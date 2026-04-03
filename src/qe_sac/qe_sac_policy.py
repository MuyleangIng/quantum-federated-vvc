"""
QE-SAC Actor: CAE encoder → VQC → Linear → Softmax

Pipeline (per forward pass):
    obs (obs_dim,)
    → CAE.encode → latent (8,)  in [-π, π]
    → VQCLayer   → vqc_out (8,) in [-1, 1]
    → Linear(8, n_actions)
    → Softmax    → action probs (n_actions,)

The CAE and VQC weights are jointly differentiable via parameter-shift
gradient in PennyLane + standard autograd for the CAE linear layers.

QE-SAC Agent wires this actor into the same SAC framework as the
classical baseline, with identical twin MLP critics and replay buffer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .autoencoder import CAE, train_cae, collect_random_observations
from .vqc import VQCLayer
from .metrics import count_parameters
from .gnn_encoder import GNNEncoder, train_gnn_encoder


class QESACActorNetwork(nn.Module):
    """
    Quantum-enhanced SAC actor: CAE + VQC + linear output head.

    Parameters
    ----------
    obs_dim      : observation dimension
    n_actions    : number of discrete actions
    noise_lambda : depolarising noise for VQC (0 = noiseless)
    """

    def __init__(self, obs_dim: int, n_actions: int, noise_lambda: float = 0.0):
        super().__init__()
        self.cae = CAE(obs_dim)
        self.vqc = VQCLayer(noise_lambda=noise_lambda)
        self.head = nn.Linear(8, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.cae.encode(obs)        # (batch, 8)  in [-π, π]
        vqc_out = self.vqc(latent)           # (batch, 8)  in [-1, 1]
        logits = self.head(vqc_out)          # (batch, n_actions)
        return F.softmax(logits, dim=-1)

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs = self.forward(obs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action.cpu().numpy()


class GNNQESACActorNetwork(nn.Module):
    """
    GNN-enhanced QE-SAC actor: GNNEncoder + VQC + linear output head.

    Replaces the flat MLP CAE with a topology-aware GNN encoder.
    Same VQC and output head as QESACActorNetwork — drop-in replacement.

    Parameters
    ----------
    obs_dim      : observation dimension (42 for DistFlow, 93 for OpenDSS)
    n_actions    : number of discrete actions
    n_buses      : number of buses in the feeder (default 15 for IEEE 13-bus)
    noise_lambda : depolarising noise for VQC (0 = noiseless)
    edge_index   : optional custom graph topology
    edge_attr    : optional custom edge features
    """

    def __init__(
        self,
        obs_dim:      int,
        n_actions:    int,
        n_buses:      int = 15,
        noise_lambda: float = 0.0,
        edge_index:   torch.Tensor | None = None,
        edge_attr:    torch.Tensor | None = None,
    ):
        super().__init__()
        self.gnn  = GNNEncoder(n_buses=n_buses, edge_index=edge_index, edge_attr=edge_attr)
        self.vqc  = VQCLayer(noise_lambda=noise_lambda)
        self.head = nn.Linear(8, n_actions)
        # keep .cae attribute for trainer compatibility (used in co-adapt update)
        self.cae  = self.gnn

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent  = self.gnn.encode(obs)       # (batch, 8)  in [-π, π]
        vqc_out = self.vqc(latent)           # (batch, 8)  in [-1, 1]
        logits  = self.head(vqc_out)         # (batch, n_actions)
        return F.softmax(logits, dim=-1)

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs = self.forward(obs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action.cpu().numpy()


class _MLPCritic(nn.Module):
    """Shared MLP critic used by both SAC variants."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),              nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class QESACAgent:
    """
    Full QE-SAC agent:
        Actor  : QESACActorNetwork (CAE + VQC)
        Critics: Twin MLP Q-networks (classical)
        Buffer : ring replay buffer (default 1M)

    Co-adaptive CAE update: every cae_update_interval gradient steps,
    re-train the CAE encoder on recent buffer observations.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 1_000_000,
        noise_lambda: float = 0.0,
        device: str = "cpu",
    ):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.gamma     = gamma
        self.tau       = tau
        self.alpha     = alpha
        self.device    = device
        self._grad_steps = 0

        self.actor   = QESACActorNetwork(obs_dim, n_actions, noise_lambda).to(device)
        self.critic1 = _MLPCritic(obs_dim, n_actions).to(device)
        self.critic2 = _MLPCritic(obs_dim, n_actions).to(device)
        self.target1 = _MLPCritic(obs_dim, n_actions).to(device)
        self.target2 = _MLPCritic(obs_dim, n_actions).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        # Replay buffer
        self._buf_obs  = np.zeros((buffer_size, obs_dim),   dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, n_actions),  dtype=np.float32)
        self._buf_rew  = np.zeros(buffer_size,               dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),   dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,               dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._max  = buffer_size

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        return self.actor.select_action(obs, deterministic)

    def _action_to_onehot(self, action: np.ndarray) -> np.ndarray:
        oh = np.zeros(self.n_actions, dtype=np.float32)
        oh[int(action)] = 1.0
        return oh

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        v_viol: float = 0.0,
    ) -> None:
        self._buf_obs[self._ptr]  = obs
        self._buf_act[self._ptr]  = self._action_to_onehot(action)
        self._buf_rew[self._ptr]  = reward
        self._buf_next[self._ptr] = next_obs
        self._buf_done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

    def update(
        self,
        batch_size: int = 256,
        cae_update_interval: int = 500,
        cae_steps: int = 50,
    ) -> dict[str, float]:
        if self._size < batch_size:
            return {}

        idx = np.random.randint(0, self._size, batch_size)
        obs  = torch.tensor(self._buf_obs[idx],  device=self.device)
        act  = torch.tensor(self._buf_act[idx],  device=self.device)
        rew  = torch.tensor(self._buf_rew[idx],  device=self.device).unsqueeze(1)
        nobs = torch.tensor(self._buf_next[idx], device=self.device)
        done = torch.tensor(self._buf_done[idx], device=self.device).unsqueeze(1)

        # --- Critic update ---
        with torch.no_grad():
            next_probs = self.actor(nobs)
            q1_next = self.target1(nobs, next_probs)
            q2_next = self.target2(nobs, next_probs)
            q_next  = torch.min(q1_next, q2_next)
            entropy = -(next_probs * torch.log(next_probs + 1e-8)).sum(dim=-1, keepdim=True)
            target_q = rew + (1 - done) * self.gamma * (q_next + self.alpha * entropy)

        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        c1_loss = F.mse_loss(q1, target_q)
        c2_loss = F.mse_loss(q2, target_q)
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

        # --- Actor update (CAE + VQC jointly) ---
        probs = self.actor(obs)
        q1_pi = self.critic1(obs, probs)
        q2_pi = self.critic2(obs, probs)
        q_pi  = torch.min(q1_pi, q2_pi)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True)
        actor_loss = -(q_pi + self.alpha * entropy).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Soft target update
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self._grad_steps += 1
        logs: dict[str, float] = {
            "critic_loss": float((c1_loss + c2_loss) / 2),
            "actor_loss":  float(actor_loss),
        }

        # --- Co-adaptive CAE update every C steps ---
        if self._grad_steps % cae_update_interval == 0:
            recent_obs = self._buf_obs[: self._size]
            cae_loss = train_cae(
                self.actor.cae,
                recent_obs,
                n_steps=cae_steps,
                device=self.device,
            )
            logs["cae_loss"] = cae_loss

        return logs

    def pretrain_cae(
        self,
        env,
        n_collect: int = 5000,
        n_train_steps: int = 200,
    ) -> float:
        """
        Offline CAE pre-training on random-policy observations (paper Section III-C).

        Collect *n_collect* observations from a random policy, then train the
        CAE encoder for *n_train_steps* gradient steps on those observations.
        Must be called before the RL training loop begins.

        Parameters
        ----------
        env           : Gymnasium VVC environment
        n_collect     : number of random-policy steps to collect
        n_train_steps : CAE gradient steps on the collected data

        Returns
        -------
        final_loss : reconstruction loss after pre-training
        """
        observations = collect_random_observations(env, n_steps=n_collect)
        return train_cae(
            self.actor.cae,
            observations,
            n_steps=n_train_steps,
            device=self.device,
        )

    def param_count(self) -> int:
        """Count actor (quantum) parameters only — matches paper metric."""
        return count_parameters(self.actor)

    def eval(self) -> None:
        self.actor.eval()

    def train(self) -> None:
        self.actor.train()

    def save(self, path: str) -> None:
        torch.save({
            "actor":   self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])


class GNNQESACAgent(QESACAgent):
    """
    QE-SAC agent with GNN encoder instead of flat MLP CAE.

    Everything identical to QESACAgent except the actor uses
    GNNQESACActorNetwork (topology-aware GNN + VQC).

    Parameters
    ----------
    obs_dim      : observation dimension
    n_actions    : number of discrete actions
    n_buses      : buses in feeder (15 for IEEE 13-bus)
    edge_index   : optional custom graph connectivity
    edge_attr    : optional custom line parameters
    All other kwargs forwarded to QESACAgent.
    """

    def __init__(
        self,
        obs_dim:    int,
        n_actions:  int,
        n_buses:    int = 15,
        edge_index: torch.Tensor | None = None,
        edge_attr:  torch.Tensor | None = None,
        **kwargs,
    ):
        # Call parent init to set up critics/buffer/optimisers
        super().__init__(obs_dim=obs_dim, n_actions=n_actions, **kwargs)

        # Replace actor with GNN-based actor (same VQC, different encoder)
        device = kwargs.get("device", "cpu")
        noise_lambda = kwargs.get("noise_lambda", 0.0)
        self.actor = GNNQESACActorNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_buses=n_buses,
            noise_lambda=noise_lambda,
            edge_index=edge_index,
            edge_attr=edge_attr,
        ).to(device)
        # Re-initialise actor optimiser for new parameters
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=kwargs.get("lr", 1e-4))

    def update(
        self,
        batch_size: int = 256,
        cae_update_interval: int = 500,
        cae_steps: int = 50,
    ) -> dict[str, float]:
        """Same as QESACAgent.update() — co-adaptive GNN update every C steps."""
        logs = super().update(batch_size=batch_size,
                              cae_update_interval=cae_update_interval,
                              cae_steps=cae_steps)
        # Rename cae_loss → gnn_loss for clarity
        if "cae_loss" in logs:
            logs["gnn_loss"] = logs.pop("cae_loss")
        return logs
