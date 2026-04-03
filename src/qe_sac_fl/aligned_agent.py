"""
AlignedQESACAgent — QE-SAC with aligned encoder for federated training.

Architecture:
    obs → LocalEncoder (private) → SharedEncoderHead (federated) → VQC (federated) → action

What gets federated each round:
    SharedEncoderHead  272 params
    VQC                 16 params
    Total:             288 params = 1,152 bytes per client

What stays local (never shared):
    LocalEncoder       (feeder-specific compression)
    LocalDecoder       (reconstruction, training only)
    Critics            (feeder-specific value estimates)
    Replay buffer      (raw grid data)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.qe_sac.vqc import VQCLayer
from src.qe_sac.metrics import count_parameters
from src.qe_sac_fl.aligned_encoder import AlignedCAE, train_aligned_cae


# ---------------------------------------------------------------------------
# Aligned actor network
# ---------------------------------------------------------------------------

class AlignedActorNetwork(nn.Module):
    """
    Actor that uses AlignedCAE instead of flat CAE.
    Forward path: obs → LocalEncoder → SharedEncoderHead → VQC → head → probs
    """
    def __init__(self, obs_dim: int, n_actions: int, noise_lambda: float = 0.0):
        super().__init__()
        self.cae  = AlignedCAE(obs_dim)       # has .local_encoder and .shared_head
        self.vqc  = VQCLayer(noise_lambda=noise_lambda)
        self.head = nn.Linear(8, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent  = self.cae.encode(obs)        # (batch, 8) in [-π, π]
        vqc_out = self.vqc(latent)            # (batch, 8) in [-1, 1]
        logits  = self.head(vqc_out)          # (batch, n_actions)
        return F.softmax(logits, dim=-1)

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs = self.forward(obs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action.cpu().numpy()

    def get_shared_weights(self) -> dict:
        """
        Return weights to send to server for FedAvg.
        Includes SharedEncoderHead + VQC weights.
        """
        return {
            "shared_head": self.cae.get_shared_weights(),
            "vqc":         self.vqc.weights.data.clone().cpu(),
        }

    def set_shared_weights(self, shared: dict) -> None:
        """Load aggregated weights from server."""
        self.cae.set_shared_weights(shared["shared_head"])
        with torch.no_grad():
            self.vqc.weights.copy_(
                shared["vqc"].to(self.vqc.weights.device)
            )


# ---------------------------------------------------------------------------
# Critic (same as QESACAgent — reused directly)
# ---------------------------------------------------------------------------

class _MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),              nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


# ---------------------------------------------------------------------------
# AlignedQESACAgent
# ---------------------------------------------------------------------------

class AlignedQESACAgent:
    """
    QE-SAC agent with aligned encoder.
    Drop-in replacement for QESACAgent in the federated trainer.
    Same external API: store / update / select_action / save / load.
    Additional methods: get_shared_weights / set_shared_weights.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 200_000,
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

        self.actor   = AlignedActorNetwork(obs_dim, n_actions, noise_lambda).to(device)
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
        self._buf_obs  = np.zeros((buffer_size, obs_dim),  dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self._buf_rew  = np.zeros(buffer_size,              dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),  dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,              dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._max  = buffer_size

    # --- Federation interface ---

    def get_shared_weights(self) -> dict:
        """Extract SharedEncoderHead + VQC weights for FedAvg."""
        return self.actor.get_shared_weights()

    def set_shared_weights(self, shared: dict) -> None:
        """Load aggregated SharedEncoderHead + VQC weights from server."""
        self.actor.set_shared_weights(shared)

    # --- Standard RL interface ---

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        return self.actor.select_action(obs, deterministic)

    def _action_to_onehot(self, action) -> np.ndarray:
        oh = np.zeros(self.n_actions, dtype=np.float32)
        oh[int(action)] = 1.0
        return oh

    def store(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        v_viol: float = 0.0,
    ) -> None:
        # `QESACTrainer` always passes `v_viol`; aligned training does not
        # currently use it in the replay buffer, so we accept and ignore it.
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
    ) -> dict:
        if self._size < batch_size:
            return {}

        idx  = np.random.randint(0, self._size, batch_size)
        obs  = torch.tensor(self._buf_obs[idx],  device=self.device)
        act  = torch.tensor(self._buf_act[idx],  device=self.device)
        rew  = torch.tensor(self._buf_rew[idx],  device=self.device).unsqueeze(1)
        nobs = torch.tensor(self._buf_next[idx], device=self.device)
        done = torch.tensor(self._buf_done[idx], device=self.device).unsqueeze(1)

        # Critic update
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

        # Actor update (local encoder + shared head + VQC jointly)
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
        logs = {
            "critic_loss": float((c1_loss + c2_loss) / 2),
            "actor_loss":  float(actor_loss),
        }

        # Co-adaptive AlignedCAE update every C steps
        if self._grad_steps % cae_update_interval == 0:
            recent_obs = self._buf_obs[: self._size]
            cae_loss = train_aligned_cae(
                self.actor.cae, recent_obs,
                n_steps=cae_steps, device=self.device,
            )
            logs["cae_loss"] = cae_loss

        return logs

    def param_count(self) -> int:
        return count_parameters(self.actor)

    def eval(self):  self.actor.eval()
    def train(self): self.actor.train()

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
