"""
Classical SAC baseline with MLP actor for VVC.

Architecture:
    Actor  : MLP(obs_dim → 256 → 256 → n_actions) + Softmax
    Critics: Twin MLP Q-networks (obs_dim + n_actions → 256 → 256 → 1)

This matches the SAC baseline in the paper (~899K params on 13-bus).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .metrics import count_parameters


class _MLPActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return F.softmax(logits, dim=-1)

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs = self.forward(obs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action.cpu().numpy()


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


class ClassicalSACAgent:
    """
    Classical SAC agent with MLP actor.
    Discrete action handling: actor outputs per-action probabilities;
    critic takes the full probability vector (soft Q approach).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 100_000,
        device: str = "cpu",
    ):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.gamma     = gamma
        self.tau       = tau
        self.alpha     = alpha
        self.device    = device

        self.actor   = _MLPActor(obs_dim, n_actions).to(device)
        self.critic1 = _MLPCritic(obs_dim, n_actions).to(device)
        self.critic2 = _MLPCritic(obs_dim, n_actions).to(device)
        self.target1 = _MLPCritic(obs_dim, n_actions).to(device)
        self.target2 = _MLPCritic(obs_dim, n_actions).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        # Replay buffer (simple ring buffer)
        self._buf_obs  = np.zeros((buffer_size, obs_dim),   dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, n_actions),  dtype=np.float32)
        self._buf_rew  = np.zeros(buffer_size,               dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),   dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,               dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._max  = buffer_size

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        return self.actor.select_action(obs, deterministic)

    def _action_to_onehot(self, action: np.ndarray) -> np.ndarray:
        """Convert scalar/vector action to one-hot for critic input."""
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
    ) -> None:
        self._buf_obs[self._ptr]  = obs
        self._buf_act[self._ptr]  = self._action_to_onehot(action)
        self._buf_rew[self._ptr]  = reward
        self._buf_next[self._ptr] = next_obs
        self._buf_done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

    def update(self, batch_size: int = 256, **kwargs) -> dict[str, float]:
        if self._size < batch_size:
            return {}

        idx = np.random.randint(0, self._size, batch_size)
        obs  = torch.tensor(self._buf_obs[idx],  device=self.device)
        act  = torch.tensor(self._buf_act[idx],  device=self.device)
        rew  = torch.tensor(self._buf_rew[idx],  device=self.device).unsqueeze(1)
        nobs = torch.tensor(self._buf_next[idx], device=self.device)
        done = torch.tensor(self._buf_done[idx], device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_probs = self.actor(nobs)
            q1_next = self.target1(nobs, next_probs)
            q2_next = self.target2(nobs, next_probs)
            q_next  = torch.min(q1_next, q2_next)
            entropy = -(next_probs * torch.log(next_probs + 1e-8)).sum(dim=-1, keepdim=True)
            target_q = rew + (1 - done) * self.gamma * (q_next + self.alpha * entropy)

        # Critic updates
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        c1_loss = F.mse_loss(q1, target_q)
        c2_loss = F.mse_loss(q2, target_q)
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

        # Actor update
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

        return {
            "critic_loss": float((c1_loss + c2_loss) / 2),
            "actor_loss":  float(actor_loss),
        }

    def param_count(self) -> int:
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
