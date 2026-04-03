"""
Classical SAC and SAC-AE baselines — paper-accurate factorized architecture.

Classical SAC : MLP(256→256) + N per-device heads + twin factorized critics
SAC-AE        : co-adaptive CAE + tiny MLP(8→8, 2 layers) + N per-device heads
                Same parameter count as QE-SAC; no VQC.
                Tests: 'is it quantum, or just compression?'  → fails to converge.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .autoencoder import CAE, train_cae, collect_random_observations
from .metrics import count_parameters
from .qe_sac_policy import _FactorizedCritic, _FactorizedSACBase


# ── Classical SAC actor ──────────────────────────────────────────────────────

class _ClassicalActor(nn.Module):
    """MLP(256→256) + N per-device Linear heads + Softmax."""

    def __init__(self, obs_dim: int, device_dims: list[int], hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in device_dims])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        h = self.net(obs)
        return [F.softmax(head(h), dim=-1) for head in self.heads]

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs_list = self.forward(obs)
        return np.array([
            int((p.argmax(-1) if deterministic else torch.multinomial(p, 1).squeeze(-1)).cpu())
            for p in probs_list
        ], dtype=np.int64)


# ── SAC-AE actor ─────────────────────────────────────────────────────────────

class _SACAEActor(nn.Module):
    """
    co-adaptive CAE encoder → tiny MLP (2 hidden layers of 8 units) → N per-device heads.

    Same parameter count as QE-SAC, no VQC.
    Ablation: proves compression alone can't replace quantum processing.
    """

    def __init__(self, obs_dim: int, device_dims: list[int]):
        super().__init__()
        self.cae = CAE(obs_dim)
        self.mlp = nn.Sequential(
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(8, d) for d in device_dims])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        z = self.cae.encode(obs)   # (B, 8) in [-π, π]
        h = self.mlp(z)            # (B, 8)
        return [F.softmax(head(h), dim=-1) for head in self.heads]

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs_list = self.forward(obs)
        return np.array([
            int((p.argmax(-1) if deterministic else torch.multinomial(p, 1).squeeze(-1)).cpu())
            for p in probs_list
        ], dtype=np.int64)


# ── ClassicalSACAgent ─────────────────────────────────────────────────────────

class ClassicalSACAgent(_FactorizedSACBase):
    """
    Classical SAC — primary performance baseline.
    Fully classical actor and critic, two hidden layers of 256 units.
    """

    def __init__(
        self,
        obs_dim:     int,
        device_dims: list[int],
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        tau:         float = 0.005,
        alpha:       float = 0.2,
        buffer_size: int   = 1_000_000,
        device:      str   = "cpu",
        n_actions:   int   = 0,
    ):
        self.obs_dim     = obs_dim
        self.device_dims = device_dims
        self.n_dev       = len(device_dims)
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.device      = device
        self._grad_steps = 0

        self.actor   = _ClassicalActor(obs_dim, device_dims).to(device)
        self.critic1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.critic2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        self._buf_obs  = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, self.n_dev), dtype=np.int32)
        self._buf_rew  = np.zeros(buffer_size,              dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),  dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,              dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._max  = buffer_size

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        return self.actor.select_action(obs, deterministic)

    def store(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        self._buf_obs[self._ptr]  = obs
        self._buf_act[self._ptr]  = action
        self._buf_rew[self._ptr]  = reward
        self._buf_next[self._ptr] = next_obs
        self._buf_done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

    def update(self, batch_size=256, **kwargs) -> dict[str, float]:
        logs = self._sac_update(batch_size)
        if logs:
            self._grad_steps += 1
        return logs

    def param_count(self) -> int:
        """Actor only (net + heads) — matches paper Table 4 reporting."""
        return count_parameters(self.actor)

    def eval(self)  -> None: self.actor.eval()
    def train(self) -> None: self.actor.train()

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


# ── SACAEAgent ────────────────────────────────────────────────────────────────

class SACAEAgent(_FactorizedSACBase):
    """
    SAC-AE — co-adaptive CAE + tiny classical MLP (8 units, 2 layers), no VQC.

    Same parameter count and compression pipeline as QE-SAC, but classical core.
    Expected to fail to converge — proving quantum processing drives QE-SAC performance.
    """

    def __init__(
        self,
        obs_dim:     int,
        device_dims: list[int],
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        tau:         float = 0.005,
        alpha:       float = 0.2,
        buffer_size: int   = 1_000_000,
        device:      str   = "cpu",
        n_actions:   int   = 0,
    ):
        self.obs_dim     = obs_dim
        self.device_dims = device_dims
        self.n_dev       = len(device_dims)
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.device      = device
        self._grad_steps = 0

        self.actor   = _SACAEActor(obs_dim, device_dims).to(device)
        self.critic1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.critic2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        self._buf_obs  = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, self.n_dev), dtype=np.int32)
        self._buf_rew  = np.zeros(buffer_size,              dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),  dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,              dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._max  = buffer_size

        self._buf_cae  = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._cae_ptr  = 0
        self._cae_size = 0

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        return self.actor.select_action(obs, deterministic)

    def store(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        self._buf_obs[self._ptr]  = obs
        self._buf_act[self._ptr]  = action
        self._buf_rew[self._ptr]  = reward
        self._buf_next[self._ptr] = next_obs
        self._buf_done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

        self._buf_cae[self._cae_ptr] = obs
        self._cae_ptr  = (self._cae_ptr + 1) % self._max
        self._cae_size = min(self._cae_size + 1, self._max)

    def pretrain_cae(self, env, n_collect=5000, n_train_steps=200) -> float:
        observations = collect_random_observations(env, n_steps=n_collect)
        n = min(len(observations), self._max)
        self._buf_cae[:n] = observations[:n]
        self._cae_ptr  = n % self._max
        self._cae_size = n
        return train_cae(self.actor.cae, observations, n_steps=n_train_steps, device=self.device)

    def update(self, batch_size=256, cae_update_interval=500, cae_steps=50, **kwargs) -> dict[str, float]:
        logs = self._sac_update(batch_size)
        if not logs:
            return logs

        self._grad_steps += 1
        if self._grad_steps % cae_update_interval == 0 and self._cae_size > 0:
            cae_obs = self._buf_cae[: self._cae_size]
            logs["cae_loss"] = train_cae(
                self.actor.cae, cae_obs, n_steps=cae_steps, device=self.device,
            )
        return logs

    def param_count(self) -> int:
        """Actor inference path (encoder + mlp + heads), no decoder or critics."""
        enc   = count_parameters(self.actor.cae.encoder)
        mlp   = count_parameters(self.actor.mlp)
        heads = count_parameters(self.actor.heads)
        return enc + mlp + heads

    def eval(self)  -> None: self.actor.eval()
    def train(self) -> None: self.actor.train()

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
