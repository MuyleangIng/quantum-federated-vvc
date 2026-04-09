"""
AlignedQESACAgent — QE-SAC with aligned encoder for federated training.

Architecture:
    obs → LocalEncoder (private) → SharedEncoderHead (federated) → VQC (federated)
        → N per-device heads → factorized action (same as QESACAgent)

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
from src.qe_sac.qe_sac_policy import _FactorizedCritic, _FactorizedSACBase
from src.qe_sac_fl.aligned_encoder import AlignedCAE, train_aligned_cae
from src.qe_sac.autoencoder import collect_random_observations


# ---------------------------------------------------------------------------
# Aligned actor network  (mirrors QESACActorNetwork but with AlignedCAE)
# ---------------------------------------------------------------------------

class AlignedActorNetwork(nn.Module):
    """
    Factorized actor: AlignedCAE → VQC → N per-device heads.

    Same structure as QESACActorNetwork except the encoder is split into
    LocalEncoder (private) + SharedEncoderHead (federated).

    obs → LocalEncoder → SharedEncoderHead → VQC → [Linear(8,|Ai|) + Softmax] × N
    """

    def __init__(self, obs_dim: int, device_dims: list[int], noise_lambda: float = 0.0):
        super().__init__()
        self.cae   = AlignedCAE(obs_dim)
        self.vqc   = VQCLayer(noise_lambda=noise_lambda)
        self.heads = nn.ModuleList([nn.Linear(8, d) for d in device_dims])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        z = self.cae.encode(obs)           # (B, 8) in [-π, π]
        q = self.vqc(z)                    # (B, 8) in [-1, 1]
        return [F.softmax(h(q), dim=-1) for h in self.heads]

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs_list = self.forward(obs)
        return np.array([
            int((p.argmax(-1) if deterministic else torch.multinomial(p, 1).squeeze(-1)).cpu())
            for p in probs_list
        ], dtype=np.int64)

    # --- Federation interface ---

    def get_shared_weights(self) -> dict:
        """Return SharedEncoderHead + VQC weights for FedAvg."""
        return {
            "shared_head": self.cae.get_shared_weights(),
            "vqc":         self.vqc.weights.data.clone().cpu(),
        }

    def set_shared_weights(self, shared: dict) -> None:
        """Load aggregated weights from server."""
        self.cae.set_shared_weights(shared["shared_head"])
        with torch.no_grad():
            self.vqc.weights.copy_(shared["vqc"].to(self.vqc.weights.device))


# ---------------------------------------------------------------------------
# AlignedQESACAgent
# ---------------------------------------------------------------------------

class AlignedQESACAgent(_FactorizedSACBase):
    """
    QE-SAC agent with aligned encoder for cross-feeder federated learning.

    Drop-in replacement for QESACAgent in the federated trainer.
    Uses identical factorized SAC update logic (inherits _FactorizedSACBase).

    Additional methods: get_shared_weights / set_shared_weights / pretrain_cae.
    """

    def __init__(
        self,
        obs_dim:      int,
        device_dims:  list[int],
        lr:           float = 3e-4,
        gamma:        float = 0.99,
        tau:          float = 0.005,
        alpha:        float = 0.2,
        buffer_size:  int   = 200_000,
        noise_lambda: float = 0.0,
        device:       str   = "cpu",
    ):
        self.obs_dim     = obs_dim
        self.device_dims = device_dims
        self.n_dev       = len(device_dims)
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.device      = device
        self._grad_steps = 0

        self.actor   = AlignedActorNetwork(obs_dim, device_dims, noise_lambda).to(device)
        self.critic1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.critic2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        # Replay buffer — MultiDiscrete actions stored as int arrays (n_dev,)
        self._buf_obs  = np.zeros((buffer_size, obs_dim),  dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, self.n_dev), dtype=np.int32)
        self._buf_rew  = np.zeros(buffer_size,              dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),  dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,              dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._max  = buffer_size

    # --- Federation interface ---

    def get_shared_weights(self) -> dict:
        return self.actor.get_shared_weights()

    def set_shared_weights(self, shared: dict) -> None:
        self.actor.set_shared_weights(shared)

    # --- Standard RL interface ---

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        return self.actor.select_action(obs, deterministic)

    def store(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,   # MultiDiscrete array (n_dev,)
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
        v_viol:   float = 0.0,
    ) -> None:
        self._buf_obs[self._ptr]  = obs
        self._buf_act[self._ptr]  = action
        self._buf_rew[self._ptr]  = reward
        self._buf_next[self._ptr] = next_obs
        self._buf_done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

    def update(
        self,
        batch_size:          int = 256,
        cae_update_interval: int = 500,
        cae_steps:           int = 50,
    ) -> dict:
        logs = self._sac_update(batch_size)
        if not logs:
            return logs

        self._grad_steps += 1

        # Co-adaptive AlignedCAE update every C steps
        if self._grad_steps % cae_update_interval == 0 and self._size > 0:
            recent_obs = self._buf_obs[: self._size]
            cae_loss = train_aligned_cae(
                self.actor.cae, recent_obs,
                n_steps=cae_steps, device=self.device,
            )
            logs["cae_loss"] = cae_loss

        return logs

    def pretrain_cae(
        self,
        env,
        n_collect:     int = 5_000,
        n_train_steps: int = 200,
    ) -> float:
        """Pre-train AlignedCAE on random-policy observations before RL."""
        observations = collect_random_observations(env, n_steps=n_collect)
        return train_aligned_cae(
            self.actor.cae, observations,
            n_steps=n_train_steps, device=self.device,
        )

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
