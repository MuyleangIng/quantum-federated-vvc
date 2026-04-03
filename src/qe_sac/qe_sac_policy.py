"""
QE-SAC and QC-SAC agents — paper-accurate factorized architecture.

Actor   : CAE encoder → VQC → N per-device Linear heads → Softmax per device
          π(at|st) = ∏ πi(a(i)t|st)           (Eq. 27 — factorized policy)

Critic  : Shared MLP encoder → N per-device Q-value heads
          Qφ(st) = [Q1φ(st,·), ..., QNφ(st,·)]  (multi-head critic, Sec 3.4)

Separate CAE replay buffer BCAE (Algorithm 1, Step 2) — distinct from RL buffer.
Co-adaptive CAE fine-tuning every C=500 gradient steps from BCAE.

QC-SAC uses a fixed PCA encoder (fitted offline, frozen during training) instead of
the co-adaptive CAE — isolates the importance of encoder adaptivity.
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


# ── Factorized critic (shared by all 4 agents) ─────────────────────────────

class _FactorizedCritic(nn.Module):
    """
    Shared MLP encoder + N per-device Q-value heads.

    forward(obs) returns a list of (B, |Ai|) tensors — one per device.
    The Q-value for taken action a_i is obtained by indexing Q_i[b, a_i].
    """

    def __init__(self, obs_dim: int, device_dims: list[int], hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in device_dims])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        h = self.encoder(obs)
        return [head(h) for head in self.heads]


# ── QE-SAC actor ────────────────────────────────────────────────────────────

class QESACActorNetwork(nn.Module):
    """
    Factorized actor: CAE encoder → VQC → N per-device heads.

    obs  →  CAE.encode  →  latent (8,) in [-π,π]
         →  VQCLayer    →  vqc_out (8,) in [-1,1]
         →  Linear(8,|Ai|) + Softmax  ×N devices
    """

    def __init__(self, obs_dim: int, device_dims: list[int], noise_lambda: float = 0.0):
        super().__init__()
        self.cae   = CAE(obs_dim)
        self.vqc   = VQCLayer(noise_lambda=noise_lambda)
        self.heads = nn.ModuleList([nn.Linear(8, d) for d in device_dims])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        z = self.cae.encode(obs)   # (B, 8) in [-π, π]
        q = self.vqc(z)            # (B, 8) in [-1, 1]
        return [F.softmax(h(q), dim=-1) for h in self.heads]

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs_list = self.forward(obs)
        return np.array([
            int((p.argmax(-1) if deterministic else torch.multinomial(p, 1).squeeze(-1)).cpu())
            for p in probs_list
        ], dtype=np.int64)


# ── QC-SAC actor (fixed PCA encoder) ────────────────────────────────────────

class QCSACActorNetwork(nn.Module):
    """
    Factorized actor: fixed PCA encoder → VQC → N per-device heads.

    PCA is fitted offline on random rollouts and frozen during training.
    Tests: 'does the adaptive encoder matter, or is any decent compression enough?'
    """

    def __init__(self, obs_dim: int, device_dims: list[int], noise_lambda: float = 0.0):
        super().__init__()
        self.vqc   = VQCLayer(noise_lambda=noise_lambda)
        self.heads = nn.ModuleList([nn.Linear(8, d) for d in device_dims])

        # PCA parameters — registered as non-grad buffers (fitted offline)
        self.register_buffer("_pca_mean",  torch.zeros(obs_dim))
        self.register_buffer("_pca_comps", torch.zeros(8, obs_dim))
        self.register_buffer("_pca_scale", torch.ones(8))
        self._pca_fitted = False

    def fit_pca(self, observations: np.ndarray) -> None:
        """Fit PCA on random-rollout observations. Call before training begins."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=8)
        pca.fit(observations)
        z_all = pca.transform(observations)
        scale = np.abs(z_all).max(axis=0).clip(min=1e-8)

        self._pca_mean.copy_(torch.tensor(pca.mean_,        dtype=torch.float32))
        self._pca_comps.copy_(torch.tensor(pca.components_, dtype=torch.float32))
        self._pca_scale.copy_(torch.tensor(scale,           dtype=torch.float32))
        self._pca_fitted = True

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._pca_fitted:
            raise RuntimeError("Call fit_pca() before training QC-SAC.")
        z = (obs - self._pca_mean) @ self._pca_comps.T  # (B, 8)
        return z / self._pca_scale * torch.pi            # scaled to [-π, π]

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        z = self.encode(obs)
        q = self.vqc(z)
        return [F.softmax(h(q), dim=-1) for h in self.heads]

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        probs_list = self.forward(obs)
        return np.array([
            int((p.argmax(-1) if deterministic else torch.multinomial(p, 1).squeeze(-1)).cpu())
            for p in probs_list
        ], dtype=np.int64)


# ── Shared factorized-SAC update mixin ──────────────────────────────────────

class _FactorizedSACBase:
    """
    Shared SAC update logic for all 4 agents.
    Subclasses set self.actor, self.critic1/2, self.target1/2, self.actor_opt, etc.
    """

    def _sac_update(
        self,
        batch_size: int = 256,
    ) -> dict[str, float]:
        if self._size < batch_size:
            return {}

        B   = batch_size
        idx = np.random.randint(0, self._size, B)
        obs  = torch.tensor(self._buf_obs[idx],  device=self.device)
        acts = torch.tensor(self._buf_act[idx],  device=self.device, dtype=torch.long)  # (B, n_dev)
        rew  = torch.tensor(self._buf_rew[idx],  device=self.device).unsqueeze(1)       # (B, 1)
        nobs = torch.tensor(self._buf_next[idx], device=self.device)
        done = torch.tensor(self._buf_done[idx], device=self.device).unsqueeze(1)       # (B, 1)

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_probs = self.actor(nobs)             # list of (B, |Ai|)
            q1n = self.target1(nobs)
            q2n = self.target2(nobs)

            # Soft value V(s') = Σ_i E_πi [min(Q1i,Q2i) − α log πi]
            v_next = torch.zeros(B, 1, device=self.device)
            for i in range(self.n_dev):
                p = next_probs[i]                                          # (B, |Ai|)
                q = torch.min(q1n[i], q2n[i])                             # (B, |Ai|)
                vi = (p * (q - self.alpha * torch.log(p + 1e-8))).sum(-1, keepdim=True)
                v_next = v_next + vi
            target_q = rew + (1 - done) * self.gamma * v_next             # (B, 1)

        q1_list = self.critic1(obs)
        q2_list = self.critic2(obs)
        c1_loss = sum(
            F.mse_loss(q1_list[i][torch.arange(B), acts[:, i]].unsqueeze(1), target_q)
            for i in range(self.n_dev)
        )
        c2_loss = sum(
            F.mse_loss(q2_list[i][torch.arange(B), acts[:, i]].unsqueeze(1), target_q)
            for i in range(self.n_dev)
        )
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

        # ── Actor update ───────────────────────────────────────────────────
        probs_list = self.actor(obs)
        q1_pi = self.critic1(obs)
        q2_pi = self.critic2(obs)
        actor_loss = sum(
            (probs_list[i] * (
                self.alpha * torch.log(probs_list[i] + 1e-8)
                - torch.min(q1_pi[i], q2_pi[i])
            )).sum(-1).mean()
            for i in range(self.n_dev)
        )
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ── Soft target update ─────────────────────────────────────────────
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": float((c1_loss + c2_loss) / 2),
            "actor_loss":  float(actor_loss),
        }


# ── QESACAgent ───────────────────────────────────────────────────────────────

class QESACAgent(_FactorizedSACBase):
    """
    Full QE-SAC agent (proposed method):
        Actor  : CAE + VQC + N per-device heads  (factorized)
        Critics: Twin factorized MLP Q-networks
        Buffer : two separate buffers — RL buffer B and CAE buffer BCAE
        CAE    : co-adaptive fine-tuning every C=500 steps from BCAE
    """

    def __init__(
        self,
        obs_dim:      int,
        device_dims:  list[int],
        lr:           float = 1e-4,
        gamma:        float = 0.99,
        tau:          float = 0.005,
        alpha:        float = 0.2,
        buffer_size:  int   = 1_000_000,
        noise_lambda: float = 0.0,
        device:       str   = "cpu",
        # legacy keyword kept for backward compat with train scripts
        n_actions:    int   = 0,
    ):
        self.obs_dim     = obs_dim
        self.device_dims = device_dims
        self.n_dev       = len(device_dims)
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.device      = device
        self._grad_steps = 0

        self.actor   = QESACActorNetwork(obs_dim, device_dims, noise_lambda).to(device)
        self.critic1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.critic2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target2 = _FactorizedCritic(obs_dim, device_dims).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        # ── RL replay buffer B ─────────────────────────────────────────────
        self._buf_obs  = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._buf_act  = np.zeros((buffer_size, self.n_dev), dtype=np.int32)
        self._buf_rew  = np.zeros(buffer_size,              dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim),  dtype=np.float32)
        self._buf_done = np.zeros(buffer_size,              dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._max  = buffer_size

        # ── Separate CAE buffer BCAE (Algorithm 1, Step 2) ─────────────────
        self._buf_cae  = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._cae_ptr  = 0
        self._cae_size = 0

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

        # Mirror to BCAE buffer
        self._buf_cae[self._cae_ptr] = obs
        self._cae_ptr  = (self._cae_ptr + 1) % self._max
        self._cae_size = min(self._cae_size + 1, self._max)

    def update(
        self,
        batch_size:          int = 256,
        cae_update_interval: int = 500,
        cae_steps:           int = 50,
    ) -> dict[str, float]:
        logs = self._sac_update(batch_size)
        if not logs:
            return logs

        self._grad_steps += 1

        # Co-adaptive CAE fine-tuning from BCAE (Algorithm 1, Step 7)
        if self._grad_steps % cae_update_interval == 0 and self._cae_size > 0:
            cae_obs = self._buf_cae[: self._cae_size]
            cae_loss = train_cae(
                self.actor.cae, cae_obs,
                n_steps=cae_steps, device=self.device,
            )
            logs["cae_loss"] = cae_loss

        return logs

    def pretrain_cae(
        self,
        env,
        n_collect:     int = 5000,
        n_train_steps: int = 200,
    ) -> float:
        """
        Offline CAE pre-training on random-policy observations (Algorithm 1, Pre step).
        Also seeds the BCAE buffer with collected observations.
        """
        observations = collect_random_observations(env, n_steps=n_collect)
        n = min(len(observations), self._max)
        self._buf_cae[:n] = observations[:n]
        self._cae_ptr  = n % self._max
        self._cae_size = n
        return train_cae(
            self.actor.cae, observations,
            n_steps=n_train_steps, device=self.device,
        )

    def param_count(self) -> int:
        """
        Count only actor inference-path parameters (encoder + VQC + heads).
        Matches paper Table 4 reporting: decoder and critics are excluded.
        """
        enc   = count_parameters(self.actor.cae.encoder)
        vqc   = count_parameters(self.actor.vqc)
        heads = count_parameters(self.actor.heads)
        return enc + vqc + heads

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


# ── QCSACAgent ───────────────────────────────────────────────────────────────

class QCSACAgent(_FactorizedSACBase):
    """
    QC-SAC agent — same VQC as QE-SAC but fixed PCA encoder.

    Tests: 'does the adaptive encoder matter, or is any decent compression enough?'
    PCA is fitted offline from random rollouts and frozen during training.
    """

    def __init__(
        self,
        obs_dim:      int,
        device_dims:  list[int],
        lr:           float = 1e-4,
        gamma:        float = 0.99,
        tau:          float = 0.005,
        alpha:        float = 0.2,
        buffer_size:  int   = 1_000_000,
        noise_lambda: float = 0.0,
        device:       str   = "cpu",
        n_actions:    int   = 0,
    ):
        self.obs_dim     = obs_dim
        self.device_dims = device_dims
        self.n_dev       = len(device_dims)
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.device      = device
        self._grad_steps = 0

        self.actor   = QCSACActorNetwork(obs_dim, device_dims, noise_lambda).to(device)
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

    def store(self, obs, action, reward, next_obs, done, v_viol=0.0) -> None:
        self._buf_obs[self._ptr]  = obs
        self._buf_act[self._ptr]  = action
        self._buf_rew[self._ptr]  = reward
        self._buf_next[self._ptr] = next_obs
        self._buf_done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

    def pretrain_pca(self, env, n_collect: int = 5000) -> None:
        """Fit PCA offline on random rollouts; encoder is then frozen."""
        observations = collect_random_observations(env, n_steps=n_collect)
        self.actor.fit_pca(observations)

    def update(self, batch_size=256, **kwargs) -> dict[str, float]:
        logs = self._sac_update(batch_size)
        if logs:
            self._grad_steps += 1
        return logs

    def param_count(self) -> int:
        """Actor inference path only: VQC + heads (no encoder, no critics)."""
        return count_parameters(self.actor.vqc) + count_parameters(self.actor.heads)

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
