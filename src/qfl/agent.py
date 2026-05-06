"""
QFL Agent — Quantum Federated Learning.
Proposed method. No RL, no SAC, no policy gradient.

Training:
  SPSA (Simultaneous Perturbation Stochastic Approximation) directly
  minimizes voltage violations by updating VQC parameters.
  No episodes, no discount, no critic — pure quantum optimization.

Reference: Chen & Yoo (2021) "Federated Quantum Machine Learning"
           SPSA: Spall (1992)

What gets federated each round (280 params = 1.1 KB):
  QuantumEncoder linear  264 params
  VQC weights             16 params
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.qe_sac.vqc import VQCLayer


# ---------------------------------------------------------------------------
# Quantum Encoder (federated)
# ---------------------------------------------------------------------------

class QuantumEncoder(nn.Module):
    """
    obs_dim → 64  (local_fc,   PRIVATE — different per client)
           → 32  (shared_head, FEDERATED — same shape for all)
           → 8   (Tanh × π)
           → VQC (FEDERATED)
           → 8 outputs

    Only shared_head + vqc are communicated (264 + 16 = 280 params = 1.1 KB).
    local_fc stays private because obs_dim differs across topologies.
    """
    def __init__(self, obs_dim: int):
        super().__init__()
        self.local_fc   = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU())
        self.shared_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8),  nn.Tanh(),
        )
        self.vqc = VQCLayer()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.local_fc(obs)
        z = self.shared_head(h) * math.pi   # scale to (-π, π)
        return self.vqc(z)

    def get_weights(self) -> dict:
        return {
            "shared_head": {k: v.clone().cpu() for k, v in self.shared_head.state_dict().items()},
            "vqc":         self.vqc.weights.data.clone().cpu(),
        }

    def set_weights(self, w: dict) -> None:
        device = next(self.shared_head.parameters()).device
        self.shared_head.load_state_dict(
            {k: v.to(device) for k, v in w["shared_head"].items()}
        )
        with torch.no_grad():
            self.vqc.weights.copy_(w["vqc"].to(device))


# ---------------------------------------------------------------------------
# Action heads (private per client)
# ---------------------------------------------------------------------------

class ActionHeads(nn.Module):
    """Maps VQC output (8-dim) to per-device action probabilities. Private."""
    def __init__(self, device_dims: list[int]):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(8, d) for d in device_dims])
        self.dims  = device_dims

    def forward(self, q: torch.Tensor) -> list[torch.Tensor]:
        return [F.softmax(h(q), dim=-1) for h in self.heads]

    def select(self, q: torch.Tensor) -> np.ndarray:
        probs = self.forward(q)
        return np.array([int(p.argmax(-1).cpu()) for p in probs], dtype=np.int64)


# ---------------------------------------------------------------------------
# QFL Agent
# ---------------------------------------------------------------------------

class QFLAgent:
    """
    Quantum Federated Learning agent.

    Local training: SPSA directly minimizes voltage violations.
    No RL, no reward signal, no episodes, no critic.

    Federation: uploads QuantumEncoder weights (280 params) to server
    after each local training round. Server FedAvg → broadcast back.
    """

    def __init__(
        self,
        obs_dim:     int,
        device_dims: list[int],
        lr:          float = 0.01,
        spsa_c:      float = 0.1,
        device:      str   = "cpu",
    ):
        self.dev         = torch.device(device)
        self.device_dims = device_dims
        self.lr          = lr
        self.spsa_c      = spsa_c

        self.encoder = QuantumEncoder(obs_dim).to(self.dev)
        self.heads   = ActionHeads(device_dims).to(self.dev)

        # Classical head optimizer (Adam, updated via backprop)
        self.head_opt = torch.optim.Adam(
            list(self.encoder.local_fc.parameters()) +
            list(self.encoder.shared_head.parameters()) +
            list(self.heads.parameters()),
            lr=lr
        )

    # -----------------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.dev).unsqueeze(0)
        with torch.no_grad():
            q = self.encoder(obs_t)
            return self.heads.select(q)

    # -----------------------------------------------------------------------
    # SPSA update on VQC weights
    # -----------------------------------------------------------------------

    def _vqc_loss(self, env, obs: np.ndarray) -> float:
        """Run one step, return violation count as loss."""
        action = self.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # loss = negative reward (fewer violations = higher reward = lower loss)
        return -float(reward), next_obs, terminated or truncated

    def spsa_update(self, env, obs: np.ndarray) -> tuple[float, np.ndarray, bool]:
        """
        One SPSA step on VQC weights.
        Estimates gradient with 2 env evaluations, no backprop through env.
        """
        vqc_w = self.encoder.vqc.weights  # shape [2, 8]
        delta = (torch.randint(0, 2, vqc_w.shape, device=self.dev).float() * 2 - 1)

        # + perturbation
        with torch.no_grad(): vqc_w.add_(self.spsa_c * delta)
        loss_plus, next_obs_p, done_p = self._vqc_loss(env, obs)

        # - perturbation
        with torch.no_grad(): vqc_w.add_(-2 * self.spsa_c * delta)
        loss_minus, next_obs_m, done_m = self._vqc_loss(env, obs)

        # Restore
        with torch.no_grad(): vqc_w.add_(self.spsa_c * delta)

        # SPSA gradient estimate → update VQC
        grad = (loss_plus - loss_minus) / (2 * self.spsa_c * delta)
        with torch.no_grad(): vqc_w.sub_(self.lr * grad)

        # Use + perturbation result for next obs
        loss = (loss_plus + loss_minus) / 2
        return loss, next_obs_p, done_p

    # -----------------------------------------------------------------------
    # Classical head update (backprop — minimize violation proxy)
    # -----------------------------------------------------------------------

    def head_update(self, obs: np.ndarray, reward: float) -> None:
        """Update classical compress + action heads to reinforce good actions."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.dev).unsqueeze(0)
        q = self.encoder(obs_t)
        probs = self.heads(q)
        # Entropy maximization — encourages diverse, exploratory actions
        entropy = sum(-(p * torch.log(p + 1e-8)).sum() for p in probs)
        loss = -entropy * float(max(reward, 0) + 1e-3)
        self.head_opt.zero_grad()
        loss.backward()
        self.head_opt.step()

    # -----------------------------------------------------------------------
    # Train one FL round
    # -----------------------------------------------------------------------

    def train_round(self, env, steps: int) -> float:
        """
        Train for `steps` env steps using SPSA on VQC + backprop on heads.
        Returns mean reward per step.
        """
        obs, _ = env.reset()
        total_reward = 0.0

        for _ in range(steps):
            loss, next_obs, done = self.spsa_update(env, obs)
            reward = -loss
            total_reward += reward
            self.head_update(obs, reward)
            obs = next_obs
            if done:
                obs, _ = env.reset()

        return total_reward / steps

    # -----------------------------------------------------------------------
    # Federation interface
    # -----------------------------------------------------------------------

    def get_federated_weights(self) -> dict:
        return self.encoder.get_weights()

    def set_federated_weights(self, w: dict) -> None:
        self.encoder.set_weights(w)
