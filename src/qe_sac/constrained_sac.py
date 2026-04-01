"""
QESACAgentConstrained — Lagrangian Constrained SAC for Volt-VAR Control.

Extends QESACAgent with a Lagrange multiplier λ that automatically tunes
the voltage violation penalty until E[v_viol per step] ≤ 0.

Standard SAC:
    maximise  E[reward]

Constrained SAC (this class):
    maximise  E[reward]
    subject to  E[v_violations per step] ≤ 0

How λ works:
    - λ starts at 0 (no extra penalty)
    - Every episode: λ += lr_λ × mean_vviol_this_episode
    - λ = max(0, λ)   (never negative)
    - Actor loss: -(Q + α·H) + λ · constraint_violation
    - As violations persist → λ grows → actor penalised more → learns to avoid them
    - When violations reach 0 → λ stops growing → stabilises

References:
    Achiam et al. (2017) "Constrained Policy Optimization" ICML
    Yang et al. (2021) "WCSAC: Worst-Case Soft Actor Critic" AAAI
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .qe_sac_policy import QESACAgent
from .autoencoder import train_cae


class QESACAgentConstrained(QESACAgent):
    """
    QE-SAC with Lagrangian safety constraint on voltage violations.

    Parameters
    ----------
    obs_dim, n_actions, lr, gamma, tau, alpha, buffer_size, noise_lambda, device
        — same as QESACAgent
    lambda_lr : float
        Learning rate for Lagrange multiplier λ (default 0.01)
    constraint_threshold : float
        Target max mean v_viol per step (default 0.0 = zero violations)
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
        lambda_lr: float = 0.01,
        constraint_threshold: float = 0.0,
    ):
        super().__init__(
            obs_dim, n_actions, lr, gamma, tau, alpha,
            buffer_size, noise_lambda, device,
        )
        self.lambda_lr             = lambda_lr
        self.constraint_threshold  = constraint_threshold
        self.lagrange_lambda       = 0.0   # λ — starts at 0, always ≥ 0

        # Extra buffer slot: per-step voltage violation count
        self._buf_vviol_c = np.zeros(buffer_size, dtype=np.float32)

    # ------------------------------------------------------------------
    # Buffer — same as parent but also stores v_viol per step
    # ------------------------------------------------------------------

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        v_viol: float = 0.0,
    ) -> None:
        super().store(obs, action, reward, next_obs, done)
        # Parent already incremented _ptr; last written index is (ptr-1) mod max
        last_idx = (self._ptr - 1 + self._max) % self._max
        self._buf_vviol_c[last_idx] = float(v_viol)

    # ------------------------------------------------------------------
    # λ update — called once per episode from the trainer
    # ------------------------------------------------------------------

    def update_lambda(self, mean_vviol: float) -> float:
        """
        Update Lagrange multiplier based on mean violation rate this episode.

        λ ← max(0,  λ + lr_λ × (mean_vviol − threshold))

        Returns the new λ value.
        """
        self.lagrange_lambda = max(
            0.0,
            self.lagrange_lambda + self.lambda_lr * (mean_vviol - self.constraint_threshold),
        )
        return self.lagrange_lambda

    # ------------------------------------------------------------------
    # Update — same as parent but with λ·constraint term in actor loss
    # ------------------------------------------------------------------

    def update(
        self,
        batch_size: int = 256,
        cae_update_interval: int = 500,
        cae_steps: int = 50,
    ) -> dict[str, float]:
        if self._size < batch_size:
            return {}

        idx   = np.random.randint(0, self._size, batch_size)
        obs   = torch.tensor(self._buf_obs[idx],      device=self.device)
        act   = torch.tensor(self._buf_act[idx],      device=self.device)
        rew   = torch.tensor(self._buf_rew[idx],      device=self.device).unsqueeze(1)
        nobs  = torch.tensor(self._buf_next[idx],     device=self.device)
        done  = torch.tensor(self._buf_done[idx],     device=self.device).unsqueeze(1)
        vviol = torch.tensor(self._buf_vviol_c[idx],  device=self.device).unsqueeze(1)

        # --- Critic update (identical to parent) ---
        with torch.no_grad():
            next_probs = self.actor(nobs)
            q1_next    = self.target1(nobs, next_probs)
            q2_next    = self.target2(nobs, next_probs)
            q_next     = torch.min(q1_next, q2_next)
            entropy    = -(next_probs * torch.log(next_probs + 1e-8)).sum(dim=-1, keepdim=True)
            target_q   = rew + (1 - done) * self.gamma * (q_next + self.alpha * entropy)

        q1      = self.critic1(obs, act)
        q2      = self.critic2(obs, act)
        c1_loss = F.mse_loss(q1, target_q)
        c2_loss = F.mse_loss(q2, target_q)
        self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

        # --- Constrained actor update ---
        probs   = self.actor(obs)
        q1_pi   = self.critic1(obs, probs)
        q2_pi   = self.critic2(obs, probs)
        q_pi    = torch.min(q1_pi, q2_pi)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True)

        # Lagrangian actor loss:
        #   standard: -(Q + α·H)
        #   +penalty: λ · mean(v_viol in this batch)
        constraint_violation = vviol.mean()
        actor_loss = (
            -(q_pi + self.alpha * entropy).mean()
            + self.lagrange_lambda * constraint_violation
        )

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft target update
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self._grad_steps += 1
        logs: dict[str, float] = {
            "critic_loss":           float((c1_loss + c2_loss) / 2),
            "actor_loss":            float(actor_loss),
            "constraint_violation":  float(constraint_violation),
            "lagrange_lambda":       self.lagrange_lambda,
        }

        # Co-adaptive CAE update every C steps
        if self._grad_steps % cae_update_interval == 0:
            recent_obs = self._buf_obs[: self._size]
            cae_loss   = train_cae(
                self.actor.cae, recent_obs,
                n_steps=cae_steps, device=self.device,
            )
            logs["cae_loss"] = cae_loss

        return logs

    def save(self, path: str) -> None:
        """Save agent state including λ."""
        import torch
        torch.save({
            "actor":           self.actor.state_dict(),
            "critic1":         self.critic1.state_dict(),
            "critic2":         self.critic2.state_dict(),
            "lagrange_lambda": self.lagrange_lambda,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state including λ."""
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.lagrange_lambda = float(ckpt.get("lagrange_lambda", 0.0))
