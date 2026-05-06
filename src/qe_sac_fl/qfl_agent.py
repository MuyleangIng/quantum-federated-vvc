"""
QFL Agent — Quantum Federated Learning without SAC.

Based on Chen & Yoo (2021) "Federated Quantum Machine Learning".
Uses REINFORCE (policy gradient) instead of actor-critic SAC.

Architecture (same encoder as QE-SAC-FL, different training algorithm):
    obs → LocalEncoder (private) → SharedEncoderHead (federated)
        → VQC (federated) → per-device action heads (private)

What gets federated:
    SharedEncoderHead  264 params
    VQC                 16 params
    Total:             280 params  (same as QE-SAC-FL)

Training:
    REINFORCE — collect episode, compute returns, gradient update.
    No critic, no replay buffer, no entropy term.
    This is the minimal quantum FL baseline (QFL) for comparison with QE-SAC-FL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.qe_sac.vqc import VQCLayer
from src.qe_sac_fl.aligned_encoder import AlignedCAE, train_aligned_cae
from src.qe_sac.autoencoder import collect_random_observations


class QFLPolicy(nn.Module):
    """
    Quantum policy: obs → LocalEncoder → SharedEncoderHead → VQC → action heads.
    Identical encoder to QE-SAC-FL but no critic.
    """

    def __init__(self, obs_dim: int, device_dims: list[int], hidden_dim: int = 32):
        super().__init__()
        self.cae   = AlignedCAE(obs_dim, hidden_dim=hidden_dim)
        self.vqc   = VQCLayer()
        self.heads = nn.ModuleList([nn.Linear(8, d) for d in device_dims])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        z = self.cae.encode(obs)
        q = self.vqc(z)
        return [F.softmax(h(q), dim=-1) for h in self.heads]

    def select_action(self, obs: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """Sample action and return (action array, summed log_prob)."""
        probs_list = self.forward(obs)
        actions, log_probs = [], []
        for probs in probs_list:
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            actions.append(int(a.cpu()))
            log_probs.append(dist.log_prob(a))
        return np.array(actions, dtype=np.int64), torch.stack(log_probs).sum()

    def get_shared_weights(self) -> dict:
        return {
            "shared_head": self.cae.get_shared_weights(),
            "vqc":         self.vqc.weights.data.clone().cpu(),
        }

    def set_shared_weights(self, shared: dict) -> None:
        self.cae.set_shared_weights(shared["shared_head"])
        with torch.no_grad():
            self.vqc.weights.copy_(shared["vqc"].to(self.vqc.weights.device))


class QFLAgent:
    """
    Quantum Federated Learning agent using REINFORCE.
    Minimal baseline — no critic, no replay buffer, no entropy.
    """

    def __init__(
        self,
        obs_dim: int,
        device_dims: list[int],
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 32,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma  = gamma
        self.policy = QFLPolicy(obs_dim, device_dims, hidden_dim).to(self.device)
        self.opt    = optim.Adam(self.policy.parameters(), lr=lr)

    def pretrain_cae(self, env, n_obs: int = 2000) -> float:
        obs = collect_random_observations(env, n_obs)
        return train_aligned_cae(self.policy.cae, obs, n_steps=50, device=str(self.device))

    def run_episode(self, env) -> tuple[float, list, list]:
        """Run one episode. Returns (total_reward, log_probs, rewards)."""
        obs, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, lp = self.policy.select_action(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(lp)
            rewards.append(reward)
        return float(sum(rewards)), log_probs, rewards

    def update(self, log_probs: list, rewards: list) -> float:
        """REINFORCE update. Returns loss value."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -torch.stack(log_probs).dot(returns_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def train_round(self, env, steps: int) -> tuple[float, float]:
        """
        Train for `steps` environment steps using REINFORCE.
        Returns (mean_reward, vqc_grad_norm).
        """
        total_reward, episode_count = 0.0, 0
        steps_done = 0

        while steps_done < steps:
            ep_reward, log_probs, rewards = self.run_episode(env)
            self.update(log_probs, rewards)
            total_reward  += ep_reward
            episode_count += 1
            steps_done    += len(rewards)

        vqc_grad = float(self.policy.vqc.weights.grad.norm().item()) if self.policy.vqc.weights.grad is not None else 0.0
        return total_reward / max(episode_count, 1), vqc_grad

    def get_shared_weights(self) -> dict:
        return self.policy.get_shared_weights()

    def set_shared_weights(self, shared: dict) -> None:
        self.policy.set_shared_weights(shared)


def fedavg_qfl(weight_list: list[dict]) -> dict:
    """Uniform FedAvg over QFL shared weights (SharedHead + VQC)."""
    n = len(weight_list)

    # Average SharedEncoderHead
    head_keys = weight_list[0]["shared_head"].keys()
    avg_head = {}
    for k in head_keys:
        avg_head[k] = torch.stack([w["shared_head"][k].float() for w in weight_list]).mean(0)

    # Average VQC weights (shape [2, 8])
    avg_vqc = torch.stack([w["vqc"].float() for w in weight_list]).mean(0)

    return {"shared_head": avg_head, "vqc": avg_vqc}
