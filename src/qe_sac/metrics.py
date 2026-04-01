"""
Metrics tracking and evaluation utilities for QE-SAC experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainingMetrics:
    """Tracks per-episode metrics during training."""
    episode_rewards: list[float] = field(default_factory=list)
    episode_v_viols: list[int]   = field(default_factory=list)
    actor_losses:    list[float] = field(default_factory=list)
    critic_losses:   list[float] = field(default_factory=list)
    cae_losses:      list[float] = field(default_factory=list)

    def record_episode(self, reward: float, v_viol: int) -> None:
        self.episode_rewards.append(reward)
        self.episode_v_viols.append(v_viol)

    def record_losses(
        self,
        actor: float | None = None,
        critic: float | None = None,
        cae: float | None = None,
    ) -> None:
        if actor  is not None: self.actor_losses.append(actor)
        if critic is not None: self.critic_losses.append(critic)
        if cae    is not None: self.cae_losses.append(cae)

    def mean_reward(self, last_n: int = 100) -> float:
        r = self.episode_rewards[-last_n:]
        return float(np.mean(r)) if r else 0.0

    def total_v_viols(self) -> int:
        return sum(self.episode_v_viols)

    def summary(self) -> dict[str, Any]:
        return {
            "n_episodes":    len(self.episode_rewards),
            "mean_reward":   self.mean_reward(),
            "total_v_viols": self.total_v_viols(),
            "mean_actor_loss":  float(np.mean(self.actor_losses))  if self.actor_losses  else None,
            "mean_critic_loss": float(np.mean(self.critic_losses)) if self.critic_losses else None,
            "mean_cae_loss":    float(np.mean(self.cae_losses))    if self.cae_losses    else None,
        }


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_policy(
    env,
    policy,
    n_episodes: int = 10,
    deterministic: bool = True,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Roll out *policy* in *env* for *n_episodes* and return mean reward and
    voltage violation count.

    policy must have a `select_action(obs, deterministic)` method that
    returns a numpy action array.
    """
    total_reward = 0.0
    total_vviols = 0
    policy.eval()

    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        terminated = False
        while not terminated:
            with torch.no_grad():
                action = policy.select_action(
                    torch.tensor(obs, dtype=torch.float32, device=device),
                    deterministic=deterministic,
                )
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_vviols += info.get("v_viol", 0)
            if truncated:
                break
        total_reward += ep_reward

    policy.train()
    return {
        "mean_reward": total_reward / n_episodes,
        "mean_v_viols": total_vviols / n_episodes,
    }
