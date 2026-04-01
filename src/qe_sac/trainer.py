"""
Training loop and comparison utilities for QE-SAC experiments.

Usage
-----
from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac.qe_sac_policy import QESACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent
from src.qe_sac.trainer import QESACTrainer, compare_agents

env = VVCEnv13Bus()
agent = QESACAgent(obs_dim=env.observation_space.shape[0],
                   n_actions=env.action_space.nvec.sum())
trainer = QESACTrainer(agent, env)
metrics = trainer.train(n_steps=10_000)
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch

from .metrics import TrainingMetrics, evaluate_policy


class QESACTrainer:
    """
    Training loop for QE-SAC (or classical SAC — same interface).

    Parameters
    ----------
    agent          : QESACAgent or ClassicalSACAgent
    env            : Gymnasium VVC environment
    batch_size     : SAC mini-batch size
    cae_update_interval : retrain CAE every this many gradient steps (QE-SAC only)
    warmup_steps   : random exploration steps before learning starts
    log_interval   : print progress every N episodes
    save_dir       : directory to save checkpoints
    device         : torch device
    """

    def __init__(
        self,
        agent,
        env,
        batch_size: int = 256,
        cae_update_interval: int = 500,
        warmup_steps: int = 1000,
        log_interval: int = 50,
        save_dir: str = "artifacts/qe_sac",
        device: str = "cpu",
    ):
        self.agent  = agent
        self.env    = env
        self.batch  = batch_size
        self.cae_interval = cae_update_interval
        self.warmup = warmup_steps
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.device   = device
        os.makedirs(save_dir, exist_ok=True)

    def train(self, n_steps: int = 50_000) -> TrainingMetrics:
        """
        Train for *n_steps* environment steps.

        Returns a TrainingMetrics object with episode-level logs.
        """
        metrics = TrainingMetrics()
        obs, _ = self.env.reset()
        ep_reward = 0.0
        ep_vviol  = 0
        total_steps = 0
        nvec = self.env.action_space.nvec  # e.g. [2, 2, 33]

        while total_steps < n_steps:
            # Action selection
            if total_steps < self.warmup:
                env_action = self.env.action_space.sample()   # MultiDiscrete array
                # Encode MultiDiscrete → scalar index (mixed-radix, same as decode inverse)
                scalar_action = 0
                for a, n in zip(env_action, nvec):
                    scalar_action = scalar_action * int(n) + int(a)
            else:
                with torch.no_grad():
                    t_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    # Agent returns a scalar index into the joint action space
                    scalar_action = int(self.agent.select_action(t_obs, deterministic=False))
                # Decode scalar → MultiDiscrete vector using mixed-radix decomposition
                env_action = np.zeros(len(nvec), dtype=np.int64)
                remainder = scalar_action % int(nvec.prod())
                for i in range(len(nvec) - 1, -1, -1):
                    env_action[i] = remainder % nvec[i]
                    remainder //= nvec[i]

            next_obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated
            v_viol_step = info.get("v_viol", 0)
            self.agent.store(obs, scalar_action, reward, next_obs, done, v_viol=v_viol_step)
            ep_reward += reward
            ep_vviol  += v_viol_step
            obs = next_obs
            total_steps += 1

            # Learning update
            if total_steps >= self.warmup:
                logs = self.agent.update(
                    batch_size=self.batch,
                    cae_update_interval=self.cae_interval,
                )
                if logs:
                    metrics.record_losses(
                        actor=logs.get("actor_loss"),
                        critic=logs.get("critic_loss"),
                        cae=logs.get("cae_loss"),
                    )

            if done:
                metrics.record_episode(ep_reward, ep_vviol)
                ep_idx = len(metrics.episode_rewards)

                # Lagrangian λ update — only for constrained agents
                lam = None
                if hasattr(self.agent, "update_lambda"):
                    ep_len = getattr(self.env, "episode_len", 24)
                    mean_vviol = ep_vviol / max(ep_len, 1)
                    lam = self.agent.update_lambda(mean_vviol)

                if ep_idx % self.log_interval == 0:
                    lam_str = f" | λ {lam:.4f}" if lam is not None else ""
                    print(
                        f"  ep {ep_idx:4d} | steps {total_steps:6d} | "
                        f"reward {ep_reward:8.3f} | vviol {ep_vviol:3d} | "
                        f"mean100 {metrics.mean_reward(100):8.3f}{lam_str}"
                    )
                obs, _ = self.env.reset()
                ep_reward = 0.0
                ep_vviol  = 0

        # Save final checkpoint
        ckpt_path = os.path.join(self.save_dir, "agent_final.pt")
        self.agent.save(ckpt_path)
        print(f"\nCheckpoint saved → {ckpt_path}")
        return metrics


def compare_agents(
    env,
    agents: dict[str, object],
    n_eval_episodes: int = 10,
    device: str = "cpu",
) -> dict[str, dict[str, float]]:
    """
    Evaluate multiple trained agents on the same environment.

    Parameters
    ----------
    env     : VVC gymnasium environment (shared, reset between agents)
    agents  : dict of {name: agent}
    n_eval_episodes : episodes per agent

    Returns
    -------
    results : dict of {name: {"mean_reward": ..., "mean_v_viols": ..., "n_params": ...}}
    """
    results = {}
    for name, agent in agents.items():
        eval_res = evaluate_policy(env, agent, n_episodes=n_eval_episodes, device=device)
        eval_res["n_params"] = agent.param_count()
        results[name] = eval_res
        print(
            f"  {name:20s} | reward {eval_res['mean_reward']:8.3f} | "
            f"vviol {eval_res['mean_v_viols']:5.2f} | "
            f"params {eval_res['n_params']:,}"
        )
    return results


def save_results(results: dict, path: str) -> None:
    """Persist comparison results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")
