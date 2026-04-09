"""
Training loop for all 4 QE-SAC paper algorithms.

Supports: QESACAgent, QCSACAgent, ClassicalSACAgent, SACAEAgent.

All agents use factorized per-device actions (MultiDiscrete array), so no
scalar action encoding/decoding is needed here.
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch

from .metrics import TrainingMetrics, evaluate_policy


class QESACTrainer:
    """
    Unified training loop for all 4 paper algorithms.

    Parameters
    ----------
    agent          : any of QESACAgent / QCSACAgent / ClassicalSACAgent / SACAEAgent
    env            : Gymnasium VVC environment (MultiDiscrete action space)
    batch_size     : SAC mini-batch size (paper: 256)
    cae_update_interval : co-adaptive CAE retrain interval (paper: C=500)
    warmup_steps   : random exploration steps before learning starts
    log_interval   : print progress every N episodes
    save_dir       : directory for checkpoints
    device         : torch device
    """

    def __init__(
        self,
        agent,
        env,
        batch_size:          int = 256,
        cae_update_interval: int = 500,
        warmup_steps:        int = 1000,
        log_interval:        int = 50,
        save_dir:            str = "artifacts/qe_sac",
        device:              str = "cpu",
        agent_name:          str = "Agent",
    ):
        self.agent        = agent
        self.env          = env
        self.batch        = batch_size
        self.cae_interval = cae_update_interval
        self.warmup       = warmup_steps
        self.log_interval = log_interval
        self.save_dir     = save_dir
        self.device       = device
        self.agent_name   = agent_name
        os.makedirs(save_dir, exist_ok=True)

    def train(self, n_steps: int = 50_000) -> TrainingMetrics:
        """Train for *n_steps* environment steps. Returns episode-level metrics."""
        metrics     = TrainingMetrics()
        obs, _      = self.env.reset()
        ep_reward   = 0.0
        ep_vviol    = 0
        total_steps = 0

        while total_steps < n_steps:
            # ── Action selection ───────────────────────────────────────────
            if total_steps < self.warmup:
                action = self.env.action_space.sample()          # MultiDiscrete array
            else:
                with torch.no_grad():
                    t_obs  = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    action = self.agent.select_action(t_obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done         = terminated or truncated
            v_viol_step  = info.get("v_viol", 0)

            self.agent.store(obs, action, reward, next_obs, done, v_viol=v_viol_step)
            ep_reward   += reward
            ep_vviol    += v_viol_step
            obs          = next_obs
            total_steps += 1

            # ── Learning update ────────────────────────────────────────────
            if total_steps >= self.warmup:
                logs = self.agent.update(
                    batch_size          = self.batch,
                    cae_update_interval = self.cae_interval,
                )
                if logs:
                    metrics.record_losses(
                        actor  = logs.get("actor_loss"),
                        critic = logs.get("critic_loss"),
                        cae    = logs.get("cae_loss"),
                    )

            # ── Episode bookkeeping ────────────────────────────────────────
            if done:
                metrics.record_episode(ep_reward, ep_vviol)
                ep_idx = len(metrics.episode_rewards)

                if ep_idx % self.log_interval == 0:
                    print(
                        f"[{self.agent_name}] ep {ep_idx:4d} | steps {total_steps:6d} | "
                        f"reward {ep_reward:8.3f} | vviol {ep_vviol:3d} | "
                        f"mean100 {metrics.mean_reward(100):8.3f}"
                    )
                obs, _    = self.env.reset()
                ep_reward = 0.0
                ep_vviol  = 0

        ckpt_path = os.path.join(self.save_dir, "agent_final.pt")
        self.agent.save(ckpt_path)
        print(f"\nCheckpoint saved → {ckpt_path}")
        return metrics


def compare_agents(
    env,
    agents:          dict[str, object],
    n_eval_episodes: int = 10,
    device:          str = "cpu",
) -> dict[str, dict[str, float]]:
    """Evaluate multiple trained agents on the same environment."""
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")
