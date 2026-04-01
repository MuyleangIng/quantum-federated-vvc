"""
Run QESACAgentConstrained on VVCEnv13Bus for seeds 1 and 2
(seed 0 checkpoint already exists from notebook).

Usage:
    python scripts/run_constrained_sac.py
"""

import sys
import json
import os

sys.path.insert(0, "/root/power-system")

import numpy as np
import torch

from src.qe_sac.env_utils import VVCEnv13Bus
from src.qe_sac.constrained_sac import QESACAgentConstrained
from src.qe_sac.trainer import QESACTrainer

SEEDS      = [1, 2]   # seed 0 already done
N_STEPS    = 50_000
BATCH_SIZE = 256
WARMUP     = 1_000
CAE_INTERVAL = 500
SAVE_DIR   = "artifacts/qe_sac"

os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    all_results = {}

    # Load seed 0 if it exists
    seed0_ckpt = os.path.join(SAVE_DIR, "qe_sac_constrained_seed0.pt")
    if os.path.exists(seed0_ckpt):
        print(f"Seed 0 checkpoint found: {seed0_ckpt}")

    for seed in SEEDS:
        env = VVCEnv13Bus(seed=seed)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.prod())

        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = QESACAgentConstrained(
            obs_dim=obs_dim,
            n_actions=n_actions,
            buffer_size=1_000_000,
            lambda_lr=0.01,
        )

        trainer = QESACTrainer(
            agent, env,
            batch_size=BATCH_SIZE,
            cae_update_interval=CAE_INTERVAL,
            warmup_steps=WARMUP,
            log_interval=100,
            save_dir=SAVE_DIR,
        )

        print(f"\n{'='*60}")
        print(f"  Constrained SAC  |  seed={seed}  |  obs={obs_dim}-dim")
        print(f"{'='*60}")
        metrics = trainer.train(n_steps=N_STEPS)

        ckpt = os.path.join(SAVE_DIR, f"qe_sac_constrained_seed{seed}.pt")
        agent.save(ckpt)
        print(f"Saved → {ckpt}")

        ep_rewards = np.array(metrics.episode_rewards)
        ep_vviols  = np.array(metrics.episode_vviols)
        all_results[f"seed{seed}"] = {
            "mean_reward": float(ep_rewards.mean()),
            "std_reward":  float(ep_rewards.std()),
            "mean_vviol":  float(ep_vviols.mean()),
            "final_lambda": float(agent.lagrange_lambda),
        }
        print(f"  reward={all_results[f'seed{seed}']['mean_reward']:.2f}"
              f"  vviol={all_results[f'seed{seed}']['mean_vviol']:.2f}"
              f"  λ={all_results[f'seed{seed}']['final_lambda']:.4f}")

    out = os.path.join(SAVE_DIR, "constrained_sac_seeds12.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
