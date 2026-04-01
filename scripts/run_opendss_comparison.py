"""
Run QE-SAC and Classical SAC on VVCEnvOpenDSS (real 3-phase AC, 93-dim obs).
Saves results to artifacts/qe_sac/opendss_results_13bus.json for direct
comparison with paper (which also uses OpenDSS).

Usage:
    python scripts/run_opendss_comparison.py
"""

import sys
import json
import os

sys.path.insert(0, "/root/power-system")

import numpy as np
import torch

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac.qe_sac_policy import QESACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent
from src.qe_sac.trainer import QESACTrainer

SEEDS      = [0, 1, 2]
N_STEPS    = 50_000
BATCH_SIZE = 256
WARMUP     = 1_000
CAE_INTERVAL = 500
SAVE_DIR   = "artifacts/qe_sac"

os.makedirs(SAVE_DIR, exist_ok=True)


def run_agent(AgentClass, env_seed, train_seed, agent_name, extra_kwargs=None):
    """Train one agent on OpenDSS env, return episode metrics."""
    extra_kwargs = extra_kwargs or {}
    env = VVCEnvOpenDSS(seed=env_seed)
    obs_dim   = env.observation_space.shape[0]   # 93
    n_actions = int(env.action_space.nvec.prod()) # 2*2*33 = 132

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    agent = AgentClass(
        obs_dim=obs_dim,
        n_actions=n_actions,
        buffer_size=200_000,
        **extra_kwargs,
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
    print(f"  {agent_name}  |  seed={train_seed}  |  obs={obs_dim}-dim  |  n_actions={n_actions}")
    print(f"{'='*60}")
    metrics = trainer.train(n_steps=N_STEPS)

    # Save checkpoint
    ckpt = os.path.join(SAVE_DIR, f"opendss_{agent_name.lower().replace(' ','_')}_seed{train_seed}.pt")
    agent.save(ckpt)

    ep_rewards = np.array(metrics.episode_rewards)
    ep_vviols  = np.array(metrics.episode_vviols)
    return {
        "mean_reward": float(ep_rewards.mean()),
        "std_reward":  float(ep_rewards.std()),
        "mean_vviol":  float(ep_vviols.mean()),
        "total_vviols": int(ep_vviols.sum()),
        "n_params": agent.param_count(),
        "obs_dim": obs_dim,
    }


def main():
    results = {}

    # --- QE-SAC on OpenDSS ---
    qe_rewards = []
    qe_vviols  = []
    for s in SEEDS:
        r = run_agent(QESACAgent, env_seed=s, train_seed=s, agent_name="QE-SAC OpenDSS")
        qe_rewards.append(r["mean_reward"])
        qe_vviols.append(r["mean_vviol"])
        print(f"  Seed {s}: reward={r['mean_reward']:.2f}, vviol={r['mean_vviol']:.2f}")

    results["QE-SAC (OpenDSS)"] = {
        "mean_reward": float(np.mean(qe_rewards)),
        "std_reward":  float(np.std(qe_rewards)),
        "mean_vviol":  float(np.mean(qe_vviols)),
        "n_params":    11430,
        "obs_dim":     93,
        "seeds":       qe_rewards,
    }

    # --- Classical SAC on OpenDSS ---
    cl_rewards = []
    cl_vviols  = []
    for s in SEEDS:
        r = run_agent(ClassicalSACAgent, env_seed=s, train_seed=s, agent_name="Classical SAC OpenDSS")
        cl_rewards.append(r["mean_reward"])
        cl_vviols.append(r["mean_vviol"])
        print(f"  Seed {s}: reward={r['mean_reward']:.2f}, vviol={r['mean_vviol']:.2f}")

    results["Classical SAC (OpenDSS)"] = {
        "mean_reward": float(np.mean(cl_rewards)),
        "std_reward":  float(np.std(cl_rewards)),
        "mean_vviol":  float(np.mean(cl_vviols)),
        "n_params":    110724,
        "obs_dim":     93,
        "seeds":       cl_rewards,
    }

    # Save
    out = os.path.join(SAVE_DIR, "opendss_results_13bus.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")

    # Print comparison
    print("\n" + "="*70)
    print("  OPENDSS RESULTS — IEEE 13-bus (3 seeds × 50K steps)")
    print("="*70)
    for name, r in results.items():
        print(f"  {name:30s}  reward={r['mean_reward']:8.2f} ±{r['std_reward']:.2f}"
              f"  vviol={r['mean_vviol']:.1f}  params={r['n_params']:,}")

    # DistFlow reference
    try:
        with open(os.path.join(SAVE_DIR, "results_13bus.json")) as f:
            df = json.load(f)
        print("\n  DistFlow reference (42-dim, same seeds):")
        for name, r in df.items():
            print(f"  {name:30s}  reward={r['mean_reward']:8.2f} ±{r['std_reward']:.2f}")
    except Exception:
        pass

    print("="*70)


if __name__ == "__main__":
    main()
