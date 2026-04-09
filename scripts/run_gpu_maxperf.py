"""
Maximum GPU utilization — all 3× RTX 4090 at full load.

Strategy: each GPU gets ONE agent, and runs all 3 seeds IN PARALLEL
(instead of sequentially). This triples GPU utilization vs the default script.

GPU assignment:
    GPU 0 → QE-SAC        (seeds 0,1,2 in parallel)
    GPU 1 → Classical-SAC (seeds 0,1,2 in parallel)
    GPU 1 → QC-SAC        (seeds 0,1,2 in parallel — tiny model, shares with Classical)
    GPU 2 → SAC-AE        (seeds 0,1,2 in parallel)

Memory estimate per GPU:
    QE-SAC:        ~0.7 Gi × 3 seeds = ~2.1 Gi  (GPU 0 has 24 Gi → fine)
    Classical-SAC: ~0.4 Gi × 3 seeds = ~1.2 Gi + QC-SAC ~0.1 Gi → fine
    SAC-AE:        ~0.4 Gi × 3 seeds = ~1.2 Gi  → fine

Usage:
    nohup python scripts/run_gpu_maxperf.py > logs/maxperf.log 2>&1 &
    tail -f logs/maxperf.log
"""

import sys
import os
import json
import time
import multiprocessing as mp

sys.path.insert(0, "/root/power-system")

import numpy as np
import torch

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac.qe_sac_policy import QESACAgent, QCSACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent, SACAEAgent
from src.qe_sac.trainer import QESACTrainer
from src.qe_sac.metrics import evaluate_policy

# ── Hyperparameters — paper-exact ─────────────────────────────────────────────
SEEDS        = [0, 1, 2]
N_STEPS      = 240_000
BATCH_SIZE   = 256
WARMUP       = 1_000
CAE_INTERVAL = 500
CAE_COLLECT  = 5_000
CAE_PRETRAIN = 200
LR           = 1e-4
GAMMA        = 0.99
TAU          = 0.005
ALPHA        = 0.2
BUFFER_SIZE  = 1_000_000
N_EVAL_EPS   = 10

SAVE_DIR = "artifacts/qe_sac_paper/opendss_gpu"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# One GPU per agent (QC-SAC shares GPU 1 with Classical-SAC — it's tiny)
AGENT_GPU = {
    "QE-SAC":        0,
    "Classical-SAC": 1,
    "QC-SAC":        1,
    "SAC-AE":        2,
}


def run_one_seed(agent_name, AgentClass, gpu_id, seed, return_dict):
    """
    Train one agent on one seed. Runs inside a subprocess on gpu_id.
    Multiple seeds run in parallel on the same GPU.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env         = VVCEnvOpenDSS(seed=seed)
    obs_dim     = env.observation_space.shape[0]
    device_dims = list(map(int, env.action_space.nvec))

    agent = AgentClass(
        obs_dim     = obs_dim,
        device_dims = device_dims,
        lr          = LR,
        gamma       = GAMMA,
        tau         = TAU,
        alpha       = ALPHA,
        buffer_size = BUFFER_SIZE,
        device      = device,
    )

    if hasattr(agent, "pretrain_cae"):
        agent.pretrain_cae(env, n_collect=CAE_COLLECT, n_train_steps=CAE_PRETRAIN)
        env = VVCEnvOpenDSS(seed=seed)
    if hasattr(agent, "pretrain_pca"):
        agent.pretrain_pca(env, n_collect=CAE_COLLECT)
        env = VVCEnvOpenDSS(seed=seed)

    trainer = QESACTrainer(
        agent, env,
        batch_size          = BATCH_SIZE,
        cae_update_interval = CAE_INTERVAL,
        warmup_steps        = WARMUP,
        log_interval        = 5_000,
        save_dir            = SAVE_DIR,
        device              = device,
    )

    t0 = time.time()
    print(f"[GPU{gpu_id}|{agent_name}|s{seed}] START  params={agent.param_count():,}", flush=True)
    trainer.train(n_steps=N_STEPS)

    tag  = agent_name.lower().replace("-", "_")
    ckpt = os.path.join(SAVE_DIR, f"{tag}_seed{seed}.pt")
    agent.save(ckpt)

    result  = evaluate_policy(VVCEnvOpenDSS(seed=seed + 100), agent,
                               n_episodes=N_EVAL_EPS, device=device)
    elapsed = (time.time() - t0) / 60
    print(f"[GPU{gpu_id}|{agent_name}|s{seed}] DONE  "
          f"reward={result['mean_reward']:.3f}  "
          f"vviol={result['mean_v_viols']:.1f}  "
          f"time={elapsed:.1f}min", flush=True)

    key = f"{agent_name}_seed{seed}"
    return_dict[key] = {
        "reward": result["mean_reward"],
        "vviol":  result["mean_v_viols"],
        "params": agent.param_count(),
        "device": device,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")

    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  MAX-PERF GPU RUN — {n_gpus}× RTX 4090")
    print(f"  Strategy: all seeds in parallel per GPU")
    print(f"  4 agents × 3 seeds = 12 parallel jobs")
    print(f"{'='*60}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    agents_cfg = [
        ("QE-SAC",        QESACAgent),
        ("Classical-SAC", ClassicalSACAgent),
        ("QC-SAC",        QCSACAgent),
        ("SAC-AE",        SACAEAgent),
    ]

    manager     = mp.Manager()
    return_dict = manager.dict()
    processes   = []

    # Launch one process per (agent, seed) — all in parallel
    for agent_name, AgentClass in agents_cfg:
        gpu_id = AGENT_GPU[agent_name] % n_gpus
        for seed in SEEDS:
            p = mp.Process(
                target = run_one_seed,
                args   = (agent_name, AgentClass, gpu_id, seed, return_dict),
            )
            p.start()
            processes.append((agent_name, seed, p))
            print(f"  Launched [{agent_name} s{seed}] → GPU {gpu_id}")

    print(f"\n  {len(processes)} processes running. Waiting...\n")

    for agent_name, seed, p in processes:
        p.join()
        print(f"  [{agent_name} s{seed}] finished")

    # ── Aggregate results per agent ───────────────────────────────────────────
    results = {}
    for agent_name, _ in agents_cfg:
        seed_rewards = [return_dict[f"{agent_name}_seed{s}"]["reward"] for s in SEEDS]
        seed_vviols  = [return_dict[f"{agent_name}_seed{s}"]["vviol"]  for s in SEEDS]
        params       =  return_dict[f"{agent_name}_seed{SEEDS[0]}"]["params"]
        device       =  return_dict[f"{agent_name}_seed{SEEDS[0]}"]["device"]
        results[agent_name] = {
            "mean":   float(np.mean(seed_rewards)),
            "std":    float(np.std(seed_rewards)),
            "seeds":  seed_rewards,
            "vviols": seed_vviols,
            "params": params,
            "obs_dim": 48,
            "device": device,
            "env":    "OpenDSS 3-phase AC (IEEE 13-bus)",
        }

    out = os.path.join(SAVE_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  RESULTS — OpenDSS IEEE 13-bus  ({len(SEEDS)} seeds × {N_STEPS:,} steps)")
    print(f"{'='*72}")
    print(f"  {'Agent':20s}  {'Mean':>8s}  {'±Std':>7s}  {'Params':>10s}  GPU")
    print(f"  {'-'*60}")
    for name, r in results.items():
        print(f"  {name:20s}  {r['mean']:+8.3f}  ±{r['std']:6.3f}  "
              f"{r['params']:>10,}  {r['device']}")
    print(f"\n  Paper (Lin et al. 2025):")
    print(f"  {'QE-SAC':20s}  {'−5.390':>8s}")
    print(f"  {'Classical-SAC':20s}  {'−5.410':>8s}")
    print(f"{'='*72}")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
