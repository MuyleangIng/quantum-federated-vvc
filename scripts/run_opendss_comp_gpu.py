"""
GPU version — runs all 4 agents in parallel across 3× RTX 4090.

Agent assignment:
    GPU 0 → QE-SAC
    GPU 1 → Classical-SAC
    GPU 2 → SAC-AE  (already running here)
    GPU 0 → QC-SAC  (after QE-SAC finishes, or run sequentially)

Each agent runs all 3 seeds on its assigned GPU simultaneously using
multiprocessing. Results saved to:
    artifacts/qe_sac_paper/opendss_gpu/results.json

Usage:
    python scripts/run_opendss_comp_gpu.py
"""

import sys
import json
import os
import multiprocessing as mp

sys.path.insert(0, "/root/power-system")

import numpy as np
import torch

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac.qe_sac_policy import QESACAgent, QCSACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent, SACAEAgent
from src.qe_sac.trainer import QESACTrainer
from src.qe_sac.metrics import evaluate_policy

# ── Hyperparameters — paper-exact (Lin et al. 2025, Table 3) ─────────────────
SEEDS        = [0, 1, 2]
N_STEPS      = 240_000      # paper: 10,000 episodes × 24 steps = 240,000
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

# Assign each agent to a GPU
AGENT_GPU = {
    "QE-SAC":        0,
    "Classical-SAC": 1,
    "SAC-AE":        2,
    "QC-SAC":        0,   # shares GPU 0 with QE-SAC (runs after)
}


def make_env(seed: int) -> VVCEnvOpenDSS:
    return VVCEnvOpenDSS(seed=seed)


def run_agent_on_gpu(agent_name, AgentClass, gpu_id, return_dict):
    """Run all seeds for one agent on a specific GPU. Called in subprocess."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    seed_rewards = []
    seed_vviols  = []
    n_params     = 0

    for seed in SEEDS:
        env = make_env(seed)
        obs_dim     = env.observation_space.shape[0]
        device_dims = list(map(int, env.action_space.nvec))

        torch.manual_seed(seed)
        np.random.seed(seed)

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

        if isinstance(agent, QESACAgent):
            agent.pretrain_cae(env, n_collect=CAE_COLLECT, n_train_steps=CAE_PRETRAIN)
            env = make_env(seed)

        if isinstance(agent, QCSACAgent):
            agent.pretrain_pca(env, n_collect=CAE_COLLECT)
            env = make_env(seed)

        if isinstance(agent, SACAEAgent):
            agent.pretrain_cae(env, n_collect=CAE_COLLECT, n_train_steps=CAE_PRETRAIN)
            env = make_env(seed)

        trainer = QESACTrainer(
            agent, env,
            batch_size          = BATCH_SIZE,
            cae_update_interval = CAE_INTERVAL,
            warmup_steps        = WARMUP,
            log_interval        = 50,
            save_dir            = SAVE_DIR,
            device              = device,
            agent_name          = f"{agent_name} seed{seed}",
        )

        print(f"\n[GPU{gpu_id}] {agent_name}  seed={seed}  params={agent.param_count():,}")
        trainer.train(n_steps=N_STEPS)

        tag  = agent_name.lower().replace(" ", "_").replace("-", "_")
        ckpt = os.path.join(SAVE_DIR, f"{tag}_seed{seed}.pt")
        agent.save(ckpt)

        eval_env = make_env(seed + 100)
        result   = evaluate_policy(eval_env, agent, n_episodes=N_EVAL_EPS, device=device)
        seed_rewards.append(result["mean_reward"])
        seed_vviols.append(result["mean_v_viols"])
        n_params = agent.param_count()
        print(f"[GPU{gpu_id}] {agent_name} seed={seed}  reward={result['mean_reward']:.3f}")

    return_dict[agent_name] = {
        "mean":    float(np.mean(seed_rewards)),
        "std":     float(np.std(seed_rewards)),
        "seeds":   seed_rewards,
        "vviols":  seed_vviols,
        "params":  n_params,
        "obs_dim": obs_dim,
        "device":  device,
        "env":     "OpenDSS 3-phase AC (IEEE 13-bus)",
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s):")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    agents_cfg = [
        ("QE-SAC",        QESACAgent),
        ("Classical-SAC", ClassicalSACAgent),
        ("SAC-AE",        SACAEAgent),
        ("QC-SAC",        QCSACAgent),
    ]

    manager     = mp.Manager()
    return_dict = manager.dict()
    processes   = []

    for agent_name, AgentClass in agents_cfg:
        gpu_id = AGENT_GPU[agent_name] % n_gpus
        p = mp.Process(
            target = run_agent_on_gpu,
            args   = (agent_name, AgentClass, gpu_id, return_dict),
        )
        p.start()
        processes.append(p)
        print(f"  Launched {agent_name} → GPU {gpu_id}")

    for p in processes:
        p.join()

    # ── Save & print results ──────────────────────────────────────────────────
    results = dict(return_dict)
    out = os.path.join(SAVE_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")

    print("\n" + "="*72)
    print("  GPU RESULTS vs PAPER — OpenDSS IEEE 13-bus")
    print(f"  ({len(SEEDS)} seeds × {N_STEPS:,} steps  |  3× RTX 4090)")
    print("="*72)
    print(f"  {'Agent':20s}  {'Mean':>8s}  {'Std':>7s}  {'Params':>10s}  GPU")
    print("-"*72)
    for name, r in results.items():
        print(f"  {name:20s}  {r['mean']:8.3f}  ±{r['std']:6.3f}  "
              f"{r['params']:>10,}  {r['device']}")

    print("\n  Paper reference (Lin et al. 2025):")
    print(f"  {'QE-SAC':20s}  {'−5.390':>8s}  {'':>7s}  {'4,872':>10s}")
    print(f"  {'Classical-SAC':20s}  {'−5.410':>8s}  {'':>7s}  {'899,729':>10s}")
    print("="*72)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
