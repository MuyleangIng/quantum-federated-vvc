"""
QFL Baseline — Quantum Federated Learning without SAC.

Runs QFL (REINFORCE + VQC + FedAvg) as a standalone baseline
to compare against QE-SAC-FL (aligned_fl).

Paper comparison:
    QE-SAC  (local_only)  — quantum RL, no federation
    QFL     (this script) — quantum FL, no actor-critic
    QE-SAC-FL (aligned)   — proposed: quantum RL + aligned FL

Run:
    python -u scripts/run_qfl_baseline.py > logs/qfl_baseline.log 2>&1 &
"""

import sys
import os
sys.path.insert(0, "/root/power-system")
os.makedirs("artifacts/qfl_baseline", exist_ok=True)
os.makedirs("logs", exist_ok=True)

import json
import time
import numpy as np
import torch

from src.qe_sac_fl.qfl_agent import QFLAgent, fedavg_qfl
from src.qe_sac_fl.federated_trainer import _make_env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS        = [0, 1, 2]
N_ROUNDS     = 50
LOCAL_STEPS  = 1_000
LR           = 3e-4
GAMMA        = 0.99
HIDDEN_DIM   = 32
LOG_INTERVAL = 10

CLIENTS = [
    dict(name="Utility_A_13bus",  env_id="13bus_fl",  obs_dim=43,  device_dims=[2,2,33,33],          seed=0, reward_scale=50.0),
    dict(name="Utility_B_34bus",  env_id="34bus_fl",  obs_dim=113, device_dims=[2,2,33,33,33],        seed=1, reward_scale=10.0),
    dict(name="Utility_C_123bus", env_id="123bus_fl", obs_dim=349, device_dims=[2,2,33,33,33,33,33],  seed=2, reward_scale=750.0),
]


# ---------------------------------------------------------------------------
# One FL seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  QFL Baseline — SEED {seed}", flush=True)
    print(f"  {N_ROUNDS} rounds × {LOCAL_STEPS} steps, REINFORCE + FedAvg", flush=True)
    print(f"{'='*60}", flush=True)

    n_gpus  = torch.cuda.device_count()
    devices = [f"cuda:{i}" if i < n_gpus else "cpu" for i in range(len(CLIENTS))]

    # Build agents and environments
    agents, envs = [], []
    for i, cfg in enumerate(CLIENTS):
        client_seed = cfg["seed"] + seed * 10
        env   = _make_env(cfg["env_id"], client_seed, cfg["reward_scale"])
        agent = QFLAgent(
            obs_dim     = cfg["obs_dim"],
            device_dims = cfg["device_dims"],
            lr          = LR,
            gamma       = GAMMA,
            hidden_dim  = HIDDEN_DIM,
            device      = devices[i],
        )
        # Pre-train CAE encoder before FL
        print(f"  Pretraining CAE: {cfg['name']} ...", flush=True)
        agent.pretrain_cae(env)
        agents.append(agent)
        envs.append(env)

    logs = []
    t0   = time.time()

    for rnd in range(N_ROUNDS):
        # --- Local training ---
        round_rewards, round_grads = [], []
        for agent, env, cfg in zip(agents, envs, CLIENTS):
            reward, vqc_grad = agent.train_round(env, LOCAL_STEPS)
            round_rewards.append(reward)
            round_grads.append(vqc_grad)
            logs.append({
                "client":       cfg["name"],
                "round":        rnd,
                "reward":       reward,
                "vqc_grad_norm": vqc_grad,
                "steps":        LOCAL_STEPS,
            })

        # --- FedAvg on SharedHead + VQC ---
        weight_list = [a.get_shared_weights() for a in agents]
        global_w    = fedavg_qfl(weight_list)
        for agent in agents:
            agent.set_shared_weights(global_w)

        if (rnd + 1) % LOG_INTERVAL == 0 or rnd == N_ROUNDS - 1:
            parts = "  |  ".join(
                f"{cfg['name'].split('_')[1]}:{r:.2f}"
                for cfg, r in zip(CLIENTS, round_rewards)
            )
            print(f"  round {rnd+1:3d}/{N_ROUNDS}  |  {parts}", flush=True)

    wall = time.time() - t0

    # --- Summary ---
    print(f"\n  Wall time: {wall/3600:.1f} h", flush=True)
    print(f"  Final rewards:", flush=True)
    for cfg, log_entry in zip(CLIENTS, [next(l for l in reversed(logs) if l["client"] == cfg["name"]) for cfg in CLIENTS]):
        print(f"    {cfg['name']:25s}: {log_entry['reward']:.4f}", flush=True)

    return {
        "condition":          "qfl_baseline",
        "seed":               seed,
        "n_rounds":           N_ROUNDS,
        "local_steps":        LOCAL_STEPS,
        "wall_time_seconds":  wall,
        "bytes_communicated": len(CLIENTS) * N_ROUNDS * 280 * 4 * 2,  # 280 params, float32, up+down
        "logs":               logs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("  QFL Baseline (REINFORCE + VQC + FedAvg)", flush=True)
    print("  Comparison: QFL vs QE-SAC-FL", flush=True)
    print("  Reference: Chen & Yoo (2021) Federated QML", flush=True)
    print("=" * 60, flush=True)

    all_results = {}

    for seed in SEEDS:
        result = run_seed(seed)
        out_path = f"artifacts/qfl_baseline/seed{seed}_qfl.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved → {out_path}", flush=True)
        all_results[seed] = {
            cfg["name"]: next(
                l["reward"] for l in reversed(result["logs"])
                if l["client"] == cfg["name"]
            )
            for cfg in CLIENTS
        }

    # --- Final comparison table ---
    print(f"\n{'='*60}", flush=True)
    print(f"  QFL Final Rewards (n={len(SEEDS)} seeds)", flush=True)
    print(f"{'='*60}", flush=True)
    for cfg in CLIENTS:
        vals = [all_results[s][cfg["name"]] for s in SEEDS]
        print(f"  {cfg['name']:25s}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}", flush=True)

    # Save summary
    summary = {
        cfg["name"]: {
            "mean": float(np.mean([all_results[s][cfg["name"]] for s in SEEDS])),
            "std":  float(np.std( [all_results[s][cfg["name"]] for s in SEEDS], ddof=1)),
            "n":    len(SEEDS),
            "seeds": [all_results[s][cfg["name"]] for s in SEEDS],
        }
        for cfg in CLIENTS
    }
    with open("artifacts/qfl_baseline/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary → artifacts/qfl_baseline/summary.json", flush=True)


if __name__ == "__main__":
    main()
