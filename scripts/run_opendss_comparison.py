"""
Run all 4 paper agents on VVCEnvOpenDSS (real 3-phase AC, 48-dim obs).

Agents compared (matches Lin et al. 2025 Table 4):
    QE-SAC       — CAE + VQC + factorized heads  (proposed)
    Classical-SAC — MLP(256→256) + factorized heads
    SAC-AE        — CAE + tiny MLP + factorized heads  (no VQC ablation)
    QC-SAC        — fixed PCA + VQC + factorized heads (no co-adapt ablation)

Environment: VVCEnvOpenDSS — real OpenDSS 3-phase AC (IEEE 13-bus)
             obs_dim=48, action=MultiDiscrete([2,2,33,33,33,33])

Paper hyperparameters (Lin et al. 2025, Table 3):
    lr            = 1e-4
    gamma         = 0.99
    tau           = 0.005
    alpha         = 0.2  (fixed entropy coefficient)
    batch_size    = 256
    buffer_size   = 1,000,000
    CAE interval  = 500  (C in Algorithm 1)
    CAE pre-train = 5,000 random steps + 200 gradient steps
    warmup        = 1,000 random steps before learning

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
from src.qe_sac.qe_sac_policy import QESACAgent, QCSACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent, SACAEAgent
from src.qe_sac.trainer import QESACTrainer
from src.qe_sac.metrics import evaluate_policy

# ── Hyperparameters — paper-exact (Lin et al. 2025, Table 3) ─────────────────
SEEDS        = [0, 1, 2]    # 3 seeds; set to list(range(10)) for full paper run
N_STEPS      = 240_000      # paper: 10,000 episodes × 24 steps = 240,000
BATCH_SIZE   = 256          # paper: 256
WARMUP       = 1_000        # random steps before learning starts
CAE_INTERVAL = 500          # paper: C=500 co-adaptive CAE update interval
CAE_COLLECT  = 5_000        # paper: 5000 random obs for CAE pre-training
CAE_PRETRAIN = 200          # paper: 200 gradient steps for CAE pre-training
LR           = 1e-4         # paper: 1e-4
GAMMA        = 0.99         # paper: 0.99
TAU          = 0.005        # paper: τ=0.005
ALPHA        = 0.2          # paper: α=0.2 (fixed entropy coefficient)
BUFFER_SIZE  = 1_000_000    # paper: 1M replay buffer
N_EVAL_EPS   = 10           # evaluation episodes after training

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "artifacts/qe_sac_paper/opendss"
os.makedirs(SAVE_DIR, exist_ok=True)


def make_env(seed: int) -> VVCEnvOpenDSS:
    return VVCEnvOpenDSS(seed=seed)


def run_agent(AgentClass, seed: int, agent_name: str):
    """Train one agent on OpenDSS env for N_STEPS, return eval metrics."""

    env = make_env(seed)
    obs_dim     = env.observation_space.shape[0]          # 48
    device_dims = list(map(int, env.action_space.nvec))   # [2,2,33,33,33,33]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Instantiate agent with paper-exact hyperparams ────────────────────────
    agent = AgentClass(
        obs_dim     = obs_dim,
        device_dims = device_dims,
        lr          = LR,
        gamma       = GAMMA,
        tau         = TAU,
        alpha       = ALPHA,
        buffer_size = BUFFER_SIZE,
        device      = DEVICE,
    )

    # ── CAE / PCA pre-training (Algorithm 1, pre-step) ────────────────────────
    if isinstance(agent, QESACAgent):
        # Paper: collect 5000 random observations, train CAE for 200 steps
        agent.pretrain_cae(env, n_collect=CAE_COLLECT, n_train_steps=CAE_PRETRAIN)
        env = make_env(seed)   # reset env after random collection

    if isinstance(agent, QCSACAgent):
        # QC-SAC: fit PCA offline on 5000 random observations (then frozen)
        agent.pretrain_pca(env, n_collect=CAE_COLLECT)
        env = make_env(seed)

    if isinstance(agent, SACAEAgent):
        # SAC-AE: same CAE pre-training as QE-SAC
        agent.pretrain_cae(env, n_collect=CAE_COLLECT, n_train_steps=CAE_PRETRAIN)
        env = make_env(seed)

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = QESACTrainer(
        agent, env,
        batch_size          = BATCH_SIZE,
        cae_update_interval = CAE_INTERVAL,
        warmup_steps        = WARMUP,
        log_interval        = 50,
        save_dir            = SAVE_DIR,
        device              = DEVICE,
    )

    print(f"\n{'='*60}")
    print(f"  {agent_name}  seed={seed}  obs={obs_dim}  devices={device_dims}")
    print(f"  params={agent.param_count():,}  steps={N_STEPS:,}")
    print(f"  lr={LR}  gamma={GAMMA}  tau={TAU}  alpha={ALPHA}  buf={BUFFER_SIZE:,}")
    print(f"{'='*60}")

    trainer.train(n_steps=N_STEPS)

    # Save checkpoint
    tag = agent_name.lower().replace(" ", "_").replace("-", "_")
    ckpt = os.path.join(SAVE_DIR, f"{tag}_seed{seed}.pt")
    agent.save(ckpt)

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_env = make_env(seed + 100)
    result = evaluate_policy(eval_env, agent, n_episodes=N_EVAL_EPS, device="cpu")
    result["n_params"] = agent.param_count()
    result["obs_dim"]  = obs_dim
    print(f"  eval  reward={result['mean_reward']:.3f}  vviol={result['mean_v_viols']:.1f}")
    return result


def main():
    agents_cfg = [
        ("QE-SAC",        QESACAgent),
        ("Classical-SAC", ClassicalSACAgent),
        ("SAC-AE",        SACAEAgent),
        ("QC-SAC",        QCSACAgent),
    ]

    results = {}

    for agent_name, AgentClass in agents_cfg:
        seed_rewards = []
        seed_vviols  = []
        n_params     = 0

        for seed in SEEDS:
            r = run_agent(AgentClass, seed, agent_name)
            seed_rewards.append(r["mean_reward"])
            seed_vviols.append(r["mean_v_viols"])
            n_params = r["n_params"]

        results[agent_name] = {
            "mean":    float(np.mean(seed_rewards)),
            "std":     float(np.std(seed_rewards)),
            "seeds":   seed_rewards,
            "vviols":  seed_vviols,
            "params":  n_params,
            "obs_dim": r["obs_dim"],
            "env":     "OpenDSS 3-phase AC (IEEE 13-bus)",
        }
        print(f"\n  {agent_name}: mean={results[agent_name]['mean']:.3f} "
              f"std=±{results[agent_name]['std']:.3f}  params={n_params:,}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = os.path.join(SAVE_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  RESULTS vs PAPER — OpenDSS IEEE 13-bus")
    print(f"  ({len(SEEDS)} seeds × {N_STEPS:,} steps,  lr={LR}, γ={GAMMA}, τ={TAU}, α={ALPHA})")
    print("="*72)
    print(f"  {'Agent':20s}  {'Mean':>8s}  {'Std':>7s}  {'Params':>10s}")
    print("-"*72)
    for name, r in results.items():
        print(f"  {name:20s}  {r['mean']:8.3f}  ±{r['std']:6.3f}  {r['params']:>10,}")

    print("\n  Paper reference (Lin et al. 2025 — full PowerGym OpenDSS):")
    print(f"  {'QE-SAC':20s}  {'−5.390':>8s}  {'':>7s}  {'4,872':>10s}")
    print(f"  {'Classical-SAC':20s}  {'−5.410':>8s}  {'':>7s}  {'899,729':>10s}")
    print("="*72)


if __name__ == "__main__":
    main()
