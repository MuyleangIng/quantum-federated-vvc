"""
13-bus paper-exact comparison — Lin et al. (2025), Table 2 settings.

All 4 agents run in parallel across 3× RTX 4090:
  GPU 0 → QE-SAC   (3 seeds)
  GPU 1 → Classical-SAC + QC-SAC  (3 seeds each)
  GPU 2 → SAC-AE   (3 seeds)

Paper hyperparameters (Table 2):
  QE-SAC / QC-SAC : 8 qubits, 2 VQC layers, lr=1e-4, γ=0.99, τ=0.005, batch=256, buf=1M
  SAC-AE          : 2 hidden layers, 8 units, same SAC params
  Classical SAC   : 2 hidden layers, 256 units, same SAC params
  Temperature α   : auto-tuned (Algorithm 1, Eq. 33)

Results saved to: artifacts/qe_sac_paper/13bus/results.json
"""

import sys, os, json, multiprocessing as mp

sys.path.insert(0, "/root/power-system")

import numpy as np
import torch

from src.qe_sac.env_opendss import VVCEnvOpenDSS
from src.qe_sac.qe_sac_policy import QESACAgent, QCSACAgent
from src.qe_sac.sac_baseline import ClassicalSACAgent, SACAEAgent
from src.qe_sac.trainer import QESACTrainer
from src.qe_sac.metrics import evaluate_policy

# ── Paper hyperparameters (Table 2) ───────────────────────────────────────────
SEEDS        = [0, 1, 2]
N_STEPS      = 240_000      # 10,000 episodes × 24 steps
BATCH_SIZE   = 256
WARMUP       = 1_000
CAE_INTERVAL = 500
CAE_COLLECT  = 5_000
CAE_PRETRAIN = 200
LR           = 1e-4
GAMMA        = 0.99
TAU          = 0.005
BUFFER_SIZE  = 1_000_000
N_EVAL_EPS   = 10

SAVE_DIR = "artifacts/qe_sac_paper/13bus"
os.makedirs(SAVE_DIR, exist_ok=True)

AGENT_GPU = {
    "QE-SAC":        0,
    "Classical-SAC": 1,
    "QC-SAC":        1,
    "SAC-AE":        2,
}


def make_env(seed: int) -> VVCEnvOpenDSS:
    return VVCEnvOpenDSS(seed=seed)


def run_agent(agent_name, AgentClass, gpu_id, return_dict):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    seed_rewards, seed_vviols = [], []
    n_params = 0

    for seed in SEEDS:
        env         = make_env(seed)
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
        )

        print(f"[GPU{gpu_id}|{agent_name}|s{seed}] START  params={agent.param_count():,}", flush=True)
        trainer.train(n_steps=N_STEPS)

        tag  = agent_name.lower().replace(" ", "_").replace("-", "_")
        agent.save(os.path.join(SAVE_DIR, f"{tag}_seed{seed}.pt"))

        result = evaluate_policy(make_env(seed + 100), agent, n_episodes=N_EVAL_EPS, device=device)
        seed_rewards.append(result["mean_reward"])
        seed_vviols.append(result["mean_v_viols"])
        n_params = agent.param_count()
        print(f"[GPU{gpu_id}|{agent_name}|s{seed}] DONE  reward={result['mean_reward']:.3f}  vviol={result['mean_v_viols']:.1f}", flush=True)

    return_dict[agent_name] = {
        "mean":   float(np.mean(seed_rewards)),
        "std":    float(np.std(seed_rewards)),
        "seeds":  seed_rewards,
        "vviols": seed_vviols,
        "params": n_params,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")

    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  13-BUS PAPER RUN — {n_gpus}× {torch.cuda.get_device_name(0)}")
    print(f"  4 agents × 3 seeds  |  {N_STEPS:,} steps  |  auto-α tuning")
    print(f"{'='*60}")

    agents_cfg = [
        ("QE-SAC",        QESACAgent),
        ("Classical-SAC", ClassicalSACAgent),
        ("QC-SAC",        QCSACAgent),
        ("SAC-AE",        SACAEAgent),
    ]

    manager     = mp.Manager()
    return_dict = manager.dict()
    processes   = []

    for agent_name, AgentClass in agents_cfg:
        gpu_id = AGENT_GPU[agent_name] % n_gpus
        p = mp.Process(target=run_agent, args=(agent_name, AgentClass, gpu_id, return_dict))
        p.start()
        processes.append(p)
        print(f"  Launched [{agent_name}] → GPU {gpu_id}")

    for p in processes:
        p.join()

    results = dict(return_dict)
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*68}")
    print(f"  RESULTS — IEEE 13-bus  ({len(SEEDS)} seeds × {N_STEPS:,} steps)")
    print(f"{'='*68}")
    print(f"  {'Agent':20s}  {'Mean':>8s}  {'Std':>7s}  {'Params':>10s}")
    print(f"  {'-'*58}")
    for name, r in results.items():
        print(f"  {name:20s}  {r['mean']:8.3f}  ±{r['std']:6.3f}  {r['params']:>10,}")

    print(f"\n  Paper (Lin et al. 2025):")
    print(f"  {'QE-SAC':20s}  {'−5.390':>8s}  {'':>7s}  {'4,872':>10s}")
    print(f"  {'Classical-SAC':20s}  {'−5.410':>8s}  {'':>7s}  {'899,729':>10s}")
    print(f"{'='*68}")
    print(f"\n  Saved → {SAVE_DIR}/results.json")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
