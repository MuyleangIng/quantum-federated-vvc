"""
Overnight run script — QE-SAC baseline + FL experiments.
Runs everything, logs everything, safe to close terminal.

Logs:
    logs/baseline_gpu.log   — baseline training progress
    logs/fl_experiment.log  — federated learning progress
    logs/status.log         — high-level status (check this first tomorrow)

Usage:
    # Start (runs in background, safe to close terminal):
    nohup python scripts/run_overnight.py > logs/main.log 2>&1 &

    # Check progress tomorrow:
    cat logs/status.log
    tail -50 logs/baseline_gpu.log
    tail -50 logs/fl_experiment.log
"""

import sys
import os
import json
import time
import traceback
import logging
from datetime import datetime

sys.path.insert(0, "/root/power-system")

# ── Log directory ─────────────────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ── Status logger (high-level — check this first) ────────────────────────────
def make_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # File handler
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console (also visible in main.log)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


status_log  = make_logger("status",   f"{LOG_DIR}/status.log")
baseline_log= make_logger("baseline", f"{LOG_DIR}/baseline_gpu.log")
fl_log      = make_logger("fl",       f"{LOG_DIR}/fl_experiment.log")


def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Baseline comparison (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_agent(args):
    """Module-level worker (picklable by multiprocessing)."""
    (agent_name, AgentClass, gpu_id, return_dict,
     use_gpu, SEEDS, N_STEPS, BATCH_SIZE, WARMUP, CAE_INTERVAL,
     CAE_COLLECT, CAE_PRETRAIN, LR, GAMMA, TAU, ALPHA,
     BUFFER_SIZE, N_EVAL_EPS, SAVE_DIR, LOG_DIR) = args

    import numpy as np
    import torch
    from src.qe_sac.env_opendss import VVCEnvOpenDSS
    from src.qe_sac.trainer import QESACTrainer
    from src.qe_sac.metrics import evaluate_policy

    device = f"cuda:{gpu_id}" if use_gpu else "cpu"
    log = make_logger(f"agent_{agent_name}", f"{LOG_DIR}/baseline_gpu.log")
    log.info(f"[{agent_name}] START  device={device}")

    seed_rewards, seed_vviols = [], []
    n_params = 0

    for seed in SEEDS:
        t0 = time.time()
        env = VVCEnvOpenDSS(seed=seed)
        obs_dim     = env.observation_space.shape[0]
        device_dims = list(map(int, env.action_space.nvec))
        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = AgentClass(
            obs_dim=obs_dim, device_dims=device_dims,
            lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA,
            buffer_size=BUFFER_SIZE, device=device,
        )

        if hasattr(agent, "pretrain_cae"):
            agent.pretrain_cae(env, n_collect=CAE_COLLECT, n_train_steps=CAE_PRETRAIN)
            env = VVCEnvOpenDSS(seed=seed)
        if hasattr(agent, "pretrain_pca"):
            agent.pretrain_pca(env, n_collect=CAE_COLLECT)
            env = VVCEnvOpenDSS(seed=seed)

        trainer = QESACTrainer(
            agent, env,
            batch_size=BATCH_SIZE, cae_update_interval=CAE_INTERVAL,
            warmup_steps=WARMUP, log_interval=5_000,
            save_dir=SAVE_DIR, device=device,
        )

        log.info(f"[{agent_name}] seed={seed}  params={agent.param_count():,}  training...")
        trainer.train(n_steps=N_STEPS)

        ckpt = os.path.join(SAVE_DIR, f"{agent_name.lower().replace('-','_')}_seed{seed}.pt")
        agent.save(ckpt)

        result = evaluate_policy(VVCEnvOpenDSS(seed=seed+100), agent,
                                 n_episodes=N_EVAL_EPS, device=device)
        seed_rewards.append(result["mean_reward"])
        seed_vviols.append(result["mean_v_viols"])
        n_params = agent.param_count()

        elapsed = time.time() - t0
        log.info(f"[{agent_name}] seed={seed}  reward={result['mean_reward']:.3f}  "
                 f"vviol={result['mean_v_viols']:.1f}  time={elapsed/60:.1f}min")

    return_dict[agent_name] = {
        "mean":   float(np.mean(seed_rewards)),
        "std":    float(np.std(seed_rewards)),
        "seeds":  seed_rewards,
        "vviols": seed_vviols,
        "params": n_params,
        "device": device,
        "env":    "OpenDSS 3-phase AC (IEEE 13-bus)",
    }
    log.info(f"[{agent_name}] DONE  mean={np.mean(seed_rewards):.3f} ± {np.std(seed_rewards):.3f}")


def run_baseline():
    status_log.info("=" * 60)
    status_log.info("PHASE 1 START — Baseline GPU experiment")
    status_log.info("  Agents: QE-SAC, Classical-SAC, SAC-AE, QC-SAC")
    status_log.info("  Steps:  240,000 per seed × 3 seeds")
    status_log.info("  Output: artifacts/qe_sac_paper/opendss_gpu/results.json")
    status_log.info("=" * 60)

    import numpy as np
    import torch
    import multiprocessing as mp

    from src.qe_sac.env_opendss import VVCEnvOpenDSS
    from src.qe_sac.qe_sac_policy import QESACAgent, QCSACAgent
    from src.qe_sac.sac_baseline import ClassicalSACAgent, SACAEAgent
    from src.qe_sac.trainer import QESACTrainer
    from src.qe_sac.metrics import evaluate_policy

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
    SAVE_DIR     = "artifacts/qe_sac_paper/opendss_gpu"
    os.makedirs(SAVE_DIR, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    n_gpus  = torch.cuda.device_count() if use_gpu else 0
    baseline_log.info(f"GPU available: {use_gpu}  |  GPUs: {n_gpus}")
    status_log.info(f"  GPU: {use_gpu}  n_gpus: {n_gpus}")

    AGENT_GPU = {"QE-SAC": 0, "Classical-SAC": 1, "SAC-AE": 2, "QC-SAC": 0}

    worker_consts = (
        use_gpu, SEEDS, N_STEPS, BATCH_SIZE, WARMUP, CAE_INTERVAL,
        CAE_COLLECT, CAE_PRETRAIN, LR, GAMMA, TAU, ALPHA,
        BUFFER_SIZE, N_EVAL_EPS, SAVE_DIR, LOG_DIR,
    )

    agents_cfg = [
        ("QE-SAC",        QESACAgent),
        ("Classical-SAC", ClassicalSACAgent),
        ("SAC-AE",        SACAEAgent),
        ("QC-SAC",        QCSACAgent),
    ]

    manager     = mp.Manager()
    return_dict = manager.dict()

    if use_gpu and n_gpus >= 2:
        # Parallel on multiple GPUs
        baseline_log.info("Running agents in PARALLEL across GPUs")
        status_log.info("  Mode: PARALLEL (multi-GPU)")
        processes = []
        for agent_name, AgentClass in agents_cfg:
            gpu_id = AGENT_GPU[agent_name] % n_gpus
            p = mp.Process(target=_run_one_agent,
                           args=((agent_name, AgentClass, gpu_id, return_dict) + worker_consts,))
            p.start()
            processes.append((agent_name, p))
            baseline_log.info(f"  Launched {agent_name} → GPU {gpu_id}")
        for name, p in processes:
            p.join()
            baseline_log.info(f"  {name} process finished")
    else:
        # Sequential (CPU or single GPU)
        baseline_log.info("Running agents SEQUENTIALLY")
        status_log.info("  Mode: SEQUENTIAL (CPU or single GPU)")
        for agent_name, AgentClass in agents_cfg:
            _run_one_agent((agent_name, AgentClass, 0, return_dict) + worker_consts)

    # Save results
    results = dict(return_dict)
    out = os.path.join(SAVE_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    status_log.info("PHASE 1 DONE — Baseline results saved")
    status_log.info(f"  File: {out}")
    status_log.info("  Results:")
    for name, r in results.items():
        status_log.info(f"    {name:20s}  mean={r['mean']:+.3f}  std=±{r['std']:.3f}  params={r['params']:,}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — FL experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_fl():
    status_log.info("=" * 60)
    status_log.info("PHASE 2 START — Federated Learning experiment")
    status_log.info("  Conditions: local_only / QE-SAC-FL / QE-SAC-FL-Aligned")
    status_log.info("  Clients: 13-bus, 34-bus, 123-bus")
    status_log.info("  Output: artifacts/qe_sac_fl/")
    status_log.info("=" * 60)

    import torch
    from src.qe_sac_fl.federated_trainer import FederatedTrainer
    from src.qe_sac_fl.fed_config import paper_config

    os.makedirs("artifacts/qe_sac_fl", exist_ok=True)

    cfg = paper_config()

    # Redirect FederatedTrainer prints to fl log
    import builtins
    _orig_print = builtins.print
    def _log_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        fl_log.info(msg)
        _orig_print(*args, **kwargs)
    builtins.print = _log_print

    try:
        trainer = FederatedTrainer(cfg)
        all_results = {}

        # --- local_only ---
        fl_log.info("\n[1/3] Running: local_only baseline...")
        status_log.info("  FL [1/3] local_only starting...")
        t0 = time.time()
        r_local = trainer.run_local_only()
        r_local.save("artifacts/qe_sac_fl/local_only.json")
        elapsed = (time.time() - t0) / 60
        all_results["local_only"] = r_local
        status_log.info(f"  FL [1/3] local_only DONE  ({elapsed:.1f} min)")
        for name, reward in r_local.final_rewards().items():
            status_log.info(f"    {name}: {reward:+.3f}")

        # --- QE-SAC-FL (VQC only) ---
        fl_log.info("\n[2/3] Running: QE-SAC-FL (VQC only, unaligned)...")
        status_log.info("  FL [2/3] QE-SAC-FL (VQC only) starting...")
        t0 = time.time()
        r_fl = trainer.run("QE-SAC-FL")
        r_fl.save("artifacts/qe_sac_fl/qe_sac_fl_vqc_only.json")
        elapsed = (time.time() - t0) / 60
        all_results["QE-SAC-FL"] = r_fl
        status_log.info(f"  FL [2/3] QE-SAC-FL DONE  ({elapsed:.1f} min)")
        for name, reward in r_fl.final_rewards().items():
            status_log.info(f"    {name}: {reward:+.3f}")

        # --- QE-SAC-FL-Aligned (SharedHead + VQC) ---
        fl_log.info("\n[3/3] Running: QE-SAC-FL-Aligned (SharedHead + VQC)...")
        status_log.info("  FL [3/3] QE-SAC-FL-Aligned starting...")
        t0 = time.time()
        r_aligned = trainer.run_aligned()
        r_aligned.save("artifacts/qe_sac_fl/qe_sac_fl_aligned.json")
        elapsed = (time.time() - t0) / 60
        all_results["QE-SAC-FL-Aligned"] = r_aligned
        status_log.info(f"  FL [3/3] QE-SAC-FL-Aligned DONE  ({elapsed:.1f} min)")
        for name, reward in r_aligned.final_rewards().items():
            status_log.info(f"    {name}: {reward:+.3f}")

        # --- Summary ---
        status_log.info("\nFL EXPERIMENT COMPLETE — H1/H2/H3/H4 RESULTS:")
        status_log.info(f"{'Condition':<30} {'13-bus':>10} {'34-bus':>10} {'123-bus':>10} {'Bytes TX':>12}")
        status_log.info("-" * 76)
        for cond_name, res in all_results.items():
            rewards = res.final_rewards()
            vals = [f"{v:+.3f}" for v in rewards.values()]
            while len(vals) < 3:
                vals.append("  N/A")
            status_log.info(
                f"  {cond_name:<28} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} "
                f"{res.bytes_communicated:>12,}"
            )

    finally:
        builtins.print = _orig_print

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    status_log.info("=" * 60)
    status_log.info("OVERNIGHT RUN STARTED")
    status_log.info(f"  Time: {stamp()}")
    status_log.info("  Jobs: [1] Baseline GPU  [2] FL experiment")
    status_log.info("  Check logs/status.log for progress anytime")
    status_log.info("=" * 60)

    try:
        # Phase 1 — Baseline
        baseline_results = run_baseline()
    except Exception:
        status_log.error("PHASE 1 FAILED:")
        status_log.error(traceback.format_exc())
        status_log.info("Continuing to Phase 2...")

    try:
        # Phase 2 — FL
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        fl_results = run_fl()
    except Exception:
        status_log.error("PHASE 2 FAILED:")
        status_log.error(traceback.format_exc())

    total = (time.time() - t_start) / 3600
    status_log.info("=" * 60)
    status_log.info(f"ALL DONE  total time: {total:.2f} hours")
    status_log.info(f"  Finished: {stamp()}")
    status_log.info("Files:")
    status_log.info("  artifacts/qe_sac_paper/opendss_gpu/results.json  ← baseline")
    status_log.info("  artifacts/qe_sac_fl/local_only.json              ← FL local")
    status_log.info("  artifacts/qe_sac_fl/qe_sac_fl_vqc_only.json     ← FL VQC only")
    status_log.info("  artifacts/qe_sac_fl/qe_sac_fl_aligned.json      ← FL aligned")
    status_log.info("=" * 60)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
