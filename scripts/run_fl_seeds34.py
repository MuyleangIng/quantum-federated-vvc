"""
Run FL experiment for seeds 3 and 4 only (seeds 0-2 already done).
Saves to same artifact directory so verify_results.py picks them up.

Run: python -u scripts/run_fl_seeds34.py > logs/fl_seeds34.log 2>&1 &
"""

import sys, os
sys.path.insert(0, "/root/power-system")
os.makedirs("artifacts/qe_sac_fl", exist_ok=True)

import json, torch, numpy as np
from src.qe_sac_fl.fed_config import FedConfig, ClientConfig
from src.qe_sac_fl.federated_trainer import FederatedTrainer


def make_cfg(seed: int) -> FedConfig:
    n_gpus  = torch.cuda.device_count()
    devices = [f"cuda:{i}" if i < n_gpus else "cpu" for i in range(3)]

    cfg = FedConfig()
    cfg.n_rounds         = 500
    cfg.local_steps      = 1_000
    cfg.warmup_steps     = 1_000
    cfg.batch_size       = 256
    cfg.lr               = 3e-4
    cfg.buffer_size      = 200_000
    cfg.log_interval     = 100
    cfg.parallel_clients = (n_gpus >= 3)
    cfg.hidden_dim       = 32
    cfg.server_momentum  = 0.3
    cfg.aggregation      = "uniform"
    cfg.clients = [
        ClientConfig(name="Utility_A_13bus",  env_id="13bus_fl",  obs_dim=43,  n_actions=132,
                     seed=seed,   device=devices[0], reward_scale=50.0),
        ClientConfig(name="Utility_B_34bus",  env_id="34bus_fl",  obs_dim=113, n_actions=132,
                     seed=seed+3, device=devices[1], reward_scale=10.0),
        ClientConfig(name="Utility_C_123bus", env_id="123bus_fl", obs_dim=349, n_actions=132,
                     seed=seed+6, device=devices[2], reward_scale=750.0),
    ]
    return cfg


def main():
    print("="*60, flush=True)
    print("  FL Seeds 3 & 4 — 3 conditions each", flush=True)
    print("  500 rounds × 1000 steps = 500K steps/client", flush=True)
    print("  Rewards in normalised units (A/50, B/10, C/750)", flush=True)
    print("="*60, flush=True)

    results_all = {}

    for seed in [3, 4]:
        print(f"\n{'='*60}", flush=True)
        print(f"  SEED {seed}", flush=True)
        print(f"{'='*60}", flush=True)

        cfg     = make_cfg(seed)
        trainer = FederatedTrainer(cfg)

        print(f"\n  [1/3] local_only ...", flush=True)
        r_local = trainer.run_local_only()
        r_local.save(f"artifacts/qe_sac_fl/seed{seed}_local_only.json")
        for name, r in r_local.final_rewards().items():
            print(f"    {name}: {r:+.3f}", flush=True)

        print(f"\n  [2/3] QE-SAC-FL naive ...", flush=True)
        r_naive = trainer.run("QE-SAC-FL")
        r_naive.save(f"artifacts/qe_sac_fl/seed{seed}_naive_fl.json")
        for name, r in r_naive.final_rewards().items():
            print(f"    {name}: {r:+.3f}", flush=True)

        print(f"\n  [3/3] QE-SAC-FL-Aligned ...", flush=True)
        r_aligned = trainer.run_aligned()
        r_aligned.save(f"artifacts/qe_sac_fl/seed{seed}_aligned_fl.json")
        for name, r in r_aligned.final_rewards().items():
            print(f"    {name}: {r:+.3f}", flush=True)

        results_all[seed] = {
            "local_only": r_local.final_rewards(),
            "naive_fl":   r_naive.final_rewards(),
            "aligned_fl": r_aligned.final_rewards(),
        }
        print(f"\n  Seed {seed} done.", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("  Seeds 3-4 complete. Run verify_results.py for full stats.", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
