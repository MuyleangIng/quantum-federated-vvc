"""
H5 — Personalised FL: aligned FL warm-start + local fine-tuning.

Phase 1: Run aligned FL for n_fl_rounds (shared VQC + SharedHead federated)
Phase 2: Each client fine-tunes locally for n_finetune_steps (no federation)

Hypothesis: FL warm-start + fine-tune > local-only > pure aligned FL
because the global shared representation provides a better initialisation
than random init, and local fine-tuning recovers feeder-specific behaviour.

Run: python -u scripts/run_fl_personalized.py > logs/fl_personalized.log 2>&1 &
"""

import sys, os
sys.path.insert(0, "/root/power-system")
os.makedirs("artifacts/qe_sac_fl", exist_ok=True)

import json, torch, numpy as np
from src.qe_sac_fl.fed_config import FedConfig, ClientConfig
from src.qe_sac_fl.federated_trainer import FederatedTrainer

CLIENTS = ["Utility_A_13bus", "Utility_B_34bus", "Utility_C_123bus"]


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
    print("="*65, flush=True)
    print("  H5 — Personalised FL  (3 seeds)", flush=True)
    print("  Phase 1: 500 rounds aligned FL  (500K steps/client)", flush=True)
    print("  Phase 2: 50K fine-tune steps per client (local, no FL)", flush=True)
    print("  Rewards in normalised units (A/50, B/10, C/750)", flush=True)
    print("="*65, flush=True)

    all_results = {}

    for seed in [0, 1, 2]:
        print(f"\n{'='*65}", flush=True)
        print(f"  SEED {seed}", flush=True)
        print(f"{'='*65}", flush=True)

        cfg     = make_cfg(seed)
        trainer = FederatedTrainer(cfg)

        results = trainer.run_personalized(
            n_fl_rounds      = 500,
            n_finetune_steps = 50_000,
        )
        results.save(f"artifacts/qe_sac_fl/seed{seed}_personalized.json")

        final = results.final_rewards()
        for name, r in final.items():
            print(f"    {name}: {r:+.3f}", flush=True)
        all_results[seed] = final
        print(f"\n  Seed {seed} done.", flush=True)

    # Summary table
    print(f"\n{'='*65}", flush=True)
    print(f"  PERSONALISED FL — mean ± std (3 seeds, normalised)", flush=True)
    print(f"{'='*65}", flush=True)

    # Compare vs local_only from existing results
    local_means = {}
    for seed in [0, 1, 2]:
        path = f"artifacts/qe_sac_fl/seed{seed}_local_only.json"
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            last = {}
            for log in raw["logs"]:
                last[log["client"]] = log["reward"]
            for cl, v in last.items():
                local_means.setdefault(cl, []).append(v)

    print(f"\n  {'Client':<25} {'personalized':>14} {'local_only':>12} {'Δ':>8}", flush=True)
    print("  " + "-"*62, flush=True)
    for cl in CLIENTS:
        vals = [all_results[s].get(cl, np.nan) for s in [0, 1, 2]]
        mean_p = np.nanmean(vals)
        std_p  = np.nanstd(vals)
        locs   = local_means.get(cl, [np.nan])
        mean_l = np.nanmean(locs)
        delta  = mean_p - mean_l
        sign   = "✓" if delta > 0 else "✗"
        print(f"  {cl:<25} {mean_p:>+10.3f} ±{std_p:.3f}  "
              f"{mean_l:>+10.3f}  {delta:>+7.3f} {sign}", flush=True)

    out = "artifacts/qe_sac_fl/personalized_summary.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out}", flush=True)
    print("="*65, flush=True)


if __name__ == "__main__":
    main()
