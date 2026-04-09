"""
FL experiment runner — all 3 conditions on 3× RTX 4090.

Conditions:
    [1] local_only      — each client trains alone, no sharing
    [2] QE-SAC-FL       — VQC weights shared (FedAvg)
    [3] QE-SAC-FL-Aligned — SharedEncoderHead + VQC shared (FedAvg)

Clients:
    Utility_A_13bus  → GPU 0
    Utility_B_34bus  → GPU 1
    Utility_C_123bus → GPU 2

Output: artifacts/qe_sac_fl/

Usage:
    nohup python scripts/run_fl.py > logs/fl.log 2>&1 &
    tail -f logs/fl.log
"""

import sys
import os
import time

sys.path.insert(0, "/root/power-system")
os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts/qe_sac_fl", exist_ok=True)


def stamp():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    from src.qe_sac_fl.federated_trainer import FederatedTrainer
    from src.qe_sac_fl.fed_config import paper_config

    cfg     = paper_config()
    trainer = FederatedTrainer(cfg)

    print(f"\n{'='*60}")
    print(f"  FL EXPERIMENT START  {stamp()}")
    print(f"  Rounds: {cfg.n_rounds}  |  Steps/round: {cfg.local_steps}")
    print(f"  Total per client: {cfg.n_rounds * cfg.local_steps:,} steps")
    print(f"{'='*60}\n")

    # ── [1/3] local_only ──────────────────────────────────────────────────────
    print(f"[1/3] local_only baseline  ({stamp()})")
    t0 = time.time()
    r_local = trainer.run_local_only()
    r_local.save("artifacts/qe_sac_fl/local_only.json")
    print(f"  DONE  ({(time.time()-t0)/60:.1f} min)")
    for name, r in r_local.final_rewards().items():
        print(f"    {name}: {r:+.3f}")

    # ── [2/3] QE-SAC-FL (VQC only) ───────────────────────────────────────────
    print(f"\n[2/3] QE-SAC-FL — VQC only  ({stamp()})")
    t0 = time.time()
    r_fl = trainer.run("QE-SAC-FL")
    r_fl.save("artifacts/qe_sac_fl/qe_sac_fl_vqc_only.json")
    print(f"  DONE  ({(time.time()-t0)/60:.1f} min)")
    for name, r in r_fl.final_rewards().items():
        print(f"    {name}: {r:+.3f}")

    # ── [3/3] QE-SAC-FL-Aligned (SharedHead + VQC) ───────────────────────────
    print(f"\n[3/3] QE-SAC-FL-Aligned — SharedHead+VQC  ({stamp()})")
    t0 = time.time()
    r_aligned = trainer.run_aligned()
    r_aligned.save("artifacts/qe_sac_fl/qe_sac_fl_aligned.json")
    print(f"  DONE  ({(time.time()-t0)/60:.1f} min)")
    for name, r in r_aligned.final_rewards().items():
        print(f"    {name}: {r:+.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FL RESULTS — H1/H2/H3")
    print(f"{'='*70}")
    print(f"  {'Condition':<28} {'13-bus':>10} {'34-bus':>10} {'123-bus':>10} {'Bytes TX':>12}")
    print(f"  {'-'*68}")
    for cond_name, res in [("local_only", r_local), ("QE-SAC-FL", r_fl), ("QE-SAC-FL-Aligned", r_aligned)]:
        rewards = list(res.final_rewards().values())
        vals = [f"{v:+.3f}" for v in rewards]
        while len(vals) < 3:
            vals.append("  N/A")
        print(f"  {cond_name:<28} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} "
              f"{res.bytes_communicated:>12,}")
    print(f"{'='*70}")
    print(f"\n  Finished: {stamp()}")


if __name__ == "__main__":
    main()
