"""
QFL — Quantum Federated Learning.

Runs two conditions per seed:
  local_only  — QE-SAC on each utility independently (baseline)
  qfl         — QFL with FedAvg on QuantumEncoder (proposed)

Usage:
  python -u scripts/run_qfl.py > logs/qfl.log 2>&1 &
"""

import sys, os
sys.path.insert(0, "/root/power-system")
os.makedirs("artifacts/qfl", exist_ok=True)
os.makedirs("logs", exist_ok=True)

import torch
from src.qfl.config import QFLConfig, ClientConfig
from src.qfl.trainer import QFLTrainer


def make_cfg(seed: int) -> QFLConfig:
    n_gpus  = torch.cuda.device_count()
    devices = [f"cuda:{i}" if i < n_gpus else "cpu" for i in range(3)]
    cfg = QFLConfig()
    cfg.n_rounds     = 50
    cfg.local_steps  = 1_000
    cfg.warmup_steps = 1_000
    cfg.batch_size   = 256
    cfg.lr           = 3e-4
    cfg.log_interval = 10
    cfg.save_dir     = "artifacts/qfl"
    cfg.clients = [
        ClientConfig(name="Utility_A_13bus",  env_id="13bus_fl",  obs_dim=43,  seed=seed,   device=devices[0], reward_scale=50.0),
        ClientConfig(name="Utility_B_34bus",  env_id="34bus_fl",  obs_dim=113, seed=seed+3, device=devices[1], reward_scale=10.0),
        ClientConfig(name="Utility_C_123bus", env_id="123bus_fl", obs_dim=349, seed=seed+6, device=devices[2], reward_scale=750.0),
    ]
    return cfg


def main():
    print("=" * 60, flush=True)
    print("  QFL — Quantum Federated Learning", flush=True)
    print("  Conditions: local_only vs qfl", flush=True)
    print("=" * 60, flush=True)

    for seed in [0, 1, 2]:
        cfg     = make_cfg(seed)
        trainer = QFLTrainer(cfg)
        trainer.run(seed=seed)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
