"""
Federated Learning configuration for QE-SAC-FL experiments.

All hyperparameters for the FL loop, clients, and hypotheses in one place.
Change values here only — do not hardcode numbers in federated_trainer.py.
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class ClientConfig:
    """Configuration for one federated client (one utility / feeder)."""
    name: str           # e.g. "Utility_A_13bus"
    env_id: str         # e.g. "13bus_fl", "34bus_fl", "123bus_fl"
    obs_dim: int        # observation space dimension
    n_actions: int      # total joint discrete actions
    seed: int = 0
    device: str = "cpu" # which GPU this client trains on


@dataclass
class FedConfig:
    """
    Master configuration for QE-SAC-FL federated training.

    Hypotheses:
        H1: Federated VQC reward > Local-only VQC reward on all clients
        H2: QE-SAC-FL converges faster (fewer steps) than local training
        H3: Total bytes communicated by QE-SAC-FL << Federated Classical SAC
    """

    # --- Federation ---
    n_rounds: int = 50
    local_steps: int = 1_000
    # Total per-client steps = n_rounds × local_steps

    # --- Local training ---
    batch_size: int = 256
    warmup_steps: int = 1_000

    # --- Agent ---
    lr: float = 3e-4        # slightly higher lr for faster convergence
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    buffer_size: int = 200_000

    # --- Federation ---
    federate_vqc_only: bool = True
    aggregation: str = "uniform"

    # --- Clients ---
    clients: List[ClientConfig] = field(default_factory=lambda: [
        ClientConfig(
            name     = "Utility_A_13bus",
            env_id   = "13bus_fl",
            obs_dim  = 42,
            n_actions= 132,
            seed     = 0,
            device   = "cpu",   # overridden by paper_config()
        ),
        ClientConfig(
            name     = "Utility_B_34bus",
            env_id   = "34bus_fl",
            obs_dim  = 105,
            n_actions= 132,
            seed     = 1,
            device   = "cpu",
        ),
        ClientConfig(
            name     = "Utility_C_123bus",
            env_id   = "123bus_fl",
            obs_dim  = 372,
            n_actions= 132,
            seed     = 2,
            device   = "cpu",
        ),
    ])

    # --- Conditions ---
    run_local_only:        bool = True
    run_centralised:       bool = False
    run_qe_sac_fl:         bool = True
    run_fed_classical_sac: bool = False

    # --- H2 threshold ---
    reward_convergence_threshold: float = -50.0

    # --- Logging ---
    log_interval: int = 5
    save_dir: str = "artifacts/qe_sac_fl"

    # --- Seeds ---
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # --- Parallelism ---
    parallel_clients: bool = False  # set True in paper_config() with multi-GPU


# ---------------------------------------------------------------------------
# Quick-test config (~2 min, CPU)
# ---------------------------------------------------------------------------

def long_run_config(n_rounds: int = 200) -> FedConfig:
    """
    Extended run to prove H1 for all 3 clients.

    50 rounds was sufficient for 13-bus but not 34-bus / 123-bus.
    200 rounds × 1,000 steps = 200K steps per client.
    Wall time: ~6 min on 3× RTX 4090 (aligned only, no local/unaligned redo).

    Use this config with trainer.run_aligned() only — local_only and
    unaligned results are already saved from the 50-round run.
    """
    n_gpus = torch.cuda.device_count()
    cfg = FedConfig()

    cfg.n_rounds         = n_rounds
    cfg.local_steps      = 1_000
    cfg.warmup_steps     = 1_000
    cfg.batch_size       = 512
    cfg.lr               = 3e-4
    cfg.buffer_size      = 200_000
    cfg.seeds            = [0, 1, 2, 3, 4]
    cfg.log_interval     = 10
    cfg.parallel_clients = (n_gpus >= 3)
    cfg.run_local_only   = False   # already have 50-round baseline
    cfg.run_qe_sac_fl    = False   # already have unaligned results

    devices = [f"cuda:{i}" if i < n_gpus else "cpu" for i in range(3)]
    cfg.clients = [
        ClientConfig(name="Utility_A_13bus",  env_id="13bus_fl",  obs_dim=42,  n_actions=132, seed=0, device=devices[0]),
        ClientConfig(name="Utility_B_34bus",  env_id="34bus_fl",  obs_dim=105, n_actions=132, seed=1, device=devices[1]),
        ClientConfig(name="Utility_C_123bus", env_id="123bus_fl", obs_dim=372, n_actions=132, seed=2, device=devices[2]),
    ]

    print(f"long_run_config: {n_rounds} rounds, {n_gpus} GPU(s)")
    for c in cfg.clients:
        print(f"  {c.name:<25} → {c.device}")
    return cfg


def quick_config() -> FedConfig:
    """Verify FL loop runs end-to-end. Not for results."""
    cfg = FedConfig()
    cfg.n_rounds     = 5
    cfg.local_steps  = 200
    cfg.warmup_steps = 100
    cfg.seeds        = [0]
    cfg.log_interval = 1
    return cfg


# ---------------------------------------------------------------------------
# Paper config — 3 GPUs, 5 seeds, full training
# ---------------------------------------------------------------------------

def paper_config() -> FedConfig:
    """
    Full experiment config for paper results.
    Assigns each client to its own GPU for parallel training.
    Total wall time: ~3-4 hours on 3× RTX 4090.
    """
    n_gpus = torch.cuda.device_count()
    cfg = FedConfig()

    cfg.n_rounds         = 50
    cfg.local_steps      = 1_000   # 50K steps per client total
    cfg.warmup_steps     = 1_000   # fill buffer before any VQC updates
    cfg.batch_size       = 512     # larger batch → more stable gradients on GPU
    cfg.lr               = 3e-4
    cfg.buffer_size      = 200_000
    cfg.seeds            = [0, 1, 2, 3, 4]
    cfg.log_interval     = 5
    cfg.parallel_clients = (n_gpus >= 3)

    # Assign each client to its own GPU if available
    devices = [f"cuda:{i}" if i < n_gpus else "cpu" for i in range(3)]
    cfg.clients = [
        ClientConfig(
            name     = "Utility_A_13bus",
            env_id   = "13bus_fl",
            obs_dim  = 42,
            n_actions= 132,
            seed     = 0,
            device   = devices[0],
        ),
        ClientConfig(
            name     = "Utility_B_34bus",
            env_id   = "34bus_fl",
            obs_dim  = 105,
            n_actions= 132,
            seed     = 1,
            device   = devices[1],
        ),
        ClientConfig(
            name     = "Utility_C_123bus",
            env_id   = "123bus_fl",
            obs_dim  = 372,
            n_actions= 132,
            seed     = 2,
            device   = devices[2],
        ),
    ]

    print(f"paper_config: {n_gpus} GPU(s) detected")
    for c in cfg.clients:
        print(f"  {c.name:<25} → {c.device}")

    return cfg
