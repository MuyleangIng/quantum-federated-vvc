"""QFL configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ClientConfig:
    name:         str
    env_id:       str
    obs_dim:      int
    seed:         int   = 0
    device:       str   = "cpu"
    reward_scale: float = 1.0


@dataclass
class QFLConfig:
    # Federation
    n_rounds:    int = 50
    local_steps: int = 1_000
    warmup_steps:int = 1_000
    conditions:  List[str] = field(default_factory=lambda: ["local_only", "qfl"])

    # Training efficiency
    update_every: int  = 10    # SAC update frequency (every N env steps)

    # SAC hyperparameters
    lr:          float = 3e-4
    gamma:       float = 0.99
    tau:         float = 0.005
    alpha:       float = 0.2
    buffer_size: int   = 200_000
    batch_size:  int   = 256

    # Logging
    log_interval: int = 10
    save_dir:     str = "artifacts/qfl"

    # Seeds
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Clients
    clients: List[ClientConfig] = field(default_factory=lambda: [
        ClientConfig(name="Utility_A_13bus",  env_id="13bus_fl",  obs_dim=43,  seed=0, reward_scale=50.0),
        ClientConfig(name="Utility_B_34bus",  env_id="34bus_fl",  obs_dim=113, seed=1, reward_scale=10.0),
        ClientConfig(name="Utility_C_123bus", env_id="123bus_fl", obs_dim=349, seed=2, reward_scale=750.0),
    ])
