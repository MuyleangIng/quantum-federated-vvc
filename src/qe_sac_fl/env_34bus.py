"""
IEEE 34-bus VVC environment — Client B for QE-SAC-FL.

Approximate IEEE 34-bus Test Feeder (EPRI / IEEE PES standard).
Medium-sized rural feeder: longer branches, higher impedance than 13-bus.

Observation dim: 34 buses × 3 (V, P, Q) + 4 caps + 2 regs = 108
Action space:    MultiDiscrete([2, 2, 2, 2, 33, 33])
                 → 4 cap banks (ON/OFF) × 2 regulators (33 taps each)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from src.qe_sac.env_utils import _VVCEnvBase, _gen_123bus_branches, _gen_123bus_loads


# ---------------------------------------------------------------------------
# IEEE 34-bus linearized network parameters
# Branch data: (from_bus, to_bus, r_pu, x_pu) on 100 MVA, 24.9 kV base
# Approximate values from IEEE 34 Node Test Feeder documentation.
# The 34-bus feeder is longer and higher-impedance than the 13-bus feeder
# (rural distribution — longer line segments, more voltage drop).
# ---------------------------------------------------------------------------

_IEEE34_BRANCHES = [
    # Main trunk: bus 800 → 808 → ... (0-indexed here)
    (0,  1,  0.1808, 0.1580),   # 800 → 802
    (1,  2,  0.4930, 0.4310),   # 802 → 806
    (2,  3,  0.2810, 0.1910),   # 806 → 808
    (3,  4,  1.0400, 0.7400),   # 808 → 810
    (3,  5,  0.1870, 0.1270),   # 808 → 812
    (5,  6,  0.5590, 0.3800),   # 812 → 814
    (6,  7,  0.0000, 0.0000),   # 814 → 850 (voltage regulator)
    (7,  8,  0.3080, 0.2230),   # 850 → 816
    (8,  9,  0.1680, 0.1150),   # 816 → 818
    (8,  10, 0.3070, 0.2200),   # 816 → 824
    (10, 11, 0.2810, 0.1910),   # 824 → 826
    (10, 12, 0.0920, 0.0650),   # 824 → 828
    (12, 13, 0.1880, 0.1280),   # 828 → 830
    (13, 14, 0.0000, 0.0000),   # 830 → 854 (voltage regulator)
    (14, 15, 0.5120, 0.3480),   # 854 → 856
    (14, 16, 0.1570, 0.1070),   # 854 → 852
    (16, 17, 0.5120, 0.3480),   # 852 → 832
    (17, 18, 0.3120, 0.2120),   # 832 → 834
    (18, 19, 0.1750, 0.1190),   # 834 → 836
    (19, 20, 0.2490, 0.1690),   # 836 → 840
    (19, 21, 0.3500, 0.2380),   # 836 → 862
    (17, 22, 0.1640, 0.1110),   # 832 → 844
    (22, 23, 0.4760, 0.3230),   # 844 → 846
    (23, 24, 0.2890, 0.1960),   # 846 → 848
    (7,  25, 0.4500, 0.3080),   # 850 → 820
    (25, 26, 0.3660, 0.2490),   # 820 → 822
    (2,  27, 0.1120, 0.0760),   # 806 → 860
    (27, 28, 0.5590, 0.3800),   # 860 → 838
    (5,  29, 0.3080, 0.2230),   # 812 → 842
    (14, 30, 0.2310, 0.1570),   # 854 → 864
    (17, 31, 0.2040, 0.1390),   # 832 → 858
    (31, 32, 0.3410, 0.2320),   # 858 → 834b
    (0,  33, 0.0500, 0.0350),   # 800 → substation branch B
]

# Base loads (P_kW, Q_kVAR) per bus — IEEE 34-bus total ≈ 1800 kW
# This is a lightly loaded rural feeder compared to 13-bus
_IEEE34_BASE_LOADS = np.array([
    [0,    0   ],  # bus 0  — substation
    [0,    0   ],  # bus 1
    [0,    0   ],  # bus 2
    [0,    0   ],  # bus 3
    [16,   8   ],  # bus 4
    [0,    0   ],  # bus 5
    [0,    0   ],  # bus 6
    [0,    0   ],  # bus 7  — regulator
    [40,   20  ],  # bus 8
    [40,   20  ],  # bus 9
    [0,    0   ],  # bus 10
    [0,    0   ],  # bus 11
    [4,    2   ],  # bus 12
    [0,    0   ],  # bus 13
    [0,    0   ],  # bus 14 — regulator
    [40,   20  ],  # bus 15
    [0,    0   ],  # bus 16
    [0,    0   ],  # bus 17
    [4,    2   ],  # bus 18
    [0,    0   ],  # bus 19
    [27,   14  ],  # bus 20
    [28,   14  ],  # bus 21
    [0,    0   ],  # bus 22
    [23,   11  ],  # bus 23
    [25,   13  ],  # bus 24
    [0,    0   ],  # bus 25
    [85,   40  ],  # bus 26
    [0,    0   ],  # bus 27
    [126,  62  ],  # bus 28
    [8,    4   ],  # bus 29
    [2,    1   ],  # bus 30
    [0,    0   ],  # bus 31
    [0,    0   ],  # bus 32
    [0,    0   ],  # bus 33
], dtype=np.float32)

# Capacitor banks: buses 11, 24, 29, 30 (typical 34-bus cap placement)
_IEEE34_CAP_BUSES = [11, 24, 29, 30]
_IEEE34_CAP_SIZES = [300.0, 150.0, 150.0, 100.0]   # kVAR

# Two voltage regulators (branches 6 and 13 above are reg branches)
_IEEE34_N_REGS = 2
_IEEE34_N_BATS = 0


class VVCEnv34Bus(_VVCEnvBase):
    """
    VVC environment for IEEE 34-bus distribution feeder.

    This is Client B in the QE-SAC-FL federated setup.
    Represents a medium rural feeder (higher impedance than 13-bus,
    smaller than 123-bus) — genuinely different from other clients.

    Devices: 4 capacitor banks, 2 voltage regulators.
    Observation dim: 34*3 + 4 + 2 = 108
    Action space: MultiDiscrete([2, 2, 2, 2, 33, 33])
                  → joint size = 2^4 × 33^2 = 17,424
    """
    _branches   = _IEEE34_BRANCHES
    _base_loads = _IEEE34_BASE_LOADS
    _cap_buses  = _IEEE34_CAP_BUSES
    _cap_sizes  = _IEEE34_CAP_SIZES
    _n_regs     = _IEEE34_N_REGS
    _n_bats     = _IEEE34_N_BATS
    _n_buses    = 34


# ---------------------------------------------------------------------------
# Simplified variants for FL experiments (2 caps + 1 reg = 132 actions)
#
# The full VVCEnv34Bus has 17,424 joint actions — too large for a one-hot
# replay buffer. For the federated experiment the differentiator between
# clients is obs_dim / topology, not device count. These simplified variants
# keep n_actions=132 (same as 13-bus) so all clients share the same VQC
# output head size while still having different observation dimensions.
# ---------------------------------------------------------------------------

_IEEE34_CAP_BUSES_SIMPLE = [11, 24]          # 2 caps only
_IEEE34_CAP_SIZES_SIMPLE = [300.0, 150.0]
_IEEE34_N_REGS_SIMPLE    = 1                 # 1 regulator


class VVCEnv34BusFL(_VVCEnvBase):
    """
    Simplified 34-bus env for federated experiments.
    Devices: 2 caps + 1 reg  → same action space as 13-bus (132 actions).
    Observation dim: 34*3 + 2 + 1 = 105
    Use this as Client B in QE-SAC-FL.
    """
    _branches   = _IEEE34_BRANCHES
    _base_loads = _IEEE34_BASE_LOADS
    _cap_buses  = _IEEE34_CAP_BUSES_SIMPLE
    _cap_sizes  = _IEEE34_CAP_SIZES_SIMPLE
    _n_regs     = _IEEE34_N_REGS_SIMPLE
    _n_bats     = 0
    _n_buses    = 34


# ---------------------------------------------------------------------------
# Simplified 123-bus for FL experiments (2 caps + 1 reg = 132 actions)
# ---------------------------------------------------------------------------

class VVCEnv123BusFL(_VVCEnvBase):
    """
    Simplified 123-bus env for federated experiments.
    Devices: 2 caps + 1 reg  → same action space as 13-bus (132 actions).
    Observation dim: 123*3 + 2 + 1 = 372
    Use this as Client C in QE-SAC-FL.
    """
    _branches   = _gen_123bus_branches(123)
    _base_loads = _gen_123bus_loads(123)
    _cap_buses  = [10, 70]          # 2 caps
    _cap_sizes  = [600.0, 600.0]
    _n_regs     = 1
    _n_bats     = 0
    _n_buses    = 123


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name, cls in [("34bus full", VVCEnv34Bus),
                      ("34bus FL",   VVCEnv34BusFL),
                      ("123bus FL",  VVCEnv123BusFL)]:
        env = cls(seed=0)
        obs, _ = env.reset()
        print(f"{name:<12}  obs={env.observation_space.shape[0]}  "
              f"nvec={env.action_space.nvec}  "
              f"n_act={int(env.action_space.nvec.prod())}")
    print("OK")
