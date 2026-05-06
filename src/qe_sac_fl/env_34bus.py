"""
IEEE 34-bus VVC environment — Client B for QE-SAC-FL.

Real IEEE 34-bus Test Feeder (IEEE PES 1992, EPRI standard).
Extracted from OpenDSS ieee34Mod1.dss — 36 buses, 32 branches, 24.9 kV base.
Medium-sized rural feeder: longer branches, higher impedance than 13-bus.

Observation dim: 36 buses × 3 (V, P, Q) + 4 caps + 2 regs = 114
Action space:    MultiDiscrete([2, 2, 2, 2, 33, 33])
                 → 4 cap banks (ON/OFF) × 2 regulators (33 taps each)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from src.qe_sac.env_utils import _VVCEnvBase, _IEEE123_BRANCHES, _IEEE123_BASE_LOADS


# ---------------------------------------------------------------------------
# IEEE 34-bus linearized network parameters
# Real data extracted from OpenDSS ieee34Mod1.dss (IEEE PES 1992 Test Feeder)
# Branch data: (from_bus, to_bus, r_pu, x_pu) on 100 MVA, 24.9 kV base
# Z_base = 24.9² / 100 = 6.2001 Ω
# 36 buses (includes regulator/transformer nodes), 32 branches
# ---------------------------------------------------------------------------

_IEEE34_BRANCHES = [
    (0,  1,  0.0241, 0.0502),
    (1,  2,  0.0162, 0.0337),
    (2,  3,  0.3015, 0.6269),
    (3,  4,  0.0543, 0.1129),
    (3,  5,  0.3508, 0.7294),
    (5,  6,  0.2781, 0.5783),
    (7,  25, 0.0001, 0.0002),
    (8,  9,  0.0160, 0.0333),
    (8,  12, 0.0955, 0.1986),
    (9,  10, 0.4504, 0.9366),
    (10, 11, 0.1285, 0.2673),
    (12, 13, 0.0283, 0.0589),
    (12, 14, 0.0079, 0.0163),
    (14, 15, 0.1912, 0.3976),
    (15, 28, 0.0049, 0.0101),
    (16, 30, 0.0458, 0.0953),
    (17, 31, 0.0189, 0.0393),
    (17, 21, 0.0026, 0.0054),
    (18, 20, 0.0080, 0.0167),
    (18, 32, 0.0026, 0.0054),
    (21, 22, 0.0126, 0.0263),
    (22, 23, 0.0341, 0.0708),
    (23, 24, 0.0050, 0.0103),
    (25, 8,  0.0029, 0.0060),
    (27, 16, 0.0001, 0.0002),
    (28, 29, 0.2182, 0.4538),
    (28, 26, 0.3445, 0.7164),
    (30, 33, 0.0152, 0.0315),
    (30, 17, 0.0545, 0.1134),
    (31, 18, 0.0251, 0.0521),
    (32, 19, 0.0455, 0.0945),
    (34, 35, 0.0988, 0.2054),
]

# Base loads (P_kW, Q_kVAR) per bus — real IEEE 34-bus total ≈ 1900 kW
_IEEE34_BASE_LOADS = np.array([
    [0.0,   0.0  ],  # bus 0  — substation
    [27.5,  14.5 ],  # bus 1
    [27.5,  14.5 ],  # bus 2
    [8.0,   4.0  ],  # bus 3
    [8.0,   4.0  ],  # bus 4
    [0.0,   0.0  ],  # bus 5
    [0.0,   0.0  ],  # bus 6
    [0.0,   0.0  ],  # bus 7  — regulator
    [2.5,   1.0  ],  # bus 8
    [17.0,  8.5  ],  # bus 9
    [84.5,  43.5 ],  # bus 10
    [67.5,  35.0 ],  # bus 11
    [24.5,  12.0 ],  # bus 12
    [20.0,  10.0 ],  # bus 13
    [5.5,   2.5  ],  # bus 14
    [48.5,  21.5 ],  # bus 15
    [7.5,   3.5  ],  # bus 16
    [89.0,  45.0 ],  # bus 17
    [61.0,  31.5 ],  # bus 18
    [14.0,  7.0  ],  # bus 19
    [47.0,  31.0 ],  # bus 20
    [4.5,   2.5  ],  # bus 21
    [432.0, 329.0],  # bus 22
    [34.0,  17.0 ],  # bus 23
    [71.5,  53.5 ],  # bus 24
    [0.0,   0.0  ],  # bus 25
    [0.0,   0.0  ],  # bus 26
    [0.0,   0.0  ],  # bus 27
    [2.0,   1.0  ],  # bus 28
    [2.0,   1.0  ],  # bus 29
    [24.5,  12.5 ],  # bus 30
    [174.0, 106.0],  # bus 31
    [14.0,  7.0  ],  # bus 32
    [1.0,   0.5  ],  # bus 33
    [0.0,   0.0  ],  # bus 34
    [450.0, 225.0],  # bus 35
], dtype=np.float32)

# Capacitor banks: buses 11, 24, 29, 30 (real 34-bus cap placement)
_IEEE34_CAP_BUSES = [11, 24, 29, 30]
_IEEE34_CAP_SIZES = [300.0, 150.0, 150.0, 100.0]   # kVAR

# Two voltage regulators
_IEEE34_N_REGS = 2
_IEEE34_N_BATS = 0


class VVCEnv34Bus(_VVCEnvBase):
    """
    VVC environment for IEEE 34-bus distribution feeder (real OpenDSS data).

    This is Client B in the QE-SAC-FL federated setup.
    36 buses, 32 branches, 24.9 kV base — rural distribution feeder.

    Devices: 4 capacitor banks, 2 voltage regulators.
    Observation dim: 36*3 + 4 + 2 = 114
    Action space: MultiDiscrete([2, 2, 2, 2, 33, 33])
                  → joint size = 2^4 × 33^2 = 17,424
    """
    _branches   = _IEEE34_BRANCHES
    _base_loads = _IEEE34_BASE_LOADS
    _cap_buses  = _IEEE34_CAP_BUSES
    _cap_sizes  = _IEEE34_CAP_SIZES
    _n_regs     = _IEEE34_N_REGS
    _n_bats     = _IEEE34_N_BATS
    _n_buses    = 36


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
    Simplified 34-bus env for federated experiments (real OpenDSS data).
    36 buses, 32 branches, 24.9 kV base.
    Devices: 2 caps + 1 reg + 2 batteries (matching Lin et al. 2025).
    Observation dim: 36*3 + 2 + 1 + 2 = 113
    Use this as Client B in QE-SAC-FL.
    """
    _branches   = _IEEE34_BRANCHES
    _base_loads = _IEEE34_BASE_LOADS
    _cap_buses  = _IEEE34_CAP_BUSES_SIMPLE
    _cap_sizes  = _IEEE34_CAP_SIZES_SIMPLE
    _n_regs     = _IEEE34_N_REGS_SIMPLE
    _n_bats     = 2
    _n_buses    = 36


# ---------------------------------------------------------------------------
# Simplified 123-bus for FL experiments (2 caps + 1 reg = 132 actions)
# ---------------------------------------------------------------------------

class VVCEnv123BusFL(_VVCEnvBase):
    """
    IEEE 123-bus env for federated experiments (real IEEE PES 1992 data).
    Extracted from OpenDSS IEEE123Master.dss — 114 active buses, 107 branches.
    Devices: 2 caps + 1 reg + 4 batteries (matching Lin et al. 2025).
    Observation dim: 114*3 + 2 + 1 + 4 = 349
    Use this as Client C in QE-SAC-FL.
    """
    _branches   = _IEEE123_BRANCHES
    _base_loads = _IEEE123_BASE_LOADS
    _cap_buses  = [10, 70]
    _cap_sizes  = [600.0, 600.0]
    _n_regs     = 1
    _n_bats     = 4
    _n_buses    = 114


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
