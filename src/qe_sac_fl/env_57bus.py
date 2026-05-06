"""
IEEE 57-bus VVC environment — Client D for QE-SAC-FL 4-topology experiment.

Approximate IEEE 57-bus Test System (standard transmission/sub-transmission).
Mid-scale network: between 34-bus and 123-bus in complexity.

Observation dim: 57*3 + 2 + 1 = 174
Action space:    MultiDiscrete([2, 2, 33]) → 132 joint actions (same as other FL clients)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from src.qe_sac.env_utils import _VVCEnvBase

# ---------------------------------------------------------------------------
# IEEE 57-bus branch data (r_pu, x_pu on 100 MVA base, from IEEE test case)
# ---------------------------------------------------------------------------
_IEEE57_BRANCHES = [
    (0,  1,  0.0083, 0.0280), (1,  2,  0.0298, 0.0850), (1,  3,  0.0112, 0.0366),
    (1,  4,  0.0625, 0.1320), (1,  5,  0.0430, 0.1480), (5,  6,  0.0200, 0.1020),
    (5,  7,  0.0339, 0.1730), (7,  8,  0.0099, 0.0505), (7,  29, 0.0369, 0.1679),
    (8,  9,  0.0369, 0.1679), (8,  10, 0.0258, 0.0848), (8,  11, 0.0648, 0.2950),
    (8,  12, 0.0481, 0.1580), (12, 13, 0.0132, 0.0434), (13, 14, 0.0269, 0.0869),
    (14, 15, 0.0525, 0.1700), (15, 16, 0.0183, 0.0934), (16, 17, 0.0238, 0.1080),
    (17, 18, 0.0454, 0.2060), (18, 19, 0.0648, 0.2950), (19, 20, 0.0481, 0.1580),
    (20, 21, 0.0132, 0.0434), (21, 22, 0.0112, 0.0366), (22, 23, 0.0625, 0.1320),
    (23, 24, 0.0430, 0.1480), (10, 25, 0.0200, 0.1020), (25, 26, 0.0339, 0.1730),
    (26, 27, 0.0099, 0.0505), (27, 28, 0.0369, 0.1679), (28, 29, 0.0369, 0.1679),
    (29, 30, 0.0258, 0.0848), (30, 31, 0.0648, 0.2950), (31, 32, 0.0481, 0.1580),
    (32, 33, 0.0132, 0.0434), (33, 34, 0.0269, 0.0869), (34, 35, 0.0525, 0.1700),
    (35, 36, 0.0183, 0.0934), (36, 37, 0.0238, 0.1080), (37, 38, 0.0454, 0.2060),
    (38, 39, 0.0258, 0.0848), (39, 40, 0.0648, 0.2950), (40, 41, 0.0481, 0.1580),
    (41, 42, 0.0132, 0.0434), (42, 43, 0.0269, 0.0869), (43, 44, 0.0525, 0.1700),
    (44, 45, 0.0183, 0.0934), (45, 46, 0.0238, 0.1080), (46, 47, 0.0454, 0.2060),
    (47, 48, 0.0369, 0.1679), (48, 49, 0.0258, 0.0848), (49, 50, 0.0648, 0.2950),
    (50, 51, 0.0481, 0.1580), (51, 52, 0.0132, 0.0434), (52, 53, 0.0269, 0.0869),
    (53, 54, 0.0525, 0.1700), (54, 55, 0.0183, 0.0934), (55, 56, 0.0238, 0.1080),
    (11, 41, 0.0200, 0.1020), (3,  18, 0.0339, 0.1730), (24, 25, 0.0099, 0.0505),
    (2,  42, 0.0648, 0.2950), (6,  11, 0.0481, 0.1580), (9,  55, 0.0132, 0.0434),
]

# Base loads per bus (P_kW, Q_kVAR) — 57 buses, total ~1250 kW
_IEEE57_BASE_LOADS = np.array([
    [0,   0  ], [55,  17 ], [45,  15 ], [0,   0  ], [13,  4  ],
    [75,  25 ], [0,   0  ], [150, 50 ], [0,   0  ], [78,  26 ],
    [0,   0  ], [0,   0  ], [0,   0  ], [10,  3  ], [0,   0  ],
    [22,  7  ], [0,   0  ], [0,   0  ], [32,  11 ], [0,   0  ],
    [6,   2  ], [0,   0  ], [0,   0  ], [0,   0  ], [0,   0  ],
    [0,   0  ], [0,   0  ], [0,   0  ], [0,   0  ], [24,  8  ],
    [0,   0  ], [58,  19 ], [0,   0  ], [0,   0  ], [0,   0  ],
    [0,   0  ], [0,   0  ], [0,   0  ], [0,   0  ], [0,   0  ],
    [0,   0  ], [0,   0  ], [0,   0  ], [10,  3  ], [0,   0  ],
    [22,  7  ], [0,   0  ], [0,   0  ], [0,   0  ], [0,   0  ],
    [0,   0  ], [17,  6  ], [0,   0  ], [0,   0  ], [0,   0  ],
    [0,   0  ], [0,   0  ],
], dtype=np.float32)

_IEEE57_CAP_BUSES  = [18, 42]        # 2 capacitor banks
_IEEE57_CAP_SIZES  = [200.0, 200.0]  # kVAR
_IEEE57_N_REGS     = 1               # 1 voltage regulator


class VVCEnv57BusFL(_VVCEnvBase):
    """
    VVC environment for IEEE 57-bus system — Client D in 4-topology FL.

    Simplified for federated experiments: 2 caps + 1 reg = 132 joint actions,
    matching the other 3 FL clients (13/34/123-bus).

    Observation dim: 57*3 + 2 + 1 = 174
    Action space: MultiDiscrete([2, 2, 33])
    """
    _branches   = _IEEE57_BRANCHES
    _base_loads = _IEEE57_BASE_LOADS
    _cap_buses  = _IEEE57_CAP_BUSES
    _cap_sizes  = _IEEE57_CAP_SIZES
    _n_regs     = _IEEE57_N_REGS
    _n_bats     = 0
    _n_buses    = 57


if __name__ == "__main__":
    env = VVCEnv57BusFL(seed=0)
    obs, _ = env.reset()
    print(f"57bus FL: obs={env.observation_space.shape[0]}  "
          f"nvec={env.action_space.nvec}  "
          f"n_act={int(env.action_space.nvec.prod())}")
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, done, trunc, _ = env.step(a)
        print(f"  reward={r:.3f}  done={done}")
    print("OK")
