"""
VVC (Volt-VAR Control) Gymnasium environments for IEEE 13-bus and 123-bus
distribution feeders.

Implements a linearized DistFlow power flow model:
    V_i^2 ≈ V_j^2 - 2*(r_ij*P_ij + x_ij*Q_ij)

Interface mirrors PowerGym so the agent code is simulator-agnostic.

State (observation):
    [V_pu at each bus | P_load at each bus | Q_load at each bus |
     cap_status | reg_tap_normalized | battery_soc]

Action (MultiDiscrete):
    capacitors: ON(1)/OFF(0)
    regulators: tap in {0..32}  → normalized to [-1, 1]
    batteries:  charge level in {0..32} → normalized to [-1, 1]

Reward:
    r = -[α*f_vv + β*f_cl + γ*f_pl]
    f_vv = sum of squared voltage violations outside [0.95, 1.05] pu
    f_cl = capacitor switching cost (number of changes × penalty)
    f_pl = total active power loss (pu)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# IEEE 13-bus linearized network parameters
# Branch data: (from_bus, to_bus, r_pu, x_pu) on 100 MVA, 4.16 kV base
# Approximate values derived from IEEE 13 Node Test Feeder documentation.
# ---------------------------------------------------------------------------

# Real IEEE 13-bus feeder — extracted from OpenDSS IEEE13Nodeckt.dss
# Source: IEEE PES 1992 Test Feeder Cases, 4.16 kV base
# Bus map: 611→0, 632→1, 633→2, 645→3, 646→4, 652→5, 670→6, 671→7,
#          675→8, 680→9, 684→10, 692→11, rg60→12
_IEEE13_BRANCHES = [
    (12, 1,  0.127,  0.264),
    (1,  6,  0.0423, 0.088),
    (6,  7,  0.0846, 0.1759),
    (7,  9,  0.0635, 0.132),
    (1,  2,  0.0317, 0.066),
    (1,  3,  0.0317, 0.066),
    (3,  4,  0.019,  0.0396),
    (11, 8,  0.0317, 0.066),
    (7,  10, 0.019,  0.0396),
    (10, 0,  0.019,  0.0396),
    (10, 5,  0.0508, 0.1056),
    (7,  11, 0.001,  0.001),
]

# Base load at each bus (P_kW, Q_kVAR) — real IEEE 13-bus loads
_IEEE13_BASE_LOADS = np.array([
    [170.0,  80.0],   # bus 0 (611)
    [0.0,    0.0],    # bus 1 (632) — junction
    [0.0,    0.0],    # bus 2 (633) — junction
    [170.0, 125.0],   # bus 3 (645)
    [230.0, 132.0],   # bus 4 (646)
    [128.0,  86.0],   # bus 5 (652)
    [200.0, 116.0],   # bus 6 (670)
    [1155.0,660.0],   # bus 7 (671) — main load
    [843.0, 462.0],   # bus 8 (675)
    [0.0,   0.0],     # bus 9 (680) — junction
    [0.0,   0.0],     # bus 10 (684) — junction
    [170.0, 151.0],   # bus 11 (692)
    [0.0,   0.0],     # bus 12 (rg60) — substation
], dtype=np.float32)

# Capacitor bank buses and sizes (kVAR)
_IEEE13_CAP_BUSES = [8, 11]        # 2 capacitors
_IEEE13_CAP_SIZES = [600.0, 200.0] # kVAR

# Voltage regulator branch index (the 4→5 branch = index 4)
_IEEE13_REG_BRANCH = 4
_IEEE13_N_REGS = 1
_REG_TAPS = 33          # 0..32 → ratio in [0.9, 1.1]
_BAT_LEVELS = 33        # 0..32 → SoC in [0, 1]

# Base MVA & kV for per-unit conversion
_BASE_MVA = 1.0         # 1 MVA base
_BASE_KV = 4.16         # 4.16 kV base
_BASE_KVA = _BASE_MVA * 1000

V_MIN, V_MAX = 0.95, 1.05  # pu voltage limits


def _build_incidence(n_buses: int, branches: list[tuple]) -> np.ndarray:
    """Bus-branch incidence matrix A (n_buses × n_branches)."""
    n_br = len(branches)
    A = np.zeros((n_buses, n_br), dtype=np.float32)
    for k, (i, j, _, _) in enumerate(branches):
        A[i, k] = 1.0
        A[j, k] = -1.0
    return A


def _tap_to_ratio(tap: int, n_taps: int = _REG_TAPS) -> float:
    """Map tap integer [0, n_taps-1] to voltage ratio [0.9, 1.1]."""
    return 0.9 + (tap / (n_taps - 1)) * 0.2


def _distflow_voltages(
    p_inj: np.ndarray,
    q_inj: np.ndarray,
    branches: list[tuple],
    n_buses: int,
    reg_tap: int = 16,
) -> tuple[np.ndarray, float]:
    """
    Linearised DistFlow: propagate voltages from substation (bus 0, V=1 pu).
    Returns (V_pu array, P_loss_pu scalar).
    """
    V2 = np.ones(n_buses, dtype=np.float32)   # V^2 in pu
    P_flow = np.zeros(len(branches), dtype=np.float32)
    Q_flow = np.zeros(len(branches), dtype=np.float32)

    # Build load flow from leaves to root (backwards sweep — BFS order reversed)
    # For a radial feeder we traverse branches in order (root → leaf = indices 0→N)
    # Forward sweep: accumulate injections, then backward update voltages
    P_net = p_inj.copy()
    Q_net = q_inj.copy()

    reg_ratio = _tap_to_ratio(reg_tap)

    for k, (fr, to, r, x) in enumerate(branches):
        P_flow[k] = -P_net[to]
        Q_flow[k] = -Q_net[to]
        P_net[fr] += P_flow[k]
        Q_net[fr] += Q_flow[k]

    for k, (fr, to, r, x) in enumerate(branches):
        dV2 = 2.0 * (r * P_flow[k] + x * Q_flow[k])
        V2[to] = V2[fr] - dV2
        # Apply regulator boost on the regulated branch
        if k == _IEEE13_REG_BRANCH:
            V2[to] = V2[to] * (reg_ratio ** 2)

    V2 = np.clip(V2, 0.5, 1.5)
    V_pu = np.sqrt(V2)

    # Total resistive losses (pu)
    P_loss = sum(
        branches[k][2] * (P_flow[k] ** 2 + Q_flow[k] ** 2) / max(V2[branches[k][0]], 0.01)
        for k in range(len(branches))
    )
    return V_pu, float(P_loss)


class _VVCEnvBase(gym.Env):
    """
    Base class for VVC environments.
    Subclasses supply network parameters via class attributes.
    """

    metadata = {"render_modes": []}

    # Override in subclasses
    _branches: list[tuple] = []
    _base_loads: np.ndarray = np.zeros((1, 2))
    _cap_buses: list[int] = []
    _cap_sizes: list[float] = []
    _n_regs: int = 0
    _n_bats: int = 0
    _n_buses: int = 0

    # Reward weights
    ALPHA = 100.0   # voltage violation weight
    BETA  = 1.0     # capacitor switching weight
    GAMMA = 0.1     # power loss weight

    def __init__(self, episode_len: int = 24, load_noise: float = 0.1, seed: int | None = None):
        super().__init__()
        self._ep_len = episode_len
        self._noise = load_noise
        self._rng = np.random.default_rng(seed)

        n_caps = len(self._cap_buses)
        n_regs = self._n_regs
        n_bats = self._n_bats

        # Action space: MultiDiscrete
        #   caps: 0/1 each, regs: 0..32 each, bats: 0..32 each
        disc = ([2] * n_caps) + ([_REG_TAPS] * n_regs) + ([_BAT_LEVELS] * n_bats)
        self.action_space = spaces.MultiDiscrete(disc)

        # Observation: voltages + P_loads + Q_loads + cap_status + reg_tap_norm + bat_soc
        obs_dim = (
            self._n_buses          # voltages
            + self._n_buses        # P_load (normalised)
            + self._n_buses        # Q_load (normalised)
            + n_caps               # cap status
            + n_regs               # reg tap normalised [0,1]
            + n_bats               # battery SoC [0,1]
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._t: int = 0
        self._cap_status = np.zeros(n_caps, dtype=np.float32)
        self._reg_tap = np.full(n_regs, 16, dtype=np.int32)
        self._bat_soc = np.full(n_bats, 16, dtype=np.int32)
        self._load_scale = 1.0

    # ------------------------------------------------------------------
    def _sample_loads(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (P_pu, Q_pu) with random noise around base loads."""
        scale = self._load_scale * (1.0 + self._rng.uniform(-self._noise, self._noise))
        P = self._base_loads[:, 0] * scale / _BASE_KVA   # kW → pu
        Q = self._base_loads[:, 1] * scale / _BASE_KVA
        return P.astype(np.float32), Q.astype(np.float32)

    def _apply_caps(self, Q: np.ndarray) -> np.ndarray:
        Q = Q.copy()
        for idx, (bus, size) in enumerate(zip(self._cap_buses, self._cap_sizes)):
            if self._cap_status[idx] > 0.5:
                Q[bus] -= size / _BASE_KVA   # capacitor injects Q
        return Q

    def _get_obs(self, V: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        n_caps = len(self._cap_buses)
        reg_norm = self._reg_tap / (_REG_TAPS - 1) if self._n_regs > 0 else np.array([])
        bat_norm = self._bat_soc / (_BAT_LEVELS - 1) if self._n_bats > 0 else np.array([])
        return np.concatenate([
            V,
            P / (P.max() + 1e-8),
            Q / (Q.max() + 1e-8),
            self._cap_status,
            reg_norm.astype(np.float32),
            bat_norm.astype(np.float32),
        ]).astype(np.float32)

    def _compute_reward(
        self,
        V: np.ndarray,
        P_loss: float,
        prev_cap: np.ndarray,
    ) -> tuple[float, int]:
        # Voltage violation: sum of squared exceedances
        viol_lo = np.maximum(V_MIN - V, 0.0) ** 2
        viol_hi = np.maximum(V - V_MAX, 0.0) ** 2
        f_vv = float(np.sum(viol_lo + viol_hi))
        n_viol = int(np.sum((V < V_MIN) | (V > V_MAX)))

        # Capacitor switching cost
        f_cl = float(np.sum(np.abs(self._cap_status - prev_cap)))

        # Power loss
        f_pl = P_loss

        reward = -(self.ALPHA * f_vv + self.BETA * f_cl + self.GAMMA * f_pl)
        return float(reward), n_viol

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._load_scale = self._rng.uniform(0.9, 1.1)
        self._cap_status = np.zeros(len(self._cap_buses), dtype=np.float32)
        self._reg_tap = np.full(self._n_regs, 16, dtype=np.int32)
        self._bat_soc = np.full(self._n_bats, 16, dtype=np.int32)

        P, Q = self._sample_loads()
        Q_eff = self._apply_caps(Q)
        V, P_loss = _distflow_voltages(
            -P, -Q_eff, self._branches, self._n_buses,
            reg_tap=int(self._reg_tap[0]) if self._n_regs > 0 else 16,
        )
        obs = self._get_obs(V, P, Q)
        return obs, {"voltage": V, "P_loss": P_loss, "v_viol": 0}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        n_caps = len(self._cap_buses)
        prev_cap = self._cap_status.copy()

        # Decode action
        self._cap_status = action[:n_caps].astype(np.float32)
        if self._n_regs > 0:
            self._reg_tap = action[n_caps: n_caps + self._n_regs].astype(np.int32)
        if self._n_bats > 0:
            self._bat_soc = action[n_caps + self._n_regs:].astype(np.int32)

        # Advance load profile
        self._t += 1
        self._load_scale = self._rng.uniform(0.9, 1.1)
        P, Q = self._sample_loads()
        Q_eff = self._apply_caps(Q)

        reg_tap = int(self._reg_tap[0]) if self._n_regs > 0 else 16
        V, P_loss = _distflow_voltages(
            -P, -Q_eff, self._branches, self._n_buses, reg_tap=reg_tap
        )

        reward, n_viol = self._compute_reward(V, P_loss, prev_cap)
        obs = self._get_obs(V, P, Q)
        terminated = self._t >= self._ep_len
        return obs, reward, terminated, False, {"voltage": V, "P_loss": P_loss, "v_viol": n_viol}


# ---------------------------------------------------------------------------
# IEEE 13-bus VVC environment
# ---------------------------------------------------------------------------

class VVCEnv13Bus(_VVCEnvBase):
    """
    VVC environment for IEEE 13-bus distribution feeder.
    Devices: 2 capacitor banks, 1 voltage regulator, 1 battery.
    Observation dim: 13*3 + 2 + 1 + 1 = 43
    """
    _branches  = _IEEE13_BRANCHES
    _base_loads = _IEEE13_BASE_LOADS
    _cap_buses  = _IEEE13_CAP_BUSES
    _cap_sizes  = _IEEE13_CAP_SIZES
    _n_regs     = _IEEE13_N_REGS
    _n_bats     = 1
    _n_buses    = 13


# ---------------------------------------------------------------------------
# IEEE 123-bus VVC environment (scaled-up synthetic)
# ---------------------------------------------------------------------------

# Real IEEE 123-bus feeder data extracted from OpenDSS IEEE123Master.dss
# Source: IEEE PES 1992 Test Feeder Cases, approved 2000 PES Summer Meeting
# 114 buses (nodes 1-123 of real feeder), 107 branches, 4.16 kV base
_IEEE123_BRANCHES = [
    (0, 1, 0.0587, 0.122),
    (0, 2, 0.0838, 0.1742),
    (0, 6, 0.1005, 0.2091),
    (2, 3, 0.067, 0.1394),
    (2, 4, 0.1089, 0.2265),
    (4, 5, 0.0838, 0.1742),
    (6, 7, 0.067, 0.1394),
    (7, 11, 0.0754, 0.1568),
    (7, 8, 0.0754, 0.1568),
    (7, 12, 0.1005, 0.2091),
    (12, 33, 0.0503, 0.1045),
    (12, 17, 0.2765, 0.5749),
    (13, 10, 0.0838, 0.1742),
    (13, 9, 0.0838, 0.1742),
    (14, 15, 0.1257, 0.2613),
    (14, 16, 0.1173, 0.2439),
    (17, 18, 0.0838, 0.1742),
    (17, 20, 0.1005, 0.2091),
    (18, 19, 0.1089, 0.2265),
    (20, 21, 0.176, 0.3659),
    (20, 22, 0.0838, 0.1742),
    (22, 23, 0.1843, 0.3833),
    (22, 24, 0.0922, 0.1916),
    (24, 27, 0.067, 0.1394),
    (25, 26, 0.0922, 0.1916),
    (25, 30, 0.0754, 0.1568),
    (26, 32, 0.1676, 0.3484),
    (27, 28, 0.1005, 0.2091),
    (28, 29, 0.1173, 0.2439),
    (30, 31, 0.1005, 0.2091),
    (33, 14, 0.0335, 0.0697),
    (34, 35, 0.2178, 0.453),
    (34, 39, 0.0838, 0.1742),
    (35, 36, 0.1005, 0.2091),
    (35, 37, 0.0838, 0.1742),
    (37, 38, 0.1089, 0.2265),
    (39, 40, 0.1089, 0.2265),
    (39, 41, 0.0838, 0.1742),
    (41, 42, 0.1676, 0.3484),
    (41, 43, 0.067, 0.1394),
    (43, 44, 0.067, 0.1394),
    (43, 46, 0.0838, 0.1742),
    (44, 45, 0.1005, 0.2091),
    (46, 47, 0.0503, 0.1045),
    (46, 48, 0.0838, 0.1742),
    (48, 49, 0.0838, 0.1742),
    (49, 50, 0.0838, 0.1742),
    (51, 52, 0.067, 0.1394),
    (52, 53, 0.0419, 0.0871),
    (53, 54, 0.0922, 0.1916),
    (53, 56, 0.1173, 0.2439),
    (54, 55, 0.0922, 0.1916),
    (56, 57, 0.0838, 0.1742),
    (56, 59, 0.2514, 0.5227),
    (57, 58, 0.0838, 0.1742),
    (59, 60, 0.1843, 0.3833),
    (59, 61, 0.0838, 0.1742),
    (61, 62, 0.0587, 0.122),
    (62, 63, 0.1173, 0.2439),
    (63, 64, 0.1424, 0.2962),
    (64, 65, 0.1089, 0.2265),
    (66, 67, 0.067, 0.1394),
    (66, 71, 0.0922, 0.1916),
    (66, 96, 0.0838, 0.1742),
    (67, 68, 0.0922, 0.1916),
    (68, 69, 0.1089, 0.2265),
    (69, 70, 0.0922, 0.1916),
    (71, 72, 0.0922, 0.1916),
    (71, 75, 0.067, 0.1394),
    (72, 73, 0.1173, 0.2439),
    (73, 74, 0.1341, 0.2788),
    (75, 76, 0.1341, 0.2788),
    (75, 85, 0.2346, 0.4878),
    (76, 77, 0.0335, 0.0697),
    (77, 78, 0.0754, 0.1568),
    (77, 79, 0.1592, 0.331),
    (79, 80, 0.0587, 0.122),
    (80, 81, 0.0838, 0.1742),
    (80, 83, 0.2262, 0.4704),
    (81, 82, 0.0838, 0.1742),
    (83, 84, 0.1592, 0.331),
    (85, 86, 0.1508, 0.3136),
    (86, 87, 0.0587, 0.122),
    (86, 88, 0.0922, 0.1916),
    (88, 89, 0.0838, 0.1742),
    (88, 90, 0.0754, 0.1568),
    (90, 91, 0.1005, 0.2091),
    (90, 92, 0.0754, 0.1568),
    (92, 93, 0.0922, 0.1916),
    (92, 94, 0.1005, 0.2091),
    (94, 95, 0.067, 0.1394),
    (96, 97, 0.0922, 0.1916),
    (97, 98, 0.1843, 0.3833),
    (98, 99, 0.1005, 0.2091),
    (100, 101, 0.0754, 0.1568),
    (100, 104, 0.0922, 0.1916),
    (101, 102, 0.1089, 0.2265),
    (102, 103, 0.2346, 0.4878),
    (104, 105, 0.0754, 0.1568),
    (104, 107, 0.1089, 0.2265),
    (105, 106, 0.1927, 0.4007),
    (107, 108, 0.1508, 0.3136),
    (108, 109, 0.1005, 0.2091),
    (109, 110, 0.1927, 0.4007),
    (109, 111, 0.0419, 0.0871),
    (111, 112, 0.176, 0.3659),
    (112, 113, 0.1089, 0.2265),
]

_IEEE123_BASE_LOADS = np.array([
    [40.0, 20.0], [20.0, 10.0], [0.0, 0.0],   [40.0, 20.0], [20.0, 10.0],
    [40.0, 20.0], [20.0, 10.0], [0.0, 0.0],   [40.0, 20.0], [20.0, 10.0],
    [40.0, 20.0], [20.0, 10.0], [0.0, 0.0],   [0.0, 0.0],   [0.0, 0.0],
    [40.0, 20.0], [20.0, 10.0], [0.0, 0.0],   [40.0, 20.0], [40.0, 20.0],
    [0.0, 0.0],   [40.0, 20.0], [0.0, 0.0],   [40.0, 20.0], [0.0, 0.0],
    [0.0, 0.0],   [0.0, 0.0],   [40.0, 20.0], [40.0, 20.0], [40.0, 20.0],
    [20.0, 10.0], [20.0, 10.0], [40.0, 20.0], [40.0, 20.0], [40.0, 20.0],
    [0.0, 0.0],   [40.0, 20.0], [20.0, 10.0], [20.0, 10.0], [0.0, 0.0],
    [20.0, 10.0], [20.0, 10.0], [40.0, 20.0], [0.0, 0.0],   [20.0, 10.0],
    [20.0, 10.0], [105.0, 75.0],[210.0,150.0],[140.0, 95.0],[40.0, 20.0],
    [20.0, 10.0], [40.0, 20.0], [40.0, 20.0], [0.0, 0.0],   [20.0, 10.0],
    [20.0, 10.0], [0.0, 0.0],   [20.0, 10.0], [20.0, 10.0], [20.0, 10.0],
    [0.0, 0.0],   [40.0, 20.0], [40.0, 20.0], [75.0, 35.0], [140.0,100.0],
    [75.0, 35.0], [0.0, 0.0],   [20.0, 10.0], [40.0, 20.0], [20.0, 10.0],
    [40.0, 20.0], [0.0, 0.0],   [40.0, 20.0], [40.0, 20.0], [40.0, 20.0],
    [245.0,180.0],[40.0, 20.0], [0.0, 0.0],   [40.0, 20.0], [40.0, 20.0],
    [0.0, 0.0],   [40.0, 20.0], [20.0, 10.0], [20.0, 10.0], [40.0, 20.0],
    [20.0, 10.0], [40.0, 20.0], [40.0, 20.0], [0.0, 0.0],   [40.0, 20.0],
    [0.0, 0.0],   [40.0, 20.0], [0.0, 0.0],   [40.0, 20.0], [20.0, 10.0],
    [20.0, 10.0], [0.0, 0.0],   [40.0, 20.0], [40.0, 20.0], [40.0, 20.0],
    [0.0, 0.0],   [20.0, 10.0], [40.0, 20.0], [40.0, 20.0], [0.0, 0.0],
    [40.0, 20.0], [40.0, 20.0], [0.0, 0.0],   [40.0, 20.0], [0.0, 0.0],
    [20.0, 10.0], [20.0, 10.0], [40.0, 20.0], [20.0, 10.0],
], dtype=np.float32)

_IEEE123_CAP_BUSES  = [10, 20, 33, 50, 70, 90, 110]   # 7 capacitors
_IEEE123_CAP_SIZES  = [600.0] * 7
_IEEE123_N_REGS     = 4
_IEEE123_N_BATS     = 0
_IEEE123_N_BUSES    = 114


class VVCEnv123Bus(_VVCEnvBase):
    """
    VVC environment for IEEE 123-bus distribution feeder (real IEEE PES 1992 data).
    Extracted from OpenDSS IEEE123Master.dss — 114 active buses, 107 branches, 4.16 kV.
    Devices: 7 capacitor banks, 4 voltage regulators.
    Observation dim: 114*3 + 7 + 4 = 353
    """
    _branches   = _IEEE123_BRANCHES
    _base_loads = _IEEE123_BASE_LOADS
    _cap_buses  = _IEEE123_CAP_BUSES
    _cap_sizes  = _IEEE123_CAP_SIZES
    _n_regs     = _IEEE123_N_REGS
    _n_bats     = 0
    _n_buses    = 114


# ---------------------------------------------------------------------------
# IEEE 34-bus VVC environment (simplified radial feeder)
# Topology: main trunk 0-20 + three laterals (21-25, 26-28, 29-33)
# Devices: 2 capacitors, 1 voltage regulator
# Observation dim: 34*3 + 2 + 1 = 105   n_actions: 2*2*33 = 132
# ---------------------------------------------------------------------------

_IEEE34_BRANCHES = [
    # Main trunk (0 → 20)
    (0,  1,  0.100, 0.074),
    (1,  2,  0.181, 0.079),
    (2,  3,  0.143, 0.061),
    (3,  4,  0.168, 0.072),
    (4,  5,  0.000, 0.000),   # voltage regulator (branch index 4)
    (5,  6,  0.154, 0.066),
    (6,  7,  0.198, 0.082),
    (7,  8,  0.127, 0.055),
    (8,  9,  0.163, 0.070),
    (9,  10, 0.175, 0.076),
    (10, 11, 0.188, 0.081),
    (11, 12, 0.142, 0.060),
    (12, 13, 0.156, 0.067),
    (13, 14, 0.193, 0.083),
    (14, 15, 0.169, 0.073),
    (15, 16, 0.147, 0.063),
    (16, 17, 0.184, 0.079),
    (17, 18, 0.161, 0.069),
    (18, 19, 0.177, 0.076),
    (19, 20, 0.135, 0.058),
    # Lateral A (from bus 5, buses 21-25)
    (5,  21, 0.221, 0.091),
    (21, 22, 0.248, 0.103),
    (22, 23, 0.215, 0.089),
    (23, 24, 0.267, 0.111),
    (24, 25, 0.239, 0.099),
    # Lateral B (from bus 10, buses 26-28)
    (10, 26, 0.233, 0.097),
    (26, 27, 0.251, 0.104),
    (27, 28, 0.218, 0.090),
    # Lateral C (from bus 16, buses 29-33)
    (16, 29, 0.244, 0.101),
    (29, 30, 0.262, 0.109),
    (30, 31, 0.228, 0.094),
    (31, 32, 0.255, 0.106),
    (32, 33, 0.235, 0.097),
]

_IEEE34_BASE_LOADS = np.array([
    [0,    0],     # bus 0  — substation
    [60,   42],    # bus 1
    [80,   56],    # bus 2
    [50,   35],    # bus 3
    [90,   63],    # bus 4
    [45,   32],    # bus 5
    [75,   53],    # bus 6
    [110,  77],    # bus 7
    [65,   46],    # bus 8
    [85,   60],    # bus 9
    [55,   39],    # bus 10
    [100,  70],    # bus 11
    [70,   49],    # bus 12
    [95,   67],    # bus 13
    [60,   42],    # bus 14
    [80,   56],    # bus 15
    [50,   35],    # bus 16
    [90,   63],    # bus 17
    [75,   53],    # bus 18
    [65,   46],    # bus 19
    [55,   39],    # bus 20
    [40,   28],    # bus 21  lateral A
    [50,   35],    # bus 22
    [35,   25],    # bus 23
    [45,   32],    # bus 24
    [30,   21],    # bus 25
    [60,   42],    # bus 26  lateral B
    [50,   35],    # bus 27
    [40,   28],    # bus 28
    [55,   39],    # bus 29  lateral C
    [45,   32],    # bus 30
    [35,   25],    # bus 31
    [60,   42],    # bus 32
    [40,   28],    # bus 33
], dtype=np.float32)

_IEEE34_CAP_BUSES = [8, 25]          # 2 capacitor banks
_IEEE34_CAP_SIZES = [450.0, 300.0]   # kVAR
_IEEE34_REG_BRANCH = 4               # regulator on branch 4 (bus 4→5)
_IEEE34_N_REGS = 1


class VVCEnv34Bus(_VVCEnvBase):
    """
    VVC environment for IEEE 34-bus distribution feeder (simplified radial).
    Devices: 2 capacitor banks, 1 voltage regulator.
    Observation dim: 34*3 + 2 + 1 = 105   n_actions: 2*2*33 = 132
    """
    _branches   = _IEEE34_BRANCHES
    _base_loads = _IEEE34_BASE_LOADS
    _cap_buses  = _IEEE34_CAP_BUSES
    _cap_sizes  = _IEEE34_CAP_SIZES
    _n_regs     = _IEEE34_N_REGS
    _n_bats     = 0
    _n_buses    = 34


# ---------------------------------------------------------------------------
# IEEE 123-bus — paper-scale config (4 caps + 1 reg, feasible action space)
# Devices: 4 capacitors, 1 voltage regulator
# Observation dim: 123*3 + 4 + 1 = 374   n_actions: 2^4 * 33 = 528
# ---------------------------------------------------------------------------

_IEEE123P_CAP_BUSES = [10, 33, 70, 110]    # 4 capacitors
_IEEE123P_CAP_SIZES = [600.0] * 4
_IEEE123P_N_REGS    = 1


class VVCEnv123BusPaper(_VVCEnvBase):
    """
    IEEE 123-bus VVC environment scaled for tractable training.
    Devices: 4 capacitors, 1 voltage regulator.
    Observation dim: 123*3 + 4 + 1 = 374   n_actions: 2^4*33 = 528
    Uses same synthetic topology as VVCEnv123Bus.
    """
    _branches   = _IEEE123_BRANCHES
    _base_loads = _IEEE123_BASE_LOADS
    _cap_buses  = _IEEE123P_CAP_BUSES
    _cap_sizes  = _IEEE123P_CAP_SIZES
    _n_regs     = _IEEE123P_N_REGS
    _n_bats     = 0
    _n_buses    = 123
