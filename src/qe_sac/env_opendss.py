"""
VVCEnvOpenDSS — IEEE 13-bus Volt-VAR Control using real OpenDSS power flow.

Replaces the linearised DistFlow approximation with exact 3-phase AC
unbalanced power flow via opendssdirect.py.

Key differences vs VVCEnv13Bus (DistFlow):
  - Physics : exact 3-phase AC (not linearised DC approximation)
  - State   : per-phase voltages at all buses (~93-dim vs 42-dim)
  - Losses  : real I²R losses from AC solution
  - Voltage : per-phase magnitudes, catches unbalanced conditions

Same interface as VVCEnv13Bus:
  - Action space : MultiDiscrete([2, 2, 33])  cap1 / cap2 / reg tap
  - Reward       : α·VViol + β·switching + γ·losses  (same formula)
  - Episode len  : 24 steps (1 simulated day)
  - Gymnasium    : reset() / step() / observation_space / action_space
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import opendssdirect as dss


# ── IEEE 13-bus DSS circuit definition ──────────────────────────────────────

_DSS_COMMANDS = [
    "clear",
    "new circuit.ieee13 basekv=115 pu=1.0001 phases=3 bus1=sourcebus mvasc3=20000 mvasc1=21000",
    "new transformer.sub phases=3 windings=2 wdg=1 bus=sourcebus conn=delta kv=115 kva=5000 %r=0.0005 xhl=0.00001 wdg=2 bus=650 conn=wye kv=4.16 kva=5000 %r=0.0005",
    "new linecode.mtx601 nphases=3 basefreq=60 rmatrix=[0.3465 | 0.1560 0.3375 | 0.1580 0.1535 0.3414] xmatrix=[1.0179 | 0.5017 1.0478 | 0.4236 0.3849 1.0348] units=mi",
    "new linecode.mtx602 nphases=3 basefreq=60 rmatrix=[0.7526 | 0.1580 0.7475 | 0.1560 0.1535 0.7436] xmatrix=[1.1814 | 0.4236 1.1983 | 0.5017 0.3849 1.2112] units=mi",
    "new line.650632 phases=3 bus1=650 bus2=632 linecode=mtx601 length=2000 units=ft",
    "new line.632670 phases=3 bus1=632 bus2=670 linecode=mtx601 length=667 units=ft",
    "new line.670671 phases=3 bus1=670 bus2=671 linecode=mtx601 length=1333 units=ft",
    "new line.671680 phases=3 bus1=671 bus2=680 linecode=mtx601 length=1000 units=ft",
    "new line.632633 phases=3 bus1=632 bus2=633 linecode=mtx602 length=500 units=ft",
    "new line.633634 phases=3 bus1=633 bus2=634 linecode=mtx602 length=500 units=ft",
    "new line.671684 phases=3 bus1=671 bus2=684 linecode=mtx601 length=300 units=ft",
    "new line.684611 phases=1 bus1=684.3 bus2=611.3 linecode=mtx602 length=300 units=ft",
    "new line.684652 phases=1 bus1=684.1 bus2=652.1 linecode=mtx602 length=800 units=ft",
    "new line.692675 phases=3 bus1=692 bus2=675 linecode=mtx601 length=500 units=ft",
    "new line.671692 phases=3 bus1=671 bus2=692 linecode=mtx601 length=1 units=ft",
    # Loads (base kW/kvar — scaled each step by load_noise)
    "new load.671  phases=3 bus1=671     kv=4.16  kw=1155 kvar=660  model=1",
    "new load.634a phases=1 bus1=634.1   kv=0.277 kw=160  kvar=110  model=1",
    "new load.634b phases=1 bus1=634.2   kv=0.277 kw=120  kvar=90   model=1",
    "new load.634c phases=1 bus1=634.3   kv=0.277 kw=120  kvar=90   model=1",
    "new load.645  phases=1 bus1=645.2   kv=2.4   kw=170  kvar=125  model=1",
    "new load.646  phases=1 bus1=646.2   kv=2.4   kw=230  kvar=132  model=1",
    "new load.692  phases=3 bus1=692     kv=4.16  kw=170  kvar=151  model=5",
    "new load.675a phases=1 bus1=675.1   kv=2.4   kw=485  kvar=190  model=1",
    "new load.675b phases=1 bus1=675.2   kv=2.4   kw=68   kvar=60   model=1",
    "new load.675c phases=1 bus1=675.3   kv=2.4   kw=290  kvar=212  model=1",
    "new load.611  phases=1 bus1=611.3   kv=2.4   kw=170  kvar=80   model=5",
    "new load.652  phases=1 bus1=652.1   kv=2.4   kw=128  kvar=86   model=1",
    "new load.632a phases=1 bus1=632.1   kv=2.4   kw=17   kvar=10   model=1",
    "new load.632b phases=1 bus1=632.2   kv=2.4   kw=66   kvar=38   model=1",
    "new load.632c phases=1 bus1=632.3   kv=2.4   kw=117  kvar=68   model=1",
    "new load.670a phases=1 bus1=670.1   kv=2.4   kw=17   kvar=10   model=1",
    "new load.670b phases=1 bus1=670.2   kv=2.4   kw=66   kvar=38   model=1",
    "new load.670c phases=1 bus1=670.3   kv=2.4   kw=117  kvar=68   model=1",
    # Capacitors
    "new capacitor.cap1 phases=3 bus1=675    kv=4.16 kvar=600",
    "new capacitor.cap2 phases=1 bus1=611.3  kv=2.4  kvar=100",
    # Regulator (tap range set by environment — not automatic)
    "new regcontrol.vreg4_a transformer=sub winding=2 vreg=122 band=2 ptratio=20 ctprim=700 r=2.7 x=1.6",
    "set voltagebases=[115, 4.16, 0.48, 0.277]",
    "calcvoltagebases",
    "solve",
]

# Base load values (kW, kvar) per load element — for noise scaling
_BASE_LOADS = {
    "671":  (1155, 660),
    "634a": (160, 110), "634b": (120, 90),  "634c": (120, 90),
    "645":  (170, 125), "646":  (230, 132),
    "692":  (170, 151),
    "675a": (485, 190), "675b": (68,  60),  "675c": (290, 212),
    "611":  (170, 80),  "652":  (128, 86),
    "632a": (17,  10),  "632b": (66,  38),  "632c": (117, 68),
    "670a": (17,  10),  "670b": (66,  38),  "670c": (117, 68),
}

# Regulator tap: 0–32 → ratio 0.90–1.10
_TAP_MIN, _TAP_MAX, _N_TAPS = 0.90, 1.10, 33

# Voltage limits
_V_MIN, _V_MAX = 0.95, 1.05

# Reward weights (same as DistFlow env)
_ALPHA = 100.0   # voltage violation penalty
_BETA  = 1.0     # capacitor switching cost
_GAMMA = 0.1     # power loss penalty (per kW)


def _build_circuit() -> None:
    """Send all DSS commands to build the IEEE 13-bus circuit."""
    for cmd in _DSS_COMMANDS:
        dss.Text.Command(cmd)


class VVCEnvOpenDSS(gym.Env):
    """
    IEEE 13-bus Volt-VAR Control with real OpenDSS 3-phase AC power flow.

    Observation (93-dim):
        [0 :45]  per-phase voltage magnitudes in pu  (15 buses × 3 phases)
        [45:90]  per-phase active load normalised     (15 loads × 3 phases, padded)
        [90]     cap1 status (0/1)
        [91]     cap2 status (0/1)
        [92]     regulator tap normalised to [0, 1]

    Action — MultiDiscrete([2, 2, 33]):
        action[0]  cap1 ON/OFF
        action[1]  cap2 ON/OFF
        action[2]  regulator tap 0–32

    Reward:
        -α·Σ(V-Vlim)²  - β·Σ|cap_changes|  - γ·P_loss_kW
    """

    metadata = {"render_modes": []}

    # observation dim breakdown
    N_BUSES   = 15
    N_PHASES  = 3
    OBS_DIM   = N_BUSES * N_PHASES + N_BUSES * N_PHASES + 3   # 93

    def __init__(
        self,
        episode_len: int = 24,
        load_noise:  float = 0.10,
        seed:        int   = 0,
    ):
        super().__init__()
        self.episode_len  = episode_len
        self.load_noise   = load_noise
        self._rng         = np.random.default_rng(seed)
        self._step_count  = 0
        self._cap_status  = np.array([0, 0], dtype=np.int32)
        self._tap_idx     = 16   # start at midpoint

        # Build circuit once
        _build_circuit()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([2, 2, _N_TAPS])

    # ── private helpers ─────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> int:
        """Apply cap + tap action, return number of cap switches."""
        new_caps = action[:2].astype(np.int32)
        new_tap  = int(action[2])
        switches = int(np.sum(np.abs(new_caps - self._cap_status)))

        # Capacitor 1 (cap1 at bus 675)
        dss.Text.Command(f"edit capacitor.cap1 enabled={'yes' if new_caps[0] else 'no'}")
        # Capacitor 2 (cap2 at bus 611)
        dss.Text.Command(f"edit capacitor.cap2 enabled={'yes' if new_caps[1] else 'no'}")

        # Regulator tap: set transformer winding ratio
        tap_ratio = _TAP_MIN + new_tap * (_TAP_MAX - _TAP_MIN) / (_N_TAPS - 1)
        dss.Text.Command(f"edit transformer.sub wdg=2 tap={tap_ratio:.6f}")

        self._cap_status = new_caps
        self._tap_idx    = new_tap
        return switches

    def _randomise_loads(self) -> None:
        """Scale all loads by a random factor in [1-noise, 1+noise]."""
        for name, (kw_base, kvar_base) in _BASE_LOADS.items():
            scale = 1.0 + self._rng.uniform(-self.load_noise, self.load_noise)
            dss.Text.Command(
                f"edit load.{name} kw={kw_base * scale:.2f} kvar={kvar_base * scale:.2f}"
            )

    def _solve(self) -> None:
        dss.Text.Command("solve")

    def _get_voltages(self) -> np.ndarray:
        """Return per-phase voltage magnitudes in pu for all buses (N_BUSES × N_PHASES)."""
        vmag = np.array(dss.Circuit.AllBusMagPu(), dtype=np.float32)
        # AllBusMagPu returns flattened [bus0_ph1, bus0_ph2, bus0_ph3, bus1_ph1, ...]
        n = self.N_BUSES * self.N_PHASES
        if len(vmag) >= n:
            return vmag[:n]
        # pad if fewer phases on some buses
        out = np.ones(n, dtype=np.float32)
        out[:len(vmag)] = vmag
        return out

    def _get_loads_norm(self) -> np.ndarray:
        """Return normalised active loads, padded to N_BUSES × N_PHASES."""
        total_max = sum(kw for kw, _ in _BASE_LOADS.values()) * (1 + self.load_noise)
        loads = []
        dss.Loads.First()
        while True:
            kw = dss.Loads.kW()
            n_ph = dss.Loads.Phases()
            loads.extend([kw / total_max] * n_ph)
            if not dss.Loads.Next():
                break
        out = np.zeros(self.N_BUSES * self.N_PHASES, dtype=np.float32)
        n = min(len(loads), len(out))
        out[:n] = loads[:n]
        return out

    def _get_obs(self) -> np.ndarray:
        v    = self._get_voltages()
        p    = self._get_loads_norm()
        caps = self._cap_status.astype(np.float32)
        tap  = np.array([self._tap_idx / (_N_TAPS - 1)], dtype=np.float32)
        return np.concatenate([v, p, caps, tap])

    def _compute_reward(self, switches: int) -> tuple[float, int, float]:
        vmag  = self._get_voltages()
        # Voltage violations
        vviol_sq = np.sum(
            np.maximum(vmag - _V_MAX, 0) ** 2 +
            np.maximum(_V_MIN - vmag, 0) ** 2
        )
        n_vviol = int(np.sum((vmag < _V_MIN) | (vmag > _V_MAX)))
        # Power losses (kW)
        p_loss = dss.Circuit.Losses()[0] / 1000.0
        reward = (
            - _ALPHA * float(vviol_sq)
            - _BETA  * float(switches)
            - _GAMMA * float(p_loss)
        )
        return reward, n_vviol, p_loss

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _build_circuit()
        self._step_count = 0
        self._cap_status = np.array([0, 0], dtype=np.int32)
        self._tap_idx    = 16
        self._randomise_loads()
        self._solve()
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        switches = self._apply_action(action)
        self._randomise_loads()
        self._solve()

        obs               = self._get_obs()
        reward, n_vviol, p_loss = self._compute_reward(switches)
        self._step_count += 1
        done              = self._step_count >= self.episode_len

        info = {
            "v_viol":  n_vviol,
            "P_loss":  p_loss,
            "voltage": self._get_voltages(),
        }
        return obs, reward, done, False, info
