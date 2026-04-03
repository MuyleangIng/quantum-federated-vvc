"""
VVCEnvOpenDSS — IEEE 13-bus Volt-VAR Control using the real PowerGym DSS file.

Physics: exact 3-phase AC unbalanced power flow via opendssdirect (same engine
         as PowerGym / dss_python).

Circuit : IEEE13Nodeckt.dss  (Siemens PowerGym dataset)
Devices : 2 capacitors (Cap1 @ bus 675, Cap2 @ bus 611)
          1 voltage regulator (Reg1, phase-A tap, 0.90–1.10 in 33 steps)

Action  — MultiDiscrete([2, 2, 33]):
    action[0]  Cap1 ON/OFF
    action[1]  Cap2 ON/OFF
    action[2]  Reg1 tap 0–32

Observation (dynamic dim = N_BUS_PHASES*2 + 3):
    per-phase voltage magnitudes (pu)  — all buses × phases
    per-phase active load (normalised) — all load buses × phases, padded
    Cap1 status, Cap2 status, Reg1 tap (normalised)

Reward : −α·Σ(V−Vlim)² − β·Σ|cap_changes| − γ·P_loss_kW
"""

from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import opendssdirect as dss

# ── DSS file ─────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_DSS_FILE = os.path.join(_HERE, "powergym_systems", "13Bus", "IEEE13Nodeckt.dss")

# Capacitor names in the circuit (exactly as defined in the DSS file)
_CAP_NAMES = ["Cap1", "Cap2"]

# Regulator transformer names (1-phase each); we control Reg1 (phase A)
# Reg2, Reg3 are set to fixed nominal tap
_REG_NAME  = "Reg1"           # controlled regulator
_REG_FIXED = ["Reg2", "Reg3"] # fixed at nominal

# All regcontrol names to disable (let RL agent control taps instead)
_REGCTRL_NAMES = ["Reg1", "Reg2", "Reg3"]

# Voltage limits and reward weights
_V_MIN, _V_MAX = 0.95, 1.05
_ALPHA = 100.0
_BETA  = 1.0
_GAMMA = 0.1

_TAP_MIN, _TAP_MAX, _N_TAPS = 0.90, 1.10, 33


# ── Circuit helpers ──────────────────────────────────────────────────────────

def _build_circuit() -> None:
    """Compile the real IEEE 13-bus DSS file and disable automatic regcontrols."""
    dss.Text.Command(f"compile [{_DSS_FILE}]")
    # Disable automatic voltage regulators — RL agent controls taps directly
    for name in _REGCTRL_NAMES:
        dss.Text.Command(f"regcontrol.{name}.enabled=no")
    # Fix Reg2 and Reg3 at nominal tap; only Reg1 is RL-controlled
    for name in _REG_FIXED:
        dss.Text.Command(f"edit transformer.{name} wdg=2 tap=1.0")
    dss.Text.Command("solve")


def _get_load_bases() -> dict[str, tuple[float, float]]:
    """Return {load_name: (base_kw, base_kvar)} after circuit is compiled."""
    bases = {}
    if not dss.Loads.First():
        return bases
    while True:
        bases[dss.Loads.Name()] = (dss.Loads.kW(), dss.Loads.kvar())
        if not dss.Loads.Next():
            break
    return bases


def _count_bus_phases() -> int:
    """Return total number of phase-voltage values from AllBusMagPu()."""
    return len(dss.Circuit.AllBusMagPu())


# ── Environment ──────────────────────────────────────────────────────────────

class VVCEnvOpenDSS(gym.Env):
    """
    IEEE 13-bus Volt-VAR Control — real PowerGym DSS circuit, OpenDSS power flow.

    Action  : MultiDiscrete([2, 2, 33])  →  132 joint actions
    Obs dim : determined at runtime from the compiled circuit
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_len: int   = 24,
        load_noise:  float = 0.10,
        seed:        int   = 0,
    ):
        super().__init__()
        self.episode_len = episode_len
        self.load_noise  = load_noise
        self._rng        = np.random.default_rng(seed)
        self._step_count = 0
        self._cap_status = np.array([0, 0], dtype=np.int32)
        self._tap_idx    = 16   # midpoint of 0–32 range

        # Compile once to determine obs dim and base loads
        _build_circuit()
        self._n_volt_phases = _count_bus_phases()
        self._load_bases    = _get_load_bases()

        obs_dim = self._n_volt_phases * 2 + 3   # voltages + loads + device states

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([2, 2, _N_TAPS])

    # ── private helpers ──────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> int:
        new_caps = action[:2].astype(np.int32)
        new_tap  = int(action[2])
        switches = int(np.sum(np.abs(new_caps - self._cap_status)))

        for name, val in zip(_CAP_NAMES, new_caps):
            dss.Text.Command(f"capacitor.{name}.enabled={'yes' if val else 'no'}")

        tap_ratio = _TAP_MIN + new_tap * (_TAP_MAX - _TAP_MIN) / (_N_TAPS - 1)
        dss.Text.Command(f"edit transformer.{_REG_NAME} wdg=2 tap={tap_ratio:.6f}")

        self._cap_status = new_caps
        self._tap_idx    = new_tap
        return switches

    def _randomise_loads(self) -> None:
        for name, (kw_base, kvar_base) in self._load_bases.items():
            scale = 1.0 + self._rng.uniform(-self.load_noise, self.load_noise)
            dss.Text.Command(
                f"edit load.{name} kw={kw_base * scale:.3f} kvar={kvar_base * scale:.3f}"
            )

    def _get_voltages(self) -> np.ndarray:
        return np.array(dss.Circuit.AllBusMagPu(), dtype=np.float32)

    def _get_loads_norm(self) -> np.ndarray:
        total_max = sum(kw for kw, _ in self._load_bases.values()) * (1 + self.load_noise)
        loads = []
        if dss.Loads.First():
            while True:
                loads.extend([dss.Loads.kW() / total_max] * dss.Loads.Phases())
                if not dss.Loads.Next():
                    break
        out = np.zeros(self._n_volt_phases, dtype=np.float32)
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
        vmag     = self._get_voltages()
        vviol_sq = np.sum(
            np.maximum(vmag - _V_MAX, 0) ** 2 +
            np.maximum(_V_MIN - vmag, 0) ** 2
        )
        n_vviol = int(np.sum((vmag < _V_MIN) | (vmag > _V_MAX)))
        p_loss  = dss.Circuit.Losses()[0] / 1000.0   # W → kW
        reward  = (
            - _ALPHA * float(vviol_sq)
            - _BETA  * float(switches)
            - _GAMMA * float(p_loss)
        )
        return reward, n_vviol, p_loss

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    int | None  = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _build_circuit()
        self._step_count = 0
        self._cap_status = np.array([0, 0], dtype=np.int32)
        self._tap_idx    = 16
        self._randomise_loads()
        dss.Text.Command("solve")
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        switches = self._apply_action(action)
        self._randomise_loads()
        dss.Text.Command("solve")

        obs               = self._get_obs()
        reward, n_vviol, p_loss = self._compute_reward(switches)
        self._step_count += 1
        done              = self._step_count >= self.episode_len

        return obs, reward, done, False, {
            "v_viol": n_vviol,
            "P_loss": p_loss,
            "voltage": self._get_voltages(),
        }
