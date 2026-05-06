"""
VVCEnvOpenDSS — IEEE 13-bus Volt-VAR Control using the real PowerGym DSS file.

Physics: exact 3-phase AC unbalanced power flow via opendssdirect (same engine
         as PowerGym / dss_python).

Circuit : IEEE13Nodeckt.dss  (Siemens PowerGym dataset)
Devices : 2 capacitors (Cap1 @ bus 675, Cap2 @ bus 611)
          3 voltage regulators (Reg1, Reg2, Reg3 — each independent single-phase)
          1 battery (batt1 @ bus 680, 200 kW, 1000 kWh)

Action  — MultiDiscrete([2, 2, 33, 33, 33, 33]):
    action[0]  Cap1  ON/OFF
    action[1]  Cap2  ON/OFF
    action[2]  Reg1  tap 0–32
    action[3]  Reg2  tap 0–32
    action[4]  Reg3  tap 0–32
    action[5]  batt1 discharge level 0–32

Observation: voltages (per-phase) | cap_statuses | reg_tap_indices | [soc, dis_ratio] per bat

Reward : −power_w·(loss/gen) − cap_w·Σ|cap_changes| − reg_w·Σ|reg_changes|
         − dis_w·dis_ratio + v_reward (linear violation penalty)
"""

from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import opendssdirect as dss
from .load_profile import LoadProfileSampler

# ── DSS file ─────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DSS_FILE = os.path.join(_HERE, "powergym_systems", "13Bus", "IEEE13Nodeckt.dss")

_CAP_NAMES     = ["Cap1", "Cap2"]
_N_CAPS        = len(_CAP_NAMES)

_REG_NAMES     = ["Reg1", "Reg2", "Reg3"]
_N_REGS        = len(_REG_NAMES)
_REGCTRL_NAMES = ["Reg1", "Reg2", "Reg3"]

# Battery: 1 battery at bus 680, 200 kW, 1000 kWh
_BAT_NAMES   = ["batt1"]
_BAT_MAX_KW  = [200.0]
_BAT_MAX_KWH = [1000.0]
_N_BATS      = len(_BAT_NAMES)
_BAT_ACT_NUM = 33
_BAT_MODE_NUM = _BAT_ACT_NUM // 2   # 16

_V_MIN, _V_MAX = 0.95, 1.05
# PowerGym reward weights for 13-bus
_POWER_W = 10.0
_V_W     = 100.0
_CAP_W   = 1.0 / 33.0
_REG_W   = 1.0 / 33.0
_DIS_W   = 6.0 / 33.0

_TAP_MIN, _TAP_MAX, _N_TAPS = 0.90, 1.10, 33
_DURATION_H = 1.0   # 1-hour time steps


def _build_circuit() -> None:
    dss.Text.Command(f"compile [{_DSS_FILE}]")
    for name in _REGCTRL_NAMES:
        dss.Text.Command(f"regcontrol.{name}.enabled=no")
    dss.Text.Command("solve")


def _get_load_bases() -> dict[str, tuple[float, float]]:
    bases = {}
    if not dss.Loads.First():
        return bases
    while True:
        bases[dss.Loads.Name()] = (dss.Loads.kW(), dss.Loads.kvar())
        if not dss.Loads.Next():
            break
    return bases


def _bat_avail_kw(max_kw: float) -> list[float]:
    """Discrete kW levels from -max_kw to +max_kw in bat_act_num steps."""
    diff = max_kw / _BAT_MODE_NUM
    return [n * diff for n in range(-_BAT_MODE_NUM, _BAT_MODE_NUM + 1)]


def _bat_state_project(state: int, avail_kw: list[float], kwh: float, max_kwh: float) -> float:
    """Clamp discharge state so SOC constraints are respected."""
    mid = len(avail_kw) // 2
    state = max(0, min(len(avail_kw) - 1, state))
    if state > mid:   # discharging
        allowed = kwh / _DURATION_H
        diff = avail_kw[1] - avail_kw[0]
        if avail_kw[state] > allowed:
            state = int(state - np.ceil((avail_kw[state] - allowed) / diff - 1e-8))
    elif state < mid:  # charging
        allowed = (kwh - max_kwh) / _DURATION_H
        diff = avail_kw[1] - avail_kw[0]
        if avail_kw[state] < allowed:
            state = int(state + np.ceil((allowed - avail_kw[state]) / diff - 1e-8))
    return avail_kw[state], state



class VVCEnvOpenDSS(gym.Env):
    """
    IEEE 13-bus Volt-VAR Control — real PowerGym DSS circuit, OpenDSS power flow.

    Action  : MultiDiscrete([2, 2, 33, 33, 33, 33])
    Obs dim : _n_volt_phases + N_CAPS + N_REGS + N_BATS*2
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
        self._cap_status = np.zeros(_N_CAPS, dtype=np.int32)
        self._tap_idx    = np.full(_N_REGS, 16, dtype=np.int32)
        self._profile    = np.ones(episode_len, dtype=np.float32)

        # Battery state: kwh and current discharge state index
        self._bat_kwh    = np.array(_BAT_MAX_KWH, dtype=np.float64)
        self._bat_state  = np.full(_N_BATS, _BAT_MODE_NUM, dtype=np.int32)  # neutral
        self._bat_avail  = [_bat_avail_kw(mkw) for mkw in _BAT_MAX_KW]

        _build_circuit()
        self._n_volt_phases = len(dss.Circuit.AllBusMagPu())
        self._load_bases    = _get_load_bases()
        self._load_sampler  = LoadProfileSampler("13Bus")

        obs_dim = self._n_volt_phases + _N_CAPS + _N_REGS + _N_BATS * 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [2] * _N_CAPS + [_N_TAPS] * _N_REGS + [_BAT_ACT_NUM] * _N_BATS
        )

    def _apply_action(self, action: np.ndarray) -> tuple[int, int, float]:
        new_caps     = action[:_N_CAPS].astype(np.int32)
        new_taps     = action[_N_CAPS:_N_CAPS + _N_REGS].astype(np.int32)
        new_bat_acts = action[_N_CAPS + _N_REGS:].astype(np.int32)

        cap_switches = int(np.sum(np.abs(new_caps - self._cap_status)))
        reg_switches = int(np.sum(np.abs(new_taps - self._tap_idx)))

        for name, val in zip(_CAP_NAMES, new_caps):
            dss.Text.Command(f"capacitor.{name}.enabled={'yes' if val else 'no'}")

        for name, tap in zip(_REG_NAMES, new_taps):
            tap_ratio = _TAP_MIN + tap * (_TAP_MAX - _TAP_MIN) / (_N_TAPS - 1)
            dss.Text.Command(f"edit transformer.{name} wdg=2 tap={tap_ratio:.6f}")

        # Apply battery actions
        total_dis_ratio = 0.0
        for i, (bat_name, avail_kw, max_kw, max_kwh) in enumerate(
            zip(_BAT_NAMES, self._bat_avail, _BAT_MAX_KW, _BAT_MAX_KWH)
        ):
            kw, clipped_state = _bat_state_project(
                int(new_bat_acts[i]), avail_kw, float(self._bat_kwh[i]), max_kwh
            )
            self._bat_state[i] = clipped_state
            dss.Text.Command(f"edit generator.{bat_name} kw={kw:.3f} kvar={kw / 0.95:.3f}")
            total_dis_ratio += max(0.0, kw) / max_kw

        self._cap_status = new_caps
        self._tap_idx    = new_taps
        return cap_switches, reg_switches, total_dis_ratio

    def _update_bat_soc(self) -> None:
        """Update battery SOC using the set kW for the current timestep."""
        for i in range(_N_BATS):
            kw_set = self._bat_avail[i][self._bat_state[i]]   # + = discharging
            self._bat_kwh[i] -= kw_set * _DURATION_H
            self._bat_kwh[i]  = float(np.clip(self._bat_kwh[i], 0.0, _BAT_MAX_KWH[i]))

    def _apply_loads(self, multiplier: float) -> None:
        for name, (kw_base, kvar_base) in self._load_bases.items():
            dss.Text.Command(
                f"edit load.{name} kw={kw_base * multiplier:.3f} kvar={kvar_base * multiplier:.3f}"
            )

    def _get_voltages(self) -> np.ndarray:
        return np.array(dss.Circuit.AllBusMagPu(), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        v    = self._get_voltages()
        caps = self._cap_status.astype(np.float32)
        taps = self._tap_idx.astype(np.float32)          # raw 0-32
        bat_obs = []
        for i in range(_N_BATS):
            soc = float(self._bat_kwh[i]) / _BAT_MAX_KWH[i]
            dis_ratio = max(0.0, self._bat_avail[i][self._bat_state[i]]) / _BAT_MAX_KW[i]
            bat_obs.extend([soc, dis_ratio])
        return np.concatenate([v, caps, taps, np.array(bat_obs, dtype=np.float32)])

    def _compute_reward(
        self, cap_switches: int, reg_switches: int, dis_ratio: float
    ) -> tuple[float, int, float]:
        vmag    = self._get_voltages()
        n_vviol = int(np.sum((vmag < _V_MIN) | (vmag > _V_MAX)))

        v_reward = _V_W * float(np.sum(
            np.minimum(0.0, _V_MAX - vmag) + np.minimum(0.0, vmag - _V_MIN)
        ))

        loss_kw = dss.Circuit.Losses()[0] / 1000.0
        gen_kw  = abs(dss.Circuit.TotalPower()[0])
        p_reward = -(loss_kw / gen_kw) * _POWER_W if gen_kw > 0 else 0.0

        t_reward = -_CAP_W * cap_switches - _REG_W * reg_switches - _DIS_W * dis_ratio

        reward = p_reward + v_reward + t_reward
        return reward, n_vviol, loss_kw

    def reset(
        self,
        seed:    int | None  = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _build_circuit()
        self._step_count = 0
        self._cap_status = np.zeros(_N_CAPS, dtype=np.int32)
        self._tap_idx    = np.full(_N_REGS, 16, dtype=np.int32)
        self._bat_kwh    = np.array(_BAT_MAX_KWH, dtype=np.float64)
        self._bat_state  = np.full(_N_BATS, _BAT_MODE_NUM, dtype=np.int32)
        self._profile    = self._load_sampler.sample(self._rng)
        self._apply_loads(self._profile[0])
        dss.Text.Command("solve")
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        cap_sw, reg_sw, dis_ratio = self._apply_action(action)
        self._apply_loads(self._profile[self._step_count])
        dss.Text.Command("solve")
        self._update_bat_soc()

        obs                     = self._get_obs()
        reward, n_vviol, p_loss = self._compute_reward(cap_sw, reg_sw, dis_ratio)
        self._step_count       += 1
        done                    = self._step_count >= self.episode_len

        return obs, reward, done, False, {
            "v_viol": n_vviol,
            "P_loss": p_loss,
            "voltage": self._get_voltages(),
        }
