from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from power_system.rl_env import IEEE30SACOTSoneStepEnv, IEEE30StochasticOTSControlEnv


class RLSetupTest(unittest.TestCase):
    def test_one_step_environment_uses_fixed_candidate_set(self):
        env = IEEE30SACOTSoneStepEnv(candidate_line_indices=(9,))
        obs, info = env.reset()
        self.assertEqual(obs.shape[0], 4)
        next_obs, reward, terminated, truncated, info = env.step([0.0])
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn("solver_result", info)
        self.assertEqual(info["solver_result"]["candidate_line_indices"], (9,))
        self.assertEqual(info["solver_result"]["forced_line_status"], (0,))
        self.assertIn(9, info["solver_result"]["switched_off_line_indices"])
        env.close()

    def test_stochastic_environment_runs_multi_step_episode(self):
        env = IEEE30StochasticOTSControlEnv(candidate_line_indices=(9, 28), episode_length=3)
        obs, info = env.reset(seed=7)
        self.assertEqual(obs.shape[0], 9)
        self.assertEqual(len(info["load_scale_path"]), 3)

        next_obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("solver_result", info)
        self.assertEqual(info["solver_result"]["candidate_line_indices"], (9, 28))
        self.assertEqual(info["solver_result"]["forced_line_status"], (1, 0))

        _, _, terminated, truncated, _ = env.step([1.0, 1.0])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        terminal_obs, _, terminated, truncated, info = env.step([0.0, 1.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(terminal_obs.shape, obs.shape)
        env.close()


if __name__ == "__main__":
    unittest.main()
