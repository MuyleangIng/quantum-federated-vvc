from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from power_system.dc_ots import (
    compare_dc_opf_vs_dc_ots_ieee30,
    compare_dc_ots_ieee30_solvers,
    solve_dc_opf_ieee30,
    solve_dc_opf_ieee30_gurobi,
    solve_dc_ots_ieee30,
    solve_dc_ots_ieee30_gurobi,
)


class DCOtsIEEE30Test(unittest.TestCase):
    def test_dc_ots_solves_with_default_candidate(self):
        result = solve_dc_ots_ieee30()
        self.assertTrue(result.success)
        self.assertEqual(result.solver_name, "scipy_highs")
        self.assertEqual(result.candidate_line_indices, (9,))
        self.assertIsNone(result.forced_line_status)
        self.assertAlmostEqual(result.load_scale, 1.0, places=6)
        self.assertEqual(len(result.line_flows_mw), 41)
        self.assertEqual(len(result.line_status), 41)
        self.assertEqual(len(result.generator_dispatch_mw), 6)
        self.assertAlmostEqual(result.total_generation_mw, result.total_load_mw, places=6)
        self.assertLessEqual(result.max_flow_violation_mw, 1e-6)

    def test_dc_ots_accepts_multiple_candidate_lines(self):
        result = solve_dc_ots_ieee30(candidate_line_indices=(9, 28))
        self.assertTrue(result.success)
        self.assertEqual(result.candidate_line_indices, (9, 28))
        self.assertLessEqual(result.max_flow_violation_mw, 1e-6)
        self.assertTrue(set(result.switched_off_line_indices).issubset({9, 28}))

    def test_dc_ots_respects_forced_line_status(self):
        result = solve_dc_ots_ieee30(candidate_line_indices=(9,), forced_line_status=(0,))
        self.assertTrue(result.success)
        self.assertEqual(result.candidate_line_indices, (9,))
        self.assertEqual(result.forced_line_status, (0,))
        self.assertIn(9, result.switched_off_line_indices)
        self.assertEqual(result.line_status[9], 0)
        self.assertLessEqual(result.max_flow_violation_mw, 1e-6)

    def test_dc_opf_load_scale_changes_total_load(self):
        base = solve_dc_opf_ieee30()
        stressed = solve_dc_opf_ieee30(load_scale=1.1)
        self.assertTrue(base.success)
        self.assertTrue(stressed.success)
        self.assertAlmostEqual(stressed.total_load_mw, base.total_load_mw * 1.1, places=6)

    def test_gurobi_dc_ots_solves_with_default_candidate(self):
        result = solve_dc_ots_ieee30_gurobi()
        self.assertTrue(result.success)
        self.assertEqual(result.solver_name, "gurobi")
        self.assertEqual(result.candidate_line_indices, (9,))
        self.assertEqual(len(result.line_flows_mw), 41)
        self.assertEqual(len(result.generator_dispatch_mw), 6)
        self.assertAlmostEqual(result.total_generation_mw, result.total_load_mw, places=6)
        self.assertLessEqual(result.max_flow_violation_mw, 1e-6)

    def test_dc_opf_baseline_solves(self):
        result = solve_dc_opf_ieee30()
        self.assertTrue(result.success)
        self.assertEqual(result.candidate_line_indices, ())
        self.assertEqual(result.switched_off_line_indices, ())
        self.assertLessEqual(result.max_flow_violation_mw, 1e-6)

    def test_gurobi_dc_opf_baseline_solves(self):
        result = solve_dc_opf_ieee30_gurobi()
        self.assertTrue(result.success)
        self.assertEqual(result.candidate_line_indices, ())
        self.assertEqual(result.switched_off_line_indices, ())
        self.assertLessEqual(result.max_flow_violation_mw, 1e-6)

    def test_solver_comparison_is_consistent(self):
        comparison = compare_dc_ots_ieee30_solvers(candidate_line_indices=(9, 28), load_scale=1.05)
        self.assertTrue(comparison.scipy_result.success)
        self.assertTrue(comparison.gurobi_result.success)
        self.assertLessEqual(comparison.objective_gap, 1e-6)
        self.assertLessEqual(comparison.dispatch_l1_gap_mw, 1e-4)
        self.assertTrue(comparison.switched_off_match)

    def test_dc_opf_vs_dc_ots_comparison_runs(self):
        comparison = compare_dc_opf_vs_dc_ots_ieee30(candidate_line_indices=(9, 28), load_scale=1.1)
        self.assertTrue(comparison.dc_opf_result.success)
        self.assertTrue(comparison.dc_ots_result.success)
        self.assertGreaterEqual(comparison.objective_delta, -1e-6)
        self.assertGreaterEqual(comparison.dispatch_l1_gap_mw, 0.0)


if __name__ == "__main__":
    unittest.main()
