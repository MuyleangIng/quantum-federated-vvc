from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from power_system.ieee_cases import (
    get_ieee14_line_loading_report,
    get_ieee30_line_loading_report,
    load_ieee14_case,
    load_ieee30_case,
    run_ieee14_power_flow_summary,
    run_ieee30_power_flow_summary,
    validate_ieee_cases,
    validate_summary,
)


class IEEE14SetupTest(unittest.TestCase):
    def test_network_metadata_matches_reference_case(self):
        net = load_ieee14_case()
        self.assertEqual(len(net.bus), 14)
        self.assertEqual(len(net.line), 15)
        self.assertEqual(len(net.trafo), 5)

    def test_power_flow_converges(self):
        _, summary = run_ieee14_power_flow_summary()
        self.assertTrue(summary.converged)
        self.assertGreater(summary.total_load_mw, 0.0)
        self.assertGreater(summary.total_generation_mw, 0.0)
        self.assertGreaterEqual(summary.min_bus_vm_pu, 0.9)
        self.assertLessEqual(summary.max_bus_vm_pu, 1.1)

    def test_validation_passes_for_default_limits(self):
        _, summary = run_ieee14_power_flow_summary()
        validation = validate_summary(summary)
        self.assertTrue(validation.passed)
        self.assertGreaterEqual(validation.network_losses_mw, 0.0)
        self.assertEqual(
            validation.notes,
            ("Validation passed within the configured operating limits.",),
        )

    def test_line_loading_report_has_no_overloads(self):
        summary, records = get_ieee14_line_loading_report()
        self.assertEqual(summary.case_name, "IEEE 14-bus")
        self.assertEqual(len(records), 15)
        self.assertFalse(any(record.overloaded for record in records))


class IEEE30SetupTest(unittest.TestCase):
    def test_network_metadata_matches_reference_case(self):
        net = load_ieee30_case()
        self.assertEqual(len(net.bus), 30)
        self.assertEqual(len(net.line), 41)
        self.assertEqual(len(net.trafo), 0)

    def test_power_flow_converges(self):
        _, summary = run_ieee30_power_flow_summary()
        self.assertTrue(summary.converged)
        self.assertGreater(summary.total_load_mw, 0.0)
        self.assertGreater(summary.total_generation_mw, 0.0)
        self.assertGreaterEqual(summary.min_bus_vm_pu, 0.9)
        self.assertLessEqual(summary.max_bus_vm_pu, 1.1)

    def test_validation_flags_line_loading_issue(self):
        _, summary = run_ieee30_power_flow_summary()
        validation = validate_summary(summary)
        self.assertFalse(validation.passed)
        self.assertTrue(validation.converged)
        self.assertFalse(validation.line_loading_ok)
        self.assertGreaterEqual(validation.network_losses_mw, 0.0)
        self.assertIn("exceeds the limit", validation.notes[0])

    def test_line_loading_report_identifies_overloaded_branch(self):
        summary, records = get_ieee30_line_loading_report()
        overloaded = [record for record in records if record.overloaded]
        self.assertEqual(summary.case_name, "IEEE 30-bus")
        self.assertEqual(len(records), 41)
        self.assertEqual(len(overloaded), 1)
        self.assertEqual(overloaded[0].from_bus, 5)
        self.assertEqual(overloaded[0].to_bus, 7)
        self.assertGreater(overloaded[0].loading_percent, 100.0)


class IEEEValidationBatchTest(unittest.TestCase):
    def test_validate_ieee_cases_returns_both_cases(self):
        summaries, validations = validate_ieee_cases()
        self.assertEqual(
            [summary.case_name for summary in summaries],
            ["IEEE 14-bus", "IEEE 30-bus"],
        )
        self.assertEqual(
            [validation.case_name for validation in validations],
            ["IEEE 14-bus", "IEEE 30-bus"],
        )


if __name__ == "__main__":
    unittest.main()
