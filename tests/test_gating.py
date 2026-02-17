from __future__ import annotations

import unittest

from exp.gating import gate_stage
from exp.models import ComparisonReport


class GatingTests(unittest.TestCase):
    def test_stage1_promotes_top_three(self) -> None:
        reports = []
        for idx, score in enumerate([6.0, 5.5, 4.0, 3.4, 2.9]):
            reports.append(
                ComparisonReport(
                    candidate_run_ids=[f"run-{idx}"],
                    baseline_run_ids=["base"],
                    delta_metrics={"composite": score},
                    significance_tests={"ci95_excludes_zero": True},
                    pass_fail={"equal_cost_pass": True, "stage_gate_pass": score >= 3.0},
                    candidate_stage=1,
                    track_id="T1",
                )
            )
        summary = gate_stage(1, reports)
        self.assertEqual(summary["promoted_count"], 3)
        self.assertEqual(summary["promoted_run_ids"], ["run-0", "run-1", "run-2"])

    def test_stage4_all_qualified_promoted(self) -> None:
        reports = [
            ComparisonReport(
                candidate_run_ids=["run-a"],
                baseline_run_ids=["base-a"],
                delta_metrics={"composite": 8.4},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"equal_cost_pass": True, "stage_gate_pass": True},
                candidate_stage=4,
                track_id="T3",
            ),
            ComparisonReport(
                candidate_run_ids=["run-b"],
                baseline_run_ids=["base-b"],
                delta_metrics={"composite": 8.1},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"equal_cost_pass": True, "stage_gate_pass": True},
                candidate_stage=4,
                track_id="T3",
            ),
        ]
        summary = gate_stage(4, reports)
        self.assertEqual(summary["promoted_count"], 2)


if __name__ == "__main__":
    unittest.main()
