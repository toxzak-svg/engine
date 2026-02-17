from __future__ import annotations

import unittest

from exp.models import ComparisonReport
from exp.reporting import build_consolidated_memo


class ReportingTests(unittest.TestCase):
    def test_build_consolidated_memo_contains_final_table(self) -> None:
        reports = [
            ComparisonReport(
                candidate_run_ids=["s1-a"],
                baseline_run_ids=["b1"],
                delta_metrics={"composite": 4.0},
                anchor_delta_metrics={"composite": 5.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"equal_cost_pass": True, "stage_gate_pass": True, "overall_pass": True},
                candidate_stage=1,
                track_id="T3",
            ),
            ComparisonReport(
                candidate_run_ids=["s2-a"],
                baseline_run_ids=["b2"],
                delta_metrics={"composite": 6.0},
                anchor_delta_metrics={"composite": 7.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"equal_cost_pass": True, "stage_gate_pass": True, "overall_pass": True},
                candidate_stage=2,
                track_id="T3",
            ),
            ComparisonReport(
                candidate_run_ids=["s3-a"],
                baseline_run_ids=["b3"],
                delta_metrics={"composite": 8.5},
                anchor_delta_metrics={"composite": 9.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"equal_cost_pass": True, "stage_gate_pass": True, "overall_pass": True},
                candidate_stage=3,
                track_id="T3",
            ),
        ]
        memo = build_consolidated_memo(reports, marker_iso_utc="2026-01-01T00:00:00+00:00")
        self.assertIn("Consolidated Final Ranking Memo", memo)
        self.assertIn("| Rank | Track | Weighted Anchor Score |", memo)
        self.assertIn("T3", memo)


if __name__ == "__main__":
    unittest.main()
