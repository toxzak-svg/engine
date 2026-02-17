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
        self.assertIn("| Rank | Track | Decision Score | Weighted Anchor Score |", memo)
        self.assertIn("Decision score formula", memo)
        self.assertIn("T3", memo)
        self.assertIn("Advance priority order (pass-adjusted)", memo)

    def test_recovery_rows_are_supplemental_when_core_present(self) -> None:
        reports = [
            ComparisonReport(
                candidate_run_ids=["batch-s2-t1-e2-201-x1"],
                baseline_run_ids=["b1"],
                delta_metrics={"composite": 5.6},
                anchor_delta_metrics={"composite": 5.6},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=2,
                track_id="T1",
            ),
            ComparisonReport(
                candidate_run_ids=["recovery-s2-t4-sweep-201-x2"],
                baseline_run_ids=["b2"],
                delta_metrics={"composite": 6.7},
                anchor_delta_metrics={"composite": 6.7},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=2,
                track_id="T4",
            ),
        ]
        memo = build_consolidated_memo(reports, marker_iso_utc="2026-01-01T00:00:00+00:00")
        self.assertIn("Candidate comparisons (core): 1", memo)
        self.assertIn("Candidate comparisons (recovery supplemental): 1", memo)
        self.assertIn("Recovery snapshot (supplemental; excluded from core ranking stats):", memo)


if __name__ == "__main__":
    unittest.main()
