from __future__ import annotations

import unittest

from exp.gating import gate_stage
from exp.models import ComparisonReport


class GatingTests(unittest.TestCase):
    def test_stage1_promotes_top_three_tracks_by_pass_adjusted_signal(self) -> None:
        reports = [
            ComparisonReport(
                candidate_run_ids=["run-t1"],
                baseline_run_ids=["base-t1"],
                delta_metrics={"composite": 6.1},
                anchor_delta_metrics={"composite": 6.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=1,
                track_id="T1",
            ),
            ComparisonReport(
                candidate_run_ids=["run-t2"],
                baseline_run_ids=["base-t2"],
                delta_metrics={"composite": 5.6},
                anchor_delta_metrics={"composite": 5.5},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=1,
                track_id="T2",
            ),
            ComparisonReport(
                candidate_run_ids=["run-t3"],
                baseline_run_ids=["base-t3"],
                delta_metrics={"composite": 4.1},
                anchor_delta_metrics={"composite": 4.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=1,
                track_id="T3",
            ),
            ComparisonReport(
                candidate_run_ids=["run-t4"],
                baseline_run_ids=["base-t4"],
                delta_metrics={"composite": 8.0},
                anchor_delta_metrics={"composite": 8.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": False},
                candidate_stage=1,
                track_id="T4",
            ),
        ]
        summary = gate_stage(1, reports)
        self.assertEqual(summary["promoted_track_count"], 3)
        self.assertEqual(summary["promoted_tracks"], ["T1", "T2", "T3"])
        self.assertEqual(summary["promoted_run_ids"], ["run-t1", "run-t2", "run-t3"])

    def test_stage_gating_excludes_recovery_when_core_exists(self) -> None:
        reports = [
            ComparisonReport(
                candidate_run_ids=["core-run"],
                baseline_run_ids=["core-base"],
                delta_metrics={"composite": 6.0},
                anchor_delta_metrics={"composite": 6.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=2,
                track_id="T1",
            ),
            ComparisonReport(
                candidate_run_ids=["recovery-s2-t4-fix"],
                baseline_run_ids=["rec-base"],
                delta_metrics={"composite": 7.0},
                anchor_delta_metrics={"composite": 7.0},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=2,
                track_id="T4",
            ),
        ]
        summary = gate_stage(1, reports)
        self.assertEqual(summary["candidate_count_all"], 0)

        summary = gate_stage(2, reports)
        self.assertEqual(summary["candidate_count_all"], 2)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["candidate_count_recovery"], 1)
        self.assertEqual(summary["promoted_tracks"], ["T1"])

    def test_stage4_promotes_none(self) -> None:
        reports = [
            ComparisonReport(
                candidate_run_ids=["run-a"],
                baseline_run_ids=["base-a"],
                delta_metrics={"composite": 8.4},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=4,
                track_id="T3",
            ),
            ComparisonReport(
                candidate_run_ids=["run-b"],
                baseline_run_ids=["base-b"],
                delta_metrics={"composite": 8.1},
                significance_tests={"ci95_excludes_zero": True},
                pass_fail={"overall_pass": True},
                candidate_stage=4,
                track_id="T3",
            ),
        ]
        summary = gate_stage(4, reports)
        self.assertEqual(summary["promoted_count"], 0)
        self.assertEqual(summary["promotion_limit"], 0)


if __name__ == "__main__":
    unittest.main()
