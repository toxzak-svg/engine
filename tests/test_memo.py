from __future__ import annotations

import unittest

from exp.memo import build_decision_memo
from exp.models import ComparisonReport


class MemoTests(unittest.TestCase):
    def test_memo_contains_ranking_table(self) -> None:
        report = ComparisonReport(
            candidate_run_ids=["run-a"],
            baseline_run_ids=["base-a"],
            delta_metrics={"composite": 4.5},
            significance_tests={"ci95_excludes_zero": True},
            pass_fail={"overall_pass": True},
            candidate_stage=1,
            track_id="T1",
        )
        memo = build_decision_memo(stage=1, reports=[report])
        self.assertIn("Stage 1 Decision Memo", memo)
        self.assertIn("| Rank | Track |", memo)


if __name__ == "__main__":
    unittest.main()
