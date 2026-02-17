from __future__ import annotations

import unittest

from exp.compare import compare_runs
from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec
from exp.simulator import simulate_run
from exp.spec_utils import build_cost_matched_baseline_spec


class CostMatchedBaselineTests(unittest.TestCase):
    def test_recovery_baseline_cost_matching_keeps_parity(self) -> None:
        candidate = ExperimentSpec.from_dict(read_yaml_or_json("specs/recovery/stage3/t4_binding_stability.yaml"))
        stage_baseline = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage3/t4_baseline.yaml"))
        matched_baseline = build_cost_matched_baseline_spec(candidate, stage_baseline)

        candidate_run = simulate_run(candidate, seed=301, run_id="cand", commit_sha="abc")
        baseline_run = simulate_run(matched_baseline, seed=301, run_id="base", commit_sha="abc")
        report = compare_runs(candidate_run, baseline_run)

        self.assertTrue(report.pass_fail["equal_cost_pass"])
        self.assertLessEqual(report.pass_fail["cost_parity_pct"], 2.0)


if __name__ == "__main__":
    unittest.main()
