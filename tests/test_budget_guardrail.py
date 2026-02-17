from __future__ import annotations

import copy
import unittest

from exp.compare import compare_runs
from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec, RunResult
from exp.simulator import simulate_run


class BudgetGuardrailTests(unittest.TestCase):
    def test_equal_cost_pass_within_tolerance(self) -> None:
        candidate_spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage1/t1_e2.yaml"))
        baseline_spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage1/t1_baseline.yaml"))
        candidate = simulate_run(candidate_spec, seed=101, run_id="cand", commit_sha="abc")
        baseline = simulate_run(baseline_spec, seed=101, run_id="base", commit_sha="abc")

        report = compare_runs(candidate, baseline)
        self.assertTrue(report.pass_fail["equal_cost_pass"])

    def test_equal_cost_fails_outside_tolerance(self) -> None:
        candidate_spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage1/t1_e2.yaml"))
        baseline_spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage1/t1_baseline.yaml"))
        candidate = simulate_run(candidate_spec, seed=101, run_id="cand", commit_sha="abc")
        baseline = simulate_run(baseline_spec, seed=101, run_id="base", commit_sha="abc")

        inflated_payload = copy.deepcopy(candidate.to_dict())
        inflated_payload["train_cost"] = candidate.train_cost * 1.30
        inflated_candidate = RunResult.from_dict(inflated_payload)
        report = compare_runs(inflated_candidate, baseline)
        self.assertFalse(report.pass_fail["equal_cost_pass"])


if __name__ == "__main__":
    unittest.main()
