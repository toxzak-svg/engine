from __future__ import annotations

import unittest

from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec
from exp.simulator import simulate_run


class DeterminismTests(unittest.TestCase):
    def test_same_seed_is_deterministic(self) -> None:
        spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage1/t3_e2.yaml"))
        run_a = simulate_run(spec, seed=102, run_id="run-a", commit_sha="abc")
        run_b = simulate_run(spec, seed=102, run_id="run-b", commit_sha="abc")

        self.assertEqual(run_a.metric_values, run_b.metric_values)
        self.assertEqual(run_a.benchmark_scores, run_b.benchmark_scores)
        self.assertEqual(run_a.failure_flags, run_b.failure_flags)


if __name__ == "__main__":
    unittest.main()
