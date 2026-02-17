from __future__ import annotations

from statistics import mean
import unittest

from exp.constants import LONG_CONTEXT_DATASETS
from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec
from exp.simulator import simulate_run


class BenchmarkPipelineTests(unittest.TestCase):
    def test_long_context_buckets_present(self) -> None:
        spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage1/t1_e2.yaml"))
        run = simulate_run(spec, seed=101, run_id="run-ctx", commit_sha="abc")

        for key in ("needle_32k", "needle_64k", "needle_128k"):
            self.assertIn(key, run.benchmark_scores)

        expected_long_context = mean([run.benchmark_scores[name] for name in LONG_CONTEXT_DATASETS])
        self.assertAlmostEqual(expected_long_context, run.metric_values["long_context"], places=4)


if __name__ == "__main__":
    unittest.main()
