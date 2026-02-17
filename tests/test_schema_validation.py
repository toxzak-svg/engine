from __future__ import annotations

import unittest

from exp.io import read_yaml_or_json
from exp.models import ComparisonReport, ExperimentSpec, RunResult
from exp.schema_validator import SchemaValidationError


class SchemaValidationTests(unittest.TestCase):
    def test_experiment_spec_valid(self) -> None:
        payload = read_yaml_or_json("specs/stage1/t1_e1.yaml")
        spec = ExperimentSpec.from_dict(payload)
        self.assertEqual(spec.track_id, "T1")
        self.assertEqual(spec.stage, 1)

    def test_experiment_spec_missing_required_field(self) -> None:
        payload = read_yaml_or_json("specs/stage1/t1_e1.yaml")
        payload.pop("id")
        with self.assertRaises(SchemaValidationError):
            ExperimentSpec.from_dict(payload)

    def test_run_result_schema_valid(self) -> None:
        payload = {
            "run_id": "run-1",
            "spec_id": "s1-t1-e1",
            "commit_sha": "abc1234",
            "seed": 101,
            "train_cost": 120.0,
            "infer_cost": 30.0,
            "latency_p50": 100.0,
            "latency_p95": 130.0,
            "energy_kwh": 57.0,
            "metric_values": {"composite": 62.0, "fluency": 89.0},
            "failure_flags": [],
            "track_id": "T1",
            "stage": 1,
            "model_variant": "T1-E1",
            "benchmark_scores": {"needle_32k": 58.0}
        }
        run = RunResult.from_dict(payload)
        self.assertEqual(run.track_id, "T1")

    def test_comparison_schema_valid(self) -> None:
        payload = {
            "candidate_run_ids": ["run-a"],
            "baseline_run_ids": ["run-b"],
            "delta_metrics": {"composite": 4.2},
            "significance_tests": {"ci95": [1.0, 7.0], "ci95_excludes_zero": True},
            "pass_fail": {"overall_pass": True},
            "candidate_stage": 1,
            "track_id": "T1"
        }
        report = ComparisonReport.from_dict(payload)
        self.assertEqual(report.candidate_stage, 1)


if __name__ == "__main__":
    unittest.main()
