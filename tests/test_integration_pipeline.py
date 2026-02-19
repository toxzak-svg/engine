import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from exp.simulator import simulate_run
from exp.models import ExperimentSpec
from exp.io import read_yaml_or_json
from exp.store import save_run_result, save_comparison
from exp.compare import compare_runs

class TestIntegrationPipeline(unittest.TestCase):

    @patch("exp.io.read_yaml_or_json")
    @patch("exp.simulator.simulate_run")
    @patch("exp.store.save_run_result")
    @patch("exp.store.save_comparison")
    def test_experiment_pipeline(self, mock_save_comparison, mock_save_run_result, mock_simulate_run, mock_read_yaml):
        # Mocking the spec generation
        mock_read_yaml.return_value = {
            "id": "test-spec",
            "stage": 1,
            "type": "main",
            "seeds": [1, 2, 3],
            "track_id": "track1",
            "model_variant": "model1",
            "composite": 0.5,
            "failure_flags": [False, False, False],
        }

        # Mocking the simulation
        mock_simulate_run.return_value = MagicMock(run_id="run-1", spec_id="test-spec")

        # Simulate running the pipeline
        spec_path = Path("specs/stage1/test_spec.yaml")
        spec = ExperimentSpec.from_dict(mock_read_yaml.return_value)
        run_result = simulate_run(spec, seed=1, run_id="run-1", commit_sha="abc")
        save_run_result(run_result)

        # Mocking comparison
        mock_save_comparison.return_value = None
        compare_runs("run-1", "baseline-run")

        # Assertions
        mock_read_yaml.assert_called_once_with(spec_path)
        mock_simulate_run.assert_called_once()
        mock_save_run_result.assert_called_once_with(run_result)
        mock_save_comparison.assert_called_once()

if __name__ == "__main__":
    unittest.main()