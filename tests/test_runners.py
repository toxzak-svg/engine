"""Tests for the experiment runners module."""

import unittest
from unittest.mock import MagicMock, patch

from exp.runners import (
    BaseRunner,
    RunnerConfig,
    RunnerResult,
    RunnerStatus,
    SimulatorRunner,
    SubprocessRunner,
    T1Runner,
    T2Runner,
    T3Runner,
    T4Runner,
    T5Runner,
    T6Runner,
    AnchorRunner,
    get_runner,
    RunnerRegistry,
)
from exp.runners.factory import create_runner_from_config
from exp.models import ExperimentSpec


class TestRunnerConfig(unittest.TestCase):
    """Test RunnerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RunnerConfig()
        self.assertEqual(config.timeout_seconds, 0.0)
        self.assertIsNone(config.working_directory)
        self.assertEqual(config.environment, {})
        self.assertTrue(config.capture_logs)
        self.assertFalse(config.dry_run)
        self.assertEqual(config.gpu_device_ids, [0])
        self.assertIsNone(config.checkpoint_dir)
        self.assertIsNone(config.artifact_dir)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RunnerConfig(
            timeout_seconds=3600.0,
            working_directory="/workspace",
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
            dry_run=True,
            gpu_device_ids=[0, 1, 2],
        )
        self.assertEqual(config.timeout_seconds, 3600.0)
        self.assertEqual(config.working_directory, "/workspace")
        self.assertEqual(config.environment, {"CUDA_VISIBLE_DEVICES": "0,1"})
        self.assertTrue(config.dry_run)
        self.assertEqual(config.gpu_device_ids, [0, 1, 2])


class TestRunnerResult(unittest.TestCase):
    """Test RunnerResult dataclass."""

    def test_success_result(self):
        """Test successful runner result."""
        result = RunnerResult(
            success=True,
            run_result=MagicMock(),
            status=RunnerStatus.COMPLETED,
            execution_time_seconds=1.5,
        )
        self.assertTrue(result.success)
        self.assertIsNotNone(result.run_result)
        self.assertEqual(result.status, RunnerStatus.COMPLETED)
        self.assertEqual(result.execution_time_seconds, 1.5)

    def test_failure_result(self):
        """Test failed runner result."""
        result = RunnerResult(
            success=False,
            run_result=None,
            error_message="Test error",
            status=RunnerStatus.FAILED,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.run_result)
        self.assertEqual(result.error_message, "Test error")
        self.assertEqual(result.status, RunnerStatus.FAILED)


class TestSimulatorRunner(unittest.TestCase):
    """Test SimulatorRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RunnerConfig(dry_run=False)
        self.runner = SimulatorRunner(self.config)
        
        # Create a sample spec (stage 1 requires at least 3 seeds)
        self.spec = ExperimentSpec.from_dict({
            "id": "test-spec",
            "track_id": "T1",
            "stage": 1,
            "hypothesis": "Test hypothesis",
            "model_variant": "T1-E1",
            "baseline_id": "test-baseline",
            "train_budget_gpu_h": 100.0,
            "infer_budget_gpu_h": 20.0,
            "max_context": 128000,
            "datasets": ["needle_32k"],
            "metrics": ["composite"],
            "seeds": [101, 102, 103],
            "promotion_gate": {"delta_composite_min": 3.0},
            "params": {"noise_model": "analog_v1"},
        })

    def test_runner_attributes(self):
        """Test runner attributes."""
        self.assertEqual(self.runner.name, "simulator")
        self.assertIn("T1", self.runner.supported_tracks)
        self.assertIn("T2", self.runner.supported_tracks)
        self.assertIn("ANCHOR", self.runner.supported_tracks)

    def test_execute_success(self):
        """Test successful execution."""
        result = self.runner.execute(
            spec=self.spec,
            seed=101,
            run_id="test-run-101",
            commit_sha="abc123",
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.run_result)
        self.assertEqual(result.status, RunnerStatus.COMPLETED)
        self.assertEqual(result.run_result.run_id, "test-run-101")
        self.assertEqual(result.run_result.spec_id, "test-spec")
        self.assertEqual(result.run_result.seed, 101)

    def test_execute_dry_run(self):
        """Test dry run mode."""
        config = RunnerConfig(dry_run=True)
        runner = SimulatorRunner(config)
        
        result = runner.execute(
            spec=self.spec,
            seed=101,
            run_id="test-run-101",
            commit_sha="abc123",
        )
        
        self.assertTrue(result.success)
        self.assertIsNone(result.run_result)
        self.assertEqual(result.status, RunnerStatus.COMPLETED)
        self.assertTrue(result.metadata.get("dry_run", False))

    def test_validate_spec_valid(self):
        """Test spec validation with valid spec."""
        errors = self.runner.validate_spec(self.spec)
        self.assertEqual(len(errors), 0)

    def test_validate_spec_invalid_track(self):
        """Test spec validation with invalid track."""
        runner = T1Runner(self.config)  # T1Runner only supports T1
        
        # Create a spec for T2
        spec_dict = self.spec.to_dict()
        spec_dict["track_id"] = "T2"
        spec_dict["id"] = "test-spec-t2"
        spec_t2 = ExperimentSpec.from_dict(spec_dict)
        
        errors = runner.validate_spec(spec_t2)
        self.assertGreater(len(errors), 0)
        self.assertIn("T2", errors[0])


class TestSubprocessRunner(unittest.TestCase):
    """Test SubprocessRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RunnerConfig(dry_run=True)
        # Use stage 0 which has no minimum seed requirement
        self.spec = ExperimentSpec.from_dict({
            "id": "test-spec",
            "track_id": "T1",
            "stage": 0,
            "hypothesis": "Test hypothesis",
            "model_variant": "T1-E1",
            "baseline_id": "test-baseline",
            "train_budget_gpu_h": 100.0,
            "infer_budget_gpu_h": 20.0,
            "max_context": 128000,
            "datasets": ["needle_32k"],
            "metrics": ["composite"],
            "seeds": [101],
            "promotion_gate": {"delta_composite_min": 3.0},
            "params": {"noise_model": "analog_v1"},
        })

    def test_command_template_substitution(self):
        """Test command template variable substitution."""
        runner = SubprocessRunner(
            config=self.config,
            command_template="python run.py --spec {spec_id} --seed {seed} --gpu {gpu_ids}",
        )
        
        command = runner._build_command(
            spec=self.spec,
            seed=101,
            run_id="test-run",
        )
        
        self.assertIn("test-spec", command)
        self.assertIn("--seed 101", command)
        self.assertIn("--gpu 0", command)

    def test_dry_run_execution(self):
        """Test dry run execution."""
        runner = SubprocessRunner(
            config=self.config,
            command_template="echo test",
        )
        
        result = runner.execute(
            spec=self.spec,
            seed=101,
            run_id="test-run",
            commit_sha="abc123",
        )
        
        self.assertTrue(result.success)
        self.assertIsNone(result.run_result)
        self.assertEqual(result.status, RunnerStatus.COMPLETED)

    def test_parse_json_result(self):
        """Test JSON result parsing."""
        runner = SubprocessRunner(config=self.config)
        
        json_output = '''
        Some log output
        {"train_cost": 100.5, "infer_cost": 20.3, "latency_p50": 50.0, "latency_p95": 80.0, "energy_kwh": 10.0, "metric_values": {"composite": 65.0}, "failure_flags": [], "benchmark_scores": {}}
        '''
        
        result = runner._parse_result(
            output=json_output,
            spec=self.spec,
            seed=101,
            run_id="test-run",
            commit_sha="abc123",
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.train_cost, 100.5)
        self.assertEqual(result.infer_cost, 20.3)


class TestTrackRunners(unittest.TestCase):
    """Test track-specific runners."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RunnerConfig(dry_run=True)

    def test_t1_runner(self):
        """Test T1 runner."""
        runner = T1Runner(self.config)
        self.assertEqual(runner.name, "t1")
        self.assertEqual(runner.supported_tracks, ["T1"])

    def test_t2_runner(self):
        """Test T2 runner."""
        runner = T2Runner(self.config)
        self.assertEqual(runner.name, "t2")
        self.assertEqual(runner.supported_tracks, ["T2"])

    def test_t3_runner(self):
        """Test T3 runner."""
        runner = T3Runner(self.config)
        self.assertEqual(runner.name, "t3")
        self.assertEqual(runner.supported_tracks, ["T3"])

    def test_t4_runner(self):
        """Test T4 runner."""
        runner = T4Runner(self.config)
        self.assertEqual(runner.name, "t4")
        self.assertEqual(runner.supported_tracks, ["T4"])

    def test_t5_runner(self):
        """Test T5 runner."""
        runner = T5Runner(self.config)
        self.assertEqual(runner.name, "t5")
        self.assertEqual(runner.supported_tracks, ["T5"])

    def test_t6_runner(self):
        """Test T6 runner."""
        runner = T6Runner(self.config)
        self.assertEqual(runner.name, "t6")
        self.assertEqual(runner.supported_tracks, ["T6"])

    def test_anchor_runner(self):
        """Test Anchor runner."""
        runner = AnchorRunner(self.config)
        self.assertEqual(runner.name, "anchor")
        self.assertEqual(runner.supported_tracks, ["ANCHOR"])

    def test_t1_validation(self):
        """Test T1-specific validation."""
        runner = T1Runner(self.config)
        
        # Use stage 0 which has no minimum seed requirement
        spec = ExperimentSpec.from_dict({
            "id": "test-spec",
            "track_id": "T1",
            "stage": 0,
            "hypothesis": "Test",
            "model_variant": "T1-E1",
            "baseline_id": "baseline",
            "train_budget_gpu_h": 100.0,
            "infer_budget_gpu_h": 20.0,
            "max_context": 128000,
            "datasets": ["needle_32k"],
            "metrics": ["composite"],
            "seeds": [101],
            "promotion_gate": {},
            "params": {"noise_model": "invalid_model"},  # Invalid
        })
        
        errors = runner.validate_spec(spec)
        self.assertGreater(len(errors), 0)
        self.assertIn("noise_model", errors[0])

    def test_t3_validation(self):
        """Test T3-specific validation."""
        runner = T3Runner(self.config)
        
        # Use stage 0 which has no minimum seed requirement
        spec = ExperimentSpec.from_dict({
            "id": "test-spec",
            "track_id": "T3",
            "stage": 0,
            "hypothesis": "Test",
            "model_variant": "T3-E1",
            "baseline_id": "baseline",
            "train_budget_gpu_h": 100.0,
            "infer_budget_gpu_h": 20.0,
            "max_context": 128000,
            "datasets": ["needle_32k"],
            "metrics": ["composite"],
            "seeds": [101],
            "promotion_gate": {},
            "params": {"compression_ratio": 1.5},  # Invalid: > 1.0
        })
        
        errors = runner.validate_spec(spec)
        self.assertGreater(len(errors), 0)
        self.assertIn("compression_ratio", errors[0])


class TestRunnerRegistry(unittest.TestCase):
    """Test RunnerRegistry class."""

    def test_list_runners(self):
        """Test listing registered runners."""
        runners = RunnerRegistry.list_runners()
        self.assertIn("simulator", runners)
        self.assertIn("subprocess", runners)
        self.assertIn("t1", runners)
        self.assertIn("t2", runners)
        self.assertIn("t3", runners)
        self.assertIn("t4", runners)
        self.assertIn("t5", runners)
        self.assertIn("t6", runners)
        self.assertIn("anchor", runners)

    def test_get_runner_class(self):
        """Test getting runner class by name."""
        runner_class = RunnerRegistry.get_runner_class("simulator")
        self.assertEqual(runner_class, SimulatorRunner)
        
        runner_class = RunnerRegistry.get_runner_class("t1")
        self.assertEqual(runner_class, T1Runner)

    def test_get_runner_for_track(self):
        """Test getting runner class for a track."""
        runner_class = RunnerRegistry.get_runner_for_track("T1")
        self.assertEqual(runner_class, T1Runner)
        
        runner_class = RunnerRegistry.get_runner_for_track("T3")
        self.assertEqual(runner_class, T3Runner)
        
        runner_class = RunnerRegistry.get_runner_for_track("ANCHOR")
        self.assertEqual(runner_class, AnchorRunner)


class TestGetRunner(unittest.TestCase):
    """Test get_runner factory function."""

    def test_get_simulator_runner(self):
        """Test getting simulator runner."""
        runner = get_runner(track_id="T1", runner_type="simulator")
        self.assertIsInstance(runner, SimulatorRunner)

    def test_get_track_runner(self):
        """Test getting track-specific runner."""
        runner = get_runner(track_id="T1")
        self.assertIsInstance(runner, T1Runner)
        
        runner = get_runner(track_id="T3")
        self.assertIsInstance(runner, T3Runner)

    def test_get_runner_with_config(self):
        """Test getting runner with configuration."""
        config = RunnerConfig(dry_run=True, timeout_seconds=3600)
        runner = get_runner(track_id="T1", config=config)
        self.assertTrue(runner.config.dry_run)
        self.assertEqual(runner.config.timeout_seconds, 3600)

    def test_get_runner_invalid_type(self):
        """Test getting runner with invalid type."""
        with self.assertRaises(ValueError):
            get_runner(track_id="T1", runner_type="invalid")


class TestCreateRunnerFromConfig(unittest.TestCase):
    """Test create_runner_from_config function."""

    def test_create_from_config_dict(self):
        """Test creating runner from configuration dictionary."""
        config_dict = {
            "type": "simulator",
            "timeout_seconds": 3600,
            "dry_run": True,
            "gpu_device_ids": [0, 1],
        }
        
        runner = create_runner_from_config(config_dict, track_id="T1")
        self.assertIsInstance(runner, SimulatorRunner)
        self.assertEqual(runner.config.timeout_seconds, 3600)
        self.assertTrue(runner.config.dry_run)
        self.assertEqual(runner.config.gpu_device_ids, [0, 1])

    def test_create_track_runner_from_config(self):
        """Test creating track runner from configuration dictionary."""
        config_dict = {
            "type": "t1",
            "command_template": "python run.py --spec {spec_id}",
        }
        
        runner = create_runner_from_config(config_dict)
        self.assertIsInstance(runner, T1Runner)


if __name__ == "__main__":
    unittest.main()