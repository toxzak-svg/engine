"""Unit tests for the preflight checklist infrastructure."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from exp.preflight import (
    CheckStatus,
    PreflightCheck,
    PreflightReport,
    check_api_credentials,
    check_comparator_model_ids,
    check_cost_latency_ceilings,
    check_dataset_hash,
    check_environment_readiness,
    check_guardrails_configured,
    format_report_text,
    load_protocol_config,
    run_preflight_checks,
    GUARDRAILS,
)


class TestPreflightCheck(unittest.TestCase):
    """Tests for PreflightCheck dataclass."""

    def test_preflight_check_creation(self):
        """Test creating a preflight check."""
        check = PreflightCheck(
            name="test_check",
            status=CheckStatus.PASS,
            message="Check passed",
        )
        self.assertEqual(check.name, "test_check")
        self.assertEqual(check.status, CheckStatus.PASS)
        self.assertEqual(check.message, "Check passed")
        self.assertIsNotNone(check.timestamp)
        self.assertEqual(check.details, {})

    def test_preflight_check_with_details(self):
        """Test creating a preflight check with details."""
        details = {"key": "value", "count": 42}
        check = PreflightCheck(
            name="test_check",
            status=CheckStatus.FAIL,
            message="Check failed",
            details=details,
        )
        self.assertEqual(check.details, details)

    def test_preflight_check_to_dict(self):
        """Test converting preflight check to dictionary."""
        check = PreflightCheck(
            name="test_check",
            status=CheckStatus.WARNING,
            message="Warning message",
            details={"foo": "bar"},
        )
        result = check.to_dict()
        self.assertEqual(result["name"], "test_check")
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["message"], "Warning message")
        self.assertEqual(result["details"]["foo"], "bar")
        self.assertIn("timestamp", result)


class TestPreflightReport(unittest.TestCase):
    """Tests for PreflightReport dataclass."""

    def test_preflight_report_creation(self):
        """Test creating a preflight report."""
        checks = [
            PreflightCheck("check1", CheckStatus.PASS, "OK"),
            PreflightCheck("check2", CheckStatus.FAIL, "Failed"),
        ]
        report = PreflightReport(
            checks=checks,
            overall_status=CheckStatus.FAIL,
        )
        self.assertEqual(len(report.checks), 2)
        self.assertEqual(report.overall_status, CheckStatus.FAIL)

    def test_preflight_report_passed_property(self):
        """Test the passed property."""
        passing_report = PreflightReport(
            checks=[PreflightCheck("check1", CheckStatus.PASS, "OK")],
            overall_status=CheckStatus.PASS,
        )
        failing_report = PreflightReport(
            checks=[PreflightCheck("check1", CheckStatus.FAIL, "Failed")],
            overall_status=CheckStatus.FAIL,
        )
        self.assertTrue(passing_report.passed)
        self.assertFalse(failing_report.passed)

    def test_preflight_report_failed_checks(self):
        """Test getting failed checks from report."""
        checks = [
            PreflightCheck("check1", CheckStatus.PASS, "OK"),
            PreflightCheck("check2", CheckStatus.FAIL, "Failed"),
            PreflightCheck("check3", CheckStatus.FAIL, "Also failed"),
        ]
        report = PreflightReport(checks=checks, overall_status=CheckStatus.FAIL)
        failed = report.failed_checks
        self.assertEqual(len(failed), 2)
        self.assertEqual(failed[0].name, "check2")
        self.assertEqual(failed[1].name, "check3")

    def test_preflight_report_warning_checks(self):
        """Test getting warning checks from report."""
        checks = [
            PreflightCheck("check1", CheckStatus.PASS, "OK"),
            PreflightCheck("check2", CheckStatus.WARNING, "Warning"),
        ]
        report = PreflightReport(checks=checks, overall_status=CheckStatus.WARNING)
        warnings = report.warning_checks
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].name, "check2")

    def test_preflight_report_to_dict(self):
        """Test converting preflight report to dictionary."""
        checks = [
            PreflightCheck("check1", CheckStatus.PASS, "OK"),
        ]
        report = PreflightReport(
            checks=checks,
            overall_status=CheckStatus.PASS,
            protocol_version="1.0.0",
        )
        result = report.to_dict()
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["protocol_version"], "1.0.0")
        self.assertTrue(result["passed"])
        self.assertEqual(result["total_checks"], 1)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["warning_count"], 0)
        self.assertIn("checks", result)


class TestCheckComparatorModelIds(unittest.TestCase):
    """Tests for check_comparator_model_ids function."""

    def test_all_model_ids_configured(self):
        """Test when all model IDs are properly configured."""
        protocol = {
            "comparators": [
                {"slot": "ext_1", "provider": "OpenAI", "model_id": "gpt-4"},
                {"slot": "ext_2", "provider": "Anthropic", "model_id": "claude-3"},
            ]
        }
        result = check_comparator_model_ids(protocol)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertIn("2 comparator", result.message)

    def test_tbd_model_ids(self):
        """Test when some model IDs are TBD."""
        protocol = {
            "comparators": [
                {"slot": "ext_1", "provider": "OpenAI", "model_id": "gpt-4"},
                {"slot": "ext_2", "provider": "Anthropic", "model_id": "TBD"},
            ]
        }
        result = check_comparator_model_ids(protocol)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("TBD", result.message)
        self.assertEqual(len(result.details["tbd_slots"]), 1)

    def test_empty_model_ids(self):
        """Test when model IDs are empty."""
        protocol = {
            "comparators": [
                {"slot": "ext_1", "provider": "OpenAI", "model_id": ""},
            ]
        }
        result = check_comparator_model_ids(protocol)
        self.assertEqual(result.status, CheckStatus.FAIL)

    def test_no_comparators(self):
        """Test when no comparators are defined."""
        protocol = {"comparators": []}
        result = check_comparator_model_ids(protocol)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("No comparators", result.message)


class TestCheckCostLatencyCeilings(unittest.TestCase):
    """Tests for check_cost_latency_ceilings function."""

    def test_all_ceilings_defined(self):
        """Test when all cost/latency ceilings are defined."""
        protocol = {
            "fairness_constraints": {
                "cost_controls": {
                    "max_cost_per_request_usd": 0.50,
                    "max_cost_per_success_usd": 1.00,
                    "equal_cost_tolerance_pct": 2.0,
                },
                "latency_controls": {
                    "target_latency_p95_ms": 5000,
                    "max_latency_regression_pct_vs_baseline": 15.0,
                },
            },
            "production_ab": {
                "guardrails": {
                    "max_failure_rate_regression_pct": 5.0,
                }
            }
        }
        result = check_cost_latency_ceilings(protocol)
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_tbd_ceilings(self):
        """Test when some ceilings are TBD."""
        protocol = {
            "fairness_constraints": {
                "cost_controls": {
                    "max_cost_per_request_usd": "TBD",
                    "max_cost_per_success_usd": 1.00,
                },
                "latency_controls": {
                    "target_latency_p95_ms": "TBD",
                },
            }
        }
        result = check_cost_latency_ceilings(protocol)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("TBD", result.message)


class TestCheckDatasetHash(unittest.TestCase):
    """Tests for check_dataset_hash function."""

    def test_dataset_hash_defined(self):
        """Test when dataset hash is defined."""
        protocol = {
            "task_matrix": {
                "dataset_snapshot_hash": "abc123def456",
                "prompt_pack_version": "t3-ext-v1",
            }
        }
        result = check_dataset_hash(protocol)
        # Hash is defined but can't be verified (no dataset file)
        self.assertIn(result.status, [CheckStatus.WARNING, CheckStatus.PASS])

    def test_dataset_hash_tbd(self):
        """Test when dataset hash is TBD."""
        protocol = {
            "task_matrix": {
                "dataset_snapshot_hash": "TBD",
                "prompt_pack_version": "t3-ext-v1",
            }
        }
        result = check_dataset_hash(protocol)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("TBD", result.message)

    def test_dataset_hash_missing(self):
        """Test when dataset hash is missing."""
        protocol = {
            "task_matrix": {
                "prompt_pack_version": "t3-ext-v1",
            }
        }
        result = check_dataset_hash(protocol)
        self.assertEqual(result.status, CheckStatus.FAIL)


class TestCheckApiCredentials(unittest.TestCase):
    """Tests for check_api_credentials function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}, clear=False)
    def test_api_credentials_present(self):
        """Test when required API credentials are present."""
        protocol = {
            "comparators": [
                {"slot": "ext_1", "provider": "OpenAI", "model_id": "gpt-4"},
            ]
        }
        result = check_api_credentials(protocol)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertIn("OpenAI", str(result.details))

    def test_api_credentials_missing(self):
        """Test when required API credentials are missing."""
        # Use a mock to simulate missing API key
        with patch('exp.preflight.os.environ.get', return_value=None):
            protocol = {
                "comparators": [
                    {"slot": "ext_1", "provider": "OpenAI", "model_id": "gpt-4"},
                ]
            }
            result = check_api_credentials(protocol)
            self.assertEqual(result.status, CheckStatus.FAIL)
            self.assertIn("missing", result.message.lower())

    def test_no_api_key_required(self):
        """Test when provider doesn't require API key."""
        protocol = {
            "comparators": [
                {"slot": "ext_4", "provider": "OpenSourceLeader", "model_id": "llama-3"},
            ]
        }
        result = check_api_credentials(protocol)
        # Should pass or warn since no API key is required
        self.assertIn(result.status, [CheckStatus.PASS, CheckStatus.WARNING])


class TestCheckEnvironmentReadiness(unittest.TestCase):
    """Tests for check_environment_readiness function."""

    def test_environment_ready(self):
        """Test when environment is ready."""
        with patch.object(Path, 'exists', return_value=True):
            result = check_environment_readiness()
            self.assertEqual(result.status, CheckStatus.PASS)

    def test_missing_directories(self):
        """Test when required directories are missing."""
        with patch.object(Path, 'exists', return_value=False):
            result = check_environment_readiness()
            self.assertEqual(result.status, CheckStatus.FAIL)
            self.assertIn("Missing directories", result.message)


class TestCheckGuardrailsConfigured(unittest.TestCase):
    """Tests for check_guardrails_configured function."""

    def test_guardrails_correct(self):
        """Test when guardrails are correctly configured."""
        protocol = {
            "fairness_constraints": {
                "cost_controls": {
                    "equal_cost_tolerance_pct": 2.0,
                },
                "latency_controls": {
                    "max_latency_regression_pct_vs_baseline": 15.0,
                },
            }
        }
        result = check_guardrails_configured(protocol)
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_guardrails_wrong_tolerance(self):
        """Test when cost tolerance is wrong."""
        protocol = {
            "fairness_constraints": {
                "cost_controls": {
                    "equal_cost_tolerance_pct": 5.0,  # Should be 2.0
                },
                "latency_controls": {
                    "max_latency_regression_pct_vs_baseline": 15.0,
                },
            }
        }
        result = check_guardrails_configured(protocol)
        self.assertEqual(result.status, CheckStatus.WARNING)


class TestRunPreflightChecks(unittest.TestCase):
    """Tests for run_preflight_checks function."""

    def test_run_all_checks(self):
        """Test running all preflight checks."""
        # Create a mock protocol that will pass most checks
        mock_protocol = {
            "version": "1.0.0",
            "comparators": [
                {"slot": "ext_1", "provider": "OpenAI", "model_id": "gpt-4"},
            ],
            "fairness_constraints": {
                "cost_controls": {
                    "max_cost_per_request_usd": 0.50,
                    "max_cost_per_success_usd": 1.00,
                    "equal_cost_tolerance_pct": 2.0,
                },
                "latency_controls": {
                    "target_latency_p95_ms": 5000,
                    "max_latency_regression_pct_vs_baseline": 15.0,
                },
            },
            "task_matrix": {
                "dataset_snapshot_hash": "abc123",
                "prompt_pack_version": "v1",
            },
        }
        
        with patch('exp.preflight.load_protocol_config', return_value=mock_protocol):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
                with patch.object(Path, 'exists', return_value=True):
                    report = run_preflight_checks()
                    
                    self.assertIsNotNone(report)
                    self.assertEqual(report.protocol_version, "1.0.0")
                    # Should have 6 checks
                    self.assertEqual(len(report.checks), 6)

    def test_skip_checks(self):
        """Test skipping specific checks."""
        mock_protocol = {
            "version": "1.0.0",
            "comparators": [],
        }
        
        with patch('exp.preflight.load_protocol_config', return_value=mock_protocol):
            report = run_preflight_checks(skip_checks=["comparator_model_ids"])
            
            # Find the skipped check
            skipped = [c for c in report.checks if c.name == "comparator_model_ids"]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0].status, CheckStatus.SKIP)


class TestFormatReportText(unittest.TestCase):
    """Tests for format_report_text function."""

    def test_format_report(self):
        """Test formatting a report as text."""
        checks = [
            PreflightCheck("check1", CheckStatus.PASS, "All good"),
            PreflightCheck("check2", CheckStatus.FAIL, "Something failed"),
        ]
        report = PreflightReport(
            checks=checks,
            overall_status=CheckStatus.FAIL,
            protocol_version="1.0.0",
        )
        text = format_report_text(report)
        
        self.assertIn("PREFLIGHT REPORT", text)
        self.assertIn("check1", text)
        self.assertIn("check2", text)
        self.assertIn("FAIL", text)
        self.assertIn("Total: 2 checks", text)


class TestGuardrailConstants(unittest.TestCase):
    """Tests for guardrail constants."""

    def test_guardrail_values(self):
        """Test that guardrail constants match expected values."""
        self.assertEqual(GUARDRAILS["equal_cost_tolerance_pct"], 2.0)
        self.assertEqual(GUARDRAILS["max_latency_regression_pct"], 15.0)
        self.assertEqual(GUARDRAILS["token_access_reduction_min_pct"], 70.0)
        self.assertEqual(GUARDRAILS["critical_fact_miss_rate_increase_max_pct"], 2.0)


class TestLoadProtocolConfig(unittest.TestCase):
    """Tests for load_protocol_config function."""

    def test_load_missing_config(self):
        """Test loading a missing config file."""
        with self.assertRaises(FileNotFoundError):
            load_protocol_config("nonexistent_file.yaml")


class TestIntegration(unittest.TestCase):
    """Integration tests for preflight module."""

    @patch('exp.preflight.load_protocol_config')
    def test_full_preflight_with_failures(self, mock_load):
        """Test full preflight run with expected failures."""
        # Use the actual protocol structure
        mock_load.return_value = {
            "version": "1.0.0",
            "comparators": [
                {"slot": "ext_1", "provider": "OpenAI", "model_id": "TBD"},
            ],
            "fairness_constraints": {
                "cost_controls": {
                    "max_cost_per_request_usd": "TBD",
                },
                "latency_controls": {
                    "target_latency_p95_ms": "TBD",
                },
            },
            "task_matrix": {
                "dataset_snapshot_hash": "TBD",
            },
        }
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('exp.preflight.os.environ.get', return_value=None):
                report = run_preflight_checks()
                
                # Should have failures
                self.assertEqual(report.overall_status, CheckStatus.FAIL)
                self.assertGreater(len(report.failed_checks), 0)
                
                # Verify JSON serialization works
                report_dict = report.to_dict()
                self.assertIsInstance(json.dumps(report_dict), str)


if __name__ == "__main__":
    unittest.main()

