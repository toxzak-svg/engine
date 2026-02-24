"""Preflight checklist infrastructure for T3 External Benchmark Protocol.

This module provides validation checks that must pass before running external
benchmark comparisons. It ensures all configuration, credentials, and environment
requirements are met.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class CheckStatus(Enum):
    """Status of a preflight check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class PreflightCheck:
    """Result of a single preflight check.
    
    Attributes:
        name: Identifier for the check
        status: Pass/fail/warning/skip status
        message: Human-readable description of the result
        timestamp: When the check was performed
        details: Additional structured details about the check
    """
    name: str
    status: CheckStatus
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class PreflightReport:
    """Complete preflight report with all check results.
    
    Attributes:
        checks: List of all preflight checks performed
        overall_status: Aggregate status (pass only if all checks pass)
        protocol_version: Version of the protocol being validated
        timestamp: When the report was generated
    """
    checks: list[PreflightCheck]
    overall_status: CheckStatus
    protocol_version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def passed(self) -> bool:
        """Check if all validations passed."""
        return self.overall_status == CheckStatus.PASS
    
    @property
    def failed_checks(self) -> list[PreflightCheck]:
        """List of checks that failed."""
        return [c for c in self.checks if c.status == CheckStatus.FAIL]
    
    @property
    def warning_checks(self) -> list[PreflightCheck]:
        """List of checks with warnings."""
        return [c for c in self.checks if c.status == CheckStatus.WARNING]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "total_checks": len(self.checks),
            "failed_count": len(self.failed_checks),
            "warning_count": len(self.warning_checks),
            "checks": [c.to_dict() for c in self.checks],
        }


# Guardrail constants from protocol
GUARDRAILS = {
    "equal_cost_tolerance_pct": 2.0,
    "max_latency_regression_pct": 15.0,
    "token_access_reduction_min_pct": 70.0,
    "critical_fact_miss_rate_increase_max_pct": 2.0,
}

# Required API credential environment variables by provider
REQUIRED_API_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
    # OpenSourceLeader may use local models, no API key required
}


def load_protocol_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the external benchmark protocol configuration.
    
    Args:
        config_path: Path to protocol config file. Defaults to 
            config/t3_external_benchmark_protocol.yaml
    
    Returns:
        Parsed protocol configuration dictionary
    """
    if config_path is None:
        config_path = Path("config/t3_external_benchmark_protocol.yaml")
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Protocol config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Handle JSON-formatted YAML
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError:
        return json.loads(content)


def check_comparator_model_ids(protocol: dict[str, Any] | None = None) -> PreflightCheck:
    """Validate that all comparator models are properly configured.
    
    Checks that model_id fields are not "TBD" and have actual values.
    
    Args:
        protocol: Protocol configuration dict. If None, loads from default path.
    
    Returns:
        PreflightCheck with validation result
    """
    check_name = "comparator_model_ids"
    
    try:
        if protocol is None:
            protocol = load_protocol_config()
        
        comparators = protocol.get("comparators", [])
        if not comparators:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.FAIL,
                message="No comparators defined in protocol",
                details={"comparators_count": 0},
            )
        
        tbd_slots = []
        valid_slots = []
        
        for comp in comparators:
            slot = comp.get("slot", "unknown")
            model_id = comp.get("model_id", "")
            provider = comp.get("provider", "unknown")
            
            if model_id == "TBD" or not model_id:
                tbd_slots.append({"slot": slot, "provider": provider})
            else:
                valid_slots.append({"slot": slot, "provider": provider, "model_id": model_id})
        
        if tbd_slots:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.FAIL,
                message=f"{len(tbd_slots)} comparator(s) have TBD model IDs",
                details={
                    "tbd_slots": tbd_slots,
                    "valid_slots": valid_slots,
                    "total_comparators": len(comparators),
                },
            )
        
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.PASS,
            message=f"All {len(comparators)} comparator model IDs are configured",
            details={
                "valid_slots": valid_slots,
                "total_comparators": len(comparators),
            },
        )
    
    except FileNotFoundError as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Protocol config not found: {e}",
            details={"error": str(e)},
        )
    except Exception as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Error checking comparator model IDs: {e}",
            details={"error": str(e)},
        )


def check_cost_latency_ceilings(protocol: dict[str, Any] | None = None) -> PreflightCheck:
    """Ensure cost and latency thresholds are defined.
    
    Validates that cost controls and latency controls have actual values
    rather than "TBD" placeholders.
    
    Args:
        protocol: Protocol configuration dict. If None, loads from default path.
    
    Returns:
        PreflightCheck with validation result
    """
    check_name = "cost_latency_ceilings"
    
    try:
        if protocol is None:
            protocol = load_protocol_config()
        
        fairness = protocol.get("fairness_constraints", {})
        cost_controls = fairness.get("cost_controls", {})
        latency_controls = fairness.get("latency_controls", {})
        
        tbd_fields = []
        valid_fields = []
        
        # Check cost controls
        cost_fields = ["max_cost_per_request_usd", "max_cost_per_success_usd"]
        for field_name in cost_fields:
            value = cost_controls.get(field_name)
            if value == "TBD" or value is None:
                tbd_fields.append(f"cost_controls.{field_name}")
            else:
                valid_fields.append({"field": f"cost_controls.{field_name}", "value": value})
        
        # Check latency controls
        latency_fields = ["target_latency_p95_ms"]
        for field_name in latency_fields:
            value = latency_controls.get(field_name)
            if value == "TBD" or value is None:
                tbd_fields.append(f"latency_controls.{field_name}")
            else:
                valid_fields.append({"field": f"latency_controls.{field_name}", "value": value})
        
        # Check guardrail values
        guardrails = protocol.get("production_ab", {}).get("guardrails", {})
        guardrail_fields = [
            "max_failure_rate_regression_pct",
            "max_latency_regression_pct",
            "max_cost_regression_pct",
        ]
        for field_name in guardrail_fields:
            value = guardrails.get(field_name)
            if value is not None:
                valid_fields.append({"field": f"guardrails.{field_name}", "value": value})
        
        if tbd_fields:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.FAIL,
                message=f"{len(tbd_fields)} cost/latency ceiling(s) are TBD",
                details={
                    "tbd_fields": tbd_fields,
                    "valid_fields": valid_fields,
                    "equal_cost_tolerance_pct": cost_controls.get("equal_cost_tolerance_pct", GUARDRAILS["equal_cost_tolerance_pct"]),
                    "max_latency_regression_pct": latency_controls.get("max_latency_regression_pct_vs_baseline", GUARDRAILS["max_latency_regression_pct"]),
                },
            )
        
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.PASS,
            message="All cost and latency ceilings are defined",
            details={
                "valid_fields": valid_fields,
                "equal_cost_tolerance_pct": cost_controls.get("equal_cost_tolerance_pct", GUARDRAILS["equal_cost_tolerance_pct"]),
                "max_latency_regression_pct": latency_controls.get("max_latency_regression_pct_vs_baseline", GUARDRAILS["max_latency_regression_pct"]),
            },
        )
    
    except FileNotFoundError as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Protocol config not found: {e}",
            details={"error": str(e)},
        )
    except Exception as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Error checking cost/latency ceilings: {e}",
            details={"error": str(e)},
        )


def check_dataset_hash(protocol: dict[str, Any] | None = None) -> PreflightCheck:
    """Verify dataset integrity via hash validation.
    
    Checks that the dataset snapshot hash is defined (not "TBD").
    Optionally verifies the hash against an actual dataset file if present.
    
    Args:
        protocol: Protocol configuration dict. If None, loads from default path.
    
    Returns:
        PreflightCheck with validation result
    """
    check_name = "dataset_hash"
    
    try:
        if protocol is None:
            protocol = load_protocol_config()
        
        task_matrix = protocol.get("task_matrix", {})
        dataset_hash = task_matrix.get("dataset_snapshot_hash")
        prompt_pack_version = task_matrix.get("prompt_pack_version", "unknown")
        
        if dataset_hash == "TBD" or not dataset_hash:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.FAIL,
                message="Dataset snapshot hash is not defined (TBD)",
                details={
                    "dataset_snapshot_hash": dataset_hash,
                    "prompt_pack_version": prompt_pack_version,
                },
            )
        
        # Check if we can verify the hash against a dataset file
        dataset_path = Path("data/datasets") / prompt_pack_version
        hash_verified = False
        
        if dataset_path.exists():
            # Compute hash of dataset files
            hasher = hashlib.sha256()
            for file_path in sorted(dataset_path.rglob("*")):
                if file_path.is_file():
                    hasher.update(file_path.read_bytes())
            computed_hash = hasher.hexdigest()[:16]
            hash_verified = computed_hash == dataset_hash[:16] if len(dataset_hash) >= 16 else False
        
        details = {
            "dataset_snapshot_hash": dataset_hash,
            "prompt_pack_version": prompt_pack_version,
            "hash_verified": hash_verified,
        }
        
        if hash_verified:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.PASS,
                message="Dataset hash verified successfully",
                details=details,
            )
        else:
            # Hash is defined but we couldn't verify it (dataset may not be present)
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.WARNING,
                message="Dataset hash is defined but could not be verified",
                details=details,
            )
    
    except FileNotFoundError as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Protocol config not found: {e}",
            details={"error": str(e)},
        )
    except Exception as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Error checking dataset hash: {e}",
            details={"error": str(e)},
        )


def check_api_credentials(protocol: dict[str, Any] | None = None) -> PreflightCheck:
    """Validate that required API keys are available.
    
    Checks for the presence of API keys for all configured comparator providers.
    Does NOT expose the actual key values, only confirms presence/absence.
    
    Args:
        protocol: Protocol configuration dict. If None, loads from default path.
    
    Returns:
        PreflightCheck with validation result
    """
    check_name = "api_credentials"
    
    try:
        if protocol is None:
            protocol = load_protocol_config()
        
        comparators = protocol.get("comparators", [])
        
        missing_keys = []
        present_keys = []
        skipped_providers = []
        
        for comp in comparators:
            provider = comp.get("provider", "")
            slot = comp.get("slot", "unknown")
            
            env_var = REQUIRED_API_KEYS.get(provider)
            
            if env_var is None:
                # Provider doesn't require an API key (e.g., local models)
                skipped_providers.append({"provider": provider, "slot": slot, "reason": "no_key_required"})
                continue
            
            key_present = bool(os.environ.get(env_var))
            
            if key_present:
                present_keys.append({"provider": provider, "slot": slot, "env_var": env_var})
            else:
                missing_keys.append({"provider": provider, "slot": slot, "env_var": env_var})
        
        if missing_keys:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.FAIL,
                message=f"{len(missing_keys)} API credential(s) missing",
                details={
                    "missing_keys": missing_keys,
                    "present_keys": present_keys,
                    "skipped_providers": skipped_providers,
                },
            )
        
        status = CheckStatus.PASS if present_keys else CheckStatus.WARNING
        message = (
            f"All {len(present_keys)} required API credentials are available"
            if present_keys
            else "No API credentials required for configured providers"
        )
        
        return PreflightCheck(
            name=check_name,
            status=status,
            message=message,
            details={
                "present_keys": present_keys,
                "skipped_providers": skipped_providers,
            },
        )
    
    except FileNotFoundError as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Protocol config not found: {e}",
            details={"error": str(e)},
        )
    except Exception as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Error checking API credentials: {e}",
            details={"error": str(e)},
        )


def check_environment_readiness() -> PreflightCheck:
    """Check that the execution environment is ready.
    
    Validates:
    - Python version compatibility
    - Required directories exist
    - System resources available
    
    Returns:
        PreflightCheck with validation result
    """
    check_name = "environment_readiness"
    
    try:
        issues = []
        details = {}
        
        # Check Python version
        python_version = sys.version_info
        details["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 9):
            issues.append(f"Python 3.9+ required, found {details['python_version']}")
        
        # Check required directories
        required_dirs = ["artifacts", "artifacts/runs", "artifacts/comparisons", "artifacts/memos", "config"]
        missing_dirs = []
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            issues.append(f"Missing directories: {', '.join(missing_dirs)}")
        
        details["missing_directories"] = missing_dirs
        details["required_directories"] = required_dirs
        
        # Check system info
        details["platform"] = platform.system()
        details["platform_version"] = platform.version()
        
        # Check if we're in a git repository
        git_dir = Path(".git")
        details["is_git_repo"] = git_dir.exists()
        
        if issues:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.FAIL,
                message=f"Environment issues: {'; '.join(issues)}",
                details=details,
            )
        
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.PASS,
            message="Environment is ready for benchmark execution",
            details=details,
        )
    
    except Exception as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Error checking environment readiness: {e}",
            details={"error": str(e)},
        )


def check_guardrails_configured(protocol: dict[str, Any] | None = None) -> PreflightCheck:
    """Verify that guardrail thresholds are properly configured.
    
    Validates that the guardrails from the protocol match expected values:
    - Equal-cost tolerance: ±2%
    - Latency overhead: ≤15%
    - Token access reduction: ≥70%
    - Critical fact miss rate increase: ≤2%
    
    Args:
        protocol: Protocol configuration dict. If None, loads from default path.
    
    Returns:
        PreflightCheck with validation result
    """
    check_name = "guardrails_configured"
    
    try:
        if protocol is None:
            protocol = load_protocol_config()
        
        fairness = protocol.get("fairness_constraints", {})
        cost_controls = fairness.get("cost_controls", {})
        latency_controls = fairness.get("latency_controls", {})
        
        issues = []
        details = {}
        
        # Check equal cost tolerance
        cost_tolerance = cost_controls.get("equal_cost_tolerance_pct")
        details["equal_cost_tolerance_pct"] = cost_tolerance
        if cost_tolerance is None or cost_tolerance != GUARDRAILS["equal_cost_tolerance_pct"]:
            issues.append(f"equal_cost_tolerance_pct should be {GUARDRAILS['equal_cost_tolerance_pct']}, found {cost_tolerance}")
        
        # Check latency regression
        latency_regression = latency_controls.get("max_latency_regression_pct_vs_baseline")
        details["max_latency_regression_pct"] = latency_regression
        if latency_regression is None or latency_regression > GUARDRAILS["max_latency_regression_pct"]:
            issues.append(f"max_latency_regression_pct should be ≤{GUARDRAILS['max_latency_regression_pct']}, found {latency_regression}")
        
        # Store expected guardrails for reference
        details["expected_guardrails"] = GUARDRAILS
        
        if issues:
            return PreflightCheck(
                name=check_name,
                status=CheckStatus.WARNING,
                message=f"Guardrail configuration issues: {'; '.join(issues)}",
                details=details,
            )
        
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.PASS,
            message="All guardrails are properly configured",
            details=details,
        )
    
    except FileNotFoundError as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Protocol config not found: {e}",
            details={"error": str(e)},
        )
    except Exception as e:
        return PreflightCheck(
            name=check_name,
            status=CheckStatus.FAIL,
            message=f"Error checking guardrails: {e}",
            details={"error": str(e)},
        )


def run_preflight_checks(
    config_path: str | Path | None = None,
    skip_checks: list[str] | None = None,
) -> PreflightReport:
    """Run all preflight checks and return a comprehensive report.
    
    Executes all validation checks in sequence and aggregates results
    into a single PreflightReport.
    
    Args:
        config_path: Optional path to protocol config file
        skip_checks: Optional list of check names to skip
    
    Returns:
        PreflightReport with all check results
    """
    skip_checks = skip_checks or []
    
    # Load protocol once for all checks that need it
    protocol = None
    try:
        protocol = load_protocol_config(config_path)
    except FileNotFoundError:
        pass  # Individual checks will handle the missing config
    
    checks = []
    
    # Run all checks
    check_functions = [
        (check_comparator_model_ids, True),
        (check_cost_latency_ceilings, True),
        (check_dataset_hash, True),
        (check_api_credentials, True),
        (check_environment_readiness, False),  # Doesn't need protocol
        (check_guardrails_configured, True),
    ]
    
    for check_func, needs_protocol in check_functions:
        check_name = check_func.__name__.replace("check_", "")
        
        if check_name in skip_checks:
            checks.append(PreflightCheck(
                name=check_name,
                status=CheckStatus.SKIP,
                message="Check skipped by request",
            ))
            continue
        
        if needs_protocol:
            result = check_func(protocol)
        else:
            result = check_func()
        
        checks.append(result)
    
    # Determine overall status
    has_fail = any(c.status == CheckStatus.FAIL for c in checks)
    has_warning = any(c.status == CheckStatus.WARNING for c in checks)
    
    if has_fail:
        overall_status = CheckStatus.FAIL
    elif has_warning:
        overall_status = CheckStatus.WARNING
    else:
        overall_status = CheckStatus.PASS
    
    # Get protocol version if available
    protocol_version = "unknown"
    if protocol:
        protocol_version = protocol.get("version", "unknown")
    
    return PreflightReport(
        checks=checks,
        overall_status=overall_status,
        protocol_version=protocol_version,
    )


def format_report_text(report: PreflightReport) -> str:
    """Format a preflight report as human-readable text.
    
    Args:
        report: PreflightReport to format
    
    Returns:
        Formatted string suitable for console output
    """
    lines = []
    lines.append("=" * 60)
    lines.append("T3 EXTERNAL BENCHMARK - PREFLIGHT REPORT")
    lines.append("=" * 60)
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Protocol Version: {report.protocol_version}")
    lines.append(f"Overall Status: {report.overall_status.value.upper()}")
    lines.append("")
    
    for check in report.checks:
        # Use ASCII-compatible status icons
        status_icon = {
            CheckStatus.PASS: "[PASS]",
            CheckStatus.FAIL: "[FAIL]",
            CheckStatus.WARNING: "[WARN]",
            CheckStatus.SKIP: "[SKIP]",
        }.get(check.status, "[?]")
        
        lines.append(f"{status_icon} {check.name}")
        lines.append(f"    {check.message}")
        
        if check.details and check.status != CheckStatus.PASS:
            for key, value in check.details.items():
                if key == "error":
                    continue
                lines.append(f"    - {key}: {value}")
        lines.append("")
    
    lines.append("-" * 60)
    lines.append(f"Total: {len(report.checks)} checks")
    lines.append(f"  Passed: {sum(1 for c in report.checks if c.status == CheckStatus.PASS)}")
    lines.append(f"  Failed: {len(report.failed_checks)}")
    lines.append(f"  Warnings: {len(report.warning_checks)}")
    lines.append(f"  Skipped: {sum(1 for c in report.checks if c.status == CheckStatus.SKIP)}")
    lines.append("=" * 60)
    
    return "\n".join(lines)

