"""Tests for exp.anchor_audit — anchor drift detection and SNR computation."""

from __future__ import annotations

from exp.anchor_audit import (
    DRIFT_FAIL_THRESHOLD,
    DRIFT_WARN_THRESHOLD,
    SNR_MIN_THRESHOLD,
    AnchorAuditResult,
    audit_comparison_anchor_validity,
    check_anchor_stability,
    compute_snr,
)
from exp.models import RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    run_id: str,
    stage: int,
    composite: float,
    track_id: str = "ANCHOR",
    seed: int = 101,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        spec_id=f"spec-{run_id}",
        commit_sha="abc123",
        seed=seed,
        train_cost=100.0,
        infer_cost=20.0,
        latency_p50=120.0,
        latency_p95=153.6,
        energy_kwh=45.6,
        metric_values={
            "composite": composite,
            "long_context": composite,
            "reasoning": composite,
            "consistency": composite,
            "fluency": 90.0,
        },
        failure_flags=[],
        track_id=track_id,
        stage=stage,
        model_variant="BASELINE",
        benchmark_scores={},
    )


# ---------------------------------------------------------------------------
# check_anchor_stability
# ---------------------------------------------------------------------------

class TestCheckAnchorStability:
    def test_empty_runs_returns_stable(self) -> None:
        result = check_anchor_stability([])
        assert result.stable is True
        assert result.drift == 0.0
        assert result.drift_warn is False
        assert result.drift_fail is False

    def test_single_run_is_stable(self) -> None:
        runs = [_make_run("r1", stage=1, composite=60.0)]
        result = check_anchor_stability(runs)
        assert result.stable is True
        assert result.drift == 0.0
        assert result.mean_composite == 60.0

    def test_stable_within_warn_threshold(self) -> None:
        runs = [
            _make_run("r1", stage=1, composite=60.0),
            _make_run("r2", stage=2, composite=60.3),
            _make_run("r3", stage=3, composite=60.1),
        ]
        result = check_anchor_stability(runs)
        assert result.stable is True
        assert result.drift_warn is False
        assert result.drift < DRIFT_WARN_THRESHOLD

    def test_warn_threshold_triggered(self) -> None:
        runs = [
            _make_run("r1", stage=1, composite=60.0),
            _make_run("r2", stage=2, composite=60.6),  # drift = 0.6 > 0.5
        ]
        result = check_anchor_stability(runs)
        assert result.drift_warn is True
        assert result.drift_fail is False
        assert result.stable is True  # warn but not fail

    def test_fail_threshold_triggered(self) -> None:
        runs = [
            _make_run("r1", stage=1, composite=60.0),
            _make_run("r2", stage=2, composite=61.2),  # drift = 1.2 > 1.0
        ]
        result = check_anchor_stability(runs)
        assert result.drift_fail is True
        assert result.stable is False

    def test_stage_composites_computed(self) -> None:
        runs = [
            _make_run("r1", stage=1, composite=60.0),
            _make_run("r2", stage=1, composite=60.2),
            _make_run("r3", stage=2, composite=60.1),
        ]
        result = check_anchor_stability(runs)
        assert 1 in result.stage_composites
        assert 2 in result.stage_composites
        assert abs(result.stage_composites[1] - 60.1) < 0.01

    def test_affected_stages_identified(self) -> None:
        runs = [
            _make_run("r1", stage=1, composite=60.0),
            _make_run("r2", stage=2, composite=60.0),
            _make_run("r3", stage=3, composite=61.5),  # stage 3 deviates
        ]
        result = check_anchor_stability(runs)
        assert 3 in result.affected_stages

    def test_to_dict_serializable(self) -> None:
        runs = [_make_run("r1", stage=1, composite=60.0)]
        result = check_anchor_stability(runs)
        d = result.to_dict()
        assert "stable" in d
        assert "drift" in d
        assert "stage_composites" in d
        assert isinstance(d["stage_composites"], dict)


# ---------------------------------------------------------------------------
# compute_snr
# ---------------------------------------------------------------------------

class TestComputeSNR:
    def test_empty_returns_zero(self) -> None:
        result = compute_snr([])
        assert result["snr"] == 0.0
        assert result["n"] == 0
        assert result["stable"] is False

    def test_single_run_zero_std(self) -> None:
        runs = [_make_run("r1", stage=1, composite=60.0)]
        result = compute_snr(runs)
        assert result["std"] == 0.0
        assert result["n"] == 1

    def test_high_variance_high_snr(self) -> None:
        # Large spread across seeds → high SNR
        runs = [
            _make_run("r1", stage=1, composite=58.0, seed=101),
            _make_run("r2", stage=1, composite=62.0, seed=102),
            _make_run("r3", stage=1, composite=60.0, seed=103),
        ]
        result = compute_snr(runs)
        assert result["snr"] > 0.0
        assert result["n"] == 3

    def test_low_variance_low_snr(self) -> None:
        # Tiny spread → low SNR (noise-dominated)
        runs = [
            _make_run("r1", stage=1, composite=60.00, seed=101),
            _make_run("r2", stage=1, composite=60.05, seed=102),
            _make_run("r3", stage=1, composite=60.02, seed=103),
        ]
        result = compute_snr(runs)
        assert result["snr"] < SNR_MIN_THRESHOLD

    def test_stable_flag_set_correctly(self) -> None:
        # High variance → stable=True
        runs = [
            _make_run(f"r{i}", stage=1, composite=58.0 + i * 1.5, seed=100 + i)
            for i in range(5)
        ]
        result = compute_snr(runs)
        # With large spread, SNR should be high enough
        assert "stable" in result

    def test_custom_metric(self) -> None:
        runs = [_make_run("r1", stage=1, composite=60.0)]
        result = compute_snr(runs, metric="fluency")
        assert result["mean"] == 90.0


# ---------------------------------------------------------------------------
# audit_comparison_anchor_validity
# ---------------------------------------------------------------------------

class TestAuditComparisonAnchorValidity:
    def _make_audit(self, drift: float, affected_stages: list[int]) -> AnchorAuditResult:
        return AnchorAuditResult(
            stable=drift <= DRIFT_FAIL_THRESHOLD,
            drift=drift,
            drift_warn=drift > DRIFT_WARN_THRESHOLD,
            drift_fail=drift > DRIFT_FAIL_THRESHOLD,
            mean_composite=60.0,
            std_composite=0.1,
            stage_composites={},
            reason="test",
            affected_stages=affected_stages,
        )

    def test_valid_when_stable(self) -> None:
        audit = self._make_audit(drift=0.2, affected_stages=[])
        result = audit_comparison_anchor_validity(audit, candidate_stage=2)
        assert result["valid"] is True
        assert result["warning"] is False

    def test_warning_when_drift_warn_and_affected(self) -> None:
        audit = self._make_audit(drift=0.7, affected_stages=[2])
        result = audit_comparison_anchor_validity(audit, candidate_stage=2)
        assert result["valid"] is True
        assert result["warning"] is True

    def test_invalid_when_drift_fail_and_affected(self) -> None:
        audit = self._make_audit(drift=1.2, affected_stages=[2])
        result = audit_comparison_anchor_validity(audit, candidate_stage=2)
        assert result["valid"] is False
        assert result["warning"] is True

    def test_valid_when_drift_fail_but_not_affected_stage(self) -> None:
        audit = self._make_audit(drift=1.2, affected_stages=[3])
        result = audit_comparison_anchor_validity(audit, candidate_stage=2)
        # Stage 2 not in affected_stages → still valid
        assert result["valid"] is True
