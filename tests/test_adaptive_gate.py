"""Tests for exp.adaptive_gate — Bayesian adaptive gate calibration."""

from __future__ import annotations

from exp.adaptive_gate import (
    FIXED_THRESHOLDS,
    calibrate_all_stages,
    calibrate_gate_threshold,
    format_adaptive_gate_report,
    _beta_cdf,  # type: ignore[attr-defined]
    _beta_ppf,  # type: ignore[attr-defined]
)
from exp.models import ComparisonReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(
    stage: int,
    overall_pass: bool,
    composite_delta: float = 5.0,
    track_id: str = "T3",
) -> ComparisonReport:
    return ComparisonReport(
        candidate_run_ids=[f"run-{stage}-{overall_pass}-{composite_delta}"],
        baseline_run_ids=["baseline-run"],
        delta_metrics={"composite": composite_delta},
        significance_tests={"ci95": [0.1, 2.0], "ci95_excludes_zero": True},
        pass_fail={"overall_pass": overall_pass},
        candidate_stage=stage,
        track_id=track_id,
        anchor_run_ids=[],
        anchor_delta_metrics={"composite": composite_delta},
    )


def _make_reports(stage: int, n_pass: int, n_fail: int) -> list[ComparisonReport]:
    reports: list[ComparisonReport] = []
    for i in range(n_pass):
        reports.append(_make_report(stage, overall_pass=True, composite_delta=6.0 + i * 0.1))
    for i in range(n_fail):
        reports.append(_make_report(stage, overall_pass=False, composite_delta=2.0 + i * 0.1))
    return reports


# ---------------------------------------------------------------------------
# Beta distribution utilities
# ---------------------------------------------------------------------------

class TestBetaDistribution:
    def test_cdf_at_zero_is_zero(self) -> None:
        assert _beta_cdf(0.0, 2.0, 2.0) == 0.0

    def test_cdf_at_one_is_one(self) -> None:
        assert _beta_cdf(1.0, 2.0, 2.0) == 1.0

    def test_cdf_symmetric_at_half_for_symmetric_beta(self) -> None:
        # Beta(2,2) is symmetric → CDF(0.5) = 0.5
        val = _beta_cdf(0.5, 2.0, 2.0)
        assert abs(val - 0.5) < 0.01

    def test_ppf_inverse_of_cdf(self) -> None:
        # ppf(cdf(x)) ≈ x
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cdf_val = _beta_cdf(x, 3.0, 2.0)
            recovered = _beta_ppf(cdf_val, 3.0, 2.0)
            assert abs(recovered - x) < 0.01, f"ppf(cdf({x})) = {recovered}, expected ~{x}"

    def test_ppf_at_zero_is_zero(self) -> None:
        assert _beta_ppf(0.0, 2.0, 2.0) == 0.0

    def test_ppf_at_one_is_one(self) -> None:
        assert _beta_ppf(1.0, 2.0, 2.0) == 1.0

    def test_ppf_p80_greater_than_p50(self) -> None:
        p50 = _beta_ppf(0.5, 3.0, 2.0)
        p80 = _beta_ppf(0.8, 3.0, 2.0)
        assert p80 > p50


# ---------------------------------------------------------------------------
# calibrate_gate_threshold
# ---------------------------------------------------------------------------

class TestCalibrateGateThreshold:
    def test_insufficient_history_returns_fixed(self) -> None:
        reports = _make_reports(stage=2, n_pass=2, n_fail=1)  # < MIN_HISTORY
        config = calibrate_gate_threshold(stage=2, historical_reports=reports)
        assert config.composite_threshold == FIXED_THRESHOLDS[2]
        assert config.adapted is False
        assert config.n_historical == 3

    def test_sufficient_history_may_adapt(self) -> None:
        # 8 passes, 2 fails → strong program → threshold should rise
        reports = _make_reports(stage=2, n_pass=8, n_fail=2)
        config = calibrate_gate_threshold(stage=2, historical_reports=reports)
        assert config.n_historical == 10
        assert config.composite_threshold >= FIXED_THRESHOLDS[2] * 0.7  # within bounds

    def test_high_pass_rate_raises_threshold(self) -> None:
        # All passes → strong program → threshold should be >= fixed
        reports = _make_reports(stage=2, n_pass=10, n_fail=0)
        config = calibrate_gate_threshold(stage=2, historical_reports=reports)
        assert config.composite_threshold >= FIXED_THRESHOLDS[2]

    def test_low_pass_rate_lowers_threshold(self) -> None:
        # All fails → struggling program → threshold should be <= fixed
        reports = _make_reports(stage=2, n_pass=0, n_fail=10)
        config = calibrate_gate_threshold(stage=2, historical_reports=reports)
        assert config.composite_threshold <= FIXED_THRESHOLDS[2]

    def test_threshold_within_bounds(self) -> None:
        # Threshold should always be within [0.7*fixed, 1.5*fixed]
        for n_pass, n_fail in [(10, 0), (5, 5), (0, 10)]:
            reports = _make_reports(stage=2, n_pass=n_pass, n_fail=n_fail)
            config = calibrate_gate_threshold(stage=2, historical_reports=reports)
            fixed = FIXED_THRESHOLDS[2]
            assert config.composite_threshold >= fixed * 0.7
            assert config.composite_threshold <= fixed * 1.5

    def test_stage_1_uses_correct_fixed_threshold(self) -> None:
        reports = _make_reports(stage=1, n_pass=2, n_fail=1)
        config = calibrate_gate_threshold(stage=1, historical_reports=reports)
        assert config.fixed_threshold == FIXED_THRESHOLDS[1]

    def test_stage_3_uses_correct_fixed_threshold(self) -> None:
        reports = _make_reports(stage=3, n_pass=2, n_fail=1)
        config = calibrate_gate_threshold(stage=3, historical_reports=reports)
        assert config.fixed_threshold == FIXED_THRESHOLDS[3]

    def test_filters_to_correct_stage(self) -> None:
        # Mix of stage 1 and stage 2 reports
        s1_reports = _make_reports(stage=1, n_pass=5, n_fail=5)
        s2_reports = _make_reports(stage=2, n_pass=8, n_fail=2)
        all_reports = s1_reports + s2_reports
        config = calibrate_gate_threshold(stage=2, historical_reports=all_reports)
        assert config.n_historical == 10  # only stage 2 reports counted

    def test_to_dict_has_required_keys(self) -> None:
        reports = _make_reports(stage=2, n_pass=3, n_fail=2)
        config = calibrate_gate_threshold(stage=2, historical_reports=reports)
        d = config.to_dict()
        for key in ["stage", "composite_threshold", "fixed_threshold", "adapted", "confidence"]:
            assert key in d

    def test_confidence_increases_with_more_history(self) -> None:
        small = _make_reports(stage=2, n_pass=3, n_fail=2)
        large = _make_reports(stage=2, n_pass=15, n_fail=5)
        config_small = calibrate_gate_threshold(stage=2, historical_reports=small)
        config_large = calibrate_gate_threshold(stage=2, historical_reports=large)
        assert config_large.confidence >= config_small.confidence

    def test_posterior_mean_reflects_pass_rate(self) -> None:
        # 8 passes, 2 fails → alpha=9, beta_param=3
        # posterior mean = alpha / (alpha + beta_param) = 9 / 12 = 0.75
        reports = _make_reports(stage=2, n_pass=8, n_fail=2)
        config = calibrate_gate_threshold(stage=2, historical_reports=reports)
        expected_mean = 9.0 / 12.0  # (passes+1) / (passes+1 + fails+1)
        assert abs(config.pass_rate_posterior_mean - expected_mean) < 0.01


# ---------------------------------------------------------------------------
# calibrate_all_stages
# ---------------------------------------------------------------------------

class TestCalibrateAllStages:
    def test_returns_all_four_stages(self) -> None:
        reports = (
            _make_reports(stage=1, n_pass=3, n_fail=2)
            + _make_reports(stage=2, n_pass=3, n_fail=2)
            + _make_reports(stage=3, n_pass=3, n_fail=2)
            + _make_reports(stage=4, n_pass=3, n_fail=2)
        )
        configs = calibrate_all_stages(reports)
        assert set(configs.keys()) == {1, 2, 3, 4}

    def test_each_stage_uses_correct_fixed_threshold(self) -> None:
        configs = calibrate_all_stages([])
        for stage in [1, 2, 3, 4]:
            assert configs[stage].fixed_threshold == FIXED_THRESHOLDS[stage]


# ---------------------------------------------------------------------------
# format_adaptive_gate_report
# ---------------------------------------------------------------------------

class TestFormatAdaptiveGateReport:
    def test_returns_markdown_string(self) -> None:
        configs = calibrate_all_stages([])
        report = format_adaptive_gate_report(configs)
        assert "## Adaptive Gate Calibration" in report
        assert "Stage" in report

    def test_all_stages_present_in_report(self) -> None:
        configs = calibrate_all_stages([])
        report = format_adaptive_gate_report(configs)
        for stage in [1, 2, 3, 4]:
            assert f"| {stage} |" in report

    def test_reasoning_section_present(self) -> None:
        configs = calibrate_all_stages([])
        report = format_adaptive_gate_report(configs)
        assert "### Calibration Reasoning" in report
