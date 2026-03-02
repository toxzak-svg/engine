"""Bayesian adaptive gate calibration.

Replaces hardcoded gate thresholds (delta >= 3/5/8) with data-driven
thresholds derived from the posterior distribution of historical pass rates.

Algorithm:
    1. Collect historical pass/fail outcomes for a given stage.
    2. Fit a Beta(alpha, beta) distribution over the pass rate.
       - alpha = number of passes + 1 (Laplace smoothing)
       - beta  = number of fails  + 1
    3. The adaptive threshold = percentile of the posterior predictive
       distribution of composite deltas, weighted by pass probability.
    4. Optionally: use the 80th percentile of the Beta posterior as the
       "expected pass rate" and scale the composite threshold accordingly.

Rationale:
    - A program consistently producing strong tracks gets harder gates
      (the posterior shifts right → higher threshold).
    - A struggling program gets fair gates (posterior stays near 0.5).
    - Eliminates the "gaming the threshold" problem where teams tune
      experiments to just barely clear a fixed bar.

Usage:
    from exp.adaptive_gate import calibrate_gate_threshold, AdaptiveGateConfig

    config = calibrate_gate_threshold(stage=2, historical_reports=reports)
    print(f"Adaptive threshold: delta_composite >= {config.composite_threshold:.2f}")
    print(f"Confidence: {config.confidence:.0%}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .models import ComparisonReport


# Fixed baseline thresholds (current hardcoded values)
FIXED_THRESHOLDS: dict[int, float] = {
    1: 3.0,
    2: 5.0,
    3: 8.0,
    4: 8.0,
}

# Adaptive gate configuration
ADAPTIVE_PERCENTILE = 0.80       # Use 80th percentile of Beta posterior
MIN_HISTORY_FOR_ADAPTATION = 5  # Minimum reports before adapting (else use fixed)
ADAPTATION_STRENGTH = 0.5       # Blend factor: 0=always fixed, 1=always adaptive


@dataclass(frozen=True)
class AdaptiveGateConfig:
    """Result of adaptive gate calibration for a stage.

    Attributes:
        stage: Stage number.
        composite_threshold: Calibrated delta_composite threshold.
        fixed_threshold: Original hardcoded threshold for comparison.
        pass_rate_posterior_mean: Mean of Beta posterior over pass rate.
        pass_rate_posterior_p80: 80th percentile of Beta posterior.
        alpha: Beta distribution alpha parameter (passes + 1).
        beta_param: Beta distribution beta parameter (fails + 1).
        n_historical: Number of historical reports used.
        adapted: True if adaptive threshold differs from fixed.
        confidence: Confidence in the adaptive threshold (0–1).
        reasoning: Human-readable explanation of the calibration.
    """
    stage: int
    composite_threshold: float
    fixed_threshold: float
    pass_rate_posterior_mean: float
    pass_rate_posterior_p80: float
    alpha: float
    beta_param: float
    n_historical: int
    adapted: bool
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "composite_threshold": round(self.composite_threshold, 4),
            "fixed_threshold": round(self.fixed_threshold, 4),
            "pass_rate_posterior_mean": round(self.pass_rate_posterior_mean, 4),
            "pass_rate_posterior_p80": round(self.pass_rate_posterior_p80, 4),
            "alpha": round(self.alpha, 4),
            "beta_param": round(self.beta_param, 4),
            "n_historical": self.n_historical,
            "adapted": self.adapted,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
        }


def calibrate_gate_threshold(
    stage: int,
    historical_reports: list[ComparisonReport],
    percentile: float = ADAPTIVE_PERCENTILE,
    adaptation_strength: float = ADAPTATION_STRENGTH,
) -> AdaptiveGateConfig:
    """Calibrate the gate threshold for a stage using historical pass rates.

    Args:
        stage: Stage number (1–4).
        historical_reports: All ComparisonReport objects for this stage
                            from previous program runs.
        percentile: Percentile of Beta posterior to use as threshold signal.
                    Default 0.80 (80th percentile).
        adaptation_strength: Blend factor between fixed and adaptive threshold.
                             0.0 = always use fixed; 1.0 = always use adaptive.

    Returns:
        AdaptiveGateConfig with calibrated threshold and diagnostics.
    """
    fixed = FIXED_THRESHOLDS.get(stage, 5.0)

    stage_reports = [r for r in historical_reports if r.candidate_stage == stage]
    n = len(stage_reports)

    if n < MIN_HISTORY_FOR_ADAPTATION:
        return AdaptiveGateConfig(
            stage=stage,
            composite_threshold=fixed,
            fixed_threshold=fixed,
            pass_rate_posterior_mean=0.5,
            pass_rate_posterior_p80=0.5,
            alpha=1.0,
            beta_param=1.0,
            n_historical=n,
            adapted=False,
            confidence=0.0,
            reasoning=(
                f"Insufficient history ({n} < {MIN_HISTORY_FOR_ADAPTATION} reports). "
                f"Using fixed threshold: delta_composite >= {fixed:.1f}."
            ),
        )

    # Fit Beta distribution
    passes = sum(1 for r in stage_reports if r.pass_fail.get("overall_pass", False))
    fails = n - passes
    alpha = float(passes + 1)   # Laplace smoothing
    beta_param = float(fails + 1)

    posterior_mean = alpha / (alpha + beta_param)
    posterior_p80 = _beta_ppf(percentile, alpha, beta_param)

    # Compute mean composite delta across historical reports
    composite_deltas = [
        r.anchor_delta_metrics.get("composite", r.delta_metrics.get("composite", 0.0))
        for r in stage_reports
    ]
    mean_delta = mean(composite_deltas) if composite_deltas else fixed

    # Adaptive threshold: scale fixed threshold by posterior pass rate signal
    # High pass rate → program is strong → raise threshold
    # Low pass rate  → program is struggling → lower threshold toward fixed
    # Formula: adaptive = fixed * (1 + (posterior_p80 - 0.5) * scale_factor)
    scale_factor = 0.4  # ±20% max adjustment at p80=0.9 or p80=0.1
    raw_adaptive = fixed * (1.0 + (posterior_p80 - 0.5) * scale_factor)

    # Blend with fixed threshold
    composite_threshold = (
        (1.0 - adaptation_strength) * fixed
        + adaptation_strength * raw_adaptive
    )
    composite_threshold = round(max(fixed * 0.7, min(fixed * 1.5, composite_threshold)), 2)

    adapted = abs(composite_threshold - fixed) > 0.05
    confidence = min(1.0, n / (MIN_HISTORY_FOR_ADAPTATION * 4))

    if adapted:
        direction = "raised" if composite_threshold > fixed else "lowered"
        reasoning = (
            f"Adaptive calibration ({n} historical reports): "
            f"pass rate posterior mean={posterior_mean:.0%}, p{int(percentile*100)}={posterior_p80:.0%}. "
            f"Threshold {direction} from {fixed:.1f} → {composite_threshold:.2f} "
            f"(adaptation_strength={adaptation_strength:.0%}, confidence={confidence:.0%}). "
            f"Historical mean delta: {mean_delta:.3f}."
        )
    else:
        reasoning = (
            f"Adaptive calibration ({n} historical reports): "
            f"pass rate posterior mean={posterior_mean:.0%}. "
            f"Threshold unchanged at {fixed:.1f} (adjustment < 0.05 pts). "
            f"Historical mean delta: {mean_delta:.3f}."
        )

    return AdaptiveGateConfig(
        stage=stage,
        composite_threshold=composite_threshold,
        fixed_threshold=fixed,
        pass_rate_posterior_mean=round(posterior_mean, 4),
        pass_rate_posterior_p80=round(posterior_p80, 4),
        alpha=alpha,
        beta_param=beta_param,
        n_historical=n,
        adapted=adapted,
        confidence=round(confidence, 4),
        reasoning=reasoning,
    )


def calibrate_all_stages(
    historical_reports: list[ComparisonReport],
    percentile: float = ADAPTIVE_PERCENTILE,
    adaptation_strength: float = ADAPTATION_STRENGTH,
) -> dict[int, AdaptiveGateConfig]:
    """Calibrate gate thresholds for all stages simultaneously.

    Args:
        historical_reports: All historical ComparisonReport objects.
        percentile: Percentile of Beta posterior to use.
        adaptation_strength: Blend factor between fixed and adaptive.

    Returns:
        Dict mapping stage → AdaptiveGateConfig.
    """
    return {
        stage: calibrate_gate_threshold(
            stage=stage,
            historical_reports=historical_reports,
            percentile=percentile,
            adaptation_strength=adaptation_strength,
        )
        for stage in [1, 2, 3, 4]
    }


def format_adaptive_gate_report(configs: dict[int, AdaptiveGateConfig]) -> str:
    """Format adaptive gate calibration results as a markdown section.

    Args:
        configs: Output of calibrate_all_stages().

    Returns:
        Markdown string with calibration table and reasoning.
    """
    lines: list[str] = []
    lines.append("## Adaptive Gate Calibration")
    lines.append("")
    lines.append(
        f"Thresholds calibrated using Beta posterior over historical pass rates "
        f"(percentile={ADAPTIVE_PERCENTILE:.0%}, "
        f"adaptation_strength={ADAPTATION_STRENGTH:.0%}, "
        f"min_history={MIN_HISTORY_FOR_ADAPTATION})."
    )
    lines.append("")
    lines.append(
        "| Stage | Fixed Threshold | Adaptive Threshold | Change | "
        "Pass Rate (mean) | Pass Rate (p80) | History | Adapted | Confidence |"
    )
    lines.append("|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|")

    for stage in sorted(configs.keys()):
        cfg = configs[stage]
        change = cfg.composite_threshold - cfg.fixed_threshold
        change_str = f"{change:+.2f}" if cfg.adapted else "—"
        adapted_icon = "✅" if cfg.adapted else "—"
        lines.append(
            f"| {stage} | {cfg.fixed_threshold:.1f} | {cfg.composite_threshold:.2f} | "
            f"{change_str} | {cfg.pass_rate_posterior_mean*100:.0f}% | "
            f"{cfg.pass_rate_posterior_p80*100:.0f}% | {cfg.n_historical} | "
            f"{adapted_icon} | {cfg.confidence*100:.0f}% |"
        )

    lines.append("")
    lines.append("### Calibration Reasoning")
    lines.append("")
    for stage in sorted(configs.keys()):
        cfg = configs[stage]
        lines.append(f"**Stage {stage}**: {cfg.reasoning}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Beta distribution utilities (no scipy dependency)
# ---------------------------------------------------------------------------

def _beta_ppf(p: float, alpha: float, beta: float) -> float:
    """Approximate Beta(alpha, beta) percent-point function (inverse CDF).

    Uses a numerical bisection method. Accurate to ~4 decimal places for
    typical alpha/beta values encountered in experiment gating (1–50).

    Args:
        p: Probability (0–1).
        alpha: Beta distribution alpha parameter.
        beta: Beta distribution beta parameter.

    Returns:
        x such that P(X <= x) = p for X ~ Beta(alpha, beta).
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    # Bisection search
    lo, hi = 0.0, 1.0
    for _ in range(64):  # 64 iterations → ~1e-19 precision
        mid = (lo + hi) / 2.0
        if _beta_cdf(mid, alpha, beta) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _beta_cdf(x: float, alpha: float, beta: float) -> float:
    """Regularised incomplete Beta function I_x(alpha, beta).

    Uses the continued fraction expansion (Lentz's method) for accuracy.

    Args:
        x: Value in [0, 1].
        alpha: Alpha parameter.
        beta: Beta parameter.

    Returns:
        I_x(alpha, beta) = P(X <= x) for X ~ Beta(alpha, beta).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use symmetry relation for numerical stability when x > (alpha)/(alpha+beta)
    if x > (alpha / (alpha + beta)):
        return 1.0 - _beta_cdf(1.0 - x, beta, alpha)

    # Continued fraction via Lentz's method
    lbeta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    front = math.exp(math.log(x) * alpha + math.log(1.0 - x) * beta - lbeta) / alpha

    # Evaluate continued fraction
    cf = _beta_cf(x, alpha, beta)
    return front * cf


def _beta_cf(x: float, alpha: float, beta: float, max_iter: int = 200, tol: float = 1e-10) -> float:
    """Continued fraction for the regularised incomplete Beta function."""
    # Lentz's algorithm (modified form from Numerical Recipes)
    tiny = 1e-30
    f = tiny
    c = 1.0   # NOTE: c must start at 1.0, not tiny
    d = 1.0 - (alpha + beta) * x / (alpha + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        m2 = 2 * m
        num = m * (beta - m) * x / ((alpha + m2 - 1.0) * (alpha + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= d * c

        # Odd step
        num = -(alpha + m) * (alpha + beta + m) * x / ((alpha + m2) * (alpha + m2 + 1.0))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    return f
