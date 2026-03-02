"""Anchor drift detection and stability auditing.

The anchor baseline is the foundation of all anchor-relative delta metrics.
Silent drift in the anchor across stages invalidates every comparison report
that uses anchor_delta_metrics. This module detects and flags such drift.

Usage:
    from exp.anchor_audit import check_anchor_stability, AnchorAuditResult

    result = check_anchor_stability(anchor_runs)
    if not result.stable:
        print(f"WARNING: anchor drifted {result.drift:.3f} pts — {result.reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any

from .models import RunResult


# Thresholds
DRIFT_WARN_THRESHOLD = 0.5   # composite pts — warn
DRIFT_FAIL_THRESHOLD = 1.0   # composite pts — fail / invalidate
SNR_MIN_THRESHOLD = 2.0      # signal-to-noise ratio minimum for Stage 3


@dataclass(frozen=True)
class AnchorAuditResult:
    """Result of an anchor stability audit.

    Attributes:
        stable: True if anchor drift is within acceptable bounds.
        drift: Max composite delta across all anchor runs (max - min).
        drift_warn: True if drift exceeds DRIFT_WARN_THRESHOLD.
        drift_fail: True if drift exceeds DRIFT_FAIL_THRESHOLD (invalidates comparisons).
        mean_composite: Mean composite score across all anchor runs.
        std_composite: Std dev of composite across anchor runs.
        stage_composites: Per-stage mean composite values.
        reason: Human-readable explanation.
        affected_stages: Stages whose anchor-relative deltas may be unreliable.
    """
    stable: bool
    drift: float
    drift_warn: bool
    drift_fail: bool
    mean_composite: float
    std_composite: float
    stage_composites: dict[int, float]
    reason: str
    affected_stages: list[int] = field(default_factory=lambda: [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable": self.stable,
            "drift": round(self.drift, 4),
            "drift_warn": self.drift_warn,
            "drift_fail": self.drift_fail,
            "mean_composite": round(self.mean_composite, 4),
            "std_composite": round(self.std_composite, 4),
            "stage_composites": {str(k): round(v, 4) for k, v in self.stage_composites.items()},
            "reason": self.reason,
            "affected_stages": self.affected_stages,
        }


def check_anchor_stability(anchor_runs: list[RunResult]) -> AnchorAuditResult:
    """Check whether the anchor baseline has drifted across stages.

    Args:
        anchor_runs: All RunResult objects for the ANCHOR track, across all stages.

    Returns:
        AnchorAuditResult with drift metrics and stability verdict.

    Example:
        >>> result = check_anchor_stability(anchor_runs)
        >>> if result.drift_fail:
        ...     raise RuntimeError(f"Anchor invalidated: {result.reason}")
    """
    if not anchor_runs:
        return AnchorAuditResult(
            stable=True,
            drift=0.0,
            drift_warn=False,
            drift_fail=False,
            mean_composite=0.0,
            std_composite=0.0,
            stage_composites={},
            reason="No anchor runs provided — cannot assess stability.",
            affected_stages=[],
        )

    composites = [r.metric_values.get("composite", 0.0) for r in anchor_runs]
    mean_c = mean(composites)
    std_c = stdev(composites) if len(composites) > 1 else 0.0
    drift = max(composites) - min(composites)

    # Per-stage breakdown
    stage_composites: dict[int, list[float]] = {}
    for run in anchor_runs:
        stage_composites.setdefault(run.stage, []).append(
            run.metric_values.get("composite", 0.0)
        )
    stage_means: dict[int, float] = {
        stage: mean(vals) for stage, vals in stage_composites.items()
    }

    # Detect which stages are affected (stages where anchor deviates > warn threshold from mean)
    affected_stages = [
        stage for stage, val in stage_means.items()
        if abs(val - mean_c) > DRIFT_WARN_THRESHOLD
    ]

    drift_warn = drift > DRIFT_WARN_THRESHOLD
    drift_fail = drift > DRIFT_FAIL_THRESHOLD
    stable = not drift_fail

    if drift_fail:
        reason = (
            f"CRITICAL: Anchor composite drifted {drift:.3f} pts (threshold={DRIFT_FAIL_THRESHOLD}). "
            f"All anchor-relative deltas for stages {sorted(affected_stages)} are unreliable. "
            f"Stage composites: {_format_stage_composites(stage_means)}."
        )
    elif drift_warn:
        reason = (
            f"WARNING: Anchor composite drifted {drift:.3f} pts (warn threshold={DRIFT_WARN_THRESHOLD}). "
            f"Monitor closely. Stage composites: {_format_stage_composites(stage_means)}."
        )
    else:
        reason = (
            f"OK: Anchor composite stable within {drift:.3f} pts across {len(anchor_runs)} runs. "
            f"Stage composites: {_format_stage_composites(stage_means)}."
        )

    return AnchorAuditResult(
        stable=stable,
        drift=round(drift, 4),
        drift_warn=drift_warn,
        drift_fail=drift_fail,
        mean_composite=round(mean_c, 4),
        std_composite=round(std_c, 4),
        stage_composites=stage_means,
        reason=reason,
        affected_stages=sorted(affected_stages),
    )


def _format_stage_composites(stage_means: dict[int, float]) -> str:
    return ", ".join(
        f"S{stage}={val:.3f}" for stage, val in sorted(stage_means.items())
    )


def compute_snr(runs: list[RunResult], metric: str = "composite") -> dict[str, float]:
    """Compute signal-to-noise ratio across seeds for a set of runs.

    Decomposes total variance into:
      - Between-seed variance (signal): real effect heterogeneity
      - Within-seed variance (noise): initialization sensitivity

    For a single set of runs (same spec, different seeds), between-seed
    variance IS the total variance. We approximate within-seed variance
    using the jitter bounds from the simulator (±0.3 pts for long_context,
    ±0.2 for reasoning/consistency).

    Args:
        runs: RunResult objects (same track/spec, different seeds).
        metric: Metric key to compute SNR for (default: "composite").

    Returns:
        Dict with keys: mean, std, snr, n, stable (snr >= SNR_MIN_THRESHOLD).
    """
    if not runs:
        return {"mean": 0.0, "std": 0.0, "snr": 0.0, "n": 0, "stable": False}

    values = [r.metric_values.get(metric, 0.0) for r in runs]
    n = len(values)
    mean_val = mean(values)
    std_val = stdev(values) if n > 1 else 0.0

    # Approximate within-seed noise floor from simulator jitter bounds
    # long_context jitter: ±0.3 → std ≈ 0.3/√3 ≈ 0.17
    # reasoning/consistency jitter: ±0.2 → std ≈ 0.12
    # composite = 0.45*lc + 0.35*r + 0.20*c → noise_floor ≈ 0.13
    noise_floor = 0.13

    signal_variance = max(0.0, std_val**2 - noise_floor**2)
    signal_std = signal_variance**0.5

    snr = signal_std / noise_floor if noise_floor > 0 else float("inf")

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "signal_std": round(signal_std, 4),
        "noise_floor": round(noise_floor, 4),
        "snr": round(snr, 4),
        "n": n,
        "stable": snr >= SNR_MIN_THRESHOLD,
    }


def audit_comparison_anchor_validity(
    anchor_audit: AnchorAuditResult,
    candidate_stage: int,
) -> dict[str, Any]:
    """Check whether anchor-relative deltas for a given stage are valid.

    Args:
        anchor_audit: Result from check_anchor_stability().
        candidate_stage: Stage of the candidate run being compared.

    Returns:
        Dict with keys: valid, warning, message.
    """
    if anchor_audit.drift_fail and candidate_stage in anchor_audit.affected_stages:
        return {
            "valid": False,
            "warning": True,
            "message": (
                f"Anchor-relative deltas for Stage {candidate_stage} are INVALID. "
                f"Anchor drifted {anchor_audit.drift:.3f} pts. "
                "Use stage-baseline-relative deltas instead."
            ),
        }
    if anchor_audit.drift_warn and candidate_stage in anchor_audit.affected_stages:
        return {
            "valid": True,
            "warning": True,
            "message": (
                f"Anchor-relative deltas for Stage {candidate_stage} may be noisy. "
                f"Anchor drifted {anchor_audit.drift:.3f} pts (warn threshold={DRIFT_WARN_THRESHOLD})."
            ),
        }
    return {
        "valid": True,
        "warning": False,
        "message": f"Anchor stable (drift={anchor_audit.drift:.3f} pts). Deltas are reliable.",
    }
