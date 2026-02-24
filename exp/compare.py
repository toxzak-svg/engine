from __future__ import annotations

from statistics import mean
from typing import Any

from .models import ComparisonReport, RunResult
from .simulator import evaluate_track_pass, stable_rng


def compare_runs(candidate: RunResult, baseline: RunResult, anchor: RunResult | None = None) -> ComparisonReport:
    delta_metrics = _delta_metrics(candidate.metric_values, baseline.metric_values)
    anchor_delta_metrics: dict[str, float] = {}
    anchor_run_ids: list[str] = []
    if anchor is not None:
        anchor_delta_metrics = _delta_metrics(candidate.metric_values, anchor.metric_values)
        anchor_run_ids = [anchor.run_id]
    parity_pct = _cost_parity_pct(candidate, baseline)
    equal_cost_pass = parity_pct <= 2.0
    fluency_drop_pct = baseline.metric_values["fluency"] - candidate.metric_values["fluency"]
    latency_overhead_pct = _latency_overhead_pct(candidate, baseline)
    stable_training = "unstable_training" not in candidate.failure_flags

    # New metrics calculations
    verbosity_penalty = _verbosity_penalty(candidate, baseline)
    robustness_delta = _robustness_delta(candidate, baseline)
    constraint_adherence_score = _constraint_adherence_score(candidate)

    significance = _bootstrap_significance(candidate, baseline)

    stage_gate_pass = _stage_gate_pass(
        stage=candidate.stage,
        delta_composite=delta_metrics.get("composite", 0.0),
        stable_training=stable_training,
        fluency_drop_pct=fluency_drop_pct,
        latency_overhead_pct=latency_overhead_pct,
        ci_excludes_zero=bool(significance["ci95_excludes_zero"]),
    )

    track_specific_pass = evaluate_track_pass(candidate, baseline, delta_metrics)

    pass_fail: dict[str, Any] = {
        "equal_cost_pass": equal_cost_pass,
        "cost_parity_pct": round(parity_pct, 4),
        "stable_training": stable_training,
        "fluency_drop_pct": round(fluency_drop_pct, 4),
        "latency_overhead_pct": round(latency_overhead_pct, 4),
        "verbosity_penalty": round(verbosity_penalty, 4),
        "robustness_delta": round(robustness_delta, 4),
        "constraint_adherence_score": round(constraint_adherence_score, 4),
        "stage_gate_pass": stage_gate_pass,
        "track_specific_pass": track_specific_pass,
        "overall_pass": bool(equal_cost_pass and stage_gate_pass and track_specific_pass),
    }

    return ComparisonReport(
        candidate_run_ids=[candidate.run_id],
        baseline_run_ids=[baseline.run_id],
        delta_metrics={k: round(v, 4) for k, v in delta_metrics.items()},
        anchor_run_ids=anchor_run_ids,
        anchor_delta_metrics={k: round(v, 4) for k, v in anchor_delta_metrics.items()},
        significance_tests=significance,
        pass_fail=pass_fail,
        candidate_stage=candidate.stage,
        track_id=candidate.track_id,
    )

# Helper functions for new metrics

def _verbosity_penalty(candidate: RunResult, baseline: RunResult) -> float:
    """Calculate verbosity penalty as correctness at fixed output length."""
    candidate_length = candidate.metric_values.get("output_length", 0)
    baseline_length = baseline.metric_values.get("output_length", 0)
    if baseline_length == 0:
        return 0.0
    return (candidate_length - baseline_length) / baseline_length


def _robustness_delta(candidate: RunResult, baseline: RunResult) -> float:
    """Calculate robustness delta: fraction of tasks where counterfactual audit flips incorrect → correct."""
    candidate_robustness = candidate.metric_values.get("robustness", 0)
    baseline_robustness = baseline.metric_values.get("robustness", 0)
    return candidate_robustness - baseline_robustness


def _constraint_adherence_score(candidate: RunResult) -> float:
    """Calculate constraint adherence score."""
    return candidate.metric_values.get("constraint_adherence", 0)


def _delta_metrics(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    shared = sorted(set(candidate) & set(baseline))
    return {key: candidate[key] - baseline[key] for key in shared}


def _cost_parity_pct(candidate: RunResult, baseline: RunResult) -> float:
    candidate_total = candidate.train_cost + candidate.infer_cost
    baseline_total = baseline.train_cost + baseline.infer_cost
    if baseline_total <= 0:
        return 100.0
    return abs(candidate_total - baseline_total) / baseline_total * 100.0


def _latency_overhead_pct(candidate: RunResult, baseline: RunResult) -> float:
    if baseline.latency_p50 <= 0:
        return 100.0
    return (candidate.latency_p50 - baseline.latency_p50) / baseline.latency_p50 * 100.0


def _bootstrap_significance(candidate: RunResult, baseline: RunResult) -> dict[str, Any]:
    benchmark_keys = sorted(set(candidate.benchmark_scores) & set(baseline.benchmark_scores))
    deltas = [candidate.benchmark_scores[key] - baseline.benchmark_scores[key] for key in benchmark_keys]
    if not deltas:
        return {
            "method": "bootstrap_mean_delta",
            "n_resamples": 0,
            "ci95": [0.0, 0.0],
            "ci95_excludes_zero": False,
            "mean_delta": 0.0,
        }

    rng = stable_rng("bootstrap", candidate.run_id, baseline.run_id)
    n_resamples = 2000
    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [deltas[rng.randrange(0, len(deltas))] for _ in deltas]
        samples.append(mean(resample))
    samples.sort()
    lower = samples[int(0.025 * (n_resamples - 1))]
    upper = samples[int(0.975 * (n_resamples - 1))]
    excludes_zero = not (lower <= 0.0 <= upper)
    return {
        "method": "bootstrap_mean_delta",
        "n_resamples": n_resamples,
        "ci95": [round(lower, 4), round(upper, 4)],
        "ci95_excludes_zero": excludes_zero,
        "mean_delta": round(mean(deltas), 4),
        "benchmark_count": len(benchmark_keys),
    }


def _stage_gate_pass(
    stage: int,
    delta_composite: float,
    stable_training: bool,
    fluency_drop_pct: float,
    latency_overhead_pct: float,
    ci_excludes_zero: bool,
) -> bool:
    if stage == 1:
        return delta_composite >= 3.0 and stable_training and fluency_drop_pct <= 2.0
    if stage == 2:
        return delta_composite >= 5.0 and latency_overhead_pct <= 15.0
    if stage in {3, 4}:
        return delta_composite >= 8.0 and ci_excludes_zero
    return False
