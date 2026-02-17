from __future__ import annotations

import hashlib
import random
from statistics import mean
from typing import Any

from .constants import (
    BASE_BENCHMARK_SCORES,
    COMPOSITE_WEIGHTS,
    CONSISTENCY_DATASETS,
    LONG_CONTEXT_DATASETS,
    QUALITY_DELTA_SCALE,
    REASONING_DATASETS,
    STAGE_GAIN_MULTIPLIER,
    TRACK_SPECIFIC_METRIC_BASELINES,
    VARIANT_EFFECTS,
)
from .models import ExperimentSpec, RunResult


def stable_rng(*parts: object) -> random.Random:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(seed)


def clamp_score(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def compute_composite(long_context: float, reasoning: float, consistency: float) -> float:
    return (
        COMPOSITE_WEIGHTS["long_context"] * long_context
        + COMPOSITE_WEIGHTS["reasoning"] * reasoning
        + COMPOSITE_WEIGHTS["consistency"] * consistency
    )


def simulate_run(spec: ExperimentSpec, seed: int, run_id: str, commit_sha: str) -> RunResult:
    rng = stable_rng(spec.id, spec.model_variant, seed, spec.stage)
    effect = VARIANT_EFFECTS.get(spec.model_variant)
    if effect is None:
        raise ValueError(f"Unknown model_variant {spec.model_variant}")

    stage_multiplier = STAGE_GAIN_MULTIPLIER.get(spec.stage, 1.0)
    benchmark_scores = _simulate_benchmarks(spec, effect, stage_multiplier, rng)

    long_context = mean([benchmark_scores[name] for name in LONG_CONTEXT_DATASETS])
    reasoning = mean([benchmark_scores[name] for name in REASONING_DATASETS])
    consistency = mean([benchmark_scores[name] for name in CONSISTENCY_DATASETS])
    fluency = benchmark_scores["fluency"]
    composite = compute_composite(long_context, reasoning, consistency)

    train_cost = spec.train_budget_gpu_h * (1.0 + rng.uniform(-0.004, 0.004))
    infer_cost = spec.infer_budget_gpu_h * (1.0 + rng.uniform(-0.004, 0.004))
    infer_cost = _adjust_infer_cost_for_params(spec, infer_cost)

    baseline_latency = 120.0 + (spec.max_context / 4000.0)
    latency_p50 = baseline_latency * (1.0 + (effect.latency_delta_pct / 100.0) * stage_multiplier)
    latency_p50 = max(20.0, latency_p50 + rng.uniform(-1.0, 1.0))
    latency_p95 = latency_p50 * 1.28

    baseline_energy = (train_cost + infer_cost) * 0.38
    energy_kwh = baseline_energy * (1.0 + (effect.energy_delta_pct / 100.0) * stage_multiplier)

    metric_values = {
        "long_context": round(long_context, 4),
        "reasoning": round(reasoning, 4),
        "consistency": round(consistency, 4),
        "fluency": round(fluency, 4),
        "composite": round(composite, 4),
    }
    metric_values.update(_track_specific_metrics(spec, effect, stage_multiplier, long_context, reasoning, consistency))

    failure_flags = _failure_flags(spec, metric_values)

    if any(flag in {"analog_drift", "reversibility_break", "repetitive_text"} for flag in failure_flags):
        failure_flags.append("unstable_training")
    failure_flags = sorted(set(failure_flags))

    return RunResult(
        run_id=run_id,
        spec_id=spec.id,
        commit_sha=commit_sha,
        seed=seed,
        train_cost=round(train_cost, 4),
        infer_cost=round(infer_cost, 4),
        latency_p50=round(latency_p50, 4),
        latency_p95=round(latency_p95, 4),
        energy_kwh=round(energy_kwh, 4),
        metric_values=metric_values,
        failure_flags=failure_flags,
        track_id=spec.track_id,
        stage=spec.stage,
        model_variant=spec.model_variant,
        benchmark_scores={k: round(v, 4) for k, v in benchmark_scores.items()},
        metadata={"params": spec.params},
    )


def _simulate_benchmarks(spec: ExperimentSpec, effect: Any, stage_multiplier: float, rng: random.Random) -> dict[str, float]:
    scores = dict(BASE_BENCHMARK_SCORES)
    long_delta = effect.long_context_delta * stage_multiplier * QUALITY_DELTA_SCALE
    reasoning_delta = effect.reasoning_delta * stage_multiplier * QUALITY_DELTA_SCALE
    consistency_delta = effect.consistency_delta * stage_multiplier * QUALITY_DELTA_SCALE
    fluency_delta = effect.fluency_delta * stage_multiplier

    long_distribution = {
        "needle_32k": 0.20,
        "needle_64k": 0.25,
        "needle_128k": 0.35,
        "longbench": 0.20,
    }

    for dataset, weight in long_distribution.items():
        jitter = rng.uniform(-0.3, 0.3)
        scores[dataset] = clamp_score(scores[dataset] + long_delta * (0.7 + weight) + jitter)

    for dataset in REASONING_DATASETS:
        jitter = rng.uniform(-0.2, 0.2)
        scores[dataset] = clamp_score(scores[dataset] + reasoning_delta + jitter)

    for dataset in CONSISTENCY_DATASETS:
        jitter = rng.uniform(-0.2, 0.2)
        scores[dataset] = clamp_score(scores[dataset] + consistency_delta + jitter)

    scores["fluency"] = clamp_score(scores["fluency"] + fluency_delta + rng.uniform(-0.1, 0.1))

    if spec.track_id == "T3":
        compression_strength = float(spec.params.get("compression_ratio", 0.7))
        penalty = max(0.0, compression_strength - 0.8) * 15.0
        scores["consistency_longform"] = clamp_score(scores["consistency_longform"] - penalty)
    if spec.track_id == "T6":
        anneal_temp = float(spec.params.get("anneal_temp", 0.55))
        if anneal_temp < 0.2:
            scores["fluency"] = clamp_score(scores["fluency"] - 5.0)

    return scores


def _adjust_infer_cost_for_params(spec: ExperimentSpec, infer_cost: float) -> float:
    if spec.track_id == "T1" and spec.model_variant == "T1-E3":
        samples = int(spec.params.get("samples", 1))
        # Photonic savings offset part of multi-sample cost.
        adjusted = infer_cost * (1.0 + 0.02 * max(0, samples - 1) - 0.06)
        return max(0.0, adjusted)
    return infer_cost


def _track_specific_metrics(
    spec: ExperimentSpec,
    effect: Any,
    stage_multiplier: float,
    long_context: float,
    reasoning: float,
    consistency: float,
) -> dict[str, float]:
    base_metrics = dict(TRACK_SPECIFIC_METRIC_BASELINES[spec.track_id])
    metrics: dict[str, float] = {}

    if spec.track_id == "T1":
        recalibration_interval = int(spec.params.get("recalibration_interval", 512))
        drift_penalty = max(0.0, recalibration_interval - 2048) / 300.0
        metrics["noise_robustness_auc"] = round(
            base_metrics["noise_robustness_auc"] + effect.long_context_delta * 3.0 * stage_multiplier - drift_penalty,
            4,
        )
        metrics["latency_energy_proxy"] = round(
            base_metrics["latency_energy_proxy"] * (1.0 - 0.03 * stage_multiplier),
            4,
        )
    elif spec.track_id == "T2":
        anchor_frequency = spec.params.get("anchor_frequency", "1/4")
        reduction = {"1/2": 38.0, "1/4": 48.0, "1/8": 54.0}.get(anchor_frequency, 45.0)
        metrics["activation_memory_reduction_pct"] = round(reduction, 4)
        metrics["peak_memory_gb"] = round(base_metrics["peak_memory_gb"] * (1.0 - reduction / 100.0), 4)
        metrics["max_feasible_context"] = round(
            base_metrics["max_feasible_context"] * (1.0 + reduction / 90.0), 4
        )
    elif spec.track_id == "T3":
        compression_ratio = float(spec.params.get("compression_ratio", 0.7))
        token_reduction = clamp_score(compression_ratio * 100.0, 0.0, 95.0)
        miss_rate = max(0.5, base_metrics["critical_fact_miss_rate"] + max(0.0, compression_ratio - 0.8) * 6.0)
        metrics["token_access_reduction_pct"] = round(token_reduction, 4)
        metrics["critical_fact_miss_rate"] = round(miss_rate, 4)
    elif spec.track_id == "T4":
        binding_noise = float(spec.params.get("role_permutation_noise", 0.2))
        error_rate = max(2.0, base_metrics["binding_error_rate"] - reasoning * 0.15 + binding_noise * 10.0)
        metrics["binding_error_rate"] = round(error_rate, 4)
        metrics["compositional_gen_acc"] = round(
            base_metrics["compositional_gen_acc"] + effect.reasoning_delta * 7.0 * stage_multiplier,
            4,
        )
    elif spec.track_id == "T5":
        max_nodes = int(spec.params.get("max_nodes", 12))
        invalid_rate = max(0.0, base_metrics["invalid_plan_rate"] + max(0, max_nodes - 12) * 0.7 - stage_multiplier)
        metrics["invalid_plan_rate"] = round(invalid_rate, 4)
        metrics["run_variance"] = round(max(0.4, base_metrics["run_variance"] - effect.consistency_delta), 4)
    elif spec.track_id == "T6":
        contradiction_reduction = max(0.0, (consistency - 60.0) * 2.8)
        constraint_gain = max(0.0, (reasoning - 56.0) * 2.2)
        metrics["contradiction_reduction_pct"] = round(contradiction_reduction, 4)
        metrics["constraint_pass_gain_pct"] = round(constraint_gain, 4)

    metrics["long_context_component"] = round(long_context, 4)
    metrics["reasoning_component"] = round(reasoning, 4)
    metrics["consistency_component"] = round(consistency, 4)
    return metrics


def _failure_flags(spec: ExperimentSpec, metric_values: dict[str, float]) -> list[str]:
    flags: list[str] = []
    params = spec.params

    if spec.track_id == "T1":
        if int(params.get("recalibration_interval", 512)) > 2048:
            flags.append("analog_drift")
    elif spec.track_id == "T2":
        if bool(params.get("disable_norm_constraints", False)) or params.get("anchor_frequency") == "1/2":
            flags.append("reversibility_break")
    elif spec.track_id == "T3":
        if float(params.get("compression_ratio", 0.7)) > 0.9:
            flags.append("critical_fact_loss")
    elif spec.track_id == "T4":
        if float(params.get("role_permutation_noise", 0.2)) > 0.7:
            flags.append("entity_role_swap")
    elif spec.track_id == "T5":
        if int(params.get("max_nodes", 12)) > 12:
            flags.append("invalid_circuit")
    elif spec.track_id == "T6":
        if float(params.get("anneal_temp", 0.55)) < 0.2:
            flags.append("repetitive_text")

    if metric_values["fluency"] < 80.0:
        flags.append("fluency_regression")
    return flags


def evaluate_track_pass(candidate: RunResult, baseline: RunResult, delta_metrics: dict[str, float]) -> bool:
    if candidate.track_id == "T1":
        return delta_metrics.get("long_context", 0.0) >= 5.0 and "analog_drift" not in candidate.failure_flags
    if candidate.track_id == "T2":
        reduction = candidate.metric_values.get("activation_memory_reduction_pct", 0.0)
        return reduction >= 45.0 and delta_metrics.get("long_context", 0.0) >= 4.0
    if candidate.track_id == "T3":
        token_reduction = candidate.metric_values.get("token_access_reduction_pct", 0.0)
        baseline_miss = baseline.metric_values.get("critical_fact_miss_rate", 8.0)
        candidate_miss = candidate.metric_values.get("critical_fact_miss_rate", 8.0)
        return token_reduction >= 70.0 and (candidate_miss - baseline_miss) <= 2.0
    if candidate.track_id == "T4":
        baseline_error = baseline.metric_values.get("binding_error_rate", 15.0)
        candidate_error = candidate.metric_values.get("binding_error_rate", 15.0)
        if baseline_error <= 0:
            return False
        reduction = (baseline_error - candidate_error) / baseline_error * 100.0
        return reduction >= 30.0 and delta_metrics.get("reasoning", 0.0) >= 4.0
    if candidate.track_id == "T5":
        return (
            delta_metrics.get("composite", 0.0) >= 5.0
            and candidate.metric_values.get("invalid_plan_rate", 100.0) < 1.0
            and _latency_overhead_pct(candidate, baseline) <= 15.0
        )
    if candidate.track_id == "T6":
        return (
            candidate.metric_values.get("contradiction_reduction_pct", 0.0) >= 25.0
            and candidate.metric_values.get("constraint_pass_gain_pct", 0.0) >= 10.0
            and (baseline.metric_values.get("fluency", 0.0) - candidate.metric_values.get("fluency", 0.0)) <= 2.0
        )
    return False


def _latency_overhead_pct(candidate: RunResult, baseline: RunResult) -> float:
    if baseline.latency_p50 <= 0:
        return 100.0
    return (candidate.latency_p50 - baseline.latency_p50) / baseline.latency_p50 * 100.0
