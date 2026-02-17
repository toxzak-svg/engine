from __future__ import annotations

from dataclasses import dataclass

COMPOSITE_WEIGHTS = {
    "long_context": 0.45,
    "reasoning": 0.35,
    "consistency": 0.20,
}

STAGE_BUDGET_GPU_HOURS = {
    0: 300.0,
    1: 150.0,
    2: 600.0,
    3: 1200.0,
    4: 900.0,
}

TRACKS = ("T1", "T2", "T3", "T4", "T5", "T6", "ANCHOR")

SEED_POLICY = {
    1: 3,
    2: 2,
    3: 2,
    4: 3,
}

BASE_BENCHMARK_SCORES = {
    "needle_32k": 55.0,
    "needle_64k": 50.0,
    "needle_128k": 45.0,
    "longbench": 52.0,
    "gsm8k": 58.0,
    "bbh": 56.0,
    "consistency_longform": 60.0,
    "fluency": 90.0,
}

LONG_CONTEXT_DATASETS = ("needle_32k", "needle_64k", "needle_128k", "longbench")
REASONING_DATASETS = ("gsm8k", "bbh")
CONSISTENCY_DATASETS = ("consistency_longform",)


@dataclass(frozen=True)
class VariantEffect:
    long_context_delta: float
    reasoning_delta: float
    consistency_delta: float
    fluency_delta: float
    latency_delta_pct: float
    energy_delta_pct: float


VARIANT_EFFECTS: dict[str, VariantEffect] = {
    "BASELINE": VariantEffect(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "T1-E1": VariantEffect(2.0, 0.3, 0.6, -0.2, -8.0, -10.0),
    "T1-E2": VariantEffect(3.5, 0.6, 0.8, -0.3, -10.0, -12.0),
    "T1-E3": VariantEffect(2.2, 1.0, 1.2, -0.4, -1.0, -9.0),
    "T2-E1": VariantEffect(1.8, 0.8, 0.5, -0.1, 3.0, -6.0),
    "T2-E2": VariantEffect(2.8, 1.0, 0.7, -0.1, 4.0, -8.0),
    "T2-E3": VariantEffect(2.0, 1.2, 0.4, -0.2, 5.0, -7.0),
    "T3-E1": VariantEffect(3.0, 0.7, 0.6, -0.1, -6.0, -5.0),
    "T3-E2": VariantEffect(3.5, 0.9, 1.0, -0.2, -7.0, -8.0),
    "T3-E3": VariantEffect(2.8, 1.1, 0.8, -0.2, -4.0, -4.0),
    "T4-E1": VariantEffect(1.0, 2.5, 1.2, -0.1, 2.0, -3.0),
    "T4-E2": VariantEffect(1.2, 2.8, 1.5, -0.1, 3.0, -4.0),
    "T4-E3": VariantEffect(0.8, 3.0, 1.6, -0.2, 4.0, -3.0),
    "T5-E1": VariantEffect(1.3, 2.0, 1.8, -0.1, 6.0, -2.0),
    "T5-E2": VariantEffect(1.0, 2.5, 2.0, -0.2, 8.0, -2.0),
    "T5-E3": VariantEffect(1.1, 2.3, 2.2, -0.2, 7.0, -2.0),
    "T6-E1": VariantEffect(0.7, 1.5, 2.8, -0.4, 9.0, -1.0),
    "T6-E2": VariantEffect(0.9, 1.7, 3.2, -0.5, 10.0, -1.0),
    "T6-E3": VariantEffect(1.1, 1.9, 3.5, -0.6, 11.0, -1.0),
}

STAGE_GAIN_MULTIPLIER = {
    0: 0.0,
    1: 1.2,
    2: 1.8,
    3: 2.4,
    4: 2.8,
}

QUALITY_DELTA_SCALE = 2.0

TRACK_SPECIFIC_METRIC_BASELINES = {
    "T1": {
        "noise_robustness_auc": 58.0,
        "latency_energy_proxy": 1.0,
    },
    "T2": {
        "peak_memory_gb": 80.0,
        "max_feasible_context": 64000.0,
    },
    "T3": {
        "token_access_reduction_pct": 20.0,
        "critical_fact_miss_rate": 8.0,
    },
    "T4": {
        "binding_error_rate": 15.0,
        "compositional_gen_acc": 55.0,
    },
    "T5": {
        "invalid_plan_rate": 3.5,
        "run_variance": 5.0,
    },
    "T6": {
        "contradiction_reduction_pct": 0.0,
        "constraint_pass_gain_pct": 0.0,
    },
    "ANCHOR": {},
}

TRACK_PASS_CRITERIA = {
    "T1": "at least one K setting has long_context_delta >= 5 and no catastrophic drift",
    "T2": "activation_memory_reduction >= 45 and long_context_delta >= 4",
    "T3": "token_access_reduction_pct >= 70 and miss_rate_increase <= 2",
    "T4": "binding_error_reduction_pct >= 30 and reasoning_delta >= 4",
    "T5": "composite_delta >= 5 and invalid_plan_rate < 1 and latency_overhead <= 15",
    "T6": "contradiction_reduction_pct >= 25 and constraint_pass_gain_pct >= 10 and fluency_drop <= 2",
    "ANCHOR": "not_applicable",
}

ENGINEERING_COMPLEXITY = {
    "T1": "very_high",
    "T2": "medium_high",
    "T3": "high",
    "T4": "medium",
    "T5": "high",
    "T6": "medium_high",
    "ANCHOR": "low",
}
