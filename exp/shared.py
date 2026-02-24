"""Shared utilities and functions for the engine project."""

import random
from statistics import mean
from typing import Any

from .constants import (
    BASE_BENCHMARK_SCORES,
    CONSISTENCY_DATASETS,
    LONG_CONTEXT_DATASETS,
    QUALITY_DELTA_SCALE,
    REASONING_DATASETS,
)
from .models import ExperimentSpec
from .inverse_arms import VariantEffect

def clamp_score(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a score to valid range."""
    return max(lower, min(upper, value))

def simulate_benchmarks(
    spec: ExperimentSpec,
    effect: VariantEffect,
    stage_multiplier: float,
    rng: random.Random,
) -> dict[str, float]:
    """Simulate benchmark scores for an inverse arm."""
    scores = dict(BASE_BENCHMARK_SCORES)

    long_delta = effect.long_context_delta * stage_multiplier * QUALITY_DELTA_SCALE
    reasoning_delta = effect.reasoning_delta * stage_multiplier * QUALITY_DELTA_SCALE
    consistency_delta = effect.consistency_delta * stage_multiplier * QUALITY_DELTA_SCALE
    fluency_delta = effect.fluency_delta * stage_multiplier

    for dataset in LONG_CONTEXT_DATASETS:
        jitter = rng.uniform(-0.3, 0.3)
        scores[dataset] = clamp_score(scores[dataset] + long_delta + jitter)

    for dataset in REASONING_DATASETS:
        jitter = rng.uniform(-0.2, 0.2)
        scores[dataset] = clamp_score(scores[dataset] + reasoning_delta + jitter)

    for dataset in CONSISTENCY_DATASETS:
        jitter = rng.uniform(-0.2, 0.2)
        scores[dataset] = clamp_score(scores[dataset] + consistency_delta + jitter)

    scores["fluency"] = clamp_score(scores["fluency"] + fluency_delta + rng.uniform(-0.1, 0.1))

    return scores