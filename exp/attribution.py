"""Shapley-value param attribution for experiment results.

Computes the marginal contribution of each hyperparameter to the composite
delta, using a cooperative game theory approach (Shapley values).

The key insight: instead of asking "which variant won?", we ask "which
*param change* drove the win?" This closes the causal gap in memos.

Algorithm:
    For each param p in a track's param space:
        Shapley(p) = weighted average of marginal contributions of p
                     across all subsets of other params.

    Since full Shapley is exponential, we use the efficient approximation:
        Shapley(p) ≈ mean over N permutations of:
            [composite(params with p) - composite(params without p)]

    In practice, we use the existing run artifacts (no new GPU hours):
    - Group runs by param values
    - For each param, compare runs where param=high vs param=low
    - Weight by how many other params are held constant

Usage:
    from exp.attribution import shapley_param_attribution, format_attribution_report

    attribution = shapley_param_attribution(track_id="T3", runs=runs, baseline_composite=57.2)
    report = format_attribution_report(attribution)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .models import RunResult


@dataclass(frozen=True)
class ParamAttribution:
    """Shapley-style attribution for a single parameter.

    Attributes:
        param_name: Name of the hyperparameter.
        shapley_value: Estimated marginal contribution to composite delta (pts).
        direction: 'positive' if higher param value → higher composite, else 'negative'.
        n_comparisons: Number of run pairs used to estimate this value.
        confidence: Fraction of comparisons where direction is consistent.
        best_value: Param value associated with highest composite.
        worst_value: Param value associated with lowest composite.
        value_composites: Dict mapping param value → mean composite.
    """
    param_name: str
    shapley_value: float
    direction: str
    n_comparisons: int
    confidence: float
    best_value: Any
    worst_value: Any
    value_composites: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_name": self.param_name,
            "shapley_value": round(self.shapley_value, 4),
            "direction": self.direction,
            "n_comparisons": self.n_comparisons,
            "confidence": round(self.confidence, 4),
            "best_value": self.best_value,
            "worst_value": self.worst_value,
            "value_composites": {k: round(v, 4) for k, v in self.value_composites.items()},
        }


@dataclass
class AttributionResult:
    """Full attribution result for a track's runs.

    Attributes:
        track_id: Track being attributed.
        stage: Stage of the runs.
        baseline_composite: Composite of the baseline (reference point).
        attributions: List of ParamAttribution, sorted by |shapley_value| desc.
        total_explained: Sum of Shapley values (should approximate total delta).
        total_delta: Actual mean composite delta vs baseline.
        explanation_ratio: total_explained / total_delta (1.0 = fully explained).
        n_runs: Number of runs used.
    """
    track_id: str
    stage: int
    baseline_composite: float
    attributions: list[ParamAttribution]
    total_explained: float
    total_delta: float
    explanation_ratio: float
    n_runs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "stage": self.stage,
            "baseline_composite": round(self.baseline_composite, 4),
            "attributions": [a.to_dict() for a in self.attributions],
            "total_explained": round(self.total_explained, 4),
            "total_delta": round(self.total_delta, 4),
            "explanation_ratio": round(self.explanation_ratio, 4),
            "n_runs": self.n_runs,
        }


def shapley_param_attribution(
    track_id: str,
    runs: list[RunResult],
    baseline_composite: float,
    metric: str = "composite",
) -> AttributionResult:
    """Compute Shapley-style param attribution for a set of runs.

    Groups runs by their param values and computes the marginal contribution
    of each param to the target metric delta vs baseline.

    Args:
        track_id: Track ID (e.g. 'T3').
        runs: All RunResult objects for this track (all variants, all seeds).
        baseline_composite: Composite score of the baseline (reference point).
        metric: Metric to attribute (default: 'composite').

    Returns:
        AttributionResult with per-param Shapley values.
    """
    if not runs:
        return AttributionResult(
            track_id=track_id,
            stage=0,
            baseline_composite=baseline_composite,
            attributions=[],
            total_explained=0.0,
            total_delta=0.0,
            explanation_ratio=0.0,
            n_runs=0,
        )

    stage = runs[0].stage

    # Collect all param keys across all runs
    all_param_keys: set[str] = set()
    for run in runs:
        params = run.metadata.get("params", {})
        all_param_keys.update(params.keys())

    if not all_param_keys:
        # No params to attribute — return empty attribution
        composites = [r.metric_values.get(metric, 0.0) for r in runs]
        total_delta = mean(composites) - baseline_composite
        return AttributionResult(
            track_id=track_id,
            stage=stage,
            baseline_composite=baseline_composite,
            attributions=[],
            total_explained=0.0,
            total_delta=round(total_delta, 4),
            explanation_ratio=0.0,
            n_runs=len(runs),
        )

    # Group runs by each param value
    attributions: list[ParamAttribution] = []
    for param_key in sorted(all_param_keys):
        attr = _attribute_single_param(
            param_key=param_key,
            runs=runs,
            baseline_composite=baseline_composite,
            metric=metric,
        )
        if attr is not None:
            attributions.append(attr)

    # Sort by absolute Shapley value descending
    attributions.sort(key=lambda a: abs(a.shapley_value), reverse=True)

    # Compute totals
    composites = [r.metric_values.get(metric, 0.0) for r in runs]
    total_delta = mean(composites) - baseline_composite
    total_explained = sum(a.shapley_value for a in attributions)
    explanation_ratio = (
        total_explained / total_delta if abs(total_delta) > 1e-6 else 0.0
    )

    return AttributionResult(
        track_id=track_id,
        stage=stage,
        baseline_composite=baseline_composite,
        attributions=attributions,
        total_explained=round(total_explained, 4),
        total_delta=round(total_delta, 4),
        explanation_ratio=round(explanation_ratio, 4),
        n_runs=len(runs),
    )


def _attribute_single_param(
    param_key: str,
    runs: list[RunResult],
    baseline_composite: float,
    metric: str,
) -> ParamAttribution | None:
    """Compute attribution for a single param by comparing runs with different values.

    Groups runs by param value, computes mean composite per group, then
    estimates Shapley value as the weighted mean of pairwise deltas.

    Args:
        param_key: The param name to attribute.
        runs: All runs for this track.
        baseline_composite: Baseline composite for reference.
        metric: Metric to use.

    Returns:
        ParamAttribution or None if insufficient data.
    """
    # Group runs by param value
    value_groups: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        params = run.metadata.get("params", {})
        if param_key in params:
            val = str(params[param_key])
            composite = run.metric_values.get(metric, 0.0)
            value_groups[val].append(composite)

    if len(value_groups) < 2:
        # Need at least 2 distinct values to compute attribution
        return None

    # Compute mean composite per param value
    value_composites: dict[str, float] = {
        val: mean(composites) for val, composites in value_groups.items()
    }

    # Compute pairwise deltas between all value pairs
    values = sorted(value_composites.keys())
    pairwise_deltas: list[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            delta = value_composites[values[j]] - value_composites[values[i]]
            pairwise_deltas.append(delta)

    if not pairwise_deltas:
        return None

    # Shapley value = mean of pairwise deltas (simplified)
    # Positive → higher param values → better composite
    shapley_value = mean(pairwise_deltas)

    # Direction: positive if higher param value → higher composite
    direction = "positive" if shapley_value > 0 else "negative"

    # Confidence: fraction of pairwise comparisons consistent with direction
    consistent = sum(1 for d in pairwise_deltas if (d > 0) == (shapley_value > 0))
    confidence = consistent / len(pairwise_deltas) if pairwise_deltas else 0.0

    # Best and worst values
    best_value = max(value_composites, key=lambda v: value_composites[v])
    worst_value = min(value_composites, key=lambda v: value_composites[v])

    n_comparisons = sum(len(g) for g in value_groups.values())

    return ParamAttribution(
        param_name=param_key,
        shapley_value=round(shapley_value, 4),
        direction=direction,
        n_comparisons=n_comparisons,
        confidence=round(confidence, 4),
        best_value=best_value,
        worst_value=worst_value,
        value_composites=value_composites,
    )


def format_attribution_report(result: AttributionResult) -> str:
    """Format an AttributionResult as a markdown section for memos.

    Args:
        result: Output of shapley_param_attribution().

    Returns:
        Markdown string with attribution table and narrative.
    """
    lines: list[str] = []
    lines.append(f"### Param Attribution — {result.track_id} (Stage {result.stage})")
    lines.append("")
    lines.append(
        f"- Runs analyzed: {result.n_runs} | "
        f"Baseline composite: {result.baseline_composite:.3f} | "
        f"Mean delta: {result.total_delta:+.3f} pts"
    )
    lines.append(
        f"- Total explained by params: {result.total_explained:+.3f} pts "
        f"(explanation ratio: {result.explanation_ratio*100:.0f}%)"
    )
    lines.append("")

    if not result.attributions:
        lines.append("_No param variation detected — cannot compute attribution._")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "| Param | Shapley Value | Direction | Best Value | Worst Value | "
        "Confidence | Comparisons |"
    )
    lines.append("|:---|---:|:---:|:---:|:---:|---:|---:|")

    for attr in result.attributions:
        direction_icon = "↑" if attr.direction == "positive" else "↓"
        lines.append(
            f"| `{attr.param_name}` | {attr.shapley_value:+.3f} | "
            f"{direction_icon} | `{attr.best_value}` | `{attr.worst_value}` | "
            f"{attr.confidence*100:.0f}% | {attr.n_comparisons} |"
        )

    lines.append("")

    # Narrative: top driver
    if result.attributions:
        top = result.attributions[0]
        lines.append(
            f"**Key driver**: `{top.param_name}` contributes "
            f"{top.shapley_value:+.3f} pts composite (confidence {top.confidence*100:.0f}%). "
            f"Best setting: `{top.best_value}`. "
            f"Recommendation: fix `{top.param_name}={top.best_value}` in next stage."
        )
        lines.append("")

    return "\n".join(lines)


def attribute_all_tracks(
    runs_by_track: dict[str, list[RunResult]],
    baseline_composites: dict[str, float],
    metric: str = "composite",
) -> dict[str, AttributionResult]:
    """Compute attribution for all tracks in a stage.

    Args:
        runs_by_track: Dict mapping track_id → list of RunResult.
        baseline_composites: Dict mapping track_id → baseline composite.
        metric: Metric to attribute.

    Returns:
        Dict mapping track_id → AttributionResult.
    """
    results: dict[str, AttributionResult] = {}
    for track_id, runs in runs_by_track.items():
        baseline = baseline_composites.get(track_id, 0.0)
        results[track_id] = shapley_param_attribution(
            track_id=track_id,
            runs=runs,
            baseline_composite=baseline,
            metric=metric,
        )
    return results
