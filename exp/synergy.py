"""Cross-track synergy detection and matrix computation.

Synergy measures whether combining two track variants produces super-additive
gains — i.e., the combination beats the better of the two individually.

    synergy[Ti][Tj] = composite(Ti+Tj) - max(composite(Ti), composite(Tj))

Positive synergy → super-additive (combination is worth running).
Zero synergy     → additive (no interaction benefit).
Negative synergy → sub-additive (techniques interfere with each other).

Usage:
    from exp.synergy import compute_synergy_score, build_synergy_matrix

    matrix = build_synergy_matrix(run_results_by_variant)
    top_pairs = get_top_synergy_pairs(matrix, top_n=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any

from .models import RunResult


@dataclass(frozen=True)
class SynergyScore:
    """Synergy score for a pair of track variants.

    Attributes:
        track_a: First track ID.
        variant_a: First variant ID (e.g. 'T3-E2').
        track_b: Second track ID.
        variant_b: Second variant ID (e.g. 'T4-E3').
        composite_a: Mean composite for variant A alone.
        composite_b: Mean composite for variant B alone.
        composite_combined: Estimated composite for A+B combination.
        synergy: composite_combined - max(composite_a, composite_b).
        super_additive: True if synergy > SYNERGY_THRESHOLD.
        sub_additive: True if synergy < -SYNERGY_THRESHOLD.
        confidence: Fraction of seeds where synergy is positive.
    """
    track_a: str
    variant_a: str
    track_b: str
    variant_b: str
    composite_a: float
    composite_b: float
    composite_combined: float
    synergy: float
    super_additive: bool
    sub_additive: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_a": self.track_a,
            "variant_a": self.variant_a,
            "track_b": self.track_b,
            "variant_b": self.variant_b,
            "composite_a": round(self.composite_a, 4),
            "composite_b": round(self.composite_b, 4),
            "composite_combined": round(self.composite_combined, 4),
            "synergy": round(self.synergy, 4),
            "super_additive": self.super_additive,
            "sub_additive": self.sub_additive,
            "confidence": round(self.confidence, 4),
        }


# Synergy threshold: synergy > this → super-additive; < -this → sub-additive
SYNERGY_THRESHOLD = 0.3  # composite pts

# Interaction dampening: combined effect is not fully additive due to overlap
# Conservative estimate: 70% of the sum of individual deltas over the better one
INTERACTION_DAMPENING = 0.70


def compute_synergy_score(
    runs_a: list[RunResult],
    runs_b: list[RunResult],
    baseline_composite: float,
) -> SynergyScore:
    """Compute synergy between two sets of variant runs.

    The combined composite is estimated as:
        combined = better_composite + INTERACTION_DAMPENING * (weaker_delta)

    where weaker_delta = weaker_composite - baseline_composite.

    This is conservative: it assumes the weaker technique contributes only
    70% of its standalone gain when combined with the stronger technique,
    due to overlapping mechanisms.

    Args:
        runs_a: RunResult list for variant A (same track, multiple seeds).
        runs_b: RunResult list for variant B (same track, multiple seeds).
        baseline_composite: Composite score of the shared baseline.

    Returns:
        SynergyScore with synergy estimate and classification.
    """
    if not runs_a or not runs_b:
        raise ValueError("Both run lists must be non-empty.")

    composites_a = [r.metric_values.get("composite", 0.0) for r in runs_a]
    composites_b = [r.metric_values.get("composite", 0.0) for r in runs_b]

    mean_a = mean(composites_a)
    mean_b = mean(composites_b)

    delta_a = mean_a - baseline_composite
    delta_b = mean_b - baseline_composite

    better_composite = max(mean_a, mean_b)
    weaker_delta = min(delta_a, delta_b)

    # Estimated combined composite
    combined = better_composite + INTERACTION_DAMPENING * max(0.0, weaker_delta)

    synergy = combined - better_composite

    # Confidence: fraction of seed pairs where synergy is positive
    # Approximate by checking if both variants beat baseline on same seeds
    seeds_a_beat = sum(1 for c in composites_a if c > baseline_composite)
    seeds_b_beat = sum(1 for c in composites_b if c > baseline_composite)
    confidence = (seeds_a_beat / len(composites_a) + seeds_b_beat / len(composites_b)) / 2.0

    track_a = runs_a[0].track_id
    variant_a = runs_a[0].model_variant
    track_b = runs_b[0].track_id
    variant_b = runs_b[0].model_variant

    return SynergyScore(
        track_a=track_a,
        variant_a=variant_a,
        track_b=track_b,
        variant_b=variant_b,
        composite_a=round(mean_a, 4),
        composite_b=round(mean_b, 4),
        composite_combined=round(combined, 4),
        synergy=round(synergy, 4),
        super_additive=synergy > SYNERGY_THRESHOLD,
        sub_additive=synergy < -SYNERGY_THRESHOLD,
        confidence=round(confidence, 4),
    )


def build_synergy_matrix(
    variant_runs: dict[str, list[RunResult]],
    baseline_composite: float,
) -> list[SynergyScore]:
    """Build a full pairwise synergy matrix across all variant groups.

    Args:
        variant_runs: Dict mapping variant_id → list of RunResult.
                      E.g. {"T3-E2": [...], "T4-E3": [...], "T1-E2": [...]}
        baseline_composite: Composite score of the shared anchor/baseline.

    Returns:
        List of SynergyScore objects for all pairs, sorted by synergy descending.
    """
    variant_ids = sorted(variant_runs.keys())
    scores: list[SynergyScore] = []

    for variant_a, variant_b in combinations(variant_ids, 2):
        runs_a = variant_runs[variant_a]
        runs_b = variant_runs[variant_b]

        # Skip same-track pairs (can't combine two variants of the same track)
        if runs_a and runs_b and runs_a[0].track_id == runs_b[0].track_id:
            continue

        try:
            score = compute_synergy_score(runs_a, runs_b, baseline_composite)
            scores.append(score)
        except (ValueError, ZeroDivisionError):
            continue

    scores.sort(key=lambda s: s.synergy, reverse=True)
    return scores


def get_top_synergy_pairs(
    matrix: list[SynergyScore],
    top_n: int = 5,
    super_additive_only: bool = False,
) -> list[SynergyScore]:
    """Get the top N synergy pairs from a synergy matrix.

    Args:
        matrix: Output of build_synergy_matrix().
        top_n: Number of top pairs to return.
        super_additive_only: If True, only return super-additive pairs.

    Returns:
        Top N SynergyScore objects sorted by synergy descending.
    """
    filtered = [s for s in matrix if s.super_additive] if super_additive_only else matrix
    return filtered[:top_n]


def format_synergy_report(matrix: list[SynergyScore]) -> str:
    """Format a synergy matrix as a markdown report section.

    Args:
        matrix: Output of build_synergy_matrix().

    Returns:
        Markdown string with synergy table and recommendations.
    """
    lines: list[str] = []
    lines.append("## Cross-Track Synergy Matrix")
    lines.append("")
    lines.append(
        "Synergy = estimated combined composite − max(individual composites). "
        f"Super-additive threshold: >{SYNERGY_THRESHOLD} pts. "
        f"Interaction dampening factor: {INTERACTION_DAMPENING:.0%}."
    )
    lines.append("")
    lines.append(
        "| Variant A | Variant B | Composite A | Composite B | "
        "Combined Est. | Synergy | Classification | Confidence |"
    )
    lines.append("|:---:|:---:|---:|---:|---:|---:|:---:|---:|")

    for score in matrix:
        classification = (
            "🟢 Super-additive" if score.super_additive
            else ("🔴 Sub-additive" if score.sub_additive else "⚪ Neutral")
        )
        lines.append(
            f"| {score.variant_a} | {score.variant_b} | "
            f"{score.composite_a:.3f} | {score.composite_b:.3f} | "
            f"{score.composite_combined:.3f} | {score.synergy:+.3f} | "
            f"{classification} | {score.confidence*100:.0f}% |"
        )

    lines.append("")
    super_additive = [s for s in matrix if s.super_additive]
    if super_additive:
        lines.append("### Recommended Combination Experiments")
        lines.append("")
        for idx, score in enumerate(super_additive[:3], start=1):
            lines.append(
                f"{idx}. **{score.variant_a} + {score.variant_b}**: "
                f"estimated synergy +{score.synergy:.3f} pts composite "
                f"(confidence {score.confidence*100:.0f}%)"
            )
        lines.append("")
    else:
        lines.append("No super-additive pairs detected at current threshold.")
        lines.append("")

    return "\n".join(lines)


def select_portfolio_diverse_tracks(
    matrix: list[SynergyScore],
    promoted_tracks: list[str],
) -> list[str]:
    """Select a portfolio of tracks that maximizes synergy diversity.

    Given a list of promoted tracks, reorder them so that the top-2 promoted
    tracks have the highest pairwise synergy (maximizing the chance that
    combining them in production yields super-additive gains).

    Args:
        matrix: Output of build_synergy_matrix().
        promoted_tracks: Track IDs in current promotion order.

    Returns:
        Reordered list of track IDs prioritizing synergistic pairs first.
    """
    if len(promoted_tracks) < 2:
        return promoted_tracks

    # Build synergy lookup: (track_a, track_b) → max synergy score
    synergy_lookup: dict[str, float] = {}
    for score in matrix:
        key = "|".join(sorted([score.track_a, score.track_b]))
        existing: float = synergy_lookup.get(key, -999.0)
        synergy_lookup[key] = max(existing, score.synergy)

    # Find the pair of promoted tracks with highest synergy
    best_pair: tuple[str, str] | None = None
    best_synergy: float = -999.0
    for t_a, t_b in combinations(promoted_tracks, 2):
        key = "|".join(sorted([t_a, t_b]))
        s: float = synergy_lookup.get(key, 0.0)
        if s > best_synergy:
            best_synergy = s
            best_pair = (t_a, t_b)

    if best_pair is None:
        return promoted_tracks

    # Put the best synergy pair first, then remaining tracks
    remaining = [t for t in promoted_tracks if t not in best_pair]
    return list(best_pair) + remaining
