from __future__ import annotations

from statistics import mean
from typing import Any

from .models import ComparisonReport, RunResult


def gate_stage(stage: int, reports: list[ComparisonReport]) -> dict[str, Any]:
    stage_reports_all = [report for report in reports if report.candidate_stage == stage]
    recovery_reports = [
        report for report in stage_reports_all if report.candidate_run_ids and report.candidate_run_ids[0].startswith("recovery-")
    ]
    core_reports = [
        report
        for report in stage_reports_all
        if not (report.candidate_run_ids and report.candidate_run_ids[0].startswith("recovery-"))
    ]
    stage_reports = core_reports if core_reports else stage_reports_all
    if stage not in {1, 2, 3, 4}:
        raise ValueError("stage must be one of 1, 2, 3, 4")
    promotion_limit = _promotion_track_limit(stage)

    track_groups: dict[str, list[ComparisonReport]] = {}
    for report in stage_reports:
        track_groups.setdefault(report.track_id, []).append(report)

    track_rows: list[dict[str, Any]] = []
    for track_id, group in track_groups.items():
        qualified = [report for report in group if report.pass_fail.get("overall_pass", False)]
        group_anchor = [report.anchor_delta_metrics.get("composite", report.delta_metrics.get("composite", 0.0)) for report in group]
        qualified_anchor = [
            report.anchor_delta_metrics.get("composite", report.delta_metrics.get("composite", 0.0)) for report in qualified
        ]
        pass_rate = sum(1 for report in group if report.pass_fail.get("overall_pass", False)) / len(group)
        best_promotable = None
        if qualified:
            best_promotable = max(
                qualified,
                key=lambda report: report.anchor_delta_metrics.get("composite", report.delta_metrics.get("composite", float("-inf"))),
            )
        track_rows.append(
            {
                "track_id": track_id,
                "report_count": len(group),
                "qualified_count": len(qualified),
                "pass_rate": pass_rate,
                "mean_anchor": mean(group_anchor) if group_anchor else float("-inf"),
                "mean_anchor_qualified": mean(qualified_anchor) if qualified_anchor else float("-inf"),
                "best_promotable": best_promotable,
            }
        )

    track_rows.sort(
        key=lambda row: (
            row["qualified_count"] > 0,
            row["pass_rate"],
            row["mean_anchor_qualified"],
            row["mean_anchor"],
        ),
        reverse=True,
    )

    promotable_rows = [row for row in track_rows if row["qualified_count"] > 0]
    promoted_rows = promotable_rows[:promotion_limit] if promotion_limit > 0 else []
    promoted_tracks = [row["track_id"] for row in promoted_rows]
    promoted = [row["best_promotable"] for row in promoted_rows if row["best_promotable"] is not None]
    promoted_run_ids = [report.candidate_run_ids[0] for report in promoted]
    promoted_run_ids_set = set(promoted_run_ids)
    rejected = [report for report in stage_reports if report.candidate_run_ids and report.candidate_run_ids[0] not in promoted_run_ids_set]
    rejected_tracks = [row["track_id"] for row in track_rows if row["track_id"] not in promoted_tracks]
    qualified_count = sum(1 for report in stage_reports if report.pass_fail.get("overall_pass", False))

    # Annotate Pareto flags (no run_lookup needed for basic mode)
    track_rows = pareto_promote(track_rows)
    pareto_tracks = [row["track_id"] for row in track_rows if row.get("pareto_promoted", False)]

    return {
        "stage": stage,
        "candidate_count_all": len(stage_reports_all),
        "candidate_count_recovery": len(recovery_reports),
        "candidate_count": len(stage_reports),
        "qualified_count": qualified_count,
        "candidate_track_count": len(track_rows),
        "qualified_track_count": len(promotable_rows),
        "promoted_count": len(promoted),
        "promoted_track_count": len(promoted_tracks),
        "promoted_tracks": promoted_tracks,
        "promoted_run_ids": promoted_run_ids,
        "rejected_run_ids": [report.candidate_run_ids[0] for report in rejected],
        "rejected_tracks": rejected_tracks,
        "promotion_limit": promotion_limit,
        "pareto_tracks": pareto_tracks,
        "thresholds": _thresholds_for_stage(stage),
    }


# ---------------------------------------------------------------------------
# Pareto-Frontier Promotion
# ---------------------------------------------------------------------------

def pareto_promote(
    track_rows: list[dict[str, Any]],
    run_lookup: dict[str, RunResult] | None = None,
) -> list[dict[str, Any]]:
    """Identify tracks on the Pareto frontier across multiple objectives.

    A track is Pareto-non-dominated if no other track beats it on ALL of:
        - mean_anchor (composite delta vs anchor, higher is better)
        - neg_latency_p50 (lower latency is better → negate for maximisation)
        - neg_energy_kwh (lower energy is better → negate for maximisation)
        - mean_fluency (higher fluency is better)

    Tracks on the Pareto frontier receive a ``pareto_promoted`` flag even if
    they miss the composite threshold, making them visible for deployment
    scenarios that prioritise latency or energy over raw composite.

    Args:
        track_rows: List of track summary dicts from gate_stage().
                    Each dict must have at least ``track_id`` and
                    ``mean_anchor``. Latency/energy/fluency are optional
                    and default to 0.0 when absent.
        run_lookup: Optional dict mapping run_id → RunResult for richer
                    latency/energy/fluency data. When provided, values are
                    pulled from the best_promotable run for each track.

    Returns:
        The same list of track_rows with a ``pareto_promoted`` bool added
        to each dict. Rows are NOT reordered.
    """
    if not track_rows:
        return track_rows

    # Build objective vectors per track
    objectives: list[dict[str, float]] = []
    for row in track_rows:
        best_run: RunResult | None = None
        if run_lookup is not None:
            best_report = row.get("best_promotable")
            if best_report is not None:
                run_id = best_report.candidate_run_ids[0] if best_report.candidate_run_ids else ""
                best_run = run_lookup.get(run_id)

        latency = best_run.latency_p50 if best_run else 0.0
        energy = best_run.energy_kwh if best_run else 0.0
        fluency = best_run.metric_values.get("fluency", 0.0) if best_run else 0.0

        objectives.append({
            "mean_anchor": float(row.get("mean_anchor", 0.0)),
            "neg_latency": -latency,          # negate: lower latency is better
            "neg_energy": -energy,            # negate: lower energy is better
            "fluency": fluency,
        })

    obj_keys = ["mean_anchor", "neg_latency", "neg_energy", "fluency"]

    # Determine Pareto non-dominated set
    n = len(objectives)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i: j >= i on all objectives AND j > i on at least one
            all_ge = all(objectives[j][k] >= objectives[i][k] for k in obj_keys)
            any_gt = any(objectives[j][k] > objectives[i][k] for k in obj_keys)
            if all_ge and any_gt:
                dominated[i] = True
                break

    # A Pareto frontier must be non-empty. If pathological inputs ever mark
    # everything dominated, keep the strongest objective vector on the frontier.
    if all(dominated):
        frontier_idx = max(
            range(n),
            key=lambda idx: tuple(objectives[idx][k] for k in obj_keys),
        )
        dominated[frontier_idx] = False

    # Annotate rows
    for row, objective, is_dominated in zip(track_rows, objectives, dominated):
        row["pareto_promoted"] = not is_dominated
        row["pareto_objectives"] = objective

    return track_rows


def _promotion_track_limit(stage: int) -> int:
    if stage == 1:
        return 3
    if stage == 2:
        return 2
    if stage == 3:
        return 1
    return 0


def _thresholds_for_stage(stage: int) -> dict[str, Any]:
    if stage == 1:
        return {
            "delta_composite": ">= 3.0",
            "stable_training": True,
            "fluency_drop_pct": "<= 2.0",
            "equal_cost_tolerance_pct": "<= 2.0",
            "promote_top_tracks_n": 3,
            "requires_overall_pass": True,
        }
    if stage == 2:
        return {
            "delta_composite": ">= 5.0",
            "latency_overhead_pct": "<= 15.0",
            "equal_cost_tolerance_pct": "<= 2.0",
            "promote_top_tracks_n": 2,
            "requires_overall_pass": True,
        }
    if stage == 3:
        return {
            "delta_composite": ">= 8.0",
            "bootstrap_ci_excludes_zero": True,
            "equal_cost_tolerance_pct": "<= 2.0",
            "promote_top_tracks_n": 1,
            "requires_overall_pass": True,
        }
    return {
        "delta_composite": ">= 8.0",
        "bootstrap_ci_excludes_zero": True,
        "track_specific_pass_required": True,
        "t3_focus": "token_access_reduction_pct >= 70 and miss_rate_increase <= 2",
        "equal_cost_tolerance_pct": "<= 2.0",
        "promote_top_tracks_n": 0,
        "requires_overall_pass": True,
    }
