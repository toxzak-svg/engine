from __future__ import annotations

from statistics import mean
from typing import Any

from .models import ComparisonReport


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
        "thresholds": _thresholds_for_stage(stage),
    }


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
