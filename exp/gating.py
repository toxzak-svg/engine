from __future__ import annotations

from typing import Any

from .models import ComparisonReport


def gate_stage(stage: int, reports: list[ComparisonReport]) -> dict[str, Any]:
    stage_reports = [report for report in reports if report.candidate_stage == stage]

    qualified = [
        report
        for report in stage_reports
        if report.pass_fail.get("equal_cost_pass", False) and report.pass_fail.get("stage_gate_pass", False)
    ]
    qualified.sort(key=lambda report: report.delta_metrics.get("composite", float("-inf")), reverse=True)

    if stage == 1:
        promotion_limit = 3
    elif stage == 2:
        promotion_limit = 2
    elif stage == 3:
        promotion_limit = len(qualified)
    else:
        raise ValueError("stage must be one of 1, 2, 3")

    promoted = qualified[:promotion_limit]
    rejected = [report for report in stage_reports if report not in promoted]

    return {
        "stage": stage,
        "candidate_count": len(stage_reports),
        "qualified_count": len(qualified),
        "promoted_count": len(promoted),
        "promoted_run_ids": [report.candidate_run_ids[0] for report in promoted],
        "rejected_run_ids": [report.candidate_run_ids[0] for report in rejected],
        "promotion_limit": promotion_limit,
        "thresholds": _thresholds_for_stage(stage),
    }


def _thresholds_for_stage(stage: int) -> dict[str, Any]:
    if stage == 1:
        return {
            "delta_composite": ">= 3.0",
            "stable_training": True,
            "fluency_drop_pct": "<= 2.0",
            "equal_cost_tolerance_pct": "<= 2.0",
            "promote_top_n": 3,
        }
    if stage == 2:
        return {
            "delta_composite": ">= 5.0",
            "latency_overhead_pct": "<= 15.0",
            "equal_cost_tolerance_pct": "<= 2.0",
            "promote_top_n": 2,
        }
    return {
        "delta_composite": ">= 8.0",
        "bootstrap_ci_excludes_zero": True,
        "equal_cost_tolerance_pct": "<= 2.0",
        "promote_top_n": "all qualified",
    }
