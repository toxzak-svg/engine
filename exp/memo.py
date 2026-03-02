from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .attribution import attribute_all_tracks, format_attribution_report
from .constants import ENGINEERING_COMPLEXITY
from .models import ComparisonReport, RunResult


def build_decision_memo(
    stage: int,
    reports: list[ComparisonReport],
    run_lookup: dict[str, RunResult] | None = None,
) -> str:
    stage_reports_all = [report for report in reports if report.candidate_stage == stage]
    recovery_reports = [
        report for report in stage_reports_all if report.candidate_run_ids and report.candidate_run_ids[0].startswith("recovery-")
    ]
    core_reports = [
        report for report in stage_reports_all if not (report.candidate_run_ids and report.candidate_run_ids[0].startswith("recovery-"))
    ]
    stage_reports = core_reports if core_reports else stage_reports_all
    grouped: dict[str, list[ComparisonReport]] = defaultdict(list)
    for report in stage_reports:
        grouped[report.track_id].append(report)

    rows = []
    for track_id, group in grouped.items():
        comp_deltas = [item.delta_metrics.get("composite", 0.0) for item in group]
        anchor_deltas = [item.anchor_delta_metrics.get("composite", item.delta_metrics.get("composite", 0.0)) for item in group]
        passes = [bool(item.pass_fail.get("overall_pass", False)) for item in group]
        ci_excludes_zero = [bool(item.significance_tests.get("ci95_excludes_zero", False)) for item in group]
        ci_widths = [
            float(item.significance_tests["ci95"][1]) - float(item.significance_tests["ci95"][0])
            for item in group
            if isinstance(item.significance_tests.get("ci95"), list) and len(item.significance_tests["ci95"]) == 2
        ]
        pass_rate = (sum(1 for item in passes if item) / len(passes)) if passes else 0.0
        mean_anchor = mean(anchor_deltas) if anchor_deltas else 0.0
        rows.append(
            {
                "track_id": track_id,
                "mean_delta_stage": mean(comp_deltas) if comp_deltas else 0.0,
                "mean_delta_anchor": mean_anchor,
                "best_delta_anchor": max(anchor_deltas) if anchor_deltas else 0.0,
                "pass_rate": pass_rate,
                "robustness_rate": (sum(1 for item in ci_excludes_zero if item) / len(ci_excludes_zero)) if ci_excludes_zero else 0.0,
                "mean_ci_width": mean(ci_widths) if ci_widths else 0.0,
                "decision_score": mean_anchor * pass_rate,
                "complexity": ENGINEERING_COMPLEXITY.get(track_id, "unknown"),
            }
        )
    rows.sort(
        key=lambda row: (row["decision_score"], row["pass_rate"], row["mean_delta_anchor"], -row["mean_ci_width"]),
        reverse=True,
    )

    lines = []
    lines.append(f"# Stage {stage} Decision Memo")
    lines.append("")
    lines.append(f"- Core comparisons included: {len(stage_reports)}")
    if recovery_reports:
        lines.append(f"- Recovery comparisons excluded from primary ranking: {len(recovery_reports)}")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append(
        "| Rank | Track | Decision Score | Mean Delta vs Anchor | Best Delta vs Anchor | Mean Delta vs Stage Baseline | "
        "Pass Rate | Robustness | Mean CI Width | Engineering Complexity |"
    )
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---|")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row['track_id']} | {row['decision_score']:.3f} | {row['mean_delta_anchor']:.3f} | "
            f"{row['best_delta_anchor']:.3f} | {row['mean_delta_stage']:.3f} | {row['pass_rate']*100:.1f}% | "
            f"{row['robustness_rate']*100:.1f}% | {row['mean_ci_width']:.3f} | {row['complexity']} |"
        )

    lines.append("")
    lines.append("## Recommendation")
    if not rows:
        lines.append("No comparison reports available for this stage.")
    else:
        top = rows[0]
        lines.append(
            f"Prioritize `{top['track_id']}` first based on strongest pass-adjusted anchor gains and current gate performance."
        )
    attribution_sections = _format_ranked_attributions(rows, grouped, run_lookup)
    if attribution_sections:
        lines.append("")
        lines.append("## Param Attribution")
        lines.append("")
        lines.extend(attribution_sections)
    lines.append("")
    return "\n".join(lines)


def _format_ranked_attributions(
    ranked_rows: list[dict[str, object]],
    grouped_reports: dict[str, list[ComparisonReport]],
    run_lookup: dict[str, RunResult] | None,
) -> list[str]:
    if not run_lookup:
        return []

    ordered_track_ids = [str(row["track_id"]) for row in ranked_rows]
    runs_by_track: dict[str, list[RunResult]] = {}
    baseline_composites: dict[str, float] = {}

    for track_id in ordered_track_ids:
        group = grouped_reports.get(track_id, [])
        candidate_runs = [
            run_lookup[run_id]
            for report in group
            for run_id in report.candidate_run_ids
            if run_id in run_lookup
        ]
        baseline_runs = [
            run_lookup[run_id]
            for report in group
            for run_id in report.baseline_run_ids
            if run_id in run_lookup
        ]
        if not candidate_runs or not baseline_runs:
            continue
        runs_by_track[track_id] = candidate_runs
        baseline_composites[track_id] = mean(
            run.metric_values.get("composite", 0.0) for run in baseline_runs
        )

    if not runs_by_track:
        return []

    attribution_results = attribute_all_tracks(runs_by_track, baseline_composites)
    return [
        format_attribution_report(attribution_results[track_id])
        for track_id in ordered_track_ids
        if track_id in attribution_results and attribution_results[track_id].attributions
    ]
