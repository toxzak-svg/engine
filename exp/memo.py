from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .constants import ENGINEERING_COMPLEXITY
from .models import ComparisonReport


def build_decision_memo(stage: int, reports: list[ComparisonReport]) -> str:
    stage_reports = [report for report in reports if report.candidate_stage == stage]
    grouped: dict[str, list[ComparisonReport]] = defaultdict(list)
    for report in stage_reports:
        grouped[report.track_id].append(report)

    rows = []
    for track_id, group in grouped.items():
        comp_deltas = [item.delta_metrics.get("composite", 0.0) for item in group]
        passes = [bool(item.pass_fail.get("overall_pass", False)) for item in group]
        ci_excludes_zero = [bool(item.significance_tests.get("ci95_excludes_zero", False)) for item in group]
        rows.append(
            {
                "track_id": track_id,
                "mean_delta": mean(comp_deltas) if comp_deltas else 0.0,
                "best_delta": max(comp_deltas) if comp_deltas else 0.0,
                "pass_rate": (sum(1 for item in passes if item) / len(passes)) if passes else 0.0,
                "robustness_rate": (sum(1 for item in ci_excludes_zero if item) / len(ci_excludes_zero)) if ci_excludes_zero else 0.0,
                "complexity": ENGINEERING_COMPLEXITY.get(track_id, "unknown"),
            }
        )
    rows.sort(key=lambda row: (row["mean_delta"], row["pass_rate"], row["robustness_rate"]), reverse=True)

    lines = []
    lines.append(f"# Stage {stage} Decision Memo")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Track | Mean Delta Composite | Best Delta | Pass Rate | Robustness | Engineering Complexity |")
    lines.append("|---:|:---:|---:|---:|---:|---:|:---|")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row['track_id']} | {row['mean_delta']:.3f} | {row['best_delta']:.3f} | "
            f"{row['pass_rate']*100:.1f}% | {row['robustness_rate']*100:.1f}% | {row['complexity']} |"
        )

    lines.append("")
    lines.append("## Recommendation")
    if not rows:
        lines.append("No comparison reports available for this stage.")
    else:
        top = rows[0]
        lines.append(
            f"Prioritize `{top['track_id']}` first based on strongest composite effect size and current gate performance."
        )
    lines.append("")
    return "\n".join(lines)
