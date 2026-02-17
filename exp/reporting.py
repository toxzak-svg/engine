from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .models import ComparisonReport


@dataclass(frozen=True)
class WindowArtifacts:
    marker_path: Path
    marker_timestamp: float
    marker_iso_utc: str
    run_files: list[Path]
    comparison_files: list[Path]


def load_window_artifacts(
    marker_path: str | Path,
    run_dir: str | Path = "artifacts/runs",
    comparison_dir: str | Path = "artifacts/comparisons",
) -> WindowArtifacts:
    marker = Path(marker_path)
    if not marker.exists():
        raise FileNotFoundError(f"Marker path not found: {marker}")
    marker_ts = marker.stat().st_mtime
    marker_iso = datetime.fromtimestamp(marker_ts, timezone.utc).isoformat()

    run_paths = sorted([path for path in Path(run_dir).glob("*.json") if path.stat().st_mtime >= marker_ts])
    comparison_paths = sorted([path for path in Path(comparison_dir).glob("*.json") if path.stat().st_mtime >= marker_ts])

    return WindowArtifacts(
        marker_path=marker,
        marker_timestamp=marker_ts,
        marker_iso_utc=marker_iso,
        run_files=run_paths,
        comparison_files=comparison_paths,
    )


def load_comparison_reports(paths: list[Path]) -> list[ComparisonReport]:
    reports: list[ComparisonReport] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports.append(ComparisonReport.from_dict(payload))
    return reports


def build_final_manifest(window: WindowArtifacts, memo_path: str | Path) -> dict[str, Any]:
    anchor_run_files = [path.name for path in window.run_files if "anchor-baseline" in path.name]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "marker_path": str(window.marker_path),
        "marker_time_utc": window.marker_iso_utc,
        "run_count": len(window.run_files),
        "comparison_count": len(window.comparison_files),
        "anchor_run_count": len(anchor_run_files),
        "run_files": [path.name for path in window.run_files],
        "anchor_run_files": anchor_run_files,
        "comparison_files": [path.name for path in window.comparison_files],
        "memo_path": str(memo_path),
    }


def build_consolidated_memo(
    reports: list[ComparisonReport],
    marker_iso_utc: str,
    stage_weights: dict[int, float] | None = None,
) -> str:
    if stage_weights is None:
        stage_weights = {1: 0.2, 2: 0.3, 3: 0.5}

    generated_iso = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# Consolidated Final Ranking Memo")
    lines.append("")
    lines.append(f"- Generated (UTC): {generated_iso}")
    lines.append(f"- Marker time (UTC): {marker_iso_utc}")
    lines.append(f"- Comparison artifacts included: {len(reports)}")
    lines.append("- Delta convention: anchor-relative where available, otherwise stage-baseline-relative fallback.")
    lines.append("")

    summary_stages = sorted(set(stage_weights.keys()) | {report.candidate_stage for report in reports})
    track_stage_stats: dict[str, dict[int, dict[str, float]]] = {}
    for stage in summary_stages:
        stage_reports = [report for report in reports if report.candidate_stage == stage]
        qualified = [
            report
            for report in stage_reports
            if report.pass_fail.get("equal_cost_pass", False) and report.pass_fail.get("stage_gate_pass", False)
        ]
        if stage == 1:
            promotion_limit = 3
        elif stage == 2:
            promotion_limit = 2
        else:
            promotion_limit = len(qualified)
        qualified = sorted(qualified, key=lambda item: item.delta_metrics.get("composite", float("-inf")), reverse=True)
        promoted = qualified[:promotion_limit]

        lines.extend(_render_stage_summary(stage, stage_reports, promoted, track_stage_stats))

    final_rows = _final_ranking_rows(track_stage_stats, stage_weights)
    lines.append("## Consolidated Final Ranking")
    lines.append("")
    lines.append(f"Weighted anchor score formula: `{_formula_string(stage_weights)}`.")
    lines.append("Anchor scores are normalized by available stage weights per track in the current run window.")
    lines.append("Tracks without weighted-stage data use Stage 4 anchor mean delta as fallback score when available.")
    if 4 in summary_stages and 4 not in stage_weights:
        lines.append("Stage 4 is reported as confirmatory evidence and is not included in weighted scoring.")
    lines.append("")
    lines.append("| Rank | Track | Weighted Anchor Score | Stage3 Anchor Mean Delta | Stage3 Overall Pass Rate |")
    lines.append("|---:|:---:|---:|---:|---:|")
    for idx, row in enumerate(final_rows, start=1):
        lines.append(
            f"| {idx} | {row['track']} | {row['weighted_anchor_score']:.3f} | "
            f"{row['stage3_anchor_delta']:.3f} | {row['stage3_pass_rate']*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if final_rows:
        top = final_rows[:3]
        lines.append("Advance priority order: " + ", ".join(f"{idx + 1}) {row['track']}" for idx, row in enumerate(top)))
    else:
        lines.append("No reports available for ranking.")
    lines.append("")
    return "\n".join(lines)


def _formula_string(stage_weights: dict[int, float]) -> str:
    terms = []
    for stage in sorted(stage_weights.keys()):
        terms.append(f"{stage_weights[stage]}*Stage{stage}AnchorMeanDelta")
    return " + ".join(terms)


def _render_stage_summary(
    stage: int,
    stage_reports: list[ComparisonReport],
    promoted: list[ComparisonReport],
    track_stage_stats: dict[str, dict[int, dict[str, float]]],
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Stage {stage} Summary")
    lines.append("")
    qualified = [
        report
        for report in stage_reports
        if report.pass_fail.get("equal_cost_pass", False) and report.pass_fail.get("stage_gate_pass", False)
    ]
    lines.append(f"- Candidate comparisons: {len(stage_reports)}")
    lines.append(f"- Qualified (equal-cost + stage gate): {len(qualified)}")
    lines.append(f"- Promoted: {len(promoted)}")
    if promoted:
        lines.append("- Top promoted run_ids: " + ", ".join(report.candidate_run_ids[0] for report in promoted[:5]))
    lines.append("")
    lines.append(
        "| Track | Mean Delta vs Anchor | Mean Delta vs Stage Baseline | Best Delta vs Anchor | Overall Pass Rate | CI Excludes 0 Rate |"
    )
    lines.append("|:---:|---:|---:|---:|---:|---:|")

    ranking_rows = []
    grouped: dict[str, list[ComparisonReport]] = {}
    for report in stage_reports:
        grouped.setdefault(report.track_id, []).append(report)

    for track in sorted(grouped.keys()):
        items = grouped[track]
        deltas_stage = [item.delta_metrics.get("composite", 0.0) for item in items]
        deltas_anchor = [
            item.anchor_delta_metrics.get("composite", item.delta_metrics.get("composite", 0.0))
            for item in items
        ]
        pass_rate = sum(1 for item in items if item.pass_fail.get("overall_pass", False)) / len(items)
        robust_rate = (
            sum(1 for item in items if item.significance_tests.get("ci95_excludes_zero", False)) / len(items)
        )
        mean_anchor = mean(deltas_anchor) if deltas_anchor else 0.0
        mean_stage = mean(deltas_stage) if deltas_stage else 0.0
        best_anchor = max(deltas_anchor) if deltas_anchor else 0.0
        lines.append(
            f"| {track} | {mean_anchor:.3f} | {mean_stage:.3f} | {best_anchor:.3f} | "
            f"{pass_rate*100:.1f}% | {robust_rate*100:.1f}% |"
        )
        ranking_rows.append(
            {"track": track, "mean_anchor": mean_anchor, "pass_rate": pass_rate, "robust_rate": robust_rate}
        )
        track_stage_stats.setdefault(track, {})[stage] = {
            "mean_anchor_delta": mean_anchor,
            "mean_stage_delta": mean_stage,
            "pass_rate": pass_rate,
            "robust_rate": robust_rate,
        }

    ranking_rows.sort(key=lambda row: (row["mean_anchor"], row["pass_rate"], row["robust_rate"]), reverse=True)
    lines.append("")
    if ranking_rows:
        lines.append(
            f"Stage {stage} ranking: "
            + ", ".join(f"{idx + 1}) {row['track']} ({row['mean_anchor']:.2f})" for idx, row in enumerate(ranking_rows))
        )
    else:
        lines.append(f"Stage {stage} ranking: no reports")
    lines.append("")
    return lines


def _final_ranking_rows(
    track_stage_stats: dict[str, dict[int, dict[str, float]]],
    stage_weights: dict[int, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for track, stage_stats in track_stage_stats.items():
        weighted_anchor = 0.0
        active_weight = 0.0
        for stage, weight in stage_weights.items():
            if stage in stage_stats:
                weighted_anchor += weight * stage_stats[stage].get(
                    "mean_anchor_delta",
                    stage_stats[stage].get("mean_stage_delta", 0.0),
                )
                active_weight += weight
        if active_weight > 0:
            weighted_anchor /= active_weight
        elif 4 in stage_stats:
            weighted_anchor = stage_stats[4].get(
                "mean_anchor_delta",
                stage_stats[4].get("mean_stage_delta", 0.0),
            )
        rows.append(
            {
                "track": track,
                "weighted_anchor_score": weighted_anchor,
                "stage3_anchor_delta": stage_stats.get(3, {}).get(
                    "mean_anchor_delta",
                    stage_stats.get(3, {}).get("mean_stage_delta", 0.0),
                ),
                "stage3_pass_rate": stage_stats.get(3, {}).get("pass_rate", 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["weighted_anchor_score"]),
            float(row["stage3_pass_rate"]),
            float(row["stage3_anchor_delta"]),
        ),
        reverse=True,
    )
    return rows
