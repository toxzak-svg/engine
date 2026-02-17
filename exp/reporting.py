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


def _anchor_delta(report: ComparisonReport) -> float:
    return report.anchor_delta_metrics.get("composite", report.delta_metrics.get("composite", 0.0))


def _ci_width(report: ComparisonReport) -> float:
    ci95 = report.significance_tests.get("ci95", [])
    if not isinstance(ci95, list) or len(ci95) != 2:
        return 0.0
    return float(ci95[1]) - float(ci95[0])


def _is_recovery_report(report: ComparisonReport) -> bool:
    candidate = report.candidate_run_ids[0] if report.candidate_run_ids else ""
    return candidate.startswith("recovery-")


def _promotion_track_limit(stage: int) -> int:
    if stage == 1:
        return 3
    if stage == 2:
        return 2
    if stage == 3:
        return 1
    return 0


def _select_promoted_reports(stage_reports: list[ComparisonReport], stage: int) -> list[ComparisonReport]:
    promotion_limit = _promotion_track_limit(stage)
    if promotion_limit <= 0:
        return []

    grouped: dict[str, list[ComparisonReport]] = {}
    for report in stage_reports:
        grouped.setdefault(report.track_id, []).append(report)

    track_rows: list[dict[str, Any]] = []
    for track, items in grouped.items():
        qualified = [item for item in items if item.pass_fail.get("overall_pass", False)]
        if not qualified:
            continue
        best = max(qualified, key=_anchor_delta)
        track_rows.append(
            {
                "track": track,
                "pass_rate": sum(1 for item in items if item.pass_fail.get("overall_pass", False)) / len(items),
                "mean_anchor_qualified": mean(_anchor_delta(item) for item in qualified),
                "mean_anchor_all": mean(_anchor_delta(item) for item in items),
                "best": best,
            }
        )

    track_rows.sort(
        key=lambda row: (
            row["pass_rate"],
            row["mean_anchor_qualified"],
            row["mean_anchor_all"],
        ),
        reverse=True,
    )
    return [row["best"] for row in track_rows[:promotion_limit]]


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
    lines.append("- Data origin: simulator-generated harness outputs; use as internal relative signal, not empirical training proof.")
    lines.append("")

    summary_stages = sorted(set(stage_weights.keys()) | {report.candidate_stage for report in reports})
    track_stage_stats: dict[str, dict[int, dict[str, float]]] = {}
    for stage in summary_stages:
        stage_reports_all = [report for report in reports if report.candidate_stage == stage]
        recovery_reports = [report for report in stage_reports_all if _is_recovery_report(report)]
        core_reports = [report for report in stage_reports_all if not _is_recovery_report(report)]
        stage_reports = core_reports if core_reports else stage_reports_all
        promoted = _select_promoted_reports(stage_reports, stage)
        lines.extend(_render_stage_summary(stage, stage_reports, recovery_reports, promoted, track_stage_stats))

    final_rows = _final_ranking_rows(track_stage_stats, stage_weights)
    lines.append("## Consolidated Final Ranking")
    lines.append("")
    lines.append(f"Weighted anchor score formula: `{_formula_string(stage_weights)}`.")
    lines.append(
        f"Decision score formula: `WeightedAnchorScore * WeightedPassRate`, where `WeightedPassRate` "
        f"uses the same stage weights ({_formula_string(stage_weights).replace('AnchorMeanDelta', 'OverallPassRate')})."
    )
    lines.append("Anchor scores are normalized by available stage weights per track in the current run window.")
    lines.append("Primary ranking stats exclude recovery comparisons when core batch comparisons are present for that stage.")
    lines.append("Tracks without weighted-stage data use Stage 4 anchor mean delta as fallback score when available.")
    if 4 in summary_stages and 4 not in stage_weights:
        lines.append("Stage 4 is reported as confirmatory evidence and is not included in weighted scoring.")
    lines.append("")
    lines.append(
        "| Rank | Track | Decision Score | Weighted Anchor Score | Weighted Pass Rate | "
        "Weighted Mean CI Width | Stage3 Anchor Mean Delta | Stage3 Overall Pass Rate |"
    )
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(final_rows, start=1):
        lines.append(
            f"| {idx} | {row['track']} | {row['decision_score']:.3f} | {row['weighted_anchor_score']:.3f} | "
            f"{row['weighted_pass_rate']*100:.1f}% | {row['weighted_ci_width']:.3f} | "
            f"{row['stage3_anchor_delta']:.3f} | {row['stage3_pass_rate']*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if final_rows:
        top = [row for row in final_rows if row["weighted_pass_rate"] > 0.0][:3]
        if not top:
            top = final_rows[:3]
        lines.append(
            "Advance priority order (pass-adjusted): "
            + ", ".join(f"{idx + 1}) {row['track']}" for idx, row in enumerate(top))
        )
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
    recovery_reports: list[ComparisonReport],
    promoted: list[ComparisonReport],
    track_stage_stats: dict[str, dict[int, dict[str, float]]],
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Stage {stage} Summary")
    lines.append("")
    qualified = [report for report in stage_reports if report.pass_fail.get("overall_pass", False)]
    promoted_tracks = sorted({report.track_id for report in promoted})
    lines.append(f"- Candidate comparisons (core): {len(stage_reports)}")
    if recovery_reports:
        lines.append(f"- Candidate comparisons (recovery supplemental): {len(recovery_reports)}")
    lines.append(f"- Qualified (overall pass): {len(qualified)}")
    lines.append(f"- Promoted tracks: {len(promoted_tracks)}")
    lines.append(f"- Promoted candidates: {len(promoted)}")
    if promoted:
        lines.append("- Top promoted run_ids: " + ", ".join(report.candidate_run_ids[0] for report in promoted[:5]))
    lines.append("")
    lines.append(
        "| Track | Mean Delta vs Anchor | Mean Delta vs Stage Baseline | Best Delta vs Anchor | "
        "Overall Pass Rate | CI Excludes 0 Rate | Mean CI Width |"
    )
    lines.append("|:---:|---:|---:|---:|---:|---:|---:|")

    ranking_rows = []
    grouped: dict[str, list[ComparisonReport]] = {}
    for report in stage_reports:
        grouped.setdefault(report.track_id, []).append(report)

    for track in sorted(grouped.keys()):
        items = grouped[track]
        deltas_stage = [item.delta_metrics.get("composite", 0.0) for item in items]
        deltas_anchor = [_anchor_delta(item) for item in items]
        ci_widths = [_ci_width(item) for item in items]
        pass_rate = sum(1 for item in items if item.pass_fail.get("overall_pass", False)) / len(items)
        robust_rate = (
            sum(1 for item in items if item.significance_tests.get("ci95_excludes_zero", False)) / len(items)
        )
        mean_anchor = mean(deltas_anchor) if deltas_anchor else 0.0
        mean_stage = mean(deltas_stage) if deltas_stage else 0.0
        best_anchor = max(deltas_anchor) if deltas_anchor else 0.0
        mean_ci_width = mean(ci_widths) if ci_widths else 0.0
        decision_score = mean_anchor * pass_rate
        lines.append(
            f"| {track} | {mean_anchor:.3f} | {mean_stage:.3f} | {best_anchor:.3f} | {pass_rate*100:.1f}% | "
            f"{robust_rate*100:.1f}% | {mean_ci_width:.3f} |"
        )
        ranking_rows.append(
            {
                "track": track,
                "mean_anchor": mean_anchor,
                "pass_rate": pass_rate,
                "robust_rate": robust_rate,
                "mean_ci_width": mean_ci_width,
                "decision_score": decision_score,
            }
        )
        track_stage_stats.setdefault(track, {})[stage] = {
            "mean_anchor_delta": mean_anchor,
            "mean_stage_delta": mean_stage,
            "pass_rate": pass_rate,
            "robust_rate": robust_rate,
            "mean_ci_width": mean_ci_width,
            "decision_score": decision_score,
        }

    ranking_rows.sort(
        key=lambda row: (row["decision_score"], row["pass_rate"], row["mean_anchor"], -row["mean_ci_width"]),
        reverse=True,
    )
    lines.append("")
    if ranking_rows:
        lines.append(
            f"Stage {stage} ranking: "
            + ", ".join(
                f"{idx + 1}) {row['track']} (decision={row['decision_score']:.2f}, anchor={row['mean_anchor']:.2f})"
                for idx, row in enumerate(ranking_rows)
            )
        )
    else:
        lines.append(f"Stage {stage} ranking: no reports")
    if recovery_reports:
        lines.append("")
        lines.append("Recovery snapshot (supplemental; excluded from core ranking stats):")
        lines.append("")
        lines.append("| Track | Recovery Comparisons | Mean Anchor Delta | Overall Pass Rate |")
        lines.append("|:---:|---:|---:|---:|")
        recovery_grouped: dict[str, list[ComparisonReport]] = {}
        for report in recovery_reports:
            recovery_grouped.setdefault(report.track_id, []).append(report)
        for track in sorted(recovery_grouped.keys()):
            items = recovery_grouped[track]
            lines.append(
                f"| {track} | {len(items)} | {mean(_anchor_delta(item) for item in items):.3f} | "
                f"{(sum(1 for item in items if item.pass_fail.get('overall_pass', False)) / len(items)) * 100:.1f}% |"
            )
    lines.append("")
    return lines


def _final_ranking_rows(
    track_stage_stats: dict[str, dict[int, dict[str, float]]],
    stage_weights: dict[int, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for track, stage_stats in track_stage_stats.items():
        weighted_anchor = 0.0
        weighted_pass = 0.0
        weighted_ci_width = 0.0
        active_weight = 0.0
        for stage, weight in stage_weights.items():
            if stage in stage_stats:
                weighted_anchor += weight * stage_stats[stage].get(
                    "mean_anchor_delta",
                    stage_stats[stage].get("mean_stage_delta", 0.0),
                )
                weighted_pass += weight * stage_stats[stage].get("pass_rate", 0.0)
                weighted_ci_width += weight * stage_stats[stage].get("mean_ci_width", 0.0)
                active_weight += weight
        if active_weight > 0:
            weighted_anchor /= active_weight
            weighted_pass /= active_weight
            weighted_ci_width /= active_weight
        elif 4 in stage_stats:
            weighted_anchor = stage_stats[4].get(
                "mean_anchor_delta",
                stage_stats[4].get("mean_stage_delta", 0.0),
            )
            weighted_pass = stage_stats[4].get("pass_rate", 0.0)
            weighted_ci_width = stage_stats[4].get("mean_ci_width", 0.0)
        decision_score = weighted_anchor * weighted_pass
        rows.append(
            {
                "track": track,
                "decision_score": decision_score,
                "weighted_anchor_score": weighted_anchor,
                "weighted_pass_rate": weighted_pass,
                "weighted_ci_width": weighted_ci_width,
                "stage3_anchor_delta": stage_stats.get(3, {}).get(
                    "mean_anchor_delta",
                    stage_stats.get(3, {}).get("mean_stage_delta", 0.0),
                ),
                "stage3_pass_rate": stage_stats.get(3, {}).get("pass_rate", 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["decision_score"]),
            float(row["weighted_anchor_score"]),
            float(row["weighted_pass_rate"]),
            -float(row["weighted_ci_width"]),
            float(row["stage3_pass_rate"]),
            float(row["stage3_anchor_delta"]),
        ),
        reverse=True,
    )
    return rows
