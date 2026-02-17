from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.compare import compare_runs
from exp.gating import gate_stage
from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec, RunResult
from exp.reporting import load_comparison_reports, load_window_artifacts
from exp.simulator import simulate_run
from exp.store import list_comparisons, save_comparison, save_run_result


def _discover_tracks(stage_dir: Path) -> list[str]:
    tracks: list[str] = []
    for baseline in sorted(stage_dir.glob("t*_baseline.yaml")):
        match = re.match(r"^(t\d+)_baseline\.yaml$", baseline.name)
        if match:
            tracks.append(match.group(1))
    return tracks


def _parse_track_overrides(value: str) -> list[str]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    normalized: list[str] = []
    for token in tokens:
        token_lower = token.lower()
        if not re.match(r"^t\d+$", token_lower):
            raise ValueError(f"Invalid track identifier '{token}'. Expected tokens like T3,T4.")
        normalized.append(token_lower)
    return normalized


def _select_tracks_for_stage(
    stage: int,
    discovered_tracks: list[str],
    manual_tracks: str | None,
    selection_marker: str | None,
) -> list[str]:
    if manual_tracks:
        requested = _parse_track_overrides(manual_tracks)
        missing = [track for track in requested if track not in discovered_tracks]
        if missing:
            raise RuntimeError(
                f"Requested tracks not found in specs/stage{stage}: {', '.join(missing)} "
                f"(available: {', '.join(discovered_tracks)})"
            )
        return requested

    if stage not in {2, 3}:
        return discovered_tracks

    if selection_marker:
        window = load_window_artifacts(selection_marker)
        reports = load_comparison_reports(window.comparison_files)
    else:
        reports = list_comparisons()

    summary = gate_stage(stage - 1, reports)
    promoted_tracks = [track.lower() for track in summary.get("promoted_tracks", [])]
    if not promoted_tracks:
        raise RuntimeError(
            f"No promoted tracks found from stage {stage - 1}. "
            "Run stage gating first or override with --tracks."
        )

    selected = [track for track in discovered_tracks if track in promoted_tracks]
    if not selected:
        raise RuntimeError(
            f"Promoted tracks from stage {stage - 1} ({', '.join(promoted_tracks)}) do not match "
            f"stage {stage} specs ({', '.join(discovered_tracks)}). Use --tracks to override."
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full stage batch (baseline + experiments per track)")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3, 4], required=True)
    parser.add_argument("--all-seeds", action="store_true", help="Run all declared seeds instead of the first seed")
    parser.add_argument(
        "--tracks",
        help="Optional comma-separated track list (e.g., T3,T4). "
        "By default, stage2/stage3 auto-select promoted tracks from the previous stage.",
    )
    parser.add_argument(
        "--selection-marker",
        help="Optional marker path used to limit prior-stage comparisons for auto track selection.",
    )
    args = parser.parse_args()

    stage_dir = Path("specs") / f"stage{args.stage}"
    if not stage_dir.exists():
        raise FileNotFoundError(f"Missing spec directory {stage_dir}")

    track_prefixes = _discover_tracks(stage_dir)
    if not track_prefixes:
        raise RuntimeError(f"No baseline specs found in {stage_dir}")
    selected_tracks = _select_tracks_for_stage(
        stage=args.stage,
        discovered_tracks=track_prefixes,
        manual_tracks=args.tracks,
        selection_marker=args.selection_marker,
    )

    anchor_spec_path = stage_dir / "anchor_baseline.yaml"
    anchor_spec = ExperimentSpec.from_dict(read_yaml_or_json(anchor_spec_path)) if anchor_spec_path.exists() else None
    anchor_runs_by_seed: dict[int, RunResult] = {}

    run_count = 0
    comparison_count = 0
    for track_prefix in selected_tracks:
        baseline_spec = ExperimentSpec.from_dict(read_yaml_or_json(stage_dir / f"{track_prefix}_baseline.yaml"))
        seeds = baseline_spec.seeds if args.all_seeds else [baseline_spec.seeds[0]]
        candidate_specs = sorted(
            [
                path
                for path in stage_dir.glob(f"{track_prefix}_*.yaml")
                if path.name != f"{track_prefix}_baseline.yaml"
            ]
        )
        for seed in seeds:
            if anchor_spec is not None and seed not in anchor_runs_by_seed:
                anchor_result = simulate_run(
                    anchor_spec,
                    seed=seed,
                    run_id=f"batch-s{args.stage}-anchor-baseline-{seed}-{uuid.uuid4().hex[:6]}",
                    commit_sha="local",
                )
                save_run_result(anchor_result)
                anchor_runs_by_seed[seed] = anchor_result
                run_count += 1
            baseline = simulate_run(
                baseline_spec,
                seed=seed,
                run_id=f"batch-s{args.stage}-{track_prefix}-baseline-{seed}-{uuid.uuid4().hex[:6]}",
                commit_sha="local",
            )
            save_run_result(baseline)
            run_count += 1
            if args.stage == 0:
                continue
            for spec_path in candidate_specs:
                candidate_spec = ExperimentSpec.from_dict(read_yaml_or_json(spec_path))
                variant = candidate_spec.id.replace(f"s{args.stage}-{candidate_spec.track_id.lower()}-", "")
                candidate = simulate_run(
                    candidate_spec,
                    seed=seed,
                    run_id=f"batch-s{args.stage}-{track_prefix}-{variant}-{seed}-{uuid.uuid4().hex[:6]}",
                    commit_sha="local",
                )
                save_run_result(candidate)
                run_count += 1
                report = compare_runs(candidate, baseline, anchor=anchor_runs_by_seed.get(seed))
                save_comparison(report)
                comparison_count += 1

    print(
        f"completed stage {args.stage}: tracks={','.join(selected_tracks)}; "
        f"wrote {run_count} run artifacts and {comparison_count} comparison artifacts"
    )


if __name__ == "__main__":
    main()
