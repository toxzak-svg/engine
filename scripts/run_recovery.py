from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.compare import compare_runs
from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec, RunResult
from exp.spec_utils import build_cost_matched_baseline_spec
from exp.simulator import simulate_run
from exp.store import save_comparison, save_run_result
from exp.runners.inverse_arm_runners import simulate_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused recovery specs for T4/T5.")
    parser.add_argument("--stage", type=int, choices=[2, 3], required=True)
    parser.add_argument("--track", choices=["T4", "T5", "all"], default="all")
    parser.add_argument("--all-seeds", action="store_true", help="Run all declared seeds instead of the first seed")
    args = parser.parse_args()

    recovery_dir = Path("specs") / "recovery" / f"stage{args.stage}"
    if not recovery_dir.exists():
        raise FileNotFoundError(f"Missing recovery directory {recovery_dir}")

    spec_paths = sorted(recovery_dir.glob("*.yaml"))
    if args.track != "all":
        spec_paths = [path for path in spec_paths if path.name.startswith(args.track.lower())]

    if not spec_paths:
        raise RuntimeError(f"No recovery specs matched in {recovery_dir}")

    anchor_spec_path = Path("specs") / f"stage{args.stage}" / "anchor_baseline.yaml"
    anchor_spec = ExperimentSpec.from_dict(read_yaml_or_json(anchor_spec_path)) if anchor_spec_path.exists() else None
    anchor_runs_by_seed: dict[int, RunResult] = {}

    run_count = 0
    comparison_count = 0
    for spec_path in spec_paths:
        spec = ExperimentSpec.from_dict(read_yaml_or_json(spec_path))
        baseline_spec_path = Path("specs") / f"stage{args.stage}" / f"{spec.track_id.lower()}_baseline.yaml"
        baseline_spec = ExperimentSpec.from_dict(read_yaml_or_json(baseline_spec_path))
        matched_baseline_spec = build_cost_matched_baseline_spec(spec, baseline_spec)
        seeds = spec.seeds if args.all_seeds else [spec.seeds[0]]
        for seed in seeds:
            if anchor_spec is not None and seed not in anchor_runs_by_seed:
                anchor = simulate_run(
                    anchor_spec,
                    seed=seed,
                    run_id=f"recovery-s{args.stage}-anchor-baseline-{seed}-{uuid.uuid4().hex[:6]}",
                    commit_sha="local",
                )
                save_run_result(anchor)
                anchor_runs_by_seed[seed] = anchor
                run_count += 1
            baseline = simulate_run(
                matched_baseline_spec,
                seed=seed,
                run_id=f"recovery-s{args.stage}-{spec.track_id.lower()}-baseline-matched-{seed}-{uuid.uuid4().hex[:6]}",
                commit_sha="local",
            )
            candidate = simulate_run(
                spec,
                seed=seed,
                run_id=f"recovery-s{args.stage}-{spec.id}-{seed}-{uuid.uuid4().hex[:6]}",
                commit_sha="local",
            )
            save_run_result(baseline)
            save_run_result(candidate)
            report = compare_runs(candidate, baseline, anchor=anchor_runs_by_seed.get(seed))
            save_comparison(report)
            run_count += 2
            comparison_count += 1

    print(
        f"completed recovery stage {args.stage}: wrote {run_count} run artifacts "
        f"and {comparison_count} comparison artifacts"
    )


if __name__ == "__main__":
    main()
