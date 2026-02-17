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
from exp.models import ExperimentSpec
from exp.simulator import simulate_run
from exp.store import save_comparison, save_run_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full stage batch (baseline + experiments per track)")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument("--all-seeds", action="store_true", help="Run all declared seeds instead of the first seed")
    args = parser.parse_args()

    stage_dir = Path("specs") / f"stage{args.stage}"
    if not stage_dir.exists():
        raise FileNotFoundError(f"Missing spec directory {stage_dir}")

    run_count = 0
    comparison_count = 0
    for track in range(1, 7):
        baseline_spec = ExperimentSpec.from_dict(read_yaml_or_json(stage_dir / f"t{track}_baseline.yaml"))
        seeds = baseline_spec.seeds if args.all_seeds else [baseline_spec.seeds[0]]
        for seed in seeds:
            baseline = simulate_run(
                baseline_spec,
                seed=seed,
                run_id=f"batch-s{args.stage}-t{track}-baseline-{seed}-{uuid.uuid4().hex[:6]}",
                commit_sha="local",
            )
            save_run_result(baseline)
            run_count += 1
            if args.stage == 0:
                continue
            for variant in ("e1", "e2", "e3"):
                candidate_spec = ExperimentSpec.from_dict(read_yaml_or_json(stage_dir / f"t{track}_{variant}.yaml"))
                candidate = simulate_run(
                    candidate_spec,
                    seed=seed,
                    run_id=f"batch-s{args.stage}-t{track}-{variant}-{seed}-{uuid.uuid4().hex[:6]}",
                    commit_sha="local",
                )
                save_run_result(candidate)
                run_count += 1
                report = compare_runs(candidate, baseline)
                save_comparison(report)
                comparison_count += 1

    print(
        f"completed stage {args.stage}: wrote {run_count} run artifacts "
        f"and {comparison_count} comparison artifacts"
    )


if __name__ == "__main__":
    main()
