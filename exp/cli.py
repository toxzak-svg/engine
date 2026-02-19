from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .compare import compare_runs
from .gating import gate_stage
from .io import read_yaml_or_json
from .memo import build_decision_memo
from .models import ExperimentSpec
from .reporting import load_comparison_reports, load_window_artifacts
from .simulator import evaluate_track_pass, simulate_run
from .store import (
    list_comparisons,
    load_run_result,
    save_comparison,
    save_run_result,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exp", description="Experiment harness CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a spec and write RunResult artifacts")
    run_parser.add_argument("--spec", required=True, help="Path to ExperimentSpec YAML/JSON")
    run_parser.add_argument("--seed", type=int, help="Single seed override")
    run_parser.add_argument("--all-seeds", action="store_true", help="Run all seeds declared in the spec")

    eval_parser = subparsers.add_parser("eval", help="Evaluate one run artifact")
    eval_parser.add_argument("--run", required=True, help="Run ID")
    eval_parser.add_argument("--baseline", help="Optional baseline run ID for delta evaluation")

    compare_parser = subparsers.add_parser("compare", help="Compare candidate and baseline runs")
    compare_parser.add_argument("--candidate", required=True, help="Candidate run ID")
    compare_parser.add_argument("--baseline", required=True, help="Baseline run ID")
    compare_parser.add_argument("--anchor", help="Optional anchor run ID for anchor-relative deltas")

    gate_parser = subparsers.add_parser("gate", help="Apply stage gates using comparison reports")
    gate_parser.add_argument("--stage", required=True, type=int, choices=[1, 2, 3, 4], help="Completed stage")
    gate_parser.add_argument(
        "--marker",
        help="Optional run-window marker path; when set, only comparison artifacts newer than marker are used.",
    )

    memo_parser = subparsers.add_parser("memo", help="Generate a stage decision memo from comparison reports")
    memo_parser.add_argument("--stage", required=True, type=int, choices=[1, 2, 3, 4], help="Stage to summarize")
    memo_parser.add_argument(
        "--marker",
        help="Optional run-window marker path; when set, only comparison artifacts newer than marker are used.",
    )

    generate_parser = subparsers.add_parser("generate", help="Generate experiment specifications")
    generate_parser.add_argument("--stage", type=int, required=True, help="Stage number to generate specs for")
    generate_parser.add_argument("--type", choices=["main", "recovery"], default="main", help="Type of specs to generate")

    package_parser = subparsers.add_parser("package", help="Package the project for PyPI publishing")
    package_parser.add_argument("--version", required=True, help="Version number for the package")
    package_parser.add_argument("--output", default="dist/", help="Output directory for the package")

    test_parser = subparsers.add_parser("test", help="Run the test suite")
    test_parser.add_argument("--ci", action="store_true", help="Run tests in CI mode")

    publish_parser = subparsers.add_parser("publish", help="Publish the package to PyPI")
    publish_parser.add_argument("--repository", default="pypi", help="Repository to publish to (default: pypi)")

    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "gate":
            return _cmd_gate(args)
        if args.command == "memo":
            return _cmd_memo(args)
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "package":
            return _cmd_package(args)
        if args.command == "test":
            return _cmd_test(args)
        if args.command == "publish":
            return _cmd_publish(args)
        parser.print_help()
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_run(args: argparse.Namespace) -> int:
    spec_payload = read_yaml_or_json(args.spec)
    spec = ExperimentSpec.from_dict(spec_payload)

    if args.all_seeds:
        seeds = spec.seeds
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [spec.seeds[0]]

    commit_sha = _git_commit_sha()
    output: list[dict[str, Any]] = []
    for seed in seeds:
        run_id = f"{spec.id}-s{seed}-{uuid.uuid4().hex[:8]}"
        result = simulate_run(spec=spec, seed=seed, run_id=run_id, commit_sha=commit_sha)
        path = save_run_result(result)
        output.append(
            {
                "run_id": result.run_id,
                "path": str(path),
                "stage": result.stage,
                "track_id": result.track_id,
                "model_variant": result.model_variant,
                "composite": result.metric_values["composite"],
                "failure_flags": result.failure_flags,
            }
        )

    print(json.dumps({"runs": output}, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    run = load_run_result(args.run)
    payload: dict[str, Any] = {
        "run_id": run.run_id,
        "spec_id": run.spec_id,
        "track_id": run.track_id,
        "stage": run.stage,
        "model_variant": run.model_variant,
        "composite": run.metric_values["composite"],
        "metrics": run.metric_values,
        "failure_flags": run.failure_flags,
    }
    if args.baseline:
        baseline = load_run_result(args.baseline)
        delta = {k: round(run.metric_values[k] - baseline.metric_values[k], 4) for k in run.metric_values if k in baseline.metric_values}
        payload["baseline"] = baseline.run_id
        payload["delta_metrics"] = delta
        payload["track_specific_pass"] = evaluate_track_pass(run, baseline, delta)
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    candidate = load_run_result(args.candidate)
    baseline = load_run_result(args.baseline)
    anchor = load_run_result(args.anchor) if args.anchor else None
    if candidate.track_id != baseline.track_id:
        raise ValueError(
            f"track mismatch: candidate track {candidate.track_id} vs baseline track {baseline.track_id}. "
            "Compare each track against its own baseline."
        )
    report = compare_runs(candidate, baseline, anchor=anchor)
    path = save_comparison(report)
    print(
        json.dumps(
            {
                "comparison_path": str(path),
                "candidate_run_id": candidate.run_id,
                "baseline_run_id": baseline.run_id,
                "anchor_run_id": anchor.run_id if anchor else None,
                "delta_composite": report.delta_metrics.get("composite", 0.0),
                "anchor_delta_composite": report.anchor_delta_metrics.get("composite", None),
                "overall_pass": report.pass_fail.get("overall_pass", False),
                "pass_fail": report.pass_fail,
                "significance_tests": report.significance_tests,
            },
            indent=2,
        )
    )
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    if args.marker:
        window = load_window_artifacts(args.marker)
        reports = load_comparison_reports(window.comparison_files)
    else:
        reports = list_comparisons()
    summary = gate_stage(args.stage, reports)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_memo(args: argparse.Namespace) -> int:
    if args.marker:
        window = load_window_artifacts(args.marker)
        reports = load_comparison_reports(window.comparison_files)
    else:
        reports = list_comparisons()
    memo = build_decision_memo(args.stage, reports)
    out_dir = Path("artifacts/memos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"stage{args.stage}_decision.md"
    out_file.write_text(memo, encoding="utf-8")
    print(json.dumps({"memo_path": str(out_file)}, indent=2))
    print("")
    print(memo)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    stage = args.stage
    spec_type = args.type
    spec_path = Path("specs") / f"stage{stage}_{spec_type}.yaml"
    spec_payload = {
        "stage": stage,
        "type": spec_type,
        "seeds": [1, 2, 3, 4, 5],
        "track_id": "track1",
        "model_variant": "model1",
        "composite": 0.5,
        "failure_flags": [False, False, False, False, False],
    }
    spec = ExperimentSpec.from_dict(spec_payload)
    print(f"Generated spec: {spec_path}")
    return 0


def _cmd_package(args: argparse.Namespace) -> int:
    version = args.version
    output_dir = args.output
    print(f"Packaging version {version} to {output_dir}")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    print("Running tests...")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    print("Publishing package...")
    return 0


def _git_commit_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
