from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io import read_json, write_json
from .models import ComparisonReport, RunResult

RUN_DIR = Path("artifacts/runs")
COMPARISON_DIR = Path("artifacts/comparisons")


def save_run_result(result: RunResult) -> Path:
    path = RUN_DIR / f"{result.run_id}.json"
    write_json(path, result.to_dict())
    return path


def load_run_result(run_id: str) -> RunResult:
    path = RUN_DIR / f"{run_id}.json"
    return RunResult.from_dict(read_json(path))


def list_run_results() -> Iterable[RunResult]:
    if not RUN_DIR.exists():
        return []
    return [RunResult.from_dict(read_json(path)) for path in sorted(RUN_DIR.glob("*.json"))]


def save_comparison(report: ComparisonReport) -> Path:
    candidate = "-".join(report.candidate_run_ids)
    baseline = "-".join(report.baseline_run_ids)
    path = COMPARISON_DIR / f"cmp-{candidate}-vs-{baseline}.json"
    write_json(path, report.to_dict())
    return path


def list_comparisons() -> list[ComparisonReport]:
    if not COMPARISON_DIR.exists():
        return []
    return [ComparisonReport.from_dict(read_json(path)) for path in sorted(COMPARISON_DIR.glob("*.json"))]
