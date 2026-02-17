from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import SEED_POLICY, STAGE_BUDGET_GPU_HOURS, TRACKS
from .schema_validator import SchemaValidationError, load_schema, validate_schema


@dataclass
class ExperimentSpec:
    id: str
    track_id: str
    stage: int
    hypothesis: str
    model_variant: str
    baseline_id: str
    train_budget_gpu_h: float
    infer_budget_gpu_h: float
    max_context: int
    datasets: list[str]
    metrics: list[str]
    seeds: list[int]
    promotion_gate: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSpec":
        validate_schema(payload, load_schema("experiment_spec"))
        instance = cls(
            id=payload["id"],
            track_id=payload["track_id"],
            stage=int(payload["stage"]),
            hypothesis=payload["hypothesis"],
            model_variant=payload["model_variant"],
            baseline_id=payload["baseline_id"],
            train_budget_gpu_h=float(payload["train_budget_gpu_h"]),
            infer_budget_gpu_h=float(payload["infer_budget_gpu_h"]),
            max_context=int(payload["max_context"]),
            datasets=list(payload["datasets"]),
            metrics=list(payload["metrics"]),
            seeds=[int(seed) for seed in payload["seeds"]],
            promotion_gate=dict(payload["promotion_gate"]),
            params=dict(payload.get("params", {})),
        )
        instance.validate_policy()
        return instance

    def validate_policy(self) -> None:
        if self.track_id not in TRACKS:
            raise SchemaValidationError(f"Unknown track_id {self.track_id}. Expected one of {TRACKS}.")
        if self.stage not in {0, 1, 2, 3, 4}:
            raise SchemaValidationError(f"Invalid stage {self.stage}. Expected 0/1/2/3/4.")
        if self.stage in SEED_POLICY and len(self.seeds) < SEED_POLICY[self.stage]:
            raise SchemaValidationError(
                f"Stage {self.stage} requires at least {SEED_POLICY[self.stage]} seeds, got {len(self.seeds)}."
            )
        if self.stage in STAGE_BUDGET_GPU_HOURS:
            stage_budget = STAGE_BUDGET_GPU_HOURS[self.stage]
            total_budget = self.train_budget_gpu_h + self.infer_budget_gpu_h
            # Keep each spec within stage envelope unless it's a stage 0 baseline that can consume all budget.
            if total_budget > stage_budget:
                raise SchemaValidationError(
                    f"Spec {self.id} total budget {total_budget} exceeds stage {self.stage} budget {stage_budget}."
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    spec_id: str
    commit_sha: str
    seed: int
    train_cost: float
    infer_cost: float
    latency_p50: float
    latency_p95: float
    energy_kwh: float
    metric_values: dict[str, float]
    failure_flags: list[str]
    track_id: str
    stage: int
    model_variant: str
    benchmark_scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunResult":
        validate_schema(payload, load_schema("run_result"))
        return cls(
            run_id=payload["run_id"],
            spec_id=payload["spec_id"],
            commit_sha=payload["commit_sha"],
            seed=int(payload["seed"]),
            train_cost=float(payload["train_cost"]),
            infer_cost=float(payload["infer_cost"]),
            latency_p50=float(payload["latency_p50"]),
            latency_p95=float(payload["latency_p95"]),
            energy_kwh=float(payload["energy_kwh"]),
            metric_values={k: float(v) for k, v in payload["metric_values"].items()},
            failure_flags=list(payload["failure_flags"]),
            track_id=payload["track_id"],
            stage=int(payload["stage"]),
            model_variant=payload["model_variant"],
            benchmark_scores={k: float(v) for k, v in payload["benchmark_scores"].items()},
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonReport:
    candidate_run_ids: list[str]
    baseline_run_ids: list[str]
    delta_metrics: dict[str, float]
    significance_tests: dict[str, Any]
    pass_fail: dict[str, Any]
    candidate_stage: int
    track_id: str
    anchor_run_ids: list[str] = field(default_factory=list)
    anchor_delta_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ComparisonReport":
        validate_schema(payload, load_schema("comparison_report"))
        return cls(
            candidate_run_ids=list(payload["candidate_run_ids"]),
            baseline_run_ids=list(payload["baseline_run_ids"]),
            delta_metrics={k: float(v) for k, v in payload["delta_metrics"].items()},
            significance_tests=dict(payload["significance_tests"]),
            pass_fail=dict(payload["pass_fail"]),
            candidate_stage=int(payload["candidate_stage"]),
            track_id=payload["track_id"],
            anchor_run_ids=list(payload.get("anchor_run_ids", [])),
            anchor_delta_metrics={k: float(v) for k, v in payload.get("anchor_delta_metrics", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
