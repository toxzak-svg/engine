from __future__ import annotations

from .models import ExperimentSpec


def build_cost_matched_baseline_spec(candidate: ExperimentSpec, baseline: ExperimentSpec) -> ExperimentSpec:
    """Create a baseline spec that matches candidate budget/context/seeds for fair equal-cost comparison."""
    payload = baseline.to_dict()
    payload["id"] = f"{candidate.id}-matched-baseline"
    payload["baseline_id"] = payload["id"]
    payload["train_budget_gpu_h"] = candidate.train_budget_gpu_h
    payload["infer_budget_gpu_h"] = candidate.infer_budget_gpu_h
    payload["max_context"] = candidate.max_context
    payload["seeds"] = list(candidate.seeds)
    payload["promotion_gate"] = {
        "mode": "matched_baseline",
        "source_baseline_id": baseline.id,
        "target_candidate_id": candidate.id,
    }
    payload["params"] = dict(baseline.params)
    payload["params"]["cost_matched_to"] = candidate.id
    return ExperimentSpec.from_dict(payload)
