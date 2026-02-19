from __future__ import annotations

from .models import ExperimentSpec


def build_spec_template(base_spec: dict, **overrides) -> dict:
    """
    Create a generic spec by applying overrides to a base spec.

    Args:
        base_spec (dict): The base specification dictionary.
        **overrides: Key-value pairs to override in the base spec.

    Returns:
        dict: A new specification dictionary with overrides applied.
    """
    spec = base_spec.copy()
    spec.update(overrides)
    return spec


def build_cost_matched_baseline_spec(candidate: ExperimentSpec, baseline: ExperimentSpec) -> ExperimentSpec:
    """Create a baseline spec that matches candidate budget/context/seeds for fair equal-cost comparison."""
    payload = build_spec_template(
        baseline.to_dict(),
        id=f"{candidate.id}-matched-baseline",
        baseline_id=f"{candidate.id}-matched-baseline",
        train_budget_gpu_h=candidate.train_budget_gpu_h,
        infer_budget_gpu_h=candidate.infer_budget_gpu_h,
        max_context=candidate.max_context,
        seeds=list(candidate.seeds),
        promotion_gate={
            "mode": "matched_baseline",
            "source_baseline_id": baseline.id,
            "target_candidate_id": candidate.id,
        },
        params=dict(baseline.params),
    )
    return ExperimentSpec.from_dict(payload)
