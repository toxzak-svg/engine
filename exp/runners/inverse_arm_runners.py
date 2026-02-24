"""Runners for inverse-directed experiment arms.

These runners implement the execution logic for each inverse arm type,
providing specialized simulation and evaluation for techniques that
improve quality through constraint rather than scale.
"""

from __future__ import annotations

import hashlib
import random
from statistics import mean
from typing import Any

from ..constants import (
    BASE_BENCHMARK_SCORES,
    COMPOSITE_WEIGHTS,
    CONSISTENCY_DATASETS,
    LONG_CONTEXT_DATASETS,
    QUALITY_DELTA_SCALE,
    REASONING_DATASETS,
    STAGE_GAIN_MULTIPLIER,
    VariantEffect,
)
from ..inverse_arms import (
    INVERSE_ARM_METRIC_BASELINES,
    INVERSE_ARM_VARIANT_EFFECTS,
    InverseArmConfig,
    InverseArmType,
    get_inverse_arm_config,
    get_inverse_arm_for_track,
    get_inverse_arm_variant_effect,
)
from ..models import ExperimentSpec, RunResult
from .base import BaseRunner, RunnerConfig, RunnerResult, RunnerStatus
from exp.shared import clamp_score, simulate_benchmarks


def stable_rng(*parts: object) -> random.Random:
    """Create a stable RNG from parts."""
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(seed)


def compute_composite(long_context: float, reasoning: float, consistency: float) -> float:
    """Compute composite score from components."""
    return (
        COMPOSITE_WEIGHTS["long_context"] * long_context
        + COMPOSITE_WEIGHTS["reasoning"] * reasoning
        + COMPOSITE_WEIGHTS["consistency"] * consistency
    )


class InverseArmRunner(BaseRunner):
    """Base runner for inverse-directed experiment arms.
    
    This runner provides specialized simulation logic for inverse arms,
    which bet against the usual "more layers / more tokens / bigger model" story.
    """
    
    name: str = "inverse_arm"
    description: str = "Base runner for inverse-directed experiment arms"
    supported_tracks: list[str] = ["T7", "T8", "T9", "T10", "T11", "T12", "T13"]
    
    def __init__(self, config: RunnerConfig | None = None, arm_type: InverseArmType | None = None):
        super().__init__(config)
        self.arm_type = arm_type
    
    def execute(
        self,
        spec: ExperimentSpec,
        seed: int,
        run_id: str,
        commit_sha: str,
    ) -> RunnerResult:
        """Execute an inverse arm experiment.
        
        Args:
            spec: The experiment specification
            seed: Random seed for reproducibility
            run_id: Unique identifier for this run
            commit_sha: Git commit SHA for tracking
            
        Returns:
            RunnerResult containing the execution outcome
        """
        self._set_status(RunnerStatus.RUNNING)
        
        try:
            # Determine arm type from track
            arm_type = self.arm_type or get_inverse_arm_for_track(spec.track_id)
            if arm_type is None:
                return RunnerResult(
                    success=False,
                    run_result=None,
                    error_message=f"Unknown inverse arm track: {spec.track_id}",
                    status=RunnerStatus.FAILED,
                )
            
            # Get variant effect
            effect = get_inverse_arm_variant_effect(spec.model_variant)
            if effect is None:
                # Fall back to arm default effect
                arm_config = get_inverse_arm_config(arm_type)
                effect = arm_config.variant_effect
            
            # Run simulation
            run_result = self._simulate_inverse_arm(
                spec=spec,
                effect=effect,
                arm_type=arm_type,
                seed=seed,
                run_id=run_id,
                commit_sha=commit_sha,
            )
            
            self._set_status(RunnerStatus.COMPLETED)
            return RunnerResult(
                success=True,
                run_result=run_result,
                status=RunnerStatus.COMPLETED,
            )
            
        except Exception as e:
            self._set_status(RunnerStatus.FAILED)
            return RunnerResult(
                success=False,
                run_result=None,
                error_message=str(e),
                status=RunnerStatus.FAILED,
            )
    
    def _simulate_inverse_arm(
        self,
        spec: ExperimentSpec,
        effect: VariantEffect,
        arm_type: InverseArmType,
        seed: int,
        run_id: str,
        commit_sha: str,
    ) -> RunResult:
        """Simulate an inverse arm run.
        
        Args:
            spec: The experiment specification
            effect: The variant effect to apply
            arm_type: The type of inverse arm
            seed: Random seed
            run_id: Unique run identifier
            commit_sha: Git commit SHA
            
        Returns:
            RunResult with simulated metrics
        """
        rng = stable_rng(spec.id, spec.model_variant, seed, spec.stage)
        stage_multiplier = STAGE_GAIN_MULTIPLIER.get(spec.stage, 1.0)
        
        # Simulate benchmark scores
        benchmark_scores = simulate_benchmarks(spec, effect, stage_multiplier, rng)
        
        # Compute component scores
        long_context = mean([benchmark_scores[name] for name in LONG_CONTEXT_DATASETS])
        reasoning = mean([benchmark_scores[name] for name in REASONING_DATASETS])
        consistency = mean([benchmark_scores[name] for name in CONSISTENCY_DATASETS])
        fluency = benchmark_scores["fluency"]
        composite = compute_composite(long_context, reasoning, consistency)
        
        # Compute costs
        train_cost = spec.train_budget_gpu_h * (1.0 + rng.uniform(-0.004, 0.004))
        infer_cost = spec.infer_budget_gpu_h * (1.0 + rng.uniform(-0.004, 0.004))
        infer_cost = self._adjust_infer_cost_for_params(spec, infer_cost, arm_type)
        
        # Compute latency
        baseline_latency = 120.0 + (spec.max_context / 4000.0)
        latency_p50 = baseline_latency * (1.0 + (effect.latency_delta_pct / 100.0) * stage_multiplier)
        latency_p50 = max(20.0, latency_p50 + rng.uniform(-1.0, 1.0))
        latency_p95 = latency_p50 * 1.28
        
        # Compute energy
        baseline_energy = (train_cost + infer_cost) * 0.38
        energy_kwh = baseline_energy * (1.0 + (effect.energy_delta_pct / 100.0) * stage_multiplier)
        
        # Build metric values
        metric_values = {
            "long_context": round(long_context, 4),
            "reasoning": round(reasoning, 4),
            "consistency": round(consistency, 4),
            "fluency": round(fluency, 4),
            "composite": round(composite, 4),
        }
        
        # Add arm-specific metrics
        arm_metrics = self._compute_arm_specific_metrics(
            spec=spec,
            effect=effect,
            arm_type=arm_type,
            stage_multiplier=stage_multiplier,
            rng=rng,
            long_context=long_context,
            reasoning=reasoning,
            consistency=consistency,
        )
        metric_values.update(arm_metrics)
        
        # Compute failure flags
        failure_flags = self._compute_failure_flags(spec, metric_values, arm_type)
        
        return RunResult(
            run_id=run_id,
            spec_id=spec.id,
            commit_sha=commit_sha,
            seed=seed,
            train_cost=round(train_cost, 4),
            infer_cost=round(infer_cost, 4),
            latency_p50=round(latency_p50, 4),
            latency_p95=round(latency_p95, 4),
            energy_kwh=round(energy_kwh, 4),
            metric_values=metric_values,
            failure_flags=failure_flags,
            track_id=spec.track_id,
            stage=spec.stage,
            model_variant=spec.model_variant,
            benchmark_scores={k: round(v, 4) for k, v in benchmark_scores.items()},
            metadata={
                "params": spec.params,
                "arm_type": arm_type.value,
            },
        )
    
    def _adjust_infer_cost_for_params(
        self,
        spec: ExperimentSpec,
        infer_cost: float,
        arm_type: InverseArmType,
    ) -> float:
        """Adjust inference cost based on arm-specific parameters."""
        params = spec.params
        
        if arm_type == InverseArmType.COUNTERFACTUAL_AUDIT:
            # K counterfactual variants add overhead
            k = int(params.get("counterfactual_k", 3))
            infer_cost *= (1.0 + 0.15 * k)
        
        elif arm_type == InverseArmType.NOISE_INJECTION_ENSEMBLE:
            # N samples add overhead
            n = int(params.get("ensemble_samples", 3))
            infer_cost *= (1.0 + 0.25 * (n - 1))
        
        elif arm_type == InverseArmType.LATENT_PLAN_SWAPPING:
            # Micro-plans add tiny overhead
            n_plans = int(params.get("num_plans", 3))
            infer_cost *= (1.0 + 0.02 * n_plans)
        
        return max(0.0, infer_cost)
    
    def _compute_arm_specific_metrics(
        self,
        spec: ExperimentSpec,
        effect: VariantEffect,
        arm_type: InverseArmType,
        stage_multiplier: float,
        rng: random.Random,
        long_context: float,
        reasoning: float,
        consistency: float,
    ) -> dict[str, float]:
        """Compute arm-specific metrics."""
        baselines = INVERSE_ARM_METRIC_BASELINES.get(spec.track_id, {})
        params = spec.params
        metrics: dict[str, float] = {}
        
        if arm_type == InverseArmType.DELIBERATION_COLLAPSE:
            # Arm A: Deliberation Collapse
            postpass_tokens = int(params.get("postpass_tokens", 48))
            verbosity_penalty = max(0.0, baselines.get("verbosity_penalty", 0.0) - 0.5 * stage_multiplier)
            accuracy_per_token = baselines.get("accuracy_per_token", 0.85) + 0.02 * stage_multiplier
            hallucination_rate = max(0.0, baselines.get("hallucination_surface_rate", 5.0) - 1.5 * stage_multiplier)
            
            metrics["verbosity_penalty"] = round(verbosity_penalty + rng.uniform(-0.1, 0.1), 4)
            metrics["accuracy_per_token"] = round(accuracy_per_token + rng.uniform(-0.01, 0.01), 4)
            metrics["hallucination_surface_rate"] = round(hallucination_rate + rng.uniform(-0.3, 0.3), 4)
            metrics["postpass_token_budget"] = float(postpass_tokens)
        
        elif arm_type == InverseArmType.COUNTERFACTUAL_AUDIT:
            # Arm B: Counterfactual Self-Audit
            k = int(params.get("counterfactual_k", 3))
            robustness_delta = 0.5 * k * stage_multiplier
            assumption_error_rate = max(0.0, baselines.get("assumption_error_rate", 12.0) - 2.0 * k * stage_multiplier)
            flip_rate = 0.1 * k * stage_multiplier
            
            metrics["robustness_delta"] = round(robustness_delta + rng.uniform(-0.1, 0.1), 4)
            metrics["assumption_error_rate"] = round(assumption_error_rate + rng.uniform(-0.5, 0.5), 4)
            metrics["counterfactual_flip_rate"] = round(flip_rate + rng.uniform(-0.02, 0.02), 4)
            metrics["counterfactual_k"] = float(k)
        
        elif arm_type == InverseArmType.ACTIVE_RECALL_ROUTER:
            # Arm C: Active Recall Router
            formatting_error_rate = max(0.0, baselines.get("formatting_error_rate", 8.0) - 2.0 * stage_multiplier)
            token_reduction = baselines.get("token_access_reduction", 15.0) + 5.0 * stage_multiplier
            pattern_completion = baselines.get("pattern_completion_rate", 75.0) + 3.0 * stage_multiplier
            meandering = max(0.0, baselines.get("meandering_score", 20.0) - 5.0 * stage_multiplier)
            
            metrics["formatting_error_rate"] = round(formatting_error_rate + rng.uniform(-0.5, 0.5), 4)
            metrics["token_access_reduction"] = round(token_reduction + rng.uniform(-1.0, 1.0), 4)
            metrics["pattern_completion_rate"] = round(pattern_completion + rng.uniform(-1.0, 1.0), 4)
            metrics["meandering_score"] = round(meandering + rng.uniform(-1.0, 1.0), 4)
        
        elif arm_type == InverseArmType.PROOF_CARRYING_OUTPUTS:
            # Arm D: Proof-Carrying Outputs
            abstention_rate = baselines.get("abstention_rate", 5.0) + 2.0 * stage_multiplier
            certificate_pass = baselines.get("certificate_pass_rate", 85.0) + 2.0 * stage_multiplier
            catastrophic_miss = max(0.0, baselines.get("catastrophic_miss_rate", 2.0) - 0.5 * stage_multiplier)
            
            metrics["abstention_rate"] = round(abstention_rate + rng.uniform(-0.3, 0.3), 4)
            metrics["certificate_pass_rate"] = round(certificate_pass + rng.uniform(-1.0, 1.0), 4)
            metrics["catastrophic_miss_rate"] = round(catastrophic_miss + rng.uniform(-0.1, 0.1), 4)
        
        elif arm_type == InverseArmType.NOISE_INJECTION_ENSEMBLE:
            # Arm E: Noise-Injection Ensemble
            n = int(params.get("ensemble_samples", 3))
            constraint_score = baselines.get("constraint_adherence_score", 80.0) + 3.0 * n * stage_multiplier
            instruction_rate = baselines.get("instruction_following_rate", 85.0) + 2.0 * n * stage_multiplier
            sample_var = baselines.get("sample_variance", 10.0) + 2.0 * n
            voting_agreement = baselines.get("voting_agreement_rate", 70.0) + 5.0 * n
            
            metrics["constraint_adherence_score"] = round(constraint_score + rng.uniform(-1.0, 1.0), 4)
            metrics["instruction_following_rate"] = round(instruction_rate + rng.uniform(-1.0, 1.0), 4)
            metrics["sample_variance"] = round(sample_var + rng.uniform(-0.5, 0.5), 4)
            metrics["voting_agreement_rate"] = round(voting_agreement + rng.uniform(-1.0, 1.0), 4)
            metrics["ensemble_samples"] = float(n)
        
        elif arm_type == InverseArmType.LATENT_PLAN_SWAPPING:
            # Arm F: Latent Plan Swapping
            n_plans = int(params.get("num_plans", 3))
            ci_width = max(1.0, baselines.get("ci_width", 8.0) - 1.5 * n_plans * stage_multiplier)
            variance_reduction = baselines.get("variance_reduction_pct", 0.0) + 5.0 * n_plans * stage_multiplier
            coverage = baselines.get("coverage_score", 85.0) + 2.0 * stage_multiplier
            
            metrics["ci_width"] = round(ci_width + rng.uniform(-0.3, 0.3), 4)
            metrics["variance_reduction_pct"] = round(variance_reduction + rng.uniform(-0.5, 0.5), 4)
            metrics["plan_selection_bias"] = round(rng.uniform(-0.5, 0.5), 4)
            metrics["coverage_score"] = round(coverage + rng.uniform(-1.0, 1.0), 4)
            metrics["num_plans"] = float(n_plans)
        
        elif arm_type == InverseArmType.ADVERSARIAL_USER_SIM:
            # Arm G: Adversarial User Simulator
            wrong_assumption = max(0.0, baselines.get("wrong_assumption_rate", 15.0) - 3.0 * stage_multiplier)
            adversarial_robust = baselines.get("adversarial_robustness_score", 70.0) + 5.0 * stage_multiplier
            ambiguity_handling = baselines.get("ambiguity_handling_rate", 75.0) + 4.0 * stage_multiplier
            abstention = baselines.get("abstention_rate", 5.0) + 1.5 * stage_multiplier
            
            metrics["wrong_assumption_rate"] = round(wrong_assumption + rng.uniform(-0.5, 0.5), 4)
            metrics["adversarial_robustness_score"] = round(adversarial_robust + rng.uniform(-1.0, 1.0), 4)
            metrics["ambiguity_handling_rate"] = round(ambiguity_handling + rng.uniform(-1.0, 1.0), 4)
            metrics["abstention_rate"] = round(abstention + rng.uniform(-0.3, 0.3), 4)
        
        # Add component scores
        metrics["long_context_component"] = round(long_context, 4)
        metrics["reasoning_component"] = round(reasoning, 4)
        metrics["consistency_component"] = round(consistency, 4)
        
        return metrics
    
    def _compute_failure_flags(
        self,
        spec: ExperimentSpec,
        metric_values: dict[str, float],
        arm_type: InverseArmType,
    ) -> list[str]:
        """Compute failure flags for an inverse arm run."""
        flags: list[str] = []
        params = spec.params
        
        if arm_type == InverseArmType.DELIBERATION_COLLAPSE:
            if int(params.get("postpass_tokens", 48)) > 100:
                flags.append("excessive_postpass")
            if metric_values.get("hallucination_surface_rate", 0.0) > 8.0:
                flags.append("hallucination_spike")
        
        elif arm_type == InverseArmType.COUNTERFACTUAL_AUDIT:
            if int(params.get("counterfactual_k", 3)) > 5:
                flags.append("excessive_counterfactuals")
            if metric_values.get("assumption_error_rate", 0.0) > 15.0:
                flags.append("assumption_audit_failure")
        
        elif arm_type == InverseArmType.ACTIVE_RECALL_ROUTER:
            if metric_values.get("formatting_error_rate", 0.0) > 10.0:
                flags.append("router_formatting_failure")
            if metric_values.get("meandering_score", 0.0) > 25.0:
                flags.append("meandering_not_reduced")
        
        elif arm_type == InverseArmType.PROOF_CARRYING_OUTPUTS:
            if metric_values.get("abstention_rate", 0.0) > 15.0:
                flags.append("excessive_abstention")
            if metric_values.get("certificate_pass_rate", 0.0) < 75.0:
                flags.append("certificate_generation_failure")
        
        elif arm_type == InverseArmType.NOISE_INJECTION_ENSEMBLE:
            if int(params.get("ensemble_samples", 3)) > 5:
                flags.append("excessive_ensemble_size")
            if metric_values.get("voting_agreement_rate", 0.0) < 60.0:
                flags.append("ensemble_disagreement")
        
        elif arm_type == InverseArmType.LATENT_PLAN_SWAPPING:
            if int(params.get("num_plans", 3)) > 5:
                flags.append("excessive_plans")
            if metric_values.get("plan_selection_bias", 0.0) > 2.0:
                flags.append("plan_selection_bias_detected")
        
        elif arm_type == InverseArmType.ADVERSARIAL_USER_SIM:
            if metric_values.get("wrong_assumption_rate", 0.0) > 20.0:
                flags.append("adversarial_robustness_failure")
            if metric_values.get("abstention_rate", 0.0) > 20.0:
                flags.append("overly_conservative")
        
        # Common fluency check
        if metric_values.get("fluency", 0.0) < 80.0:
            flags.append("fluency_regression")
        
        return sorted(set(flags))


# Specialized runners for each arm type
class DeliberationCollapseRunner(InverseArmRunner):
    """Runner for Arm A: Deliberation Collapse (anti-chain-of-thought)."""
    
    name: str = "deliberation_collapse"
    description: str = "Deliberation Collapse - Less thinking for better answers"
    supported_tracks: list[str] = ["T7"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config, arm_type=InverseArmType.DELIBERATION_COLLAPSE)


class CounterfactualAuditRunner(InverseArmRunner):
    """Runner for Arm B: Counterfactual Self-Audit."""
    
    name: str = "counterfactual_audit"
    description: str = "Counterfactual Self-Audit - Search alternate worlds"
    supported_tracks: list[str] = ["T8"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config, arm_type=InverseArmType.COUNTERFACTUAL_AUDIT)


class ActiveRecallRouterRunner(InverseArmRunner):
    """Runner for Arm C: Active Recall Router."""
    
    name: str = "active_recall_router"
    description: str = "Active Recall Router - Retrieve first, reason last"
    supported_tracks: list[str] = ["T9"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config, arm_type=InverseArmType.ACTIVE_RECALL_ROUTER)


class ProofCarryingOutputsRunner(InverseArmRunner):
    """Runner for Arm D: Proof-Carrying Outputs."""
    
    name: str = "proof_carrying_outputs"
    description: str = "Proof-Carrying Outputs - Self-verifying certificates"
    supported_tracks: list[str] = ["T10"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config, arm_type=InverseArmType.PROOF_CARRYING_OUTPUTS)


class NoiseInjectionEnsembleRunner(InverseArmRunner):
    """Runner for Arm E: Noise-Injection Ensemble."""
    
    name: str = "noise_injection_ensemble"
    description: str = "Noise-Injection Ensemble - Chaos as compute"
    supported_tracks: list[str] = ["T11"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config, arm_type=InverseArmType.NOISE_INJECTION_ENSEMBLE)


class LatentPlanSwappingRunner(InverseArmRunner):
    """Runner for Arm F: Latent Plan Swapping."""
    
    name: str = "latent_plan_swapping"
    description: str = "Latent Plan Swapping - Multiple micro-plans, one execution"
    supported_tracks: list[str] = ["T12"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config, arm_type=InverseArmType.LATENT_PLAN_SWAPPING)


class AdversarialUserSimRunner(InverseArmRunner):
    """Runner for Arm G: Adversarial User Simulator."""
    
    name: str = "adversarial_user_sim"
    description: str = "Adversarial User Simulator - Train for worst-case users"
    supported_tracks: list[str] = ["T13"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config, arm_type=InverseArmType.ADVERSARIAL_USER_SIM)


# Evaluation functions for inverse arms
def evaluate_inverse_arm_pass(
    candidate: RunResult,
    baseline: RunResult,
    delta_metrics: dict[str, float],
) -> bool:
    """Evaluate whether an inverse arm run passes its gate.
    
    Args:
        candidate: The candidate run result
        baseline: The baseline run result
        delta_metrics: Delta metrics between candidate and baseline
        
    Returns:
        True if the run passes its gate
    """
    arm_type_str = candidate.metadata.get("arm_type", "")
    
    try:
        arm_type = InverseArmType(arm_type_str)
    except ValueError:
        return False
    
    if arm_type == InverseArmType.DELIBERATION_COLLAPSE:
        # Arm A: Pass if accuracy_per_token improved and hallucination reduced
        return (
            delta_metrics.get("accuracy_per_token", 0.0) >= 0.02
            and candidate.metric_values.get("hallucination_surface_rate", 100.0) < baseline.metric_values.get("hallucination_surface_rate", 100.0)
            and "hallucination_spike" not in candidate.failure_flags
        )
    
    elif arm_type == InverseArmType.COUNTERFACTUAL_AUDIT:
        # Arm B: Pass if robustness improved and assumption errors reduced
        return (
            delta_metrics.get("robustness_delta", 0.0) >= 0.3
            and candidate.metric_values.get("assumption_error_rate", 100.0) < baseline.metric_values.get("assumption_error_rate", 100.0)
        )
    
    elif arm_type == InverseArmType.ACTIVE_RECALL_ROUTER:
        # Arm C: Pass if formatting errors reduced and meandering reduced
        return (
            candidate.metric_values.get("formatting_error_rate", 100.0) < baseline.metric_values.get("formatting_error_rate", 100.0)
            and candidate.metric_values.get("meandering_score", 100.0) < baseline.metric_values.get("meandering_score", 100.0)
            and delta_metrics.get("composite", 0.0) >= 1.0
        )
    
    elif arm_type == InverseArmType.PROOF_CARRYING_OUTPUTS:
        # Arm D: Pass if catastrophic misses reduced (even with some abstention)
        return (
            candidate.metric_values.get("catastrophic_miss_rate", 100.0) < baseline.metric_values.get("catastrophic_miss_rate", 100.0)
            and candidate.metric_values.get("certificate_pass_rate", 0.0) >= 80.0
            and candidate.metric_values.get("abstention_rate", 0.0) <= 15.0
        )
    
    elif arm_type == InverseArmType.NOISE_INJECTION_ENSEMBLE:
        # Arm E: Pass if constraint adherence improved
        return (
            delta_metrics.get("constraint_adherence_score", 0.0) >= 3.0
            and delta_metrics.get("instruction_following_rate", 0.0) >= 2.0
            and "ensemble_disagreement" not in candidate.failure_flags
        )
    
    elif arm_type == InverseArmType.LATENT_PLAN_SWAPPING:
        # Arm F: Pass if variance reduced and composite improved
        return (
            delta_metrics.get("variance_reduction_pct", 0.0) >= 3.0
            and delta_metrics.get("composite", 0.0) >= 2.0
            and abs(candidate.metric_values.get("plan_selection_bias", 0.0)) <= 1.0
        )
    
    elif arm_type == InverseArmType.ADVERSARIAL_USER_SIM:
        # Arm G: Pass if wrong assumptions reduced and robustness improved
        return (
            candidate.metric_values.get("wrong_assumption_rate", 100.0) < baseline.metric_values.get("wrong_assumption_rate", 100.0)
            and delta_metrics.get("adversarial_robustness_score", 0.0) >= 3.0
            and candidate.metric_values.get("abstention_rate", 0.0) <= 15.0
        )
    
    return False
