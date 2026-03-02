"""Inverse-directed experiment arms for LLM evaluation.

These arms bet against the usual "more layers / more tokens / bigger model" story,
focusing on techniques that improve quality through constraint, audit, and structure
rather than scale.

Arms:
    A: Deliberation Collapse - Less thinking → better answers under tight budgets
    B: Counterfactual Self-Audit - Search nearby alternate worlds for robustness
    C: Active Recall Router - Retrieve first, reason last
    D: Proof-Carrying Outputs - Self-verifying minimal certificates
    E: Noise-Injection Ensemble - Chaos as compute for constraint satisfaction
    F: Latent Plan Swapping - Multiple micro-plans, one execution
    G: Adversarial User Simulator - Train for worst-case users
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .constants import VariantEffect


class InverseArmType(str, Enum):
    """Types of inverse-directed experiment arms."""
    DELIBERATION_COLLAPSE = "A"  # Anti-chain-of-thought
    COUNTERFACTUAL_AUDIT = "B"   # Imagination-first, bounded
    ACTIVE_RECALL_ROUTER = "C"   # Memory-first, not reasoning-first
    PROOF_CARRYING_OUTPUTS = "D" # Self-verifying, minimal
    NOISE_INJECTION_ENSEMBLE = "E"  # Chaos as compute
    LATENT_PLAN_SWAPPING = "F"  # Multiple micro-plans, one execution
    ADVERSARIAL_USER_SIM = "G"  # Inverse of friendly prompting


@dataclass(frozen=True)
class InverseArmConfig:
    """Configuration for an inverse-directed arm.
    
    Attributes:
        arm_type: The type of inverse arm
        description: Human-readable description
        inversion_principle: Core principle being inverted
        variant_effect: Effect on metrics when this arm is applied
        track_id: Track ID for this arm (T7-T13)
        metrics: List of metrics to track for this arm
        failure_modes: Failure modes this arm targets
        implementation_hints: Hints for implementation
    """
    arm_type: InverseArmType
    description: str
    inversion_principle: str
    variant_effect: VariantEffect
    track_id: str
    metrics: list[str]
    failure_modes: list[str]
    implementation_hints: list[str] = field(default_factory=list)


# Inverse arm variant effects - designed to show improvement through constraint
INVERSE_ARM_VARIANT_EFFECTS: dict[str, VariantEffect] = {
    # Arm A: Deliberation Collapse
    # Forces single-pass generation, reduces hallucination surfaces
    "INV-A-E1": VariantEffect(
        long_context_delta=0.5,   # Slight reduction - no scratchpad
        reasoning_delta=1.5,       # Better - focused answers
        consistency_delta=2.0,     # Better - fewer contradictions
        fluency_delta=0.3,         # Better - more concise
        latency_delta_pct=-25.0,   # Much faster - single pass
        energy_delta_pct=-20.0,    # Less energy - fewer tokens
    ),
    "INV-A-E2": VariantEffect(
        long_context_delta=0.8,
        reasoning_delta=2.0,
        consistency_delta=2.5,
        fluency_delta=0.5,
        latency_delta_pct=-30.0,
        energy_delta_pct=-25.0,
    ),
    "INV-A-E3": VariantEffect(
        long_context_delta=1.0,
        reasoning_delta=1.8,
        consistency_delta=2.2,
        fluency_delta=0.4,
        latency_delta_pct=-28.0,
        energy_delta_pct=-22.0,
    ),
    
    # Arm B: Counterfactual Self-Audit
    # Generates K counterfactual variants, selects most robust
    "INV-B-E1": VariantEffect(
        long_context_delta=1.0,
        reasoning_delta=2.5,       # Better - catches assumption errors
        consistency_delta=3.0,     # Much better - robustness selection
        fluency_delta=-0.1,        # Slight cost - audit overhead
        latency_delta_pct=15.0,    # Slower - counterfactual generation
        energy_delta_pct=10.0,     # More energy - K variants
    ),
    "INV-B-E2": VariantEffect(
        long_context_delta=1.2,
        reasoning_delta=3.0,
        consistency_delta=3.5,
        fluency_delta=-0.2,
        latency_delta_pct=18.0,
        energy_delta_pct=12.0,
    ),
    "INV-B-E3": VariantEffect(
        long_context_delta=1.5,
        reasoning_delta=2.8,
        consistency_delta=3.2,
        fluency_delta=-0.15,
        latency_delta_pct=16.0,
        energy_delta_pct=11.0,
    ),
    
    # Arm C: Active Recall Router
    # Route to direct answer / recall cue / schema cue
    "INV-C-E1": VariantEffect(
        long_context_delta=1.5,    # Better - structured retrieval
        reasoning_delta=1.0,       # Similar - pattern completion
        consistency_delta=1.8,     # Better - structure-first
        fluency_delta=0.2,         # Better - less meandering
        latency_delta_pct=-10.0,   # Faster - no long reasoning
        energy_delta_pct=-8.0,     # Less energy - fewer tokens
    ),
    "INV-C-E2": VariantEffect(
        long_context_delta=1.8,
        reasoning_delta=1.2,
        consistency_delta=2.0,
        fluency_delta=0.3,
        latency_delta_pct=-12.0,
        energy_delta_pct=-10.0,
    ),
    "INV-C-E3": VariantEffect(
        long_context_delta=2.0,
        reasoning_delta=1.5,
        consistency_delta=2.2,
        fluency_delta=0.4,
        latency_delta_pct=-15.0,
        energy_delta_pct=-12.0,
    ),
    
    # Arm D: Proof-Carrying Outputs
    # Require compact certificate for each response
    "INV-D-E1": VariantEffect(
        long_context_delta=0.5,
        reasoning_delta=2.0,       # Better - verification required
        consistency_delta=3.5,     # Much better - honesty bias
        fluency_delta=-0.3,        # Slight cost - certificate overhead
        latency_delta_pct=8.0,     # Slower - certificate generation
        energy_delta_pct=5.0,      # More energy - verification
    ),
    "INV-D-E2": VariantEffect(
        long_context_delta=0.7,
        reasoning_delta=2.3,
        consistency_delta=4.0,
        fluency_delta=-0.4,
        latency_delta_pct=10.0,
        energy_delta_pct=6.0,
    ),
    "INV-D-E3": VariantEffect(
        long_context_delta=0.8,
        reasoning_delta=2.5,
        consistency_delta=3.8,
        fluency_delta=-0.35,
        latency_delta_pct=9.0,
        energy_delta_pct=5.5,
    ),
    
    # Arm E: Noise-Injection Ensemble
    # Vote on constraint satisfaction, not content
    "INV-E-E1": VariantEffect(
        long_context_delta=1.0,
        reasoning_delta=1.5,
        consistency_delta=2.5,     # Better - constraint voting
        fluency_delta=0.1,         # Better - instruction following
        latency_delta_pct=20.0,    # Slower - 3 samples
        energy_delta_pct=15.0,     # More energy - ensemble
    ),
    "INV-E-E2": VariantEffect(
        long_context_delta=1.2,
        reasoning_delta=1.8,
        consistency_delta=2.8,
        fluency_delta=0.2,
        latency_delta_pct=22.0,
        energy_delta_pct=16.0,
    ),
    "INV-E-E3": VariantEffect(
        long_context_delta=1.4,
        reasoning_delta=2.0,
        consistency_delta=3.0,
        fluency_delta=0.3,
        latency_delta_pct=25.0,
        energy_delta_pct=18.0,
    ),
    
    # Arm F: Latent Plan Swapping
    # Generate 3 micro-plans, execute one
    "INV-F-E1": VariantEffect(
        long_context_delta=1.2,
        reasoning_delta=2.2,       # Better - plan selection
        consistency_delta=2.0,     # Better - reduced variance
        fluency_delta=0.2,         # Better - structured output
        latency_delta_pct=5.0,     # Slight overhead - plan generation
        energy_delta_pct=3.0,      # Small overhead - tiny plans
    ),
    "INV-F-E2": VariantEffect(
        long_context_delta=1.5,
        reasoning_delta=2.5,
        consistency_delta=2.3,
        fluency_delta=0.3,
        latency_delta_pct=6.0,
        energy_delta_pct=4.0,
    ),
    "INV-F-E3": VariantEffect(
        long_context_delta=1.8,
        reasoning_delta=2.8,
        consistency_delta=2.5,
        fluency_delta=0.4,
        latency_delta_pct=7.0,
        energy_delta_pct=5.0,
    ),
    
    # Arm G: Adversarial User Simulator
    # Rewrite request with ambiguity, answer safely
    "INV-G-E1": VariantEffect(
        long_context_delta=0.8,
        reasoning_delta=1.8,       # Better - assumption bracketing
        consistency_delta=3.0,     # Much better - robust to ambiguity
        fluency_delta=-0.2,        # Slight cost - conservative output
        latency_delta_pct=12.0,    # Slower - adversarial simulation
        energy_delta_pct=8.0,      # More energy - adversary pass
    ),
    "INV-G-E2": VariantEffect(
        long_context_delta=1.0,
        reasoning_delta=2.0,
        consistency_delta=3.3,
        fluency_delta=-0.3,
        latency_delta_pct=14.0,
        energy_delta_pct=9.0,
    ),
    "INV-G-E3": VariantEffect(
        long_context_delta=1.2,
        reasoning_delta=2.2,
        consistency_delta=3.5,
        fluency_delta=-0.25,
        latency_delta_pct=13.0,
        energy_delta_pct=8.5,
    ),
}


# Inverse arm configurations with full metadata
INVERSE_ARM_CONFIGS: dict[InverseArmType, InverseArmConfig] = {
    InverseArmType.DELIBERATION_COLLAPSE: InverseArmConfig(
        arm_type=InverseArmType.DELIBERATION_COLLAPSE,
        description="Deliberation Collapse (anti-chain-of-thought)",
        inversion_principle="Less thinking → better answers under tight budgets",
        variant_effect=INVERSE_ARM_VARIANT_EFFECTS["INV-A-E1"],
        track_id="T7",
        metrics=[
            "pass_rate",
            "critical_fact_miss_rate",
            "variance_across_seeds",
            "verbosity_penalty",
            "accuracy_per_token",
            "hallucination_surface_rate",
        ],
        failure_modes=[
            "overthinking",
            "hallucination_from_reasoning",
            "verbosity_without_value",
        ],
        implementation_hints=[
            "Force single-pass generation with no scratchpad",
            "Disable self-consistency voting",
            "Disable long reasoning chains",
            "Add tiny answer-shaping postpass (32-64 tokens)",
            "Postpass only fixes format + obvious contradictions",
        ],
    ),
    
    InverseArmType.COUNTERFACTUAL_AUDIT: InverseArmConfig(
        arm_type=InverseArmType.COUNTERFACTUAL_AUDIT,
        description="Counterfactual Self-Audit (imagination-first, bounded)",
        inversion_principle="Don't search the world; search nearby alternate worlds",
        variant_effect=INVERSE_ARM_VARIANT_EFFECTS["INV-B-E1"],
        track_id="T8",
        metrics=[
            "robustness_delta",
            "assumption_error_rate",
            "counterfactual_flip_rate",
            "latency_overhead",
        ],
        failure_modes=[
            "assumption_errors",
            "constraint_misreading",
            "single-path_failing",
        ],
        implementation_hints=[
            "After producing answer, generate K counterfactual variants (K=2-4)",
            "Vary assumptions, not the answer",
            "Ask: 'What if key constraint was opposite?'",
            "Ask: 'What if user intended alternate meaning?'",
            "Select answer valid across most counterfactuals",
        ],
    ),
    
    InverseArmType.ACTIVE_RECALL_ROUTER: InverseArmConfig(
        arm_type=InverseArmType.ACTIVE_RECALL_ROUTER,
        description="Active Recall Router (memory-first, not reasoning-first)",
        inversion_principle="Retrieve first, reason last",
        variant_effect=INVERSE_ARM_VARIANT_EFFECTS["INV-C-E1"],
        track_id="T9",
        metrics=[
            "formatting_error_rate",
            "token_access_reduction",
            "pattern_completion_rate",
            "meandering_score",
        ],
        failure_modes=[
            "formatting_errors",
            "meandering_output",
            "structure_violation",
        ],
        implementation_hints=[
            "Train/engineer router that decides among:",
            "  1. direct answer",
            "  2. recall cue → generate 5-10 key facts/keywords",
            "  3. schema cue → generate outline/template, then fill",
            "No long reasoning, just routing + schema approach",
            "Force structure-first pass to prevent meandering",
        ],
    ),
    
    InverseArmType.PROOF_CARRYING_OUTPUTS: InverseArmConfig(
        arm_type=InverseArmType.PROOF_CARRYING_OUTPUTS,
        description="Proof-Carrying Outputs (self-verifying, minimal)",
        inversion_principle="Answers must carry a tiny certificate, or they don't ship",
        variant_effect=INVERSE_ARM_VARIANT_EFFECTS["INV-D-E1"],
        track_id="T10",
        metrics=[
            "critical_fact_miss_rate",
            "abstention_rate",
            "certificate_pass_rate",
            "catastrophic_miss_rate",
        ],
        failure_modes=[
            "catastrophic_misses",
            "hallucination",
            "unverifiable_claims",
        ],
        implementation_hints=[
            "Require compact certificate for each response:",
            "  '3 bullet claims + 1 supporting derivation/check per claim'",
            "  or 'unit tests' in natural language (inputs → expected outputs)",
            "If certificate fails internal consistency check:",
            "  return 'insufficient info' instead of guessing",
            "Honesty bias: lose coverage, reduce catastrophic misses",
        ],
    ),
    
    InverseArmType.NOISE_INJECTION_ENSEMBLE: InverseArmConfig(
        arm_type=InverseArmType.NOISE_INJECTION_ENSEMBLE,
        description="Noise-Injection Ensemble (chaos as compute)",
        inversion_principle="Randomness is not a bug; it's a search operator",
        variant_effect=INVERSE_ARM_VARIANT_EFFECTS["INV-E-E1"],
        track_id="T11",
        metrics=[
            "constraint_adherence_score",
            "instruction_following_rate",
            "sample_variance",
            "voting_agreement_rate",
        ],
        failure_modes=[
            "instruction_following_failures",
            "constraint_violation",
            "format_disobedience",
        ],
        implementation_hints=[
            "Inject controlled noise into:",
            "  - decoding temperature schedule (e.g., 0.2→0.7)",
            "  - prompt perturbations (synonym swaps, reorder constraints)",
            "Take 3 samples",
            "Vote on CONSTRAINT SATISFACTION, not content",
            "Select best constraint-satisfying output",
        ],
    ),
    
    InverseArmType.LATENT_PLAN_SWAPPING: InverseArmConfig(
        arm_type=InverseArmType.LATENT_PLAN_SWAPPING,
        description="Latent Plan Swapping (multiple micro-plans, one execution)",
        inversion_principle="Generate multiple plans, execute only one",
        variant_effect=INVERSE_ARM_VARIANT_EFFECTS["INV-F-E1"],
        track_id="T12",
        metrics=[
            "ci_width",
            "variance_reduction_pct",
            "plan_selection_bias",
            "coverage_score",
        ],
        failure_modes=[
            "high_variance",
            "plan_instability",
            "execution_drift",
        ],
        implementation_hints=[
            "Step 1 (tiny): generate 3 micro-plans (each ≤ 20 tokens)",
            "Step 2: choose plan via learned/rule scorer:",
            "  - coverage of constraints",
            "  - simplicity score",
            "Step 3: execute chosen plan with strict budget",
            "Get planning benefit without full CoT cost",
        ],
    ),
    
    InverseArmType.ADVERSARIAL_USER_SIM: InverseArmConfig(
        arm_type=InverseArmType.ADVERSARIAL_USER_SIM,
        description="Adversarial User Simulator (inverse of friendly prompting)",
        inversion_principle="Train for worst-case users, not average users",
        variant_effect=INVERSE_ARM_VARIANT_EFFECTS["INV-G-E1"],
        track_id="T13",
        metrics=[
            "wrong_assumption_rate",
            "adversarial_robustness_score",
            "abstention_rate",
            "ambiguity_handling_rate",
        ],
        failure_modes=[
            "wrong_assumption",
            "hallucination_from_ambiguity",
            "constraint_confusion",
        ],
        implementation_hints=[
            "Before answering, run cheap adversary that rewrites request:",
            "  - add ambiguity",
            "  - add conflicting constraints",
            "  - add bait for hallucination",
            "Answer adversarial version safely:",
            "  - ask-for-info style or conservative output",
            "  - explicitly bracket assumptions",
            "  - no interactive clarification (harness forbids)",
        ],
    ),
}


# Track-specific metric baselines for inverse arms
INVERSE_ARM_METRIC_BASELINES: dict[str, dict[str, float]] = {
    "T7": {  # Deliberation Collapse
        "verbosity_penalty": 0.0,
        "accuracy_per_token": 0.85,
        "hallucination_surface_rate": 5.0,
    },
    "T8": {  # Counterfactual Self-Audit
        "robustness_delta": 0.0,
        "assumption_error_rate": 12.0,
        "counterfactual_flip_rate": 0.0,
    },
    "T9": {  # Active Recall Router
        "formatting_error_rate": 8.0,
        "token_access_reduction": 15.0,
        "pattern_completion_rate": 75.0,
        "meandering_score": 20.0,
    },
    "T10": {  # Proof-Carrying Outputs
        "abstention_rate": 5.0,
        "certificate_pass_rate": 85.0,
        "catastrophic_miss_rate": 2.0,
    },
    "T11": {  # Noise-Injection Ensemble
        "constraint_adherence_score": 80.0,
        "instruction_following_rate": 85.0,
        "sample_variance": 10.0,
        "voting_agreement_rate": 70.0,
    },
    "T12": {  # Latent Plan Swapping
        "ci_width": 8.0,
        "variance_reduction_pct": 0.0,
        "plan_selection_bias": 0.0,
        "coverage_score": 85.0,
    },
    "T13": {  # Adversarial User Simulator
        "wrong_assumption_rate": 15.0,
        "adversarial_robustness_score": 70.0,
        "ambiguity_handling_rate": 75.0,
    },
}


# Bundle configurations
INVERSE_ARM_BUNDLES: dict[str, dict[str, Any]] = {
    "bundle_1_cheap_high_win": {
        "name": "Bundle 1 (cheap, high chance of win)",
        "arms": [InverseArmType.DELIBERATION_COLLAPSE, InverseArmType.LATENT_PLAN_SWAPPING],
        "goal": "Improve accuracy-per-token and reduce variance",
        "expected_cost_multiplier": 1.1,  # 10% overhead
        "risk_level": "low",
    },
    "bundle_2_risky_massive_upside": {
        "name": "Bundle 2 (risky, potentially massive upside)",
        "arms": [InverseArmType.COUNTERFACTUAL_AUDIT, InverseArmType.PROOF_CARRYING_OUTPUTS],
        "goal": "Crush critical misses + increase robustness even if pass rate stays flat",
        "expected_cost_multiplier": 1.25,  # 25% overhead
        "risk_level": "high",
    },
    "bundle_3_top3_recommended": {
        "name": "Top 3 Recommended Arms",
        "arms": [
            InverseArmType.DELIBERATION_COLLAPSE,
            InverseArmType.COUNTERFACTUAL_AUDIT,
            InverseArmType.NOISE_INJECTION_ENSEMBLE,
        ],
        "goal": "Attack three different failure modes: overthinking, wrong assumptions, instruction-following",
        "expected_cost_multiplier": 1.2,
        "risk_level": "medium",
    },
}


def get_inverse_arm_config(arm_type: InverseArmType) -> InverseArmConfig:
    """Get configuration for an inverse arm type.
    
    Args:
        arm_type: The type of inverse arm
        
    Returns:
        Configuration for the arm
    """
    return INVERSE_ARM_CONFIGS[arm_type]


def get_inverse_arm_variant_effect(variant_id: str) -> VariantEffect | None:
    """Get variant effect for an inverse arm variant.
    
    Args:
        variant_id: The variant ID (e.g., 'INV-A-E1')
        
    Returns:
        VariantEffect if found, None otherwise
    """
    return INVERSE_ARM_VARIANT_EFFECTS.get(variant_id)


def get_inverse_arm_for_track(track_id: str) -> InverseArmType | None:
    """Get the inverse arm type for a track ID.
    
    Args:
        track_id: Track ID (T7-T13)
        
    Returns:
        InverseArmType if found, None otherwise
    """
    track_mapping = {
        "T7": InverseArmType.DELIBERATION_COLLAPSE,
        "T8": InverseArmType.COUNTERFACTUAL_AUDIT,
        "T9": InverseArmType.ACTIVE_RECALL_ROUTER,
        "T10": InverseArmType.PROOF_CARRYING_OUTPUTS,
        "T11": InverseArmType.NOISE_INJECTION_ENSEMBLE,
        "T12": InverseArmType.LATENT_PLAN_SWAPPING,
        "T13": InverseArmType.ADVERSARIAL_USER_SIM,
    }
    return track_mapping.get(track_id.upper())


def list_inverse_arms() -> list[InverseArmType]:
    """List all available inverse arm types."""
    return list(InverseArmType)


def list_inverse_arm_tracks() -> list[str]:
    """List all track IDs for inverse arms."""
    return ["T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T-META"]
