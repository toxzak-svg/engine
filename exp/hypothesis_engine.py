"""LLM Hypothesis Generation Engine — AI-powered hypothesis generation.

Leverages LLMs to generate novel hypotheses based on:
    - Prior experiment results
    - Track literature and known failure modes
    - Cross-track patterns

This enables:
    - "What if" reasoning about untested parameter combinations
    - Generation of adversarial failure scenarios
    - Synthesis of insights across tracks

Usage:
    engine = HypothesisEngine()
    
    # Generate hypotheses from experiment history
    hypotheses = engine.generate_hypotheses(
        track_id="T3",
        context={"runs": [...], "reports": [...]},
        num_hypotheses=5,
    )
    
    # Rank hypotheses by novelty and plausibility
    ranked = engine.rank_hypotheses(hypotheses)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exp.constants import TRACKS


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A generated hypothesis."""
    id: str
    track_id: str
    description: str
    mechanism: str  # Why this might work
    expected_effect: dict[str, float]  # Predicted metric deltas
    testable_predictions: list[str]  # How to verify
    novelty_score: float  # 0-1, how novel vs existing variants
    risk_level: str  # low/medium/high
    related_tracks: list[str] = field(default_factory=list)


@dataclass
class HypothesisGenerationResult:
    """Result of hypothesis generation."""
    hypotheses: list[Hypothesis]
    generation_method: str
    context_summary: str


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

HYPOTHESIS_SYSTEM_PROMPT = """You are an expert ML researcher specializing in LLM architecture innovations.
Your task is to generate novel, testable hypotheses for improving LLM performance.

Think deeply about:
1. What mechanisms could improve reasoning, long-context, or consistency?
2. What combinations of techniques haven't been tried?
3. What failure modes could be addressed?

Generate hypotheses that are:
- Mechanistically grounded
- Testable with the experiment harness
- Potentially high-impact
"""


HYPOTHESIS_USER_PROMPT_TEMPLATE = """Generate {num_hypotheses} hypotheses for track {track_id}.

Track Description:
{track_description}

Current Best Results:
{best_results}

Failure Modes to Address:
{failure_modes}

For each hypothesis, provide:
1. A brief description
2. The underlying mechanism
3. Expected effect on metrics (long_context, reasoning, consistency, fluency)
4. How to test it
5. Risk level (low/medium/high)

Respond in JSON format:
{{
  "hypotheses": [
    {{
      "description": "...",
      "mechanism": "...",
      "expected_effect": {{"long_context": 0.0, "reasoning": 0.0, "consistency": 0.0, "fluency": 0.0}},
      "testable_predictions": ["...", "..."],
      "novelty_score": 0.8,
      "risk_level": "medium"
    }}
  ]
}}
"""


# Track descriptions for context
TRACK_DESCRIPTIONS = {
    "T1": "Hybrid photonic-digital attention acceleration. Uses analog components for attention computation.",
    "T2": "Reversible-state Transformer blocks. Maintains activation checkpoints for efficient backprop.",
    "T3": "Compression-first hierarchical memory. Compresses KV cache with learned representations.",
    "T4": "Vector-symbolic scratchpad reasoning. Uses Holographic Reduced Representations for multi-step reasoning.",
    "T5": "Self-assembling modular circuits. Dynamically composes sub-networks based on input.",
    "T6": "Energy-based global decoding. Uses energy-based models for constrained generation.",
    "T7": "Deliberation Collapse - Anti-chain-of-thought for tight budgets.",
    "T8": "Counterfactual Self-Audit - Imagination-first bounded search.",
    "T9": "Active Recall Router - Memory-first retrieval before reasoning.",
    "T10": "Proof-Carrying Outputs - Self-verifying minimal certificates.",
    "T11": "Noise-Injection Ensemble - Chaos as compute for constraint satisfaction.",
    "T12": "Latent Plan Swapping - Multiple micro-plans, single execution.",
    "T13": "Adversarial User Simulator - Worst-case user training.",
}


# Known failure modes per track
TRACK_FAILURE_MODES = {
    "T1": ["analog_drift", "noise_amplification"],
    "T2": ["reversibility_break", "checkpoint_overhead"],
    "T3": ["critical_fact_loss", "compression_artifacts"],
    "T4": ["entity_role_swap", "binding_errors"],
    "T5": ["invalid_circuit", "routing_failures"],
    "T6": ["repetitive_text", "energy_minima_trapping"],
}


# ---------------------------------------------------------------------------
# Hypothesis Engine
# ---------------------------------------------------------------------------


class HypothesisEngine:
    """LLM-powered hypothesis generation engine.

    Generates novel hypotheses by:
    1. Analyzing experiment history
    2. Using LLM to propose mechanisms
    3. Scoring for novelty and plausibility

    Can work with or without actual LLM API (fallback to template-based).
    """

    def __init__(self, llm_client: Any = None):
        """Initialize engine with optional LLM client.

        Args:
            llm_client: Optional LLM API client (OpenAI, Anthropic, etc.)
        """
        self.llm_client = llm_client
        self._hypothesis_cache: list[Hypothesis] = []

    def generate_hypotheses(
        self,
        track_id: str,
        context: dict[str, Any],
        num_hypotheses: int = 5,
    ) -> HypothesisGenerationResult:
        """Generate hypotheses for a track.

        Args:
            track_id: Track ID (e.g., "T3")
            context: Dict with 'runs', 'reports', 'comparison' data
            num_hypotheses: Number of hypotheses to generate

        Returns:
            HypothesisGenerationResult with generated hypotheses
        """
        if self.llm_client:
            return self._generate_with_llm(track_id, context, num_hypotheses)
        else:
            return self._generate_template_based(track_id, context, num_hypotheses)

    def _generate_with_llm(
        self,
        track_id: str,
        context: dict[str, Any],
        num_hypotheses: int,
    ) -> HypothesisGenerationResult:
        """Generate hypotheses using LLM API."""
        # Build context summary
        context_summary = self._build_context_summary(track_id, context)

        # Build prompt
        user_prompt = HYPOTHESIS_USER_PROMPT_TEMPLATE.format(
            num_hypotheses=num_hypotheses,
            track_id=track_id,
            track_description=TRACK_DESCRIPTIONS.get(track_id, "Unknown track"),
            best_results=self._format_best_results(context.get("runs", [])),
            failure_modes=TRACK_FAILURE_MODES.get(track_id, []),
        )

        # Call LLM
        response = self.llm_client.chat(
            system=HYPOTHESIS_SYSTEM_PROMPT,
            user=user_prompt,
        )

        # Parse response
        try:
            data = json.loads(response)
            hypotheses = [
                Hypothesis(
                    id=f"hypo-{track_id}-{i}",
                    track_id=track_id,
                    description=h["description"],
                    mechanism=h["mechanism"],
                    expected_effect=h.get("expected_effect", {}),
                    testable_predictions=h.get("testable_predictions", []),
                    novelty_score=h.get("novelty_score", 0.5),
                    risk_level=h.get("risk_level", "medium"),
                )
                for i, h in enumerate(data.get("hypotheses", []))
            ]
        except (json.JSONDecodeError, KeyError):
            # Fallback on parse error
            hypotheses = self._generate_template_based(track_id, context, num_hypotheses).hypotheses

        self._hypothesis_cache.extend(hypotheses)

        return HypothesisGenerationResult(
            hypotheses=hypotheses,
            generation_method="llm",
            context_summary=context_summary,
        )

    def _generate_template_based(
        self,
        track_id: str,
        context: dict[str, Any],
        num_hypotheses: int,
    ) -> HypothesisGenerationResult:
        """Generate hypotheses using templates (no LLM)."""
        hypotheses = []

        # Get template hypotheses for this track
        templates = self._get_hypothesis_templates(track_id)

        for i, template in enumerate(templates[:num_hypotheses]):
            hypo = Hypothesis(
                id=f"hypo-{track_id}-{i}",
                track_id=track_id,
                description=template["description"],
                mechanism=template["mechanism"],
                expected_effect=template.get("expected_effect", {}),
                testable_predictions=template.get("testable_predictions", []),
                novelty_score=template.get("novelty_score", 0.5),
                risk_level=template.get("risk_level", "medium"),
                related_tracks=template.get("related_tracks", []),
            )
            hypotheses.append(hypo)

        self._hypothesis_cache.extend(hypotheses)

        return HypothesisGenerationResult(
            hypotheses=hypotheses,
            generation_method="template",
            context_summary=self._build_context_summary(track_id, context),
        )

    def _get_hypothesis_templates(self, track_id: str) -> list[dict[str, Any]]:
        """Get template hypotheses for a track."""
        templates = {
            "T1": [
                {
                    "description": "Hybrid sparse-dense attention with learned routing",
                    "mechanism": "Route easy tokens to cheap dense attention, hard tokens to precise analog attention",
                    "expected_effect": {"long_context": 3.0, "reasoning": 1.0, "consistency": 0.5, "fluency": -0.2},
                    "testable_predictions": ["Routing accuracy > 80%", "Energy reduction > 15%"],
                    "novelty_score": 0.7,
                    "risk_level": "medium",
                    "related_tracks": ["T5"],
                },
                {
                    "description": "Temporal error correction for analog drift",
                    "mechanism": "Add learnable correction layer after analog attention to compensate for drift",
                    "expected_effect": {"long_context": 1.5, "reasoning": 0.5, "consistency": 2.0, "fluency": 0.0},
                    "testable_predictions": ["Drift reduction > 50%", "Minimal latency overhead"],
                    "novelty_score": 0.6,
                    "risk_level": "low",
                    "related_tracks": [],
                },
                {
                    "description": "Noise-aware token weighting",
                    "mechanism": "Weight attention scores by estimated noise level in analog computation",
                    "expected_effect": {"long_context": 2.0, "reasoning": 0.8, "consistency": 1.0, "fluency": -0.1},
                    "testable_predictions": ["Noise robustness improved", "Signal-to-noise ratio increased"],
                    "novelty_score": 0.5,
                    "risk_level": "low",
                    "related_tracks": [],
                },
            ],
            "T3": [
                {
                    "description": "Importance-weighted compression",
                    "mechanism": "Compress tokens non-uniformly based on attention importance scores",
                    "expected_effect": {"long_context": 4.0, "reasoning": 0.5, "consistency": 0.8, "fluency": -0.1},
                    "testable_predictions": ["Token reduction > 50%", "Critical fact miss rate < 5%"],
                    "novelty_score": 0.8,
                    "risk_level": "medium",
                    "related_tracks": [],
                },
                {
                    "description": "Hierarchical retrieval with verification",
                    "mechanism": "First compress to summary, then retrieve with verification step",
                    "expected_effect": {"long_context": 3.5, "reasoning": 1.5, "consistency": 1.2, "fluency": -0.2},
                    "testable_predictions": ["Retrieval accuracy improved", "Miss rate decrease"],
                    "novelty_score": 0.7,
                    "risk_level": "medium",
                    "related_tracks": ["T4", "T9"],
                },
                {
                    "description": "Adaptive compression granularity",
                    "mechanism": "Dynamically choose compression ratio based on context complexity",
                    "expected_effect": {"long_context": 3.0, "reasoning": 1.0, "consistency": 0.6, "fluency": 0.0},
                    "testable_predictions": ["Variable compression ratios observed", "Quality maintained"],
                    "novelty_score": 0.6,
                    "risk_level": "low",
                    "related_tracks": [],
                },
            ],
            "T4": [
                {
                    "description": "Symbolic constraint propagation",
                    "mechanism": "Propagate binding constraints across reasoning steps to reduce errors",
                    "expected_effect": {"long_context": 0.5, "reasoning": 3.5, "consistency": 2.0, "fluency": -0.1},
                    "testable_predictions": ["Binding error reduction > 40%", "Multi-step accuracy improved"],
                    "novelty_score": 0.8,
                    "risk_level": "medium",
                    "related_tracks": ["T10"],
                },
                {
                    "description": "Role-aware attention masking",
                    "mechanism": "Add learned masks to prevent entity-role confusion in attention",
                    "expected_effect": {"long_context": 0.3, "reasoning": 2.5, "consistency": 2.5, "fluency": 0.0},
                    "testable_predictions": ["Entity swap errors reduced", "Consistency improved"],
                    "novelty_score": 0.7,
                    "risk_level": "low",
                    "related_tracks": [],
                },
            ],
            "T5": [
                {
                    "description": "Learnable circuit topology",
                    "mechanism": "Learn optimal circuit structure during training, use during inference",
                    "expected_effect": {"long_context": 1.5, "reasoning": 2.0, "consistency": 1.8, "fluency": -0.2},
                    "testable_predictions": ["Valid plans > 99%", "Latency overhead < 10%"],
                    "novelty_score": 0.7,
                    "risk_level": "high",
                    "related_tracks": [],
                },
                {
                    "description": "Ensemble of circuit experts",
                    "mechanism": "Train separate experts for different task types, route dynamically",
                    "expected_effect": {"long_context": 2.0, "reasoning": 2.5, "consistency": 2.0, "fluency": -0.1},
                    "testable_predictions": ["Expert utilization balanced", "Quality improvements across tasks"],
                    "novelty_score": 0.6,
                    "risk_level": "medium",
                    "related_tracks": [],
                },
            ],
            "T6": [
                {
                    "description": "Adaptive temperature annealing",
                    "mechanism": "Start high-temp for exploration, anneal based on constraint satisfaction",
                    "mechanism": "Use constraint satisfaction signal to guide annealing schedule",
                    "expected_effect": {"long_context": 0.5, "reasoning": 2.0, "consistency": 3.5, "fluency": -0.3},
                    "testable_predictions": ["Contradiction reduction > 30%", "Fluency maintained"],
                    "novelty_score": 0.7,
                    "risk_level": "medium",
                    "related_tracks": [],
                },
                {
                    "description": "Constraint-aware beam search",
                    "mechanism": "Incorporate constraint satisfaction into beam scoring",
                    "expected_effect": {"long_context": 0.8, "reasoning": 1.5, "consistency": 3.0, "fluency": -0.2},
                    "testable_predictions": ["Constraint pass rate improved", "Minimal quality loss"],
                    "novelty_score": 0.6,
                    "risk_level": "low",
                    "related_tracks": [],
                },
            ],
        }

        return templates.get(track_id, [])

    def rank_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        weights: dict[str, float] | None = None,
    ) -> list[Hypothesis]:
        """Rank hypotheses by composite score.

        Args:
            hypotheses: List of hypotheses to rank
            weights: Optional weights for ranking factors

        Returns:
            Sorted list (best first)
        """
        if weights is None:
            weights = {"novelty": 0.3, "impact": 0.4, "risk_adjusted": 0.3}

        ranked = []
        for hypo in hypotheses:
            # Compute impact score
            impact = sum(hypo.expected_effect.values())

            # Risk adjustment: penalize high risk
            risk_map = {"low": 1.0, "medium": 0.7, "high": 0.4}
            risk_factor = risk_map.get(hypo.risk_level, 0.5)

            # Composite score
            score = (
                weights["novelty"] * hypo.novelty_score
                + weights["impact"] * (impact / 10.0)  # Normalize
                + weights["risk_adjusted"] * risk_factor
            )

            ranked.append((score, hypo))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in ranked]

    def find_synergy_hypotheses(
        self,
        tracks: list[str],
    ) -> list[dict[str, Any]]:
        """Find hypotheses that could combine well across tracks.

        Args:
            tracks: List of track IDs to consider

        Returns:
            List of synergy suggestions
        """
        synergies = []

        # Track known good combinations
        known_synergies = [
            {
                "tracks": ["T3", "T4"],
                "description": "Hierarchical memory + symbolic reasoning",
                "rationale": "T3 compression provides efficient context, T4 provides structured reasoning",
            },
            {
                "tracks": ["T1", "T5"],
                "description": "Photonic attention + modular circuits",
                "rationale": "Fast attention enables more routing decisions",
            },
            {
                "tracks": ["T6", "T4"],
                "description": "Energy-based decoding + vector symbolism",
                "rationale": "Energy model can enforce symbolic constraints",
            },
            {
                "tracks": ["T3", "T9"],
                "description": "Compression + active recall",
                "rationale": "Compressed memory enables faster retrieval",
            },
        ]

        for synergy in known_synergies:
            synergy_tracks = set(synergy["tracks"])
            if synergy_tracks.issubset(set(tracks)):
                synergies.append(synergy)

        return synergies

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _build_context_summary(
        self,
        track_id: str,
        context: dict[str, Any],
    ) -> str:
        """Build text summary of context for hypothesis generation."""
        runs = context.get("runs", [])
        reports = context.get("reports", [])

        lines = [f"Track: {track_id}"]

        if runs:
            composites = [r.get("metric_values", {}).get("composite", 0) for r in runs]
            lines.append(f"  Runs: {len(runs)}, mean composite: {sum(composites)/len(composites):.1f}")

        if reports:
            passed = sum(1 for r in reports if r.get("pass_fail", {}).get("overall_pass"))
            lines.append(f"  Reports: {len(reports)}, pass rate: {passed/len(reports):.1%}")

        return "\n".join(lines)

    def _format_best_results(self, runs: list[dict]) -> str:
        """Format best results from runs."""
        if not runs:
            return "No prior results"

        # Sort by composite
        sorted_runs = sorted(
            runs,
            key=lambda r: r.get("metric_values", {}).get("composite", 0),
            reverse=True,
        )

        best = sorted_runs[0] if sorted_runs else {}
        comp = best.get("metric_values", {}).get("composite", 0)

        return f"Best composite: {comp:.1f}"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def generate_hypothesis_report(
    engine: HypothesisEngine,
    track_id: str,
    context: dict[str, Any],
    num_hypotheses: int = 5,
) -> str:
    """Generate a markdown hypothesis report.

    Args:
        engine: HypothesisEngine instance
        track_id: Track ID
        context: Experiment context
        num_hypotheses: Number to generate

    Returns:
        Markdown-formatted report
    """
    result = engine.generate_hypotheses(track_id, context, num_hypotheses)
    ranked = engine.rank_hypotheses(result.hypotheses)

    lines = []
    lines.append(f"# Hypothesis Generation Report: {track_id}")
    lines.append("")
    lines.append(f"**Generation Method**: {result.generation_method}")
    lines.append(f"**Context**: {result.context_summary}")
    lines.append("")
    lines.append("## Generated Hypotheses")
    lines.append("")

    for i, hypo in enumerate(ranked, 1):
        lines.append(f"### {i}. {hypo.description}")
        lines.append("")
        lines.append(f"**Mechanism**: {hypo.mechanism}")
        lines.append("")
        lines.append(f"**Expected Effects**:")
        for metric, value in hypo.expected_effect.items():
            sign = "+" if value > 0 else ""
            lines.append(f"  - {metric}: {sign}{value:.1f}")
        lines.append("")
        lines.append(f"**Novelty**: {hypo.novelty_score:.1f}/1.0 | **Risk**: {hypo.risk_level}")
        lines.append("")

        if hypo.testable_predictions:
            lines.append("**Testable Predictions**:")
            for pred in hypo.testable_predictions:
                lines.append(f"  - {pred}")
            lines.append("")

    return "\n".join(lines)

