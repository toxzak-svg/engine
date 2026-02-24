"""Track-specific runners for each experiment track (T1-T6)."""

from __future__ import annotations

import time
from typing import Any

from ..models import ExperimentSpec, RunResult
from .base import BaseRunner, RunnerConfig, RunnerResult, RunnerStatus
from .subprocess_runner import SubprocessRunner


class T1Runner(SubprocessRunner):
    """Runner for T1: Hybrid photonic-digital attention acceleration.
    
    This track experiments with photonic computing for attention mechanisms.
    Key parameters:
    - noise_model: Type of analog noise model
    - recalibration_interval: How often to recalibrate analog components
    
    The runner can execute photonic simulation frameworks or real hardware drivers.
    """
    
    name = "t1"
    description = "Hybrid photonic-digital attention acceleration runner"
    supported_tracks = ["T1"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
    ):
        if command_template is None:
            # Default command for T1 experiments
            command_template = (
                "python -m engine.tracks.t1.run "
                "--spec {spec_id} --seed {seed} --run-id {run_id} "
                "--gpu {gpu_ids} --variant {variant} "
                "--params '{params_json}'"
            )
        super().__init__(config, command_template)
    
    def pre_execute(self, spec: ExperimentSpec, seed: int) -> None:
        """T1-specific pre-execution setup."""
        # Validate photonic-specific parameters
        noise_model = spec.params.get("noise_model", "analog_v1")
        recal_interval = spec.params.get("recalibration_interval", 512)
        
        # Could check for hardware availability here
        # Or set up photonic simulation environment
    
    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        errors = super().validate_spec(spec)
        
        # T1-specific validation
        recal_interval = spec.params.get("recalibration_interval", 512)
        if not isinstance(recal_interval, int) or recal_interval < 1:
            errors.append("recalibration_interval must be a positive integer")
        
        noise_model = spec.params.get("noise_model", "analog_v1")
        valid_models = ["analog_v1", "analog_v2", "digital", "hybrid"]
        if noise_model not in valid_models:
            errors.append(f"noise_model must be one of {valid_models}")
        
        return errors


class T2Runner(SubprocessRunner):
    """Runner for T2: Reversible-state Transformer blocks.
    
    This track experiments with reversible computation for memory efficiency.
    Key parameters:
    - anchor_frequency: How often to place reversible anchors
    - disable_norm_constraints: Whether to disable normalization constraints
    
    The runner manages reversible checkpointing and memory profiling.
    """
    
    name = "t2"
    description = "Reversible-state Transformer blocks runner"
    supported_tracks = ["T2"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
    ):
        if command_template is None:
            command_template = (
                "python -m engine.tracks.t2.run "
                "--spec {spec_id} --seed {seed} --run-id {run_id} "
                "--gpu {gpu_ids} --variant {variant} "
                "--params '{params_json}'"
            )
        super().__init__(config, command_template)
    
    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        errors = super().validate_spec(spec)
        
        anchor_freq = spec.params.get("anchor_frequency", "1/4")
        valid_freqs = ["1/2", "1/4", "1/8"]
        if anchor_freq not in valid_freqs:
            errors.append(f"anchor_frequency must be one of {valid_freqs}")
        
        return errors


class T3Runner(SubprocessRunner):
    """Runner for T3: Compression-first hierarchical memory.
    
    This track experiments with hierarchical memory compression.
    Key parameters:
    - compression_ratio: Ratio of memory compression
    
    The runner handles compression benchmarks and fact retrieval tests.
    """
    
    name = "t3"
    description = "Compression-first hierarchical memory runner"
    supported_tracks = ["T3"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
    ):
        if command_template is None:
            command_template = (
                "python -m engine.tracks.t3.run "
                "--spec {spec_id} --seed {seed} --run-id {run_id} "
                "--gpu {gpu_ids} --variant {variant} "
                "--params '{params_json}'"
            )
        super().__init__(config, command_template)
    
    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        errors = super().validate_spec(spec)
        
        compression_ratio = spec.params.get("compression_ratio", 0.7)
        if not isinstance(compression_ratio, (int, float)):
            errors.append("compression_ratio must be a number")
        elif not 0.0 <= compression_ratio <= 1.0:
            errors.append("compression_ratio must be between 0 and 1")
        
        return errors


class T4Runner(SubprocessRunner):
    """Runner for T4: Vector-symbolic scratchpad reasoning.
    
    This track experiments with vector-symbolic architectures for reasoning.
    Key parameters:
    - role_permutation_noise: Noise level for role permutations
    
    The runner executes VSA reasoning benchmarks.
    """
    
    name = "t4"
    description = "Vector-symbolic scratchpad reasoning runner"
    supported_tracks = ["T4"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
    ):
        if command_template is None:
            command_template = (
                "python -m engine.tracks.t4.run "
                "--spec {spec_id} --seed {seed} --run-id {run_id} "
                "--gpu {gpu_ids} --variant {variant} "
                "--params '{params_json}'"
            )
        super().__init__(config, command_template)
    
    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        errors = super().validate_spec(spec)
        
        noise = spec.params.get("role_permutation_noise", 0.2)
        if not isinstance(noise, (int, float)):
            errors.append("role_permutation_noise must be a number")
        elif not 0.0 <= noise <= 1.0:
            errors.append("role_permutation_noise must be between 0 and 1")
        
        return errors


class T5Runner(SubprocessRunner):
    """Runner for T5: Self-assembling modular circuits.
    
    This track experiments with self-assembling neural circuits.
    Key parameters:
    - max_nodes: Maximum number of assembly nodes
    - typed_io_enforced: Whether to enforce typed I/O
    - deterministic_fallback: Whether to use deterministic fallback
    - planner_prune: Whether to use planner pruning
    
    The runner handles circuit assembly and planning benchmarks.
    """
    
    name = "t5"
    description = "Self-assembling modular circuits runner"
    supported_tracks = ["T5"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
    ):
        if command_template is None:
            command_template = (
                "python -m engine.tracks.t5.run "
                "--spec {spec_id} --seed {seed} --run-id {run_id} "
                "--gpu {gpu_ids} --variant {variant} "
                "--params '{params_json}'"
            )
        super().__init__(config, command_template)
    
    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        errors = super().validate_spec(spec)
        
        max_nodes = spec.params.get("max_nodes", 12)
        if not isinstance(max_nodes, int) or max_nodes < 1:
            errors.append("max_nodes must be a positive integer")
        
        return errors


class T6Runner(SubprocessRunner):
    """Runner for T6: Energy-based global decoding.
    
    This track experiments with energy-based models for decoding.
    Key parameters:
    - anneal_temp: Annealing temperature for energy minimization
    
    The runner handles energy-based decoding benchmarks.
    """
    
    name = "t6"
    description = "Energy-based global decoding runner"
    supported_tracks = ["T6"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
    ):
        if command_template is None:
            command_template = (
                "python -m engine.tracks.t6.run "
                "--spec {spec_id} --seed {seed} --run-id {run_id} "
                "--gpu {gpu_ids} --variant {variant} "
                "--params '{params_json}'"
            )
        super().__init__(config, command_template)
    
    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        errors = super().validate_spec(spec)
        
        anneal_temp = spec.params.get("anneal_temp", 0.55)
        if not isinstance(anneal_temp, (int, float)):
            errors.append("anneal_temp must be a number")
        elif not 0.0 <= anneal_temp <= 1.0:
            errors.append("anneal_temp must be between 0 and 1")
        
        return errors


class AnchorRunner(SubprocessRunner):
    """Runner for ANCHOR: Permanent cross-stage reference engine.
    
    The anchor runner provides baseline experiments that are consistent
    across all stages for stable longitudinal comparisons.
    """
    
    name = "anchor"
    description = "Anchor baseline runner for cross-stage comparisons"
    supported_tracks = ["ANCHOR"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
    ):
        if command_template is None:
            command_template = (
                "python -m engine.tracks.anchor.run "
                "--spec {spec_id} --seed {seed} --run-id {run_id} "
                "--gpu {gpu_ids} --stage {stage}"
            )
        super().__init__(config, command_template)