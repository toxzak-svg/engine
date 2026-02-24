"""Runner factory and registry for managing runner instances."""

from __future__ import annotations

from typing import Callable, Type

from ..constants import TRACKS
from ..models import ExperimentSpec
from .base import BaseRunner, RunnerConfig
from .simulator_runner import SimulatorRunner
from .subprocess_runner import SubprocessRunner
from .track_runners import (
    AnchorRunner,
    T1Runner,
    T2Runner,
    T3Runner,
    T4Runner,
    T5Runner,
    T6Runner,
)
from .inverse_arm_runners import (
    InverseArmRunner,
    DeliberationCollapseRunner,
    CounterfactualAuditRunner,
    ActiveRecallRouterRunner,
    ProofCarryingOutputsRunner,
    NoiseInjectionEnsembleRunner,
    LatentPlanSwappingRunner,
    AdversarialUserSimRunner,
)


class RunnerRegistry:
    """Registry for runner types.
    
    Allows registration and lookup of runner types by name or track.
    """
    
    _runners: dict[str, Type[BaseRunner]] = {}
    _track_mapping: dict[str, str] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        runner_class: Type[BaseRunner],
        tracks: list[str] | None = None,
    ) -> None:
        """Register a runner type.
        
        Args:
            name: Unique name for the runner
            runner_class: The runner class to register
            tracks: List of track IDs this runner supports
        """
        cls._runners[name] = runner_class
        if tracks:
            for track in tracks:
                cls._track_mapping[track.upper()] = name
    
    @classmethod
    def get_runner_class(cls, name: str) -> Type[BaseRunner] | None:
        """Get a runner class by name.
        
        Args:
            name: The runner name
            
        Returns:
            The runner class, or None if not found
        """
        return cls._runners.get(name)
    
    @classmethod
    def get_runner_for_track(cls, track_id: str) -> Type[BaseRunner] | None:
        """Get the appropriate runner class for a track.
        
        Args:
            track_id: The track ID (e.g., 'T1', 'T2')
            
        Returns:
            The runner class for the track, or None if not found
        """
        runner_name = cls._track_mapping.get(track_id.upper())
        if runner_name:
            return cls._runners.get(runner_name)
        return None
    
    @classmethod
    def list_runners(cls) -> list[str]:
        """List all registered runner names."""
        return list(cls._runners.keys())
    
    @classmethod
    def list_track_mappings(cls) -> dict[str, str]:
        """List all track-to-runner mappings."""
        return dict(cls._track_mapping)


# Register built-in runners
RunnerRegistry.register("simulator", SimulatorRunner, list(TRACKS))
RunnerRegistry.register("subprocess", SubprocessRunner, list(TRACKS))
RunnerRegistry.register("t1", T1Runner, ["T1"])
RunnerRegistry.register("t2", T2Runner, ["T2"])
RunnerRegistry.register("t3", T3Runner, ["T3"])
RunnerRegistry.register("t4", T4Runner, ["T4"])
RunnerRegistry.register("t5", T5Runner, ["T5"])
RunnerRegistry.register("t6", T6Runner, ["T6"])
RunnerRegistry.register("anchor", AnchorRunner, ["ANCHOR"])

# Register inverse arm runners (T7-T13)
RunnerRegistry.register("inverse_arm", InverseArmRunner, ["T7", "T8", "T9", "T10", "T11", "T12", "T13"])
RunnerRegistry.register("deliberation_collapse", DeliberationCollapseRunner, ["T7"])
RunnerRegistry.register("counterfactual_audit", CounterfactualAuditRunner, ["T8"])
RunnerRegistry.register("active_recall_router", ActiveRecallRouterRunner, ["T9"])
RunnerRegistry.register("proof_carrying_outputs", ProofCarryingOutputsRunner, ["T10"])
RunnerRegistry.register("noise_injection_ensemble", NoiseInjectionEnsembleRunner, ["T11"])
RunnerRegistry.register("latent_plan_swapping", LatentPlanSwappingRunner, ["T12"])
RunnerRegistry.register("adversarial_user_sim", AdversarialUserSimRunner, ["T13"])


def get_runner(
    track_id: str,
    config: RunnerConfig | None = None,
    runner_type: str | None = None,
    command_template: str | None = None,
) -> BaseRunner:
    """Get the appropriate runner for a track.
    
    Args:
        track_id: The track ID (e.g., 'T1', 'T2')
        config: Optional runner configuration
        runner_type: Optional runner type override ('simulator', 'subprocess', etc.)
        command_template: Optional command template for subprocess runners
        
    Returns:
        Configured runner instance
        
    Raises:
        ValueError: If no suitable runner is found
    """
    track_id = track_id.upper()
    
    # If runner_type is specified, use that
    if runner_type:
        runner_class = RunnerRegistry.get_runner_class(runner_type)
        if runner_class is None:
            raise ValueError(f"Unknown runner type: {runner_type}")
        
        # Handle subprocess-based runners with custom command
        if command_template and hasattr(runner_class, '__init__'):
            try:
                return runner_class(config=config, command_template=command_template)
            except TypeError:
                return runner_class(config=config)
        
        return runner_class(config=config)
    
    # Otherwise, use track-specific runner
    runner_class = RunnerRegistry.get_runner_for_track(track_id)
    
    if runner_class is None:
        # Fall back to simulator for unknown tracks
        runner_class = SimulatorRunner
    
    # Handle subprocess-based runners with custom command
    if command_template and hasattr(runner_class, '__init__'):
        try:
            return runner_class(config=config, command_template=command_template)
        except TypeError:
            return runner_class(config=config)
    
    return runner_class(config=config)


def create_runner_from_config(
    config_dict: dict,
    track_id: str | None = None,
) -> BaseRunner:
    """Create a runner from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary with keys:
            - type: Runner type ('simulator', 'subprocess', 't1', etc.)
            - timeout_seconds: Maximum execution time
            - working_directory: Directory to execute from
            - environment: Environment variables
            - gpu_device_ids: List of GPU device IDs
            - command_template: Command template for subprocess runners
            - dry_run: Whether to simulate execution
        track_id: Optional track ID for track-specific defaults
        
    Returns:
        Configured runner instance
    """
    runner_config = RunnerConfig(
        timeout_seconds=config_dict.get("timeout_seconds", 0.0),
        working_directory=config_dict.get("working_directory"),
        environment=config_dict.get("environment", {}),
        capture_logs=config_dict.get("capture_logs", True),
        dry_run=config_dict.get("dry_run", False),
        gpu_device_ids=config_dict.get("gpu_device_ids", [0]),
        checkpoint_dir=config_dict.get("checkpoint_dir"),
        artifact_dir=config_dict.get("artifact_dir"),
    )
    
    runner_type = config_dict.get("type")
    command_template = config_dict.get("command_template")
    
    # Infer track from runner type if not provided
    if track_id is None and runner_type and runner_type.startswith("t"):
        track_id = runner_type.upper()
    
    return get_runner(
        track_id=track_id or "T1",
        config=runner_config,
        runner_type=runner_type,
        command_template=command_template,
    )