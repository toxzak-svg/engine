"""Base runner interface for experiment execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models import ExperimentSpec, RunResult


class RunnerStatus(Enum):
    """Status of a runner execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class RunnerResult:
    """Result from a runner execution.
    
    Attributes:
        success: Whether the execution was successful
        run_result: The RunResult if successful, None otherwise
        error_message: Error message if failed, None otherwise
        status: The final status of the execution
        execution_time_seconds: Time taken to execute
        logs: Captured logs from execution
        metadata: Additional metadata about the execution
    """
    success: bool
    run_result: RunResult | None
    error_message: str | None = None
    status: RunnerStatus = RunnerStatus.COMPLETED
    execution_time_seconds: float = 0.0
    logs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    """Configuration for a runner.
    
    Attributes:
        timeout_seconds: Maximum execution time in seconds (0 = no timeout)
        working_directory: Directory to execute from
        environment: Environment variables to set
        capture_logs: Whether to capture execution logs
        dry_run: If True, simulate execution without actually running
        gpu_device_ids: List of GPU device IDs to use
        checkpoint_dir: Directory for checkpoints
        artifact_dir: Directory for artifacts
    """
    timeout_seconds: float = 0.0
    working_directory: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    capture_logs: bool = True
    dry_run: bool = False
    gpu_device_ids: list[int] = field(default_factory=lambda: [0])
    checkpoint_dir: str | None = None
    artifact_dir: str | None = None


class BaseRunner(ABC):
    """Abstract base class for experiment runners.
    
    All runners must implement the execute method which takes an ExperimentSpec
    and returns a RunnerResult containing the RunResult or error information.
    """
    
    name: str = "base"
    description: str = "Base runner - do not use directly"
    supported_tracks: list[str] = []
    
    def __init__(self, config: RunnerConfig | None = None):
        self.config = config or RunnerConfig()
        self._status = RunnerStatus.PENDING
    
    @property
    def status(self) -> RunnerStatus:
        """Current status of the runner."""
        return self._status
    
    @abstractmethod
    def execute(
        self,
        spec: ExperimentSpec,
        seed: int,
        run_id: str,
        commit_sha: str,
    ) -> RunnerResult:
        """Execute an experiment specification.
        
        Args:
            spec: The experiment specification to execute
            seed: Random seed for reproducibility
            run_id: Unique identifier for this run
            commit_sha: Git commit SHA for tracking
            
        Returns:
            RunnerResult containing the execution outcome
        """
        ...
    
    def validate_spec(self, spec: ExperimentSpec) -> list[str]:
        """Validate that the spec is compatible with this runner.
        
        Args:
            spec: The experiment specification to validate
            
        Returns:
            List of validation error messages, empty if valid
        """
        errors: list[str] = []
        
        if self.supported_tracks and spec.track_id not in self.supported_tracks:
            errors.append(
                f"Runner '{self.name}' does not support track '{spec.track_id}'. "
                f"Supported tracks: {', '.join(self.supported_tracks)}"
            )
        
        return errors
    
    def pre_execute(self, spec: ExperimentSpec, seed: int) -> None:
        """Hook called before execution.
        
        Override to implement custom pre-execution logic like:
        - Setting up directories
        - Validating GPU availability
        - Checking dependencies
        """
        pass
    
    def post_execute(
        self,
        spec: ExperimentSpec,
        result: RunnerResult,
    ) -> None:
        """Hook called after execution.
        
        Override to implement custom post-execution logic like:
        - Cleanup
        - Artifact collection
        - Notification
        """
        pass
    
    def cancel(self) -> bool:
        """Cancel a running execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        self._status = RunnerStatus.CANCELLED
        return True
    
    def _set_status(self, status: RunnerStatus) -> None:
        """Set the runner status."""
        self._status = status