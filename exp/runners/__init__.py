"""Experiment runners for executing model-engine experiments.

This module provides a runner abstraction for executing experiments.
Runners can be:
- SimulatorRunner: Uses the existing simulator for mock results
- SubprocessRunner: Executes external commands/scripts
- Track-specific runners: Custom implementations for each track (T1-T6)
"""

from __future__ import annotations

from .base import BaseRunner, RunnerResult, RunnerConfig, RunnerStatus
from .simulator_runner import SimulatorRunner
from .subprocess_runner import SubprocessRunner, DockerRunner
from .track_runners import (
    T1Runner,
    T2Runner,
    T3Runner,
    T4Runner,
    T5Runner,
    T6Runner,
    AnchorRunner,
)
from .factory import get_runner, RunnerRegistry, create_runner_from_config

__all__ = [
    "BaseRunner",
    "RunnerResult",
    "RunnerConfig",
    "RunnerStatus",
    "SimulatorRunner",
    "SubprocessRunner",
    "DockerRunner",
    "T1Runner",
    "T2Runner",
    "T3Runner",
    "T4Runner",
    "T5Runner",
    "T6Runner",
    "AnchorRunner",
    "get_runner",
    "RunnerRegistry",
    "create_runner_from_config",
]
