"""Simulator runner that wraps the existing simulator for backward compatibility."""

from __future__ import annotations

import time
from typing import Any

from ..models import ExperimentSpec
from ..simulator import simulate_run
from .base import BaseRunner, RunnerConfig, RunnerResult, RunnerStatus


class SimulatorRunner(BaseRunner):
    """Runner that uses the existing simulator for mock experiment results.
    
    This runner provides backward compatibility with the existing simulation
    system while allowing the codebase to use the new runner interface.
    It's useful for:
    - Testing the pipeline without real experiments
    - Development and debugging
    - CI/CD environments without GPU access
    """
    
    name = "simulator"
    description = "Uses the existing simulator for mock experiment results"
    supported_tracks = ["T1", "T2", "T3", "T4", "T5", "T6", "ANCHOR"]
    
    def __init__(self, config: RunnerConfig | None = None):
        super().__init__(config)
        self._simulation_delay: float = config.extra.get("simulation_delay", 0.0) if config and hasattr(config, 'extra') else 0.0
    
    def execute(
        self,
        spec: ExperimentSpec,
        seed: int,
        run_id: str,
        commit_sha: str,
    ) -> RunnerResult:
        """Execute a simulated experiment run.
        
        Args:
            spec: The experiment specification to simulate
            seed: Random seed for reproducibility
            run_id: Unique identifier for this run
            commit_sha: Git commit SHA for tracking
            
        Returns:
            RunnerResult containing the simulated RunResult
        """
        start_time = time.time()
        logs: list[str] = []
        
        # Validate spec
        errors = self.validate_spec(spec)
        if errors:
            return RunnerResult(
                success=False,
                run_result=None,
                error_message="; ".join(errors),
                status=RunnerStatus.FAILED,
                execution_time_seconds=time.time() - start_time,
                logs=logs,
            )
        
        self._set_status(RunnerStatus.RUNNING)
        logs.append(f"[SIMULATOR] Starting simulation for spec {spec.id}")
        logs.append(f"[SIMULATOR] Track: {spec.track_id}, Variant: {spec.model_variant}, Seed: {seed}")
        
        # Check for dry run
        if self.config.dry_run:
            logs.append("[SIMULATOR] Dry run mode - skipping actual simulation")
            self._set_status(RunnerStatus.COMPLETED)
            return RunnerResult(
                success=True,
                run_result=None,
                status=RunnerStatus.COMPLETED,
                execution_time_seconds=time.time() - start_time,
                logs=logs,
                metadata={"dry_run": True},
            )
        
        try:
            # Add optional simulation delay for realistic testing
            if self._simulation_delay > 0:
                time.sleep(self._simulation_delay)
            
            # Use the existing simulator
            run_result = simulate_run(
                spec=spec,
                seed=seed,
                run_id=run_id,
                commit_sha=commit_sha,
            )
            
            logs.append(f"[SIMULATOR] Simulation completed successfully")
            logs.append(f"[SIMULATOR] Composite score: {run_result.metric_values.get('composite', 'N/A')}")
            
            if run_result.failure_flags:
                logs.append(f"[SIMULATOR] Failure flags: {', '.join(run_result.failure_flags)}")
            
            self._set_status(RunnerStatus.COMPLETED)
            
            return RunnerResult(
                success=True,
                run_result=run_result,
                status=RunnerStatus.COMPLETED,
                execution_time_seconds=time.time() - start_time,
                logs=logs,
                metadata={
                    "simulated": True,
                    "seed": seed,
                    "variant": spec.model_variant,
                },
            )
            
        except Exception as e:
            logs.append(f"[SIMULATOR] Error during simulation: {str(e)}")
            self._set_status(RunnerStatus.FAILED)
            
            return RunnerResult(
                success=False,
                run_result=None,
                error_message=str(e),
                status=RunnerStatus.FAILED,
                execution_time_seconds=time.time() - start_time,
                logs=logs,
            )
    
    def execute_batch(
        self,
        specs: list[ExperimentSpec],
        seeds: list[int],
        commit_sha: str,
    ) -> list[RunnerResult]:
        """Execute a batch of simulations.
        
        Args:
            specs: List of experiment specifications
            seeds: List of seeds for each spec
            commit_sha: Git commit SHA for tracking
            
        Returns:
            List of RunnerResults
        """
        results: list[RunnerResult] = []
        
        for spec in specs:
            for seed in seeds:
                import uuid
                run_id = f"{spec.id}-s{seed}-{uuid.uuid4().hex[:8]}"
                result = self.execute(spec, seed, run_id, commit_sha)
                results.append(result)
        
        return results