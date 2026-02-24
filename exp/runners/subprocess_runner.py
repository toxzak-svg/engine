"""Subprocess runner that executes external commands/scripts."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..models import ExperimentSpec, RunResult
from .base import BaseRunner, RunnerConfig, RunnerResult, RunnerStatus


class SubprocessRunner(BaseRunner):
    """Runner that executes external commands or scripts.
    
    This runner can execute:
    - Shell scripts
    - Python scripts
    - Docker containers
    - Any executable command
    
    The command can use template variables that are substituted with spec values:
    - {spec_id}: The spec ID
    - {track_id}: The track ID
    - {stage}: The stage number
    - {seed}: The random seed
    - {run_id}: The unique run ID
    - {variant}: The model variant
    - {gpu_ids}: Comma-separated GPU device IDs
    - {checkpoint_dir}: Checkpoint directory
    - {artifact_dir}: Artifact output directory
    """
    
    name = "subprocess"
    description = "Executes external commands or scripts"
    supported_tracks = ["T1", "T2", "T3", "T4", "T5", "T6", "ANCHOR"]
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        command_template: str | None = None,
        result_parser: str = "json",
    ):
        """Initialize the subprocess runner.
        
        Args:
            config: Runner configuration
            command_template: Command template with variable substitution
            result_parser: How to parse the output ('json', 'lines', 'regex')
        """
        super().__init__(config)
        self.command_template = command_template
        self.result_parser = result_parser
        self._process: subprocess.Popen | None = None
    
    def _build_command(
        self,
        spec: ExperimentSpec,
        seed: int,
        run_id: str,
    ) -> str:
        """Build the command with variable substitution.
        
        Args:
            spec: Experiment specification
            seed: Random seed
            run_id: Unique run ID
            
        Returns:
            The command string with variables substituted
        """
        if not self.command_template:
            raise ValueError("command_template is required for SubprocessRunner")
        
        gpu_ids = ",".join(str(gpu) for gpu in self.config.gpu_device_ids)
        
        replacements = {
            "{spec_id}": spec.id,
            "{track_id}": spec.track_id,
            "{stage}": str(spec.stage),
            "{seed}": str(seed),
            "{run_id}": run_id,
            "{variant}": spec.model_variant,
            "{gpu_ids}": gpu_ids,
            "{checkpoint_dir}": self.config.checkpoint_dir or "",
            "{artifact_dir}": self.config.artifact_dir or "",
            "{train_budget}": str(spec.train_budget_gpu_h),
            "{infer_budget}": str(spec.infer_budget_gpu_h),
            "{max_context}": str(spec.max_context),
        }
        
        command = self.command_template
        for placeholder, value in replacements.items():
            command = command.replace(placeholder, str(value))
        
        # Handle JSON params
        if "{params_json}" in command:
            command = command.replace("{params_json}", json.dumps(spec.params))
        
        return command
    
    def _build_environment(self) -> dict[str, str]:
        """Build the environment variables for the subprocess.
        
        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        
        # Add configured environment variables
        env.update(self.config.environment)
        
        # Set GPU visibility
        if self.config.gpu_device_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu) for gpu in self.config.gpu_device_ids
            )
        
        return env
    
    def _parse_result(
        self,
        output: str,
        spec: ExperimentSpec,
        seed: int,
        run_id: str,
        commit_sha: str,
    ) -> RunResult | None:
        """Parse the subprocess output to create a RunResult.
        
        Args:
            output: The stdout from the subprocess
            spec: Experiment specification
            seed: Random seed
            run_id: Unique run ID
            commit_sha: Git commit SHA
            
        Returns:
            RunResult if parsing succeeds, None otherwise
        """
        if self.result_parser == "json":
            return self._parse_json_result(output, spec, seed, run_id, commit_sha)
        return None
    
    def _parse_json_result(
        self,
        output: str,
        spec: ExperimentSpec,
        seed: int,
        run_id: str,
        commit_sha: str,
    ) -> RunResult | None:
        """Parse JSON output to create a RunResult.
        
        Expected JSON format:
        {
            "train_cost": float,
            "infer_cost": float,
            "latency_p50": float,
            "latency_p95": float,
            "energy_kwh": float,
            "metric_values": {...},
            "failure_flags": [...],
            "benchmark_scores": {...}
        }
        """
        try:
            # Try to find JSON in the output
            lines = output.strip().split("\n")
            json_line = None
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    json_line = line
                    break
                # Handle multi-line JSON
                if line.startswith("{"):
                    json_line = line
                    break
            
            if not json_line:
                # Try parsing entire output as JSON
                json_line = output.strip()
            
            data = json.loads(json_line)
            
            return RunResult(
                run_id=run_id,
                spec_id=spec.id,
                commit_sha=commit_sha,
                seed=seed,
                train_cost=float(data.get("train_cost", 0.0)),
                infer_cost=float(data.get("infer_cost", 0.0)),
                latency_p50=float(data.get("latency_p50", 0.0)),
                latency_p95=float(data.get("latency_p95", 0.0)),
                energy_kwh=float(data.get("energy_kwh", 0.0)),
                metric_values=dict(data.get("metric_values", {})),
                failure_flags=list(data.get("failure_flags", [])),
                track_id=spec.track_id,
                stage=spec.stage,
                model_variant=spec.model_variant,
                benchmark_scores=dict(data.get("benchmark_scores", {})),
                metadata=dict(data.get("metadata", {})),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return None
    
    def execute(
        self,
        spec: ExperimentSpec,
        seed: int,
        run_id: str,
        commit_sha: str,
    ) -> RunnerResult:
        """Execute an experiment via subprocess.
        
        Args:
            spec: The experiment specification to execute
            seed: Random seed for reproducibility
            run_id: Unique identifier for this run
            commit_sha: Git commit SHA for tracking
            
        Returns:
            RunnerResult containing the execution outcome
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
        
        # Check for dry run
        if self.config.dry_run:
            logs.append("[SUBPROCESS] Dry run mode - skipping execution")
            self._set_status(RunnerStatus.COMPLETED)
            return RunnerResult(
                success=True,
                run_result=None,
                status=RunnerStatus.COMPLETED,
                execution_time_seconds=time.time() - start_time,
                logs=logs,
                metadata={"dry_run": True, "command": self.command_template},
            )
        
        self._set_status(RunnerStatus.RUNNING)
        
        try:
            command = self._build_command(spec, seed, run_id)
            logs.append(f"[SUBPROCESS] Executing: {command}")
            
            env = self._build_environment()
            cwd = self.config.working_directory
            
            # Execute the subprocess
            self._process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd,
            )
            
            try:
                stdout, stderr = self._process.communicate(
                    timeout=self.config.timeout_seconds if self.config.timeout_seconds > 0 else None
                )
            except subprocess.TimeoutExpired:
                self._process.kill()
                logs.append(f"[SUBPROCESS] Timeout after {self.config.timeout_seconds}s")
                self._set_status(RunnerStatus.TIMEOUT)
                return RunnerResult(
                    success=False,
                    run_result=None,
                    error_message=f"Timeout after {self.config.timeout_seconds} seconds",
                    status=RunnerStatus.TIMEOUT,
                    execution_time_seconds=time.time() - start_time,
                    logs=logs,
                )
            
            # Capture output
            if stdout:
                logs.append(f"[SUBPROCESS] stdout: {stdout[:1000]}")
            if stderr:
                logs.append(f"[SUBPROCESS] stderr: {stderr[:1000]}")
            
            # Check return code
            if self._process.returncode != 0:
                logs.append(f"[SUBPROCESS] Process failed with return code {self._process.returncode}")
                self._set_status(RunnerStatus.FAILED)
                return RunnerResult(
                    success=False,
                    run_result=None,
                    error_message=f"Process exited with code {self._process.returncode}: {stderr}",
                    status=RunnerStatus.FAILED,
                    execution_time_seconds=time.time() - start_time,
                    logs=logs,
                )
            
            # Parse result
            run_result = self._parse_result(stdout, spec, seed, run_id, commit_sha)
            
            if run_result is None:
                logs.append("[SUBPROCESS] Warning: Could not parse RunResult from output")
            
            self._set_status(RunnerStatus.COMPLETED)
            logs.append("[SUBPROCESS] Execution completed successfully")
            
            return RunnerResult(
                success=True,
                run_result=run_result,
                status=RunnerStatus.COMPLETED,
                execution_time_seconds=time.time() - start_time,
                logs=logs,
                metadata={
                    "command": command,
                    "return_code": self._process.returncode,
                    "seed": seed,
                },
            )
            
        except Exception as e:
            logs.append(f"[SUBPROCESS] Error: {str(e)}")
            self._set_status(RunnerStatus.FAILED)
            return RunnerResult(
                success=False,
                run_result=None,
                error_message=str(e),
                status=RunnerStatus.FAILED,
                execution_time_seconds=time.time() - start_time,
                logs=logs,
            )
        finally:
            self._process = None
    
    def cancel(self) -> bool:
        """Cancel a running subprocess.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception:
                return False
        
        self._set_status(RunnerStatus.CANCELLED)
        return True


class DockerRunner(SubprocessRunner):
    """Runner that executes experiments in Docker containers.
    
    This is a specialized SubprocessRunner that wraps commands in Docker.
    """
    
    name = "docker"
    description = "Executes experiments in Docker containers"
    
    def __init__(
        self,
        config: RunnerConfig | None = None,
        image: str = "engine-experiment:latest",
        command_template: str | None = None,
        volumes: dict[str, str] | None = None,
    ):
        """Initialize the Docker runner.
        
        Args:
            config: Runner configuration
            image: Docker image to use
            command_template: Command template (will be wrapped in docker run)
            volumes: Dictionary mapping host paths to container paths
        """
        self.image = image
        self.volumes = volumes or {}
        
        # Build docker command wrapper
        docker_cmd = f"docker run --rm"
        
        # Add GPU support
        if config and config.gpu_device_ids:
            docker_cmd += f" --gpus {'"device=' + ','.join(str(g) for g in config.gpu_device_ids) + '"'}"
        
        # Add volumes
        for host_path, container_path in self.volumes.items():
            docker_cmd += f" -v {host_path}:{container_path}"
        
        # Add environment variables
        if config and config.environment:
            for key, value in config.environment.items():
                docker_cmd += f" -e {key}={value}"
        
        # Add checkpoint and artifact directories
        if config:
            if config.checkpoint_dir:
                docker_cmd += f" -v {config.checkpoint_dir}:/checkpoints"
            if config.artifact_dir:
                docker_cmd += f" -v {config.artifact_dir}:/artifacts"
        
        # Wrap command in docker
        full_template = f'{docker_cmd} {image} bash -c "{command_template}"'
        
        super().__init__(config, full_template)