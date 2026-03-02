"""Experiment Knowledge Graph — Graph-based experiment history management.

Provides a graph database interface for experiment runs, enabling:
    - Query: "Which params have historically correlated with Stage 3 promotion?"
    - Find analogous tracks by failure mode profile
    - Suggest next specs based on graph patterns
    - Counterfactual reasoning: "What if we had run X instead of Y?"

Usage:
    kg = ExperimentKnowledgeGraph()
    kg.load_from_artifacts("artifacts/runs")
    kg.load_from_artifacts("artifacts/comparisons")
    
    # Query: which params predict Stage 3 promotion?
    predictors = kg.query_promotion_predictors("T3")
    
    # Find analogous tracks
    similar = kg.find_analogous_tracks("T4")
    
    # Suggest next spec
    suggestion = kg.suggest_next_spec("T3", stage=2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from exp.models import ComparisonReport, RunResult


# ---------------------------------------------------------------------------
# Graph Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A node in the experiment knowledge graph."""
    id: str
    type: str  # "run", "spec", "stage", "track", "param"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the experiment knowledge graph."""
    source: str
    target: str
    relation: str  # "ran_with", "compared_to", "promoted_from", "caused_failure", "correlated_with"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of a graph query."""
    query_type: str
    results: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge Graph Implementation
# ---------------------------------------------------------------------------


class ExperimentKnowledgeGraph:
    """Graph-based experiment history.

    Nodes represent: runs, specs, stages, tracks, params
    Edges represent: relationships between them
    """

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._run_to_spec: dict[str, str] = {}
        self._track_stages: dict[str, set[int]] = {}
        self._param_values: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------------------
    # Loading Data
    # ---------------------------------------------------------------------------

    def load_run(self, run: RunResult) -> None:
        """Load a single run into the graph."""
        # Create run node
        run_node = GraphNode(
            id=run.run_id,
            type="run",
            properties={
                "track_id": run.track_id,
                "stage": run.stage,
                "model_variant": run.model_variant,
                "composite": run.metric_values.get("composite", 0.0),
                "passed": "failure_flags" not in run or len(run.failure_flags) == 0,
                "seed": run.seed,
                "train_cost": run.train_cost,
                "infer_cost": run.infer_cost,
                "latency_p50": run.latency_p50,
            },
        )
        self.nodes[run.run_id] = run_node

        # Create spec node if we have spec_id
        if hasattr(run, "spec_id") and run.spec_id:
            spec_id = run.spec_id
            if spec_id not in self.nodes:
                self.nodes[spec_id] = GraphNode(
                    id=spec_id,
                    type="spec",
                    properties={"track_id": run.track_id, "stage": run.stage},
                )
            # Edge: run -> spec (ran_with)
            self.edges.append(GraphEdge(run.run_id, spec_id, "ran_with"))

        # Track stage info
        if run.track_id not in self._track_stages:
            self._track_stages[run.track_id] = set()
        self._track_stages[run.track_id].add(run.stage)

        # Store param values
        if hasattr(run, "metadata") and "params" in run.metadata:
            self._param_values[run.run_id] = run.metadata["params"]

    def load_comparison(self, report: ComparisonReport) -> None:
        """Load a comparison report into the graph."""
        # Create comparison node
        comp_id = f"comp-{report.candidate_run_ids[0]}-{report.baseline_run_ids[0]}"
        comp_node = GraphNode(
            id=comp_id,
            type="comparison",
            properties={
                "track_id": report.track_id,
                "stage": report.candidate_stage,
                "delta_composite": report.delta_metrics.get("composite", 0.0),
                "anchor_delta": report.anchor_delta_metrics.get("composite", 0.0),
                "overall_pass": report.pass_fail.get("overall_pass", False),
            },
        )
        self.nodes[comp_id] = comp_node

        # Edges: comparison -> candidate runs
        for cand_id in report.candidate_run_ids:
            if cand_id in self.nodes:
                self.edges.append(GraphEdge(comp_id, cand_id, "evaluated"))

        # Edges: comparison -> baseline runs
        for base_id in report.baseline_run_ids:
            if base_id in self.nodes:
                self.edges.append(GraphEdge(comp_id, base_id, "baseline_of"))

        # Edge: comparison -> track
        track_node_id = f"track-{report.track_id}"
        if track_node_id not in self.nodes:
            self.nodes[track_node_id] = GraphNode(
                id=track_node_id,
                type="track",
                properties={"track_id": report.track_id},
            )
        self.edges.append(GraphEdge(comp_id, track_node_id, "for_track"))

    def load_from_artifacts(
        self,
        run_dir: str | Path = "artifacts/runs",
        comparison_dir: str | Path = "artifacts/comparisons",
    ) -> None:
        """Load all runs and comparisons from artifact directories."""
        # Load runs
        run_path = Path(run_dir)
        if run_path.exists():
            for json_file in run_path.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    if "run_id" in data:
                        run = RunResult.from_dict(data)
                        self.load_run(run)
                except Exception:
                    pass  # Skip invalid files

        # Load comparisons
        comp_path = Path(comparison_dir)
        if comp_path.exists():
            for json_file in comp_path.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    if "candidate_run_ids" in data:
                        report = ComparisonReport.from_dict(data)
                        self.load_comparison(report)
                except Exception:
                    pass  # Skip invalid files

    # ---------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------

    def query_promotion_predictors(
        self,
        track_id: str,
        min_stage: int = 2,
    ) -> list[dict[str, float]]:
        """Which params most predict Stage 3+ promotion for this track?

        Returns list of (param, correlation) sorted by absolute correlation.
        """
        # Find Stage 3+ runs for this track
        promoted_runs = []
        non_promoted_runs = []

        for node_id, node in self.nodes.items():
            if node.type == "run" and node.properties.get("track_id") == track_id:
                stage = node.properties.get("stage", 0)
                if stage >= min_stage:
                    if node.properties.get("passed", False):
                        promoted_runs.append(node_id)
                    else:
                        non_promoted_runs.append(node_id)

        if not promoted_runs or not non_promoted_runs:
            return []

        # Collect param values
        param_presence_promoted: dict[str, float] = {}
        param_presence_non: dict[str, float] = {}

        for run_id in promoted_runs:
            params = self._param_values.get(run_id, {})
            for key, val in params.items():
                param_presence_promoted[key] = param_presence_promoted.get(key, 0) + 1

        for run_id in non_promoted_runs:
            params = self._param_values.get(run_id, {})
            for key, val in params.items():
                param_presence_non[key] = param_presence_non.get(key, 0) + 1

        # Compute simple correlation as presence rate difference
        n_promoted = len(promoted_runs)
        n_non = len(non_promoted_runs)

        correlations = []
        all_params = set(param_presence_promoted.keys()) | set(param_presence_non.keys())
        for param in all_params:
            rate_promoted = param_presence_promoted.get(param, 0) / n_promoted
            rate_non = param_presence_non.get(param, 0) / n_non
            correlation = rate_promoted - rate_non
            correlations.append({"param": param, "correlation": correlation})

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return correlations[:10]  # Top 10

    def find_analogous_tracks(
        self,
        track_id: str,
        similarity_metric: str = "failure_mode",
    ) -> list[dict[str, Any]]:
        """Find tracks with similar behavior patterns.

        Args:
            track_id: Reference track
            similarity_metric: "failure_mode" or "performance"

        Returns list of (track_id, similarity_score) sorted by similarity.
        """
        # Get failure mode profile for reference track
        ref_profile = self._get_track_profile(track_id, similarity_metric)
        if not ref_profile:
            return []

        # Compare to all other tracks
        similarities = []
        for other_id in self._track_stages:
            if other_id == track_id:
                continue

            other_profile = self._get_track_profile(other_id, similarity_metric)
            if other_profile:
                sim = self._cosine_similarity(ref_profile, other_profile)
                similarities.append({"track_id": other_id, "similarity": sim})

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:5]  # Top 5

    def suggest_next_spec(
        self,
        track_id: str,
        stage: int,
        budget_hours: float | None = None,
    ) -> dict[str, Any]:
        """Suggest next spec based on graph patterns.

        Args:
            track_id: Current track
            stage: Current stage
            budget_hours: Optional budget constraint

        Returns suggested spec dict with rationale.
        """
        # Find the best-performing run at this stage
        best_run_id = None
        best_composite = float("-inf")

        for node_id, node in self.nodes.items():
            if (
                node.type == "run"
                and node.properties.get("track_id") == track_id
                and node.properties.get("stage") == stage
            ):
                comp = node.properties.get("composite", float("-inf"))
                if comp > best_composite:
                    best_composite = comp
                    best_run_id = node_id

        if not best_run_id:
            return {"suggestion": None, "rationale": "No runs found for this track/stage"}

        # Get params from best run
        best_params = self._param_values.get(best_run_id, {})

        # Find params that correlate with promotion
        predictors = self.query_promotion_predictors(track_id, min_stage=stage + 1)

        # Build suggestion
        suggested_params = best_params.copy()

        # Apply top predictor if it's different from current
        if predictors:
            top_predictor = predictors[0]
            param_name = top_predictor["param"]
            if param_name not in suggested_params:
                suggested_params[param_name] = "optimized_value"  # Would need actual optimization

        return {
            "suggestion": {
                "track_id": track_id,
                "stage": stage + 1,
                "params": suggested_params,
                "based_on_run": best_run_id,
                "expected_composite": best_composite + 0.5,  # Optimistic estimate
            },
            "rationale": f"Based on best run {best_run_id} (composite={best_composite:.2f}) "
            f"and top predictor: {predictors[0]['param'] if predictors else 'none'}",
        }

    def get_track_stage_history(
        self,
        track_id: str,
    ) -> list[dict[str, Any]]:
        """Get full stage-by-stage history for a track."""
        history = []

        for node_id, node in self.nodes.items():
            if node.type == "run" and node.properties.get("track_id") == track_id:
                history.append(
                    {
                        "run_id": node_id,
                        "stage": node.properties.get("stage"),
                        "composite": node.properties.get("composite"),
                        "passed": node.properties.get("passed"),
                        "params": self._param_values.get(node_id, {}),
                    }
                )

        # Sort by stage
        history.sort(key=lambda x: x["stage"] if x["stage"] else 0)
        return history

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _get_track_profile(self, track_id: str, metric: str) -> dict[str, float]:
        """Get behavioral profile for a track."""
        profile: dict[str, float] = {}

        for node_id, node in self.nodes.items():
            if node.type == "run" and node.properties.get("track_id") == track_id:
                if metric == "failure_mode":
                    # Profile based on failure flags
                    passed = node.properties.get("passed", True)
                    profile["pass_rate"] = profile.get("pass_rate", 0.0) + (1.0 if passed else 0.0)
                    profile["count"] = profile.get("count", 0.0) + 1.0

                elif metric == "performance":
                    # Profile based on composite score
                    comp = node.properties.get("composite", 0.0)
                    profile["mean_composite"] = profile.get("mean_composite", 0.0) + comp
                    profile["count"] = profile.get("count", 0.0) + 1.0

        # Normalize
        if "count" in profile and profile["count"] > 0:
            for key in profile:
                if key != "count":
                    profile[key] /= profile["count"]

        return profile

    def _cosine_similarity(self, a: dict[str, float], b: dict[str, float]) -> float:
        """Compute cosine similarity between two profiles."""
        all_keys = set(a.keys()) | set(b.keys())

        dot_product = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in all_keys)
        norm_a = sum(a.get(k, 0.0) ** 2 for k in all_keys) ** 0.5
        norm_b = sum(b.get(k, 0.0) ** 2 for k in all_keys) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dict."""
        return {
            "nodes": [
                {"id": n.id, "type": n.type, "properties": n.properties}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentKnowledgeGraph":
        """Deserialize graph from dict."""
        kg = cls()
        for node_data in data.get("nodes", []):
            node = GraphNode(
                id=node_data["id"],
                type=node_data["type"],
                properties=node_data.get("properties", {}),
            )
            kg.nodes[node.id] = node

        for edge_data in data.get("edges", []):
            edge = GraphEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                relation=edge_data["relation"],
            )
            kg.edges.append(edge)

        return kg


# ---------------------------------------------------------------------------
# Integration with Memo
# ---------------------------------------------------------------------------


def format_knowledge_graph_summary(kg: ExperimentKnowledgeGraph) -> str:
    """Format knowledge graph summary as markdown."""
    lines = []
    lines.append("## Experiment Knowledge Graph Summary")
    lines.append("")
    lines.append(f"- **Total Nodes**: {len(kg.nodes)}")
    lines.append(f"- **Total Edges**: {len(kg.edges)}")
    lines.append("")

    # Track summaries
    lines.append("### Tracks in Graph")
    lines.append("")
    for track_id in sorted(kg._track_stages.keys()):
        stages = sorted(kg._track_stages[track_id])
        lines.append(f"- **{track_id}**: Stages {stages[0]}–{stages[-1] if len(stages) > 1 else stages[0]}")

    return "\n".join(lines)

