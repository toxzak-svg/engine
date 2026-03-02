"""Tests for exp.knowledge_graph — Experiment Knowledge Graph."""

from __future__ import annotations

from exp.knowledge_graph import (
    ExperimentKnowledgeGraph,
    GraphNode,
    GraphEdge,
    QueryResult,
    format_knowledge_graph_summary,
)
from exp.models import ComparisonReport, RunResult


# ---------------------------------------------------------------------------
# Test Data Helpers
# ---------------------------------------------------------------------------

def _make_run(
    run_id: str,
    track_id: str,
    stage: int,
    composite: float = 60.0,
    passed: bool = True,
    params: dict | None = None,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        spec_id=f"spec-{run_id}",
        commit_sha="abc123",
        seed=42,
        train_cost=50.0,
        infer_cost=10.0,
        latency_p50=100.0,
        latency_p95=128.0,
        energy_kwh=0.5,
        metric_values={"composite": composite},
        failure_flags=[] if passed else ["some_failure"],
        track_id=track_id,
        stage=stage,
        model_variant="T1-E1",
        benchmark_scores={"needle_32k": 55.0},
        metadata={"params": params or {}},
    )


def _make_comparison(
    candidate_ids: list[str],
    baseline_ids: list[str],
    track_id: str,
    stage: int,
    delta_composite: float = 2.0,
    pass_overall: bool = True,
) -> ComparisonReport:
    return ComparisonReport(
        candidate_run_ids=candidate_ids,
        baseline_run_ids=baseline_ids,
        delta_metrics={"composite": delta_composite},
        significance_tests={"ci95_excludes_zero": True},
        pass_fail={"overall_pass": pass_overall},
        candidate_stage=stage,
        track_id=track_id,
    )


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------

class TestGraphNode:
    def test_create_node(self) -> None:
        node = GraphNode(id="test-1", type="run", properties={"key": "value"})
        assert node.id == "test-1"
        assert node.type == "run"
        assert node.properties["key"] == "value"

    def test_node_default_properties(self) -> None:
        node = GraphNode(id="test-2", type="spec")
        assert node.properties == {}


# ---------------------------------------------------------------------------
# GraphEdge
# ---------------------------------------------------------------------------

class TestGraphEdge:
    def test_create_edge(self) -> None:
        edge = GraphEdge(source="run-1", target="spec-1", relation="ran_with")
        assert edge.source == "run-1"
        assert edge.target == "spec-1"
        assert edge.relation == "ran_with"

    def test_edge_with_properties(self) -> None:
        edge = GraphEdge(
            source="run-1",
            target="run-2",
            relation="compared_to",
            properties={"delta": 2.5},
        )
        assert edge.properties["delta"] == 2.5


# ---------------------------------------------------------------------------
# ExperimentKnowledgeGraph
# ---------------------------------------------------------------------------

class TestKnowledgeGraphBasics:
    def test_init_empty(self) -> None:
        kg = ExperimentKnowledgeGraph()
        assert len(kg.nodes) == 0
        assert len(kg.edges) == 0

    def test_load_run_creates_node(self) -> None:
        kg = ExperimentKnowledgeGraph()
        run = _make_run("run-1", "T1", stage=1, composite=65.0)
        kg.load_run(run)

        assert "run-1" in kg.nodes
        assert kg.nodes["run-1"].type == "run"
        assert kg.nodes["run-1"].properties["track_id"] == "T1"
        assert kg.nodes["run-1"].properties["composite"] == 65.0

    def test_load_run_tracks_stage(self) -> None:
        kg = ExperimentKnowledgeGraph()
        run = _make_run("run-1", "T3", stage=2)
        kg.load_run(run)

        assert "T3" in kg._track_stages
        assert 2 in kg._track_stages["T3"]

    def test_load_run_stores_params(self) -> None:
        kg = ExperimentKnowledgeGraph()
        run = _make_run("run-1", "T3", stage=2, params={"compression_ratio": 0.8})
        kg.load_run(run)

        assert "run-1" in kg._param_values
        assert kg._param_values["run-1"]["compression_ratio"] == 0.8

    def test_load_run_failed_flag(self) -> None:
        kg = ExperimentKnowledgeGraph()
        run_failed = _make_run("run-1", "T1", stage=1, passed=False)
        kg.load_run(run_failed)

        assert kg.nodes["run-1"].properties["passed"] is False

    def test_load_comparison_creates_node(self) -> None:
        kg = ExperimentKnowledgeGraph()
        # Need runs first for edges
        run1 = _make_run("cand-1", "T1", stage=1)
        run2 = _make_run("base-1", "T1", stage=1)
        kg.load_run(run1)
        kg.load_run(run2)

        comp = _make_comparison(["cand-1"], ["base-1"], "T1", stage=1)
        kg.load_comparison(comp)

        # Check comparison node was created
        comp_nodes = [n for n in kg.nodes.values() if n.type == "comparison"]
        assert len(comp_nodes) == 1


class TestQueries:
    def test_query_promotion_predictors_empty(self) -> None:
        kg = ExperimentKnowledgeGraph()
        result = kg.query_promotion_predictors("T3")
        assert result == []

    def test_query_promotion_predictors_with_data(self) -> None:
        kg = ExperimentKnowledgeGraph()

        # Add promoted runs
        for i in range(3):
            run = _make_run(
                f"promoted-{i}",
                "T3",
                stage=3,
                composite=70.0,
                passed=True,
                params={"key_param": "value_a"},
            )
            kg.load_run(run)

        # Add non-promoted runs
        for i in range(2):
            run = _make_run(
                f"nonpromoted-{i}",
                "T3",
                stage=3,
                composite=55.0,
                passed=False,
                params={"key_param": "value_b"},
            )
            kg.load_run(run)

        predictors = kg.query_promotion_predictors("T3", min_stage=3)
        assert len(predictors) > 0
        assert predictors[0]["param"] == "key_param"

    def test_find_analogous_tracks_empty(self) -> None:
        kg = ExperimentKnowledgeGraph()
        result = kg.find_analogous_tracks("T1")
        assert result == []

    def test_find_analogous_tracks_with_data(self) -> None:
        kg = ExperimentKnowledgeGraph()

        # Add runs for T1
        for i in range(3):
            run = _make_run(f"t1-{i}", "T1", stage=1, composite=60.0 + i, passed=True)
            kg.load_run(run)

        # Add runs for T2 (similar behavior)
        for i in range(3):
            run = _make_run(f"t2-{i}", "T2", stage=1, composite=61.0 + i, passed=True)
            kg.load_run(run)

        similar = kg.find_analogous_tracks("T1")
        # Should find T2 as similar
        assert any(r["track_id"] == "T2" for r in similar)

    def test_suggest_next_spec_no_runs(self) -> None:
        kg = ExperimentKnowledgeGraph()
        suggestion = kg.suggest_next_spec("T3", stage=1)

        assert suggestion["suggestion"] is None
        assert "No runs found" in suggestion["rationale"]

    def test_suggest_next_spec_with_runs(self) -> None:
        kg = ExperimentKnowledgeGraph()

        # Add a run
        run = _make_run(
            "best-run",
            "T3",
            stage=2,
            composite=68.0,
            passed=True,
            params={"compression_ratio": 0.8},
        )
        kg.load_run(run)

        suggestion = kg.suggest_next_spec("T3", stage=2)
        assert suggestion["suggestion"] is not None
        assert suggestion["suggestion"]["track_id"] == "T3"
        assert suggestion["suggestion"]["stage"] == 3

    def test_get_track_stage_history(self) -> None:
        kg = ExperimentKnowledgeGraph()

        run1 = _make_run("run-1", "T3", stage=1, composite=60.0)
        run2 = _make_run("run-2", "T3", stage=2, composite=65.0)
        run3 = _make_run("run-3", "T3", stage=3, composite=70.0)

        kg.load_run(run1)
        kg.load_run(run2)
        kg.load_run(run3)

        history = kg.get_track_stage_history("T3")
        assert len(history) == 3
        # Should be sorted by stage
        assert history[0]["stage"] == 1
        assert history[1]["stage"] == 2
        assert history[2]["stage"] == 3


class TestSerialization:
    def test_to_dict_serializes(self) -> None:
        kg = ExperimentKnowledgeGraph()
        node = GraphNode(id="test", type="run", properties={"key": "value"})
        kg.nodes["test"] = node
        kg.edges.append(GraphEdge("test", "other", "relates_to"))

        d = kg.to_dict()

        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["id"] == "test"
        assert len(d["edges"]) == 1

    def test_from_dict_deserializes(self) -> None:
        data = {
            "nodes": [
                {"id": "run-1", "type": "run", "properties": {"composite": 65.0}}
            ],
            "edges": [
                {"source": "run-1", "target": "spec-1", "relation": "ran_with"}
            ],
        }
        kg = ExperimentKnowledgeGraph.from_dict(data)

        assert "run-1" in kg.nodes
        assert kg.nodes["run-1"].properties["composite"] == 65.0
        assert len(kg.edges) == 1


class TestIntegration:
    def test_load_from_artifacts_empty_dir(self) -> None:
        kg = ExperimentKnowledgeGraph()
        # Should not raise
        kg.load_from_artifacts("/nonexistent/path")
        assert len(kg.nodes) == 0


class TestFormatSummary:
    def test_format_summary(self) -> None:
        kg = ExperimentKnowledgeGraph()
        # Add some data
        node = GraphNode(id="test", type="run", properties={"track_id": "T1", "stage": 1})
        kg.nodes["test"] = node
        kg._track_stages["T1"] = {1, 2}

        summary = format_knowledge_graph_summary(kg)

        assert "Knowledge Graph Summary" in summary
        assert "Total Nodes" in summary
        assert "T1" in summary


class TestEdgeCases:
    def test_load_run_without_metadata(self) -> None:
        # Create run without metadata
        run = RunResult(
            run_id="test-run",
            spec_id="spec-test",
            commit_sha="abc",
            seed=1,
            train_cost=10.0,
            infer_cost=2.0,
            latency_p50=50.0,
            latency_p95=60.0,
            energy_kwh=0.1,
            metric_values={"composite": 60.0},
            failure_flags=[],
            track_id="T1",
            stage=1,
            model_variant="baseline",
            benchmark_scores={},
            metadata={},  # Empty metadata
        )

        kg = ExperimentKnowledgeGraph()
        kg.load_run(run)

        # Should handle empty metadata gracefully
        assert "test-run" in kg.nodes

    def test_multiple_runs_same_track_stage(self) -> None:
        kg = ExperimentKnowledgeGraph()

        for i in range(5):
            run = _make_run(f"run-{i}", "T3", stage=2, composite=60.0 + i)
            kg.load_run(run)

        history = kg.get_track_stage_history("T3")
        assert len(history) == 5

