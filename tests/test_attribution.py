"""Tests for exp.attribution — Shapley param attribution engine."""

from __future__ import annotations

from exp.attribution import (
    AttributionResult,
    attribute_all_tracks,
    format_attribution_report,
    shapley_param_attribution,
)
from exp.models import RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    run_id: str,
    track_id: str,
    composite: float,
    params: dict[str, object],
    stage: int = 1,
    seed: int = 101,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        spec_id=f"spec-{run_id}",
        commit_sha="abc123",
        seed=seed,
        train_cost=100.0,
        infer_cost=20.0,
        latency_p50=120.0,
        latency_p95=153.6,
        energy_kwh=45.6,
        metric_values={"composite": composite, "fluency": 90.0},
        failure_flags=[],
        track_id=track_id,
        stage=stage,
        model_variant="T3-E2",
        benchmark_scores={},
        metadata={"params": params},
    )


# ---------------------------------------------------------------------------
# shapley_param_attribution
# ---------------------------------------------------------------------------

class TestShapleyParamAttribution:
    def test_empty_runs_returns_empty_result(self) -> None:
        result = shapley_param_attribution("T3", [], baseline_composite=60.0)
        assert result.n_runs == 0
        assert result.attributions == []
        assert result.track_id == "T3"

    def test_no_params_returns_empty_attributions(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={}),
            _make_run("r2", "T3", 64.0, params={}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        assert result.attributions == []
        assert result.n_runs == 2

    def test_single_param_two_values_attributed(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
            _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85}),
            _make_run("r3", "T3", 64.5, params={"compression_ratio": 0.70}),
            _make_run("r4", "T3", 67.5, params={"compression_ratio": 0.85}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        assert len(result.attributions) == 1
        attr = result.attributions[0]
        assert attr.param_name == "compression_ratio"
        # Higher compression_ratio → higher composite → positive direction
        assert attr.direction == "positive"
        assert attr.shapley_value > 0.0

    def test_multiple_params_all_attributed(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70, "max_context": 64000}),
            _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85, "max_context": 128000}),
            _make_run("r3", "T3", 66.0, params={"compression_ratio": 0.70, "max_context": 128000}),
            _make_run("r4", "T3", 67.0, params={"compression_ratio": 0.85, "max_context": 64000}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        param_names = [a.param_name for a in result.attributions]
        assert "compression_ratio" in param_names
        assert "max_context" in param_names

    def test_sorted_by_absolute_shapley_descending(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70, "max_context": 64000}),
            _make_run("r2", "T3", 70.0, params={"compression_ratio": 0.85, "max_context": 128000}),
            _make_run("r3", "T3", 65.5, params={"compression_ratio": 0.70, "max_context": 128000}),
            _make_run("r4", "T3", 69.5, params={"compression_ratio": 0.85, "max_context": 64000}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        if len(result.attributions) >= 2:
            assert abs(result.attributions[0].shapley_value) >= abs(result.attributions[1].shapley_value)

    def test_total_delta_computed(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
            _make_run("r2", "T3", 67.0, params={"compression_ratio": 0.85}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        assert abs(result.total_delta - 6.0) < 0.01  # mean(65, 67) - 60 = 6.0

    def test_track_id_and_stage_preserved(self) -> None:
        runs = [_make_run("r1", "T4", 65.0, params={"role_permutation_noise": 0.2}, stage=2)]
        result = shapley_param_attribution("T4", runs, baseline_composite=60.0)
        assert result.track_id == "T4"
        assert result.stage == 2

    def test_to_dict_serializable(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
            _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        d = result.to_dict()
        assert "track_id" in d
        assert "attributions" in d
        assert isinstance(d["attributions"], list)

    def test_confidence_between_0_and_1(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
            _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        for attr in result.attributions:
            assert 0.0 <= attr.confidence <= 1.0

    def test_best_worst_values_identified(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
            _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        attr = result.attributions[0]
        assert attr.best_value == "0.85"   # higher composite
        assert attr.worst_value == "0.7"   # lower composite (str(0.70) == '0.7')


# ---------------------------------------------------------------------------
# format_attribution_report
# ---------------------------------------------------------------------------

class TestFormatAttributionReport:
    def test_returns_markdown_string(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
            _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        report = format_attribution_report(result)
        assert "### Param Attribution" in report
        assert "compression_ratio" in report

    def test_empty_attributions_no_crash(self) -> None:
        result = AttributionResult(
            track_id="T3",
            stage=1,
            baseline_composite=60.0,
            attributions=[],
            total_explained=0.0,
            total_delta=5.0,
            explanation_ratio=0.0,
            n_runs=0,
        )
        report = format_attribution_report(result)
        assert "### Param Attribution" in report
        assert "No param variation" in report

    def test_key_driver_narrative_present(self) -> None:
        runs = [
            _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
            _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85}),
        ]
        result = shapley_param_attribution("T3", runs, baseline_composite=60.0)
        report = format_attribution_report(result)
        assert "Key driver" in report


# ---------------------------------------------------------------------------
# attribute_all_tracks
# ---------------------------------------------------------------------------

class TestAttributeAllTracks:
    def test_returns_dict_per_track(self) -> None:
        runs_by_track = {
            "T3": [
                _make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70}),
                _make_run("r2", "T3", 68.0, params={"compression_ratio": 0.85}),
            ],
            "T4": [
                _make_run("r3", "T4", 64.0, params={"role_permutation_noise": 0.2}),
                _make_run("r4", "T4", 66.0, params={"role_permutation_noise": 0.1}),
            ],
        }
        baseline_composites = {"T3": 60.0, "T4": 60.0}
        results = attribute_all_tracks(runs_by_track, baseline_composites)
        assert "T3" in results
        assert "T4" in results
        assert isinstance(results["T3"], AttributionResult)
        assert isinstance(results["T4"], AttributionResult)

    def test_missing_baseline_defaults_to_zero(self) -> None:
        runs_by_track = {
            "T3": [_make_run("r1", "T3", 65.0, params={"compression_ratio": 0.70})],
        }
        results = attribute_all_tracks(runs_by_track, baseline_composites={})
        assert "T3" in results
        assert results["T3"].baseline_composite == 0.0
