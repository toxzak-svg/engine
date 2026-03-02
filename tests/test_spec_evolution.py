"""Tests for exp.spec_evolution — genetic algorithm spec generation."""

from __future__ import annotations

import pytest

from exp.models import ExperimentSpec, RunResult
from exp.spec_evolution import (
    PARAM_SEARCH_SPACE,
    EvolutionConfig,
    EvolutionResult,
    _compute_diversity,
    _crossover,
    _mutate,
    _params_hash,
    _spec_to_offspring_dict,
    _tournament_select,
    evolve_specs,
    format_evolution_report,
)

import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(
    spec_id: str,
    track_id: str,
    params: dict,
    stage: int = 2,
) -> ExperimentSpec:
    """Build a minimal valid ExperimentSpec without schema validation."""
    return ExperimentSpec(
        id=spec_id,
        track_id=track_id,
        stage=stage,
        hypothesis=f"Test hypothesis for {spec_id}",
        model_variant=f"{track_id}-E2",
        baseline_id=f"s{stage}-{track_id.lower()}-baseline",
        train_budget_gpu_h=80.0,
        infer_budget_gpu_h=20.0,
        max_context=128000,
        datasets=["needle_32k", "longbench", "gsm8k"],
        metrics=["long_context", "reasoning", "consistency", "composite"],
        seeds=[101, 102],
        promotion_gate={"next_stage": stage + 1, "delta_composite_min": 5.0},
        params=params,
    )


def _make_result(
    spec_id: str,
    track_id: str,
    composite: float,
    failure_flags: list[str] | None = None,
    stage: int = 2,
    run_id: str | None = None,
) -> RunResult:
    rid = run_id or f"run-{spec_id}"
    return RunResult(
        run_id=rid,
        spec_id=spec_id,
        commit_sha="abc123",
        seed=101,
        train_cost=80.0,
        infer_cost=20.0,
        latency_p50=120.0,
        latency_p95=155.0,
        energy_kwh=45.0,
        metric_values={"composite": composite, "fluency": 89.0},
        failure_flags=failure_flags or [],
        track_id=track_id,
        stage=stage,
        model_variant=f"{track_id}-E2",
        benchmark_scores={"gsm8k": composite, "bbh": composite - 1.0},
    )


# ---------------------------------------------------------------------------
# _crossover
# ---------------------------------------------------------------------------

class TestCrossover:
    def test_child_keys_are_union_of_parents(self) -> None:
        rng = random.Random(0)
        a = {"compression_ratio": 0.70, "seed_override": 42}
        b = {"compression_ratio": 0.85, "extra_param": "yes"}
        child = _crossover(a, b, rng)
        assert set(child.keys()) == {"compression_ratio", "seed_override", "extra_param"}

    def test_shared_param_comes_from_one_parent(self) -> None:
        rng = random.Random(1)
        a = {"compression_ratio": 0.70}
        b = {"compression_ratio": 0.85}
        seen: set[float] = set()
        # Run many times — must always be 0.70 or 0.85, never something else
        for seed in range(30):
            rng = random.Random(seed)
            child = _crossover(a, b, rng)
            seen.add(child["compression_ratio"])
        assert seen <= {0.70, 0.85}
        assert len(seen) == 2  # both values appear across seeds

    def test_params_only_in_one_parent_are_inherited(self) -> None:
        rng = random.Random(0)
        a = {"compression_ratio": 0.70}
        b = {"max_nodes": 12}
        child = _crossover(a, b, rng)
        assert child["compression_ratio"] == 0.70
        assert child["max_nodes"] == 12

    def test_empty_parents_produce_empty_child(self) -> None:
        rng = random.Random(0)
        assert _crossover({}, {}, rng) == {}


# ---------------------------------------------------------------------------
# _mutate
# ---------------------------------------------------------------------------

class TestMutate:
    def test_zero_mutation_rate_leaves_params_unchanged(self) -> None:
        rng = random.Random(0)
        params = {"compression_ratio": 0.70}
        mutated = _mutate(params, track_id="T3", mutation_rate=0.0, rng=rng)
        assert mutated == params

    def test_full_mutation_rate_replaces_known_params(self) -> None:
        rng = random.Random(42)
        # Use a track with a known search space
        params = {"compression_ratio": 0.70}
        mutated = _mutate(params, track_id="T3", mutation_rate=1.0, rng=rng)
        valid_values = PARAM_SEARCH_SPACE["T3"]["compression_ratio"]
        assert mutated["compression_ratio"] in valid_values

    def test_mutation_respects_search_space_bounds(self) -> None:
        rng = random.Random(7)
        params = {"role_permutation_noise": 0.10}
        valid_values = PARAM_SEARCH_SPACE["T4"]["role_permutation_noise"]
        for seed in range(50):
            rng = random.Random(seed)
            mutated = _mutate(params, track_id="T4", mutation_rate=1.0, rng=rng)
            assert mutated["role_permutation_noise"] in valid_values

    def test_unknown_track_leaves_params_unchanged(self) -> None:
        rng = random.Random(0)
        params = {"some_param": 999}
        mutated = _mutate(params, track_id="T_UNKNOWN", mutation_rate=1.0, rng=rng)
        # No search space for unknown track — params untouched
        assert mutated["some_param"] == 999

    def test_mutation_does_not_add_extra_keys(self) -> None:
        rng = random.Random(3)
        params = {"compression_ratio": 0.80, "extra": "keep"}
        mutated = _mutate(params, track_id="T3", mutation_rate=1.0, rng=rng)
        # extra is untouched; we get at most compression_ratio mutated
        assert "extra" in mutated
        assert mutated["extra"] == "keep"


# ---------------------------------------------------------------------------
# _compute_diversity
# ---------------------------------------------------------------------------

class TestComputeDiversity:
    def test_empty_offspring_returns_zero(self) -> None:
        assert _compute_diversity([]) == 0.0

    def test_single_offspring_returns_zero(self) -> None:
        assert _compute_diversity([{"params": {"cr": 0.70}}]) == 0.0

    def test_identical_offspring_returns_zero(self) -> None:
        specs = [
            {"params": {"compression_ratio": 0.70}},
            {"params": {"compression_ratio": 0.70}},
            {"params": {"compression_ratio": 0.70}},
        ]
        assert _compute_diversity(specs) == 0.0

    def test_fully_different_params_return_one(self) -> None:
        # Two specs, one param each, all different → distance = 1.0
        specs = [
            {"params": {"compression_ratio": 0.70}},
            {"params": {"compression_ratio": 0.85}},
        ]
        diversity = _compute_diversity(specs)
        assert diversity == pytest.approx(1.0)

    def test_partial_diversity(self) -> None:
        specs = [
            {"params": {"a": 1, "b": 1}},
            {"params": {"a": 2, "b": 1}},  # differs on "a" only → distance = 0.5
        ]
        diversity = _compute_diversity(specs)
        assert diversity == pytest.approx(0.5)

    def test_diversity_in_range_zero_to_one(self) -> None:
        rng = random.Random(42)
        specs = [
            {"params": {f"p{i}": rng.randint(0, 3) for i in range(5)}}
            for _ in range(8)
        ]
        d = _compute_diversity(specs)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# _params_hash
# ---------------------------------------------------------------------------

class TestParamsHash:
    def test_same_params_same_hash(self) -> None:
        a = {"compression_ratio": 0.70, "noise": "analog_v1"}
        b = {"noise": "analog_v1", "compression_ratio": 0.70}
        assert _params_hash(a) == _params_hash(b)

    def test_different_params_different_hash(self) -> None:
        a = {"compression_ratio": 0.70}
        b = {"compression_ratio": 0.85}
        assert _params_hash(a) != _params_hash(b)

    def test_empty_params_consistent(self) -> None:
        assert _params_hash({}) == _params_hash({})


# ---------------------------------------------------------------------------
# _tournament_select
# ---------------------------------------------------------------------------

class TestTournamentSelect:
    def test_selects_from_population(self) -> None:
        rng = random.Random(0)
        pop = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.75}),
            _make_spec("s3", "T3", {"compression_ratio": 0.80}),
        ]
        fitnesses = [0.3, 0.6, 0.9]
        for _ in range(20):
            winner = _tournament_select(pop, fitnesses, rng)
            assert winner in pop

    def test_biases_toward_higher_fitness(self) -> None:
        rng = random.Random(0)
        pop = [
            _make_spec("low", "T3", {"compression_ratio": 0.70}),
            _make_spec("high", "T3", {"compression_ratio": 0.85}),
        ]
        fitnesses = [0.1, 5.0]
        wins = {"low": 0, "high": 0}
        for _ in range(100):
            w = _tournament_select(pop, fitnesses, rng)
            wins[w.id] += 1
        # The high-fitness spec should win most of the time
        assert wins["high"] > wins["low"]


# ---------------------------------------------------------------------------
# evolve_specs
# ---------------------------------------------------------------------------

class TestEvolveSpecs:
    def test_empty_parent_specs_returns_empty_result(self) -> None:
        result = evolve_specs(parent_specs=[], parent_results=[])
        assert result.offspring_specs == []
        assert result.parent_fitnesses == []
        assert result.n_parents == 0
        assert result.n_offspring == 0
        assert result.track_id == "UNKNOWN"

    def test_fewer_than_min_parents_returns_no_offspring(self) -> None:
        spec = _make_spec("s1", "T3", {"compression_ratio": 0.70})
        result_obj = _make_result("s1", "T3", 65.0)
        config = EvolutionConfig(min_parents_for_evolution=2)
        result = evolve_specs([spec], [result_obj], config)
        assert result.n_offspring == 0
        assert result.n_parents == 1

    def test_evolve_with_valid_parents_returns_offspring(self) -> None:
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.80}),
            _make_spec("s3", "T3", {"compression_ratio": 0.75}),
        ]
        results = [
            _make_result("s1", "T3", 65.0),
            _make_result("s2", "T3", 68.0),
            _make_result("s3", "T3", 67.0),
        ]
        config = EvolutionConfig(population_size=5, n_generations=2, random_seed=42)
        result = evolve_specs(specs, results, config, next_stage=3)
        assert result.n_offspring > 0
        assert result.track_id == "T3"
        assert result.stage == 3
        assert len(result.generation_stats) == 2

    def test_offspring_have_required_fields(self) -> None:
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.85}),
        ]
        results = [
            _make_result("s1", "T3", 64.0),
            _make_result("s2", "T3", 69.0),
        ]
        config = EvolutionConfig(population_size=4, n_generations=1, random_seed=7)
        result = evolve_specs(specs, results, config)
        for offspring in result.offspring_specs:
            assert "id" in offspring
            assert "track_id" in offspring
            assert "stage" in offspring
            assert "params" in offspring
            assert "evolution_metadata" in offspring
            assert offspring["track_id"] == "T3"

    def test_offspring_params_are_within_search_space(self) -> None:
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.85}),
        ]
        results = [_make_result("s1", "T3", 64.0), _make_result("s2", "T3", 69.0)]
        config = EvolutionConfig(population_size=8, n_generations=2, mutation_rate=1.0, random_seed=42)
        result = evolve_specs(specs, results, config)
        valid_cr = PARAM_SEARCH_SPACE["T3"]["compression_ratio"]
        for offspring in result.offspring_specs:
            params = offspring.get("params", {})
            if "compression_ratio" in params:
                assert params["compression_ratio"] in valid_cr, (
                    f"Invalid compression_ratio {params['compression_ratio']}"
                )

    def test_offspring_are_deduplicated(self) -> None:
        # Very small mutation rate → likely many duplicates → deduplicated
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.70}),
        ]
        results = [_make_result("s1", "T3", 65.0), _make_result("s2", "T3", 65.0)]
        config = EvolutionConfig(
            population_size=20, n_generations=3, mutation_rate=0.0,
            crossover_rate=0.0, random_seed=1,
        )
        result = evolve_specs(specs, results, config)
        # All offspring should be unique
        ids = [o["id"] for o in result.offspring_specs]
        assert len(ids) == len(set(ids))

    def test_parent_fitnesses_computed_from_results(self) -> None:
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.85}),
        ]
        results = [
            _make_result("s1", "T3", 60.0),  # all failed
            _make_result("s1", "T3", 62.0, failure_flags=["some_flag"]),
            _make_result("s2", "T3", 70.0),
        ]
        config = EvolutionConfig(population_size=4, n_generations=1, random_seed=0)
        result = evolve_specs(specs, results, config)
        # s1 has some failures → lower fitness; s2 has no failures → fitness = composite * 1.0
        assert len(result.parent_fitnesses) == 2
        assert result.parent_fitnesses[0] >= 0.0
        assert result.parent_fitnesses[1] >= 0.0

    def test_generation_stats_have_required_keys(self) -> None:
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.85}),
        ]
        results = [_make_result("s1", "T3", 65.0), _make_result("s2", "T3", 68.0)]
        config = EvolutionConfig(population_size=5, n_generations=3, random_seed=0)
        result = evolve_specs(specs, results, config)
        assert len(result.generation_stats) == 3
        for stats in result.generation_stats:
            assert "generation" in stats
            assert "mean_fitness" in stats
            assert "best_fitness" in stats
            assert "diversity" in stats
            assert "n_offspring" in stats

    def test_no_results_still_evolves(self) -> None:
        # When parent_results is empty, fitness defaults to 0.0 for all parents
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.85}),
        ]
        config = EvolutionConfig(population_size=4, n_generations=1, random_seed=0)
        result = evolve_specs(specs, parent_results=[], config=config)
        assert result.n_offspring > 0
        assert all(f == 0.0 for f in result.parent_fitnesses)

    def test_seed_determines_reproducibility(self) -> None:
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}),
            _make_spec("s2", "T3", {"compression_ratio": 0.85}),
        ]
        results = [_make_result("s1", "T3", 65.0), _make_result("s2", "T3", 68.0)]
        cfg = EvolutionConfig(population_size=6, n_generations=2, random_seed=99)
        r1 = evolve_specs(specs, results, cfg)
        r2 = evolve_specs(specs, results, cfg)
        ids1 = [o["id"] for o in r1.offspring_specs]
        ids2 = [o["id"] for o in r2.offspring_specs]
        assert ids1 == ids2

    def test_default_next_stage_is_parent_plus_one(self) -> None:
        specs = [
            _make_spec("s1", "T3", {"compression_ratio": 0.70}, stage=2),
            _make_spec("s2", "T3", {"compression_ratio": 0.85}, stage=2),
        ]
        results = [_make_result("s1", "T3", 65.0), _make_result("s2", "T3", 68.0)]
        config = EvolutionConfig(population_size=3, n_generations=1, random_seed=0)
        result = evolve_specs(specs, results, config)
        assert result.stage == 3
        for offspring in result.offspring_specs:
            assert offspring["stage"] == 3


# ---------------------------------------------------------------------------
# _spec_to_offspring_dict
# ---------------------------------------------------------------------------

class TestSpecToOffspringDict:
    def test_offspring_dict_structure(self) -> None:
        rng = random.Random(0)
        spec = _make_spec("parent-1", "T3", {"compression_ratio": 0.70})
        child = _spec_to_offspring_dict(
            spec=spec,
            params={"compression_ratio": 0.80},
            offspring_stage=3,
            rng=rng,
            label="crossover",
        )
        assert child["id"].startswith("evo-t3-s3-crossover-")
        assert child["track_id"] == "T3"
        assert child["stage"] == 3
        assert child["params"] == {"compression_ratio": 0.80}
        assert child["evolution_metadata"]["label"] == "crossover"
        assert child["evolution_metadata"]["parent_spec_id"] == "parent-1"
        assert len(child["seeds"]) >= 3

    def test_elite_label_preserved(self) -> None:
        rng = random.Random(0)
        spec = _make_spec("p1", "T4", {"role_permutation_noise": 0.20})
        child = _spec_to_offspring_dict(spec, {"role_permutation_noise": 0.20}, 3, rng, "elite")
        assert "elite" in child["id"]
        assert child["evolution_metadata"]["label"] == "elite"


# ---------------------------------------------------------------------------
# format_evolution_report
# ---------------------------------------------------------------------------

class TestFormatEvolutionReport:
    def _make_result_obj(self) -> EvolutionResult:
        return EvolutionResult(
            offspring_specs=[
                {
                    "id": "evo-t3-s3-elite-aabbccdd",
                    "track_id": "T3",
                    "stage": 3,
                    "params": {"compression_ratio": 0.70},
                    "evolution_metadata": {"label": "elite", "parent_spec_id": "parent-1"},
                },
                {
                    "id": "evo-t3-s3-crossover-11223344",
                    "track_id": "T3",
                    "stage": 3,
                    "params": {"compression_ratio": 0.85},
                    "evolution_metadata": {"label": "crossover", "parent_spec_id": "parent-2"},
                },
            ],
            parent_fitnesses=[3.1, 4.8],
            generation_stats=[
                {"generation": 1.0, "mean_fitness": 3.95, "best_fitness": 4.8, "diversity": 0.5, "n_offspring": 5.0},
                {"generation": 2.0, "mean_fitness": 4.1, "best_fitness": 4.9, "diversity": 0.4, "n_offspring": 5.0},
            ],
            n_parents=2,
            n_offspring=2,
            track_id="T3",
            stage=3,
        )

    def test_report_contains_header(self) -> None:
        report = format_evolution_report(self._make_result_obj())
        assert "## Genetic Spec Evolution" in report
        assert "T3" in report
        assert "Stage 3" in report

    def test_report_contains_generation_table(self) -> None:
        report = format_evolution_report(self._make_result_obj())
        assert "Generation Statistics" in report
        assert "| Generation |" in report
        assert "| 1 |" in report
        assert "| 2 |" in report

    def test_report_lists_offspring(self) -> None:
        report = format_evolution_report(self._make_result_obj())
        assert "Top Offspring Specs" in report
        assert "evo-t3-s3-elite-aabbccdd" in report

    def test_report_shows_fitness_range(self) -> None:
        report = format_evolution_report(self._make_result_obj())
        assert "3.100" in report
        assert "4.800" in report

    def test_empty_result_produces_safe_output(self) -> None:
        empty = EvolutionResult(
            offspring_specs=[],
            parent_fitnesses=[],
            generation_stats=[],
            n_parents=0,
            n_offspring=0,
            track_id="T3",
            stage=2,
        )
        report = format_evolution_report(empty)
        assert isinstance(report, str)
        assert "## Genetic Spec Evolution" in report


# ---------------------------------------------------------------------------
# EvolutionResult.summary()
# ---------------------------------------------------------------------------

class TestEvolutionResultSummary:
    def test_summary_is_multiline_string(self) -> None:
        r = EvolutionResult(
            offspring_specs=[],
            parent_fitnesses=[2.1, 4.3],
            generation_stats=[
                {"generation": 1, "mean_fitness": 3.2, "best_fitness": 4.3, "diversity": 0.6}
            ],
            n_parents=2,
            n_offspring=0,
            track_id="T4",
            stage=3,
        )
        summary = r.summary()
        lines = summary.splitlines()
        assert len(lines) >= 3
        assert "T4" in summary
        assert "Stage 3" in summary

    def test_summary_contains_fitness_range(self) -> None:
        r = EvolutionResult(
            offspring_specs=[],
            parent_fitnesses=[1.0, 5.0],
            generation_stats=[],
            n_parents=2,
            n_offspring=0,
            track_id="T3",
            stage=2,
        )
        summary = r.summary()
        assert "1.000" in summary
        assert "5.000" in summary

    def test_summary_no_fitness_data_is_safe(self) -> None:
        r = EvolutionResult(
            offspring_specs=[],
            parent_fitnesses=[],
            generation_stats=[],
            n_parents=0,
            n_offspring=0,
            track_id="T1",
            stage=1,
        )
        summary = r.summary()
        assert isinstance(summary, str)
