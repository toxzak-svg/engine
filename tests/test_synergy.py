"""Tests for exp.synergy — cross-track synergy matrix."""

from __future__ import annotations

from exp.models import RunResult
from exp.synergy import (
    SynergyScore,
    build_synergy_matrix,
    compute_synergy_score,
    format_synergy_report,
    get_top_synergy_pairs,
    select_portfolio_diverse_tracks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    run_id: str,
    track_id: str,
    variant: str,
    composite: float,
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
        model_variant=variant,
        benchmark_scores={},
    )


# ---------------------------------------------------------------------------
# compute_synergy_score
# ---------------------------------------------------------------------------

class TestComputeSynergyScore:
    def test_raises_on_empty_runs(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 65.0)]
        try:
            compute_synergy_score([], runs_a, 60.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_positive_synergy_when_both_beat_baseline(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 65.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 64.0)]
        score = compute_synergy_score(runs_a, runs_b, baseline_composite=60.0)
        # Both beat baseline → combined should be > better individual
        assert score.synergy >= 0.0
        assert score.composite_a == 65.0
        assert score.composite_b == 64.0

    def test_zero_synergy_when_weaker_below_baseline(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 65.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 58.0)]  # below baseline
        score = compute_synergy_score(runs_a, runs_b, baseline_composite=60.0)
        # Weaker delta is negative → combined = better_composite + 0
        assert score.synergy == 0.0

    def test_super_additive_flag(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 66.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 65.0)]
        score = compute_synergy_score(runs_a, runs_b, baseline_composite=60.0)
        # synergy = 0.70 * 5.0 = 3.5 > SYNERGY_THRESHOLD
        assert score.super_additive is True

    def test_sub_additive_flag_not_set_for_positive_synergy(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 65.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 64.0)]
        score = compute_synergy_score(runs_a, runs_b, baseline_composite=60.0)
        assert score.sub_additive is False

    def test_track_ids_captured(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 65.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 64.0)]
        score = compute_synergy_score(runs_a, runs_b, baseline_composite=60.0)
        assert score.track_a == "T3"
        assert score.track_b == "T4"
        assert score.variant_a == "T3-E2"
        assert score.variant_b == "T4-E3"

    def test_confidence_between_0_and_1(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 65.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 64.0)]
        score = compute_synergy_score(runs_a, runs_b, baseline_composite=60.0)
        assert 0.0 <= score.confidence <= 1.0

    def test_to_dict_has_required_keys(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 65.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 64.0)]
        score = compute_synergy_score(runs_a, runs_b, baseline_composite=60.0)
        d = score.to_dict()
        for key in ["track_a", "track_b", "synergy", "super_additive", "confidence"]:
            assert key in d


# ---------------------------------------------------------------------------
# build_synergy_matrix
# ---------------------------------------------------------------------------

class TestBuildSynergyMatrix:
    def _make_variant_runs(self) -> dict[str, list[RunResult]]:
        return {
            "T3-E2": [_make_run("r1", "T3", "T3-E2", 65.0)],
            "T4-E3": [_make_run("r2", "T4", "T4-E3", 64.0)],
            "T1-E2": [_make_run("r3", "T1", "T1-E2", 63.0)],
        }

    def test_returns_list_of_synergy_scores(self) -> None:
        variant_runs = self._make_variant_runs()
        matrix = build_synergy_matrix(variant_runs, baseline_composite=60.0)
        assert isinstance(matrix, list)
        assert all(isinstance(s, SynergyScore) for s in matrix)

    def test_skips_same_track_pairs(self) -> None:
        variant_runs = {
            "T3-E1": [_make_run("r1", "T3", "T3-E1", 63.0)],
            "T3-E2": [_make_run("r2", "T3", "T3-E2", 65.0)],
        }
        matrix = build_synergy_matrix(variant_runs, baseline_composite=60.0)
        # Same track → should be skipped
        assert len(matrix) == 0

    def test_sorted_by_synergy_descending(self) -> None:
        variant_runs = self._make_variant_runs()
        matrix = build_synergy_matrix(variant_runs, baseline_composite=60.0)
        if len(matrix) >= 2:
            assert matrix[0].synergy >= matrix[1].synergy

    def test_empty_variant_runs(self) -> None:
        matrix = build_synergy_matrix({}, baseline_composite=60.0)
        assert matrix == []

    def test_cross_track_pairs_generated(self) -> None:
        variant_runs = self._make_variant_runs()
        matrix = build_synergy_matrix(variant_runs, baseline_composite=60.0)
        # 3 tracks → 3 cross-track pairs
        assert len(matrix) == 3


# ---------------------------------------------------------------------------
# get_top_synergy_pairs
# ---------------------------------------------------------------------------

class TestGetTopSynergyPairs:
    def _make_matrix(self) -> list[SynergyScore]:
        runs_a = [_make_run("r1", "T3", "T3-E2", 66.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 65.0)]
        runs_c = [_make_run("r3", "T1", "T1-E2", 63.0)]
        return [
            compute_synergy_score(runs_a, runs_b, 60.0),
            compute_synergy_score(runs_a, runs_c, 60.0),
            compute_synergy_score(runs_b, runs_c, 60.0),
        ]

    def test_returns_top_n(self) -> None:
        matrix = self._make_matrix()
        top = get_top_synergy_pairs(matrix, top_n=2)
        assert len(top) <= 2

    def test_super_additive_only_filter(self) -> None:
        matrix = self._make_matrix()
        top = get_top_synergy_pairs(matrix, super_additive_only=True)
        assert all(s.super_additive for s in top)


# ---------------------------------------------------------------------------
# format_synergy_report
# ---------------------------------------------------------------------------

class TestFormatSynergyReport:
    def test_returns_markdown_string(self) -> None:
        runs_a = [_make_run("r1", "T3", "T3-E2", 66.0)]
        runs_b = [_make_run("r2", "T4", "T4-E3", 65.0)]
        matrix = [compute_synergy_score(runs_a, runs_b, 60.0)]
        report = format_synergy_report(matrix)
        assert "## Cross-Track Synergy Matrix" in report
        assert "T3-E2" in report
        assert "T4-E3" in report

    def test_empty_matrix_no_crash(self) -> None:
        report = format_synergy_report([])
        assert "## Cross-Track Synergy Matrix" in report


# ---------------------------------------------------------------------------
# select_portfolio_diverse_tracks
# ---------------------------------------------------------------------------

class TestSelectPortfolioDiverseTracks:
    def test_single_track_unchanged(self) -> None:
        result = select_portfolio_diverse_tracks([], ["T3"])
        assert result == ["T3"]

    def test_reorders_by_synergy(self) -> None:
        runs_t3 = [_make_run("r1", "T3", "T3-E2", 66.0)]
        runs_t4 = [_make_run("r2", "T4", "T4-E3", 65.0)]
        runs_t1 = [_make_run("r3", "T1", "T1-E2", 63.0)]
        matrix = [
            compute_synergy_score(runs_t3, runs_t4, 60.0),
            compute_synergy_score(runs_t3, runs_t1, 60.0),
            compute_synergy_score(runs_t4, runs_t1, 60.0),
        ]
        result = select_portfolio_diverse_tracks(matrix, ["T1", "T3", "T4"])
        # Result should be a permutation of the input
        assert sorted(result) == sorted(["T1", "T3", "T4"])

    def test_empty_matrix_returns_original_order(self) -> None:
        result = select_portfolio_diverse_tracks([], ["T3", "T4", "T1"])
        assert result == ["T3", "T4", "T1"]
