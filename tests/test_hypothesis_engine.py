"""Tests for exp.hypothesis_engine — LLM Hypothesis Generation Engine."""

from __future__ import annotations

from exp.hypothesis_engine import (
    HypothesisEngine,
    Hypothesis,
    HypothesisGenerationResult,
    generate_hypothesis_report,
)


# ---------------------------------------------------------------------------
# Test Data Helpers
# ---------------------------------------------------------------------------

def _make_mock_runs() -> list[dict]:
    return [
        {
            "metric_values": {"composite": 65.0},
            "track_id": "T3",
            "stage": 2,
        },
        {
            "metric_values": {"composite": 62.0},
            "track_id": "T3",
            "stage": 2,
        },
    ]


def _make_mock_reports() -> list[dict]:
    return [
        {
            "pass_fail": {"overall_pass": True},
            "track_id": "T3",
        },
        {
            "pass_fail": {"overall_pass": False},
            "track_id": "T3",
        },
    ]


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    def test_create_hypothesis(self) -> None:
        hypo = Hypothesis(
            id="hypo-1",
            track_id="T3",
            description="Test hypothesis",
            mechanism="Test mechanism",
            expected_effect={"long_context": 2.0, "reasoning": 1.0},
            testable_predictions=["prediction 1"],
            novelty_score=0.8,
            risk_level="medium",
        )
        assert hypo.id == "hypo-1"
        assert hypo.track_id == "T3"
        assert hypo.novelty_score == 0.8

    def test_hypothesis_default_related_tracks(self) -> None:
        hypo = Hypothesis(
            id="hypo-1",
            track_id="T3",
            description="Test",
            mechanism="Test",
            expected_effect={},
            testable_predictions=[],
            novelty_score=0.5,
            risk_level="low",
        )
        assert hypo.related_tracks == []


# ---------------------------------------------------------------------------
# HypothesisEngine
# ---------------------------------------------------------------------------

class TestHypothesisEngineBasics:
    def test_init_without_llm(self) -> None:
        engine = HypothesisEngine()
        assert engine.llm_client is None
        assert engine._hypothesis_cache == []

    def test_init_with_llm(self) -> None:
        class MockLLM:
            def chat(self, system: str, user: str) -> str:
                return '{"hypotheses": []}'

        engine = HypothesisEngine(llm_client=MockLLM())
        assert engine.llm_client is not None


class TestHypothesisGeneration:
    def test_generate_template_based_t3(self) -> None:
        engine = HypothesisEngine()
        context = {"runs": _make_mock_runs(), "reports": _make_mock_reports()}

        result = engine.generate_hypotheses("T3", context, num_hypotheses=3)

        assert isinstance(result, HypothesisGenerationResult)
        assert len(result.hypotheses) == 3
        assert result.generation_method == "template"
        assert all(h.track_id == "T3" for h in result.hypotheses)

    def test_generate_template_based_unknown_track(self) -> None:
        engine = HypothesisEngine()
        context = {}

        result = engine.generate_hypotheses("UNKNOWN", context, num_hypotheses=3)

        # Should return empty for unknown track
        assert len(result.hypotheses) == 0

    def test_generate_stores_in_cache(self) -> None:
        engine = HypothesisEngine()
        context = {"runs": _make_mock_runs(), "reports": _make_mock_reports()}

        engine.generate_hypotheses("T3", context, num_hypotheses=2)

        assert len(engine._hypothesis_cache) >= 2


class TestHypothesisRanking:
    def test_rank_hypotheses(self) -> None:
        engine = HypothesisEngine()
        hypotheses = [
            Hypothesis(
                id="h1",
                track_id="T3",
                description="Low novelty high impact",
                mechanism="",
                expected_effect={"long_context": 5.0, "reasoning": 5.0},
                testable_predictions=[],
                novelty_score=0.2,
                risk_level="low",
            ),
            Hypothesis(
                id="h2",
                track_id="T3",
                description="High novelty low impact",
                mechanism="",
                expected_effect={"long_context": 1.0, "reasoning": 1.0},
                testable_predictions=[],
                novelty_score=0.9,
                risk_level="low",
            ),
        ]

        ranked = engine.rank_hypotheses(hypotheses)

        # Both should be returned
        assert len(ranked) == 2

    def test_rank_with_default_weights(self) -> None:
        engine = HypothesisEngine()
        hypo = Hypothesis(
            id="h1",
            track_id="T3",
            description="Test",
            mechanism="",
            expected_effect={"long_context": 3.0},
            testable_predictions=[],
            novelty_score=0.5,
            risk_level="medium",
        )

        ranked = engine.rank_hypotheses([hypo])
        assert len(ranked) == 1


class TestSynergyHypotheses:
    def test_find_synergy_t3_t4(self) -> None:
        engine = HypothesisEngine()
        synergies = engine.find_synergy_hypotheses(["T3", "T4"])

        # Should find T3+T4 synergy
        assert any(
            set(s["tracks"]) == {"T3", "T4"}
            for s in synergies
        )

    def test_find_synergy_empty(self) -> None:
        engine = HypothesisEngine()
        synergies = engine.find_synergy_hypotheses(["T99"])

        # Should return empty for unknown tracks
        assert synergies == []


class TestReportGeneration:
    def test_generate_hypothesis_report(self) -> None:
        engine = HypothesisEngine()
        context = {"runs": _make_mock_runs(), "reports": _make_mock_reports()}

        report = generate_hypothesis_report(engine, "T3", context, num_hypotheses=2)

        assert "Hypothesis Generation Report" in report
        assert "T3" in report
        assert "Generated Hypotheses" in report

    def test_report_contains_hypotheses(self) -> None:
        engine = HypothesisEngine()
        context = {"runs": _make_mock_runs(), "reports": _make_mock_reports()}

        report = generate_hypothesis_report(engine, "T3", context, num_hypotheses=2)

        # Should contain hypothesis details
        assert "###" in report  # Markdown headings for hypotheses


class TestEdgeCases:
    def test_generate_zero_hypotheses(self) -> None:
        engine = HypothesisEngine()
        context = {}

        result = engine.generate_hypotheses("T3", context, num_hypotheses=0)

        assert len(result.hypotheses) == 0

    def test_rank_empty_hypotheses(self) -> None:
        engine = HypothesisEngine()
        ranked = engine.rank_hypotheses([])

        assert ranked == []

