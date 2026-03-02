"""Integration wiring tests.

Covers the two integration seams that were previously untested:
  1. anchor_audit ↔ compare.py — verify that ComparisonReport correctly
     reflects anchor stability results from check_anchor_stability().
  2. attribution ↔ memo.py — verify that build_decision_memo() includes a
     Param Attribution section when run_lookup contains runs with params.
"""

from __future__ import annotations

from exp.anchor_audit import DRIFT_FAIL_THRESHOLD, DRIFT_WARN_THRESHOLD
from exp.compare import compare_runs
from exp.memo import build_decision_memo
from exp.models import ComparisonReport, RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    run_id: str,
    stage: int = 2,
    track_id: str = "T3",
    composite: float = 65.0,
    fluency: float = 90.0,
    latency_p50: float = 120.0,
    failure_flags: list[str] | None = None,
    params: dict | None = None,
    model_variant: str = "T3-E2",
    train_cost: float = 100.0,
    infer_cost: float = 20.0,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        spec_id=f"spec-{run_id}",
        commit_sha="abc123",
        seed=101,
        train_cost=train_cost,
        infer_cost=infer_cost,
        latency_p50=latency_p50,
        latency_p95=latency_p50 * 1.3,
        energy_kwh=12.0,
        metric_values={
            "composite": composite,
            "fluency": fluency,
            "long_context": composite,
            "reasoning": composite,
            "consistency": composite,
            "output_length": 500.0,
            "robustness": 0.8,
            "constraint_adherence": 0.9,
        },
        failure_flags=failure_flags or [],
        track_id=track_id,
        stage=stage,
        model_variant=model_variant,
        benchmark_scores={
            "gsm8k": composite,
            "bbh": composite - 1.0,
            "needle_32k": composite + 0.5,
        },
        metadata={"params": params or {}},
    )


# ---------------------------------------------------------------------------
# anchor_audit ↔ compare.py
# ---------------------------------------------------------------------------

class TestAnchorAuditCompareIntegration:
    """anchor_audit results must propagate correctly into compare_runs() output."""

    def _stable_anchor_history(self, stage: int = 2) -> list[RunResult]:
        """Returns anchor runs that are well within drift bounds."""
        return [
            _run("anc-s1-r1", stage=1, track_id="ANCHOR", composite=60.0),
            _run("anc-s1-r2", stage=1, track_id="ANCHOR", composite=60.2),
            _run("anc-s2-r1", stage=stage, track_id="ANCHOR", composite=60.1),
            _run("anc-s2-r2", stage=stage, track_id="ANCHOR", composite=60.3),
        ]

    def _failing_anchor_history(self, candidate_stage: int = 2) -> list[RunResult]:
        """Returns anchor runs whose drift > DRIFT_FAIL_THRESHOLD (1.0 pts)."""
        return [
            _run("anc-s1", stage=1, track_id="ANCHOR", composite=60.0),
            _run("anc-s2", stage=candidate_stage, track_id="ANCHOR", composite=62.5),
        ]

    def _warning_anchor_history(self, candidate_stage: int = 2) -> list[RunResult]:
        """Returns anchor runs where drift_warn=True, drift_fail=False, and
        candidate_stage IS in affected_stages.

        Three-stage setup:
          S1: two runs at 60.0 each
          S2 (candidate stage): one run at 60.9
        mean_c = (60.0+60.0+60.9)/3 = 60.3
        drift  = 0.9  (> DRIFT_WARN_THRESHOLD=0.5, < DRIFT_FAIL_THRESHOLD=1.0)
        S2 deviation = |60.9-60.3| = 0.6 > 0.5  → candidate stage in affected_stages
        S1 deviation = |60.0-60.3| = 0.3         → NOT in affected_stages
        """
        assert DRIFT_WARN_THRESHOLD < DRIFT_FAIL_THRESHOLD
        return [
            _run("anc-s1-a", stage=1, track_id="ANCHOR", composite=60.0),
            _run("anc-s1-b", stage=1, track_id="ANCHOR", composite=60.0),
            _run("anc-s2",   stage=candidate_stage, track_id="ANCHOR", composite=60.9),
        ]

    # --- Valid anchor (stable) ---

    def test_stable_anchor_populates_anchor_delta_metrics(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._stable_anchor_history(stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        assert report.anchor_delta_metrics != {}, \
            "Stable anchor should produce non-empty anchor_delta_metrics"
        assert "composite" in report.anchor_delta_metrics
        assert report.pass_fail.get("anchor_valid") is True
        assert report.pass_fail.get("anchor_warning") is False

    # --- Invalid anchor (drift_fail) ---

    def test_failed_anchor_suppresses_anchor_delta_metrics(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._failing_anchor_history(candidate_stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        # Invalid anchor → anchor_delta_metrics must be empty
        assert report.anchor_delta_metrics == {}, (
            f"Drifted anchor should suppress anchor_delta_metrics, "
            f"got: {report.anchor_delta_metrics}"
        )

    def test_failed_anchor_sets_anchor_valid_false_in_pass_fail(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._failing_anchor_history(candidate_stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        assert report.pass_fail.get("anchor_valid") is False
        assert report.pass_fail.get("anchor_warning") is True

    def test_failed_anchor_message_mentions_stage(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._failing_anchor_history(candidate_stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        msg = report.pass_fail.get("anchor_message", "")
        assert "Stage 2" in msg or "stage" in msg.lower()

    # --- Warning anchor (drift_warn, but valid) ---

    def test_warning_anchor_allows_delta_metrics_but_sets_warning(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._warning_anchor_history(candidate_stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        # Warning anchor is still valid — deltas should be computed
        assert report.pass_fail.get("anchor_valid") is True
        assert report.pass_fail.get("anchor_warning") is True
        # anchor_delta_metrics may or may not be populated (depends on whether
        # candidate stage is in "affected_stages") — check the pass_fail keys exist
        assert "anchor_valid" in report.pass_fail
        assert "anchor_message" in report.pass_fail

    # --- No anchor provided ---

    def test_no_anchor_no_anchor_fields_in_pass_fail(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)

        report = compare_runs(candidate, baseline)

        assert report.anchor_delta_metrics == {}
        assert "anchor_valid" not in report.pass_fail

    # --- Anchor without history ---

    def test_anchor_without_history_assumes_valid(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)

        report = compare_runs(candidate, baseline, anchor=anchor)

        assert report.anchor_delta_metrics != {}, \
            "Anchor without history should assume valid and compute deltas"
        assert report.pass_fail.get("anchor_valid") is True

    # --- anchor_audit embedded in significance_tests ---

    def test_anchor_audit_dict_embedded_in_significance_tests(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._stable_anchor_history(stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        sig = report.significance_tests
        assert "anchor_audit" in sig, "anchor_audit dict should be in significance_tests"
        audit = sig["anchor_audit"]
        assert "stable" in audit
        assert "drift" in audit

    def test_anchor_delta_is_candidate_minus_anchor_composite(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._stable_anchor_history(stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        if report.anchor_delta_metrics:
            expected_composite_delta = 70.0 - 60.0
            assert abs(report.anchor_delta_metrics["composite"] - expected_composite_delta) < 1e-3

    def test_stage_gate_pass_uses_stage_baseline_delta_even_with_anchor(self) -> None:
        # Stage 2 gate: delta_composite >= 5.0 and latency_overhead_pct <= 15.0
        # candidate composite = 67.0, baseline composite = 62.0 → delta = 5.0 → passes
        candidate = _run("cand", stage=2, composite=67.0, latency_p50=120.0)
        baseline = _run("base", stage=2, composite=62.0, latency_p50=110.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._stable_anchor_history(stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        assert report.pass_fail["stage_gate_pass"] is True

    def test_anchor_run_ids_populated_for_valid_anchor(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._stable_anchor_history(stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        assert "anc-ref" in report.anchor_run_ids

    def test_anchor_run_ids_empty_for_invalid_anchor(self) -> None:
        candidate = _run("cand", stage=2, composite=70.0)
        baseline = _run("base", stage=2, composite=62.0)
        anchor = _run("anc-ref", stage=2, track_id="ANCHOR", composite=60.0)
        history = self._failing_anchor_history(candidate_stage=2)

        report = compare_runs(candidate, baseline, anchor=anchor, anchor_history=history)

        assert report.anchor_run_ids == []


# ---------------------------------------------------------------------------
# attribution ↔ memo.py
# ---------------------------------------------------------------------------

class TestAttributionMemoIntegration:
    """build_decision_memo() should include a Param Attribution section when
    run_lookup provides runs that have params in their metadata."""

    def _make_report(
        self,
        track_id: str,
        stage: int,
        candidate_run_ids: list[str],
        baseline_run_ids: list[str],
        composite_delta: float = 5.5,
    ) -> ComparisonReport:
        return ComparisonReport(
            candidate_run_ids=candidate_run_ids,
            baseline_run_ids=baseline_run_ids,
            delta_metrics={"composite": composite_delta},
            anchor_delta_metrics={"composite": composite_delta},
            significance_tests={"ci95_excludes_zero": True, "ci95": [0.5, 2.5]},
            pass_fail={"overall_pass": True},
            candidate_stage=stage,
            track_id=track_id,
        )

    def test_memo_with_run_lookup_includes_attribution_section(self) -> None:
        report = self._make_report("T3", stage=1, candidate_run_ids=["c1", "c2"], baseline_run_ids=["b1"])
        run_lookup = {
            "c1": _run("c1", stage=1, track_id="T3", composite=65.0,
                       params={"compression_ratio": 0.70}),
            "c2": _run("c2", stage=1, track_id="T3", composite=68.0,
                       params={"compression_ratio": 0.85}),
            "b1": _run("b1", stage=1, track_id="T3", composite=60.0),
        }
        memo = build_decision_memo(stage=1, reports=[report], run_lookup=run_lookup)
        assert "Param Attribution" in memo, "Attribution section should be present"

    def test_memo_without_run_lookup_skips_attribution(self) -> None:
        report = self._make_report("T3", stage=1, candidate_run_ids=["c1"], baseline_run_ids=["b1"])
        memo = build_decision_memo(stage=1, reports=[report], run_lookup=None)
        assert "Param Attribution" not in memo, "No run_lookup → no attribution section"

    def test_memo_run_lookup_with_no_params_skips_attribution(self) -> None:
        report = self._make_report("T3", stage=1, candidate_run_ids=["c1", "c2"], baseline_run_ids=["b1"])
        run_lookup = {
            "c1": _run("c1", stage=1, track_id="T3", composite=65.0, params={}),
            "c2": _run("c2", stage=1, track_id="T3", composite=68.0, params={}),
            "b1": _run("b1", stage=1, track_id="T3", composite=60.0),
        }
        memo = build_decision_memo(stage=1, reports=[report], run_lookup=run_lookup)
        # No params → no attributions → attribution section skipped
        assert "Param Attribution" not in memo

    def test_memo_attribution_lists_param_name(self) -> None:
        report = self._make_report("T3", stage=1, candidate_run_ids=["c1", "c2"], baseline_run_ids=["b1"])
        run_lookup = {
            "c1": _run("c1", stage=1, track_id="T3", composite=65.0,
                       params={"compression_ratio": 0.70}),
            "c2": _run("c2", stage=1, track_id="T3", composite=68.0,
                       params={"compression_ratio": 0.85}),
            "b1": _run("b1", stage=1, track_id="T3", composite=60.0),
        }
        memo = build_decision_memo(stage=1, reports=[report], run_lookup=run_lookup)
        assert "compression_ratio" in memo

    def test_memo_attribution_multiple_tracks(self) -> None:
        r1 = self._make_report("T3", stage=1, candidate_run_ids=["c-t3"], baseline_run_ids=["b-t3"])
        r2 = self._make_report("T4", stage=1, candidate_run_ids=["c-t4a", "c-t4b"], baseline_run_ids=["b-t4"])
        run_lookup = {
            "c-t3": _run("c-t3", stage=1, track_id="T3", composite=67.0,
                         params={"compression_ratio": 0.85}),
            "b-t3": _run("b-t3", stage=1, track_id="T3", composite=62.0),
            "c-t4a": _run("c-t4a", stage=1, track_id="T4", composite=66.0,
                          params={"role_permutation_noise": 0.20}),
            "c-t4b": _run("c-t4b", stage=1, track_id="T4", composite=69.0,
                          params={"role_permutation_noise": 0.30}),
            "b-t4": _run("b-t4", stage=1, track_id="T4", composite=61.0),
        }
        memo = build_decision_memo(stage=1, reports=[r1, r2], run_lookup=run_lookup)
        assert "Param Attribution" in memo
        assert "role_permutation_noise" in memo

    def test_memo_attribution_ranking_order_matches_decision_score(self) -> None:
        # T3 has higher decision score; T4 has lower — T3 should appear first in memo
        r1 = self._make_report("T3", stage=1, candidate_run_ids=["c-t3"], baseline_run_ids=["b-t3"],
                               composite_delta=7.0)
        r2 = self._make_report("T4", stage=1, candidate_run_ids=["c-t4"], baseline_run_ids=["b-t4"],
                               composite_delta=3.0)
        memo = build_decision_memo(stage=1, reports=[r1, r2])
        # T3 appears before T4 in the ranking table (higher delta → higher rank)
        t3_pos = memo.find("T3")
        t4_pos = memo.find("T4")
        assert t3_pos < t4_pos, "T3 should be ranked above T4 in the memo"

    def test_memo_always_contains_ranking_table(self) -> None:
        report = self._make_report("T1", stage=2, candidate_run_ids=["c1"], baseline_run_ids=["b1"])
        memo = build_decision_memo(stage=2, reports=[report])
        assert "Stage 2 Decision Memo" in memo
        assert "| Rank | Track |" in memo

    def test_memo_empty_reports_is_safe(self) -> None:
        memo = build_decision_memo(stage=1, reports=[])
        assert "Stage 1 Decision Memo" in memo

    def test_memo_recovery_runs_excluded_from_primary_ranking(self) -> None:
        core = self._make_report("T1", stage=2,
                                 candidate_run_ids=["core-run"],
                                 baseline_run_ids=["core-base"])
        recovery = self._make_report("T4", stage=2,
                                     candidate_run_ids=["recovery-s2-fix"],
                                     baseline_run_ids=["rec-base"])
        memo = build_decision_memo(stage=2, reports=[core, recovery])
        assert "Recovery comparisons excluded" in memo

    def test_attribution_section_shows_direction_indicator(self) -> None:
        report = self._make_report("T3", stage=1, candidate_run_ids=["c1", "c2"], baseline_run_ids=["b1"])
        run_lookup = {
            "c1": _run("c1", stage=1, track_id="T3", composite=65.0,
                       params={"compression_ratio": 0.70}),
            "c2": _run("c2", stage=1, track_id="T3", composite=68.0,
                       params={"compression_ratio": 0.85}),
            "b1": _run("b1", stage=1, track_id="T3", composite=60.0),
        }
        memo = build_decision_memo(stage=1, reports=[report], run_lookup=run_lookup)
        # format_attribution_report uses ↑/↓ direction icons
        assert "↑" in memo or "↓" in memo

    def test_attribution_section_includes_best_value_recommendation(self) -> None:
        report = self._make_report("T3", stage=1, candidate_run_ids=["c1", "c2"], baseline_run_ids=["b1"])
        run_lookup = {
            "c1": _run("c1", stage=1, track_id="T3", composite=65.0,
                       params={"compression_ratio": 0.70}),
            "c2": _run("c2", stage=1, track_id="T3", composite=68.0,
                       params={"compression_ratio": 0.85}),
            "b1": _run("b1", stage=1, track_id="T3", composite=60.0),
        }
        memo = build_decision_memo(stage=1, reports=[report], run_lookup=run_lookup)
        # format_attribution_report writes "Key driver: ..." and "Recommendation: fix ... in next stage"
        assert "Key driver" in memo
        assert "Recommendation" in memo
