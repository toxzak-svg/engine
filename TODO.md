# Innovation Implementation TODO

## Phase 0 — Quick Wins ✅ COMPLETE

- [x] P0-1: Anchor Drift Detection
  - [x] Create `exp/anchor_audit.py` with `check_anchor_stability()`, `compute_snr()`, `audit_comparison_anchor_validity()`
  - [x] Create `tests/test_anchor_audit.py` — 22 tests passing

- [x] P0-2: Cross-Seed SNR Decomposition
  - [x] `compute_snr()` implemented in `exp/anchor_audit.py`
  - [x] SNR decomposition: signal_std / noise_floor with simulator-calibrated noise floor

- [x] P0-3: Cross-Track Synergy Matrix
  - [x] Create `exp/synergy.py` — `compute_synergy_score()`, `build_synergy_matrix()`, `format_synergy_report()`, `select_portfolio_diverse_tracks()`
  - [x] Create `tests/test_synergy.py` — 17 tests passing

## Phase 1 — Core Intelligence ✅ COMPLETE

- [x] P1-1: Shapley Param Attribution Engine
  - [x] Create `exp/attribution.py` with `shapley_param_attribution()`, `format_attribution_report()`, `attribute_all_tracks()`
  - [x] Create `tests/test_attribution.py` — 15 tests passing

- [x] P1-2: Pareto-Frontier Promotion
  - [x] Add `pareto_promote()` to `exp/gating.py`
  - [x] Add `pareto_tracks` to gate output dict
  - [x] All existing gating tests still pass (3/3)

- [x] P1-3: Bayesian Adaptive Gate Calibration
  - [x] Create `exp/adaptive_gate.py` with `calibrate_gate_threshold()`, `calibrate_all_stages()`, `format_adaptive_gate_report()`
  - [x] Pure-Python Beta CDF/PPF (no scipy dependency) — Lentz algorithm fixed
  - [x] Create `tests/test_adaptive_gate.py` — 22 tests passing

## Phase 2 — Simulation Upgrade ✅ COMPLETE

- [x] P2-2: Adversarial Spec Generation
  - [x] Create `scripts/generate_adversarial_specs.py` — sweeps failure boundaries for T1–T6
  - [x] Failure boundaries: analog_drift (T1), reversibility_break (T2), critical_fact_loss (T3), entity_role_swap (T4), invalid_circuit (T5), repetitive_text (T6)

- [x] P2-3: Genetic Algorithm Spec Evolution
  - [x] Create `exp/spec_evolution.py` — tournament selection, uniform crossover, mutation, diversity metric
  - [x] `PARAM_SEARCH_SPACE` defined for all 6 tracks

- [x] P2-1: Gaussian Process Effect Surface
  - [x] Create `exp/effect_surface.py` with `GPEffectSurface` class
  - [x] Hybrid prediction: GP surface + fallback to simulator prior
  - [x] Acquisition function for next-spec recommendation
  - [x] Tests

## Phase 3 — Meta-System ✅ COMPLETE

- [x] P3-1: Experiment Knowledge Graph
  - [x] Create `exp/knowledge_graph.py` — graph-based experiment history
  - [x] Query: promotion predictors, analogous tracks, next specs
  - [x] Tests

- [x] P3-2: LLM Hypothesis Generation Engine
  - [x] Create `exp/hypothesis_engine.py` — AI-powered hypothesis generation
  - [x] Template-based + LLM-based generation
  - [x] Synergy discovery across tracks
  - [x] Tests

- [x] P3-3: Real-Time Dashboard
  - [x] Create `exp/dashboard.py` — live monitoring
  - [x] HTML/JSON rendering, stage status, leaderboard
  - [x] Alert system, CLI server
  - [x] Tests

## Phase 4 — New Tracks ✅ COMPLETE

- [x] P4-1: T14 Speculative Decoding + Symbolic Verification
  - [x] Add VARIANT_EFFECTS to `exp/constants.py`
  - [x] Add metrics to TRACK_SPECIFIC_METRIC_BASELINES
  - [x] Add TRACK_PASS_CRITERIA and ENGINEERING_COMPLEXITY

- [x] P4-2: T15 Sparse MoE Routing
  - [x] Add VARIANT_EFFECTS to `exp/constants.py`
  - [x] Add metrics to TRACK_SPECIFIC_METRIC_BASELINES
  - [x] Add TRACK_PASS_CRITERIA and ENGINEERING_COMPLEXITY

- [x] P4-3: T16 Constitutional Self-Improvement
  - [x] Add VARIANT_EFFECTS + config

- [x] P4-4: T17 Temporal Consistency Transformer
  - [x] Add VARIANT_EFFECTS + config

- [x] P4-5: T-META Self-Modifying Harness Track
  - [x] Design + implement (included in TRACKS and constants)

## Test Summary

| Suite | Tests | Status |
|-------|-------|--------|
| test_anchor_audit.py | 22 | ✅ All pass |
| test_synergy.py | 17 | ✅ All pass |
| test_attribution.py | 15 | ✅ All pass |
| test_adaptive_gate.py | 22 | ✅ All pass |
| test_effect_surface.py | 18 | ✅ All pass |
| test_knowledge_graph.py | 18 | ✅ All pass |
| test_hypothesis_engine.py | 13 | ✅ All pass |
| test_dashboard.py | 17 | ✅ All pass |
| test_gating.py (existing) | 3 | ✅ All pass |
| Full suite (existing) | 213 | ✅ All pass |
| Pre-existing failures | 4 | ⚠️ Pre-existing (unrelated to new code) |
| **Total new tests** | **145** | **✅ All pass** |
