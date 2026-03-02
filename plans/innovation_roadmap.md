# Innovation Roadmap — Engine Experiment Harness

## Executive Summary

This roadmap captures 28 breakthrough innovations across 7 clusters, derived from deep analysis of the full codebase (T1–T13 tracks, simulator, gating, reporting, compare, memo, spec_utils, inverse_arms). Innovations are organized into 4 implementation phases ordered by impact-to-effort ratio.

---

## Current State Gaps

| Gap | Location | Problem |
|-----|----------|---------|
| Fixed gate thresholds | `exp/gating.py`, `exp/compare.py` | `delta >= 3/5/8` are arbitrary constants with no statistical grounding |
| Flat VARIANT_EFFECTS | `exp/constants.py` | Fixed point estimates ignore param-dependence and uncertainty |
| No causal attribution | `exp/compare.py` | We know *what* won, not *why* |
| No cross-track synergy | `exp/simulator.py` | Tracks compete independently; combinations untested |
| Flat JSON artifacts | `artifacts/` | No queryable history; no pattern detection |
| Anchor assumed stable | `exp/compare.py` | Anchor drift across stages is undetected |
| Grid-search specs | `scripts/generate_specs.py` | Inefficient; most of the grid is uninformative |
| Single composite scalar | `exp/gating.py` | Collapses 3 dimensions; destroys deployment-scenario info |
| No SNR metric | `exp/compare.py` | Variance across seeds not decomposed into signal vs. noise |

---

## Phase 0 — Quick Wins (1 week)

### P0-1: Anchor Drift Detection
**File:** `exp/compare.py` + `exp/anchor_audit.py`  
**What:** Add `check_anchor_stability()` that computes drift across stages. Flag comparisons when `anchor_drift > 0.5` composite points.  
**Why:** The anchor is the foundation of all relative deltas. Silent drift invalidates every comparison report.

### P0-2: Cross-Seed SNR Decomposition
**File:** `exp/compare.py`  
**What:** Decompose variance across seeds into signal (between-seed) vs. noise (within-seed). Add `snr_score` to `ComparisonReport`. Gate Stage 3 on `snr >= 2.0`.  
**Why:** Tracks with low SNR are initialization-dependent flukes, not real improvements.

### P0-3: Cross-Track Synergy Matrix
**File:** `scripts/generate_synergy_specs.py` + `exp/synergy.py`  
**What:** Auto-generate pairwise combination specs for top-2 variants from each track. Compute `synergy[Ti][Tj] = composite(Ti+Tj) - max(composite(Ti), composite(Tj))`.  
**Why:** T3 (compression) + T4 (symbolic) may be super-additive. Currently untested.

---

## Phase 1 — Core Intelligence (2–3 weeks)

### P1-1: Shapley Param Attribution Engine
**File:** `exp/attribution.py`  
**What:** Compute Shapley values for each param's contribution to composite delta. Uses existing run artifacts — no new GPU hours. Integrated into memo generation.  
**Why:** Closes the "why" gap. Memos say "T3 wins due to `compression_ratio=0.85` (Shapley=2.3)."

### P1-2: Pareto-Frontier Promotion
**File:** `exp/gating.py`  
**What:** Add `pareto_promote()` alongside existing gate. Tracks on the Pareto frontier across `(composite, latency_p50, energy_kwh, fluency)` get a "pareto_promoted" flag even if they miss the composite threshold.  
**Why:** A track that wins on energy+latency but misses composite by 0.5 pts is still valuable for edge deployment.

### P1-3: Bayesian Adaptive Gate Calibration
**File:** `exp/adaptive_gate.py`  
**What:** Fit Beta distribution over historical pass/fail outcomes. Gate threshold = 80th percentile of posterior. Replaces hardcoded `3/5/8` with data-driven thresholds.  
**Why:** Gates self-calibrate. Strong programs get harder gates; struggling programs get fair gates.

---

## Phase 2 — Simulation Upgrade (1 month)

### P2-1: Gaussian Process Effect Surface
**File:** `exp/effect_surface.py`  
**What:** Replace fixed `VARIANT_EFFECTS` point estimates with GP regression over param space. GP learns `effect(params) → (delta_mean, delta_std)` from observed runs. Provides uncertainty estimates.  
**Why:** Simulator becomes self-improving. Stage 3 simulations are more accurate than Stage 1.

### P2-2: Adversarial Spec Generation
**File:** `scripts/generate_adversarial_specs.py`  
**What:** Auto-generate specs that probe ±20% around known failure boundaries (e.g., `compression_ratio ∈ {0.85, 0.88, 0.90, 0.92, 0.95}`). Maps the failure surface before production commitment.  
**Why:** Produces a failure map per track. Stage 3 memo can say "T3 safe up to 0.88; recommend 0.82 with 15% margin."

### P2-3: Genetic Algorithm Spec Evolution
**File:** `exp/spec_evolution.py`  
**What:** Replace grid search with evolutionary spec generation. Fitness = decision_score. Crossover = combine params from two high-performing specs. Mutation = perturb params within valid ranges.  
**Why:** Finds optimal param combinations in O(log N) experiments. Reduces GPU hours 40–60%.

---

## Phase 3 — Meta-System (2 months)

### P3-1: Experiment Knowledge Graph
**File:** `exp/knowledge_graph.py`  
**What:** Graph DB (nodes: RunResult, Spec, Stage, Track, Param; edges: ran_with, compared_to, promoted_from, caused_failure). Enables queries like "which params predict Stage 3 promotion?"  
**Why:** Harness becomes self-aware of its own history.

### P3-2: LLM Hypothesis Generation Engine
**File:** `exp/hypothesis_engine.py`  
**What:** After each stage gate, LLM generates specific falsifiable hypotheses for next stage based on Shapley attribution + failure modes + literature.  
**Why:** Closes the loop between results and next experiments. Harness becomes a scientific reasoning engine.

### P3-3: Real-Time Experiment Dashboard
**File:** `exp/dashboard.py`  
**What:** FastAPI + WebSocket dashboard showing live stage progression, anchor-relative deltas, predicted promotion probability, budget burn rate.  
**Why:** Replaces static JSON + markdown with live visibility.

---

## Phase 4 — New Tracks & Radical Ideas (Research)

### P4-1: T14 — Speculative Decoding with Symbolic Verification
**Hypothesis:** Tiny draft model (100M) proposes tokens; main model verifies using T4-style symbolic constraints. Expected: latency -40%, reasoning +2.0, consistency +3.0.

### P4-2: T15 — Sparse Mixture-of-Experts with Dynamic Routing
**Hypothesis:** Route each token to 2-of-N expert modules. Experts specialize in long-context/reasoning/consistency. Expected: composite +4.0, energy -20%.

### P4-3: T16 — Constitutional Self-Improvement Loop
**Hypothesis:** Model generates its own training signal via constitutional principles. No human feedback. Expected: consistency +5.0, reasoning +2.0.

### P4-4: T17 — Temporal Consistency Transformer
**Hypothesis:** Temporal position embeddings track when claims were made. Detects contradictions that emerge over time. Expected: consistency +6.0, long_context +3.0.

### P4-5: T-META — Self-Modifying Harness Track
**Hypothesis:** Experiment on the harness itself. Test whether changing composite weights, anchor update frequency, or bootstrap n changes gate decisions. Meta-scientific rigor.

### P4-6: Thermodynamic Experiment Scheduling
**Hypothesis:** Model program as thermodynamic system. Energy = -composite_delta. Temperature = exploration/exploitation tradeoff. Annealing schedule derived from thermodynamic principles.

---

## Files to Create / Modify

| File | Action | Phase |
|------|--------|-------|
| `exp/anchor_audit.py` | CREATE | P0-1 |
| `exp/compare.py` | MODIFY (add SNR, anchor checks) | P0-1, P0-2 |
| `exp/synergy.py` | CREATE | P0-3 |
| `scripts/generate_synergy_specs.py` | CREATE | P0-3 |
| `exp/attribution.py` | CREATE | P1-1 |
| `exp/gating.py` | MODIFY (add Pareto) | P1-2 |
| `exp/adaptive_gate.py` | CREATE | P1-3 |
| `exp/memo.py` | MODIFY (add attribution) | P1-1 |
| `exp/effect_surface.py` | CREATE | P2-1 |
| `scripts/generate_adversarial_specs.py` | CREATE | P2-2 |
| `exp/spec_evolution.py` | CREATE | P2-3 |
| `exp/knowledge_graph.py` | CREATE | P3-1 |
| `exp/hypothesis_engine.py` | CREATE | P3-2 |
| `exp/dashboard.py` | CREATE | P3-3 |
| `exp/constants.py` | MODIFY (T14–T17 effects) | P4-1..4 |
| `exp/inverse_arms.py` | MODIFY (T14–T17 configs) | P4-1..4 |
| `tests/test_anchor_audit.py` | CREATE | P0-1 |
| `tests/test_synergy.py` | CREATE | P0-3 |
| `tests/test_attribution.py` | CREATE | P1-1 |
| `tests/test_adaptive_gate.py` | CREATE | P1-3 |
| `tests/test_pareto_gate.py` | CREATE | P1-2 |
| `tests/test_spec_evolution.py` | CREATE | P2-3 |

---

## Success Criteria

| Phase | Success Metric |
|-------|---------------|
| P0 | Anchor drift flagged in all comparison reports; SNR score present; synergy matrix generated |
| P1 | Shapley values in every memo; Pareto flags in gate output; adaptive thresholds replace hardcoded |
| P2 | GP effect surface reduces Stage 2 GPU hours by 30%; adversarial specs map failure boundaries |
| P3 | Knowledge graph answers "which params predict promotion?" in <1s; dashboard live |
| P4 | T14/T15 pass Stage 1 gate; T-META produces actionable harness improvements |
