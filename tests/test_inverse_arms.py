"""Tests for inverse-directed experiment arms."""

from __future__ import annotations

import pytest

from exp.constants import VariantEffect
from exp.inverse_arms import (
    INVERSE_ARM_BUNDLES,
    INVERSE_ARM_CONFIGS,
    INVERSE_ARM_METRIC_BASELINES,
    INVERSE_ARM_VARIANT_EFFECTS,
    InverseArmConfig,
    InverseArmType,
    get_inverse_arm_config,
    get_inverse_arm_for_track,
    get_inverse_arm_variant_effect,
    list_inverse_arm_tracks,
    list_inverse_arms,
)
from exp.models import ExperimentSpec, RunResult
from exp.runners.inverse_arm_runners import (
    AdversarialUserSimRunner,
    ActiveRecallRouterRunner,
    CounterfactualAuditRunner,
    DeliberationCollapseRunner,
    InverseArmRunner,
    LatentPlanSwappingRunner,
    NoiseInjectionEnsembleRunner,
    ProofCarryingOutputsRunner,
    evaluate_inverse_arm_pass,
)


class TestInverseArmTypes:
    """Tests for inverse arm type enumeration."""
    
    def test_all_arm_types_defined(self):
        """All 7 arm types should be defined."""
        assert len(InverseArmType) == 7
        assert InverseArmType.DELIBERATION_COLLAPSE.value == "A"
        assert InverseArmType.COUNTERFACTUAL_AUDIT.value == "B"
        assert InverseArmType.ACTIVE_RECALL_ROUTER.value == "C"
        assert InverseArmType.PROOF_CARRYING_OUTPUTS.value == "D"
        assert InverseArmType.NOISE_INJECTION_ENSEMBLE.value == "E"
        assert InverseArmType.LATENT_PLAN_SWAPPING.value == "F"
        assert InverseArmType.ADVERSARIAL_USER_SIM.value == "G"


class TestInverseArmConfigs:
    """Tests for inverse arm configurations."""
    
    def test_all_configs_present(self):
        """All arm types should have configurations."""
        for arm_type in InverseArmType:
            assert arm_type in INVERSE_ARM_CONFIGS
            config = INVERSE_ARM_CONFIGS[arm_type]
            assert isinstance(config, InverseArmConfig)
            assert config.arm_type == arm_type
            assert len(config.description) > 0
            assert len(config.inversion_principle) > 0
            assert len(config.metrics) > 0
            assert len(config.failure_modes) > 0
    
    def test_get_inverse_arm_config(self):
        """get_inverse_arm_config should return correct config."""
        config = get_inverse_arm_config(InverseArmType.DELIBERATION_COLLAPSE)
        assert config.arm_type == InverseArmType.DELIBERATION_COLLAPSE
        assert config.track_id == "T7"
        assert "verbosity_penalty" in config.metrics
    
    def test_track_ids_unique(self):
        """Each arm should have a unique track ID."""
        track_ids = [config.track_id for config in INVERSE_ARM_CONFIGS.values()]
        assert len(track_ids) == len(set(track_ids))
    
    def test_track_ids_in_t7_t13_range(self):
        """Track IDs should be T7-T13."""
        for config in INVERSE_ARM_CONFIGS.values():
            assert config.track_id.startswith("T")
            track_num = int(config.track_id[1:])
            assert 7 <= track_num <= 13


class TestInverseArmVariantEffects:
    """Tests for inverse arm variant effects."""
    
    def test_all_variants_have_effects(self):
        """All variant IDs should have defined effects."""
        for arm_type in InverseArmType:
            # Check E1, E2, E3 variants
            for exp_num in [1, 2, 3]:
                variant_id = f"INV-{arm_type.value}-E{exp_num}"
                assert variant_id in INVERSE_ARM_VARIANT_EFFECTS, f"Missing variant: {variant_id}"
    
    def test_variant_effects_are_valid(self):
        """All variant effects should have valid values."""
        for variant_id, effect in INVERSE_ARM_VARIANT_EFFECTS.items():
            assert isinstance(effect, VariantEffect)
            # Latency and energy deltas can be negative (improvements)
            assert -100.0 <= effect.latency_delta_pct <= 100.0
            assert -100.0 <= effect.energy_delta_pct <= 100.0
    
    def test_deliberation_collapse_reduces_latency(self):
        """Deliberation Collapse variants should reduce latency."""
        for exp_num in [1, 2, 3]:
            variant_id = f"INV-A-E{exp_num}"
            effect = INVERSE_ARM_VARIANT_EFFECTS[variant_id]
            assert effect.latency_delta_pct < 0, f"{variant_id} should reduce latency"
            assert effect.energy_delta_pct < 0, f"{variant_id} should reduce energy"
    
    def test_counterfactual_audit_increases_latency(self):
        """Counterfactual Audit variants should increase latency."""
        for exp_num in [1, 2, 3]:
            variant_id = f"INV-B-E{exp_num}"
            effect = INVERSE_ARM_VARIANT_EFFECTS[variant_id]
            assert effect.latency_delta_pct > 0, f"{variant_id} should increase latency (audit overhead)"
    
    def test_get_inverse_arm_variant_effect(self):
        """get_inverse_arm_variant_effect should return correct effect."""
        effect = get_inverse_arm_variant_effect("INV-A-E1")
        assert effect is not None
        assert effect.latency_delta_pct < 0
        
        # Unknown variant should return None
        assert get_inverse_arm_variant_effect("UNKNOWN") is None


class TestInverseArmMetricBaselines:
    """Tests for inverse arm metric baselines."""
    
    def test_all_tracks_have_baselines(self):
        """All inverse arm tracks should have metric baselines."""
        for track_id in ["T7", "T8", "T9", "T10", "T11", "T12", "T13"]:
            assert track_id in INVERSE_ARM_METRIC_BASELINES
            baselines = INVERSE_ARM_METRIC_BASELINES[track_id]
            assert len(baselines) > 0
    
    def test_baseline_values_in_valid_range(self):
        """Baseline values should be in reasonable ranges."""
        for track_id, baselines in INVERSE_ARM_METRIC_BASELINES.items():
            for metric, value in baselines.items():
                if "rate" in metric or "pct" in metric:
                    assert 0.0 <= value <= 100.0, f"{track_id}.{metric} = {value} out of range"
                elif "score" in metric:
                    assert 0.0 <= value <= 100.0, f"{track_id}.{metric} = {value} out of range"


class TestInverseArmBundles:
    """Tests for inverse arm bundles."""
    
    def test_bundle_1_defined(self):
        """Bundle 1 should be defined with correct arms."""
        bundle = INVERSE_ARM_BUNDLES["bundle_1_cheap_high_win"]
        assert bundle["name"] == "Bundle 1 (cheap, high chance of win)"
        assert len(bundle["arms"]) == 2
        assert InverseArmType.DELIBERATION_COLLAPSE in bundle["arms"]
        assert InverseArmType.LATENT_PLAN_SWAPPING in bundle["arms"]
        assert bundle["risk_level"] == "low"
    
    def test_bundle_2_defined(self):
        """Bundle 2 should be defined with correct arms."""
        bundle = INVERSE_ARM_BUNDLES["bundle_2_risky_massive_upside"]
        assert bundle["name"] == "Bundle 2 (risky, potentially massive upside)"
        assert len(bundle["arms"]) == 2
        assert InverseArmType.COUNTERFACTUAL_AUDIT in bundle["arms"]
        assert InverseArmType.PROOF_CARRYING_OUTPUTS in bundle["arms"]
        assert bundle["risk_level"] == "high"
    
    def test_bundle_3_defined(self):
        """Bundle 3 (top 3 recommended) should be defined."""
        bundle = INVERSE_ARM_BUNDLES["bundle_3_top3_recommended"]
        assert len(bundle["arms"]) == 3
        assert InverseArmType.DELIBERATION_COLLAPSE in bundle["arms"]
        assert InverseArmType.COUNTERFACTUAL_AUDIT in bundle["arms"]
        assert InverseArmType.NOISE_INJECTION_ENSEMBLE in bundle["arms"]


class TestInverseArmUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_inverse_arm_for_track(self):
        """get_inverse_arm_for_track should return correct arm type."""
        assert get_inverse_arm_for_track("T7") == InverseArmType.DELIBERATION_COLLAPSE
        assert get_inverse_arm_for_track("T8") == InverseArmType.COUNTERFACTUAL_AUDIT
        assert get_inverse_arm_for_track("T9") == InverseArmType.ACTIVE_RECALL_ROUTER
        assert get_inverse_arm_for_track("T10") == InverseArmType.PROOF_CARRYING_OUTPUTS
        assert get_inverse_arm_for_track("T11") == InverseArmType.NOISE_INJECTION_ENSEMBLE
        assert get_inverse_arm_for_track("T12") == InverseArmType.LATENT_PLAN_SWAPPING
        assert get_inverse_arm_for_track("T13") == InverseArmType.ADVERSARIAL_USER_SIM
        assert get_inverse_arm_for_track("T99") is None
    
    def test_list_inverse_arms(self):
        """list_inverse_arms should return all arm types."""
        arms = list_inverse_arms()
        assert len(arms) == 7
        assert InverseArmType.DELIBERATION_COLLAPSE in arms
    
    def test_list_inverse_arm_tracks(self):
        """list_inverse_arm_tracks should return T7-T13."""
        tracks = list_inverse_arm_tracks()
        assert tracks == ["T7", "T8", "T9", "T10", "T11", "T12", "T13"]


class TestInverseArmRunner:
    """Tests for InverseArmRunner base class."""
    
    def test_runner_supports_inverse_tracks(self):
        """InverseArmRunner should support T7-T13."""
        runner = InverseArmRunner()
        assert "T7" in runner.supported_tracks
        assert "T8" in runner.supported_tracks
        assert "T13" in runner.supported_tracks
        assert "T1" not in runner.supported_tracks
    
    def test_runner_executes_spec(self):
        """InverseArmRunner should execute a spec and return results."""
        runner = InverseArmRunner()
        spec = ExperimentSpec(
            id="test-t7-e1",
            track_id="T7",
            stage=1,
            hypothesis="Test hypothesis",
            model_variant="INV-A-E1",
            baseline_id="test-t7-baseline",
            train_budget_gpu_h=100.0,
            infer_budget_gpu_h=50.0,
            max_context=64000,
            datasets=["needle_32k", "gsm8k"],
            metrics=["composite", "accuracy_per_token"],
            seeds=[101],
            promotion_gate={},
            params={"postpass_tokens": 48},
        )
        
        result = runner.execute(
            spec=spec,
            seed=101,
            run_id="test-run-001",
            commit_sha="abc123",
        )
        
        assert result.success
        assert result.run_result is not None
        assert result.run_result.track_id == "T7"
        assert "accuracy_per_token" in result.run_result.metric_values
        assert "hallucination_surface_rate" in result.run_result.metric_values


class TestSpecializedRunners:
    """Tests for specialized inverse arm runners."""
    
    def test_deliberation_collapse_runner(self):
        """DeliberationCollapseRunner should be configured correctly."""
        runner = DeliberationCollapseRunner()
        assert runner.name == "deliberation_collapse"
        assert runner.supported_tracks == ["T7"]
        assert runner.arm_type == InverseArmType.DELIBERATION_COLLAPSE
    
    def test_counterfactual_audit_runner(self):
        """CounterfactualAuditRunner should be configured correctly."""
        runner = CounterfactualAuditRunner()
        assert runner.name == "counterfactual_audit"
        assert runner.supported_tracks == ["T8"]
        assert runner.arm_type == InverseArmType.COUNTERFACTUAL_AUDIT
    
    def test_active_recall_router_runner(self):
        """ActiveRecallRouterRunner should be configured correctly."""
        runner = ActiveRecallRouterRunner()
        assert runner.name == "active_recall_router"
        assert runner.supported_tracks == ["T9"]
        assert runner.arm_type == InverseArmType.ACTIVE_RECALL_ROUTER
    
    def test_proof_carrying_outputs_runner(self):
        """ProofCarryingOutputsRunner should be configured correctly."""
        runner = ProofCarryingOutputsRunner()
        assert runner.name == "proof_carrying_outputs"
        assert runner.supported_tracks == ["T10"]
        assert runner.arm_type == InverseArmType.PROOF_CARRYING_OUTPUTS
    
    def test_noise_injection_ensemble_runner(self):
        """NoiseInjectionEnsembleRunner should be configured correctly."""
        runner = NoiseInjectionEnsembleRunner()
        assert runner.name == "noise_injection_ensemble"
        assert runner.supported_tracks == ["T11"]
        assert runner.arm_type == InverseArmType.NOISE_INJECTION_ENSEMBLE
    
    def test_latent_plan_swapping_runner(self):
        """LatentPlanSwappingRunner should be configured correctly."""
        runner = LatentPlanSwappingRunner()
        assert runner.name == "latent_plan_swapping"
        assert runner.supported_tracks == ["T12"]
        assert runner.arm_type == InverseArmType.LATENT_PLAN_SWAPPING
    
    def test_adversarial_user_sim_runner(self):
        """AdversarialUserSimRunner should be configured correctly."""
        runner = AdversarialUserSimRunner()
        assert runner.name == "adversarial_user_sim"
        assert runner.supported_tracks == ["T13"]
        assert runner.arm_type == InverseArmType.ADVERSARIAL_USER_SIM


class TestEvaluateInverseArmPass:
    """Tests for inverse arm pass evaluation."""
    
    def _make_result(
        self,
        track_id: str,
        arm_type: str,
        metric_values: dict,
        failure_flags: list[str] | None = None,
    ) -> RunResult:
        """Helper to create a RunResult."""
        return RunResult(
            run_id="test-run",
            spec_id="test-spec",
            commit_sha="abc123",
            seed=101,
            train_cost=100.0,
            infer_cost=50.0,
            latency_p50=100.0,
            latency_p95=128.0,
            energy_kwh=50.0,
            metric_values=metric_values,
            failure_flags=failure_flags or [],
            track_id=track_id,
            stage=1,
            model_variant=f"INV-{arm_type}-E1",
            benchmark_scores={},
            metadata={"arm_type": arm_type},
        )
    
    def test_deliberation_collapse_pass(self):
        """Deliberation Collapse should pass with improved accuracy and reduced hallucination."""
        candidate = self._make_result(
            track_id="T7",
            arm_type="A",
            metric_values={
                "accuracy_per_token": 0.88,
                "hallucination_surface_rate": 3.0,
                "fluency": 90.0,
            },
        )
        baseline = self._make_result(
            track_id="T7",
            arm_type="A",
            metric_values={
                "accuracy_per_token": 0.85,
                "hallucination_surface_rate": 5.0,
                "fluency": 90.0,
            },
        )
        delta = {"accuracy_per_token": 0.03}
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is True
    
    def test_deliberation_collapse_fail_hallucination_spike(self):
        """Deliberation Collapse should fail with hallucination spike flag."""
        candidate = self._make_result(
            track_id="T7",
            arm_type="A",
            metric_values={
                "accuracy_per_token": 0.88,
                "hallucination_surface_rate": 3.0,
                "fluency": 90.0,
            },
            failure_flags=["hallucination_spike"],
        )
        baseline = self._make_result(
            track_id="T7",
            arm_type="A",
            metric_values={
                "accuracy_per_token": 0.85,
                "hallucination_surface_rate": 5.0,
                "fluency": 90.0,
            },
        )
        delta = {"accuracy_per_token": 0.03}
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is False
    
    def test_counterfactual_audit_pass(self):
        """Counterfactual Audit should pass with improved robustness."""
        candidate = self._make_result(
            track_id="T8",
            arm_type="B",
            metric_values={
                "robustness_delta": 0.5,
                "assumption_error_rate": 8.0,
                "fluency": 90.0,
            },
        )
        baseline = self._make_result(
            track_id="T8",
            arm_type="B",
            metric_values={
                "robustness_delta": 0.0,
                "assumption_error_rate": 12.0,
                "fluency": 90.0,
            },
        )
        delta = {"robustness_delta": 0.5}
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is True
    
    def test_proof_carrying_outputs_pass(self):
        """Proof-Carrying Outputs should pass with reduced catastrophic misses."""
        candidate = self._make_result(
            track_id="T10",
            arm_type="D",
            metric_values={
                "catastrophic_miss_rate": 1.0,
                "certificate_pass_rate": 85.0,
                "abstention_rate": 10.0,
                "fluency": 90.0,
            },
        )
        baseline = self._make_result(
            track_id="T10",
            arm_type="D",
            metric_values={
                "catastrophic_miss_rate": 2.0,
                "certificate_pass_rate": 80.0,
                "abstention_rate": 5.0,
                "fluency": 90.0,
            },
        )
        delta = {}
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is True
    
    def test_proof_carrying_outputs_fail_high_abstention(self):
        """Proof-Carrying Outputs should fail with excessive abstention."""
        candidate = self._make_result(
            track_id="T10",
            arm_type="D",
            metric_values={
                "catastrophic_miss_rate": 1.0,
                "certificate_pass_rate": 85.0,
                "abstention_rate": 20.0,  # Too high
                "fluency": 90.0,
            },
        )
        baseline = self._make_result(
            track_id="T10",
            arm_type="D",
            metric_values={
                "catastrophic_miss_rate": 2.0,
                "certificate_pass_rate": 80.0,
                "abstention_rate": 5.0,
                "fluency": 90.0,
            },
        )
        delta = {}
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is False
    
    def test_noise_injection_ensemble_pass(self):
        """Noise-Injection Ensemble should pass with improved constraint adherence."""
        candidate = self._make_result(
            track_id="T11",
            arm_type="E",
            metric_values={
                "constraint_adherence_score": 85.0,
                "instruction_following_rate": 90.0,
                "fluency": 90.0,
            },
        )
        baseline = self._make_result(
            track_id="T11",
            arm_type="E",
            metric_values={
                "constraint_adherence_score": 80.0,
                "instruction_following_rate": 85.0,
                "fluency": 90.0,
            },
        )
        delta = {
            "constraint_adherence_score": 5.0,
            "instruction_following_rate": 5.0,
        }
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is True
    
    def test_latent_plan_swapping_pass(self):
        """Latent Plan Swapping should pass with reduced variance."""
        candidate = self._make_result(
            track_id="T12",
            arm_type="F",
            metric_values={
                "variance_reduction_pct": 5.0,
                "plan_selection_bias": 0.5,
                "composite": 60.0,
                "fluency": 90.0,
            },
        )
        baseline = self._make_result(
            track_id="T12",
            arm_type="F",
            metric_values={
                "variance_reduction_pct": 0.0,
                "plan_selection_bias": 0.0,
                "composite": 55.0,
                "fluency": 90.0,
            },
        )
        delta = {
            "variance_reduction_pct": 5.0,
            "composite": 5.0,
        }
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is True
    
    def test_adversarial_user_sim_pass(self):
        """Adversarial User Simulator should pass with reduced wrong assumptions."""
        candidate = self._make_result(
            track_id="T13",
            arm_type="G",
            metric_values={
                "wrong_assumption_rate": 10.0,
                "adversarial_robustness_score": 75.0,
                "abstention_rate": 10.0,
                "fluency": 90.0,
            },
        )
        baseline = self._make_result(
            track_id="T13",
            arm_type="G",
            metric_values={
                "wrong_assumption_rate": 15.0,
                "adversarial_robustness_score": 70.0,
                "abstention_rate": 5.0,
                "fluency": 90.0,
            },
        )
        delta = {"adversarial_robustness_score": 5.0}
        
        assert evaluate_inverse_arm_pass(candidate, baseline, delta) is True


class TestInverseArmFailureFlags:
    """Tests for inverse arm failure flag computation."""
    
    def test_deliberation_collapse_excessive_postpass(self):
        """Deliberation Collapse should flag excessive postpass tokens."""
        runner = DeliberationCollapseRunner()
        spec = ExperimentSpec(
            id="test-t7-e1",
            track_id="T7",
            stage=1,
            hypothesis="Test",
            model_variant="INV-A-E1",
            baseline_id="test-baseline",
            train_budget_gpu_h=100.0,
            infer_budget_gpu_h=50.0,
            max_context=64000,
            datasets=["needle_32k"],
            metrics=["composite"],
            seeds=[101],
            promotion_gate={},
            params={"postpass_tokens": 150},  # Excessive
        )
        
        result = runner.execute(spec, 101, "test-run", "abc123")
        assert "excessive_postpass" in result.run_result.failure_flags
    
    def test_counterfactual_audit_excessive_k(self):
        """Counterfactual Audit should flag excessive counterfactuals."""
        runner = CounterfactualAuditRunner()
        spec = ExperimentSpec(
            id="test-t8-e1",
            track_id="T8",
            stage=1,
            hypothesis="Test",
            model_variant="INV-B-E1",
            baseline_id="test-baseline",
            train_budget_gpu_h=100.0,
            infer_budget_gpu_h=50.0,
            max_context=64000,
            datasets=["needle_32k"],
            metrics=["composite"],
            seeds=[101],
            promotion_gate={},
            params={"counterfactual_k": 10},  # Excessive
        )
        
        result = runner.execute(spec, 101, "test-run", "abc123")
        assert "excessive_counterfactuals" in result.run_result.failure_flags
    
    def test_noise_injection_excessive_samples(self):
        """Noise-Injection Ensemble should flag excessive samples."""
        runner = NoiseInjectionEnsembleRunner()
        spec = ExperimentSpec(
            id="test-t11-e1",
            track_id="T11",
            stage=1,
            hypothesis="Test",
            model_variant="INV-E-E1",
            baseline_id="test-baseline",
            train_budget_gpu_h=100.0,
            infer_budget_gpu_h=50.0,
            max_context=64000,
            datasets=["needle_32k"],
            metrics=["composite"],
            seeds=[101],
            promotion_gate={},
            params={"ensemble_samples": 10},  # Excessive
        )
        
        result = runner.execute(spec, 101, "test-run", "abc123")
        assert "excessive_ensemble_size" in result.run_result.failure_flags


class TestInverseArmCostAdjustment:
    """Tests for inverse arm cost adjustment."""
    
    def test_counterfactual_audit_increases_infer_cost(self):
        """Counterfactual Audit should increase inference cost."""
        runner = CounterfactualAuditRunner()
        spec = ExperimentSpec(
            id="test-t8-e1",
            track_id="T8",
            stage=1,
            hypothesis="Test",
            model_variant="INV-B-E1",
            baseline_id="test-baseline",
            train_budget_gpu_h=100.0,
            infer_budget_gpu_h=50.0,
            max_context=64000,
            datasets=["needle_32k"],
            metrics=["composite"],
            seeds=[101],
            promotion_gate={},
            params={"counterfactual_k": 3},
        )
        
        result = runner.execute(spec, 101, "test-run", "abc123")
        # With K=3, infer cost should be ~1.45x base
        assert result.run_result.infer_cost > 50.0 * 1.3
    
    def test_noise_injection_increases_infer_cost(self):
        """Noise-Injection Ensemble should increase inference cost."""
        runner = NoiseInjectionEnsembleRunner()
        spec = ExperimentSpec(
            id="test-t11-e1",
            track_id="T11",
            stage=1,
            hypothesis="Test",
            model_variant="INV-E-E1",
            baseline_id="test-baseline",
            train_budget_gpu_h=100.0,
            infer_budget_gpu_h=50.0,
            max_context=64000,
            datasets=["needle_32k"],
            metrics=["composite"],
            seeds=[101],
            promotion_gate={},
            params={"ensemble_samples": 3},
        )
        
        result = runner.execute(spec, 101, "test-run", "abc123")
        # With 3 samples, infer cost should be ~1.5x base
        assert result.run_result.infer_cost > 50.0 * 1.4
    
    def test_deliberation_collapse_reduces_latency(self):
        """Deliberation Collapse should reduce latency."""
        runner = DeliberationCollapseRunner()
        spec = ExperimentSpec(
            id="test-t7-e1",
            track_id="T7",
            stage=1,
            hypothesis="Test",
            model_variant="INV-A-E1",
            baseline_id="test-baseline",
            train_budget_gpu_h=100.0,
            infer_budget_gpu_h=50.0,
            max_context=64000,
            datasets=["needle_32k"],
            metrics=["composite"],
            seeds=[101],
            promotion_gate={},
            params={"postpass_tokens": 48},
        )
        
        result = runner.execute(spec, 101, "test-run", "abc123")
        # Latency should be reduced due to single-pass generation
        # Baseline latency is ~120ms, with -25% delta should be ~90ms
        assert result.run_result.latency_p50 < 110.0
