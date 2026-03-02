"""Tests for exp.effect_surface — Gaussian Process Effect Surface."""

from __future__ import annotations

import math
from exp.effect_surface import (
    GPEffectSurface,
    GPHyperparameters,
    PredictionResult,
    AcquisitionResult,
    gp_predict_with_prior,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_run(
    composite: float,
    params: dict,
) -> dict:
    return {
        "params": params,
        "metric_values": {"composite": composite},
    }


# ---------------------------------------------------------------------------
# GPEffectSurface
# ---------------------------------------------------------------------------

class TestGPEffectSurface:
    def test_init_creates_empty_surface(self) -> None:
        surface = GPEffectSurface()
        assert "T1" in surface._training_data
        assert "T3" in surface._training_data

    def test_hyperparameters_defaults(self) -> None:
        surface = GPEffectSurface()
        assert surface.hyperparameters.length_scale == 1.0
        assert surface.hyperparameters.signal_variance == 1.0
        assert surface.hyperparameters.noise_variance == 0.1

    def test_param_to_features_t3(self) -> None:
        surface = GPEffectSurface()
        features = surface._param_to_features("T3", {"compression_ratio": 0.85})
        assert 0.85 in features

    def test_param_to_features_t4(self) -> None:
        surface = GPEffectSurface()
        features = surface._param_to_features("T4", {"role_permutation_noise": 0.2})
        assert 0.2 in features

    def test_update_from_runs_stores_data(self) -> None:
        surface = GPEffectSurface()
        runs = [
            _make_mock_run(65.0, {"compression_ratio": 0.70}),
            _make_mock_run(68.0, {"compression_ratio": 0.85}),
        ]
        surface.update_from_runs("T3", runs)
        assert len(surface._training_data["T3"]) == 2

    def test_predict_returns_prediction_result(self) -> None:
        surface = GPEffectSurface()
        runs = [
            _make_mock_run(65.0, {"compression_ratio": 0.70}),
            _make_mock_run(68.0, {"compression_ratio": 0.85}),
        ]
        surface.update_from_runs("T3", runs)

        pred = surface.predict("T3", {"compression_ratio": 0.80})

        assert isinstance(pred, PredictionResult)
        assert hasattr(pred, "mean")
        assert hasattr(pred, "std")
        assert hasattr(pred, "lower_95")
        assert hasattr(pred, "upper_95")
        assert hasattr(pred, "confidence")

    def test_predict_with_no_data_uses_prior(self) -> None:
        surface = GPEffectSurface()
        # No training data for T3
        pred = surface.predict("T3", {"compression_ratio": 0.80})

        # Should return prior with high uncertainty
        assert pred.std > 1.0
        assert pred.confidence < 0.5

    def test_prediction_with_data_has_reasonable_uncertainty(self) -> None:
        surface = GPEffectSurface()
        runs = [
            _make_mock_run(65.0, {"compression_ratio": 0.70}),
            _make_mock_run(68.0, {"compression_ratio": 0.85}),
            _make_mock_run(66.0, {"compression_ratio": 0.75}),
            _make_mock_run(67.0, {"compression_ratio": 0.80}),
        ]
        surface.update_from_runs("T3", runs)

        pred = surface.predict("T3", {"compression_ratio": 0.80})

        # With 4 data points, uncertainty should be reasonable
        assert pred.std < 3.0
        assert pred.confidence > 0.1

    def test_confidence_increases_with_more_data(self) -> None:
        surface = GPEffectSurface()

        # Few points
        runs_few = [
            _make_mock_run(65.0, {"compression_ratio": 0.70}),
            _make_mock_run(68.0, {"compression_ratio": 0.85}),
        ]
        surface.update_from_runs("T3", runs_few)
        pred_few = surface.predict("T3", {"compression_ratio": 0.80})

        # Many points
        runs_many = runs_few + [
            _make_mock_run(66.0, {"compression_ratio": 0.75}),
            _make_mock_run(67.0, {"compression_ratio": 0.78}),
            _make_mock_run(65.5, {"compression_ratio": 0.72}),
            _make_mock_run(67.5, {"compression_ratio": 0.82}),
        ]
        surface.update_from_runs("T3", runs_many)
        pred_many = surface.predict("T3", {"compression_ratio": 0.80})

        # More data should give higher confidence
        assert pred_many.confidence >= pred_few.confidence

    def test_normalize_features(self) -> None:
        surface = GPEffectSurface()
        surface._feature_mins["T3"] = [0.5, 0.0]
        surface._feature_maxs["T3"] = [1.0, 100.0]

        # Within range
        normalized = surface._normalize_features("T3", [0.75, 50.0])
        assert normalized[0] == 0.5  # (0.75-0.5)/(1.0-0.5) = 0.5
        assert normalized[1] == 0.5  # (50-0)/(100-0) = 0.5

        # Outside range should clip
        normalized_clip = surface._normalize_features("T3", [0.0, 200.0])
        assert normalized_clip[0] == 0.0


class TestAcquisitionFunction:
    def test_acquisition_returns_result(self) -> None:
        surface = GPEffectSurface()
        runs = [
            _make_mock_run(65.0, {"compression_ratio": 0.70}),
            _make_mock_run(68.0, {"compression_ratio": 0.85}),
        ]
        surface.update_from_runs("T3", runs)

        result = surface.acquisition(track_id="T3", budget_hours=100.0)

        assert isinstance(result, AcquisitionResult)
        assert hasattr(result, "recommended_params")
        assert hasattr(result, "expected_improvement")
        assert hasattr(result, "uncertainty")
        assert hasattr(result, "rationale")

    def test_acquisition_returns_params(self) -> None:
        surface = GPEffectSurface()
        runs = [
            _make_mock_run(65.0, {"compression_ratio": 0.70}),
            _make_mock_run(68.0, {"compression_ratio": 0.85}),
        ]
        surface.update_from_runs("T3", runs)

        result = surface.acquisition(track_id="T3", budget_hours=100.0)

        assert isinstance(result.recommended_params, dict)


class TestSerialization:
    def test_to_dict_serializes(self) -> None:
        surface = GPEffectSurface(GPHyperparameters(length_scale=2.0))
        d = surface.to_dict()

        assert "hyperparameters" in d
        assert d["hyperparameters"]["length_scale"] == 2.0
        assert "training_counts" in d
        assert "T1" in d["training_counts"]

    def test_from_dict_deserializes(self) -> None:
        data = {
            "hyperparameters": {
                "length_scale": 1.5,
                "signal_variance": 2.0,
                "noise_variance": 0.05,
            },
            "feature_mins": {"T3": [0.5]},
            "feature_maxs": {"T3": [1.0]},
        }
        surface = GPEffectSurface.from_dict(data)

        assert surface.hyperparameters.length_scale == 1.5
        assert surface.hyperparameters.signal_variance == 2.0
        assert surface.hyperparameters.noise_variance == 0.05


class TestGPWithPrior:
    def test_gp_predict_with_prior_returns_tuple(self) -> None:
        surface = GPEffectSurface()
        mean, std = gp_predict_with_prior(
            track_id="T3",
            params={"compression_ratio": 0.80},
            stage=1,
            surface=surface,
        )

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert not math.isnan(mean)
        assert not math.isnan(std)

    def test_gp_predict_with_prior_no_surface(self) -> None:
        # Should work even without surface (creates default)
        mean, std = gp_predict_with_prior(
            track_id="T3",
            params={"compression_ratio": 0.80},
            stage=1,
            surface=None,
        )

        assert isinstance(mean, float)
        assert isinstance(std, float)


class TestPredictionResult:
    def test_prediction_has_valid_ci(self) -> None:
        pred = PredictionResult(
            mean=5.0,
            std=1.0,
            lower_95=3.04,
            upper_95=6.96,
            confidence=0.8,
        )

        assert pred.lower_95 < pred.mean < pred.upper_95
        assert pred.confidence > 0 and pred.confidence <= 1

    def test_prediction_high_confidence_narrow_ci(self) -> None:
        pred_high_conf = PredictionResult(
            mean=5.0,
            std=0.5,
            lower_95=4.02,
            upper_95=5.98,
            confidence=0.9,
        )
        pred_low_conf = PredictionResult(
            mean=5.0,
            std=2.0,
            lower_95=1.08,
            upper_95=8.92,
            confidence=0.2,
        )

        # High confidence should have narrower CI
        ci_width_high = pred_high_conf.upper_95 - pred_high_conf.lower_95
        ci_width_low = pred_low_conf.upper_95 - pred_low_conf.lower_95
        assert ci_width_high < ci_width_low

