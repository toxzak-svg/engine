from __future__ import annotations

import json
from pathlib import Path


TRACK_HYPOTHESES = {
    "T1": "Hybrid photonic-digital acceleration enables longer effective context and ensemble gains at equal cost.",
    "T2": "Mostly reversible blocks remove activation-memory bottlenecks for deeper/longer-context training.",
    "T3": "Compression-first hierarchical memory scales context while preserving task-critical details via uncertainty-aware zoom.",
    "T4": "Vector-symbolic scratchpad operations improve compositional variable/entity handling.",
    "T5": "Self-assembled constrained neural circuits improve specialization and reliability over fixed graphs.",
    "T6": "Energy-based global decoding improves long-range coherence and constraint satisfaction.",
}

TRACK_METRICS = {
    "T1": ["long_context", "reasoning", "consistency", "fluency", "composite", "noise_robustness_auc", "latency_energy_proxy"],
    "T2": [
        "long_context",
        "reasoning",
        "consistency",
        "fluency",
        "composite",
        "activation_memory_reduction_pct",
        "peak_memory_gb",
        "max_feasible_context",
    ],
    "T3": [
        "long_context",
        "reasoning",
        "consistency",
        "fluency",
        "composite",
        "token_access_reduction_pct",
        "critical_fact_miss_rate",
    ],
    "T4": [
        "long_context",
        "reasoning",
        "consistency",
        "fluency",
        "composite",
        "binding_error_rate",
        "compositional_gen_acc",
    ],
    "T5": ["long_context", "reasoning", "consistency", "fluency", "composite", "invalid_plan_rate", "run_variance"],
    "T6": [
        "long_context",
        "reasoning",
        "consistency",
        "fluency",
        "composite",
        "contradiction_reduction_pct",
        "constraint_pass_gain_pct",
    ],
}

EXPERIMENT_PARAMS = {
    "T1": {
        "E1": {"noise_model": "analog_v1"},
        "E2": {"recalibration_interval": 512},
        "E3": {"recalibration_interval": 512, "samples": 4},
    },
    "T2": {
        "E1": {"anchor_frequency": "1/2"},
        "E2": {"anchor_frequency": "1/4"},
        "E3": {"anchor_frequency": "1/8"},
    },
    "T3": {
        "E1": {"compression_ratio": 0.65},
        "E2": {"compression_ratio": 0.75, "zoom_uncertainty_threshold": 0.25},
        "E3": {"compression_ratio": 0.85, "critical_fact_stress_test": True},
    },
    "T4": {
        "E1": {"role_permutation_noise": 0.25},
        "E2": {"role_permutation_noise": 0.2, "scratchpad_consistency_loss": 0.15},
        "E3": {"role_permutation_noise": 0.3, "ood_role_permutation_eval": True},
    },
    "T5": {
        "E1": {"max_nodes": 8},
        "E2": {"max_nodes": 12, "typed_io_enforced": True},
        "E3": {"max_nodes": 12, "deterministic_fallback": True},
    },
    "T6": {
        "E1": {"anneal_temp": 0.7},
        "E2": {"anneal_temp": 0.55},
        "E3": {"anneal_temp": 0.4},
    },
}

STAGE_SETTINGS = {
    0: {
        "train_budget_gpu_h": 40.0,
        "infer_budget_gpu_h": 10.0,
        "max_context": 128000,
        "seeds": [11],
        "promotion_gate": {"target": "harness-freeze"},
    },
    1: {
        "train_budget_gpu_h": 120.0,
        "infer_budget_gpu_h": 30.0,
        "max_context": 128000,
        "seeds": [101, 102, 103],
        "promotion_gate": {
            "next_stage": 2,
            "delta_composite_min": 3.0,
            "stable_training_required": True,
            "max_fluency_drop_pct": 2.0,
        },
    },
    2: {
        "train_budget_gpu_h": 500.0,
        "infer_budget_gpu_h": 100.0,
        "max_context": 256000,
        "seeds": [201, 202],
        "promotion_gate": {
            "next_stage": 3,
            "delta_composite_min": 5.0,
            "max_latency_overhead_pct": 15.0,
        },
    },
    3: {
        "train_budget_gpu_h": 1000.0,
        "infer_budget_gpu_h": 200.0,
        "max_context": 1000000,
        "seeds": [301, 302],
        "promotion_gate": {
            "next_stage": "final",
            "delta_composite_min": 8.0,
            "bootstrap_ci_excludes_zero": True,
        },
    },
}

DATASETS = ["needle_32k", "needle_64k", "needle_128k", "longbench", "gsm8k", "bbh", "consistency_longform"]


def build_spec(track_id: str, stage: int, variant: str, baseline_id: str, params: dict) -> dict:
    settings = STAGE_SETTINGS[stage]
    return {
        "id": f"s{stage}-{track_id.lower()}-{variant.lower()}",
        "track_id": track_id,
        "stage": stage,
        "hypothesis": TRACK_HYPOTHESES[track_id],
        "model_variant": f"{track_id}-{variant}" if variant != "BASELINE" else "BASELINE",
        "baseline_id": baseline_id,
        "train_budget_gpu_h": settings["train_budget_gpu_h"],
        "infer_budget_gpu_h": settings["infer_budget_gpu_h"],
        "max_context": settings["max_context"],
        "datasets": DATASETS,
        "metrics": TRACK_METRICS[track_id],
        "seeds": settings["seeds"],
        "promotion_gate": settings["promotion_gate"],
        "params": params,
    }


def write_spec(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def main() -> None:
    for stage in [0, 1, 2, 3]:
        stage_dir = Path("specs") / f"stage{stage}"
        for track_id in TRACK_HYPOTHESES:
            baseline_id = f"s{stage}-{track_id.lower()}-baseline"
            baseline_payload = build_spec(track_id, stage, "BASELINE", baseline_id=baseline_id, params={})
            write_spec(stage_dir / f"{track_id.lower()}_baseline.yaml", baseline_payload)
            if stage == 0:
                continue
            for variant, params in EXPERIMENT_PARAMS[track_id].items():
                payload = build_spec(track_id, stage, variant, baseline_id=baseline_id, params=params)
                write_spec(stage_dir / f"{track_id.lower()}_{variant.lower()}.yaml", payload)


if __name__ == "__main__":
    main()
