"""Generate adversarial experiment specs that probe failure boundaries.

Instead of testing at "reasonable" param values, adversarial specs
systematically probe ±20% around known failure boundaries to map the
exact failure surface before committing to production.

For each track, this script generates a sweep of specs around the
known failure threshold for the most sensitive parameter, producing
a failure map that the Stage 3 memo can reference.

Example output:
    specs/adversarial/t3_compression_sweep/t3_adv_cr085.yaml
    specs/adversarial/t3_compression_sweep/t3_adv_cr088.yaml
    specs/adversarial/t3_compression_sweep/t3_adv_cr090.yaml
    specs/adversarial/t3_compression_sweep/t3_adv_cr092.yaml
    specs/adversarial/t3_compression_sweep/t3_adv_cr095.yaml

Usage:
    python scripts/generate_adversarial_specs.py --track T3 --stage 3
    python scripts/generate_adversarial_specs.py --track all --stage 2
    python scripts/generate_adversarial_specs.py --list-boundaries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Known failure boundaries per track
# Derived from simulator._failure_flags() and TRACK_PASS_CRITERIA
# ---------------------------------------------------------------------------

FAILURE_BOUNDARIES: dict[str, dict[str, object]] = {
    "T1": {
        "param": "recalibration_interval",
        "failure_threshold": 2048,
        "failure_flag": "analog_drift",
        "sweep_values": [1024, 1536, 2048, 2304, 2560, 3072],
        "description": "Probe analog drift onset around recalibration_interval=2048",
        "baseline_params": {"noise_model": "analog_v1"},
    },
    "T2": {
        "param": "anchor_frequency",
        "failure_threshold": "1/2",
        "failure_flag": "reversibility_break",
        "sweep_values": ["1/8", "1/4", "1/2"],
        "description": "Probe reversibility break at anchor_frequency=1/2",
        "baseline_params": {"disable_norm_constraints": False},
    },
    "T3": {
        "param": "compression_ratio",
        "failure_threshold": 0.90,
        "failure_flag": "critical_fact_loss",
        "sweep_values": [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95],
        "description": "Map critical_fact_loss onset around compression_ratio=0.90",
        "baseline_params": {},
    },
    "T4": {
        "param": "role_permutation_noise",
        "failure_threshold": 0.70,
        "failure_flag": "entity_role_swap",
        "sweep_values": [0.10, 0.20, 0.40, 0.60, 0.70, 0.80, 0.90],
        "description": "Map entity_role_swap onset around role_permutation_noise=0.70",
        "baseline_params": {},
    },
    "T5": {
        "param": "max_nodes",
        "failure_threshold": 12,
        "failure_flag": "invalid_circuit",
        "sweep_values": [8, 10, 12, 14, 16, 18, 20],
        "description": "Map invalid_circuit onset around max_nodes=12",
        "baseline_params": {
            "typed_io_enforced": True,
            "deterministic_fallback": True,
            "planner_prune": True,
        },
    },
    "T6": {
        "param": "anneal_temp",
        "failure_threshold": 0.20,
        "failure_flag": "repetitive_text",
        "sweep_values": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55],
        "description": "Map repetitive_text onset around anneal_temp=0.20",
        "baseline_params": {},
    },
}

# Base spec template per track (stage 3 confirmatory context)
BASE_SPEC_TEMPLATE: dict[str, object] = {
    "track_id": "",
    "stage": 3,
    "hypothesis": "",
    "model_variant": "",
    "baseline_id": "",
    "train_budget_gpu_h": 100.0,
    "infer_budget_gpu_h": 20.0,
    "max_context": 128000,
    "datasets": [
        "needle_32k", "needle_64k", "needle_128k",
        "longbench", "gsm8k", "bbh", "consistency_longform",
    ],
    "metrics": [
        "long_context", "reasoning", "consistency",
        "fluency", "composite",
    ],
    "seeds": [101, 102, 103, 104, 105],
    "promotion_gate": {
        "next_stage": "adversarial_analysis",
        "delta_composite_min": 0.0,
        "purpose": "failure_boundary_mapping",
    },
    "params": {},
}

TRACK_VARIANTS: dict[str, str] = {
    "T1": "T1-E2",
    "T2": "T2-E2",
    "T3": "T3-E2",
    "T4": "T4-E2",
    "T5": "T5-E2",
    "T6": "T6-E2",
}

TRACK_METRICS: dict[str, list[str]] = {
    "T1": ["noise_robustness_auc", "latency_energy_proxy"],
    "T2": ["activation_memory_reduction_pct", "peak_memory_gb", "max_feasible_context"],
    "T3": ["token_access_reduction_pct", "critical_fact_miss_rate"],
    "T4": ["binding_error_rate", "compositional_gen_acc"],
    "T5": ["invalid_plan_rate", "run_variance"],
    "T6": ["contradiction_reduction_pct", "constraint_pass_gain_pct"],
}


def generate_adversarial_specs(
    track_id: str,
    stage: int,
    output_dir: Path,
    dry_run: bool = False,
) -> list[Path]:
    """Generate adversarial specs for a single track.

    Args:
        track_id: Track ID (e.g. 'T3').
        stage: Stage number for the specs.
        output_dir: Directory to write specs into.
        dry_run: If True, print specs but don't write files.

    Returns:
        List of paths to generated spec files.
    """
    if track_id not in FAILURE_BOUNDARIES:
        print(f"[WARN] No failure boundary defined for {track_id}. Skipping.")
        return []

    boundary: dict[str, object] = FAILURE_BOUNDARIES[track_id]
    param_name: str = str(boundary["param"])
    sweep_values: list[object] = list(boundary["sweep_values"])  # type: ignore[arg-type]
    description: str = str(boundary["description"])
    baseline_params: dict[str, object] = dict(boundary["baseline_params"])  # type: ignore[arg-type]

    variant = TRACK_VARIANTS.get(track_id, f"{track_id}-E2")
    extra_metrics = TRACK_METRICS.get(track_id, [])

    sweep_dir = output_dir / f"{track_id.lower()}_adv_{param_name}_sweep"
    if not dry_run:
        sweep_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for val in sweep_values:
        # Build safe filename from param value
        safe_val = str(val).replace("/", "div").replace(".", "p").replace("-", "neg")
        spec_id = f"adv-{track_id.lower()}-{param_name.replace('_', '-')}-{safe_val}-s{stage}"
        filename = f"{track_id.lower()}_adv_{safe_val}.yaml"
        filepath = sweep_dir / filename

        # Build params: baseline params + swept param
        params: dict[str, object] = dict(baseline_params)
        params[param_name] = val

        # Determine if this value is at/beyond the failure threshold
        threshold: object = boundary["failure_threshold"]
        at_boundary = _is_at_boundary(val, threshold)
        boundary_note = f" [AT/BEYOND FAILURE THRESHOLD {threshold}]" if at_boundary else ""

        spec: dict[str, object] = dict(BASE_SPEC_TEMPLATE)
        spec["id"] = spec_id
        spec["track_id"] = track_id
        spec["stage"] = stage
        spec["hypothesis"] = (
            f"Adversarial boundary probe: {description}. "
            f"Testing {param_name}={val}{boundary_note}. "
            f"Expected failure flag: '{boundary['failure_flag']}' "
            f"at/beyond threshold={threshold}."
        )
        spec["model_variant"] = variant
        spec["baseline_id"] = f"s{stage}-{track_id.lower()}-baseline"
        base_metrics = BASE_SPEC_TEMPLATE["metrics"]
        spec["metrics"] = list(base_metrics) + extra_metrics  # type: ignore[arg-type]
        spec["params"] = params
        spec["promotion_gate"] = {
            "next_stage": "adversarial_analysis",
            "delta_composite_min": 0.0,
            "purpose": "failure_boundary_mapping",
            "swept_param": param_name,
            "swept_value": val,
            "failure_threshold": threshold,
            "expected_failure_flag": boundary["failure_flag"],
        }

        if dry_run:
            print(f"\n--- {filepath} ---")
            print(json.dumps(spec, indent=2))
        else:
            filepath.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            print(f"  Written: {filepath}")

        generated.append(filepath)

    return generated


def _is_at_boundary(val: object, threshold: object) -> bool:
    """Check if a value is at or beyond the failure threshold."""
    try:
        return float(str(val)) >= float(str(threshold))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return str(val) == str(threshold)


def list_boundaries() -> None:
    """Print all known failure boundaries."""
    print("\nKnown Failure Boundaries:")
    print("=" * 70)
    for track_id, bnd in FAILURE_BOUNDARIES.items():
        print(f"\n{track_id}: {bnd['description']}")
        print(f"  Param:     {bnd['param']}")
        print(f"  Threshold: {bnd['failure_threshold']}")
        print(f"  Flag:      {bnd['failure_flag']}")
        print(f"  Sweep:     {bnd['sweep_values']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate adversarial experiment specs that probe failure boundaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--track",
        default="all",
        help="Track ID (T1–T6) or 'all'. Default: all.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        help="Stage number for generated specs. Default: 3.",
    )
    parser.add_argument(
        "--output-dir",
        default="specs/adversarial",
        help="Output directory for generated specs. Default: specs/adversarial.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print specs to stdout without writing files.",
    )
    parser.add_argument(
        "--list-boundaries",
        action="store_true",
        help="List all known failure boundaries and exit.",
    )

    args = parser.parse_args()

    if args.list_boundaries:
        list_boundaries()
        return 0

    output_dir = Path(args.output_dir)
    tracks: list[str] = list(FAILURE_BOUNDARIES.keys()) if args.track == "all" else [args.track.upper()]

    total = 0
    for track_id in tracks:
        print(f"\nGenerating adversarial specs for {track_id} (Stage {args.stage})...")
        paths = generate_adversarial_specs(
            track_id=track_id,
            stage=args.stage,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        total += len(paths)
        print(f"  {len(paths)} specs generated for {track_id}.")

    print(f"\nTotal: {total} adversarial specs generated.")
    if not args.dry_run:
        print(f"Output directory: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
