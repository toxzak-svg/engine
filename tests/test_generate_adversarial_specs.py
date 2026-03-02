"""Tests for scripts.generate_adversarial_specs — CLI smoke test and per-track spec output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.generate_adversarial_specs import (
    FAILURE_BOUNDARIES,
    TRACK_METRICS,
    TRACK_VARIANTS,
    _is_at_boundary,
    generate_adversarial_specs,
    list_boundaries,
)


# ---------------------------------------------------------------------------
# _is_at_boundary
# ---------------------------------------------------------------------------

class TestIsAtBoundary:
    def test_value_below_threshold_returns_false(self) -> None:
        assert _is_at_boundary(0.80, 0.90) is False

    def test_value_at_threshold_returns_true(self) -> None:
        assert _is_at_boundary(0.90, 0.90) is True

    def test_value_above_threshold_returns_true(self) -> None:
        assert _is_at_boundary(0.95, 0.90) is True

    def test_integer_comparison(self) -> None:
        assert _is_at_boundary(2048, 2048) is True
        assert _is_at_boundary(1024, 2048) is False

    def test_string_equality_fallback(self) -> None:
        # Non-numeric strings fall back to str equality
        assert _is_at_boundary("1/2", "1/2") is True
        assert _is_at_boundary("1/4", "1/2") is False


# ---------------------------------------------------------------------------
# generate_adversarial_specs — file output per track
# ---------------------------------------------------------------------------

class TestGenerateAdversarialSpecs:
    def test_t3_generates_correct_number_of_specs(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs(
            track_id="T3",
            stage=3,
            output_dir=tmp_path,
        )
        expected_count = len(FAILURE_BOUNDARIES["T3"]["sweep_values"])
        assert len(paths) == expected_count

    def test_t3_creates_output_directory(self, tmp_path: Path) -> None:
        generate_adversarial_specs("T3", 3, tmp_path)
        sweep_dir = tmp_path / "t3_adv_compression_ratio_sweep"
        assert sweep_dir.is_dir()

    def test_t3_files_are_valid_json(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        for p in paths:
            assert p.suffix == ".yaml"
            text = p.read_text(encoding="utf-8")
            spec = json.loads(text)
            assert isinstance(spec, dict)

    def test_t3_spec_contains_required_fields(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert spec["track_id"] == "T3"
            assert spec["stage"] == 3
            assert "id" in spec
            assert "params" in spec
            assert "promotion_gate" in spec
            assert "hypothesis" in spec

    def test_t3_spec_swept_param_embedded_in_params(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        sweep_values = list(FAILURE_BOUNDARIES["T3"]["sweep_values"])
        found_values = set()
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            cr = spec["params"].get("compression_ratio")
            assert cr in sweep_values, f"Unexpected value: {cr}"
            found_values.add(cr)
        assert found_values == set(sweep_values)

    def test_t3_boundary_specs_have_boundary_note_in_hypothesis(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        threshold = FAILURE_BOUNDARIES["T3"]["failure_threshold"]
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            cr = spec["params"]["compression_ratio"]
            if cr >= threshold:
                assert "AT/BEYOND FAILURE THRESHOLD" in spec["hypothesis"]

    def test_t3_extra_metrics_appended(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        extra = TRACK_METRICS["T3"]
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            for m in extra:
                assert m in spec["metrics"]

    def test_t3_model_variant_is_correct(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        expected_variant = TRACK_VARIANTS["T3"]
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert spec["model_variant"] == expected_variant

    def test_t3_promotion_gate_has_swept_param_metadata(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            gate = spec["promotion_gate"]
            assert gate["swept_param"] == "compression_ratio"
            assert gate["expected_failure_flag"] == FAILURE_BOUNDARIES["T3"]["failure_flag"]
            assert "swept_value" in gate
            assert "failure_threshold" in gate

    # Per-track sweep coverage

    def test_t1_generates_specs_for_recalibration_interval(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T1", 2, tmp_path)
        assert len(paths) == len(FAILURE_BOUNDARIES["T1"]["sweep_values"])
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert spec["track_id"] == "T1"
            assert "recalibration_interval" in spec["params"]

    def test_t2_generates_specs_for_anchor_frequency(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T2", 2, tmp_path)
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert "anchor_frequency" in spec["params"]
            assert spec["params"]["disable_norm_constraints"] is False

    def test_t4_generates_specs_for_role_permutation_noise(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T4", 3, tmp_path)
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert "role_permutation_noise" in spec["params"]

    def test_t5_baseline_params_present(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T5", 3, tmp_path)
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert spec["params"]["typed_io_enforced"] is True
            assert spec["params"]["deterministic_fallback"] is True
            assert spec["params"]["planner_prune"] is True

    def test_t5_max_nodes_sweep_progresses_over_threshold(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T5", 3, tmp_path)
        threshold = FAILURE_BOUNDARIES["T5"]["failure_threshold"]
        at_or_beyond = [
            p for p in paths
            if json.loads(p.read_text(encoding="utf-8"))["params"]["max_nodes"] >= threshold
        ]
        below = [
            p for p in paths
            if json.loads(p.read_text(encoding="utf-8"))["params"]["max_nodes"] < threshold
        ]
        assert len(at_or_beyond) > 0, "Should have specs at/beyond threshold"
        assert len(below) > 0, "Should have specs below threshold"

    def test_t6_generates_specs_for_anneal_temp(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T6", 3, tmp_path)
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert "anneal_temp" in spec["params"]

    def test_unknown_track_generates_no_specs(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T_UNKNOWN", 3, tmp_path)
        assert paths == []

    def test_dry_run_does_not_write_files(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path, dry_run=True)
        # dry_run returns paths (planned, not written)
        assert len(paths) == len(FAILURE_BOUNDARIES["T3"]["sweep_values"])
        sweep_dir = tmp_path / "t3_adv_compression_ratio_sweep"
        assert not sweep_dir.exists()

    def test_spec_ids_are_unique_within_track(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 3, tmp_path)
        specs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        ids = [s["id"] for s in specs]
        assert len(ids) == len(set(ids))

    def test_stage_number_embedded_in_spec_id(self, tmp_path: Path) -> None:
        paths = generate_adversarial_specs("T3", 2, tmp_path)
        for p in paths:
            spec = json.loads(p.read_text(encoding="utf-8"))
            assert "-s2" in spec["id"]

    def test_all_tracks_generate_without_error(self, tmp_path: Path) -> None:
        total = 0
        for track_id in FAILURE_BOUNDARIES:
            paths = generate_adversarial_specs(track_id, 3, tmp_path)
            assert len(paths) > 0, f"No specs generated for {track_id}"
            total += len(paths)
        assert total > 0


# ---------------------------------------------------------------------------
# list_boundaries — smoke test
# ---------------------------------------------------------------------------

class TestListBoundaries:
    def test_list_boundaries_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        list_boundaries()
        captured = capsys.readouterr()
        assert "Failure Boundaries" in captured.out
        for track_id in FAILURE_BOUNDARIES:
            assert track_id in captured.out


# ---------------------------------------------------------------------------
# CLI smoke tests via subprocess
# ---------------------------------------------------------------------------

class TestCLISmokeTest:
    def test_list_boundaries_flag_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/generate_adversarial_specs.py", "--list-boundaries"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "Failure Boundaries" in result.stdout

    def test_dry_run_single_track_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_adversarial_specs.py",
                "--track", "T3",
                "--stage", "3",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "T3" in result.stdout

    def test_dry_run_single_track_outputs_spec_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_adversarial_specs.py",
                "--track", "T3",
                "--stage", "3",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        # Output should contain the swept param name
        assert "compression_ratio" in result.stdout

    def test_dry_run_all_tracks_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_adversarial_specs.py",
                "--track", "all",
                "--stage", "3",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        # Each track should be mentioned in output
        for track_id in FAILURE_BOUNDARIES:
            assert track_id in result.stdout

    def test_output_dir_creates_files(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_adversarial_specs.py",
                "--track", "T3",
                "--stage", "3",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        sweep_dir = tmp_path / "t3_adv_compression_ratio_sweep"
        yaml_files = list(sweep_dir.glob("*.yaml"))
        assert len(yaml_files) == len(FAILURE_BOUNDARIES["T3"]["sweep_values"])

    def test_invalid_track_still_exits_zero(self) -> None:
        # Unknown tracks print a warning but don't crash
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_adversarial_specs.py",
                "--track", "T_INVALID",
                "--stage", "3",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "WARN" in result.stdout or "Total: 0" in result.stdout
