"""Tests for exp.dashboard — Real-Time Dashboard."""

from __future__ import annotations

from exp.dashboard import (
    Dashboard,
    RunStatus,
    StageStatus,
    DashboardSnapshot,
)


# ---------------------------------------------------------------------------
# Test Data Helpers
# ---------------------------------------------------------------------------

def _make_run_status(
    run_id: str,
    track_id: str,
    stage: int,
    status: str = "completed",
    composite: float | None = 60.0,
    failure_flags: list[str] | None = None,
) -> RunStatus:
    return RunStatus(
        run_id=run_id,
        spec_id=f"spec-{run_id}",
        track_id=track_id,
        stage=stage,
        status=status,
        composite=composite,
        failure_flags=failure_flags or [],
        start_time=None,
        end_time=None,
    )


# ---------------------------------------------------------------------------
# RunStatus
# ---------------------------------------------------------------------------

class TestRunStatus:
    def test_create_run_status(self) -> None:
        status = _make_run_status("run-1", "T1", stage=1)
        assert status.run_id == "run-1"
        assert status.track_id == "T1"
        assert status.stage == 1
        assert status.status == "completed"

    def test_run_status_default_composite(self) -> None:
        status = _make_run_status("run-1", "T1", stage=1, composite=None)
        assert status.composite is None


# ---------------------------------------------------------------------------
# StageStatus
# ---------------------------------------------------------------------------

class TestStageStatus:
    def test_create_stage_status(self) -> None:
        status = StageStatus(
            stage=1,
            total_runs=10,
            completed_runs=8,
            failed_runs=2,
            pending_runs=0,
            best_composite=72.5,
            pass_rate=0.75,
            gate_decision=None,
        )
        assert status.stage == 1
        assert status.total_runs == 10
        assert status.best_composite == 72.5


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class TestDashboardBasics:
    def test_init_creates_empty_dashboard(self) -> None:
        dashboard = Dashboard()
        assert dashboard.run_dir == dashboard.run_dir
        assert len(dashboard._runs) == 0

    def test_init_with_custom_dirs(self) -> None:
        dashboard = Dashboard(run_dir="/custom/runs", comparison_dir="/custom/comparisons")
        assert dashboard.run_dir == dashboard.run_dir


class TestDataLoading:
    def test_load_runs_nonexistent_dir(self) -> None:
        dashboard = Dashboard(run_dir="/nonexistent/path")
        dashboard._load_runs()
        # Should handle gracefully
        assert len(dashboard._runs) == 0

    def test_load_comparisons_nonexistent_dir(self) -> None:
        dashboard = Dashboard(comparison_dir="/nonexistent/path")
        dashboard._load_comparisons()
        # Should handle gracefully
        assert len(dashboard._comparisons) == 0


class TestStageStatus:
    def test_get_stage_status_empty(self) -> None:
        dashboard = Dashboard()
        status = dashboard.get_stage_status(1)

        assert status.stage == 1
        assert status.total_runs == 0
        assert status.best_composite is None
        assert status.pass_rate is None

    def test_get_stage_status_with_runs(self) -> None:
        dashboard = Dashboard()
        dashboard._runs = {
            "run-1": _make_run_status("run-1", "T1", stage=1, composite=65.0),
            "run-2": _make_run_status("run-2", "T2", stage=1, composite=68.0),
            "run-3": _make_run_status("run-3", "T3", stage=1, composite=70.0, failure_flags=["error"]),
        }

        status = dashboard.get_stage_status(1)

        assert status.total_runs == 3
        assert status.completed_runs == 3
        assert status.best_composite == 70.0

    def test_get_stage_status_only_failed(self) -> None:
        dashboard = Dashboard()
        dashboard._runs = {
            "run-1": _make_run_status("run-1", "T1", stage=1, status="failed", composite=None),
        }

        status = dashboard.get_stage_status(1)

        assert status.completed_runs == 0
        assert status.failed_runs == 1


class TestAllStagesStatus:
    def test_get_all_stages_status(self) -> None:
        dashboard = Dashboard()
        dashboard._runs = {
            "run-1": _make_run_status("run-1", "T1", stage=1),
            "run-2": _make_run_status("run-2", "T2", stage=2),
            "run-3": _make_run_status("run-3", "T3", stage=3),
        }

        all_status = dashboard.get_all_stages_status()

        assert 1 in all_status
        assert 2 in all_status
        assert 3 in all_status
        assert 4 in all_status  # Even with no runs


class TestAlerts:
    def test_check_alerts_no_issues(self) -> None:
        dashboard = Dashboard()
        dashboard._runs = {
            "run-1": _make_run_status("run-1", "T1", stage=1),
        }

        alerts = dashboard.check_alerts()

        # Should have success alert for high score
        assert any(a["type"] == "high_score" for a in alerts)

    def test_check_alerts_failed_runs(self) -> None:
        dashboard = Dashboard()
        dashboard._runs = {
            "run-1": _make_run_status("run-1", "T1", stage=1, status="failed"),
        }

        alerts = dashboard.check_alerts()

        assert any(a["type"] == "failed_runs" for a in alerts)


class TestSummary:
    def test_get_summary_empty(self) -> None:
        dashboard = Dashboard()
        summary = dashboard.get_summary()

        assert summary["total_runs"] == 0
        assert summary["completed_runs"] == 0


class TestLeaderboard:
    def test_get_track_leaderboard_empty(self) -> None:
        dashboard = Dashboard()
        leaderboard = dashboard.get_track_leaderboard()

        assert leaderboard == []


class TestRendering:
    def test_render_html_returns_string(self) -> None:
        dashboard = Dashboard()
        html = dashboard.render_html()

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Engine Experiment Dashboard" in html

    def test_render_json_returns_string(self) -> None:
        dashboard = Dashboard()
        json_str = dashboard.render_json()

        assert isinstance(json_str, str)
        assert "timestamp" in json_str


class TestSnapshot:
    def test_get_snapshot_returns_snapshot(self) -> None:
        dashboard = Dashboard()
        snapshot = dashboard.get_snapshot()

        assert isinstance(snapshot, DashboardSnapshot)
        assert isinstance(snapshot.stages, dict)
        assert isinstance(snapshot.alerts, list)


class TestEdgeCases:
    def test_refresh_handles_errors(self) -> None:
        dashboard = Dashboard(run_dir="/invalid/path")
        # Should not raise
        dashboard.refresh()

    def test_stage_status_with_all_pending(self) -> None:
        dashboard = Dashboard()
        dashboard._runs = {
            "run-1": _make_run_status("run-1", "T1", stage=1, status="pending"),
        }

        status = dashboard.get_stage_status(1)

        assert status.pending_runs == 1
        assert status.completed_runs == 0

