"""Real-Time Dashboard — Live experiment monitoring and visualization.

Provides:
    - Live run status tracking
    - Stage gate progress visualization
    - Metric streaming updates
    - Comparison charts
    - Alert notifications

Usage:
    dashboard = Dashboard()
    
    # Start monitoring
    dashboard.watch_directory("artifacts/runs")
    dashboard.watch_directory("artifacts/comparisons")
    
    # Get status
    status = dashboard.get_stage_status(stage=2)
    
    # Render HTML
    html = dashboard.render_html()
    
    # Start web server (optional)
    dashboard.serve(port=8080)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from exp.gating import gate_stage
from exp.models import ComparisonReport, RunResult


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class RunStatus:
    """Status of a single run."""
    run_id: str
    spec_id: str | None
    track_id: str
    stage: int
    status: str  # pending, running, completed, failed
    composite: float | None
    failure_flags: list[str]
    start_time: datetime | None
    end_time: datetime | None


@dataclass
class StageStatus:
    """Status of all runs in a stage."""
    stage: int
    total_runs: int
    completed_runs: int
    failed_runs: int
    pending_runs: int
    best_composite: float | None
    pass_rate: float | None
    gate_decision: dict[str, Any] | None


@dataclass
class DashboardSnapshot:
    """Complete dashboard state."""
    timestamp: datetime
    stages: dict[int, StageStatus]
    recent_comparisons: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Dashboard Implementation
# ---------------------------------------------------------------------------


class Dashboard:
    """Real-time experiment dashboard.

    Monitors artifact directories and provides:
    - Live status updates
    - Stage progress tracking
    - Gate decision preview
    - HTML/JSON rendering
    """

    def __init__(
        self,
        run_dir: str | Path = "artifacts/runs",
        comparison_dir: str | Path = "artifacts/comparisons",
    ):
        self.run_dir = Path(run_dir)
        self.comparison_dir = Path(comparison_dir)

        # Cached state
        self._runs: dict[str, RunStatus] = {}
        self._comparisons: list[ComparisonReport] = []
        self._last_update: datetime = datetime.now(timezone.utc)

        # Watchers
        self._watchers: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Callbacks
        self._on_update: list[Callable[[], None]] = []
        self._on_alert: list[Callable[[str, str], None]] = []  # (title, message)

    # ---------------------------------------------------------------------------
    # Data Loading
    # ---------------------------------------------------------------------------

    def refresh(self) -> None:
        """Refresh all data from disk."""
        self._load_runs()
        self._load_comparisons()
        self._last_update = datetime.now(timezone.utc)

    def _load_runs(self) -> None:
        """Load runs from artifact directory."""
        if not self.run_dir.exists():
            return

        self._runs.clear()

        for json_file in self.run_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                if "run_id" in data:
                    run = RunResult.from_dict(data)
                    self._runs[run.run_id] = RunStatus(
                        run_id=run.run_id,
                        spec_id=run.spec_id,
                        track_id=run.track_id,
                        stage=run.stage,
                        status="completed",  # If in artifacts, completed
                        composite=run.metric_values.get("composite"),
                        failure_flags=run.failure_flags,
                        start_time=None,  # Would need to track separately
                        end_time=None,
                    )
            except Exception:
                pass  # Skip invalid files

    def _load_comparisons(self) -> None:
        """Load comparisons from artifact directory."""
        if not self.comparison_dir.exists():
            return

        self._comparisons.clear()

        for json_file in self.comparison_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                if "candidate_run_ids" in data:
                    report = ComparisonReport.from_dict(data)
                    self._comparisons.append(report)
            except Exception:
                pass  # Skip invalid files

    # ---------------------------------------------------------------------------
    # Status Queries
    # ---------------------------------------------------------------------------

    def get_stage_status(self, stage: int) -> StageStatus:
        """Get status for a specific stage."""
        stage_runs = [r for r in self._runs.values() if r.stage == stage]

        if not stage_runs:
            return StageStatus(
                stage=stage,
                total_runs=0,
                completed_runs=0,
                failed_runs=0,
                pending_runs=0,
                best_composite=None,
                pass_rate=None,
                gate_decision=None,
            )

        completed = [r for r in stage_runs if r.status == "completed"]
        failed = [r for r in stage_runs if r.status == "failed"]
        pending = [r for r in stage_runs if r.status == "pending"]

        composites = [r.composite for r in completed if r.composite is not None]
        best_composite = max(composites) if composites else None

        passed = [r for r in completed if not r.failure_flags]
        pass_rate = len(passed) / len(completed) if completed else None

        # Get gate decision if we have comparisons
        stage_comparisons = [c for c in self._comparisons if c.candidate_stage == stage]
        gate_decision = None
        if stage_comparisons and stage in {1, 2, 3, 4}:
            try:
                gate_decision = gate_stage(stage, stage_comparisons)
            except Exception:
                pass

        return StageStatus(
            stage=stage,
            total_runs=len(stage_runs),
            completed_runs=len(completed),
            failed_runs=len(failed),
            pending_runs=len(pending),
            best_composite=best_composite,
            pass_rate=pass_rate,
            gate_decision=gate_decision,
        )

    def get_all_stages_status(self) -> dict[int, StageStatus]:
        """Get status for all stages."""
        return {stage: self.get_stage_status(stage) for stage in range(5)}

    def get_recent_comparisons(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most recent comparisons."""
        sorted_comps = sorted(
            self._comparisons,
            key=lambda c: c.candidate_run_ids[0] if c.candidate_run_ids else "",
            reverse=True,
        )

        return [
            {
                "track_id": c.track_id,
                "stage": c.candidate_stage,
                "delta_composite": c.delta_metrics.get("composite", 0),
                "anchor_delta": c.anchor_delta_metrics.get("composite", 0),
                "overall_pass": c.pass_fail.get("overall_pass", False),
            }
            for c in sorted_comps[:limit]
        ]

    def get_track_leaderboard(self) -> list[dict[str, Any]]:
        """Get best performing tracks across all stages."""
        track_scores: dict[str, dict[int, float]] = {}

        for comp in self._comparisons:
            track_id = comp.track_id
            stage = comp.candidate_stage

            if track_id not in track_scores:
                track_scores[track_id] = {}

            anchor_delta = comp.anchor_delta_metrics.get("composite", comp.delta_metrics.get("composite", 0))

            if stage not in track_scores[track_id] or anchor_delta > track_scores[track_id][stage]:
                track_scores[track_id][stage] = anchor_delta

        # Compute weighted scores
        leaderboard = []
        for track_id, stage_scores in track_scores.items():
            weighted = sum(score * (stage + 1) for stage, score in stage_scores.items())
            leaderboard.append(
                {
                    "track_id": track_id,
                    "stages": stage_scores,
                    "weighted_score": weighted,
                }
            )

        leaderboard.sort(key=lambda x: x["weighted_score"], reverse=True)
        return leaderboard

    # ---------------------------------------------------------------------------
    # Alerts
    # ---------------------------------------------------------------------------

    def check_alerts(self) -> list[dict[str, Any]]:
        """Check for any alert conditions."""
        alerts = []

        # Check for failed runs
        failed_runs = [r for r in self._runs.values() if r.status == "failed"]
        if failed_runs:
            alerts.append(
                {
                    "type": "failed_runs",
                    "severity": "error",
                    "title": f"{len(failed_runs)} run(s) failed",
                    "message": ", ".join(r.run_id for r in failed_runs[:3]),
                }
            )

        # Check for low pass rates
        for stage in range(1, 5):
            status = self.get_stage_status(stage)
            if status.pass_rate is not None and status.pass_rate < 0.3 and status.completed_runs >= 3:
                alerts.append(
                    {
                        "type": "low_pass_rate",
                        "severity": "warning",
                        "title": f"Stage {stage} low pass rate",
                        "message": f"Pass rate: {status.pass_rate:.1%}",
                    }
                )

        # Check for best performers
        for stage in range(1, 5):
            status = self.get_stage_status(stage)
            if status.best_composite is not None and status.best_composite > 70:
                alerts.append(
                    {
                        "type": "high_score",
                        "severity": "success",
                        "title": f"Stage {stage} new best!",
                        "message": f"Best composite: {status.best_composite:.1f}",
                    }
                )

        return alerts

    # ---------------------------------------------------------------------------
    # Snapshot
    # ---------------------------------------------------------------------------

    def get_snapshot(self) -> DashboardSnapshot:
        """Get complete dashboard snapshot."""
        return DashboardSnapshot(
            timestamp=self._last_update,
            stages=self.get_all_stages_status(),
            recent_comparisons=self.get_recent_comparisons(),
            alerts=self.check_alerts(),
            summary=self.get_summary(),
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        total_runs = len(self._runs)
        completed = len([r for r in self._runs.values() if r.status == "completed"])
        failed = len([r for r in self._runs.values() if r.status == "failed"])

        all_composites = [r.composite for r in self._runs.values() if r.composite is not None]
        mean_composite = sum(all_composites) / len(all_composites) if all_composites else None

        return {
            "total_runs": total_runs,
            "completed_runs": completed,
            "failed_runs": failed,
            "total_comparisons": len(self._comparisons),
            "mean_composite": mean_composite,
            "last_update": self._last_update.isoformat(),
        }

    # ---------------------------------------------------------------------------
    # HTML Rendering
    # ---------------------------------------------------------------------------

    def render_html(self) -> str:
        """Render dashboard as HTML."""
        snapshot = self.get_snapshot()

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Engine Experiment Dashboard</title>",
            "<style>",
            self._get_css(),
            "</style>",
            "</head><body>",
            self._render_header(),
            self._render_alerts(snapshot.alerts),
            self._render_summary(snapshot.summary),
            self._render_stages(snapshot.stages),
            self._render_recent_comparisons(snapshot.recent_comparisons),
            self._render_leaderboard(),
            "<script>",
            self._get_js(),
            "</script>",
            "</body></html>",
        ]

        return "\n".join(html_parts)

    def _get_css(self) -> str:
        """Get dashboard CSS."""
        return """
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #333; }
            h2 { color: #555; margin-top: 30px; }
            .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stage-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .stage-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stage-card h3 { margin: 0 0 10px 0; color: #333; }
            .metric { display: flex; justify-content: space-between; padding: 5px 0; }
            .metric-label { color: #666; }
            .metric-value { font-weight: bold; }
            .pass { color: #28a745; }
            .fail { color: #dc3545; }
            .pending { color: #ffc107; }
            .alert { padding: 12px; border-radius: 4px; margin-bottom: 10px; }
            .alert-error { background: #f8d7da; color: #721c24; }
            .alert-warning { background: #fff3cd; color: #856404; }
            .alert-success { background: #d4edda; color: #155724; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; }
            .progress-bar { background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; }
            .progress-fill { background: #28a745; height: 100%; transition: width 0.3s; }
        """

    def _get_js(self) -> str:
        """Get dashboard JavaScript."""
        return """
            // Auto-refresh every 30 seconds
            setInterval(function() {
                location.reload();
            }, 30000);
        """

    def _render_header(self) -> str:
        return """
            <div class="container">
                <h1>🚀 Engine Experiment Dashboard</h1>
                <p>Real-time monitoring for the experiment harness</p>
            </div>
        """

    def _render_alerts(self, alerts: list[dict[str, Any]]) -> str:
        if not alerts:
            return ""

        html = '<div class="container">'
        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            html += f'<div class="alert {severity_class}"><strong>{alert["title"]}</strong>: {alert["message"]}</div>'
        html += '</div>'
        return html

    def _render_summary(self, summary: dict[str, Any]) -> str:
        return f"""
            <div class="container">
                <div class="card">
                    <h2>Summary</h2>
                    <div class="metric">
                        <span class="metric-label">Total Runs</span>
                        <span class="metric-value">{summary.get('total_runs', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Completed</span>
                        <span class="metric-value pass">{summary.get('completed_runs', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Failed</span>
                        <span class="metric-value fail">{summary.get('failed_runs', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Comparisons</span>
                        <span class="metric-value">{summary.get('total_comparisons', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Last Update</span>
                        <span class="metric-value">{summary.get('last_update', 'N/A')}</span>
                    </div>
                </div>
            </div>
        """

    def _render_stages(self, stages: dict[int, StageStatus]) -> str:
        html = '<div class="container"><h2>Stage Status</h2><div class="stage-grid">'

        for stage in range(1, 5):
            status = stages.get(stage)
            if status:
                progress = (status.completed_runs / max(status.total_runs, 1)) * 100
                pass_rate = f"{status.pass_rate*100:.1f}%" if status.pass_rate else "N/A"
                best = f"{status.best_composite:.1f}" if status.best_composite else "N/A"

                html += f"""
                    <div class="stage-card">
                        <h3>Stage {stage}</h3>
                        <div class="metric">
                            <span class="metric-label">Runs</span>
                            <span class="metric-value">{status.completed_runs}/{status.total_runs}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {progress}%"></div>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Pass Rate</span>
                            <span class="metric-value">{pass_rate}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Best</span>
                            <span class="metric-value">{best}</span>
                        </div>
                    </div>
                """

        html += '</div></div>'
        return html

    def _render_recent_comparisons(self, comparisons: list[dict[str, Any]]) -> str:
        if not comparisons:
            return '<div class="container"><div class="card"><h2>Recent Comparisons</h2><p>No comparisons yet.</p></div></div>'

        html = '<div class="container"><div class="card"><h2>Recent Comparisons</h2><table><thead><tr><th>Track</th><th>Stage</th><th>Delta</th><th>Anchor Delta</th><th>Pass</th></tr></thead><tbody>'

        for comp in comparisons:
            pass_class = "pass" if comp["overall_pass"] else "fail"
            html += f"""
                <tr>
                    <td>{comp['track_id']}</td>
                    <td>{comp['stage']}</td>
                    <td>{comp['delta_composite']:.2f}</td>
                    <td>{comp['anchor_delta']:.2f}</td>
                    <td class="{pass_class}">{'✓' if comp['overall_pass'] else '✗'}</td>
                </tr>
            """

        html += '</tbody></table></div></div>'
        return html

    def _render_leaderboard(self) -> str:
        leaderboard = self.get_track_leaderboard()

        if not leaderboard:
            return '<div class="container"><div class="card"><h2>Track Leaderboard</h2><p>No data yet.</p></div></div>'

        html = '<div class="container"><div class="card"><h2>Track Leaderboard</h2><table><thead><tr><th>Rank</th><th>Track</th><th>Weighted Score</th></tr></thead><tbody>'

        for i, entry in enumerate(leaderboard[:10], 1):
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{entry['track_id']}</td>
                    <td>{entry['weighted_score']:.2f}</td>
                </tr>
            """

        html += '</tbody></table></div></div>'
        return html

    # ---------------------------------------------------------------------------
    # JSON API
    # ---------------------------------------------------------------------------

    def render_json(self) -> str:
        """Render dashboard as JSON."""
        return json.dumps(self.get_snapshot(), default=str)

    # ---------------------------------------------------------------------------
    # Server (Optional)
    # ---------------------------------------------------------------------------

    def serve(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start a simple HTTP server for the dashboard."""
        try:
            from http.server import HTTPServer, SimpleHTTPRequestHandler
        except ImportError:
            print("HTTP server not available")
            return

        class DashboardHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(self.render_html().encode())
                elif self.path == "/api":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(self.render_json().encode())
                else:
                    self.send_response(404)

        server = HTTPServer((host, port), DashboardHandler)
        print(f"Dashboard running at http://{host}:{port}")
        server.serve_forever()


# ---------------------------------------------------------------------------
# CLI Integration
# ---------------------------------------------------------------------------


def create_dashboard_cli():
    """Create CLI commands for dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Engine Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    parser.add_argument("--html", action="store_true", help="Output HTML to stdout")

    args = parser.parse_args()

    dashboard = Dashboard()

    if args.html:
        print(dashboard.render_html())
    else:
        dashboard.serve(port=args.port)


if __name__ == "__main__":
    create_dashboard_cli()

