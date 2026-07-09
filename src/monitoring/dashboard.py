"""
Dashboard Utilities / Ray Dashboard Extensions

Provides a tiny, dependency-free HTTP endpoint that serves:
    - /metrics : Prometheus exposition (from MetricsCollector)
    - /healthz : liveness probe for Kubernetes
    - /        : a minimal HTML summary of cluster + training state

This complements the Ray dashboard (port 8265) with an app-specific view and
a Prometheus scrape target, without pulling in a heavy web framework.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, Optional

from src.monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class DashboardState:
    """Mutable holder for the latest cluster/training snapshot."""

    def __init__(self):
        self.cluster_stats: Dict = {}
        self.training_stats: Dict = {}

    def update_cluster(self, stats: Dict) -> None:
        self.cluster_stats = stats

    def update_training(self, stats: Dict) -> None:
        self.training_stats = stats


def _render_html(state: DashboardState) -> str:
    """Render a minimal HTML dashboard page."""
    cluster = state.cluster_stats
    training = state.training_stats
    return f"""<!doctype html>
<html><head><title>Ray + K8s GPU Scaling</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
</style></head>
<body>
  <h1>Ray + Kubernetes GPU Scaling</h1>
  <div class="card">
    <h2>Cluster</h2>
    <pre>{json.dumps(cluster, indent=2, default=str)}</pre>
  </div>
  <div class="card">
    <h2>Training</h2>
    <pre>{json.dumps(training, indent=2, default=str)}</pre>
  </div>
</body></html>"""


class DashboardServer:
    """
    A lightweight HTTP dashboard + metrics endpoint.

    Example:
        collector = MetricsCollector()
        state = DashboardState()
        server = DashboardServer(collector, state, port=8266)
        server.serve_forever()
    """

    def __init__(
        self,
        collector: MetricsCollector,
        state: Optional[DashboardState] = None,
        host: str = "0.0.0.0",
        port: int = 8266,
    ):
        self.collector = collector
        self.state = state or DashboardState()
        self.host = host
        self.port = port
        self._httpd: Optional[HTTPServer] = None

    def _handler_factory(self) -> Callable:
        collector = self.collector
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, code: int, body: str, content_type: str):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                encoded = body.encode("utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
                if self.path == "/metrics":
                    self._respond(200, collector.render(), "text/plain; version=0.0.4")
                elif self.path == "/healthz":
                    self._respond(200, json.dumps({"status": "ok"}), "application/json")
                elif self.path in ("/", "/index.html"):
                    self._respond(200, _render_html(state), "text/html")
                else:
                    self._respond(404, "not found", "text/plain")

            def log_message(self, *args):  # silence default stderr logging
                pass

        return Handler

    def serve_forever(self) -> None:
        """Start the blocking HTTP server."""
        self._httpd = HTTPServer((self.host, self.port), self._handler_factory())
        logger.info("Dashboard serving on http://%s:%d", self.host, self.port)
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        """Stop the HTTP server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
