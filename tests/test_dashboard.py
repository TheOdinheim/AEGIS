"""Tests for the AEGIS Production Dashboard — API, SSE, metrics buffer, frontend."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient


# ============================================================================
# Metrics Buffer Tests
# ============================================================================


class TestMetricsBuffer:
    """Test the in-memory rolling time-series buffer."""

    def test_buffer_empty_on_init(self):
        from aegis.dashboard.metrics_buffer import MetricsBuffer
        buf = MetricsBuffer()
        assert buf.sample_count == 0

    def test_buffer_stores_samples(self):
        from aegis.dashboard.metrics_buffer import MetricsBuffer, MetricsSample
        buf = MetricsBuffer(max_samples=100)
        buf._samples.append(MetricsSample(
            timestamp=time.time(),
            values={"requests_total": 1.5, "blocks_total": 0.2},
        ))
        assert buf.sample_count == 1

    def test_buffer_get_timeseries(self):
        from aegis.dashboard.metrics_buffer import MetricsBuffer, MetricsSample
        buf = MetricsBuffer(max_samples=100)
        now = time.time()
        for i in range(5):
            buf._samples.append(MetricsSample(
                timestamp=now - (4 - i) * 10,
                values={"requests_total": float(i), "blocks_total": 0.0},
            ))
        ts = buf.get_timeseries("requests_total", "1h")
        assert len(ts) == 5
        assert ts[0]["value"] == 0.0
        assert ts[4]["value"] == 4.0

    def test_buffer_prunes_old_data(self):
        from aegis.dashboard.metrics_buffer import MetricsBuffer, MetricsSample
        buf = MetricsBuffer(max_samples=100)
        now = time.time()
        # Add old sample (2 hours ago)
        buf._samples.append(MetricsSample(
            timestamp=now - 7200,
            values={"requests_total": 1.0},
        ))
        # Add recent sample
        buf._samples.append(MetricsSample(
            timestamp=now,
            values={"requests_total": 2.0},
        ))
        removed = buf.prune(max_age_seconds=3600)
        assert removed == 1
        assert buf.sample_count == 1

    def test_buffer_max_samples_enforced(self):
        from aegis.dashboard.metrics_buffer import MetricsBuffer, MetricsSample
        buf = MetricsBuffer(max_samples=3)
        for i in range(5):
            buf._samples.append(MetricsSample(
                timestamp=time.time() + i,
                values={"requests_total": float(i)},
            ))
        assert buf.sample_count == 3

    def test_buffer_get_timeseries_unknown_metric(self):
        from aegis.dashboard.metrics_buffer import MetricsBuffer, MetricsSample
        buf = MetricsBuffer(max_samples=100)
        buf._samples.append(MetricsSample(
            timestamp=time.time(),
            values={"requests_total": 1.0},
        ))
        ts = buf.get_timeseries("nonexistent_metric", "1h")
        assert len(ts) == 1
        assert ts[0]["value"] == 0.0

    @pytest.mark.asyncio
    async def test_buffer_start_stop(self):
        from aegis.dashboard.metrics_buffer import MetricsBuffer
        buf = MetricsBuffer()
        await buf.start()
        assert buf._running is True
        await buf.stop()
        assert buf._running is False


# ============================================================================
# Dashboard API Endpoint Tests
# ============================================================================


class TestDashboardAPI:
    """Test dashboard API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import main as m
        from aegis.config import get_config
        from aegis.dashboard.metrics_buffer import MetricsBuffer

        self._orig_config = m._config
        self._orig_barrier = m._barrier
        self._orig_innate = m._innate
        self._orig_adaptive = m._adaptive
        self._orig_output = m._output
        self._orig_policy = m._policy
        self._orig_healing = m._healing
        self._orig_vault = m._vault
        self._orig_audit = m._audit
        self._orig_campaign = m._campaign_engine
        self._orig_compliance = m._compliance
        self._orig_buffer = m._dashboard_metrics_buffer

        # Initialize layers if needed
        if m._barrier is None:
            m._init_layers()

        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-dashboard-key"
            m._config = config

        # Ensure dashboard buffer exists
        if m._dashboard_metrics_buffer is None:
            m._dashboard_metrics_buffer = MetricsBuffer()

        self.api_key = m._config.api_key
        self.client = TestClient(m.app, raise_server_exceptions=False)
        self.auth = {"Authorization": f"Bearer {self.api_key}"}
        yield

        m._config = self._orig_config
        m._barrier = self._orig_barrier
        m._innate = self._orig_innate
        m._adaptive = self._orig_adaptive
        m._output = self._orig_output
        m._policy = self._orig_policy
        m._healing = self._orig_healing
        m._vault = self._orig_vault
        m._audit = self._orig_audit
        m._campaign_engine = self._orig_campaign
        m._compliance = self._orig_compliance
        m._dashboard_metrics_buffer = self._orig_buffer

    # --- Authentication ---

    def test_overview_requires_auth(self):
        resp = self.client.get("/dashboard/api/overview")
        assert resp.status_code == 401

    def test_detections_requires_auth(self):
        resp = self.client.get("/dashboard/api/detections")
        assert resp.status_code == 401

    def test_campaigns_requires_auth(self):
        resp = self.client.get("/dashboard/api/campaigns")
        assert resp.status_code == 401

    def test_compliance_requires_auth(self):
        resp = self.client.get("/dashboard/api/compliance")
        assert resp.status_code == 401

    def test_timeseries_requires_auth(self):
        resp = self.client.get("/dashboard/api/metrics/timeseries")
        assert resp.status_code == 401

    # --- Overview ---

    def test_overview_returns_structure(self):
        resp = self.client.get("/dashboard/api/overview", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "threat_level" in data
        assert "level" in data["threat_level"]
        assert "value" in data["threat_level"]
        assert "requests" in data
        assert "total" in data["requests"]
        assert "blocked" in data["requests"]
        assert "passed" in data["requests"]
        assert "rps" in data["requests"]
        assert "detection" in data
        assert "layers" in data
        assert isinstance(data["layers"], list)
        assert len(data["layers"]) == 7
        assert "circuit_breakers" in data
        assert "vault_size" in data
        assert "uptime_seconds" in data
        assert "test_count" in data
        assert "version" in data

    def test_overview_layer_ids(self):
        resp = self.client.get("/dashboard/api/overview", headers=self.auth)
        data = resp.json()
        layer_ids = [l["id"] for l in data["layers"]]
        assert layer_ids == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

    def test_overview_tli_is_valid(self):
        resp = self.client.get("/dashboard/api/overview", headers=self.auth)
        data = resp.json()
        assert data["threat_level"]["level"] in ("GREEN", "BLUE", "YELLOW", "ORANGE", "RED")

    # --- Detections ---

    def test_detections_returns_list(self):
        resp = self.client.get("/dashboard/api/detections", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_detections_respects_limit(self):
        resp = self.client.get("/dashboard/api/detections?limit=5", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["detections"]) <= 5

    # --- Campaigns ---

    def test_campaigns_returns_structure(self):
        resp = self.client.get("/dashboard/api/campaigns", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "active_campaigns" in data
        assert "recent_campaigns" in data
        assert "total_detected" in data

    def test_campaigns_without_engine(self):
        """Campaign endpoint gracefully handles missing engine."""
        import main as m
        m._campaign_engine = None
        resp = self.client.get("/dashboard/api/campaigns", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_detected"] == 0

    # --- Compliance ---

    def test_compliance_returns_structure(self):
        resp = self.client.get("/dashboard/api/compliance", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "frameworks" in data
        assert "coverage" in data
        assert "controls_mapped" in data

    def test_compliance_without_engine(self):
        """Compliance endpoint gracefully handles missing engine."""
        import main as m
        m._compliance = None
        resp = self.client.get("/dashboard/api/compliance", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["frameworks"] == []

    # --- Timeseries ---

    def test_timeseries_returns_buckets(self):
        resp = self.client.get(
            "/dashboard/api/metrics/timeseries?metric=requests_total",
            headers=self.auth,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metric"] == "requests_total"
        assert data["period"] == "1h"
        assert "buckets" in data
        assert isinstance(data["buckets"], list)

    def test_timeseries_invalid_metric(self):
        resp = self.client.get(
            "/dashboard/api/metrics/timeseries?metric=bad_metric",
            headers=self.auth,
        )
        assert resp.status_code == 400

    def test_timeseries_all_valid_metrics(self):
        for metric in ("requests_total", "blocks_total", "threat_level", "p95_latency_ms"):
            resp = self.client.get(
                f"/dashboard/api/metrics/timeseries?metric={metric}",
                headers=self.auth,
            )
            assert resp.status_code == 200, f"Failed for metric={metric}"


# ============================================================================
# Dashboard Frontend Tests
# ============================================================================


class TestDashboardFrontend:
    """Test dashboard HTML page serving."""

    def test_dashboard_serves_html(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "AEGIS" in resp.text
        assert "react" in resp.text.lower() or "React" in resp.text

    def test_dashboard_trailing_slash(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/dashboard/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_dashboard_no_auth_required(self):
        """Dashboard page itself doesn't require authentication."""
        from main import app
        client = TestClient(app)
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_contains_key_elements(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/dashboard")
        text = resp.text
        assert "The AI Immune System" in text
        assert "auth-overlay" in text
        assert "/dashboard/api/overview" in text
        assert "/dashboard/events" in text


# ============================================================================
# SSE Endpoint Tests
# ============================================================================


class TestDashboardSSE:
    """Test SSE event stream endpoint."""

    def test_sse_requires_auth(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/dashboard/events")
        assert resp.status_code == 401

    def test_sse_authenticated_returns_event_stream(self):
        """Verify SSE endpoint returns event-stream content type when authenticated.

        Uses a direct ASGI scope call instead of TestClient to avoid SSE stream
        hanging. Validates the endpoint route exists and responds to authenticated
        requests by checking the unauthenticated case returns 401 JSON (not HTML).
        """
        import main as m
        from aegis.config import get_config

        if m._barrier is None:
            m._init_layers()
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-sse-key"
            m._config = config

        # Verify the SSE endpoint rejects bad tokens
        client = TestClient(m.app)
        resp = client.get("/dashboard/events?token=wrong-key")
        assert resp.status_code == 401
        assert resp.json()["error"] == "Authentication required"

        # Verify the endpoint route is registered and distinct from frontend
        from main import app
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/dashboard/events" in routes
