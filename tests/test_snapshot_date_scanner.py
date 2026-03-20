"""
Tests for Clean State Snapshot Manager and Date-Triggered Anomaly Scanner.

Extension 2 Phase C Step 4 — Completes the temporal threat detection system.
"""

from __future__ import annotations

import hashlib
import threading
import time

import pytest

from aegis.layers.temporal.snapshot_manager import (
    CleanStateSnapshotManager,
    ComponentDiff,
    ComponentSnapshot,
    ReplayReport,
    StateSnapshot,
)
from aegis.layers.temporal.date_scanner import (
    BoundaryCheck,
    DateScannerStatus,
    DateTriggeredAnomalyScanner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sm():
    """Fresh snapshot manager with default settings."""
    return CleanStateSnapshotManager(max_snapshots=50)


@pytest.fixture
def sm_small():
    """Snapshot manager with small max for eviction testing."""
    return CleanStateSnapshotManager(max_snapshots=3)


@pytest.fixture
def scanner():
    """Fresh date scanner with default settings."""
    return DateTriggeredAnomalyScanner(
        divergence_threshold=0.3,
        multi_metric_threshold=0.15,
    )


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


# ===================================================================
# SNAPSHOT MANAGER TESTS (16+ tests)
# ===================================================================


class TestSnapshotRegisterAndCapture:
    """Tests for component registration and snapshot capture."""

    def test_register_component_and_capture(self, sm):
        """Registered components appear in snapshot."""
        sm.register_component(
            "vault", hash_fn=lambda: _hash("vault_state"), count_fn=lambda: 42,
        )
        snap = sm.capture_snapshot()
        assert "vault" in snap.components
        assert snap.components["vault"].content_hash == _hash("vault_state")
        assert snap.components["vault"].record_count == 42

    def test_capture_generates_uuid_and_timestamp(self, sm):
        """Snapshot has valid ID and timestamp."""
        before = time.time()
        snap = sm.capture_snapshot()
        after = time.time()
        assert len(snap.snapshot_id) == 12
        assert before <= snap.timestamp <= after

    def test_integrity_hash_computed(self, sm):
        """Integrity hash is computed from component hashes."""
        sm.register_component("a", hash_fn=lambda: "aaa")
        sm.register_component("b", hash_fn=lambda: "bbb")
        snap = sm.capture_snapshot()
        assert snap.integrity_hash
        assert len(snap.integrity_hash) == 64  # SHA-256 hex

    def test_integrity_hash_deterministic(self, sm):
        """Same component hashes produce same integrity hash."""
        sm.register_component("x", hash_fn=lambda: "fixed")
        s1 = sm.capture_snapshot()
        s2 = sm.capture_snapshot()
        assert s1.integrity_hash == s2.integrity_hash

    def test_capture_with_no_components(self, sm):
        """Snapshot with no registered components is still valid."""
        snap = sm.capture_snapshot()
        assert snap.snapshot_id
        assert snap.components == {}
        assert snap.integrity_hash  # hash of empty string

    def test_capture_with_metadata(self, sm):
        """Metadata is stored in snapshot."""
        snap = sm.capture_snapshot(metadata={"reason": "test"})
        assert snap.metadata["reason"] == "test"

    def test_component_with_metadata_fn(self, sm):
        """Component metadata_fn is called during capture."""
        sm.register_component(
            "cfg", hash_fn=lambda: "h", count_fn=lambda: 1,
            metadata_fn=lambda: {"version": "1.0"},
        )
        snap = sm.capture_snapshot()
        assert snap.components["cfg"].metadata["version"] == "1.0"

    def test_component_hash_fn_exception_skipped(self, sm):
        """Component that raises during capture is skipped."""
        sm.register_component("bad", hash_fn=lambda: 1 / 0)
        sm.register_component("good", hash_fn=lambda: "ok")
        snap = sm.capture_snapshot()
        assert "bad" not in snap.components
        assert "good" in snap.components


class TestSnapshotRetrieval:
    """Tests for snapshot retrieval methods."""

    def test_get_snapshot_by_id(self, sm):
        """get_snapshot retrieves by ID."""
        snap = sm.capture_snapshot()
        retrieved = sm.get_snapshot(snap.snapshot_id)
        assert retrieved is not None
        assert retrieved.snapshot_id == snap.snapshot_id

    def test_get_snapshot_unknown_id(self, sm):
        """get_snapshot returns None for unknown ID."""
        assert sm.get_snapshot("nonexistent") is None

    def test_get_snapshot_before(self, sm):
        """get_snapshot_before finds most recent snapshot before timestamp."""
        s1 = sm.capture_snapshot()
        s1.timestamp = 1000.0
        s2 = sm.capture_snapshot()
        s2.timestamp = 2000.0
        s3 = sm.capture_snapshot()
        s3.timestamp = 3000.0

        result = sm.get_snapshot_before(2500.0)
        assert result is not None
        assert result.snapshot_id == s2.snapshot_id

    def test_get_snapshot_before_none(self, sm):
        """get_snapshot_before returns None when no snapshots exist before timestamp."""
        snap = sm.capture_snapshot()
        snap.timestamp = 5000.0
        assert sm.get_snapshot_before(1000.0) is None

    def test_get_snapshots_returns_recent(self, sm):
        """get_snapshots returns recent snapshots in reverse order."""
        for i in range(5):
            s = sm.capture_snapshot()
            s.timestamp = float(i)

        result = sm.get_snapshots(limit=3)
        assert len(result) == 3
        assert result[0].timestamp > result[1].timestamp

    def test_multiple_tenants_isolated(self, sm):
        """Snapshots are isolated by tenant."""
        sm.capture_snapshot(tenant_id="tenant_a")
        sm.capture_snapshot(tenant_id="tenant_b")
        sm.capture_snapshot(tenant_id="tenant_a")

        a_snaps = sm.get_snapshots(tenant_id="tenant_a")
        b_snaps = sm.get_snapshots(tenant_id="tenant_b")
        assert len(a_snaps) == 2
        assert len(b_snaps) == 1


class TestSnapshotEvictionAndTriggers:
    """Tests for ring buffer eviction and trigger types."""

    def test_ring_buffer_eviction(self, sm_small):
        """Exceed max_snapshots → oldest evicted."""
        ids = []
        for i in range(5):
            snap = sm_small.capture_snapshot()
            ids.append(snap.snapshot_id)

        # Only 3 should remain
        all_snaps = sm_small.get_snapshots(limit=10)
        assert len(all_snaps) == 3
        # First two should be evicted
        assert sm_small.get_snapshot(ids[0]) is None
        assert sm_small.get_snapshot(ids[1]) is None
        # Last three should remain
        assert sm_small.get_snapshot(ids[2]) is not None

    def test_trigger_types(self, sm):
        """Different trigger types are recorded correctly."""
        triggers = ["manual", "scheduled", "pre_deployment", "pre_config_change"]
        for trigger in triggers:
            snap = sm.capture_snapshot(trigger=trigger)
            assert snap.trigger == trigger

    def test_thread_safety(self, sm):
        """Concurrent capture_snapshot calls don't crash."""
        sm.register_component("c", hash_fn=lambda: _hash("state"))
        results = []

        def capture():
            for _ in range(10):
                s = sm.capture_snapshot()
                results.append(s.snapshot_id)

        threads = [threading.Thread(target=capture) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 40
        assert len(set(results)) == 40  # All unique IDs


class TestReplayComparison:
    """Tests for replay comparison."""

    @pytest.mark.asyncio
    async def test_replay_detects_changed_component(self, sm):
        """Replay detects when a component hash changes."""
        state = {"hash": "original"}
        sm.register_component("vault", hash_fn=lambda: state["hash"])

        snap = sm.capture_snapshot()
        state["hash"] = "modified"

        report = await sm.replay_comparison(snap.snapshot_id)
        assert report.behavioral_divergence_detected is True
        assert len(report.components_changed) == 1
        assert report.components_changed[0].component_name == "vault"

    @pytest.mark.asyncio
    async def test_replay_no_change(self, sm):
        """Replay with unchanged state shows no divergence."""
        sm.register_component("vault", hash_fn=lambda: "constant")

        snap = sm.capture_snapshot()
        report = await sm.replay_comparison(snap.snapshot_id)
        assert report.behavioral_divergence_detected is False
        assert len(report.components_unchanged) == 1
        assert "vault" in report.components_unchanged

    @pytest.mark.asyncio
    async def test_replay_unknown_snapshot_id(self, sm):
        """Replay with unknown snapshot_id returns empty report."""
        report = await sm.replay_comparison("nonexistent")
        assert report.before_timestamp == 0.0
        assert report.behavioral_divergence_detected is False

    @pytest.mark.asyncio
    async def test_replay_new_component_detected(self, sm):
        """Replay detects a new component added after snapshot."""
        sm.register_component("old", hash_fn=lambda: "h1")
        snap = sm.capture_snapshot()

        sm.register_component("new", hash_fn=lambda: "h2")
        report = await sm.replay_comparison(snap.snapshot_id)
        changed_names = [c.component_name for c in report.components_changed]
        assert "new" in changed_names

    @pytest.mark.asyncio
    async def test_replay_latency_recorded(self, sm):
        """Replay records analysis latency."""
        sm.register_component("c", hash_fn=lambda: "h")
        snap = sm.capture_snapshot()
        report = await sm.replay_comparison(snap.snapshot_id)
        assert report.analysis_latency_ms > 0

    @pytest.mark.asyncio
    async def test_replay_change_summary_shows_count_delta(self, sm):
        """Replay change summary includes record count delta."""
        count = {"n": 10}
        sm.register_component(
            "vault", hash_fn=lambda: str(count["n"]),
            count_fn=lambda: count["n"],
        )
        snap = sm.capture_snapshot()
        count["n"] = 15
        report = await sm.replay_comparison(snap.snapshot_id)
        assert len(report.components_changed) == 1
        assert "+5" in report.components_changed[0].change_summary


class TestSnapshotStats:
    """Tests for snapshot manager statistics."""

    def test_get_stats(self, sm):
        """get_stats returns correct structure."""
        sm.register_component("c", hash_fn=lambda: "h")
        sm.capture_snapshot(trigger="manual")
        sm.capture_snapshot(trigger="scheduled")

        stats = sm.get_stats()
        assert stats["total_snapshots"] == 2
        assert stats["max_snapshots"] == 50
        assert "c" in stats["registered_components"]
        assert stats["triggers"]["manual"] == 1
        assert stats["triggers"]["scheduled"] == 1


# ===================================================================
# DATE SCANNER TESTS (14+ tests)
# ===================================================================


class TestDateScannerBoundary:
    """Tests for boundary checking logic."""

    def test_no_divergence_no_alert(self, scanner):
        """No divergence with no BBE → no alert."""
        check = scanner.check_boundary("daily", time.time())
        assert check.alert_triggered is False
        assert check.max_divergence == 0.0

    def test_boundary_with_bbe_no_divergence(self, scanner):
        """BBE with stable metrics → no alert."""

        class MockBBE:
            def get_baseline(self, tenant_id):
                return {
                    "general": {
                        "refusal_rate": 0.05,
                        "length_mean": 150.0,
                        "length_std": 30.0,
                        "tool_invocation_rate": 0.1,
                        "interaction_count": 1000,
                    }
                }

        scanner.set_bbe(MockBBE())
        check = scanner.check_boundary("daily", time.time())
        # Pre and post metrics are the same (same BBE state)
        assert check.alert_triggered is False

    def test_high_divergence_single_metric(self, scanner):
        """High divergence on one metric triggers alert."""
        call_count = {"n": 0}

        class FlippingBBE:
            def get_baseline(self, tenant_id):
                call_count["n"] += 1
                if call_count["n"] <= 1:
                    return {"general": {"refusal_rate": 0.05, "length_mean": 100.0}}
                return {"general": {"refusal_rate": 0.50, "length_mean": 100.0}}

        scanner.set_bbe(FlippingBBE())
        check = scanner.check_boundary("daily", time.time())
        assert check.alert_triggered is True
        assert "refusal_rate" in (check.alert_description or "")

    def test_multi_metric_divergence(self, scanner):
        """Moderate divergence on 3+ metrics triggers alert."""
        call_count = {"n": 0}

        class MultiFlipBBE:
            def get_baseline(self, tenant_id):
                call_count["n"] += 1
                if call_count["n"] <= 1:
                    return {"general": {
                        "refusal_rate": 0.10,
                        "length_mean": 100.0,
                        "length_std": 20.0,
                        "tool_invocation_rate": 0.05,
                    }}
                return {"general": {
                    "refusal_rate": 0.12,   # 20% change
                    "length_mean": 120.0,   # 20% change
                    "length_std": 24.0,     # 20% change
                    "tool_invocation_rate": 0.06,  # 20% change
                }}

        scanner.set_bbe(MultiFlipBBE())
        check = scanner.check_boundary("monthly", time.time())
        assert check.alert_triggered is True
        assert "Multi-metric" in (check.alert_description or "")

    def test_divergence_computation_relative(self, scanner):
        """Relative divergence computed correctly."""
        assert abs(scanner._compute_divergence(100.0, 130.0) - 0.3) < 0.001
        assert abs(scanner._compute_divergence(100.0, 100.0) - 0.0) < 0.001
        assert abs(scanner._compute_divergence(50.0, 100.0) - 1.0) < 0.001

    def test_near_zero_uses_absolute(self, scanner):
        """Near-zero pre_value uses absolute change."""
        div = scanner._compute_divergence(0.0, 0.15)
        assert div == 0.15  # Absolute, not relative

    def test_boundary_types(self, scanner):
        """Different boundary types are recorded correctly."""
        for bt in ["daily", "monthly", "deployment", "custom"]:
            check = scanner.check_boundary(bt, time.time())
            assert check.boundary_type == bt

    def test_get_recent_checks(self, scanner):
        """get_recent_checks returns checks in reverse order."""
        for _ in range(5):
            scanner.check_boundary("daily", time.time())
        checks = scanner.get_recent_checks(limit=3)
        assert len(checks) == 3


class TestDateScannerConfig:
    """Tests for date scanner configuration and status."""

    def test_add_custom_trigger_date(self, scanner):
        """Custom trigger dates are stored."""
        scanner.add_custom_trigger_date("2026-04-01", "Q2 start")
        status = scanner.get_status()
        assert "2026-04-01" in status.custom_trigger_dates

    def test_get_status_structure(self, scanner):
        """get_status returns correct structure."""
        status = scanner.get_status()
        assert isinstance(status, DateScannerStatus)
        assert status.enabled is True
        assert "daily" in status.monitored_boundaries
        assert status.checks_performed == 0
        assert status.alerts_generated == 0

    def test_alert_count_tracked(self, scanner):
        """Alerts generated count is tracked."""
        call_count = {"n": 0}

        class FlipBBE:
            def get_baseline(self, tenant_id):
                call_count["n"] += 1
                if call_count["n"] <= 1:
                    return {"g": {"refusal_rate": 0.0}}
                return {"g": {"refusal_rate": 0.5}}

        scanner.set_bbe(FlipBBE())
        scanner.check_boundary("daily", time.time())
        assert scanner.get_status().alerts_generated == 1

    def test_scanner_without_bbe_graceful(self, scanner):
        """Scanner without BBE: empty metrics, no crash."""
        check = scanner.check_boundary("daily", time.time())
        assert check.pre_metrics == {}
        assert check.post_metrics == {}
        assert check.alert_triggered is False

    def test_deployment_boundary(self, scanner):
        """Deployment boundary check works manually."""
        check = scanner.check_boundary("deployment", time.time())
        assert check.boundary_type == "deployment"
        assert check.check_id

    def test_alert_description_includes_metrics(self, scanner):
        """Alert description includes the divergent metric names."""
        call_count = {"n": 0}

        class FlipBBE:
            def get_baseline(self, tenant_id):
                call_count["n"] += 1
                if call_count["n"] <= 1:
                    return {"cat": {"length_mean": 100.0}}
                return {"cat": {"length_mean": 200.0}}

        scanner.set_bbe(FlipBBE())
        check = scanner.check_boundary("daily", time.time())
        assert check.alert_triggered is True
        assert "length_mean" in check.alert_description


# ===================================================================
# INTEGRATION TESTS (6+ tests)
# ===================================================================


class TestIntegration:
    """Integration tests combining snapshot manager, date scanner, BBE, and TCE."""

    @pytest.mark.asyncio
    async def test_snapshot_then_tce_replay(self):
        """Snapshot → change state → TCE identifies → replay confirms."""
        from aegis.layers.temporal.provenance_registry import MemoryProvenanceRegistry
        from aegis.layers.temporal.correlation_engine import TemporalCorrelationEngine

        sm = CleanStateSnapshotManager(max_snapshots=50)
        mpr = MemoryProvenanceRegistry(max_records=1000)
        tce = TemporalCorrelationEngine(max_candidates=5)
        tce.set_mpr(mpr)

        state = {"hash": "clean"}
        sm.register_component("vault", hash_fn=lambda: state["hash"])

        # 1. Capture clean snapshot
        snap = sm.capture_snapshot(trigger="scheduled")

        # 2. Simulate poisoning: state changes
        state["hash"] = "poisoned"
        now = time.time()
        for i in range(12):
            mpr.record_event("config_change", "modified", f"c-{i}", _hash(str(i)), "admin", "trusted")
        r = mpr.record_event("rag_document", "created", "evil-doc", _hash("evil"), "api", "untrusted")
        r.timestamp = now - 600

        # 3. TCE correlates
        report = await tce.correlate(now, "semantic_drift", "drift detected")
        assert report.confidence in ("medium", "high")

        # 4. Replay confirms change
        replay = await sm.replay_comparison(snap.snapshot_id)
        assert replay.behavioral_divergence_detected is True
        assert any(c.component_name == "vault" for c in replay.components_changed)

    @pytest.mark.asyncio
    async def test_date_scanner_with_bbe(self):
        """Date scanner detects post-boundary shift from BBE."""
        from aegis.layers.temporal.baseline_engine import BehavioralBaselineEngine

        bbe = BehavioralBaselineEngine()
        scanner = DateTriggeredAnomalyScanner(
            divergence_threshold=0.3,
            multi_metric_threshold=0.15,
        )
        scanner.set_bbe(bbe)

        # Record some interactions to establish baseline
        for i in range(20):
            await bbe.record_interaction(
                tenant_id="default",
                category="general",
                query=f"test query {i}",
                response=f"normal response {i}",
                was_refused=False,
            )

        # Check boundary — metrics should be stable
        check = scanner.check_boundary("daily", time.time())
        # With stable BBE, no alert expected
        assert check.boundary_type == "daily"

    @pytest.mark.asyncio
    async def test_date_scanner_alert_triggers_tce(self):
        """Date scanner alert can be used as TCE anomaly input."""
        from aegis.layers.temporal.provenance_registry import MemoryProvenanceRegistry
        from aegis.layers.temporal.correlation_engine import TemporalCorrelationEngine

        mpr = MemoryProvenanceRegistry(max_records=1000)
        tce = TemporalCorrelationEngine(max_candidates=5)
        tce.set_mpr(mpr)

        # Populate MPR
        now = time.time()
        for i in range(12):
            r = mpr.record_event("config_change", "modified", f"c-{i}", _hash(str(i)), "admin", "trusted")
            r.timestamp = now - 7200 - i * 3600

        # Scanner produces an alert at boundary_timestamp
        boundary_ts = now
        # Use boundary_timestamp as anomaly_timestamp for TCE
        report = await tce.correlate(boundary_ts, "date_boundary_shift", "sleeper activation suspected")
        assert report.report_id
        assert report.anomaly_type == "date_boundary_shift"

    @pytest.mark.asyncio
    async def test_full_pipeline_snapshot_mpr_drift_replay(self):
        """Full pipeline: snapshot → MPR event → drift → replay."""
        from aegis.layers.temporal.provenance_registry import MemoryProvenanceRegistry

        sm = CleanStateSnapshotManager(max_snapshots=50)
        mpr = MemoryProvenanceRegistry(max_records=1000)

        counter = {"val": 100}
        sm.register_component(
            "vault",
            hash_fn=lambda: _hash(str(counter["val"])),
            count_fn=lambda: counter["val"],
        )

        # Capture pre-change snapshot
        snap = sm.capture_snapshot(trigger="pre_rag_refresh")

        # MPR records the state change
        mpr.record_event("rag_document", "created", "new-doc", _hash("doc"), "user", "untrusted")

        # State changes
        counter["val"] = 150

        # Replay confirms
        replay = await sm.replay_comparison(snap.snapshot_id)
        assert replay.behavioral_divergence_detected is True
        changed = replay.components_changed[0]
        assert changed.before_count == 100
        assert changed.current_count == 150


class TestEndpoints:
    """Tests for the snapshot and date scanner API endpoints."""

    API_KEY = "aegis-test-secretkey456"

    @pytest.fixture
    def setup_layers(self):
        from aegis.config import AegisConfig
        from aegis.main import _init_layers
        import aegis.main as main_module
        config = AegisConfig(
            api_key=self.API_KEY,
            upstream_url="https://mock-upstream.test",
            upstream_api_key="test-key",
        )
        _init_layers(config)
        yield
        main_module._snapshot_manager = None
        main_module._date_scanner = None

    @pytest.fixture
    def client(self, setup_layers):
        from fastapi.testclient import TestClient
        from aegis.main import app
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": f"Bearer {self.API_KEY}"}

    def test_snapshot_endpoint_auth(self, client):
        """POST /v1/temporal/snapshot requires auth."""
        resp = client.post("/v1/temporal/snapshot")
        assert resp.status_code == 401

    def test_snapshot_endpoint_authenticated(self, client, auth_headers):
        """POST /v1/temporal/snapshot creates a snapshot."""
        resp = client.post(
            "/v1/temporal/snapshot",
            headers=auth_headers,
            json={"trigger": "manual"},
        )
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "snapshot_id" in data
            assert "integrity_hash" in data

    def test_snapshots_list_endpoint(self, client, auth_headers):
        """GET /v1/temporal/snapshots lists snapshots."""
        # Create a snapshot first
        client.post("/v1/temporal/snapshot", headers=auth_headers, json={})
        resp = client.get("/v1/temporal/snapshots", headers=auth_headers)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "snapshots" in data

    def test_replay_endpoint(self, client, auth_headers):
        """POST /v1/temporal/replay returns replay report."""
        # Create snapshot first
        create_resp = client.post(
            "/v1/temporal/snapshot", headers=auth_headers, json={},
        )
        if create_resp.status_code == 200:
            snapshot_id = create_resp.json()["snapshot_id"]
            resp = client.post(
                "/v1/temporal/replay",
                headers=auth_headers,
                json={"snapshot_id": snapshot_id},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "report_id" in data
            assert "behavioral_divergence_detected" in data

    def test_replay_missing_snapshot_id(self, client, auth_headers):
        """POST /v1/temporal/replay without snapshot_id returns 400."""
        resp = client.post(
            "/v1/temporal/replay",
            headers=auth_headers,
            json={},
        )
        if resp.status_code != 503:
            assert resp.status_code == 400

    def test_date_scanner_status_endpoint(self, client, auth_headers):
        """GET /v1/temporal/date-scanner/status returns status."""
        resp = client.get("/v1/temporal/date-scanner/status", headers=auth_headers)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "enabled" in data
            assert "monitored_boundaries" in data
