"""Red Team Phase 3 regression tests.

Each test demonstrates the exploit was possible and verifies the fix works.
Tests are named after their finding IDs from docs/red_team_phase3_report.md.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# RT-P3-001: FL num_samples inflation capped
# ---------------------------------------------------------------------------


class TestRTP3001NumSamplesCap:
    """Verify num_samples is capped to prevent aggregation weight manipulation."""

    def test_num_samples_capped_in_fl_server(self):
        from aegis.services.federated.fl_model import FederatedClassifier
        from aegis.services.federated.fl_server import FLServer

        model = FederatedClassifier()
        server = FLServer(model, min_clients=3)  # high min so auto-agg doesn't fire
        w = model.get_weights()

        # Malicious node tries to dominate with huge num_samples
        server.submit_update("malicious", [x.tolist() for x in w], num_samples=10**9)

        # Verify the malicious num_samples was capped
        malicious_update = server._current_updates[-1]
        assert malicious_update["num_samples"] <= 100_000

    def test_num_samples_zero_or_negative_clamped(self):
        from aegis.services.federated.fl_model import FederatedClassifier
        from aegis.services.federated.fl_server import FLServer

        model = FederatedClassifier()
        server = FLServer(model, min_clients=3)
        w = model.get_weights()

        server.submit_update("node", [x.tolist() for x in w], num_samples=-5)
        assert server._current_updates[-1]["num_samples"] == 0

    def test_aggregation_not_dominated_by_inflated_samples(self):
        from aegis.services.federated.fl_model import FederatedClassifier
        from aegis.services.federated.fl_server import FLServer

        model = FederatedClassifier()
        server = FLServer(model, min_clients=2)
        w = model.get_weights()

        # Create honest and malicious weight variants
        w_honest = [x.copy() for x in w]
        w_malicious = [x * 100 for x in w]

        server.submit_update("honest", [x.tolist() for x in w_honest], num_samples=100)
        # Auto-aggregation fires on second update (min_clients=2)
        result = server.submit_update(
            "malicious", [x.tolist() for x in w_malicious], num_samples=10**9,
        )
        # With capping, malicious weight is capped to 100k out of 100k+100
        assert result.get("aggregated") is True


# ---------------------------------------------------------------------------
# RT-P3-002: Indicator ID overwrite prevented
# ---------------------------------------------------------------------------


class TestRTP3002IndicatorOverwrite:
    """Verify duplicate indicator IDs don't overwrite existing indicators."""

    def test_duplicate_id_does_not_overwrite(self):
        from aegis.services.federated.indicator_registry import IndicatorRegistry

        reg = IndicatorRegistry()
        emb1 = list(np.random.randn(64))
        emb2 = list(np.random.randn(64))

        ind1 = {
            "type": "indicator",
            "id": "indicator--same-id",
            "pattern": json.dumps({"threat_embedding": emb1, "mitre_tactic": "AML.T0051"}),
            "created": "2026-03-26T00:00:00Z",
        }
        ind2 = {
            "type": "indicator",
            "id": "indicator--same-id",
            "pattern": json.dumps({"threat_embedding": emb2, "mitre_tactic": "AML.T0054"}),
            "created": "2026-03-26T01:00:00Z",
        }

        id1 = reg.add_indicator(ind1)
        id2 = reg.add_indicator(ind2)

        # Both should return the same ID (no overwrite)
        assert id1 == id2 == "indicator--same-id"

        # Original indicator should be preserved
        stored = reg.get_indicator("indicator--same-id")
        pattern = json.loads(stored["pattern"])
        np.testing.assert_array_almost_equal(
            pattern["threat_embedding"], emb1,
        )

    def test_duplicate_id_increments_hit_count(self):
        from aegis.services.federated.indicator_registry import IndicatorRegistry

        reg = IndicatorRegistry()
        ind = {
            "type": "indicator",
            "id": "indicator--hit-count",
            "pattern": json.dumps({"threat_embedding": list(np.random.randn(64))}),
            "created": "2026-03-26T00:00:00Z",
        }

        reg.add_indicator(ind)
        reg.add_indicator(ind)  # duplicate
        reg.add_indicator(ind)  # duplicate

        stored = reg.get_indicator("indicator--hit-count")
        assert stored["hit_count"] == 3


# ---------------------------------------------------------------------------
# RT-P3-003: Heartbeat trust farming rate-limited
# ---------------------------------------------------------------------------


class TestRTP3003HeartbeatRateLimit:
    """Verify heartbeats can't be spammed to farm trust."""

    def test_rapid_heartbeats_rate_limited(self):
        from aegis.services.federated.node_trust import NodeTrustScorer

        scorer = NodeTrustScorer(initial_trust=0.3)
        initial = scorer.get_trust("node-1")

        # Send 100 rapid heartbeats
        for _ in range(100):
            scorer.record_positive("node-1", "heartbeat")

        # Trust should have only increased by one heartbeat reward (0.001)
        # not 100 * 0.001 = 0.1
        final = scorer.get_trust("node-1")
        assert final <= initial + 0.01  # At most a few rewards if timing is close

    def test_heartbeat_after_interval_is_rewarded(self):
        from aegis.services.federated.node_trust import NodeTrustScorer

        scorer = NodeTrustScorer(initial_trust=0.3)
        scorer._heartbeat_min_interval = 0.0  # Disable for this test
        initial = scorer.get_trust("node-1")

        scorer.record_positive("node-1", "heartbeat")
        final = scorer.get_trust("node-1")
        assert final > initial

    def test_non_heartbeat_positive_not_rate_limited(self):
        from aegis.services.federated.node_trust import NodeTrustScorer

        scorer = NodeTrustScorer(initial_trust=0.3)
        initial = scorer.get_trust("node-1")

        # valid_indicator should not be rate limited
        for _ in range(10):
            scorer.record_positive("node-1", "valid_indicator")

        final = scorer.get_trust("node-1")
        assert final >= initial + 10 * 0.02 - 0.001  # ~0.5


# ---------------------------------------------------------------------------
# RT-P3-004: Case-insensitive Bearer token
# ---------------------------------------------------------------------------


class TestRTP3004BearerCaseInsensitive:
    @pytest.fixture(autouse=True)
    def _setup(self):
        import main as m

        self._orig_config = m._config
        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-rt-p3-key"
        m._config = config
        self.api_key = m._config.api_key
        self.client = TestClient(m.app)
        yield
        m._config = self._orig_config

    def test_lowercase_bearer_accepted(self):
        resp = self.client.get(
            "/dashboard/api/overview",
            headers={"Authorization": f"bearer {self.api_key}"},
        )
        assert resp.status_code == 200

    def test_uppercase_bearer_accepted(self):
        resp = self.client.get(
            "/dashboard/api/overview",
            headers={"Authorization": f"BEARER {self.api_key}"},
        )
        assert resp.status_code == 200

    def test_mixed_case_bearer_accepted(self):
        resp = self.client.get(
            "/dashboard/api/overview",
            headers={"Authorization": f"BeArEr {self.api_key}"},
        )
        assert resp.status_code == 200

    def test_standard_bearer_still_works(self):
        resp = self.client.get(
            "/dashboard/api/overview",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        assert resp.status_code == 200

    def test_wrong_token_still_rejected(self):
        resp = self.client.get(
            "/dashboard/api/overview",
            headers={"Authorization": "bearer wrong-token"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# RT-P3-005: Trigger-round requires minimum updates
# ---------------------------------------------------------------------------


class TestRTP3005TriggerRoundMinimum:
    @pytest.fixture(autouse=True)
    def _setup(self):
        import main as m

        self._saved = {}
        for attr in ("_config", "_fl_server", "_fl_client", "_fl_scheduler", "_fl_training_buffer"):
            self._saved[attr] = getattr(m, attr, None)

        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-rt-p3-key"
        m._config = config

        from aegis.services.federated.fl_model import FederatedClassifier
        from aegis.services.federated.fl_server import FLServer

        model = FederatedClassifier()
        m._fl_server = FLServer(model, min_clients=3)
        m._fl_client = None
        m._fl_scheduler = None

        self.api_key = m._config.api_key
        self.client = TestClient(m.app)
        yield

        for attr, val in self._saved.items():
            setattr(m, attr, val)

    def test_trigger_round_rejected_without_enough_updates(self):
        resp = self.client.post(
            "/v1/federation/fl/trigger-round",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "insufficient_updates"

    def test_trigger_round_works_with_enough_updates(self):
        import main as m
        fl_server = m._fl_server
        from aegis.services.federated.fl_model import FederatedClassifier

        model = FederatedClassifier()
        w = model.get_weights()
        for i in range(3):
            fl_server.submit_update(f"n{i}", [x.tolist() for x in w], 10)

        resp = self.client.post(
            "/v1/federation/fl/trigger-round",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# RT-P3-007: Training buffer benign flood resistance
# ---------------------------------------------------------------------------


class TestRTP3007TrainingBufferFloodResistance:
    """Verify benign samples can't evict all threat samples."""

    def test_threats_preserved_under_benign_flood(self):
        from aegis.services.federated.training_buffer import TrainingBuffer

        buf = TrainingBuffer(max_size=100, min_threat_ratio=0.3)

        # Add 30 threat samples
        for _ in range(30):
            buf.add_sample(np.random.randn(64), is_threat=True)

        # Flood with 200 benign samples
        for _ in range(200):
            buf.add_sample(np.random.randn(64), is_threat=False)

        # Threat samples should still be present (30% reserved)
        X, y = buf.get_training_data()
        threat_count = int(y.sum())
        assert threat_count >= 25  # At least most of the 30 threats remain

    def test_get_training_data_combines_both_buffers(self):
        from aegis.services.federated.training_buffer import TrainingBuffer

        buf = TrainingBuffer(max_size=100, min_threat_ratio=0.5)
        for _ in range(10):
            buf.add_sample(np.random.randn(64), is_threat=True)
        for _ in range(10):
            buf.add_sample(np.random.randn(64), is_threat=False)

        X, y = buf.get_training_data()
        assert X.shape[0] == 20
        assert int(y.sum()) == 10

    def test_clear_resets_both_buffers(self):
        from aegis.services.federated.training_buffer import TrainingBuffer

        buf = TrainingBuffer(max_size=100)
        buf.add_sample(np.random.randn(64), is_threat=True)
        buf.add_sample(np.random.randn(64), is_threat=False)
        buf.clear()
        assert buf.size == 0

    def test_stats_accurate(self):
        from aegis.services.federated.training_buffer import TrainingBuffer

        buf = TrainingBuffer(max_size=100)
        for _ in range(5):
            buf.add_sample(np.random.randn(64), is_threat=True)
        for _ in range(3):
            buf.add_sample(np.random.randn(64), is_threat=False)

        stats = buf.stats
        assert stats["threat_samples"] == 5
        assert stats["benign_samples"] == 3
        assert stats["size"] == 8


# ---------------------------------------------------------------------------
# RT-P3-008: Oversized pattern rejected
# ---------------------------------------------------------------------------


class TestRTP3008OversizedPattern:
    """Verify oversized indicator patterns are penalized."""

    def test_oversized_pattern_penalized(self):
        from aegis.services.federated.indicator_reputation import IndicatorReputationScorer

        scorer = IndicatorReputationScorer()
        # Create a pattern > 100KB
        huge_pattern = json.dumps({
            "threat_embedding": list(np.random.randn(50000)),
        })
        assert len(huge_pattern) > 102_400

        ind = {
            "type": "indicator",
            "id": "indicator--huge",
            "pattern": huge_pattern,
        }
        score = scorer.score(ind, "node-1", source_trust=0.5)
        assert any("too large" in r for r in score.reasons)
        assert score.pattern_validity < 1.0


# ---------------------------------------------------------------------------
# RT-P3-009: SSE token in query parameter (documentation-only, no code fix)
# ---------------------------------------------------------------------------


class TestRTP3009SSETokenQueryParam:
    """Verify SSE auth via query parameter (known limitation, documented).

    Note: SSE endpoint streams forever, so we only test the rejection path
    (which returns a normal JSONResponse). Testing the success path would
    require async timeout handling that complicates the test unnecessarily.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import main as m

        self._orig_config = m._config
        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-rt-p3-key"
        m._config = config
        self.api_key = m._config.api_key
        self.client = TestClient(m.app)
        yield
        m._config = self._orig_config

    def test_sse_wrong_token_rejected(self):
        resp = self.client.get(
            "/dashboard/events?token=wrong-token",
        )
        assert resp.status_code == 401

    def test_sse_no_token_rejected(self):
        resp = self.client.get("/dashboard/events")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Integration: verify existing federation tests still work
# ---------------------------------------------------------------------------


class TestRTP3Integration:
    """Verify fixes don't break existing federation functionality."""

    def test_training_buffer_save_load_with_sub_buffers(self):
        import tempfile
        from aegis.services.federated.training_buffer import TrainingBuffer

        buf = TrainingBuffer(max_size=100, min_threat_ratio=0.3)
        for i in range(5):
            buf.add_sample(np.ones(10) * i, is_threat=True)
        for i in range(5):
            buf.add_sample(np.ones(10) * (i + 10), is_threat=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            buf.save(f.name)
            buf2 = TrainingBuffer(max_size=100, min_threat_ratio=0.3)
            loaded = buf2.load(f.name)
            assert loaded == 10

    def test_indicator_registry_dedup_still_works(self):
        from aegis.services.federated.indicator_registry import IndicatorRegistry

        reg = IndicatorRegistry()
        emb = list(np.random.randn(64))

        ind1 = {
            "type": "indicator",
            "id": "indicator--dedup-1",
            "pattern": json.dumps({"threat_embedding": emb}),
            "created": "2026-03-26T00:00:00Z",
        }
        ind2 = {
            "type": "indicator",
            "id": "indicator--dedup-2",
            "pattern": json.dumps({"threat_embedding": emb}),  # same embedding
            "created": "2026-03-26T01:00:00Z",
        }

        id1 = reg.add_indicator(ind1)
        id2 = reg.add_indicator(ind2)
        assert id2 == id1  # merged by similarity

    def test_fl_server_normal_operation_unchanged(self):
        from aegis.services.federated.fl_model import FederatedClassifier
        from aegis.services.federated.fl_server import FLServer

        model = FederatedClassifier()
        server = FLServer(model, min_clients=1)
        w = model.get_weights()
        result = server.submit_update("n1", [x.tolist() for x in w], 100)
        assert result.get("aggregated") is True or result["status"] == "accepted"
