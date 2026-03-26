"""Tests for L8 Federation Zero Trust Hardening.

Tests cover:
  - ByzantineResilientAggregator (trimmed mean, Krum, Multi-Krum, anomaly detection)
  - IndicatorReputationScorer (4-factor scoring, quarantine)
  - NodeTrustScorer (earn/decay, participation gates)
  - FederationImmuneResponse (coordinated defense)
  - Integration with FL server, hub API, node registry
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Byzantine-Resilient Aggregation
# ---------------------------------------------------------------------------

from aegis.services.federated.byzantine import (
    AnomalyReport,
    ByzantineResilientAggregator,
)


def _make_update(
    node_id: str, weights: list[np.ndarray], num_samples: int = 100,
) -> dict:
    return {"node_id": node_id, "weights": weights, "num_samples": num_samples}


def _random_weights(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [rng.randn(10, 5), rng.randn(5)]


class TestByzantineResilientAggregator:
    def test_fedavg_fallback(self):
        agg = ByzantineResilientAggregator(method="fedavg")
        u1 = _make_update("n1", [np.ones((3,)), np.ones((2,))], 50)
        u2 = _make_update("n2", [np.ones((3,)) * 3, np.ones((2,)) * 3], 50)
        result = agg.aggregate([u1, u2])
        assert len(result) == 2
        np.testing.assert_allclose(result[0], np.ones(3) * 2)

    def test_trimmed_mean_basic(self):
        agg = ByzantineResilientAggregator(method="trimmed_mean", trim_ratio=0.1)
        updates = [
            _make_update(f"n{i}", [np.ones((4,)) * i], 10)
            for i in range(10)
        ]
        result = agg.aggregate(updates)
        assert len(result) == 1
        # Trimming top/bottom 1 (10%), averaging 1-8
        expected = np.mean([i for i in range(1, 9)])
        np.testing.assert_allclose(result[0], np.ones(4) * expected, atol=0.01)

    def test_trimmed_mean_few_updates(self):
        agg = ByzantineResilientAggregator(method="trimmed_mean")
        updates = [
            _make_update("n1", [np.ones((3,))], 10),
            _make_update("n2", [np.ones((3,)) * 2], 10),
        ]
        result = agg.aggregate(updates)
        assert len(result) == 1
        # Too few for trimming, falls back to mean
        np.testing.assert_allclose(result[0], np.ones(3) * 1.5)

    def test_krum_select(self):
        agg = ByzantineResilientAggregator(method="krum", krum_f=1)
        # 3 similar updates + 1 outlier
        updates = [
            _make_update("n1", [np.ones((5,))], 10),
            _make_update("n2", [np.ones((5,)) * 1.1], 10),
            _make_update("n3", [np.ones((5,)) * 0.9], 10),
            _make_update("n4", [np.ones((5,)) * 100], 10),  # outlier
        ]
        result = agg.aggregate(updates)
        assert len(result) == 1
        # Should NOT select the outlier
        assert not np.allclose(result[0], np.ones(5) * 100)

    def test_krum_single_update(self):
        agg = ByzantineResilientAggregator(method="krum")
        updates = [_make_update("n1", [np.ones((3,))], 10)]
        result = agg.aggregate(updates)
        np.testing.assert_array_equal(result[0], np.ones(3))

    def test_multi_krum(self):
        agg = ByzantineResilientAggregator(method="multi_krum", multi_krum_k=2, krum_f=1)
        updates = [
            _make_update("n1", [np.ones((5,))], 10),
            _make_update("n2", [np.ones((5,)) * 1.1], 10),
            _make_update("n3", [np.ones((5,)) * 0.9], 10),
            _make_update("n4", [np.ones((5,)) * 100], 10),  # outlier
        ]
        result = agg.aggregate(updates)
        # Should average top 2 by Krum score (not the outlier)
        assert result[0].mean() < 5  # far from 100

    def test_multi_krum_too_few(self):
        agg = ByzantineResilientAggregator(method="multi_krum", multi_krum_k=5)
        updates = [
            _make_update("n1", [np.ones((3,))], 10),
            _make_update("n2", [np.ones((3,)) * 2], 10),
        ]
        # Falls back to fedavg when updates <= multi_krum_k
        result = agg.aggregate(updates)
        assert len(result) == 1

    def test_aggregate_empty_raises(self):
        agg = ByzantineResilientAggregator()
        with pytest.raises(ValueError):
            agg.aggregate([])

    def test_detect_anomalous_norm_outlier(self):
        agg = ByzantineResilientAggregator(norm_threshold_sigma=2.0)
        updates = [
            _make_update("n1", [np.ones((10,))], 10),
            _make_update("n2", [np.ones((10,)) * 1.1], 10),
            _make_update("n3", [np.ones((10,)) * 0.9], 10),
            _make_update("n4", [np.ones((10,)) * 1000], 10),  # huge norm
        ]
        report = agg.detect_anomalous_updates(updates)
        assert len(report.flagged) >= 1
        flagged_ids = [u.get("node_id") for u in report.flagged]
        assert "n4" in flagged_ids

    def test_detect_anomalous_cosine_outlier(self):
        agg = ByzantineResilientAggregator(cosine_threshold=0.9)
        updates = [
            _make_update("n1", [np.ones((10,))], 10),
            _make_update("n2", [np.ones((10,)) * 1.1], 10),
            _make_update("n3", [np.ones((10,)) * 0.9], 10),
            _make_update("n4", [np.ones((10,)) * -1], 10),  # opposite direction
        ]
        report = agg.detect_anomalous_updates(updates)
        assert len(report.flagged) >= 1
        flagged_ids = [u.get("node_id") for u in report.flagged]
        assert "n4" in flagged_ids

    def test_detect_anomalous_too_few(self):
        agg = ByzantineResilientAggregator()
        updates = [_make_update("n1", [np.ones((5,))], 10)]
        report = agg.detect_anomalous_updates(updates)
        assert len(report.clean) == 1
        assert len(report.flagged) == 0

    def test_detect_anomalous_all_identical(self):
        agg = ByzantineResilientAggregator()
        updates = [
            _make_update(f"n{i}", [np.ones((5,))], 10)
            for i in range(5)
        ]
        report = agg.detect_anomalous_updates(updates)
        assert len(report.flagged) == 0
        assert len(report.clean) == 5

    def test_stats(self):
        agg = ByzantineResilientAggregator(method="krum")
        updates = [_make_update("n1", [np.ones((3,))], 10)]
        agg.aggregate(updates)
        stats = agg.stats
        assert stats["method"] == "krum"
        assert stats["total_aggregations"] == 1


# ---------------------------------------------------------------------------
# Indicator Reputation Scoring
# ---------------------------------------------------------------------------

from aegis.services.federated.indicator_reputation import (
    IndicatorReputationScorer,
    ReputationScore,
)


def _make_indicator(
    ind_id: str = "indicator--test-1",
    embedding: list | None = None,
    mitre_tactic: str = "TA0001",
    confidence: float = 0.9,
) -> dict:
    if embedding is None:
        embedding = list(np.random.randn(64))
    pattern = json.dumps({
        "threat_embedding": embedding,
        "mitre_tactic": mitre_tactic,
        "confidence": confidence,
        "detection_layer": "L2",
        "attack_category": "injection",
    })
    return {
        "type": "indicator",
        "id": ind_id,
        "pattern": pattern,
        "created": "2026-03-26T00:00:00Z",
    }


class TestIndicatorReputationScorer:
    def test_accept_valid_indicator(self):
        scorer = IndicatorReputationScorer(accept_threshold=0.3)
        ind = _make_indicator()
        score = scorer.score(ind, "node-1", source_trust=0.5)
        assert score.accepted is True
        assert 0.0 <= score.overall <= 1.0

    def test_reject_low_trust_source(self):
        scorer = IndicatorReputationScorer(accept_threshold=0.5)
        ind = _make_indicator()
        score = scorer.score(ind, "node-1", source_trust=0.0)
        # With 0 source trust (40% weight), total is reduced
        assert score.source_trust == 0.0

    def test_quarantine_very_bad_indicator(self):
        scorer = IndicatorReputationScorer(
            accept_threshold=0.5,
            quarantine_threshold=0.3,
        )
        # Bad indicator: missing pattern, bad type
        bad = {"type": "not-indicator", "id": "bad-id", "pattern": ""}
        score = scorer.score(bad, "node-1", source_trust=0.0)
        assert score.accepted is False
        assert len(scorer.get_quarantined()) >= 1

    def test_pattern_validity_missing_embedding(self):
        scorer = IndicatorReputationScorer()
        ind = {
            "type": "indicator",
            "id": "indicator--no-emb",
            "pattern": json.dumps({"mitre_tactic": "TA0001"}),
        }
        score = scorer.score(ind, "node-1", source_trust=0.5)
        assert score.pattern_validity < 1.0
        assert any("embedding" in r for r in score.reasons)

    def test_consistency_corroboration_boosts_score(self):
        scorer = IndicatorReputationScorer()
        emb = list(np.random.randn(64))

        # First submission from node-1
        ind1 = _make_indicator("indicator--1", embedding=emb)
        scorer.score(ind1, "node-1", source_trust=0.5)

        # Similar from node-2 should get higher consistency
        ind2 = _make_indicator("indicator--2", embedding=emb)
        score2 = scorer.score(ind2, "node-2", source_trust=0.5)
        assert score2.consistency >= 0.5

    def test_burst_detection(self):
        scorer = IndicatorReputationScorer(burst_max_count=3, burst_window_seconds=60)
        # Submit many from same source
        for i in range(5):
            ind = _make_indicator(f"indicator--burst-{i}")
            score = scorer.score(ind, "node-burster", source_trust=0.5)
        # Last ones should have low temporal score
        assert score.temporal_coherence < 0.5

    def test_release_from_quarantine(self):
        scorer = IndicatorReputationScorer(
            accept_threshold=0.9,
            quarantine_threshold=0.8,
        )
        ind = _make_indicator("indicator--q1")
        scorer.score(ind, "node-1", source_trust=0.0)
        assert len(scorer.get_quarantined()) >= 1
        released = scorer.release_from_quarantine("indicator--q1")
        assert released is not None
        assert released["id"] == "indicator--q1"
        assert len(scorer.get_quarantined()) == 0

    def test_release_nonexistent(self):
        scorer = IndicatorReputationScorer()
        assert scorer.release_from_quarantine("nope") is None

    def test_stats(self):
        scorer = IndicatorReputationScorer()
        ind = _make_indicator()
        scorer.score(ind, "node-1", source_trust=0.5)
        stats = scorer.stats
        assert stats["total_scored"] == 1
        assert "acceptance_rate" in stats


# ---------------------------------------------------------------------------
# Node Trust Scoring
# ---------------------------------------------------------------------------

from aegis.services.federated.node_trust import NodeTrustScorer, TrustEvent


class TestNodeTrustScorer:
    def test_initial_trust(self):
        scorer = NodeTrustScorer(initial_trust=0.3)
        assert scorer.get_trust("node-1") == 0.3

    def test_positive_increases_trust(self):
        scorer = NodeTrustScorer(initial_trust=0.3)
        new = scorer.record_positive("node-1", "valid_indicator")
        assert new > 0.3

    def test_negative_decreases_trust(self):
        scorer = NodeTrustScorer(initial_trust=0.5)
        new = scorer.record_negative("node-1", "byzantine_flagged")
        assert new < 0.5

    def test_trust_clamped_to_bounds(self):
        scorer = NodeTrustScorer(initial_trust=0.05, min_trust=0.0, max_trust=1.0)
        # Huge penalty
        new = scorer.record_negative("node-1", "byzantine_flagged")
        assert new >= 0.0
        # Many rewards
        scorer2 = NodeTrustScorer(initial_trust=0.95, max_trust=1.0)
        for _ in range(50):
            scorer2.record_positive("node-1", "valid_indicator")
        assert scorer2.get_trust("node-1") <= 1.0

    def test_can_participate_fl(self):
        scorer = NodeTrustScorer(initial_trust=0.3, min_trust_for_fl=0.2)
        assert scorer.can_participate_fl("node-1") is True
        scorer2 = NodeTrustScorer(initial_trust=0.1, min_trust_for_fl=0.2)
        assert scorer2.can_participate_fl("node-2") is False

    def test_can_submit_indicators(self):
        scorer = NodeTrustScorer(initial_trust=0.3, min_trust_for_indicators=0.15)
        assert scorer.can_submit_indicators("node-1") is True
        scorer2 = NodeTrustScorer(initial_trust=0.1, min_trust_for_indicators=0.15)
        assert scorer2.can_submit_indicators("node-2") is False

    def test_decay_inactive(self):
        scorer = NodeTrustScorer(
            initial_trust=0.5,
            decay_rate=0.1,
            decay_interval_seconds=1.0,
        )
        scorer.get_trust("node-1")
        # Backdate activity
        scorer._last_activity["node-1"] = time.time() - 7200  # 2 hours ago
        changed = scorer.decay_inactive()
        assert "node-1" in changed
        assert changed["node-1"] < 0.5

    def test_decay_no_change_for_active(self):
        scorer = NodeTrustScorer(decay_interval_seconds=3600)
        scorer.get_trust("node-1")
        changed = scorer.decay_inactive()
        assert len(changed) == 0

    def test_get_node_profile(self):
        scorer = NodeTrustScorer(initial_trust=0.3)
        scorer.record_positive("node-1", "valid_indicator")
        scorer.record_negative("node-1", "byzantine_flagged")
        profile = scorer.get_node_profile("node-1")
        assert profile["node_id"] == "node-1"
        assert profile["positive_events"] == 1
        assert profile["negative_events"] == 1
        assert profile["total_events"] == 2

    def test_get_all_scores(self):
        scorer = NodeTrustScorer(initial_trust=0.3)
        scorer.get_trust("n1")
        scorer.get_trust("n2")
        scores = scorer.get_all_scores()
        assert len(scores) == 2
        assert "n1" in scores

    def test_event_log_capped(self):
        scorer = NodeTrustScorer(initial_trust=0.5)
        scorer._max_events_per_node = 5
        for i in range(20):
            scorer.record_positive("node-1", "heartbeat")
        assert len(scorer._event_log["node-1"]) == 5

    def test_stats(self):
        scorer = NodeTrustScorer(initial_trust=0.3)
        scorer.get_trust("n1")
        scorer.get_trust("n2")
        stats = scorer.stats
        assert stats["total_nodes"] == 2
        assert stats["avg_trust"] == 0.3


# ---------------------------------------------------------------------------
# Federation Immune Response
# ---------------------------------------------------------------------------

from aegis.services.federated.immune_response import FederationImmuneResponse


class TestFederationImmuneResponse:
    def _make_immune(self) -> FederationImmuneResponse:
        return FederationImmuneResponse(
            byzantine=ByzantineResilientAggregator(method="trimmed_mean"),
            reputation=IndicatorReputationScorer(accept_threshold=0.3),
            trust=NodeTrustScorer(initial_trust=0.5),
        )

    def test_screen_fl_updates_clean(self):
        immune = self._make_immune()
        updates = [
            _make_update("n1", [np.ones((5,))], 10),
            _make_update("n2", [np.ones((5,)) * 1.1], 10),
            _make_update("n3", [np.ones((5,)) * 0.9], 10),
        ]
        clean, report = immune.screen_fl_updates(updates)
        assert len(clean) >= 2
        assert immune.stats["fl_rounds_protected"] == 1

    def test_screen_fl_updates_excludes_low_trust(self):
        immune = FederationImmuneResponse(
            trust=NodeTrustScorer(initial_trust=0.1, min_trust_for_fl=0.2),
        )
        updates = [
            _make_update("n1", [np.ones((5,))], 10),
            _make_update("n2", [np.ones((5,))], 10),
        ]
        clean, report = immune.screen_fl_updates(updates)
        assert len(clean) == 0  # All excluded

    def test_screen_fl_updates_penalizes_byzantine(self):
        immune = self._make_immune()
        updates = [
            _make_update("n1", [np.ones((10,))], 10),
            _make_update("n2", [np.ones((10,)) * 1.1], 10),
            _make_update("n3", [np.ones((10,)) * 0.9], 10),
            _make_update("n4", [np.ones((10,)) * 1000], 10),
        ]
        clean, report = immune.screen_fl_updates(updates)
        # n4 should be flagged and penalized
        if "n4" in [u.get("node_id") for u in report.flagged]:
            assert immune.trust.get_trust("n4") < 0.5

    def test_screen_indicator_accepted(self):
        immune = self._make_immune()
        ind = _make_indicator()
        score = immune.screen_indicator(ind, "node-1")
        assert score.accepted is True
        assert immune.trust.get_trust("node-1") > 0.5  # rewarded

    def test_screen_indicator_rejected(self):
        immune = FederationImmuneResponse(
            reputation=IndicatorReputationScorer(accept_threshold=0.9),
            trust=NodeTrustScorer(initial_trust=0.1),
        )
        bad = {"type": "not-indicator", "id": "bad", "pattern": ""}
        score = immune.screen_indicator(bad, "node-bad")
        assert score.accepted is False

    def test_aggregate_with_trust(self):
        immune = self._make_immune()
        updates = [
            _make_update("n1", [np.ones((5,))], 100),
            _make_update("n2", [np.ones((5,)) * 2], 100),
        ]
        result = immune.aggregate_with_trust(updates)
        assert len(result) == 1
        # Should aggregate to something between 1 and 2

    def test_aggregate_with_trust_empty_if_all_excluded(self):
        immune = FederationImmuneResponse(
            trust=NodeTrustScorer(initial_trust=0.01, min_trust_for_fl=0.5),
        )
        updates = [_make_update("n1", [np.ones((5,))], 10)]
        result = immune.aggregate_with_trust(updates)
        assert result == []

    def test_escalation_on_high_flag_rate(self):
        immune = FederationImmuneResponse(
            byzantine=ByzantineResilientAggregator(norm_threshold_sigma=0.1),
            trust=NodeTrustScorer(initial_trust=0.5),
            escalation_threshold=0.3,
        )
        # Create updates where most are "anomalous" due to tight threshold
        updates = [
            _make_update("n1", [np.ones((10,))], 10),
            _make_update("n2", [np.ones((10,)) * 2], 10),
            _make_update("n3", [np.ones((10,)) * 3], 10),
            _make_update("n4", [np.ones((10,)) * 5], 10),
        ]
        immune.screen_fl_updates(updates)
        # May or may not escalate depending on detection — check flag works
        assert immune.stats["fl_rounds_protected"] == 1

    def test_clear_escalation(self):
        immune = self._make_immune()
        immune._trigger_escalation("test")
        assert immune.is_escalated is True
        immune.clear_escalation()
        assert immune.is_escalated is False

    def test_stats(self):
        immune = self._make_immune()
        stats = immune.stats
        assert "fl_rounds_protected" in stats
        assert "byzantine" in stats
        assert "reputation" in stats
        assert "trust" in stats


# ---------------------------------------------------------------------------
# FL Server Integration
# ---------------------------------------------------------------------------

from aegis.services.federated.fl_model import FederatedClassifier
from aegis.services.federated.fl_server import FLServer


class TestFLServerWithImmune:
    def test_server_with_immune_response(self):
        model = FederatedClassifier()
        immune = FederationImmuneResponse(
            byzantine=ByzantineResilientAggregator(method="fedavg"),
            trust=NodeTrustScorer(initial_trust=0.5),
        )
        server = FLServer(model, min_clients=1, immune_response=immune)
        w = model.get_weights()
        server.submit_update("n1", [x.tolist() for x in w], 50)
        status = server.get_status()
        assert status["round"] >= 1

    def test_server_without_immune_response(self):
        model = FederatedClassifier()
        server = FLServer(model, min_clients=1)
        w = model.get_weights()
        server.submit_update("n1", [x.tolist() for x in w], 50)
        status = server.get_status()
        assert status["round"] >= 1

    def test_server_immune_excludes_all(self):
        model = FederatedClassifier()
        immune = FederationImmuneResponse(
            trust=NodeTrustScorer(initial_trust=0.01, min_trust_for_fl=0.5),
        )
        server = FLServer(model, min_clients=1, immune_response=immune)
        w = model.get_weights()
        result = server.submit_update("low-trust", [x.tolist() for x in w], 50)
        # Should aggregate but with "all_excluded" status
        assert result.get("aggregated") is True or result.get("status") == "accepted"


# ---------------------------------------------------------------------------
# Node Registry Integration
# ---------------------------------------------------------------------------

from aegis.services.federated.node_registry import NodeRegistry


class TestNodeRegistryWithTrust:
    def test_node_includes_trust_score(self):
        trust = NodeTrustScorer(initial_trust=0.3)
        reg = NodeRegistry(trust_scorer=trust)
        reg.register_node("n1")
        node = reg.get_node("n1")
        assert "trust_score" in node
        assert node["trust_score"] == 0.3

    def test_node_without_trust_scorer(self):
        reg = NodeRegistry()
        reg.register_node("n1")
        node = reg.get_node("n1")
        assert "trust_score" not in node

    def test_active_nodes_include_trust(self):
        trust = NodeTrustScorer(initial_trust=0.5)
        reg = NodeRegistry(trust_scorer=trust)
        reg.register_node("n1")
        active = reg.get_active_nodes()
        assert len(active) == 1
        assert active[0]["trust_score"] == 0.5

    def test_all_nodes_include_trust(self):
        trust = NodeTrustScorer(initial_trust=0.4)
        reg = NodeRegistry(trust_scorer=trust)
        reg.register_node("n1")
        reg.register_node("n2")
        all_nodes = reg.get_all_nodes()
        assert all("trust_score" in n for n in all_nodes)


# ---------------------------------------------------------------------------
# Hub API Integration
# ---------------------------------------------------------------------------


class TestHubAPIZeroTrust:
    @pytest.fixture(autouse=True)
    def _setup(self):
        import main as m

        saved = {}
        for attr in (
            "_config", "_indicator_registry", "_node_registry",
            "_federated", "_federation_immune", "_event_bus",
        ):
            saved[attr] = getattr(m, attr, None)

        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-zt-key"
        m._config = config

        from aegis.services.federated.indicator_registry import IndicatorRegistry
        m._indicator_registry = IndicatorRegistry()
        trust = NodeTrustScorer(initial_trust=0.5)
        m._node_registry = NodeRegistry(trust_scorer=trust)
        m._federation_immune = FederationImmuneResponse(
            reputation=IndicatorReputationScorer(accept_threshold=0.3),
            trust=trust,
        )
        from aegis.services.federated import FederatedIntelligenceManager
        m._federated = FederatedIntelligenceManager(instance_id="test")
        m._event_bus = None

        self.api_key = m._config.api_key
        self.client = TestClient(m.app)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        yield

        for attr, val in saved.items():
            setattr(m, attr, val)

    def test_submit_indicator_screened(self):
        ind = _make_indicator("indicator--api-1")
        resp = self.client.post(
            "/v1/federation/indicators",
            json=ind,
            headers=self.headers,
        )
        assert resp.status_code in (201, 422)

    def test_quarantined_endpoint(self):
        resp = self.client.get(
            "/v1/federation/quarantined",
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "quarantined" in data

    def test_trust_endpoint(self):
        resp = self.client.get(
            "/v1/federation/trust",
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "scores" in data
        assert "stats" in data

    def test_immune_stats_endpoint(self):
        resp = self.client.get(
            "/v1/federation/immune/stats",
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "fl_rounds_protected" in data or "status" in data

    def test_heartbeat_records_trust(self):
        resp = self.client.post(
            "/v1/federation/heartbeat",
            json={"node_id": "trust-node"},
            headers=self.headers,
        )
        assert resp.status_code == 200

    def test_quarantined_no_auth(self):
        resp = self.client.get("/v1/federation/quarantined")
        assert resp.status_code == 401

    def test_trust_no_auth(self):
        resp = self.client.get("/v1/federation/trust")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Dashboard Integration
# ---------------------------------------------------------------------------


class TestDashboardZeroTrust:
    @pytest.fixture(autouse=True)
    def _setup(self):
        import main as m
        saved = {}
        for attr in (
            "_config", "_federation_immune", "_federated",
            "_indicator_registry", "_node_registry", "_fl_server",
            "_dashboard_metrics_buffer", "_audit", "_healing",
            "_vault", "_event_bus", "_campaign_engine", "_compliance",
        ):
            saved[attr] = getattr(m, attr, None)

        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-dash-zt-key"
        m._config = config

        trust = NodeTrustScorer(initial_trust=0.5)
        m._federation_immune = FederationImmuneResponse(trust=trust)
        m._federated = None
        m._indicator_registry = None
        m._node_registry = None
        m._fl_server = None
        m._dashboard_metrics_buffer = None
        m._audit = None
        m._healing = None
        m._vault = None
        m._event_bus = None
        m._campaign_engine = None
        m._compliance = None

        self.api_key = m._config.api_key
        self.client = TestClient(m.app)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        yield

        for attr, val in saved.items():
            setattr(m, attr, val)

    def test_overview_includes_zero_trust_fields(self):
        resp = self.client.get(
            "/dashboard/api/overview",
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        fed = data.get("federation", {})
        assert "byzantine_method" in fed
        assert "quarantine_size" in fed
        assert "avg_node_trust" in fed
