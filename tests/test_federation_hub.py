"""
Tests for L8 Federation Hub — Indicator Sharing Infrastructure.

Covers:
- STIX Generator: valid structure, required fields, MITRE mapping, relationships, bundles
- Indicator Registry: add/retrieve, since filter, dedup by cosine similarity, dormant lifecycle, stats
- Hub API: POST/GET indicators, stats, heartbeat, auth requirements
- Node Registry: register, heartbeat, active filtering, network stats
- Federation Pipeline: novel attack triggers, DP noise, registry storage, event bus
- Dashboard integration: overview includes federation, /dashboard/api/federation returns data
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from aegis.services.federated.stix_generator import (
    generate_indicator,
    generate_relationship,
    generate_stix_bundle,
)
from aegis.services.federated.indicator_registry import (
    IndicatorRegistry,
    _cosine_similarity,
    _extract_embedding,
)
from aegis.services.federated.node_registry import NodeRegistry
from aegis.services.federated.pipeline import FederationPipeline


# ---------------------------------------------------------------------------
# STIX Generator Tests
# ---------------------------------------------------------------------------


class TestStixGeneratorIndicator:
    """Test STIX 2.1 indicator generation."""

    def _make_embedding(self, dim: int = 384) -> np.ndarray:
        rng = np.random.RandomState(42)
        emb = rng.randn(dim).astype(np.float64)
        return emb / np.linalg.norm(emb)

    def test_indicator_has_required_stix_fields(self):
        emb = self._make_embedding()
        ind = generate_indicator(noised_embedding=emb)
        assert ind["type"] == "indicator"
        assert ind["spec_version"] == "2.1"
        assert ind["id"].startswith("indicator--")
        assert "created" in ind
        assert "modified" in ind
        assert "name" in ind
        assert "pattern" in ind
        assert "pattern_type" in ind
        assert ind["pattern_type"] == "aegis"
        assert "valid_from" in ind
        assert "labels" in ind
        assert "confidence" in ind

    def test_indicator_pattern_contains_embedding(self):
        emb = self._make_embedding()
        ind = generate_indicator(noised_embedding=emb, confidence=0.95)
        pattern = json.loads(ind["pattern"])
        assert "threat_embedding" in pattern
        assert len(pattern["threat_embedding"]) == 384
        assert pattern["confidence"] == 0.95

    def test_indicator_mitre_tactic_in_labels(self):
        emb = self._make_embedding()
        ind = generate_indicator(
            noised_embedding=emb, mitre_tactic="AML.T0051"
        )
        assert "AML.T0051" in ind["labels"]
        assert "malicious-activity" in ind["labels"]

    def test_indicator_name_uses_attack_category(self):
        emb = self._make_embedding()
        ind = generate_indicator(
            noised_embedding=emb, attack_category="prompt_injection"
        )
        assert "prompt_injection" in ind["name"]

    def test_indicator_name_uses_tactic_when_no_category(self):
        emb = self._make_embedding()
        ind = generate_indicator(
            noised_embedding=emb, mitre_tactic="AML.T0051"
        )
        assert "LLM Prompt Injection" in ind["name"]

    def test_indicator_confidence_clamped_0_100(self):
        emb = self._make_embedding()
        ind = generate_indicator(noised_embedding=emb, confidence=1.5)
        assert ind["confidence"] == 100
        ind2 = generate_indicator(noised_embedding=emb, confidence=-0.1)
        assert ind2["confidence"] == 0

    def test_indicator_default_description(self):
        emb = self._make_embedding()
        ind = generate_indicator(noised_embedding=emb, detection_layer="L3")
        assert "L3" in ind["description"]

    def test_indicator_custom_description(self):
        emb = self._make_embedding()
        ind = generate_indicator(
            noised_embedding=emb, description="Custom attack description"
        )
        assert ind["description"] == "Custom attack description"

    def test_indicator_pattern_detection_layer(self):
        emb = self._make_embedding()
        ind = generate_indicator(noised_embedding=emb, detection_layer="L3")
        pattern = json.loads(ind["pattern"])
        assert pattern["detection_layer"] == "L3"

    def test_indicator_unique_ids(self):
        emb = self._make_embedding()
        ids = {generate_indicator(noised_embedding=emb)["id"] for _ in range(10)}
        assert len(ids) == 10


class TestStixGeneratorRelationship:
    """Test STIX 2.1 relationship generation."""

    def test_relationship_structure(self):
        rel = generate_relationship(
            indicator_id="indicator--abc123", mitre_tactic="AML.T0051"
        )
        assert rel["type"] == "relationship"
        assert rel["spec_version"] == "2.1"
        assert rel["id"].startswith("relationship--")
        assert rel["relationship_type"] == "indicates"
        assert rel["source_ref"] == "indicator--abc123"
        assert rel["target_ref"].startswith("attack-pattern--")

    def test_relationship_target_deterministic(self):
        rel1 = generate_relationship(
            indicator_id="indicator--a", mitre_tactic="AML.T0051"
        )
        rel2 = generate_relationship(
            indicator_id="indicator--b", mitre_tactic="AML.T0051"
        )
        # Same tactic → same attack-pattern UUID5
        assert rel1["target_ref"] == rel2["target_ref"]

    def test_different_tactics_different_targets(self):
        rel1 = generate_relationship(
            indicator_id="indicator--a", mitre_tactic="AML.T0051"
        )
        rel2 = generate_relationship(
            indicator_id="indicator--a", mitre_tactic="AML.T0054"
        )
        assert rel1["target_ref"] != rel2["target_ref"]


class TestStixGeneratorBundle:
    """Test STIX 2.1 bundle generation."""

    def _make_embedding(self) -> np.ndarray:
        return np.random.RandomState(42).randn(384).astype(np.float64)

    def test_bundle_with_tactic(self):
        bundle = generate_stix_bundle(
            noised_embedding=self._make_embedding(),
            mitre_tactic="AML.T0051",
            confidence=0.9,
        )
        assert bundle["type"] == "bundle"
        assert bundle["id"].startswith("bundle--")
        assert len(bundle["objects"]) == 2
        assert bundle["objects"][0]["type"] == "indicator"
        assert bundle["objects"][1]["type"] == "relationship"

    def test_bundle_without_tactic(self):
        bundle = generate_stix_bundle(
            noised_embedding=self._make_embedding(),
        )
        assert len(bundle["objects"]) == 1
        assert bundle["objects"][0]["type"] == "indicator"


# ---------------------------------------------------------------------------
# Indicator Registry Tests
# ---------------------------------------------------------------------------


class TestIndicatorRegistry:
    """Test in-memory indicator store."""

    def _make_indicator(self, dim: int = 384, seed: int = 42, **kwargs) -> dict:
        rng = np.random.RandomState(seed)
        emb = rng.randn(dim).astype(np.float64)
        emb = emb / np.linalg.norm(emb)
        return generate_indicator(
            noised_embedding=emb,
            mitre_tactic=kwargs.get("mitre_tactic", "AML.T0051"),
            confidence=kwargs.get("confidence", 0.9),
            detection_layer=kwargs.get("detection_layer", "L3"),
        )

    def test_add_and_retrieve(self):
        reg = IndicatorRegistry()
        ind = self._make_indicator()
        ind_id = reg.add_indicator(ind)
        assert ind_id == ind["id"]

        retrieved = reg.get_indicator(ind_id)
        assert retrieved is not None
        assert retrieved["type"] == "indicator"
        assert retrieved["hit_count"] == 1
        assert retrieved["status"] == "active"

    def test_add_requires_id(self):
        reg = IndicatorRegistry()
        with pytest.raises(ValueError, match="id"):
            reg.add_indicator({"type": "indicator"})

    def test_get_nonexistent_returns_none(self):
        reg = IndicatorRegistry()
        assert reg.get_indicator("indicator--nope") is None

    def test_dedup_by_cosine_similarity(self):
        reg = IndicatorRegistry(similarity_threshold=0.95)
        ind1 = self._make_indicator(seed=42)
        ind2 = self._make_indicator(seed=42)  # Same embedding
        id1 = reg.add_indicator(ind1)
        id2 = reg.add_indicator(ind2)
        assert id1 == id2  # Merged
        retrieved = reg.get_indicator(id1)
        assert retrieved["hit_count"] == 2

    def test_different_embeddings_not_deduped(self):
        reg = IndicatorRegistry()
        ind1 = self._make_indicator(seed=1)
        ind2 = self._make_indicator(seed=999)
        id1 = reg.add_indicator(ind1)
        id2 = reg.add_indicator(ind2)
        assert id1 != id2
        stats = reg.get_stats()
        assert stats["total"] == 2

    def test_get_indicators_since(self):
        reg = IndicatorRegistry()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        ind = self._make_indicator()
        reg.add_indicator(ind)
        results = reg.get_indicators_since(old_time)
        assert len(results) == 1

    def test_get_indicators_since_excludes_old(self):
        reg = IndicatorRegistry()
        ind = self._make_indicator()
        reg.add_indicator(ind)
        future_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        results = reg.get_indicators_since(future_time)
        assert len(results) == 0

    def test_dormant_lifecycle(self):
        reg = IndicatorRegistry(dormant_age_seconds=0)  # Instant dormancy
        ind = self._make_indicator()
        ind_id = reg.add_indicator(ind)

        # Force _added_at to the past
        reg._indicators[ind_id]["_added_at"] = time.time() - 1

        aged = reg.age_indicators()
        assert aged == 1

        # Dormant indicators excluded from get_all
        all_active = reg.get_all_indicators(include_dormant=False)
        assert len(all_active) == 0

        # But included with flag
        all_with_dormant = reg.get_all_indicators(include_dormant=True)
        assert len(all_with_dormant) == 1
        assert all_with_dormant[0]["status"] == "dormant"

    def test_reactivate_dormant(self):
        reg = IndicatorRegistry(dormant_age_seconds=0)
        ind = self._make_indicator()
        ind_id = reg.add_indicator(ind)
        reg._indicators[ind_id]["_added_at"] = time.time() - 1
        reg.age_indicators()

        assert reg.reactivate(ind_id) is True
        retrieved = reg.get_indicator(ind_id)
        assert retrieved["status"] == "active"

    def test_reactivate_nonexistent_returns_false(self):
        reg = IndicatorRegistry()
        assert reg.reactivate("indicator--nope") is False

    def test_reactivate_active_returns_false(self):
        reg = IndicatorRegistry()
        ind = self._make_indicator()
        ind_id = reg.add_indicator(ind)
        assert reg.reactivate(ind_id) is False

    def test_get_indicators_since_excludes_dormant(self):
        reg = IndicatorRegistry(dormant_age_seconds=0)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        ind = self._make_indicator()
        ind_id = reg.add_indicator(ind)
        reg._indicators[ind_id]["_added_at"] = time.time() - 1
        reg.age_indicators()

        results = reg.get_indicators_since(old_time)
        assert len(results) == 0

    def test_stats(self):
        reg = IndicatorRegistry()
        ind1 = self._make_indicator(seed=1, mitre_tactic="AML.T0051")
        ind2 = self._make_indicator(seed=2, mitre_tactic="AML.T0054")
        reg.add_indicator(ind1)
        reg.add_indicator(ind2)

        stats = reg.get_stats()
        assert stats["total"] == 2
        assert stats["active"] == 2
        assert stats["dormant"] == 0
        assert "AML.T0051" in stats["top_tactics"]
        assert stats["newest"] is not None
        assert stats["oldest"] is not None

    def test_stats_empty(self):
        reg = IndicatorRegistry()
        stats = reg.get_stats()
        assert stats["total"] == 0
        assert stats["newest"] is None

    def test_internal_fields_stripped_from_output(self):
        reg = IndicatorRegistry()
        ind = self._make_indicator()
        ind_id = reg.add_indicator(ind)
        retrieved = reg.get_indicator(ind_id)
        assert "_status" not in retrieved
        assert "_added_at" not in retrieved


class TestCosineHelpers:
    """Test cosine similarity and embedding extraction helpers."""

    def test_cosine_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine_similarity(a, b) == 0.0

    def test_extract_embedding_valid(self):
        emb = np.array([1.0, 2.0, 3.0])
        ind = generate_indicator(noised_embedding=emb)
        extracted = _extract_embedding(ind)
        assert extracted is not None
        np.testing.assert_array_almost_equal(extracted, emb)

    def test_extract_embedding_no_pattern(self):
        assert _extract_embedding({}) is None

    def test_extract_embedding_invalid_json(self):
        assert _extract_embedding({"pattern": "not json"}) is None


# ---------------------------------------------------------------------------
# Node Registry Tests
# ---------------------------------------------------------------------------


class TestNodeRegistry:
    """Test federation node tracking."""

    def test_register_node(self):
        reg = NodeRegistry()
        reg.register_node("node-1", {"region": "us-east"})
        node = reg.get_node("node-1")
        assert node is not None
        assert node["node_id"] == "node-1"
        assert node["metadata"]["region"] == "us-east"
        assert node["heartbeat_count"] == 1
        assert node["is_active"] is True

    def test_heartbeat_increments_count(self):
        reg = NodeRegistry()
        reg.register_node("node-1")
        reg.heartbeat("node-1")
        reg.heartbeat("node-1")
        node = reg.get_node("node-1")
        assert node["heartbeat_count"] == 3  # 1 register + 2 heartbeats

    def test_heartbeat_registers_new_node(self):
        reg = NodeRegistry()
        reg.heartbeat("new-node", {"version": "2.0"})
        node = reg.get_node("new-node")
        assert node is not None
        assert node["metadata"]["version"] == "2.0"

    def test_inactive_after_threshold(self):
        reg = NodeRegistry(active_threshold_seconds=10)
        reg.register_node("old-node")
        # Backdate the heartbeat
        reg._nodes["old-node"]["last_heartbeat"] = time.time() - 20
        node = reg.get_node("old-node")
        assert node["is_active"] is False

    def test_get_active_nodes_filters(self):
        reg = NodeRegistry(active_threshold_seconds=10)
        reg.register_node("active-node")
        reg.register_node("stale-node")
        reg._nodes["stale-node"]["last_heartbeat"] = time.time() - 20

        active = reg.get_active_nodes()
        assert len(active) == 1
        assert active[0]["node_id"] == "active-node"

    def test_get_all_nodes(self):
        reg = NodeRegistry(active_threshold_seconds=10)
        reg.register_node("node-a")
        reg.register_node("node-b")
        reg._nodes["node-b"]["last_heartbeat"] = time.time() - 20

        all_nodes = reg.get_all_nodes()
        assert len(all_nodes) == 2

    def test_get_nonexistent_node(self):
        reg = NodeRegistry()
        assert reg.get_node("missing") is None

    def test_network_stats(self):
        reg = NodeRegistry()
        reg.register_node("n1")
        reg.register_node("n2")
        reg.heartbeat("n1")

        stats = reg.get_network_stats()
        assert stats["total_nodes"] == 2
        assert stats["active_nodes"] == 2
        assert stats["total_heartbeats"] == 3  # n1=2, n2=1

    def test_metadata_update_on_heartbeat(self):
        reg = NodeRegistry()
        reg.register_node("n1", {"v": "1.0"})
        reg.heartbeat("n1", {"v": "2.0"})
        node = reg.get_node("n1")
        assert node["metadata"]["v"] == "2.0"


# ---------------------------------------------------------------------------
# Federation Pipeline Tests
# ---------------------------------------------------------------------------


class TestFederationPipeline:
    """Test event bus to indicator generation pipeline."""

    def _make_pipeline(self, **overrides):
        dp_engine = MagicMock()
        dp_engine.is_budget_exhausted = False
        dp_engine.add_noise_to_embedding = lambda emb: emb + np.random.randn(*emb.shape) * 0.01

        federated = MagicMock()
        federated.dp_engine = dp_engine

        registry = IndicatorRegistry()
        event_bus = AsyncMock()
        event_bus.subscribe = AsyncMock()
        event_bus.publish = AsyncMock()

        kwargs = {
            "federated": federated,
            "indicator_registry": registry,
            "event_bus": event_bus,
        }
        kwargs.update(overrides)
        pipeline = FederationPipeline(**kwargs)
        return pipeline, registry, event_bus, dp_engine

    def test_initial_stats(self):
        pipeline, *_ = self._make_pipeline()
        stats = pipeline.stats
        assert stats["running"] is False
        assert stats["indicators_generated"] == 0
        assert stats["events_processed"] == 0

    def test_start_subscribes_to_event_bus(self):
        pipeline, _, event_bus, _ = self._make_pipeline()
        asyncio.get_event_loop().run_until_complete(pipeline.start())
        assert pipeline._running is True
        event_bus.subscribe.assert_awaited_once_with(
            "threat_detected", pipeline._on_threat_detected
        )

    def test_start_idempotent(self):
        pipeline, _, event_bus, _ = self._make_pipeline()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())
        loop.run_until_complete(pipeline.start())
        assert event_bus.subscribe.await_count == 1

    def test_stop_sets_not_running(self):
        pipeline, *_ = self._make_pipeline()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())
        loop.run_until_complete(pipeline.stop())
        assert pipeline._running is False

    def test_novel_attack_generates_indicator(self):
        pipeline, registry, _, _ = self._make_pipeline()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())

        event = SimpleNamespace(payload={
            "is_novel": True,
            "adaptive_caught": True,
            "embedding": np.random.randn(384).tolist(),
            "mitre_tactic": "AML.T0051",
            "confidence": 0.92,
            "detection_layer": "L3",
            "threat_category": "prompt_injection",
        })
        loop.run_until_complete(pipeline._on_threat_detected(event))

        assert pipeline.stats["events_processed"] == 1
        assert pipeline.stats["indicators_generated"] == 1
        assert registry.get_stats()["total"] == 1

    def test_non_novel_attack_skipped(self):
        pipeline, registry, _, _ = self._make_pipeline()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())

        event = SimpleNamespace(payload={
            "is_novel": False,
            "adaptive_caught": False,
            "confidence": 0.5,
        })
        loop.run_until_complete(pipeline._on_threat_detected(event))

        assert pipeline.stats["events_processed"] == 1
        assert pipeline.stats["indicators_generated"] == 0

    def test_dp_budget_exhausted_skips(self):
        pipeline, registry, _, dp_engine = self._make_pipeline()
        dp_engine.is_budget_exhausted = True

        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())

        event = SimpleNamespace(payload={
            "is_novel": True,
            "embedding": np.random.randn(384).tolist(),
        })
        loop.run_until_complete(pipeline._on_threat_detected(event))

        assert pipeline.stats["indicators_generated"] == 0

    def test_hash_based_fallback_embedding(self):
        pipeline, registry, _, _ = self._make_pipeline()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())

        # No embedding provided, no vault — should use hash fallback
        event = SimpleNamespace(payload={
            "is_novel": True,
            "prompt": "test attack prompt",
            "mitre_tactic": "AML.T0054",
            "confidence": 0.88,
        })
        loop.run_until_complete(pipeline._on_threat_detected(event))
        assert pipeline.stats["indicators_generated"] == 1

    def test_pipeline_publishes_indicator_event(self):
        pipeline, _, event_bus, _ = self._make_pipeline()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())

        event = SimpleNamespace(payload={
            "is_novel": True,
            "embedding": np.random.randn(384).tolist(),
        })
        loop.run_until_complete(pipeline._on_threat_detected(event))

        # Should publish indicator_generated event
        assert event_bus.publish.await_count >= 1

    def test_stopped_pipeline_ignores_events(self):
        pipeline, *_ = self._make_pipeline()
        # Don't start — pipeline is not running
        event = SimpleNamespace(payload={"is_novel": True})
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline._on_threat_detected(event))
        assert pipeline.stats["events_processed"] == 0

    def test_dict_event_payload(self):
        """Pipeline handles dict events (no .payload attr)."""
        pipeline, registry, _, _ = self._make_pipeline()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())

        event = {
            "is_novel": True,
            "embedding": np.random.randn(384).tolist(),
            "mitre_tactic": "AML.T0043",
            "confidence": 0.85,
        }
        loop.run_until_complete(pipeline._on_threat_detected(event))
        assert pipeline.stats["indicators_generated"] == 1

    def test_no_federated_returns_none(self):
        pipeline = FederationPipeline(
            federated=None,
            indicator_registry=IndicatorRegistry(),
            event_bus=None,
        )
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            pipeline.generate_indicator_from_detection({"is_novel": True})
        )
        assert result is None

    def test_no_registry_returns_none(self):
        federated = MagicMock()
        pipeline = FederationPipeline(
            federated=federated,
            indicator_registry=None,
            event_bus=None,
        )
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            pipeline.generate_indicator_from_detection({"is_novel": True})
        )
        assert result is None

    def test_start_without_event_bus(self):
        pipeline = FederationPipeline(
            federated=MagicMock(),
            indicator_registry=IndicatorRegistry(),
            event_bus=None,
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pipeline.start())
        assert pipeline._running is True


# ---------------------------------------------------------------------------
# Hub API Tests
# ---------------------------------------------------------------------------


class TestFederationHubAPI:
    """Test federation hub API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import main as m

        self._orig_config = m._config
        self._orig_indicator_registry = getattr(m, "_indicator_registry", None)
        self._orig_node_registry = getattr(m, "_node_registry", None)
        self._orig_federated = getattr(m, "_federated", None)
        self._orig_event_bus = getattr(m, "_event_bus", None)

        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-federation-key"
        m._config = config

        m._indicator_registry = IndicatorRegistry()
        m._node_registry = NodeRegistry()

        # Minimal federated mock with dp_engine
        dp_engine = MagicMock()
        dp_engine.stats = {"budget_remaining": 95.0, "queries_used": 5}
        m._federated = MagicMock()
        m._federated.dp_engine = dp_engine

        m._event_bus = AsyncMock()
        m._event_bus.publish = AsyncMock()

        self.api_key = m._config.api_key
        self.auth = {"Authorization": f"Bearer {self.api_key}"}

        from fastapi.testclient import TestClient
        self.client = TestClient(m.app, raise_server_exceptions=False)
        yield

        m._config = self._orig_config
        m._indicator_registry = self._orig_indicator_registry
        m._node_registry = self._orig_node_registry
        m._federated = self._orig_federated
        m._event_bus = self._orig_event_bus

    def _make_stix_indicator(self, seed: int = 42) -> dict:
        emb = np.random.RandomState(seed).randn(384).astype(np.float64)
        return generate_indicator(
            noised_embedding=emb,
            mitre_tactic="AML.T0051",
            confidence=0.9,
            detection_layer="L3",
        )

    # --- Auth ---

    def test_submit_requires_auth(self):
        resp = self.client.post("/v1/federation/indicators", json={})
        assert resp.status_code == 401

    def test_list_requires_auth(self):
        resp = self.client.get("/v1/federation/indicators")
        assert resp.status_code == 401

    def test_stats_requires_auth(self):
        resp = self.client.get("/v1/federation/stats")
        assert resp.status_code == 401

    def test_heartbeat_requires_auth(self):
        resp = self.client.post("/v1/federation/heartbeat", json={})
        assert resp.status_code == 401

    def test_get_indicator_requires_auth(self):
        resp = self.client.get("/v1/federation/indicators/indicator--abc")
        assert resp.status_code == 401

    # --- POST /v1/federation/indicators ---

    def test_submit_indicator_success(self):
        ind = self._make_stix_indicator()
        resp = self.client.post(
            "/v1/federation/indicators", json=ind, headers=self.auth
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["indicator_id"] == ind["id"]

    def test_submit_rejects_non_indicator_type(self):
        resp = self.client.post(
            "/v1/federation/indicators",
            json={"type": "malware", "id": "indicator--x", "pattern": "{}"},
            headers=self.auth,
        )
        assert resp.status_code == 400
        assert "indicator" in resp.json()["error"].lower()

    def test_submit_rejects_bad_id(self):
        resp = self.client.post(
            "/v1/federation/indicators",
            json={"type": "indicator", "id": "bad-id", "pattern": "{}"},
            headers=self.auth,
        )
        assert resp.status_code == 400

    def test_submit_rejects_missing_pattern(self):
        resp = self.client.post(
            "/v1/federation/indicators",
            json={"type": "indicator", "id": "indicator--abc"},
            headers=self.auth,
        )
        assert resp.status_code == 400

    # --- GET /v1/federation/indicators ---

    def test_list_indicators_empty(self):
        resp = self.client.get("/v1/federation/indicators", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["indicators"] == []
        assert data["total"] == 0

    def test_list_indicators_after_submit(self):
        ind = self._make_stix_indicator()
        self.client.post("/v1/federation/indicators", json=ind, headers=self.auth)

        resp = self.client.get("/v1/federation/indicators", headers=self.auth)
        data = resp.json()
        assert data["total"] == 1
        assert len(data["indicators"]) == 1
        # Embedding stripped, pattern_meta present
        listed = data["indicators"][0]
        assert "pattern" not in listed
        assert "pattern_meta" in listed
        assert listed["pattern_meta"]["embedding_dim"] == 384

    def test_list_with_since_filter(self):
        ind = self._make_stix_indicator()
        self.client.post("/v1/federation/indicators", json=ind, headers=self.auth)

        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        resp = self.client.get(
            f"/v1/federation/indicators?since={future}", headers=self.auth
        )
        assert resp.json()["total"] == 0

    def test_list_with_limit(self):
        for seed in range(5):
            ind = self._make_stix_indicator(seed=seed * 100)
            self.client.post("/v1/federation/indicators", json=ind, headers=self.auth)

        resp = self.client.get(
            "/v1/federation/indicators?limit=2", headers=self.auth
        )
        data = resp.json()
        assert data["total"] == 5
        assert len(data["indicators"]) == 2

    # --- GET /v1/federation/indicators/{id} ---

    def test_get_indicator_success(self):
        ind = self._make_stix_indicator()
        self.client.post("/v1/federation/indicators", json=ind, headers=self.auth)

        resp = self.client.get(
            f"/v1/federation/indicators/{ind['id']}", headers=self.auth
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "indicator"
        assert data["hit_count"] == 1

    def test_get_indicator_not_found(self):
        resp = self.client.get(
            "/v1/federation/indicators/indicator--missing", headers=self.auth
        )
        assert resp.status_code == 404

    # --- GET /v1/federation/stats ---

    def test_stats_response(self):
        resp = self.client.get("/v1/federation/stats", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "indicators" in data
        assert "network" in data
        assert "privacy" in data

    # --- POST /v1/federation/heartbeat ---

    def test_heartbeat_success(self):
        resp = self.client.post(
            "/v1/federation/heartbeat",
            json={"node_id": "node-test-1", "indicators_ingested": 42},
            headers=self.auth,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["network_size"] == 1

    def test_heartbeat_missing_node_id(self):
        resp = self.client.post(
            "/v1/federation/heartbeat",
            json={},
            headers=self.auth,
        )
        assert resp.status_code == 400

    def test_heartbeat_increases_network_size(self):
        self.client.post(
            "/v1/federation/heartbeat",
            json={"node_id": "n1"},
            headers=self.auth,
        )
        self.client.post(
            "/v1/federation/heartbeat",
            json={"node_id": "n2"},
            headers=self.auth,
        )
        resp = self.client.post(
            "/v1/federation/heartbeat",
            json={"node_id": "n3"},
            headers=self.auth,
        )
        assert resp.json()["network_size"] == 3


# ---------------------------------------------------------------------------
# Dashboard Integration Tests
# ---------------------------------------------------------------------------


class TestDashboardFederationIntegration:
    """Test that dashboard endpoints include federation data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import main as m

        self._orig_config = m._config
        self._orig_barrier = m._barrier
        self._orig_indicator_registry = getattr(m, "_indicator_registry", None)
        self._orig_node_registry = getattr(m, "_node_registry", None)
        self._orig_federated = getattr(m, "_federated", None)
        self._orig_buffer = m._dashboard_metrics_buffer

        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-dash-fed-key"
        m._config = config

        if m._barrier is None:
            m._init_layers()

        from aegis.dashboard.metrics_buffer import MetricsBuffer
        if m._dashboard_metrics_buffer is None:
            m._dashboard_metrics_buffer = MetricsBuffer()

        m._indicator_registry = IndicatorRegistry()
        m._node_registry = NodeRegistry()

        dp_engine = MagicMock()
        dp_engine.get_budget_status = MagicMock(return_value=MagicMock(
            to_dict=lambda: {"budget_remaining": 90.0, "queries_used": 10}
        ))
        m._federated = MagicMock()
        m._federated.dp_engine = dp_engine
        m._federated.stats = {
            "indicators_shared": 5,
            "indicators_received": 3,
            "privacy_budget_remaining": 90.0,
        }

        self.api_key = m._config.api_key
        self.auth = {"Authorization": f"Bearer {self.api_key}"}

        from fastapi.testclient import TestClient
        self.client = TestClient(m.app, raise_server_exceptions=False)
        yield

        m._config = self._orig_config
        m._barrier = self._orig_barrier
        m._indicator_registry = self._orig_indicator_registry
        m._node_registry = self._orig_node_registry
        m._federated = self._orig_federated
        m._dashboard_metrics_buffer = self._orig_buffer

    def test_overview_includes_federation(self):
        resp = self.client.get("/dashboard/api/overview", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "federation" in data
        fed = data["federation"]
        assert "status" in fed
        assert "indicators_shared" in fed
        assert "active_nodes" in fed
        assert "privacy_budget_remaining" in fed

    def test_federation_dashboard_endpoint(self):
        resp = self.client.get("/dashboard/api/federation", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "indicators" in data
        assert "nodes" in data
        assert "privacy_budget" in data
        assert "pipeline" in data
        assert "registry_stats" in data
        assert "network_stats" in data

    def test_federation_dashboard_requires_auth(self):
        resp = self.client.get("/dashboard/api/federation")
        assert resp.status_code == 401

    def test_federation_overview_with_nodes(self):
        import main as m
        m._node_registry.register_node("test-node-1")
        m._node_registry.register_node("test-node-2")

        resp = self.client.get("/dashboard/api/overview", headers=self.auth)
        data = resp.json()
        assert data["federation"]["active_nodes"] == 2

    def test_federation_dashboard_with_indicators(self):
        import main as m
        emb = np.random.randn(384)
        ind = generate_indicator(
            noised_embedding=emb, mitre_tactic="AML.T0051", confidence=0.9
        )
        m._indicator_registry.add_indicator(ind)

        resp = self.client.get("/dashboard/api/federation", headers=self.auth)
        data = resp.json()
        assert data["registry_stats"]["total"] == 1
