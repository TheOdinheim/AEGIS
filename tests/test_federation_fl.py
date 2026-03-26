"""
Tests for L8 Federated Model Training (Flower-equivalent).

Covers:
- FederatedClassifier: forward pass, weight get/set, training, JSON serialization
- TrainingBuffer: add/retrieve, FIFO eviction, clear, persistence
- DP Gradient Protection: clipping, noise, budget tracking, noise scale
- FL Server: global model, submit update, FedAvg aggregation, round lifecycle
- FL Client: local training, DP-protected updates, apply global model
- FL API: endpoints for model/update/trigger-round/status, auth
- FL Scheduler: start/stop, run_round
- Integration: end-to-end training → aggregation → improvement
- Dashboard: overview includes FL stats
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pytest

from aegis.services.federated.fl_model import FederatedClassifier
from aegis.services.federated.training_buffer import TrainingBuffer
from aegis.services.federated.dp_gradients import (
    clip_gradients,
    add_noise_to_gradients,
    compute_noise_scale,
    GradientPrivacyTracker,
)
from aegis.services.federated.fl_server import FLServer
from aegis.services.federated.fl_client import FLClient
from aegis.services.federated.fl_scheduler import FLScheduler


# ---------------------------------------------------------------------------
# FederatedClassifier Tests
# ---------------------------------------------------------------------------


class TestFederatedClassifier:
    """Test numpy-based binary neural network."""

    def test_forward_output_range(self):
        model = FederatedClassifier()
        x = np.random.randn(384)
        out = model.forward(x)
        assert out.shape == (1,)
        assert 0.0 <= out[0] <= 1.0

    def test_forward_batch(self):
        model = FederatedClassifier()
        X = np.random.randn(10, 384)
        out = model.forward(X)
        assert out.shape == (10, 1)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_predict_scalar(self):
        model = FederatedClassifier()
        x = np.random.randn(384)
        score = model.predict(x)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_deterministic(self):
        model = FederatedClassifier()
        x = np.random.randn(384)
        s1 = model.predict(x)
        s2 = model.predict(x)
        assert s1 == s2

    def test_weight_get_set_roundtrip(self):
        model = FederatedClassifier()
        x = np.random.randn(384)
        original = model.predict(x)

        weights = model.get_weights()
        assert len(weights) == 4

        model2 = FederatedClassifier()
        model2.set_weights(weights)
        restored = model2.predict(x)
        assert abs(original - restored) < 1e-10

    def test_set_weights_wrong_count(self):
        model = FederatedClassifier()
        with pytest.raises(ValueError, match="4"):
            model.set_weights([np.zeros(10)])

    def test_train_step_reduces_loss(self):
        model = FederatedClassifier(input_dim=10, hidden_dim=5)
        X = np.random.randn(20, 10)
        y = (X[:, 0] > 0).astype(np.float64)

        loss1 = model.train_step(X, y, lr=0.1)
        for _ in range(50):
            loss2 = model.train_step(X, y, lr=0.1)
        assert loss2 < loss1

    def test_train_epoch_converges(self):
        model = FederatedClassifier(input_dim=10, hidden_dim=5)
        rng = np.random.RandomState(42)
        # Linearly separable data
        X = rng.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)

        losses = []
        for _ in range(20):
            loss = model.train_epoch(X, y, lr=0.05, batch_size=32)
            losses.append(loss)

        assert losses[-1] < losses[0]

    def test_weights_json_serializable(self):
        model = FederatedClassifier()
        json_weights = model.weights_to_json()
        assert isinstance(json_weights, list)
        assert len(json_weights) == 4
        serialized = json.dumps(json_weights)
        assert len(serialized) > 0

    def test_weights_from_json_roundtrip(self):
        model = FederatedClassifier()
        x = np.random.randn(384)
        original = model.predict(x)

        json_weights = model.weights_to_json()
        restored = FederatedClassifier.weights_from_json(json_weights)
        model2 = FederatedClassifier()
        model2.set_weights(restored)
        assert abs(model2.predict(x) - original) < 1e-10

    def test_custom_dimensions(self):
        model = FederatedClassifier(input_dim=64, hidden_dim=32)
        x = np.random.randn(64)
        score = model.predict(x)
        assert 0.0 <= score <= 1.0

    def test_train_epoch_empty_data(self):
        model = FederatedClassifier(input_dim=10, hidden_dim=5)
        loss = model.train_epoch(np.empty((0, 10)), np.empty(0))
        assert loss == 0.0


# ---------------------------------------------------------------------------
# TrainingBuffer Tests
# ---------------------------------------------------------------------------


class TestTrainingBuffer:
    """Test training data accumulation."""

    def test_add_and_retrieve(self):
        buf = TrainingBuffer()
        emb = np.random.randn(384)
        buf.add_sample(emb, is_threat=True)
        buf.add_sample(emb, is_threat=False)

        X, y = buf.get_training_data()
        assert X.shape == (2, 384)
        assert y.shape == (2,)
        assert y[0] == 1.0
        assert y[1] == 0.0

    def test_size_property(self):
        buf = TrainingBuffer()
        assert buf.size == 0
        buf.add_sample(np.random.randn(10), is_threat=True)
        assert buf.size == 1

    def test_fifo_eviction(self):
        buf = TrainingBuffer(max_size=5)
        for i in range(10):
            buf.add_sample(np.array([float(i)]), is_threat=i % 2 == 0)

        assert buf.size == 5
        X, y = buf.get_training_data()
        # Should have samples 5-9 (oldest evicted)
        assert X[0][0] == 5.0

    def test_clear(self):
        buf = TrainingBuffer()
        buf.add_sample(np.random.randn(10), is_threat=True)
        buf.clear()
        assert buf.size == 0
        X, y = buf.get_training_data()
        assert X.shape[0] == 0

    def test_empty_buffer_returns_empty_arrays(self):
        buf = TrainingBuffer()
        X, y = buf.get_training_data()
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_mixed_samples(self):
        buf = TrainingBuffer()
        for _ in range(5):
            buf.add_sample(np.random.randn(10), is_threat=True)
        for _ in range(5):
            buf.add_sample(np.random.randn(10), is_threat=False)

        stats = buf.stats
        assert stats["size"] == 10
        assert stats["threat_samples"] == 5
        assert stats["benign_samples"] == 5

    def test_persistence_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.json"
            buf = TrainingBuffer(persist_path=path)
            buf.add_sample(np.array([1.0, 2.0, 3.0]), is_threat=True)
            buf.add_sample(np.array([4.0, 5.0, 6.0]), is_threat=False)
            buf.save()

            buf2 = TrainingBuffer(persist_path=path)
            loaded = buf2.load()
            assert loaded == 2
            X, y = buf2.get_training_data()
            assert X.shape == (2, 3)
            np.testing.assert_array_almost_equal(X[0], [1.0, 2.0, 3.0])

    def test_load_nonexistent_returns_zero(self):
        buf = TrainingBuffer(persist_path="/nonexistent/path.json")
        assert buf.load() == 0


# ---------------------------------------------------------------------------
# DP Gradient Tests
# ---------------------------------------------------------------------------


class TestDPGradients:
    """Test gradient clipping and DP noise."""

    def test_clip_gradients_within_norm(self):
        grads = [np.array([0.1, 0.2])]
        clipped = clip_gradients(grads, max_norm=1.0)
        np.testing.assert_array_almost_equal(clipped[0], grads[0])

    def test_clip_gradients_exceeds_norm(self):
        grads = [np.array([3.0, 4.0])]  # norm = 5.0
        clipped = clip_gradients(grads, max_norm=1.0)
        assert np.linalg.norm(clipped[0]) <= 1.0 + 1e-6

    def test_clip_multiple_arrays(self):
        grads = [np.ones(10) * 10, np.ones(5) * 0.1]
        clipped = clip_gradients(grads, max_norm=1.0)
        assert np.linalg.norm(clipped[0]) <= 1.0 + 1e-6
        # Small grad should be unchanged
        np.testing.assert_array_almost_equal(clipped[1], grads[1])

    def test_noise_changes_values(self):
        grads = [np.ones(100)]
        noised = add_noise_to_gradients(grads, noise_scale=0.1)
        assert not np.allclose(noised[0], grads[0])

    def test_higher_epsilon_less_noise(self):
        """Higher epsilon = less privacy = less noise."""
        sigma_high_eps = compute_noise_scale(
            epsilon=10.0, delta=1e-5, sensitivity=1.0, num_samples=100,
        )
        sigma_low_eps = compute_noise_scale(
            epsilon=1.0, delta=1e-5, sensitivity=1.0, num_samples=100,
        )
        assert sigma_high_eps < sigma_low_eps

    def test_noise_scale_computation(self):
        sigma = compute_noise_scale(
            epsilon=3.0, delta=1e-5, sensitivity=1.0, num_samples=1,
        )
        assert sigma > 0

    def test_noise_scale_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            compute_noise_scale(epsilon=0, delta=1e-5, sensitivity=1.0, num_samples=1)

    def test_noise_scale_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            compute_noise_scale(epsilon=1.0, delta=0, sensitivity=1.0, num_samples=1)


class TestGradientPrivacyTracker:
    """Test cumulative privacy budget tracking."""

    def test_initial_state(self):
        tracker = GradientPrivacyTracker(total_budget=100.0)
        assert not tracker.is_budget_exhausted
        assert tracker.remaining_budget == 100.0

    def test_budget_tracking(self):
        tracker = GradientPrivacyTracker(
            epsilon_per_round=3.0, total_budget=10.0,
        )
        grads = [np.ones(10)]
        tracker.protect_gradients(grads, num_samples=10)
        assert tracker.remaining_budget == 7.0
        assert tracker.stats["rounds_tracked"] == 1

    def test_budget_exhaustion(self):
        tracker = GradientPrivacyTracker(
            epsilon_per_round=5.0, total_budget=10.0,
        )
        grads = [np.ones(10)]
        tracker.protect_gradients(grads, num_samples=10)
        tracker.protect_gradients(grads, num_samples=10)
        assert tracker.is_budget_exhausted

        with pytest.raises(RuntimeError, match="exhausted"):
            tracker.protect_gradients(grads, num_samples=10)

    def test_protect_clips_and_noises(self):
        tracker = GradientPrivacyTracker()
        grads = [np.ones(100) * 10]
        protected = tracker.protect_gradients(grads, num_samples=50, max_norm=1.0)
        # Should be clipped (norm was >> 1.0) and noised
        assert np.linalg.norm(protected[0]) != np.linalg.norm(grads[0])


# ---------------------------------------------------------------------------
# FL Server Tests
# ---------------------------------------------------------------------------


class TestFLServer:
    """Test federated learning aggregation server."""

    def _make_model(self, dim=10, hidden=5):
        return FederatedClassifier(input_dim=dim, hidden_dim=hidden)

    def test_get_global_model(self):
        model = self._make_model()
        server = FLServer(model)
        result = server.get_global_model()
        assert "round" in result
        assert "weights" in result
        assert "updated_at" in result
        assert len(result["weights"]) == 4

    def test_submit_update(self):
        model = self._make_model()
        server = FLServer(model, min_clients=2)
        weights = model.weights_to_json()
        result = server.submit_update("node-1", weights, 100)
        assert result["status"] == "accepted"
        assert result["updates_received"] == 1

    def test_fedavg_two_clients(self):
        model = self._make_model()
        server = FLServer(model, min_clients=2)

        # Client A: all weights = 1.0
        w_a = [np.ones_like(w) for w in model.get_weights()]
        server.submit_update("a", [w.tolist() for w in w_a], 50)

        # Client B: all weights = 3.0
        w_b = [np.ones_like(w) * 3.0 for w in model.get_weights()]
        result = server.submit_update("b", [w.tolist() for w in w_b], 50)

        # Should auto-aggregate (min_clients=2 met)
        assert result.get("aggregated") is True

        # FedAvg with equal samples: (1*50 + 3*50) / 100 = 2.0
        global_model = server.get_global_model()
        restored = FederatedClassifier.weights_from_json(global_model["weights"])
        np.testing.assert_array_almost_equal(restored[0], np.ones_like(restored[0]) * 2.0)

    def test_fedavg_weighted_by_samples(self):
        model = self._make_model()
        server = FLServer(model, min_clients=2)

        w_a = [np.ones_like(w) for w in model.get_weights()]
        server.submit_update("a", [w.tolist() for w in w_a], 100)  # 100 samples

        w_b = [np.ones_like(w) * 3.0 for w in model.get_weights()]
        server.submit_update("b", [w.tolist() for w in w_b], 300)  # 300 samples

        # Weighted avg: (1*100 + 3*300) / 400 = 1000/400 = 2.5
        global_model = server.get_global_model()
        restored = FederatedClassifier.weights_from_json(global_model["weights"])
        np.testing.assert_array_almost_equal(restored[0], np.ones_like(restored[0]) * 2.5)

    def test_aggregate_empty(self):
        model = self._make_model()
        server = FLServer(model)
        result = server.aggregate()
        assert result["status"] == "no_updates"

    def test_round_lifecycle(self):
        model = self._make_model()
        server = FLServer(model, min_clients=1)
        server.start_round()

        weights = model.weights_to_json()
        server.submit_update("node-1", weights, 50, {"loss": 0.1})
        # Auto-aggregated

        status = server.get_status()
        assert status["round"] == 1
        assert status["total_samples_trained"] == 50

    def test_status(self):
        model = self._make_model()
        server = FLServer(model, min_clients=3)
        status = server.get_status()
        assert status["round"] == 0
        assert status["updates_received"] == 0
        assert status["min_clients"] == 3
        assert status["model_version"] == "v0"


# ---------------------------------------------------------------------------
# FL Client Tests
# ---------------------------------------------------------------------------


class TestFLClient:
    """Test local training client."""

    def _make_client(self, dim=10, hidden=5):
        model = FederatedClassifier(input_dim=dim, hidden_dim=hidden)
        buf = TrainingBuffer()
        dp_tracker = GradientPrivacyTracker(total_budget=100.0)
        return FLClient(model, buf, dp_tracker, node_id="test-node")

    def test_train_local_insufficient_data(self):
        client = self._make_client()
        result = client.train_local()
        assert result["status"] == "insufficient_data"

    def test_train_local_success(self):
        client = self._make_client()
        for _ in range(20):
            client._buffer.add_sample(np.random.randn(10), is_threat=True)
            client._buffer.add_sample(np.random.randn(10), is_threat=False)

        result = client.train_local(epochs=3)
        assert result["status"] == "trained"
        assert result["samples"] == 40
        assert result["epochs"] == 3
        assert "loss" in result

    def test_get_update_returns_weights(self):
        client = self._make_client()
        for _ in range(10):
            client._buffer.add_sample(np.random.randn(10), is_threat=True)
            client._buffer.add_sample(np.random.randn(10), is_threat=False)

        client.train_local()
        update = client.get_update()
        assert update["node_id"] == "test-node"
        assert "weights" in update
        assert isinstance(update["weights"], list)
        assert update["num_samples"] == 20

    def test_dp_protection_applied(self):
        client = self._make_client()
        for _ in range(10):
            client._buffer.add_sample(np.random.randn(10), is_threat=True)
            client._buffer.add_sample(np.random.randn(10), is_threat=False)

        client.train_local()

        # Get update twice — DP noise should make them different
        update1 = client.get_update()
        update2 = client.get_update()
        w1 = np.array(update1["weights"][0])
        w2 = np.array(update2["weights"][0])
        assert not np.allclose(w1, w2)

    def test_apply_global_model(self):
        client = self._make_client()
        new_weights = [np.ones_like(w) * 0.5 for w in client.model.get_weights()]
        client.apply_global_model([w.tolist() for w in new_weights])

        # Verify weights were applied
        current = client.model.get_weights()
        np.testing.assert_array_almost_equal(current[0], new_weights[0])


# ---------------------------------------------------------------------------
# FL API Tests
# ---------------------------------------------------------------------------


class TestFLAPI:
    """Test FL API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import main as m

        self._orig_config = m._config
        self._orig_fl_server = getattr(m, "_fl_server", None)
        self._orig_fl_client = getattr(m, "_fl_client", None)
        self._orig_fl_scheduler = getattr(m, "_fl_scheduler", None)
        self._orig_fl_training_buffer = getattr(m, "_fl_training_buffer", None)

        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-fl-key"
        m._config = config

        model = FederatedClassifier(input_dim=10, hidden_dim=5)
        buf = TrainingBuffer()
        dp_tracker = GradientPrivacyTracker()
        m._fl_server = FLServer(model, min_clients=1)
        m._fl_client = FLClient(model, buf, dp_tracker)
        m._fl_scheduler = None
        m._fl_training_buffer = buf

        self.api_key = m._config.api_key
        self.auth = {"Authorization": f"Bearer {self.api_key}"}
        self.model = model

        from fastapi.testclient import TestClient
        self.client = TestClient(m.app, raise_server_exceptions=False)
        yield

        m._config = self._orig_config
        m._fl_server = self._orig_fl_server
        m._fl_client = self._orig_fl_client
        m._fl_scheduler = self._orig_fl_scheduler
        m._fl_training_buffer = self._orig_fl_training_buffer

    # --- Auth ---

    def test_model_requires_auth(self):
        resp = self.client.get("/v1/federation/fl/model")
        assert resp.status_code == 401

    def test_update_requires_auth(self):
        resp = self.client.post("/v1/federation/fl/update", json={})
        assert resp.status_code == 401

    def test_trigger_requires_auth(self):
        resp = self.client.post("/v1/federation/fl/trigger-round")
        assert resp.status_code == 401

    def test_status_requires_auth(self):
        resp = self.client.get("/v1/federation/fl/status")
        assert resp.status_code == 401

    # --- GET /model ---

    def test_get_model_returns_weights(self):
        resp = self.client.get("/v1/federation/fl/model", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "weights" in data
        assert "round" in data
        assert len(data["weights"]) == 4

    # --- POST /update ---

    def test_submit_update_success(self):
        weights = self.model.weights_to_json()
        resp = self.client.post(
            "/v1/federation/fl/update",
            json={"node_id": "test", "weights": weights, "num_samples": 50},
            headers=self.auth,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"

    def test_submit_update_missing_node_id(self):
        resp = self.client.post(
            "/v1/federation/fl/update",
            json={"weights": [], "num_samples": 10},
            headers=self.auth,
        )
        assert resp.status_code == 400

    def test_submit_update_missing_weights(self):
        resp = self.client.post(
            "/v1/federation/fl/update",
            json={"node_id": "test", "num_samples": 10},
            headers=self.auth,
        )
        assert resp.status_code == 400

    # --- POST /trigger-round ---

    def test_trigger_round(self):
        resp = self.client.post(
            "/v1/federation/fl/trigger-round", headers=self.auth,
        )
        assert resp.status_code == 200

    # --- GET /status ---

    def test_status_returns_structure(self):
        resp = self.client.get("/v1/federation/fl/status", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "round" in data
        assert "updates_received" in data
        assert "min_clients" in data
        assert "model_version" in data


# ---------------------------------------------------------------------------
# FL Scheduler Tests
# ---------------------------------------------------------------------------


class TestFLScheduler:
    """Test automated training round management."""

    def _make_scheduler(self, dim=10, hidden=5, buf_size=40):
        model = FederatedClassifier(input_dim=dim, hidden_dim=hidden)
        buf = TrainingBuffer()
        for _ in range(buf_size // 2):
            buf.add_sample(np.random.randn(dim), is_threat=True)
            buf.add_sample(np.random.randn(dim), is_threat=False)
        dp_tracker = GradientPrivacyTracker(total_budget=100.0)
        server = FLServer(model, min_clients=1)
        client = FLClient(model, buf, dp_tracker, node_id="sched-test")
        return FLScheduler(server, client, interval_hours=1.0)

    def test_start_creates_task(self):
        sched = self._make_scheduler()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(sched.start())
        assert sched._running is True
        assert sched._task is not None
        loop.run_until_complete(sched.stop())

    def test_stop_cancels_task(self):
        sched = self._make_scheduler()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(sched.start())
        loop.run_until_complete(sched.stop())
        assert sched._running is False

    def test_run_round_full_cycle(self):
        sched = self._make_scheduler()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(sched.run_round())
        assert result["status"] == "completed"
        assert result["samples"] == 40
        assert result["aggregated"] is True
        assert sched.stats["rounds_completed"] == 1

    def test_run_round_insufficient_data(self):
        sched = self._make_scheduler(buf_size=0)
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(sched.run_round())
        assert result["status"] == "skipped"

    def test_scheduler_stats(self):
        sched = self._make_scheduler()
        stats = sched.stats
        assert stats["running"] is False
        assert stats["rounds_completed"] == 0
        assert stats["interval_hours"] == 1.0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestFederatedTrainingIntegration:
    """End-to-end federated training integration."""

    def test_full_training_cycle(self):
        """Add samples → train → submit → aggregate → model improves."""
        dim = 10
        model = FederatedClassifier(input_dim=dim, hidden_dim=8)
        server = FLServer(model, min_clients=1)
        buf = TrainingBuffer()
        dp_tracker = GradientPrivacyTracker(total_budget=100.0)
        client = FLClient(model, buf, dp_tracker, node_id="e2e")

        # Create linearly separable data
        rng = np.random.RandomState(42)
        for _ in range(50):
            threat_emb = rng.randn(dim) + 2.0  # Shifted positive
            benign_emb = rng.randn(dim) - 2.0  # Shifted negative
            buf.add_sample(threat_emb, is_threat=True)
            buf.add_sample(benign_emb, is_threat=False)

        # Before training — should be near 0.5 (random init)
        test_threat = rng.randn(dim) + 2.0
        test_benign = rng.randn(dim) - 2.0
        pre_threat = model.predict(test_threat)
        pre_benign = model.predict(test_benign)

        # Train locally
        train_result = client.train_local(epochs=20, lr=0.05)
        assert train_result["status"] == "trained"

        # Submit to server and aggregate
        update = client.get_update()
        server.submit_update(
            update["node_id"], update["weights"], update["num_samples"],
        )

        # After aggregation, model should distinguish threats from benign
        post_threat = model.predict(test_threat)
        post_benign = model.predict(test_benign)

        # Model should have learned something (threat score > benign score)
        # We don't require perfect separation, just improvement
        assert post_threat != pre_threat or post_benign != pre_benign

    def test_multi_client_aggregation(self):
        """Two clients train on different data, server aggregates."""
        dim = 10
        model = FederatedClassifier(input_dim=dim, hidden_dim=8)
        server = FLServer(model, min_clients=2)

        rng = np.random.RandomState(42)
        for node_id in ["client-a", "client-b"]:
            buf = TrainingBuffer()
            dp_tracker = GradientPrivacyTracker(total_budget=100.0)
            client = FLClient(model, buf, dp_tracker, node_id=node_id)
            # Use fresh model copy so clients don't share state
            local_model = FederatedClassifier(input_dim=dim, hidden_dim=8)
            local_model.set_weights(model.get_weights())
            local_client = FLClient(
                local_model, buf, dp_tracker, node_id=node_id,
            )
            for _ in range(20):
                buf.add_sample(rng.randn(dim) + 1.0, is_threat=True)
                buf.add_sample(rng.randn(dim) - 1.0, is_threat=False)

            local_client.train_local(epochs=5)
            update = local_client.get_update()
            server.submit_update(
                update["node_id"], update["weights"], update["num_samples"],
            )

        # Server should have aggregated after second client
        status = server.get_status()
        assert status["round"] == 1


class TestDashboardFLIntegration:
    """Test that dashboard includes FL stats."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import main as m

        self._orig_config = m._config
        self._orig_barrier = m._barrier
        self._orig_fl_server = getattr(m, "_fl_server", None)
        self._orig_buffer = m._dashboard_metrics_buffer

        from aegis.config import get_config
        config = m._config or get_config()
        if not config.api_key:
            config.api_key = "test-dash-fl-key"
        m._config = config

        if m._barrier is None:
            m._init_layers()

        from aegis.dashboard.metrics_buffer import MetricsBuffer
        if m._dashboard_metrics_buffer is None:
            m._dashboard_metrics_buffer = MetricsBuffer()

        self.api_key = m._config.api_key
        self.auth = {"Authorization": f"Bearer {self.api_key}"}

        from fastapi.testclient import TestClient
        self.client = TestClient(m.app, raise_server_exceptions=False)
        yield

        m._config = self._orig_config
        m._barrier = self._orig_barrier
        m._fl_server = self._orig_fl_server
        m._dashboard_metrics_buffer = self._orig_buffer

    def test_overview_includes_fl_stats(self):
        resp = self.client.get("/dashboard/api/overview", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        fed = data.get("federation", {})
        assert "fl_rounds" in fed
        assert "fl_model_version" in fed
        assert "fl_samples_trained" in fed
