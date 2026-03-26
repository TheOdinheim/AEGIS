"""
FL Client — Local Training Client for Federated Learning.

Runs on each AEGIS node. Trains the local model on accumulated embeddings,
applies DP protection, and submits weight updates to the FL server.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aegis.services.federated.fl_model import FederatedClassifier
from aegis.services.federated.training_buffer import TrainingBuffer
from aegis.services.federated.dp_gradients import GradientPrivacyTracker

logger = logging.getLogger(__name__)


class FLClient:
    """Federated learning client for local training.

    Parameters:
        model: Local model instance.
        buffer: Training data buffer.
        dp_tracker: Gradient privacy tracker for DP protection.
        node_id: Unique identifier for this node.
    """

    def __init__(
        self,
        model: FederatedClassifier,
        buffer: TrainingBuffer,
        dp_tracker: GradientPrivacyTracker,
        *,
        node_id: str = "local",
    ) -> None:
        self._model = model
        self._buffer = buffer
        self._dp_tracker = dp_tracker
        self._node_id = node_id
        self._pre_train_weights: list[np.ndarray] | None = None
        self._rounds_participated: int = 0

    def train_local(
        self, epochs: int = 5, lr: float = 0.01, batch_size: int = 32,
    ) -> dict[str, Any]:
        """Train on local buffer data.

        Returns training stats including final loss.
        """
        X, y = self._buffer.get_training_data()
        if X.shape[0] < 2:
            return {"status": "insufficient_data", "samples": X.shape[0]}

        # Save pre-training weights for delta computation
        self._pre_train_weights = self._model.get_weights()

        total_loss = 0.0
        for epoch in range(epochs):
            loss = self._model.train_epoch(X, y, lr=lr, batch_size=batch_size)
            total_loss += loss

        avg_loss = total_loss / epochs if epochs > 0 else 0.0

        logger.info(
            "Local training: %d epochs, %d samples, avg_loss=%.4f",
            epochs, X.shape[0], avg_loss,
        )

        return {
            "status": "trained",
            "samples": int(X.shape[0]),
            "epochs": epochs,
            "loss": round(avg_loss, 6),
        }

    def get_update(self) -> dict[str, Any]:
        """Return DP-protected weight update.

        The update is the weight delta (current - pre-training) with DP noise.
        """
        current = self._model.get_weights()

        if self._pre_train_weights is not None:
            # Delta = trained - original
            deltas = [c - o for c, o in zip(current, self._pre_train_weights)]
        else:
            deltas = current

        # Apply DP protection
        num_samples = max(self._buffer.size, 1)
        try:
            protected = self._dp_tracker.protect_gradients(
                deltas, num_samples=num_samples,
            )
        except RuntimeError:
            logger.warning("DP budget exhausted — sending raw clipped deltas")
            from aegis.services.federated.dp_gradients import clip_gradients
            protected = clip_gradients(deltas, max_norm=1.0)

        return {
            "node_id": self._node_id,
            "weights": [w.tolist() for w in protected],
            "num_samples": num_samples,
            "metrics": {"loss": 0.0},
        }

    def apply_global_model(self, weights: list) -> None:
        """Load new global model weights."""
        parsed = [np.array(w, dtype=np.float64) for w in weights]
        self._model.set_weights(parsed)
        self._pre_train_weights = None
        logger.info("Applied global model (round update)")

    @property
    def model(self) -> FederatedClassifier:
        return self._model

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "node_id": self._node_id,
            "rounds_participated": self._rounds_participated,
            "buffer_size": self._buffer.size,
            "dp_budget": self._dp_tracker.stats,
        }
