"""
Federated Classifier — Numpy-Based Binary Neural Network.

A lightweight 2-layer neural network for prompt injection detection.
Takes 384-dim MiniLM embeddings as input, outputs a threat/benign score.

Supplements DeBERTa (L3) with federated knowledge from across the network.
Uses numpy only — no PyTorch, no TensorFlow dependency.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FederatedClassifier:
    """Two-layer binary classifier using numpy.

    Architecture:
        Input (384) → Hidden (128, ReLU) → Output (1, Sigmoid)

    Parameters:
        input_dim: Input dimension (384 for MiniLM embeddings).
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128) -> None:
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        # Xavier initialization
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        limit2 = np.sqrt(6.0 / (hidden_dim + 1))

        rng = np.random.RandomState(42)
        self._W1 = rng.uniform(-limit1, limit1, (input_dim, hidden_dim)).astype(np.float64)
        self._b1 = np.zeros(hidden_dim, dtype=np.float64)
        self._W2 = rng.uniform(-limit2, limit2, (hidden_dim, 1)).astype(np.float64)
        self._b2 = np.zeros(1, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (batch, input_dim) or (input_dim,). Returns (batch, 1) or (1,)."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        # Hidden layer: ReLU
        h = x @ self._W1 + self._b1
        h = np.maximum(h, 0.0)  # ReLU

        # Output layer: Sigmoid
        z = h @ self._W2 + self._b2
        out = self._sigmoid(z)

        return out.flatten() if single else out

    def predict(self, x: np.ndarray) -> float:
        """Single prediction (0-1 score)."""
        return float(self.forward(x).flatten()[0])

    def get_weights(self) -> list[np.ndarray]:
        """Return [W1, b1, W2, b2]."""
        return [self._W1.copy(), self._b1.copy(), self._W2.copy(), self._b2.copy()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Load weights from [W1, b1, W2, b2]."""
        if len(weights) != 4:
            raise ValueError("Expected 4 weight arrays [W1, b1, W2, b2]")
        self._W1 = weights[0].astype(np.float64)
        self._b1 = weights[1].astype(np.float64)
        self._W2 = weights[2].astype(np.float64)
        self._b2 = weights[3].astype(np.float64)

    def weights_to_json(self) -> list[list]:
        """Serialize weights to JSON-compatible lists."""
        return [w.tolist() for w in self.get_weights()]

    @staticmethod
    def weights_from_json(data: list[list]) -> list[np.ndarray]:
        """Deserialize weights from JSON lists."""
        return [np.array(w, dtype=np.float64) for w in data]

    def train_step(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01,
    ) -> float:
        """One batch SGD step. Returns loss."""
        batch_size = X.shape[0]

        # Forward
        h = X @ self._W1 + self._b1
        h_relu = np.maximum(h, 0.0)
        z = h_relu @ self._W2 + self._b2
        y_pred = self._sigmoid(z)

        # Target shape
        y_col = y.reshape(-1, 1) if y.ndim == 1 else y

        # Binary cross-entropy loss
        eps = 1e-12
        loss = -np.mean(
            y_col * np.log(y_pred + eps) + (1 - y_col) * np.log(1 - y_pred + eps)
        )

        # Backward
        dz = (y_pred - y_col) / batch_size  # (batch, 1)

        dW2 = h_relu.T @ dz  # (hidden, 1)
        db2 = dz.sum(axis=0)  # (1,)

        dh_relu = dz @ self._W2.T  # (batch, hidden)
        dh = dh_relu * (h > 0).astype(np.float64)  # ReLU gradient

        dW1 = X.T @ dh  # (input, hidden)
        db1 = dh.sum(axis=0)  # (hidden,)

        # Update
        self._W1 -= lr * dW1
        self._b1 -= lr * db1
        self._W2 -= lr * dW2
        self._b2 -= lr * db2

        return float(loss)

    def train_epoch(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, batch_size: int = 32,
    ) -> float:
        """Train on full dataset. Returns average loss."""
        n = X.shape[0]
        if n == 0:
            return 0.0

        indices = np.arange(n)
        np.random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            loss = self.train_step(X[batch_idx], y[batch_idx], lr=lr)
            total_loss += loss
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )
