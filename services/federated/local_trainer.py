"""
Local Trainer — Federated Learning Local Model Training.

Biological Analog: Each lymph node independently trains immune cells against
locally encountered pathogens. The trained cells' *knowledge* (not the
pathogens themselves) is shared with the broader immune network.

Each AEGIS instance trains a local prompt injection classifier on its own data
using logistic regression on TF-IDF features. Only model weight updates are
shared — never training data — preserving tenant privacy.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LocalModelUpdate:
    """Model update from a local training round.

    Contains only weights and metadata — never raw training data.
    This is the unit shared in federated averaging.
    """
    instance_id: str
    round_number: int
    weights: dict[str, np.ndarray]
    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "round_number": self.round_number,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "weight_shapes": {k: list(v.shape) for k, v in self.weights.items()},
        }


class LocalTrainer:
    """Trains a local prompt injection classifier using logistic regression on TF-IDF.

    This is a lightweight classifier suitable for federated learning. The full
    DeBERTa model in L3 remains the primary classifier; this model supplements
    it with federated knowledge from across the network.

    The trainer maintains a vocabulary (TF-IDF feature names) and learns
    logistic regression weights over those features.
    """

    def __init__(
        self,
        *,
        instance_id: str = "local",
        vocab_size: int = 5000,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        regularization: float = 0.01,
    ) -> None:
        self._instance_id = instance_id
        self._vocab_size = vocab_size
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._regularization = regularization

        # Model parameters (initialized on first train)
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._vocabulary: list[str] = []
        self._round_number: int = 0

        # Training data buffer
        self._training_texts: list[str] = []
        self._training_labels: list[int] = []  # 1 = threat, 0 = benign

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def round_number(self) -> int:
        return self._round_number

    @property
    def has_model(self) -> bool:
        return self._weights is not None

    @property
    def training_samples(self) -> int:
        return len(self._training_labels)

    def add_training_sample(self, text: str, is_threat: bool) -> None:
        """Add a labeled sample to the training buffer.

        Args:
            text: The prompt text.
            is_threat: True if this is a confirmed threat, False if benign.
        """
        self._training_texts.append(text)
        self._training_labels.append(1 if is_threat else 0)

    def add_training_batch(self, texts: list[str], labels: list[int]) -> None:
        """Add a batch of labeled samples."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        self._training_texts.extend(texts)
        self._training_labels.extend(labels)

    def _build_tfidf(self, texts: list[str]) -> np.ndarray:
        """Build TF-IDF feature matrix from texts.

        Simple implementation: character n-grams (3-5) as features.
        Uses hashing trick to keep vocabulary bounded.
        """
        n_docs = len(texts)
        features = np.zeros((n_docs, self._vocab_size), dtype=np.float64)

        for i, text in enumerate(texts):
            text_lower = text.lower()
            # Extract character n-grams (3 to 5)
            ngrams: dict[int, int] = {}
            for n in range(3, 6):
                for j in range(len(text_lower) - n + 1):
                    gram = text_lower[j:j + n]
                    h = int(hashlib.md5(gram.encode()).hexdigest(), 16) % self._vocab_size
                    ngrams[h] = ngrams.get(h, 0) + 1

            # TF: normalize by document length
            total = sum(ngrams.values()) or 1
            for h, count in ngrams.items():
                features[i, h] = count / total

        # IDF: log(N / (1 + df))
        doc_freq = (features > 0).sum(axis=0) + 1  # +1 smoothing
        idf = np.log(n_docs / doc_freq)
        features *= idf

        return features

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )

    def train(self) -> LocalModelUpdate:
        """Train on buffered data and return a model update.

        Returns:
            LocalModelUpdate with trained weights.

        Raises:
            ValueError: If insufficient training data.
        """
        if len(self._training_labels) < 2:
            raise ValueError("Need at least 2 training samples")

        n_samples = len(self._training_labels)
        X = self._build_tfidf(self._training_texts)
        y = np.array(self._training_labels, dtype=np.float64)

        # Initialize weights if needed
        if self._weights is None or len(self._weights) != self._vocab_size:
            self._weights = np.zeros(self._vocab_size, dtype=np.float64)
            self._bias = 0.0

        # Gradient descent with L2 regularization
        for _ in range(self._max_iterations):
            z = X @ self._weights + self._bias
            predictions = self._sigmoid(z)
            error = predictions - y

            # Gradients
            grad_w = (X.T @ error) / n_samples + self._regularization * self._weights
            grad_b = error.mean()

            self._weights -= self._learning_rate * grad_w
            self._bias -= self._learning_rate * grad_b

        # Compute metrics
        final_predictions = self._sigmoid(X @ self._weights + self._bias)
        binary_preds = (final_predictions >= 0.5).astype(int)
        accuracy = (binary_preds == y).mean()
        loss = -np.mean(
            y * np.log(final_predictions + 1e-10)
            + (1 - y) * np.log(1 - final_predictions + 1e-10)
        )

        self._round_number += 1

        update = LocalModelUpdate(
            instance_id=self._instance_id,
            round_number=self._round_number,
            weights={"classifier": self._weights.copy(), "bias": np.array([self._bias])},
            num_samples=n_samples,
            metrics={"accuracy": float(accuracy), "loss": float(loss)},
        )

        logger.info(
            "Local training round %d: %d samples, accuracy=%.3f, loss=%.4f",
            self._round_number, n_samples, accuracy, loss,
        )

        return update

    def apply_global_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Apply aggregated global weights from FedAvg.

        Args:
            weights: Dict with 'classifier' and 'bias' arrays.
        """
        if "classifier" in weights:
            self._weights = weights["classifier"].copy()
        if "bias" in weights:
            self._bias = float(weights["bias"][0])
        logger.info("Applied global weights from federated round")

    def predict(self, text: str) -> float:
        """Predict threat probability for a single text.

        Args:
            text: Input prompt text.

        Returns:
            Probability of threat (0.0 to 1.0).
        """
        if self._weights is None:
            return 0.5  # No model yet, return uncertain

        X = self._build_tfidf([text])
        z = X @ self._weights + self._bias
        return float(self._sigmoid(z)[0])

    def clear_training_buffer(self) -> None:
        """Clear training data buffer after a round."""
        self._training_texts.clear()
        self._training_labels.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Trainer statistics."""
        return {
            "instance_id": self._instance_id,
            "round_number": self._round_number,
            "has_model": self.has_model,
            "vocab_size": self._vocab_size,
            "training_samples_buffered": len(self._training_labels),
            "total_threat_samples": sum(self._training_labels),
            "total_benign_samples": len(self._training_labels) - sum(self._training_labels),
        }
