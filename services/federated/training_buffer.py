"""
Training Buffer — Embedding/Label Accumulation for Federated Learning.

Stores (embedding, label) pairs from confirmed detections for local model
training. FIFO eviction when buffer is full. Optional JSON persistence.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 10_000


class TrainingBuffer:
    """Accumulates (embedding, label) pairs for federated training.

    Parameters:
        max_size: Maximum samples before FIFO eviction.
        persist_path: Optional path for JSON save/load.
    """

    def __init__(
        self,
        *,
        max_size: int = _DEFAULT_MAX_SIZE,
        persist_path: str | Path | None = None,
    ) -> None:
        self._max_size = max_size
        self._persist_path = Path(persist_path) if persist_path else None
        self._embeddings: deque[np.ndarray] = deque(maxlen=max_size)
        self._labels: deque[int] = deque(maxlen=max_size)

    @property
    def size(self) -> int:
        return len(self._labels)

    @property
    def max_size(self) -> int:
        return self._max_size

    def add_sample(self, embedding: np.ndarray, is_threat: bool) -> None:
        """Add a training example. FIFO eviction at max_size."""
        self._embeddings.append(embedding.astype(np.float64))
        self._labels.append(1 if is_threat else 0)

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays. X: (n, dim), y: (n,)."""
        if not self._labels:
            return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
        X = np.array(list(self._embeddings), dtype=np.float64)
        y = np.array(list(self._labels), dtype=np.float64)
        return X, y

    def clear(self) -> None:
        """Reset buffer."""
        self._embeddings.clear()
        self._labels.clear()

    def save(self, path: str | Path | None = None) -> None:
        """Save buffer to JSON file."""
        p = Path(path) if path else self._persist_path
        if not p:
            return
        data = {
            "embeddings": [e.tolist() for e in self._embeddings],
            "labels": list(self._labels),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        logger.debug("Training buffer saved to %s (%d samples)", p, self.size)

    def load(self, path: str | Path | None = None) -> int:
        """Load buffer from JSON file. Returns number of samples loaded."""
        p = Path(path) if path else self._persist_path
        if not p or not p.exists():
            return 0
        try:
            data = json.loads(p.read_text())
            embeddings = data.get("embeddings", [])
            labels = data.get("labels", [])
            count = 0
            for emb, lbl in zip(embeddings, labels):
                self._embeddings.append(np.array(emb, dtype=np.float64))
                self._labels.append(int(lbl))
                count += 1
            logger.debug("Loaded %d samples from %s", count, p)
            return count
        except Exception as e:
            logger.warning("Failed to load training buffer from %s: %s", p, e)
            return 0

    @property
    def stats(self) -> dict[str, Any]:
        n_threat = sum(1 for l in self._labels if l == 1)
        return {
            "size": self.size,
            "max_size": self._max_size,
            "threat_samples": n_threat,
            "benign_samples": self.size - n_threat,
        }
