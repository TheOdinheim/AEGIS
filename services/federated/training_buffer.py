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
        min_threat_ratio: Minimum fraction reserved for threat samples (RT-P3-007).
    """

    def __init__(
        self,
        *,
        max_size: int = _DEFAULT_MAX_SIZE,
        persist_path: str | Path | None = None,
        min_threat_ratio: float = 0.3,
    ) -> None:
        self._max_size = max_size
        self._persist_path = Path(persist_path) if persist_path else None
        # RT-P3-007: Separate sub-buffers to prevent benign flood evicting threats
        threat_capacity = max(1, int(max_size * min_threat_ratio))
        benign_capacity = max(1, max_size - threat_capacity)
        self._threat_embeddings: deque[np.ndarray] = deque(maxlen=threat_capacity)
        self._threat_labels: deque[int] = deque(maxlen=threat_capacity)
        self._benign_embeddings: deque[np.ndarray] = deque(maxlen=benign_capacity)
        self._benign_labels: deque[int] = deque(maxlen=benign_capacity)
        # Legacy unified view for backward compat
        self._embeddings: deque[np.ndarray] = deque(maxlen=max_size)
        self._labels: deque[int] = deque(maxlen=max_size)

    @property
    def size(self) -> int:
        return len(self._threat_labels) + len(self._benign_labels)

    @property
    def max_size(self) -> int:
        return self._max_size

    def add_sample(self, embedding: np.ndarray, is_threat: bool) -> None:
        """Add a training example. FIFO eviction at max_size.

        RT-P3-007: Threat and benign samples use separate sub-buffers to
        prevent benign flood from evicting all attack training data.
        """
        emb = embedding.astype(np.float64)
        label = 1 if is_threat else 0
        if is_threat:
            self._threat_embeddings.append(emb)
            self._threat_labels.append(label)
        else:
            self._benign_embeddings.append(emb)
            self._benign_labels.append(label)
        # Also maintain unified view
        self._embeddings.append(emb)
        self._labels.append(label)

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays from both sub-buffers. X: (n, dim), y: (n,)."""
        all_emb = list(self._threat_embeddings) + list(self._benign_embeddings)
        all_lbl = list(self._threat_labels) + list(self._benign_labels)
        if not all_lbl:
            return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
        X = np.array(all_emb, dtype=np.float64)
        y = np.array(all_lbl, dtype=np.float64)
        return X, y

    def clear(self) -> None:
        """Reset buffer."""
        self._embeddings.clear()
        self._labels.clear()
        self._threat_embeddings.clear()
        self._threat_labels.clear()
        self._benign_embeddings.clear()
        self._benign_labels.clear()

    def save(self, path: str | Path | None = None) -> None:
        """Save buffer to JSON file."""
        p = Path(path) if path else self._persist_path
        if not p:
            return
        # Save from sub-buffers (authoritative source)
        all_emb = list(self._threat_embeddings) + list(self._benign_embeddings)
        all_lbl = list(self._threat_labels) + list(self._benign_labels)
        data = {
            "embeddings": [e.tolist() for e in all_emb],
            "labels": all_lbl,
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
                e = np.array(emb, dtype=np.float64)
                is_threat = int(lbl) == 1
                self.add_sample(e, is_threat)
                count += 1
            logger.debug("Loaded %d samples from %s", count, p)
            return count
        except Exception as e:
            logger.warning("Failed to load training buffer from %s: %s", p, e)
            return 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "threat_samples": len(self._threat_labels),
            "benign_samples": len(self._benign_labels),
        }
