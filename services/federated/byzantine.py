"""
Byzantine-Resilient Aggregation — Robust Federated Averaging.

Replaces naive FedAvg with aggregation methods that tolerate up to f
Byzantine (malicious or faulty) participants out of n total.

Methods:
- Trimmed Mean: sort values per parameter, trim top/bottom 10%, average rest
- Krum: select update most consistent with majority (min sum of distances)
- Multi-Krum: select top k by Krum score, average them
- FedAvg: standard weighted average (no Byzantine resilience)

Anomaly detection flags updates that are statistical outliers by L2 norm
or cosine similarity before aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Report from anomaly detection on FL updates."""
    clean: list[dict[str, Any]] = field(default_factory=list)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)


class ByzantineResilientAggregator:
    """Byzantine-resilient federated aggregation.

    Parameters:
        method: Aggregation method (trimmed_mean, krum, multi_krum, fedavg).
        trim_ratio: Fraction to trim from each end (trimmed_mean only).
        krum_f: Number of Byzantine participants to tolerate (Krum).
        multi_krum_k: Number of updates to select (Multi-Krum).
        norm_threshold_sigma: MAD multiplier for norm outlier detection.
        cosine_threshold: Min cosine similarity to mean update.
    """

    def __init__(
        self,
        *,
        method: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        krum_f: int = 1,
        multi_krum_k: int = 3,
        norm_threshold_sigma: float = 3.0,
        cosine_threshold: float = 0.5,
    ) -> None:
        self._method = method
        self._trim_ratio = trim_ratio
        self._krum_f = krum_f
        self._multi_krum_k = multi_krum_k
        self._norm_threshold_sigma = norm_threshold_sigma
        self._cosine_threshold = cosine_threshold
        self._flagged_count: int = 0
        self._total_aggregations: int = 0

    def detect_anomalous_updates(
        self, updates: list[dict[str, Any]],
    ) -> AnomalyReport:
        """Separate updates into clean and anomalous.

        Flags updates where:
        - L2 norm > median + norm_threshold_sigma * MAD
        - Cosine similarity to mean update < cosine_threshold
        """
        report = AnomalyReport()
        if len(updates) < 3:
            # Need at least 3 to meaningfully detect outliers
            report.clean = list(updates)
            return report

        # Flatten each update's weights into a single vector
        flat_vectors = []
        for u in updates:
            flat = np.concatenate([w.flatten() for w in u["weights"]])
            flat_vectors.append(flat)

        # L2 norms
        norms = np.array([np.linalg.norm(v) for v in flat_vectors])
        median_norm = float(np.median(norms))
        mad = float(np.median(np.abs(norms - median_norm)))
        if mad == 0:
            mad = 1e-8
        norm_threshold = median_norm + self._norm_threshold_sigma * mad

        # Mean vector for cosine similarity
        mean_vec = np.mean(flat_vectors, axis=0)
        mean_norm = np.linalg.norm(mean_vec)

        for i, u in enumerate(updates):
            node_id = u.get("node_id", f"unknown-{i}")
            reasons: list[str] = []

            # Norm check
            if norms[i] > norm_threshold:
                reasons.append(
                    f"L2 norm {norms[i]:.4f} > threshold {norm_threshold:.4f}"
                )

            # Cosine similarity check
            if mean_norm > 0:
                cos_sim = float(
                    np.dot(flat_vectors[i], mean_vec) /
                    (np.linalg.norm(flat_vectors[i]) * mean_norm + 1e-12)
                )
                if cos_sim < self._cosine_threshold:
                    reasons.append(
                        f"cosine similarity {cos_sim:.4f} < threshold {self._cosine_threshold}"
                    )

            if reasons:
                report.flagged.append(u)
                report.reasons[node_id] = reasons
                self._flagged_count += 1
                logger.warning(
                    "Byzantine anomaly from %s: %s", node_id, "; ".join(reasons),
                )
            else:
                report.clean.append(u)

        return report

    def aggregate(
        self, updates: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        """Aggregate weight updates using configured method.

        Each update: {"weights": list[np.ndarray], "num_samples": int, "node_id": str}
        Returns aggregated weights as list of np.ndarray.
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        self._total_aggregations += 1

        if self._method == "trimmed_mean":
            return self._trimmed_mean(updates)
        elif self._method == "krum":
            return self._krum_select(updates)
        elif self._method == "multi_krum":
            return self._multi_krum(updates)
        else:  # fedavg
            return self._fedavg(updates)

    def _trimmed_mean(self, updates: list[dict[str, Any]]) -> list[np.ndarray]:
        """Coordinate-wise trimmed mean."""
        n = len(updates)
        trim_count = max(1, int(n * self._trim_ratio)) if n >= 4 else 0
        n_params = len(updates[0]["weights"])

        result = []
        for i in range(n_params):
            stacked = np.stack([u["weights"][i] for u in updates])
            if trim_count > 0 and n > 2 * trim_count:
                sorted_vals = np.sort(stacked, axis=0)
                trimmed = sorted_vals[trim_count:n - trim_count]
                result.append(trimmed.mean(axis=0))
            else:
                result.append(stacked.mean(axis=0))
        return result

    def _krum_select(
        self, updates: list[dict[str, Any]], f: int | None = None,
    ) -> list[np.ndarray]:
        """Select the update most consistent with the majority."""
        if len(updates) == 1:
            return [w.copy() for w in updates[0]["weights"]]

        f = f if f is not None else self._krum_f
        n = len(updates)
        k = max(1, n - f - 2) if n > f + 2 else 1

        # Flatten weights
        flat = [np.concatenate([w.flatten() for w in u["weights"]]) for u in updates]

        # Pairwise distances
        scores = np.zeros(n)
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dists.append(float(np.linalg.norm(flat[i] - flat[j])) ** 2)
            dists.sort()
            scores[i] = sum(dists[:k])

        best = int(np.argmin(scores))
        return [w.copy() for w in updates[best]["weights"]]

    def _multi_krum(self, updates: list[dict[str, Any]]) -> list[np.ndarray]:
        """Select top k updates by Krum score and average."""
        if len(updates) <= self._multi_krum_k:
            return self._fedavg(updates)

        n = len(updates)
        f = self._krum_f
        k_nearest = max(1, n - f - 2) if n > f + 2 else 1

        flat = [np.concatenate([w.flatten() for w in u["weights"]]) for u in updates]

        scores = np.zeros(n)
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dists.append(float(np.linalg.norm(flat[i] - flat[j])) ** 2)
            dists.sort()
            scores[i] = sum(dists[:k_nearest])

        top_k_idx = np.argsort(scores)[:self._multi_krum_k]
        selected = [updates[i] for i in top_k_idx]
        return self._fedavg(selected)

    def _fedavg(self, updates: list[dict[str, Any]]) -> list[np.ndarray]:
        """Standard FedAvg weighted by num_samples."""
        total_samples = sum(u.get("num_samples", 1) for u in updates)
        if total_samples == 0:
            total_samples = len(updates)

        n_params = len(updates[0]["weights"])
        result = []
        for i in range(n_params):
            weighted_sum = np.zeros_like(updates[0]["weights"][i], dtype=np.float64)
            for u in updates:
                w = u.get("num_samples", 1)
                weighted_sum += u["weights"][i].astype(np.float64) * w
            result.append(weighted_sum / total_samples)
        return result

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "method": self._method,
            "total_aggregations": self._total_aggregations,
            "flagged_updates": self._flagged_count,
            "trim_ratio": self._trim_ratio,
        }
