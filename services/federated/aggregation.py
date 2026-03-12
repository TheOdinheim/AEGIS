"""
Federated Aggregator — FedAvg with Poisoning Defense.

Biological Analog: The thymus screens developing T-cells, destroying those
with dangerously strong self-reactivity (negative selection). Similarly,
the aggregator screens model updates, clipping those with abnormally large
weight norms that could indicate poisoning attacks.

Implements Federated Averaging (FedAvg) with Byzantine-robust aggregation:
- Weighted average of model updates proportional to num_samples
- L2 norm clipping: updates with norm > 10× median are clipped
- Minimum participant threshold for aggregation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from aegis.services.federated.local_trainer import LocalModelUpdate

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of a federated aggregation round."""
    round_number: int
    global_weights: dict[str, np.ndarray]
    num_participants: int
    num_clipped: int
    total_samples: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "num_participants": self.num_participants,
            "num_clipped": self.num_clipped,
            "total_samples": self.total_samples,
            "timestamp": self.timestamp.isoformat(),
            "weight_shapes": {k: list(v.shape) for k, v in self.global_weights.items()},
        }


class FederatedAggregator:
    """Federated Averaging with weight poisoning defense.

    Aggregates model updates from multiple AEGIS instances using weighted
    averaging (FedAvg). Defends against poisoning by clipping updates
    whose L2 norm exceeds 10× the median norm.

    Parameters:
        min_participants: Minimum updates required for aggregation.
        norm_clip_multiplier: Updates with L2 norm > multiplier × median are clipped.
    """

    def __init__(
        self,
        *,
        min_participants: int = 1,
        norm_clip_multiplier: float = 10.0,
    ) -> None:
        if min_participants < 1:
            raise ValueError("min_participants must be >= 1")
        if norm_clip_multiplier <= 0:
            raise ValueError("norm_clip_multiplier must be positive")

        self._min_participants = min_participants
        self._norm_clip_multiplier = norm_clip_multiplier
        self._round_number: int = 0
        self._history: list[AggregationResult] = []

    @property
    def round_number(self) -> int:
        return self._round_number

    @property
    def min_participants(self) -> int:
        return self._min_participants

    def _compute_update_norm(self, update: LocalModelUpdate) -> float:
        """Compute L2 norm across all weight arrays in an update."""
        total_sq = 0.0
        for w in update.weights.values():
            total_sq += float(np.sum(w ** 2))
        return float(np.sqrt(total_sq))

    def _clip_update(
        self, update: LocalModelUpdate, max_norm: float,
    ) -> LocalModelUpdate:
        """Clip update weights to max_norm if they exceed it.

        Returns a new LocalModelUpdate with clipped weights.
        """
        current_norm = self._compute_update_norm(update)
        if current_norm <= max_norm:
            return update

        scale = max_norm / current_norm
        clipped_weights = {
            name: w * scale for name, w in update.weights.items()
        }

        return LocalModelUpdate(
            instance_id=update.instance_id,
            round_number=update.round_number,
            weights=clipped_weights,
            num_samples=update.num_samples,
            metrics=update.metrics,
            timestamp=update.timestamp,
        )

    def aggregate(self, updates: list[LocalModelUpdate]) -> AggregationResult:
        """Aggregate model updates using FedAvg with poisoning defense.

        Args:
            updates: List of LocalModelUpdate from participating instances.

        Returns:
            AggregationResult with global weights.

        Raises:
            ValueError: If fewer than min_participants updates provided.
        """
        if len(updates) < self._min_participants:
            raise ValueError(
                f"Need at least {self._min_participants} participants, got {len(updates)}"
            )

        # Step 1: Compute L2 norms
        norms = [self._compute_update_norm(u) for u in updates]

        # Step 2: Poisoning defense — clip outliers
        num_clipped = 0
        if len(norms) >= 2:
            median_norm = float(np.median(norms))
            max_norm = median_norm * self._norm_clip_multiplier

            processed = []
            for i, update in enumerate(updates):
                if norms[i] > max_norm:
                    processed.append(self._clip_update(update, max_norm))
                    num_clipped += 1
                    logger.warning(
                        "Clipped update from %s: norm %.4f > max %.4f (%.1fx median)",
                        update.instance_id, norms[i], max_norm,
                        norms[i] / median_norm if median_norm > 0 else float('inf'),
                    )
                else:
                    processed.append(update)
        else:
            processed = list(updates)

        # Step 3: FedAvg — weighted average by num_samples
        total_samples = sum(u.num_samples for u in processed)
        if total_samples == 0:
            total_samples = len(processed)  # Equal weight fallback

        # Collect all weight keys
        all_keys: set[str] = set()
        for u in processed:
            all_keys.update(u.weights.keys())

        global_weights: dict[str, np.ndarray] = {}
        for key in all_keys:
            # Find reference shape from first update that has this key
            ref_shape = None
            for u in processed:
                if key in u.weights:
                    ref_shape = u.weights[key].shape
                    break
            if ref_shape is None:
                continue

            weighted_sum = np.zeros(ref_shape, dtype=np.float64)
            weight_total = 0

            for u in processed:
                if key not in u.weights:
                    continue
                w = u.num_samples if total_samples > 0 else 1
                weighted_sum += u.weights[key].astype(np.float64) * w
                weight_total += w

            if weight_total > 0:
                global_weights[key] = weighted_sum / weight_total

        self._round_number += 1

        result = AggregationResult(
            round_number=self._round_number,
            global_weights=global_weights,
            num_participants=len(updates),
            num_clipped=num_clipped,
            total_samples=total_samples,
        )
        self._history.append(result)

        logger.info(
            "FedAvg round %d: %d participants, %d clipped, %d total samples",
            self._round_number, len(updates), num_clipped, total_samples,
        )

        return result

    @property
    def stats(self) -> dict[str, Any]:
        """Aggregator statistics."""
        return {
            "round_number": self._round_number,
            "min_participants": self._min_participants,
            "norm_clip_multiplier": self._norm_clip_multiplier,
            "total_rounds": len(self._history),
            "last_round": self._history[-1].to_dict() if self._history else None,
        }
