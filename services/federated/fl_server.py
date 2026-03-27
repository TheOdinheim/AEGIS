"""
FL Server — Federated Learning Aggregation Server.

HTTP-based federated learning server running inside the AEGIS FastAPI app.
Implements FedAvg with optional Byzantine-resilient aggregation and trust-weighted
updates via FederationImmuneResponse.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from aegis.services.federated.fl_model import FederatedClassifier

logger = logging.getLogger(__name__)


class FLServer:
    """Federated learning aggregation server.

    Parameters:
        model: Global model to aggregate into.
        min_clients: Minimum client updates before aggregation.
        immune_response: Optional FederationImmuneResponse for zero-trust aggregation.
    """

    def __init__(
        self,
        model: FederatedClassifier,
        *,
        min_clients: int = 1,
        rounds_completed: int = 0,
        immune_response: Any | None = None,
    ) -> None:
        self._model = model
        self._min_clients = min_clients
        self._rounds_completed = rounds_completed
        self._current_updates: list[dict[str, Any]] = []
        self._round_active = False
        self._last_aggregation: str | None = None
        self._total_samples_trained: int = 0
        self._immune_response = immune_response
        self._max_samples_per_update = 100_000

    def get_global_model(self) -> dict[str, Any]:
        """Return current global model weights as JSON-serializable dict."""
        return {
            "round": self._rounds_completed,
            "weights": self._model.weights_to_json(),
            "updated_at": self._last_aggregation or datetime.now(timezone.utc).isoformat(),
        }

    def submit_update(
        self,
        node_id: str,
        weights: list,
        num_samples: int,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Accept a client's updated weights.

        Returns status dict with round info.
        """
        # RT-P3-001: Cap num_samples to prevent aggregation weight manipulation
        capped_samples = max(0, min(num_samples, self._max_samples_per_update))
        self._current_updates.append({
            "node_id": node_id,
            "weights": [np.array(w, dtype=np.float64) for w in weights],
            "num_samples": capped_samples,
            "metrics": metrics or {},
        })

        logger.info(
            "FL update from %s: %d samples (updates: %d/%d)",
            node_id, num_samples,
            len(self._current_updates), self._min_clients,
        )

        result = {
            "status": "accepted",
            "round": self._rounds_completed,
            "updates_received": len(self._current_updates),
            "updates_needed": self._min_clients,
        }

        # Auto-aggregate if enough updates
        if len(self._current_updates) >= self._min_clients:
            agg = self.aggregate()
            result["aggregated"] = True
            result["round"] = agg["round"]

        return result

    def aggregate(self) -> dict[str, Any]:
        """Run FedAvg aggregation on collected updates.

        When immune_response is set, uses Byzantine-resilient aggregation
        with trust-weighted updates. Otherwise falls back to standard FedAvg.
        """
        if not self._current_updates:
            return {"status": "no_updates", "round": self._rounds_completed}

        updates = self._current_updates
        flagged_count = 0

        if self._immune_response is not None:
            # Zero-trust aggregation path
            global_weights = self._immune_response.aggregate_with_trust(updates)
            if not global_weights:
                self._current_updates = []
                return {
                    "status": "all_excluded",
                    "round": self._rounds_completed,
                    "clients": 0,
                    "total_samples": 0,
                    "global_loss": 0.0,
                }
            flagged_count = self._immune_response.byzantine.stats.get("flagged_updates", 0)
        else:
            # Standard FedAvg path
            total_samples = sum(u["num_samples"] for u in updates)
            if total_samples == 0:
                total_samples = len(updates)

            n_params = len(updates[0]["weights"])
            global_weights = []
            for i in range(n_params):
                weighted_sum = np.zeros_like(updates[0]["weights"][i], dtype=np.float64)
                for u in updates:
                    w = u["num_samples"] if total_samples > 0 else 1
                    weighted_sum += u["weights"][i].astype(np.float64) * w
                global_weights.append(weighted_sum / total_samples)

        total_samples = sum(u["num_samples"] for u in updates)

        # Apply to global model
        self._model.set_weights(global_weights)
        self._rounds_completed += 1
        self._last_aggregation = datetime.now(timezone.utc).isoformat()
        self._total_samples_trained += total_samples

        # Compute average client loss
        avg_loss = 0.0
        loss_count = 0
        for u in updates:
            if "loss" in u.get("metrics", {}):
                avg_loss += u["metrics"]["loss"]
                loss_count += 1
        if loss_count > 0:
            avg_loss /= loss_count

        n_clients = len(updates)
        self._current_updates = []

        logger.info(
            "FedAvg round %d: %d clients, %d total samples, avg_loss=%.4f",
            self._rounds_completed, n_clients, total_samples, avg_loss,
        )

        result = {
            "status": "aggregated",
            "round": self._rounds_completed,
            "clients": n_clients,
            "total_samples": total_samples,
            "global_loss": round(avg_loss, 6),
        }
        if flagged_count:
            result["flagged_updates"] = flagged_count
        return result

    def start_round(self) -> None:
        """Begin a new training round (clear pending updates)."""
        self._current_updates = []
        self._round_active = True

    def get_status(self) -> dict[str, Any]:
        """Current server status."""
        return {
            "round": self._rounds_completed,
            "updates_received": len(self._current_updates),
            "min_clients": self._min_clients,
            "last_aggregation": self._last_aggregation,
            "model_version": f"v{self._rounds_completed}",
            "total_samples_trained": self._total_samples_trained,
        }
