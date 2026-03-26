"""
Node Registry — Connected Instance Tracking.

Tracks AEGIS instances participating in the federated intelligence network.
In-memory storage with heartbeat-based active/inactive determination.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_ACTIVE_THRESHOLD_SECONDS = 900  # 15 minutes


class NodeRegistry:
    """Track connected federation nodes.

    Parameters:
        active_threshold_seconds: Seconds since last heartbeat to consider active.
        trust_scorer: Optional NodeTrustScorer for trust-aware node info.
    """

    def __init__(
        self,
        *,
        active_threshold_seconds: int = _ACTIVE_THRESHOLD_SECONDS,
        trust_scorer: Any | None = None,
    ) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._active_threshold = active_threshold_seconds
        self._trust_scorer = trust_scorer

    def register_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Register or update a node."""
        now = time.time()
        existing = self._nodes.get(node_id)
        if existing:
            existing["last_heartbeat"] = now
            existing["heartbeat_count"] = existing.get("heartbeat_count", 0) + 1
            if metadata:
                existing["metadata"].update(metadata)
        else:
            self._nodes[node_id] = {
                "node_id": node_id,
                "registered_at": now,
                "last_heartbeat": now,
                "heartbeat_count": 1,
                "metadata": metadata or {},
            }
            logger.info("New federation node registered: %s", node_id)

    def heartbeat(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Update heartbeat for a node. Registers if new."""
        self.register_node(node_id, metadata)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get info for a single node."""
        node = self._nodes.get(node_id)
        if node is None:
            return None
        result = {
            **node,
            "is_active": self._is_active(node),
        }
        if self._trust_scorer:
            result["trust_score"] = self._trust_scorer.get_trust(node_id)
        return result

    def get_active_nodes(self) -> list[dict[str, Any]]:
        """Get nodes that heartbeated within the active threshold."""
        result = []
        for n in self._nodes.values():
            if self._is_active(n):
                entry = {**n, "is_active": True}
                if self._trust_scorer:
                    entry["trust_score"] = self._trust_scorer.get_trust(n["node_id"])
                result.append(entry)
        return result

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Get all registered nodes."""
        result = []
        for n in self._nodes.values():
            entry = {**n, "is_active": self._is_active(n)}
            if self._trust_scorer:
                entry["trust_score"] = self._trust_scorer.get_trust(n["node_id"])
            result.append(entry)
        return result

    def _is_active(self, node: dict[str, Any]) -> bool:
        return (time.time() - node.get("last_heartbeat", 0)) < self._active_threshold

    def get_network_stats(self) -> dict[str, Any]:
        """Network statistics."""
        active = [n for n in self._nodes.values() if self._is_active(n)]
        return {
            "total_nodes": len(self._nodes),
            "active_nodes": len(active),
            "total_heartbeats": sum(n.get("heartbeat_count", 0) for n in self._nodes.values()),
        }
