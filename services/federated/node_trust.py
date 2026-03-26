"""
Node Trust Scoring — Earn/Decay Trust Model.

Every federation node starts with a low trust score (0.3) and must earn
higher trust through consistent, accurate contributions. Trust influences
what weight a node's updates receive in aggregation and whether its
indicators are accepted.

Trust factors:
  - Positive: valid indicator submissions, consistent FL updates, heartbeats
  - Negative: anomalous updates flagged by Byzantine detection, invalid
    indicators, quarantined submissions, burst flooding
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrustEvent:
    """A recorded trust-affecting event."""
    event_type: str  # positive or negative
    action: str      # e.g. "valid_indicator", "byzantine_flagged"
    delta: float     # trust change
    timestamp: float
    details: str = ""


class NodeTrustScorer:
    """Earn/decay trust model for federation nodes.

    Parameters:
        initial_trust: Starting trust score for new nodes (default 0.3).
        max_trust: Ceiling (default 1.0).
        min_trust: Floor (default 0.0).
        decay_rate: Trust decays by this per hour of inactivity (default 0.001).
        decay_interval_seconds: How often to check for decay (default 3600).
        valid_indicator_reward: Trust boost for accepted indicator (default 0.02).
        valid_fl_update_reward: Trust boost for clean FL update (default 0.01).
        heartbeat_reward: Trust boost for heartbeat (default 0.001).
        byzantine_penalty: Trust penalty for flagged update (default 0.10).
        quarantine_penalty: Trust penalty for quarantined indicator (default 0.05).
        invalid_indicator_penalty: Trust penalty for rejected indicator (default 0.03).
        min_trust_for_fl: Minimum trust to participate in FL (default 0.2).
        min_trust_for_indicators: Minimum trust to submit indicators (default 0.15).
    """

    def __init__(
        self,
        *,
        initial_trust: float = 0.3,
        max_trust: float = 1.0,
        min_trust: float = 0.0,
        decay_rate: float = 0.001,
        decay_interval_seconds: float = 3600.0,
        valid_indicator_reward: float = 0.02,
        valid_fl_update_reward: float = 0.01,
        heartbeat_reward: float = 0.001,
        byzantine_penalty: float = 0.10,
        quarantine_penalty: float = 0.05,
        invalid_indicator_penalty: float = 0.03,
        min_trust_for_fl: float = 0.2,
        min_trust_for_indicators: float = 0.15,
    ) -> None:
        self._initial_trust = initial_trust
        self._max_trust = max_trust
        self._min_trust = min_trust
        self._decay_rate = decay_rate
        self._decay_interval = decay_interval_seconds
        self._rewards = {
            "valid_indicator": valid_indicator_reward,
            "valid_fl_update": valid_fl_update_reward,
            "heartbeat": heartbeat_reward,
        }
        self._penalties = {
            "byzantine_flagged": byzantine_penalty,
            "quarantined_indicator": quarantine_penalty,
            "invalid_indicator": invalid_indicator_penalty,
        }
        self._min_trust_fl = min_trust_for_fl
        self._min_trust_indicators = min_trust_for_indicators

        self._scores: dict[str, float] = {}
        self._last_activity: dict[str, float] = {}
        self._event_log: dict[str, list[TrustEvent]] = {}
        self._max_events_per_node = 100

    def get_trust(self, node_id: str) -> float:
        """Get current trust score for a node. Creates with initial if new."""
        if node_id not in self._scores:
            self._scores[node_id] = self._initial_trust
            self._last_activity[node_id] = time.time()
        return self._scores[node_id]

    def record_positive(self, node_id: str, action: str, details: str = "") -> float:
        """Record a positive trust event. Returns new score."""
        delta = self._rewards.get(action, 0.01)
        return self._apply_delta(node_id, delta, "positive", action, details)

    def record_negative(self, node_id: str, action: str, details: str = "") -> float:
        """Record a negative trust event. Returns new score."""
        delta = -self._penalties.get(action, 0.03)
        return self._apply_delta(node_id, delta, "negative", action, details)

    def _apply_delta(
        self, node_id: str, delta: float,
        event_type: str, action: str, details: str,
    ) -> float:
        """Apply trust delta and record event."""
        old = self.get_trust(node_id)
        new = max(self._min_trust, min(self._max_trust, old + delta))
        self._scores[node_id] = new
        self._last_activity[node_id] = time.time()

        event = TrustEvent(
            event_type=event_type,
            action=action,
            delta=delta,
            timestamp=time.time(),
            details=details,
        )
        if node_id not in self._event_log:
            self._event_log[node_id] = []
        self._event_log[node_id].append(event)
        if len(self._event_log[node_id]) > self._max_events_per_node:
            self._event_log[node_id] = self._event_log[node_id][-self._max_events_per_node:]

        if event_type == "negative":
            logger.warning(
                "Node %s trust: %.3f → %.3f (%s: %s)",
                node_id, old, new, action, details,
            )

        return new

    def decay_inactive(self) -> dict[str, float]:
        """Apply decay to all inactive nodes. Returns {node_id: new_score}."""
        now = time.time()
        changed: dict[str, float] = {}

        for node_id, last in list(self._last_activity.items()):
            elapsed = now - last
            if elapsed < self._decay_interval:
                continue

            hours = elapsed / 3600
            decay = self._decay_rate * hours
            old = self._scores.get(node_id, self._initial_trust)
            new = max(self._min_trust, old - decay)
            if new != old:
                self._scores[node_id] = new
                changed[node_id] = new

        return changed

    def can_participate_fl(self, node_id: str) -> bool:
        """Check if a node has enough trust for federated learning."""
        return self.get_trust(node_id) >= self._min_trust_fl

    def can_submit_indicators(self, node_id: str) -> bool:
        """Check if a node has enough trust to submit indicators."""
        return self.get_trust(node_id) >= self._min_trust_indicators

    def get_node_profile(self, node_id: str) -> dict[str, Any]:
        """Get detailed trust profile for a node."""
        trust = self.get_trust(node_id)
        events = self._event_log.get(node_id, [])
        positive = sum(1 for e in events if e.event_type == "positive")
        negative = sum(1 for e in events if e.event_type == "negative")

        return {
            "node_id": node_id,
            "trust_score": round(trust, 4),
            "can_fl": self.can_participate_fl(node_id),
            "can_submit_indicators": self.can_submit_indicators(node_id),
            "positive_events": positive,
            "negative_events": negative,
            "total_events": len(events),
            "last_activity": self._last_activity.get(node_id),
        }

    def get_all_scores(self) -> dict[str, float]:
        """Return all node trust scores."""
        return dict(self._scores)

    @property
    def stats(self) -> dict[str, Any]:
        scores = list(self._scores.values())
        return {
            "total_nodes": len(self._scores),
            "avg_trust": round(sum(scores) / max(1, len(scores)), 4),
            "min_trust": round(min(scores), 4) if scores else 0.0,
            "max_trust": round(max(scores), 4) if scores else 0.0,
            "nodes_eligible_fl": sum(1 for s in scores if s >= self._min_trust_fl),
            "nodes_eligible_indicators": sum(1 for s in scores if s >= self._min_trust_indicators),
        }
