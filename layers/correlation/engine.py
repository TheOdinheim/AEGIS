"""
Campaign Correlation Engine — Main orchestrator for Extension 6.

Subscribes to the event bus, feeds events to the fingerprint detector
and campaign graph simultaneously. When any fingerprint fires with
confidence above threshold, constructs a CampaignAlert and publishes
it through the existing AEGIS event bus.

ASSUMED-BREACH POSTURE: The engine is advisory and additive — it does
not modify request flow. If the engine fails, AEGIS continues to operate
with its existing per-request detection. Engine errors are logged but
never crash the main proxy.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aegis.layers.correlation.events import (
    AgentActionEvent,
    AlertLevel,
    CampaignAlert,
    FingerprintType,
)
from aegis.layers.correlation.fingerprint_detector import FingerprintDetector
from aegis.layers.correlation.campaign_graph import CampaignGraph

logger = logging.getLogger(__name__)


class CampaignCorrelationEngine:
    """Orchestrates campaign detection from distributed agent action events.

    Lifecycle:
    - Instantiate with configuration
    - Call start() to begin processing
    - Feed events via ingest_event() or event bus subscription
    - Call stop() to shut down

    Integration:
    - Publishes CampaignAlerts to existing AEGIS event bus ("threat_detected")
    - Integrates with existing TLI escalation and audit logging
    - Does NOT modify existing layer implementations
    """

    def __init__(
        self,
        *,
        window_sizes: list[int] | None = None,
        temporal_cluster_cv_threshold: float = 0.5,
        temporal_cluster_min_agents: int = 10,
        enumeration_coverage_threshold: float = 0.6,
        fuzzing_entropy_std_threshold: float = 2.0,
        recon_exploit_transition_threshold: float = 0.7,
        info_flow_min_links: int = 3,
        campaign_alert_threshold: float = 0.7,
        campaign_escalation_threshold: float = 0.85,
        graph_retention_seconds: int = 300,
        event_bus: Any | None = None,
    ) -> None:
        self._alert_threshold = campaign_alert_threshold
        self._escalation_threshold = campaign_escalation_threshold
        self._event_bus = event_bus
        self._running = False

        self._detector = FingerprintDetector(
            window_sizes=window_sizes,
            temporal_cluster_cv_threshold=temporal_cluster_cv_threshold,
            temporal_cluster_min_agents=temporal_cluster_min_agents,
            enumeration_coverage_threshold=enumeration_coverage_threshold,
            fuzzing_entropy_std_threshold=fuzzing_entropy_std_threshold,
            recon_exploit_transition_threshold=recon_exploit_transition_threshold,
            info_flow_min_links=info_flow_min_links,
        )

        self._graph = CampaignGraph(
            retention_seconds=graph_retention_seconds,
        )

        # Track active campaigns
        self._active_campaigns: dict[str, CampaignAlert] = {}
        self._alert_history: list[CampaignAlert] = []
        self._max_history = 200

        # Stats
        self._events_processed = 0
        self._alerts_generated = 0

    async def start(self) -> None:
        """Start the correlation engine."""
        self._running = True
        logger.info("Campaign Correlation Engine started")

    async def stop(self) -> None:
        """Stop the correlation engine."""
        self._running = False
        logger.info("Campaign Correlation Engine stopped")

    async def ingest_event(self, event: AgentActionEvent) -> list[CampaignAlert]:
        """Ingest an event and return any generated alerts.

        Thread-safe for concurrent callers. Never raises — errors
        are logged and empty list returned.
        """
        try:
            return await self._process_event(event)
        except Exception as e:
            logger.error("Campaign correlation error: %s", e)
            return []

    async def _process_event(self, event: AgentActionEvent) -> list[CampaignAlert]:
        """Internal event processing."""
        self._events_processed += 1

        # Feed to campaign graph
        self._graph.add_event(event)

        # Feed to fingerprint detector
        matches = self._detector.add_event(event)

        # Generate alerts for matches above threshold
        alerts: list[CampaignAlert] = []
        for match in matches:
            if match.confidence < self._alert_threshold:
                continue

            # Determine alert level
            if match.confidence >= self._escalation_threshold:
                alert_level = AlertLevel.HIGH
            elif match.confidence >= 0.8:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW

            # Get contributing agent IDs
            agent_ids = set()
            for eid in match.contributing_events:
                node = self._graph.get_node(eid)
                if node:
                    agent_ids.add(node.agent_id)
            # Also find from detector events
            for e in self._detector._events:
                if e.event_id in match.contributing_events:
                    agent_ids.add(e.agent_id)

            # Determine recommended action
            if match.confidence >= self._escalation_threshold:
                recommended_action = "escalate_tli"
            elif match.signature_type == FingerprintType.INFORMATION_FLOW:
                recommended_action = "quarantine_agents"
            else:
                recommended_action = "investigate"

            alert = CampaignAlert(
                confidence=match.confidence,
                alert_level=alert_level,
                contributing_agent_ids=sorted(agent_ids),
                fingerprint_type=match.signature_type,
                campaign_graph_snapshot=self._graph.to_serializable(),
                recommended_action=recommended_action,
                evidence=match.evidence,
                tenant_id=event.tenant_id,
            )

            alerts.append(alert)
            self._alerts_generated += 1
            self._alert_history.append(alert)
            if len(self._alert_history) > self._max_history:
                self._alert_history = self._alert_history[-self._max_history:]

            logger.warning(
                "Campaign alert: type=%s, confidence=%.2f, agents=%d, action=%s",
                match.signature_type.value, match.confidence,
                len(agent_ids), recommended_action,
            )

            # Publish to existing AEGIS event bus
            if self._event_bus:
                try:
                    await self._event_bus.publish("threat_detected", {
                        "type": "campaign_alert",
                        "campaign_id": alert.campaign_id,
                        "fingerprint_type": alert.fingerprint_type.value,
                        "confidence": alert.confidence,
                        "alert_level": alert.alert_level.value,
                        "contributing_agents": alert.contributing_agent_ids,
                        "recommended_action": alert.recommended_action,
                    })
                except Exception as e:
                    logger.debug("Failed to publish campaign alert: %s", e)

        return alerts

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "running": self._running,
            "events_processed": self._events_processed,
            "alerts_generated": self._alerts_generated,
            "graph_nodes": self._graph.node_count(),
            "active_campaigns": len(self._graph.get_campaign_subgraphs()),
            "alert_history_count": len(self._alert_history),
        }

    def get_recent_alerts(self, limit: int = 20) -> list[CampaignAlert]:
        """Get recent campaign alerts."""
        return list(reversed(self._alert_history[-limit:]))

    @property
    def detector(self) -> FingerprintDetector:
        """Access the fingerprint detector."""
        return self._detector

    @property
    def graph(self) -> CampaignGraph:
        """Access the campaign graph."""
        return self._graph
