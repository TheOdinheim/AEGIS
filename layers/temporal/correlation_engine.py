"""
Temporal Correlation Engine (TCE) — Correlates behavioral anomalies
against the provenance timeline to identify suspected poisoning events.

When the BBE detects drift or the canary system detects divergence, the
TCE answers: "The system started behaving differently at time T. What
state change happened shortly before T that could explain the shift?"

The scoring model assigns correlation scores based on:
- Temporal proximity: events closer to the anomaly score higher
- Trust level: untrusted sources get 2x weighting
- Category relevance: RAG/memory events score higher (primary poisoning vectors)
- Anomaly type match: semantic drift → RAG events; refusal rate → memory events

ASSUMED-BREACH POSTURE: The TCE is advisory — it provides ranked
candidates and recommended actions but does NOT take automated action.
L7 Self-Healing acts on TCE recommendations based on policy. A
compromised TCE that suppresses candidates could delay forensic
attribution — the TCE's reports are persisted and auditable.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.temporal.provenance_registry import (
    MemoryProvenanceRegistry,
    ProvenanceCategory,
    ProvenanceRecord,
)

logger = logging.getLogger(__name__)

# Anomaly types that map to specific category relevance boosts
_SEMANTIC_ANOMALIES = {"semantic_drift", "topic_distribution", "canary_failure"}
_REFUSAL_ANOMALIES = {"refusal_rate", "response_consistency"}
_LENGTH_ANOMALIES = {"output_length"}


@dataclass
class CorrelationCandidate:
    """A single candidate state change that may explain a behavioral anomaly."""
    provenance_record: ProvenanceRecord
    correlation_score: float  # 0.0-1.0
    temporal_proximity: float
    trust_factor: float
    category_factor: float
    anomaly_match_factor: float
    explanation: str


@dataclass
class CorrelationReport:
    """Result of correlating a behavioral anomaly against provenance records."""
    report_id: str
    anomaly_timestamp: float
    anomaly_type: str
    anomaly_description: str
    correlation_window_hours: float
    candidates_examined: int
    top_candidates: list[CorrelationCandidate]
    confidence: str  # "high", "medium", "low", "insufficient_data"
    recommended_action: str  # "investigate", "quarantine", "rollback", "monitor"
    analysis_latency_ms: float
    tenant_id: str = "default"


class TemporalCorrelationEngine:
    """Correlates behavioral anomalies from BBE/canaries against MPR timeline.

    Integration points:
    - MPR: provides the provenance timeline of state changes
    - BBE: provides drift alerts with anomaly timestamps
    - Canary system: provides failure alerts
    - Event bus: subscribes to "temporal_drift" for auto-correlation;
      publishes correlation reports
    - L7: receives recommendations (quarantine, rollback, investigate)
    """

    def __init__(
        self,
        *,
        default_window_hours: float = 72.0,
        max_candidates: int = 5,
        auto_correlate_on_critical: bool = True,
        event_bus: Any | None = None,
    ) -> None:
        self._default_window_hours = default_window_hours
        self._max_candidates = max_candidates
        self._auto_correlate = auto_correlate_on_critical
        self._event_bus = event_bus

        self._mpr: MemoryProvenanceRegistry | None = None
        self._bbe: Any | None = None

        # Report history (bounded)
        self._reports: list[CorrelationReport] = []
        self._max_reports = 200

    def set_mpr(self, mpr: MemoryProvenanceRegistry) -> None:
        """Set the MPR reference."""
        self._mpr = mpr

    def set_bbe(self, bbe: Any) -> None:
        """Set the BBE reference for accessing drift history."""
        self._bbe = bbe

    async def correlate(
        self,
        anomaly_timestamp: float,
        anomaly_type: str,
        anomaly_description: str,
        tenant_id: str = "default",
        window_hours: float | None = None,
    ) -> CorrelationReport:
        """Correlate a behavioral anomaly against the provenance timeline."""
        start = time.perf_counter()
        window = window_hours or self._default_window_hours

        if self._mpr is None:
            return self._insufficient_data_report(
                anomaly_timestamp, anomaly_type, anomaly_description,
                window, tenant_id, start,
            )

        # Check if MPR has enough data
        stats = self._mpr.get_stats()
        if stats["total_records"] < 10:
            return self._insufficient_data_report(
                anomaly_timestamp, anomaly_type, anomaly_description,
                window, tenant_id, start,
            )

        # Get candidate state changes in the lookback window
        lookback_start = anomaly_timestamp - (window * 3600.0)
        timeline = self._mpr.get_timeline(
            tenant_id=tenant_id,
            since=lookback_start,
            until=anomaly_timestamp,
        )

        if not timeline.records:
            return self._insufficient_data_report(
                anomaly_timestamp, anomaly_type, anomaly_description,
                window, tenant_id, start,
            )

        # Score each candidate
        candidates: list[CorrelationCandidate] = []
        for record in timeline.records:
            candidate = self._score_candidate(
                record, anomaly_timestamp, anomaly_type, window,
            )
            candidates.append(candidate)

        # Rank by correlation score
        candidates.sort(key=lambda c: c.correlation_score, reverse=True)
        top = candidates[:self._max_candidates]

        # Determine confidence and recommended action
        confidence = self._determine_confidence(top, anomaly_type)
        action = self._determine_action(confidence, top)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        report = CorrelationReport(
            report_id=uuid.uuid4().hex[:12],
            anomaly_timestamp=anomaly_timestamp,
            anomaly_type=anomaly_type,
            anomaly_description=anomaly_description,
            correlation_window_hours=window,
            candidates_examined=len(timeline.records),
            top_candidates=top,
            confidence=confidence,
            recommended_action=action,
            analysis_latency_ms=elapsed_ms,
            tenant_id=tenant_id,
        )

        # Store report
        self._reports.append(report)
        if len(self._reports) > self._max_reports:
            self._reports = self._reports[-self._max_reports:]

        # Publish high-confidence reports to event bus
        if self._event_bus and confidence in ("high", "medium"):
            try:
                await self._event_bus.publish("temporal_drift", {
                    "type": "correlation_report",
                    "report_id": report.report_id,
                    "confidence": confidence,
                    "recommended_action": action,
                    "anomaly_type": anomaly_type,
                    "top_candidate_entity": top[0].provenance_record.entity_id if top else None,
                    "top_candidate_score": round(top[0].correlation_score, 3) if top else 0,
                })
            except Exception as e:
                logger.debug("Failed to publish correlation report: %s", e)

        logger.info(
            "TCE: correlation complete — confidence=%s, action=%s, candidates=%d, latency=%.1fms",
            confidence, action, len(timeline.records), elapsed_ms,
        )
        return report

    def _score_candidate(
        self,
        record: ProvenanceRecord,
        anomaly_timestamp: float,
        anomaly_type: str,
        window_hours: float,
    ) -> CorrelationCandidate:
        """Score a single provenance record as a candidate cause."""
        window_seconds = window_hours * 3600.0

        # 1. Temporal proximity: closer events score higher
        time_gap = anomaly_timestamp - record.timestamp
        if time_gap < 0:
            time_gap = 0.0
        temporal_proximity = max(0.0, 1.0 - (time_gap / window_seconds))

        # 2. Trust level weighting
        trust_factor = {
            "untrusted": 2.0,
            "unknown": 1.0,
            "trusted": 0.5,
        }.get(record.trust_level, 1.0)

        # 3. Category relevance
        category_factor = {
            ProvenanceCategory.RAG_DOCUMENT: 1.5,
            ProvenanceCategory.MEMORY_ENTRY: 1.5,
            ProvenanceCategory.CONFIG_CHANGE: 1.0,
            ProvenanceCategory.MODEL_STATE: 0.8,
        }.get(record.category, 1.0)

        # 4. Anomaly type match
        anomaly_match_factor = 1.0
        if anomaly_type in _SEMANTIC_ANOMALIES:
            if record.category == ProvenanceCategory.RAG_DOCUMENT:
                anomaly_match_factor = 1.5
        elif anomaly_type in _REFUSAL_ANOMALIES:
            if record.category == ProvenanceCategory.MEMORY_ENTRY:
                anomaly_match_factor = 1.5
        elif anomaly_type in _LENGTH_ANOMALIES:
            if record.category in (ProvenanceCategory.CONFIG_CHANGE, ProvenanceCategory.MEMORY_ENTRY):
                anomaly_match_factor = 1.2

        # Combined score (normalized to 0-1)
        raw_score = temporal_proximity * trust_factor * category_factor * anomaly_match_factor
        # Normalize: max possible = 1.0 * 2.0 * 1.5 * 1.5 = 4.5
        correlation_score = min(1.0, raw_score / 4.5)

        # Build explanation
        parts = [
            f"{record.category.value} '{record.entity_id}'",
            f"from {record.source} ({record.trust_level})",
            f"occurred {time_gap/3600:.1f}h before anomaly",
        ]
        if record.flagged:
            parts.append(f"FLAGGED: {record.flag_reason}")

        return CorrelationCandidate(
            provenance_record=record,
            correlation_score=correlation_score,
            temporal_proximity=temporal_proximity,
            trust_factor=trust_factor,
            category_factor=category_factor,
            anomaly_match_factor=anomaly_match_factor,
            explanation="; ".join(parts),
        )

    def _determine_confidence(
        self,
        top_candidates: list[CorrelationCandidate],
        anomaly_type: str,
    ) -> str:
        """Determine confidence level based on top candidates."""
        if not top_candidates:
            return "insufficient_data"

        top = top_candidates[0]

        # HIGH: score > 0.8 AND untrusted AND anomaly type matches category
        if (
            top.correlation_score > 0.8
            and top.provenance_record.trust_level == "untrusted"
            and top.anomaly_match_factor > 1.0
        ):
            return "high"

        # MEDIUM: score > 0.5 OR multiple candidates cluster
        if top.correlation_score > 0.5:
            return "medium"

        # Check for clustering: multiple candidates with similar timestamps
        if len(top_candidates) >= 3:
            time_spread = (
                top_candidates[0].temporal_proximity
                - top_candidates[2].temporal_proximity
            )
            if abs(time_spread) < 0.1:
                return "medium"

        return "low"

    def _determine_action(
        self,
        confidence: str,
        top_candidates: list[CorrelationCandidate],
    ) -> str:
        """Determine recommended action based on confidence and candidate type."""
        if confidence == "insufficient_data":
            return "monitor"

        if confidence == "high" and top_candidates:
            cat = top_candidates[0].provenance_record.category
            if cat in (ProvenanceCategory.RAG_DOCUMENT, ProvenanceCategory.MEMORY_ENTRY):
                return "quarantine"
            if cat == ProvenanceCategory.CONFIG_CHANGE:
                return "rollback"
            if cat == ProvenanceCategory.MODEL_STATE:
                return "rollback"
            return "quarantine"

        if confidence == "medium":
            return "investigate"

        return "monitor"

    def _insufficient_data_report(
        self,
        anomaly_timestamp: float,
        anomaly_type: str,
        anomaly_description: str,
        window_hours: float,
        tenant_id: str,
        start: float,
    ) -> CorrelationReport:
        """Generate an insufficient_data report."""
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        report = CorrelationReport(
            report_id=uuid.uuid4().hex[:12],
            anomaly_timestamp=anomaly_timestamp,
            anomaly_type=anomaly_type,
            anomaly_description=anomaly_description,
            correlation_window_hours=window_hours,
            candidates_examined=0,
            top_candidates=[],
            confidence="insufficient_data",
            recommended_action="monitor",
            analysis_latency_ms=elapsed_ms,
            tenant_id=tenant_id,
        )
        self._reports.append(report)
        if len(self._reports) > self._max_reports:
            self._reports = self._reports[-self._max_reports:]
        return report

    def get_recent_reports(
        self,
        tenant_id: str = "default",
        limit: int = 10,
    ) -> list[CorrelationReport]:
        """Get recent correlation reports, optionally filtered by tenant."""
        reports = [r for r in self._reports if r.tenant_id == tenant_id]
        return reports[-limit:]

    async def handle_drift_event(self, event_data: dict) -> None:
        """Handle a temporal_drift event for auto-correlation.

        Called when subscribed to the event bus. Only triggers correlation
        for critical-severity events (drift alerts or canary failures).
        """
        if not self._auto_correlate:
            return

        event_type = event_data.get("type", "")

        if event_type == "drift_alert":
            confidence = event_data.get("confidence", "")
            if confidence != "critical":
                return
            await self.correlate(
                anomaly_timestamp=time.time(),
                anomaly_type=event_data.get("metric_name", "unknown"),
                anomaly_description=f"BBE drift alert: {event_data.get('metric_name', 'unknown')} "
                                    f"deviated {event_data.get('deviation_sigmas', 0):.1f} sigmas",
                tenant_id=event_data.get("tenant_id", "default"),
            )

        elif event_type == "canary_alert":
            level = event_data.get("level", "")
            if level not in ("high", "critical"):
                return
            await self.correlate(
                anomaly_timestamp=time.time(),
                anomaly_type="canary_failure",
                anomaly_description=f"Canary '{event_data.get('canary_id', 'unknown')}' "
                                    f"failed ({event_data.get('consecutive_failures', 0)} consecutive)",
                tenant_id="default",
            )
