"""
Memory Provenance Registry (MPR) — Tracks the provenance of every piece
of persistent state in the protected system.

When the system gets sick (BBE detects drift or canary fails), the MPR
provides the epidemiological record: what state changes happened, when,
and from what source. The Temporal Correlation Engine (TCE) uses this
timeline to identify suspected poisoning events.

Research grounding: MINJA (NeurIPS 2025) showed >95% memory injection
success rates. Microsoft identified 31 companies exploiting memory
poisoning (February 2026). RAGForensics (ACM Web Conference 2025)
demonstrated post-attack forensic attribution via iterative retrieval.

ASSUMED-BREACH POSTURE: The MPR is append-only. A compromised MPR that
drops records would blind forensic attribution. Records are indexed by
content_hash for retroactive matching when new threats are discovered.
An attacker who can delete MPR records can cover their tracks — the
registry must be treated as a critical audit surface.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ProvenanceCategory(enum.Enum):
    """Categories of tracked state changes."""
    RAG_DOCUMENT = "rag_document"
    MEMORY_ENTRY = "memory_entry"
    CONFIG_CHANGE = "config_change"
    MODEL_STATE = "model_state"


@dataclass
class ProvenanceRecord:
    """A single provenance record tracking a state change."""
    record_id: str
    category: ProvenanceCategory
    event_type: str  # "created", "modified", "removed", "quarantined"
    timestamp: float
    entity_id: str
    content_hash: str
    source: str
    trust_level: str  # "trusted", "untrusted", "unknown"
    tenant_id: str = "default"
    metadata: dict = field(default_factory=dict)
    flagged: bool = False
    flag_reason: str | None = None


@dataclass
class ProvenanceTimeline:
    """A timeline view of provenance records for a tenant."""
    tenant_id: str
    records: list[ProvenanceRecord]
    total_records: int
    earliest_timestamp: float | None
    latest_timestamp: float | None


class MemoryProvenanceRegistry:
    """Tracks provenance metadata for persistent state changes.

    In-memory storage with FIFO eviction when max records is exceeded.
    Indexed by entity_id, content_hash, and tenant_id for efficient queries.
    Thread-safe via threading.Lock.
    """

    def __init__(self, *, max_records: int = 100_000, event_bus: Any | None = None) -> None:
        self._max_records = max_records
        self._event_bus = event_bus
        self._lock = threading.Lock()

        # Primary storage: ordered list of all records
        self._records: list[ProvenanceRecord] = []
        # Indexes
        self._by_entity: dict[str, list[ProvenanceRecord]] = defaultdict(list)
        self._by_hash: dict[str, list[ProvenanceRecord]] = defaultdict(list)
        self._by_tenant: dict[str, list[ProvenanceRecord]] = defaultdict(list)

    def record_event(
        self,
        category: str | ProvenanceCategory,
        event_type: str,
        entity_id: str,
        content_hash: str,
        source: str,
        trust_level: str = "unknown",
        tenant_id: str = "default",
        metadata: dict | None = None,
    ) -> ProvenanceRecord:
        """Record a state change event. Returns the created record."""
        if isinstance(category, str):
            category = ProvenanceCategory(category)

        record = ProvenanceRecord(
            record_id=uuid.uuid4().hex[:12],
            category=category,
            event_type=event_type,
            timestamp=time.time(),
            entity_id=entity_id,
            content_hash=content_hash,
            source=source,
            trust_level=trust_level,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            self._by_entity[entity_id].append(record)
            self._by_hash[content_hash].append(record)
            self._by_tenant[tenant_id].append(record)

            # FIFO eviction
            while len(self._records) > self._max_records:
                oldest = self._records.pop(0)
                self._by_entity[oldest.entity_id] = [
                    r for r in self._by_entity[oldest.entity_id] if r.record_id != oldest.record_id
                ]
                self._by_hash[oldest.content_hash] = [
                    r for r in self._by_hash[oldest.content_hash] if r.record_id != oldest.record_id
                ]
                self._by_tenant[oldest.tenant_id] = [
                    r for r in self._by_tenant[oldest.tenant_id] if r.record_id != oldest.record_id
                ]

        logger.debug(
            "MPR: recorded %s %s for entity %s (source=%s, trust=%s)",
            category.value, event_type, entity_id, source, trust_level,
        )
        return record

    def get_timeline(
        self,
        tenant_id: str = "default",
        since: float | None = None,
        until: float | None = None,
        category: str | None = None,
    ) -> ProvenanceTimeline:
        """Get provenance timeline, optionally filtered by time range and category."""
        with self._lock:
            records = list(self._by_tenant.get(tenant_id, []))

        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]
        if category is not None:
            cat = ProvenanceCategory(category) if isinstance(category, str) else category
            records = [r for r in records if r.category == cat]

        records.sort(key=lambda r: r.timestamp)

        return ProvenanceTimeline(
            tenant_id=tenant_id,
            records=records,
            total_records=len(records),
            earliest_timestamp=records[0].timestamp if records else None,
            latest_timestamp=records[-1].timestamp if records else None,
        )

    def get_events_for_entity(self, entity_id: str) -> list[ProvenanceRecord]:
        """Get all provenance records for a specific entity."""
        with self._lock:
            return list(self._by_entity.get(entity_id, []))

    def get_events_in_window(
        self,
        tenant_id: str,
        center_timestamp: float,
        window_hours: float = 24.0,
    ) -> list[ProvenanceRecord]:
        """Get all events within a time window centered on a timestamp."""
        half_window = window_hours * 3600.0 / 2.0
        since = center_timestamp - half_window
        until = center_timestamp + half_window

        with self._lock:
            records = [
                r for r in self._by_tenant.get(tenant_id, [])
                if since <= r.timestamp <= until
            ]
        records.sort(key=lambda r: r.timestamp)
        return records

    def retroactive_flag(
        self,
        threat_signature: str,
        tenant_id: str | None = None,
    ) -> list[ProvenanceRecord]:
        """Scan records and flag any whose content_hash matches the threat signature.

        Called when a new threat is added to L4's immune memory vault.
        """
        flagged: list[ProvenanceRecord] = []
        with self._lock:
            # Match by content_hash
            matching = list(self._by_hash.get(threat_signature, []))
            if tenant_id:
                matching = [r for r in matching if r.tenant_id == tenant_id]

            for record in matching:
                if not record.flagged:
                    record.flagged = True
                    record.flag_reason = f"Retroactive match: content_hash={threat_signature[:16]}..."
                    flagged.append(record)

        if flagged:
            logger.warning(
                "MPR: retroactive flag matched %d record(s) for signature %s",
                len(flagged), threat_signature[:32],
            )

        return flagged

    def quarantine_entity(self, entity_id: str, reason: str) -> bool:
        """Mark an entity as quarantined. Returns True if found and updated."""
        with self._lock:
            records = self._by_entity.get(entity_id, [])
            if not records:
                return False
            for record in records:
                record.metadata["quarantined"] = True
                record.metadata["quarantine_reason"] = reason
                record.metadata["quarantine_timestamp"] = time.time()
                if not record.flagged:
                    record.flagged = True
                    record.flag_reason = f"Quarantined: {reason}"
        return True

    def get_untrusted_events(
        self,
        tenant_id: str = "default",
        hours: float = 168.0,
    ) -> list[ProvenanceRecord]:
        """Get all events from untrusted sources in the given time window."""
        cutoff = time.time() - hours * 3600.0
        with self._lock:
            records = [
                r for r in self._by_tenant.get(tenant_id, [])
                if r.trust_level == "untrusted" and r.timestamp >= cutoff
            ]
        records.sort(key=lambda r: r.timestamp)
        return records

    def get_stats(self) -> dict:
        """Get registry statistics."""
        with self._lock:
            total = len(self._records)
            by_category: dict[str, int] = defaultdict(int)
            by_trust: dict[str, int] = defaultdict(int)
            flagged_count = 0
            for r in self._records:
                by_category[r.category.value] += 1
                by_trust[r.trust_level] += 1
                if r.flagged:
                    flagged_count += 1

        return {
            "total_records": total,
            "by_category": dict(by_category),
            "by_trust_level": dict(by_trust),
            "flagged_count": flagged_count,
            "max_records": self._max_records,
        }
