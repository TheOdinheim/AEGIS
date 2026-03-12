"""
Audit Log Sink — Append-Only Tamper-Evident Security Decision Logger

Every security decision across all AEGIS layers is recorded as a structured
JSON record with SHA-256 chain hashing for tamper evidence. Each record
includes the hash of the previous record, creating an append-only chain
that detects modifications to historical entries.

Bootstrap writes to JSONL file. Production upgrade: PostgreSQL with
row-level encryption and WORM storage.

Compliance Mapping:
    NIST AI RMF GOVERN 1.4 — AI risk management process documentation
    ISO 42001 Clause 9     — Performance evaluation and monitoring
    EU AI Act Articles 12, 19 — Transparency, logging, human oversight
    SOC 2 CC7.2            — System operations monitoring

ASSUMED-BREACH POSTURE: The audit log is a high-value target. An attacker
who compromises the log can cover their tracks. The SHA-256 chain hash
provides tamper evidence — any modification to a historical record breaks
the chain and is detectable. In production, logs are written to WORM
(Write Once Read Many) storage and replicated to a separate security domain.
The log file path is never exposed through the API; only recent records are
served via an authenticated internal endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Genesis hash — the "previous hash" for the first record
_GENESIS_HASH = "0" * 64


@dataclass
class AuditRecord:
    """A single audit log entry."""
    timestamp: str
    request_id: str
    tenant_id: str
    layer: str
    action: str
    decision_context: dict[str, Any] = field(default_factory=dict)
    source_ip: str = ""
    session_id: str = ""
    threat_categories: list[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    compliance: dict[str, str] = field(default_factory=dict)
    previous_hash: str = ""
    record_hash: str = ""

    def compute_hash(self, previous_hash: str) -> str:
        """Compute SHA-256 hash of this record chained with previous hash."""
        self.previous_hash = previous_hash
        # Hash all fields except record_hash itself
        hash_input = {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "layer": self.layer,
            "action": self.action,
            "decision_context": self.decision_context,
            "source_ip": self.source_ip,
            "session_id": self.session_id,
            "threat_categories": self.threat_categories,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "compliance": self.compliance,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(hash_input, sort_keys=True, separators=(",", ":"))
        self.record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.record_hash


# Compliance mapping applied to every audit record
_COMPLIANCE_MAP = {
    "nist_ai_rmf": "GOVERN 1.4",
    "iso_42001": "Clause 9",
    "eu_ai_act": "Articles 12, 19",
    "soc2": "CC7.2",
}


class AuditLogger:
    """Append-only JSONL audit logger with SHA-256 chain hashing.

    Thread-safe. Each record includes the hash of the previous record,
    creating a tamper-evident chain. Integrity can be verified by replaying
    the chain and checking each record's hash.
    """

    def __init__(self, log_path: str | Path | None = None, max_recent: int = 100):
        self._log_path = Path(log_path) if log_path else Path("aegis_audit.jsonl")
        self._max_recent = max_recent
        self._lock = threading.Lock()
        self._last_hash = _GENESIS_HASH
        self._recent: list[dict[str, Any]] = []
        self._record_count = 0

        # Ensure parent directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Recover chain hash from existing log file
        self._recover_chain()

    def _recover_chain(self) -> None:
        """Recover the last hash from existing log file for chain continuity."""
        if not self._log_path.exists():
            return
        try:
            with self._log_path.open("r") as f:
                last_line = None
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                        self._record_count += 1
                if last_line:
                    record = json.loads(last_line)
                    self._last_hash = record.get("record_hash", _GENESIS_HASH)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to recover audit chain: %s", e)

    def log(
        self,
        request_id: str,
        layer: str,
        action: str,
        *,
        tenant_id: str = "",
        decision_context: dict[str, Any] | None = None,
        source_ip: str = "",
        session_id: str = "",
        threat_categories: list[str] | None = None,
        confidence: float = 0.0,
        latency_ms: float = 0.0,
    ) -> AuditRecord:
        """Log a security decision. Returns the created AuditRecord."""
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id,
            tenant_id=tenant_id,
            layer=layer,
            action=action,
            decision_context=decision_context or {},
            source_ip=source_ip,
            session_id=session_id,
            threat_categories=threat_categories or [],
            confidence=confidence,
            latency_ms=latency_ms,
            compliance=_COMPLIANCE_MAP.copy(),
        )

        with self._lock:
            record.compute_hash(self._last_hash)
            self._last_hash = record.record_hash
            self._record_count += 1

            record_dict = asdict(record)

            # Append to JSONL file — flush after every write to prevent
            # log chain corruption on crash (last event must be persisted)
            try:
                with self._log_path.open("a") as f:
                    f.write(json.dumps(record_dict, separators=(",", ":")) + "\n")
                    f.flush()
            except OSError as e:
                logger.error("Failed to write audit record: %s", e)

            # Maintain recent buffer
            self._recent.append(record_dict)
            if len(self._recent) > self._max_recent:
                self._recent = self._recent[-self._max_recent:]

        return record

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent audit records."""
        with self._lock:
            return list(self._recent[-limit:])

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the integrity of the entire audit chain.

        Returns (is_valid, record_count). If any record has been tampered
        with, is_valid is False.
        """
        if not self._log_path.exists():
            return True, 0

        previous_hash = _GENESIS_HASH
        count = 0

        try:
            with self._log_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    count += 1

                    # Verify previous_hash chain
                    if record.get("previous_hash") != previous_hash:
                        logger.error(
                            "Chain break at record %d: expected previous_hash %s, got %s",
                            count, previous_hash, record.get("previous_hash"),
                        )
                        return False, count

                    # Recompute hash
                    stored_hash = record.get("record_hash", "")
                    recomputed = AuditRecord(
                        timestamp=record["timestamp"],
                        request_id=record["request_id"],
                        tenant_id=record["tenant_id"],
                        layer=record["layer"],
                        action=record["action"],
                        decision_context=record.get("decision_context", {}),
                        source_ip=record.get("source_ip", ""),
                        session_id=record.get("session_id", ""),
                        threat_categories=record.get("threat_categories", []),
                        confidence=record.get("confidence", 0.0),
                        latency_ms=record.get("latency_ms", 0.0),
                        compliance=record.get("compliance", {}),
                    )
                    recomputed.compute_hash(previous_hash)

                    if recomputed.record_hash != stored_hash:
                        logger.error(
                            "Hash mismatch at record %d: stored %s, computed %s",
                            count, stored_hash, recomputed.record_hash,
                        )
                        return False, count

                    previous_hash = stored_hash

        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.error("Chain verification error: %s", e)
            return False, count

        return True, count

    @property
    def record_count(self) -> int:
        with self._lock:
            return self._record_count


# Module-level singleton
_audit_logger: AuditLogger | None = None


def get_audit_logger(log_path: str | Path | None = None) -> AuditLogger:
    """Get or create the singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_path=log_path)
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset the singleton (for testing)."""
    global _audit_logger
    _audit_logger = None
