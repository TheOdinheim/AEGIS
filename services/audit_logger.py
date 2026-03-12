"""
PostgreSQL Audit Logger — Fire-and-forget audit log sink.

Writes to the ``audit_log`` table with structured JSONB columns for
per-layer detection results. CRITICAL: Raw prompts are NEVER stored.
Only SHA-256 hashes of prompt content are persisted.

Fire-and-forget pattern: audit writes use ``asyncio.create_task()`` so
they never add latency to the request path. When PostgreSQL is unavailable,
records buffer in-memory (max 10,000) and flush automatically when the
connection recovers.

ASSUMED-BREACH POSTURE: If the audit database is compromised, an attacker
can read SHA-256 prompt hashes (which cannot be reversed to reconstruct
prompts) and detection metadata. They cannot read the actual prompts.
In production, audit_log should be replicated to WORM storage and the
database should use TLS + row-level security.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Maximum buffered records when DB is unavailable
_MAX_BUFFER_SIZE = 10_000

# Retry interval for flushing buffered records (seconds)
_FLUSH_RETRY_INTERVAL = 30


@dataclass
class PgAuditRecord:
    """A single audit log entry destined for PostgreSQL."""
    request_id: str
    tenant_id: str = ""
    user_id: str = ""
    session_id: str = ""
    source_ip: str = ""
    model: str = ""
    prompt_hash: str = ""  # SHA-256 of concatenated message content
    token_count: int = 0

    # L2 Innate results
    innate_is_threat: bool = False
    innate_confidence: float = 0.0
    innate_scanners: dict[str, Any] = field(default_factory=dict)

    # L3 Adaptive results
    adaptive_is_threat: bool = False
    adaptive_confidence: float = 0.0
    adaptive_mcav: float = 0.0
    adaptive_analyzers: dict[str, Any] = field(default_factory=dict)

    # L5 Output validation
    output_pii_found: bool = False
    output_toxicity: float = 0.0
    output_leakage: bool = False
    output_validation: dict[str, Any] = field(default_factory=dict)

    # Decision
    final_action: str = "allow"
    block_reason: str = ""
    threat_level: int = 1

    # Performance
    latency_innate_ms: float = 0.0
    latency_adaptive_ms: float = 0.0
    latency_output_ms: float = 0.0
    latency_total_ms: float = 0.0

    # System state
    circuit_state: str = "closed"


def compute_prompt_hash(messages: list[dict[str, Any]] | None) -> str:
    """Compute SHA-256 hash of concatenated message content.

    NEVER stores raw prompt text. Only the hash is persisted.
    """
    if not messages:
        return hashlib.sha256(b"").hexdigest()
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            parts.append(str(content))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class PgAuditLogger:
    """PostgreSQL audit logger with fire-and-forget writes and in-memory buffer.

    Usage::

        pg_audit = PgAuditLogger(session_factory)
        pg_audit.log(record)  # fire-and-forget, returns immediately
        await pg_audit.flush_buffer()  # manual flush of buffered records
    """

    def __init__(self, session_factory: Any | None = None):
        self._session_factory = session_factory
        self._buffer: deque[PgAuditRecord] = deque(maxlen=_MAX_BUFFER_SIZE)
        self._flush_task: asyncio.Task | None = None
        self._total_logged: int = 0
        self._total_buffered: int = 0
        self._total_flushed: int = 0

    @property
    def session_factory(self) -> Any | None:
        return self._session_factory

    @session_factory.setter
    def session_factory(self, value: Any | None) -> None:
        self._session_factory = value

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_logged(self) -> int:
        return self._total_logged

    @property
    def total_buffered(self) -> int:
        return self._total_buffered

    @property
    def total_flushed(self) -> int:
        return self._total_flushed

    def log(self, record: PgAuditRecord) -> None:
        """Fire-and-forget audit log write.

        Creates an asyncio task to write the record to PostgreSQL.
        If DB is unavailable, buffers the record in-memory.
        Never raises, never blocks.
        """
        self._total_logged += 1
        try:
            asyncio.create_task(self._write_record(record))
        except RuntimeError:
            # No event loop — buffer for later
            self._buffer_record(record)

    async def _write_record(self, record: PgAuditRecord) -> None:
        """Write a single record to PostgreSQL."""
        if self._session_factory is None:
            self._buffer_record(record)
            return

        try:
            async with self._session_factory() as session:
                from sqlalchemy import text
                await session.execute(
                    text("""
                        INSERT INTO audit_log (
                            request_id, tenant_id, user_id, session_id, source_ip,
                            model, prompt_hash, token_count,
                            innate_is_threat, innate_confidence, innate_scanners,
                            adaptive_is_threat, adaptive_confidence, adaptive_mcav,
                            adaptive_analyzers,
                            output_pii_found, output_toxicity, output_leakage,
                            output_validation,
                            final_action, block_reason, threat_level,
                            latency_innate_ms, latency_adaptive_ms,
                            latency_output_ms, latency_total_ms,
                            circuit_state
                        ) VALUES (
                            :request_id, :tenant_id, :user_id, :session_id,
                            :source_ip, :model, :prompt_hash, :token_count,
                            :innate_is_threat, :innate_confidence,
                            :innate_scanners::jsonb,
                            :adaptive_is_threat, :adaptive_confidence,
                            :adaptive_mcav, :adaptive_analyzers::jsonb,
                            :output_pii_found, :output_toxicity, :output_leakage,
                            :output_validation::jsonb,
                            :final_action, :block_reason, :threat_level,
                            :latency_innate_ms, :latency_adaptive_ms,
                            :latency_output_ms, :latency_total_ms,
                            :circuit_state
                        )
                    """),
                    _record_to_params(record),
                )
                await session.commit()
        except Exception as e:
            logger.warning("Audit log DB write failed (%s) — buffering", e)
            self._buffer_record(record)

    def _buffer_record(self, record: PgAuditRecord) -> None:
        """Add a record to the in-memory buffer."""
        self._buffer.append(record)
        self._total_buffered += 1
        if len(self._buffer) == _MAX_BUFFER_SIZE:
            logger.warning("Audit buffer at capacity (%d) — oldest records will be dropped", _MAX_BUFFER_SIZE)

    async def flush_buffer(self) -> int:
        """Flush buffered records to PostgreSQL.

        Returns the number of records successfully flushed.
        """
        if not self._buffer or self._session_factory is None:
            return 0

        flushed = 0
        records_to_retry: list[PgAuditRecord] = []

        while self._buffer:
            record = self._buffer.popleft()
            try:
                async with self._session_factory() as session:
                    from sqlalchemy import text
                    await session.execute(
                        text("""
                            INSERT INTO audit_log (
                                request_id, tenant_id, user_id, session_id,
                                source_ip, model, prompt_hash, token_count,
                                innate_is_threat, innate_confidence,
                                innate_scanners,
                                adaptive_is_threat, adaptive_confidence,
                                adaptive_mcav, adaptive_analyzers,
                                output_pii_found, output_toxicity,
                                output_leakage, output_validation,
                                final_action, block_reason, threat_level,
                                latency_innate_ms, latency_adaptive_ms,
                                latency_output_ms, latency_total_ms,
                                circuit_state
                            ) VALUES (
                                :request_id, :tenant_id, :user_id, :session_id,
                                :source_ip, :model, :prompt_hash, :token_count,
                                :innate_is_threat, :innate_confidence,
                                :innate_scanners::jsonb,
                                :adaptive_is_threat, :adaptive_confidence,
                                :adaptive_mcav, :adaptive_analyzers::jsonb,
                                :output_pii_found, :output_toxicity,
                                :output_leakage, :output_validation::jsonb,
                                :final_action, :block_reason, :threat_level,
                                :latency_innate_ms, :latency_adaptive_ms,
                                :latency_output_ms, :latency_total_ms,
                                :circuit_state
                            )
                        """),
                        _record_to_params(record),
                    )
                    await session.commit()
                    flushed += 1
            except Exception as e:
                # Put back and stop — DB is likely still down
                records_to_retry.append(record)
                break

        # Put unprocessed records back at the front
        for r in reversed(records_to_retry):
            self._buffer.appendleft(r)

        if flushed > 0:
            self._total_flushed += flushed
            logger.info("Flushed %d buffered audit records to PostgreSQL", flushed)

        return flushed

    async def start_flush_loop(self) -> None:
        """Periodically flush buffered records when DB recovers."""
        while True:
            await asyncio.sleep(_FLUSH_RETRY_INTERVAL)
            if self._buffer and self._session_factory is not None:
                try:
                    await self.flush_buffer()
                except Exception as e:
                    logger.warning("Buffer flush failed: %s", e)

    def start_flush_task(self) -> None:
        """Start the background flush loop as an asyncio task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._flush_task = loop.create_task(self.start_flush_loop())
        except RuntimeError:
            pass

    def stop_flush_task(self) -> None:
        """Cancel the background flush task."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            self._flush_task = None


def _record_to_params(record: PgAuditRecord) -> dict[str, Any]:
    """Convert a PgAuditRecord to SQL parameter dict."""
    import json
    return {
        "request_id": record.request_id,
        "tenant_id": record.tenant_id or None,
        "user_id": record.user_id or None,
        "session_id": record.session_id or None,
        "source_ip": record.source_ip or None,
        "model": record.model or None,
        "prompt_hash": record.prompt_hash,
        "token_count": record.token_count,
        "innate_is_threat": record.innate_is_threat,
        "innate_confidence": record.innate_confidence,
        "innate_scanners": json.dumps(record.innate_scanners),
        "adaptive_is_threat": record.adaptive_is_threat,
        "adaptive_confidence": record.adaptive_confidence,
        "adaptive_mcav": record.adaptive_mcav,
        "adaptive_analyzers": json.dumps(record.adaptive_analyzers),
        "output_pii_found": record.output_pii_found,
        "output_toxicity": record.output_toxicity,
        "output_leakage": record.output_leakage,
        "output_validation": json.dumps(record.output_validation),
        "final_action": record.final_action,
        "block_reason": record.block_reason or None,
        "threat_level": record.threat_level,
        "latency_innate_ms": record.latency_innate_ms,
        "latency_adaptive_ms": record.latency_adaptive_ms,
        "latency_output_ms": record.latency_output_ms,
        "latency_total_ms": record.latency_total_ms,
        "circuit_state": record.circuit_state,
    }


# ---------------------------------------------------------------------------
# Circuit Breaker Event Logger
# ---------------------------------------------------------------------------


async def log_circuit_breaker_event(
    session_factory: Any | None,
    endpoint: str,
    previous_state: str,
    new_state: str,
    trigger_reason: str = "",
    failure_rate: float = 0.0,
    window_seconds: int = 60,
    cooldown_seconds: int = 30,
) -> None:
    """Write a circuit breaker state transition to PostgreSQL.

    Fire-and-forget. Silently fails if DB is unavailable.
    """
    if session_factory is None:
        return

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO circuit_breaker_events (
                        endpoint, previous_state, new_state,
                        trigger_reason, failure_rate,
                        window_seconds, cooldown_seconds
                    ) VALUES (
                        :endpoint, :previous_state, :new_state,
                        :trigger_reason, :failure_rate,
                        :window_seconds, :cooldown_seconds
                    )
                """),
                {
                    "endpoint": endpoint,
                    "previous_state": previous_state,
                    "new_state": new_state,
                    "trigger_reason": trigger_reason,
                    "failure_rate": failure_rate,
                    "window_seconds": window_seconds,
                    "cooldown_seconds": cooldown_seconds,
                },
            )
            await session.commit()
    except Exception as e:
        logger.warning("Circuit breaker event DB write failed: %s", e)


# ---------------------------------------------------------------------------
# Threat Indicator Persistence
# ---------------------------------------------------------------------------


async def upsert_threat_indicator(
    session_factory: Any | None,
    indicator_id: str,
    mitre_tactic: str = "",
    threat_category: str = "",
    confidence: float = 0.0,
    source: str = "adaptive_detection",
    prompt_hash: str = "",
    embedding_ref: str = "",
    status: str = "acute",
    hit_count: int = 1,
    source_count: int = 1,
) -> bool:
    """Upsert a threat indicator into PostgreSQL.

    Returns True if successful, False if DB unavailable.
    """
    if session_factory is None:
        return False

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO threat_indicators (
                        indicator_id, mitre_tactic, threat_category,
                        confidence, source, prompt_hash, embedding_ref,
                        status, hit_count, source_count
                    ) VALUES (
                        :indicator_id, :mitre_tactic, :threat_category,
                        :confidence, :source, :prompt_hash, :embedding_ref,
                        :status, :hit_count, :source_count
                    )
                    ON CONFLICT (indicator_id) DO UPDATE SET
                        hit_count = threat_indicators.hit_count + 1,
                        last_seen = NOW(),
                        source_count = GREATEST(
                            threat_indicators.source_count, :source_count
                        ),
                        status = :status,
                        updated_at = NOW()
                """),
                {
                    "indicator_id": indicator_id,
                    "mitre_tactic": mitre_tactic,
                    "threat_category": threat_category,
                    "confidence": confidence,
                    "source": source,
                    "prompt_hash": prompt_hash,
                    "embedding_ref": embedding_ref,
                    "status": status,
                    "hit_count": hit_count,
                    "source_count": source_count,
                },
            )
            await session.commit()
            return True
    except Exception as e:
        logger.warning("Threat indicator DB write failed: %s", e)
        return False


async def update_indicator_status(
    session_factory: Any | None,
    indicator_id: str,
    new_status: str,
) -> bool:
    """Update the lifecycle status of a threat indicator."""
    if session_factory is None:
        return False

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    UPDATE threat_indicators
                    SET status = :status, updated_at = NOW()
                    WHERE indicator_id = :indicator_id
                """),
                {"indicator_id": indicator_id, "status": new_status},
            )
            await session.commit()
            return True
    except Exception as e:
        logger.warning("Indicator status update failed: %s", e)
        return False


async def load_threat_indicators(
    session_factory: Any | None,
) -> list[dict[str, Any]]:
    """Load all threat indicators from PostgreSQL.

    Returns a list of dicts, or empty list if DB unavailable.
    """
    if session_factory is None:
        return []

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            result = await session.execute(
                text("""
                    SELECT indicator_id, mitre_tactic, threat_category,
                           confidence, source, prompt_hash, embedding_ref,
                           status, hit_count, first_seen, last_seen,
                           source_count
                    FROM threat_indicators
                    ORDER BY last_seen DESC
                """)
            )
            rows = result.fetchall()
            return [
                {
                    "indicator_id": str(row[0]),
                    "mitre_tactic": row[1],
                    "threat_category": row[2],
                    "confidence": row[3],
                    "source": row[4],
                    "prompt_hash": row[5],
                    "embedding_ref": row[6],
                    "status": row[7],
                    "hit_count": row[8],
                    "first_seen": row[9].isoformat() if row[9] else None,
                    "last_seen": row[10].isoformat() if row[10] else None,
                    "source_count": row[11],
                }
                for row in rows
            ]
    except Exception as e:
        logger.warning("Failed to load threat indicators from DB: %s", e)
        return []


# ---------------------------------------------------------------------------
# Signature Persistence
# ---------------------------------------------------------------------------


async def upsert_signature(
    session_factory: Any | None,
    signature_id: str,
    pattern_type: str = "regex",
    pattern_text: str = "",
    source_indicator_id: str | None = None,
    affinity_score: float = 0.0,
    tpr: float = 0.0,
    fpr: float = 0.0,
    status: str = "active",
    match_count: int = 0,
) -> bool:
    """Upsert a signature into PostgreSQL."""
    if session_factory is None:
        return False

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            await session.execute(
                text("""
                    INSERT INTO signatures (
                        signature_id, pattern_type, pattern_text,
                        source_indicator_id, affinity_score,
                        tpr, fpr, status, match_count
                    ) VALUES (
                        :signature_id, :pattern_type, :pattern_text,
                        :source_indicator_id, :affinity_score,
                        :tpr, :fpr, :status, :match_count
                    )
                    ON CONFLICT (signature_id) DO UPDATE SET
                        tpr = :tpr,
                        fpr = :fpr,
                        status = :status,
                        match_count = :match_count,
                        updated_at = NOW()
                """),
                {
                    "signature_id": signature_id,
                    "pattern_type": pattern_type,
                    "pattern_text": pattern_text,
                    "source_indicator_id": source_indicator_id or None,
                    "affinity_score": affinity_score,
                    "tpr": tpr,
                    "fpr": fpr,
                    "status": status,
                    "match_count": match_count,
                },
            )
            await session.commit()
            return True
    except Exception as e:
        logger.warning("Signature DB write failed: %s", e)
        return False


async def update_signature_status(
    session_factory: Any | None,
    signature_id: str,
    new_status: str,
    match_count: int | None = None,
    fpr: float | None = None,
) -> bool:
    """Update signature status and optional metrics."""
    if session_factory is None:
        return False

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            params: dict[str, Any] = {
                "signature_id": signature_id,
                "status": new_status,
            }
            set_clauses = ["status = :status", "updated_at = NOW()"]
            if match_count is not None:
                set_clauses.append("match_count = :match_count")
                params["match_count"] = match_count
            if fpr is not None:
                set_clauses.append("fpr = :fpr")
                params["fpr"] = fpr

            await session.execute(
                text(f"""
                    UPDATE signatures
                    SET {', '.join(set_clauses)}
                    WHERE signature_id = :signature_id
                """),
                params,
            )
            await session.commit()
            return True
    except Exception as e:
        logger.warning("Signature status update failed: %s", e)
        return False


async def load_signatures(
    session_factory: Any | None,
) -> list[dict[str, Any]]:
    """Load all signatures from PostgreSQL."""
    if session_factory is None:
        return []

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            result = await session.execute(
                text("""
                    SELECT signature_id, pattern_type, pattern_text,
                           source_indicator_id, affinity_score,
                           tpr, fpr, status, match_count, last_matched
                    FROM signatures
                    ORDER BY created_at DESC
                """)
            )
            rows = result.fetchall()
            return [
                {
                    "signature_id": str(row[0]),
                    "pattern_type": row[1],
                    "pattern_text": row[2],
                    "source_indicator_id": str(row[3]) if row[3] else None,
                    "affinity_score": row[4],
                    "tpr": row[5],
                    "fpr": row[6],
                    "status": row[7],
                    "match_count": row[8],
                    "last_matched": row[9].isoformat() if row[9] else None,
                }
                for row in rows
            ]
    except Exception as e:
        logger.warning("Failed to load signatures from DB: %s", e)
        return []


# ---------------------------------------------------------------------------
# Per-Tenant Audit Query
# ---------------------------------------------------------------------------


async def get_tenant_audit_log(
    session_factory: Any | None,
    tenant_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Query audit log entries for a specific tenant.

    Returns a list of audit records, or empty list if DB unavailable.
    """
    if session_factory is None:
        return []

    try:
        async with session_factory() as session:
            from sqlalchemy import text
            result = await session.execute(
                text("""
                    SELECT request_id, timestamp, user_id, session_id,
                           model, final_action, block_reason, threat_level,
                           innate_confidence, adaptive_mcav, circuit_state
                    FROM audit_log
                    WHERE tenant_id = :tenant_id
                    ORDER BY timestamp DESC
                    LIMIT :limit OFFSET :offset
                """),
                {"tenant_id": tenant_id, "limit": limit, "offset": offset},
            )
            rows = result.fetchall()
            return [
                {
                    "request_id": str(row[0]),
                    "timestamp": row[1].isoformat() if row[1] else None,
                    "user_id": row[2],
                    "session_id": row[3],
                    "model": row[4],
                    "final_action": row[5],
                    "block_reason": row[6],
                    "threat_level": row[7],
                    "innate_confidence": row[8],
                    "adaptive_mcav": row[9],
                    "circuit_state": row[10],
                }
                for row in rows
            ]
    except Exception as e:
        logger.warning("Tenant audit log query failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_pg_audit: PgAuditLogger | None = None


def get_pg_audit_logger() -> PgAuditLogger:
    """Get or create the singleton PostgreSQL audit logger."""
    global _pg_audit
    if _pg_audit is None:
        _pg_audit = PgAuditLogger()
    return _pg_audit


def reset_pg_audit_logger() -> None:
    """Reset the singleton (for testing)."""
    global _pg_audit
    if _pg_audit:
        _pg_audit.stop_flush_task()
    _pg_audit = None
