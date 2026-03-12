"""
PostgreSQL Integration Tests — Stage 3 Production Hardening

Tests for:
- services/db.py — async engine initialization, graceful degradation, health check
- services/audit_logger.py — PgAuditLogger, prompt hash verification, buffer, circuit breaker events
- Threat vault PostgreSQL dual-write and lifecycle sync
- Signature store PostgreSQL dual-write and auto-deprecation
- Graceful degradation when DB is unavailable

All tests run without a real PostgreSQL instance — they mock the session factory
to verify the application logic. Real integration tests against PostgreSQL run
in Docker (docker compose up).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# -------------------------------------------------------------------------
# services/db.py tests
# -------------------------------------------------------------------------


class TestDbModule:
    """Tests for services/db.py — async engine singleton."""

    def test_import(self):
        """Module imports without error even if asyncpg not installed."""
        from aegis.services.db import init_db, close_db, db_health, get_engine, get_session_factory, get_session

    def test_get_engine_returns_none_initially(self):
        from aegis.services import db
        # Reset state
        db._engine = None
        db._session_factory = None
        assert db.get_engine() is None
        assert db.get_session_factory() is None

    @pytest.mark.asyncio
    async def test_init_db_no_url(self):
        """init_db with no URL returns None."""
        from aegis.services.db import init_db
        with patch.dict("os.environ", {}, clear=False):
            result = await init_db("")
            assert result is None

    @pytest.mark.asyncio
    async def test_init_db_no_asyncpg(self):
        """init_db returns None when sqlalchemy not available."""
        from aegis.services import db
        original = db.HAS_ASYNCPG
        db.HAS_ASYNCPG = False
        try:
            result = await db.init_db("postgresql://test:test@localhost/test")
            assert result is None
        finally:
            db.HAS_ASYNCPG = original

    @pytest.mark.asyncio
    async def test_close_db_noop_when_none(self):
        """close_db is safe to call when no engine exists."""
        from aegis.services import db
        db._engine = None
        await db.close_db()

    @pytest.mark.asyncio
    async def test_db_health_unavailable(self):
        """db_health returns unavailable when engine is None."""
        from aegis.services import db
        db._engine = None
        result = await db.db_health()
        assert result["status"] == "unavailable"
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_get_session_returns_none_when_no_factory(self):
        from aegis.services import db
        db._session_factory = None
        result = await db.get_session()
        assert result is None

    def test_mask_url_hides_password(self):
        from aegis.services.db import _mask_url
        url = "postgresql+asyncpg://aegis:secretpass@localhost:5432/aegis"
        masked = _mask_url(url)
        assert "secretpass" not in masked
        assert "***" in masked
        assert "localhost:5432/aegis" in masked

    def test_mask_url_no_password(self):
        from aegis.services.db import _mask_url
        url = "postgresql+asyncpg://localhost:5432/aegis"
        masked = _mask_url(url)
        assert masked == url

    @pytest.mark.asyncio
    async def test_init_db_converts_scheme(self):
        """init_db converts postgresql:// to postgresql+asyncpg://."""
        from aegis.services import db
        if not db.HAS_ASYNCPG:
            pytest.skip("asyncpg not installed")
        # This will fail to connect but should attempt with correct scheme
        result = await db.init_db("postgresql://test:test@nonexistent:5432/test")
        assert result is None  # Can't connect, graceful degradation


# -------------------------------------------------------------------------
# services/audit_logger.py — PgAuditRecord and prompt hash tests
# -------------------------------------------------------------------------


class TestPromptHash:
    """Tests for SHA-256 prompt hash — raw prompts are NEVER stored."""

    def test_compute_prompt_hash_basic(self):
        from aegis.services.audit_logger import compute_prompt_hash
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello world"},
        ]
        h = compute_prompt_hash(messages)
        assert len(h) == 64  # SHA-256 hex
        # Verify it matches manual computation
        expected = hashlib.sha256("You are a helper.|Hello world".encode()).hexdigest()
        assert h == expected

    def test_compute_prompt_hash_empty(self):
        from aegis.services.audit_logger import compute_prompt_hash
        h = compute_prompt_hash([])
        assert len(h) == 64
        assert h == hashlib.sha256(b"").hexdigest()

    def test_compute_prompt_hash_none(self):
        from aegis.services.audit_logger import compute_prompt_hash
        h = compute_prompt_hash(None)
        assert len(h) == 64

    def test_compute_prompt_hash_no_content(self):
        """Messages without content field are skipped."""
        from aegis.services.audit_logger import compute_prompt_hash
        messages = [{"role": "assistant", "tool_calls": []}]
        h = compute_prompt_hash(messages)
        assert len(h) == 64

    def test_compute_prompt_hash_deterministic(self):
        """Same messages always produce the same hash."""
        from aegis.services.audit_logger import compute_prompt_hash
        messages = [{"role": "user", "content": "test prompt"}]
        h1 = compute_prompt_hash(messages)
        h2 = compute_prompt_hash(messages)
        assert h1 == h2

    def test_compute_prompt_hash_different_content(self):
        """Different messages produce different hashes."""
        from aegis.services.audit_logger import compute_prompt_hash
        h1 = compute_prompt_hash([{"role": "user", "content": "hello"}])
        h2 = compute_prompt_hash([{"role": "user", "content": "world"}])
        assert h1 != h2


class TestPgAuditRecord:
    """Tests for PgAuditRecord data model."""

    def test_default_values(self):
        from aegis.services.audit_logger import PgAuditRecord
        record = PgAuditRecord(request_id="req-1")
        assert record.request_id == "req-1"
        assert record.final_action == "allow"
        assert record.prompt_hash == ""
        assert record.innate_is_threat is False
        assert record.adaptive_mcav == 0.0
        assert record.circuit_state == "closed"
        assert record.threat_level == 1

    def test_record_with_all_fields(self):
        from aegis.services.audit_logger import PgAuditRecord
        record = PgAuditRecord(
            request_id="req-2",
            tenant_id="tenant-1",
            user_id="user-1",
            session_id="sess-1",
            source_ip="10.0.0.1",
            model="gpt-4",
            prompt_hash="a" * 64,
            token_count=100,
            innate_is_threat=True,
            innate_confidence=0.95,
            innate_scanners={"regex": {"matched": True}},
            adaptive_is_threat=True,
            adaptive_confidence=0.88,
            adaptive_mcav=0.92,
            adaptive_analyzers={"deberta": {"score": 0.99}},
            output_pii_found=True,
            output_toxicity=0.3,
            output_leakage=False,
            output_validation={"should_block": False},
            final_action="block",
            block_reason="innate_threshold",
            threat_level=3,
            latency_innate_ms=2.5,
            latency_adaptive_ms=18.0,
            latency_output_ms=12.0,
            latency_total_ms=45.0,
            circuit_state="open",
        )
        assert record.final_action == "block"
        assert record.innate_scanners == {"regex": {"matched": True}}
        assert record.threat_level == 3

    def test_record_to_params(self):
        """_record_to_params produces correct SQL parameter dict."""
        from aegis.services.audit_logger import PgAuditRecord, _record_to_params
        record = PgAuditRecord(
            request_id="req-3",
            prompt_hash="b" * 64,
            innate_scanners={"test": True},
        )
        params = _record_to_params(record)
        assert params["request_id"] == "req-3"
        assert params["prompt_hash"] == "b" * 64
        # JSONB fields must be JSON strings
        assert json.loads(params["innate_scanners"]) == {"test": True}
        assert json.loads(params["adaptive_analyzers"]) == {}


# -------------------------------------------------------------------------
# PgAuditLogger tests
# -------------------------------------------------------------------------


class TestPgAuditLogger:
    """Tests for PgAuditLogger — fire-and-forget with buffer."""

    def test_create_logger(self):
        from aegis.services.audit_logger import PgAuditLogger
        logger = PgAuditLogger()
        assert logger.buffer_size == 0
        assert logger.total_logged == 0
        assert logger.session_factory is None

    def test_log_buffers_without_session_factory(self):
        """When DB unavailable, records are buffered in-memory."""
        from aegis.services.audit_logger import PgAuditLogger, PgAuditRecord
        logger = PgAuditLogger(session_factory=None)

        # Without event loop, log() buffers directly
        record = PgAuditRecord(request_id="req-buf-1", prompt_hash="x" * 64)
        logger.log(record)
        assert logger.total_logged == 1
        assert logger.buffer_size == 1
        assert logger.total_buffered == 1

    def test_buffer_max_size(self):
        """Buffer respects max size (oldest records dropped)."""
        from aegis.services.audit_logger import PgAuditLogger, PgAuditRecord, _MAX_BUFFER_SIZE
        logger = PgAuditLogger(session_factory=None)

        for i in range(_MAX_BUFFER_SIZE + 5):
            record = PgAuditRecord(request_id=f"req-{i}", prompt_hash="x" * 64)
            logger._buffer_record(record)

        assert logger.buffer_size == _MAX_BUFFER_SIZE

    def test_session_factory_setter(self):
        from aegis.services.audit_logger import PgAuditLogger
        logger = PgAuditLogger()
        assert logger.session_factory is None
        mock_factory = MagicMock()
        logger.session_factory = mock_factory
        assert logger.session_factory is mock_factory

    @pytest.mark.asyncio
    async def test_write_record_buffers_on_db_failure(self):
        """_write_record buffers when DB write fails."""
        from aegis.services.audit_logger import PgAuditLogger, PgAuditRecord

        # Create a mock session factory that raises
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_session.__aexit__ = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)

        logger = PgAuditLogger(session_factory=mock_factory)
        record = PgAuditRecord(request_id="req-fail", prompt_hash="y" * 64)
        await logger._write_record(record)
        assert logger.buffer_size == 1

    @pytest.mark.asyncio
    async def test_flush_buffer_empty(self):
        """flush_buffer on empty buffer returns 0."""
        from aegis.services.audit_logger import PgAuditLogger
        logger = PgAuditLogger(session_factory=None)
        result = await logger.flush_buffer()
        assert result == 0

    @pytest.mark.asyncio
    async def test_flush_buffer_no_factory(self):
        """flush_buffer with no session factory returns 0."""
        from aegis.services.audit_logger import PgAuditLogger, PgAuditRecord
        logger = PgAuditLogger(session_factory=None)
        logger._buffer_record(PgAuditRecord(request_id="r1", prompt_hash="z" * 64))
        result = await logger.flush_buffer()
        assert result == 0
        assert logger.buffer_size == 1  # Still buffered

    def test_singleton_pattern(self):
        from aegis.services.audit_logger import get_pg_audit_logger, reset_pg_audit_logger
        reset_pg_audit_logger()
        l1 = get_pg_audit_logger()
        l2 = get_pg_audit_logger()
        assert l1 is l2
        reset_pg_audit_logger()

    def test_stop_flush_task_noop(self):
        from aegis.services.audit_logger import PgAuditLogger
        logger = PgAuditLogger()
        logger.stop_flush_task()  # No error when no task exists


# -------------------------------------------------------------------------
# Circuit breaker event persistence tests
# -------------------------------------------------------------------------


class TestCircuitBreakerEventPersistence:
    """Tests for log_circuit_breaker_event."""

    @pytest.mark.asyncio
    async def test_log_cb_event_no_factory(self):
        """log_circuit_breaker_event is no-op with no session factory."""
        from aegis.services.audit_logger import log_circuit_breaker_event
        # Should not raise
        await log_circuit_breaker_event(
            session_factory=None,
            endpoint="primary",
            previous_state="closed",
            new_state="open",
            trigger_reason="test",
        )

    @pytest.mark.asyncio
    async def test_log_cb_event_db_failure(self):
        """log_circuit_breaker_event handles DB failure gracefully."""
        from aegis.services.audit_logger import log_circuit_breaker_event

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_session.__aexit__ = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)

        # Should not raise
        await log_circuit_breaker_event(
            session_factory=mock_factory,
            endpoint="primary",
            previous_state="closed",
            new_state="open",
            trigger_reason="Error rate 80%",
            failure_rate=0.8,
        )


# -------------------------------------------------------------------------
# Threat indicator persistence tests
# -------------------------------------------------------------------------


class TestThreatIndicatorPersistence:
    """Tests for threat indicator upsert/load/update."""

    @pytest.mark.asyncio
    async def test_upsert_indicator_no_factory(self):
        from aegis.services.audit_logger import upsert_threat_indicator
        result = await upsert_threat_indicator(
            session_factory=None,
            indicator_id="ind-1",
            prompt_hash="h" * 64,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_indicator_status_no_factory(self):
        from aegis.services.audit_logger import update_indicator_status
        result = await update_indicator_status(
            session_factory=None,
            indicator_id="ind-1",
            new_status="persistent",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_load_indicators_no_factory(self):
        from aegis.services.audit_logger import load_threat_indicators
        result = await load_threat_indicators(session_factory=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_upsert_indicator_db_failure(self):
        from aegis.services.audit_logger import upsert_threat_indicator

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_session.__aexit__ = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)

        result = await upsert_threat_indicator(
            session_factory=mock_factory,
            indicator_id="ind-1",
            prompt_hash="h" * 64,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_load_indicators_db_failure(self):
        from aegis.services.audit_logger import load_threat_indicators

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_session.__aexit__ = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)

        result = await load_threat_indicators(session_factory=mock_factory)
        assert result == []


# -------------------------------------------------------------------------
# Signature persistence tests
# -------------------------------------------------------------------------


class TestSignaturePersistence:
    """Tests for signature upsert/load/update."""

    @pytest.mark.asyncio
    async def test_upsert_signature_no_factory(self):
        from aegis.services.audit_logger import upsert_signature
        result = await upsert_signature(
            session_factory=None,
            signature_id="sig-1",
            pattern_text="test.*pattern",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_signature_status_no_factory(self):
        from aegis.services.audit_logger import update_signature_status
        result = await update_signature_status(
            session_factory=None,
            signature_id="sig-1",
            new_status="deprecated",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_load_signatures_no_factory(self):
        from aegis.services.audit_logger import load_signatures
        result = await load_signatures(session_factory=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_upsert_signature_db_failure(self):
        from aegis.services.audit_logger import upsert_signature

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_session.__aexit__ = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)

        result = await upsert_signature(
            session_factory=mock_factory,
            signature_id="sig-1",
            pattern_text="test",
        )
        assert result is False


# -------------------------------------------------------------------------
# ThreatVault PostgreSQL dual-write tests
# -------------------------------------------------------------------------


class TestThreatVaultDualWrite:
    """Tests for ThreatVault with PostgreSQL session_factory."""

    def _make_vault(self, session_factory=None):
        from aegis.config import MemoryConfig
        from aegis.layers.memory.threat_vault import ThreatVault
        config = MemoryConfig()
        return ThreatVault(config=config, session_factory=session_factory)

    def _make_indicator(self, indicator_id=None):
        from aegis.models.threat_indicator import ThreatIndicator, IndicatorSource, MemoryPhase
        from aegis.models.scan_result import ThreatCategory
        import numpy as np
        return ThreatIndicator(
            indicator_id=indicator_id or f"test-{uuid.uuid4().hex[:8]}",
            source=IndicatorSource.ADAPTIVE_DETECTION,
            confirmed=False,
            threat_category=ThreatCategory.PROMPT_INJECTION,
            confidence=0.95,
            embedding=list(np.random.randn(384).astype(float)),
            payload_hash=hashlib.sha256(b"test").hexdigest(),
            payload_summary="Test attack",
        )

    def test_vault_accepts_session_factory(self):
        mock_factory = MagicMock()
        vault = self._make_vault(session_factory=mock_factory)
        assert vault.session_factory is mock_factory

    def test_vault_session_factory_setter(self):
        vault = self._make_vault()
        assert vault.session_factory is None
        mock_factory = MagicMock()
        vault.session_factory = mock_factory
        assert vault.session_factory is mock_factory

    def test_add_without_session_factory(self):
        """add() works without PostgreSQL — in-memory only."""
        vault = self._make_vault()
        ind = self._make_indicator()
        vault.add(ind)
        assert vault.size == 1

    def test_add_with_session_factory_no_event_loop(self):
        """add() with session_factory but no event loop doesn't crash."""
        mock_factory = MagicMock()
        vault = self._make_vault(session_factory=mock_factory)
        ind = self._make_indicator()
        # _pg_upsert_indicator should catch RuntimeError (no event loop)
        vault.add(ind)
        assert vault.size == 1

    def test_record_hit_without_pg(self):
        """record_hit works without PostgreSQL."""
        vault = self._make_vault()
        ind = self._make_indicator(indicator_id="hit-test")
        vault.add(ind)
        vault.record_hit("hit-test")
        assert vault.get_indicator("hit-test").frequency == 2

    def test_promote_without_pg(self):
        """promote() works without PostgreSQL."""
        vault = self._make_vault()
        ind = self._make_indicator(indicator_id="promo-test")
        ind.confirmed = True
        vault.add(ind)
        result = vault.promote("promo-test")
        assert result is True
        from aegis.models.threat_indicator import MemoryPhase
        assert vault.get_indicator("promo-test").phase == MemoryPhase.PERSISTENT

    def test_demote_without_pg(self):
        """demote() works without PostgreSQL."""
        vault = self._make_vault()
        ind = self._make_indicator(indicator_id="demote-test")
        vault.add(ind)
        result = vault.demote("demote-test")
        assert result is True
        from aegis.models.threat_indicator import MemoryPhase
        assert vault.get_indicator("demote-test").phase == MemoryPhase.DORMANT

    @pytest.mark.asyncio
    async def test_load_from_postgres_no_factory(self):
        vault = self._make_vault()
        result = await vault.load_from_postgres()
        assert result == 0

    @pytest.mark.asyncio
    async def test_load_from_postgres_empty_db(self):
        """load_from_postgres returns 0 when DB has no indicators."""
        from aegis.services.audit_logger import load_threat_indicators
        with patch("aegis.services.audit_logger.load_threat_indicators", return_value=[]):
            vault = self._make_vault(session_factory=MagicMock())
            result = await vault.load_from_postgres()
            assert result == 0

    def test_lifecycle_update_without_pg(self):
        """lifecycle_update works without PostgreSQL."""
        vault = self._make_vault()
        ind = self._make_indicator(indicator_id="lifecycle-test")
        ind.confirmed = True
        vault.add(ind)
        transitions = vault.lifecycle_update()
        assert transitions["to_persistent"] == 1


# -------------------------------------------------------------------------
# SignatureStore PostgreSQL dual-write tests
# -------------------------------------------------------------------------


class TestSignatureStoreDualWrite:
    """Tests for SignatureStore with PostgreSQL session_factory."""

    def _make_store(self, session_factory=None):
        from aegis.config import MemoryConfig
        from aegis.layers.memory.signatures import SignatureStore
        return SignatureStore(config=MemoryConfig(), session_factory=session_factory)

    def test_store_accepts_session_factory(self):
        mock_factory = MagicMock()
        store = self._make_store(session_factory=mock_factory)
        assert store.session_factory is mock_factory

    def test_store_session_factory_setter(self):
        store = self._make_store()
        assert store.session_factory is None
        mock_factory = MagicMock()
        store.session_factory = mock_factory
        assert store.session_factory is mock_factory

    def test_add_without_session_factory(self):
        """add() works without PostgreSQL."""
        store = self._make_store()
        sig = store.add(pattern="test.*pattern", category="prompt_injection")
        assert store.size == 1
        assert sig.pattern == "test.*pattern"

    def test_add_with_session_factory_no_event_loop(self):
        """add() with session_factory but no event loop doesn't crash."""
        mock_factory = MagicMock()
        store = self._make_store(session_factory=mock_factory)
        sig = store.add(pattern="test.*pattern")
        assert store.size == 1

    def test_deprecate_without_pg(self):
        store = self._make_store()
        sig = store.add(pattern="dep.*test")
        result = store.deprecate(sig.signature_id)
        assert result is True
        assert store.get_by_id(sig.signature_id).deprecated is True

    def test_deprecation_check_without_pg(self):
        from aegis.config import MemoryConfig
        from aegis.layers.memory.signatures import SignatureStore
        config = MemoryConfig()
        config_dict = config.model_dump()
        store = SignatureStore(config=MemoryConfig())
        sig = store.add(pattern="fpr.*test")
        # Simulate enough matches with high FPR
        sig.match_count = 25
        sig.false_positive_count = 5  # FPR = 0.2 > 0.05
        deprecated = store.deprecation_check()
        assert sig.signature_id in deprecated


# -------------------------------------------------------------------------
# Graceful degradation integration tests
# -------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests verifying AEGIS never crashes when PostgreSQL is unavailable."""

    def test_pg_audit_log_without_db(self):
        """PgAuditLogger.log() never raises even without DB."""
        from aegis.services.audit_logger import PgAuditLogger, PgAuditRecord
        logger = PgAuditLogger(session_factory=None)
        record = PgAuditRecord(
            request_id="graceful-1",
            prompt_hash="g" * 64,
            final_action="block",
        )
        # Should not raise
        logger.log(record)
        assert logger.total_logged == 1

    def test_vault_add_with_broken_factory(self):
        """Vault add() survives a broken session factory."""
        from aegis.config import MemoryConfig
        from aegis.layers.memory.threat_vault import ThreatVault
        from aegis.models.threat_indicator import ThreatIndicator, IndicatorSource
        from aegis.models.scan_result import ThreatCategory
        import numpy as np

        broken_factory = MagicMock(side_effect=Exception("Connection refused"))
        vault = ThreatVault(config=MemoryConfig(), session_factory=broken_factory)
        ind = ThreatIndicator(
            indicator_id="broken-1",
            source=IndicatorSource.ADAPTIVE_DETECTION,
            confirmed=False,
            threat_category=ThreatCategory.PROMPT_INJECTION,
            confidence=0.9,
            embedding=list(np.random.randn(384).astype(float)),
            payload_hash="test",
            payload_summary="test",
        )
        # Should not raise — falls back to FAISS-only
        vault.add(ind)
        assert vault.size == 1

    def test_signature_store_with_broken_factory(self):
        """SignatureStore add() survives a broken session factory."""
        from aegis.config import MemoryConfig
        from aegis.layers.memory.signatures import SignatureStore
        broken_factory = MagicMock(side_effect=Exception("Connection refused"))
        store = SignatureStore(config=MemoryConfig(), session_factory=broken_factory)
        sig = store.add(pattern="broken.*test")
        assert store.size == 1

    @pytest.mark.asyncio
    async def test_vault_load_from_postgres_handles_failure(self):
        """load_from_postgres handles DB failure gracefully."""
        from aegis.config import MemoryConfig
        from aegis.layers.memory.threat_vault import ThreatVault

        with patch("aegis.services.audit_logger.load_threat_indicators", side_effect=Exception("DB exploded")):
            vault = ThreatVault(config=MemoryConfig(), session_factory=MagicMock())
            result = await vault.load_from_postgres()
            assert result == 0  # Graceful fallback


# -------------------------------------------------------------------------
# _fire_pg_audit helper test (verifies main.py wiring)
# -------------------------------------------------------------------------


class TestFirePgAudit:
    """Tests for the _fire_pg_audit helper in main.py."""

    def test_fire_pg_audit_no_logger(self):
        """_fire_pg_audit is no-op when _pg_audit is None."""
        import aegis.main as main_mod
        original = main_mod._pg_audit
        main_mod._pg_audit = None
        try:
            main_mod._fire_pg_audit("req-test")
        finally:
            main_mod._pg_audit = original

    def test_fire_pg_audit_with_mock_logger(self):
        """_fire_pg_audit creates a PgAuditRecord and calls log()."""
        import aegis.main as main_mod
        from aegis.services.audit_logger import PgAuditLogger

        mock_logger = MagicMock(spec=PgAuditLogger)
        original = main_mod._pg_audit
        main_mod._pg_audit = mock_logger
        try:
            main_mod._fire_pg_audit(
                "req-test-2",
                body={"messages": [{"role": "user", "content": "hello"}]},
                final_action="allow",
            )
            mock_logger.log.assert_called_once()
            call_args = mock_logger.log.call_args
            record = call_args[0][0]
            assert record.request_id == "req-test-2"
            assert record.final_action == "allow"
            assert len(record.prompt_hash) == 64  # SHA-256
        finally:
            main_mod._pg_audit = original

    def test_fire_pg_audit_prompt_hash_not_raw(self):
        """Verify _fire_pg_audit stores hash, not raw prompt text."""
        import aegis.main as main_mod
        from aegis.services.audit_logger import PgAuditLogger

        mock_logger = MagicMock(spec=PgAuditLogger)
        original = main_mod._pg_audit
        main_mod._pg_audit = mock_logger
        try:
            secret_prompt = "This is my secret system prompt with PII: SSN 123-45-6789"
            main_mod._fire_pg_audit(
                "req-test-3",
                body={"messages": [{"role": "system", "content": secret_prompt}]},
            )
            record = mock_logger.log.call_args[0][0]
            # prompt_hash must NOT contain raw text
            assert secret_prompt not in record.prompt_hash
            assert "123-45-6789" not in record.prompt_hash
            assert len(record.prompt_hash) == 64
        finally:
            main_mod._pg_audit = original
