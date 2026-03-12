"""
Chaos runner — simulates component failures via monkey-patching.

Provides ChaosRunner class with methods to break individual AEGIS components
and ChaosContext context manager for automatic cleanup. All chaos operations
save original state and restore on exit.

CI-safe: operates on in-process objects, no live servers needed.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock

logger = logging.getLogger(__name__)


class ChaosRunner:
    """Simulates component failures by monkey-patching live AEGIS objects.

    Each method saves original state and provides restore capability.
    Use ChaosContext for automatic restore on exit.
    """

    def __init__(self) -> None:
        self._saved_state: dict[str, Any] = {}

    # ---- Redis failure ----

    @contextmanager
    def kill_redis_client(self, module: Any) -> Generator[None, None, None]:
        """Set the redis_client module-level _client to None.

        Args:
            module: The aegis.services.redis_client module or any object
                    with a _client attribute.
        """
        original = getattr(module, "_client", None)
        try:
            module._client = None
            yield
        finally:
            module._client = original

    # ---- PostgreSQL failure ----

    @contextmanager
    def kill_session_factory(self, target: Any, attr: str = "_session_factory") -> Generator[None, None, None]:
        """Replace a session_factory attribute with a broken factory.

        The broken factory raises OperationalError on __call__.

        Args:
            target: Object with session_factory (e.g., PgAuditLogger, ThreatVault).
            attr: Attribute name to replace.
        """
        original = getattr(target, attr, None)
        try:
            broken_factory = AsyncMock(side_effect=Exception("Simulated DB failure"))
            setattr(target, attr, broken_factory)
            yield
        finally:
            setattr(target, attr, original)

    # ---- FAISS corruption ----

    @contextmanager
    def corrupt_faiss_index(self, vault: Any) -> Generator[None, None, None]:
        """Set vault._index to None, forcing brute-force numpy fallback.

        Args:
            vault: ThreatVault instance.
        """
        original = vault._index
        try:
            vault._index = None
            yield
        finally:
            vault._index = original

    # ---- TLI manipulation ----

    @contextmanager
    def force_threat_level(self, policy_engine: Any, level: Any) -> Generator[None, None, None]:
        """Force the policy engine to a specific threat level.

        Args:
            policy_engine: PolicyEngine instance.
            level: ThreatLevel enum value.
        """
        original = policy_engine._threat_level
        try:
            policy_engine.threat_level = level
            yield
        finally:
            policy_engine.threat_level = original

    # ---- Audit buffer exhaustion ----

    @staticmethod
    def fill_audit_buffer(audit_logger: Any, count: int | None = None) -> int:
        """Fill the audit logger's deque with dummy entries.

        Args:
            audit_logger: PgAuditLogger instance.
            count: Number of entries to add. Defaults to maxlen.

        Returns:
            Number of entries added.
        """
        from aegis.services.audit_logger import PgAuditRecord

        target = count or audit_logger._buffer.maxlen or 10_000
        added = 0
        for i in range(target):
            record = PgAuditRecord(
                request_id=f"chaos-{i}",
                tenant_id="chaos-test",
                final_action="allow",
            )
            audit_logger._buffer.append(record)
            added += 1
        return added

    # ---- Upstream behavior ----

    @contextmanager
    def slow_upstream(self, mock_upstream_module: Any, latency_ms: int) -> Generator[None, None, None]:
        """Configure mock upstream latency.

        Args:
            mock_upstream_module: Module with configurable latency.
            latency_ms: Desired latency in milliseconds.
        """
        import os
        original = os.environ.get("MOCK_LATENCY_MS", "")
        try:
            os.environ["MOCK_LATENCY_MS"] = str(latency_ms)
            yield
        finally:
            if original:
                os.environ["MOCK_LATENCY_MS"] = original
            else:
                os.environ.pop("MOCK_LATENCY_MS", None)

    @contextmanager
    def upstream_errors(self, error_rate: float) -> Generator[None, None, None]:
        """Configure mock upstream error rate.

        Args:
            error_rate: Fraction of requests that should fail (0.0-1.0).
        """
        import os
        original = os.environ.get("MOCK_ERROR_RATE", "")
        try:
            os.environ["MOCK_ERROR_RATE"] = str(error_rate)
            yield
        finally:
            if original:
                os.environ["MOCK_ERROR_RATE"] = original
            else:
                os.environ.pop("MOCK_ERROR_RATE", None)


@asynccontextmanager
async def chaos_context(
    chaos: ChaosRunner,
    method_name: str,
    *args: Any,
    duration_seconds: float = 0.0,
    **kwargs: Any,
) -> AsyncGenerator[None, None]:
    """Async context manager wrapping a chaos action with automatic restore.

    Usage:
        chaos = ChaosRunner()
        async with chaos_context(chaos, "corrupt_faiss_index", vault):
            # vault._index is None here
            results = vault.search(embedding)
        # vault._index is restored here
    """
    method = getattr(chaos, method_name)
    with method(*args, **kwargs):
        if duration_seconds > 0:
            await asyncio.sleep(duration_seconds)
        yield
