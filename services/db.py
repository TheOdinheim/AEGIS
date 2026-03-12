"""
PostgreSQL Database — Singleton async engine with graceful degradation.

Provides a shared async SQLAlchemy engine and session factory. If PostgreSQL
is unavailable (not configured, unreachable, or connection lost), all callers
get ``None`` and must fall back to in-memory implementations or buffer writes.

AEGIS must NEVER crash because PostgreSQL is down. The database is a
persistence and compliance enhancement, not a hard dependency.

ASSUMED-BREACH POSTURE: A compromised PostgreSQL instance could tamper with
audit logs, inject false threat indicators, or exfiltrate query patterns.
In production, PostgreSQL should use TLS, scram-sha-256 auth, network
isolation, and row-level security. Audit logs should be replicated to WORM
storage in a separate security domain.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Optional imports — asyncpg/sqlalchemy may not be installed in dev/test
try:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

_engine: Any | None = None
_session_factory: Any | None = None


async def init_db(database_url: str | None = None) -> Any | None:
    """Initialize the singleton async SQLAlchemy engine.

    Call once at application startup (in lifespan). Returns the engine if
    connection succeeds, or ``None`` if PostgreSQL is unavailable.

    Accepts ``postgresql+asyncpg://`` scheme URLs.
    """
    global _engine, _session_factory

    if not HAS_ASYNCPG:
        logger.warning("sqlalchemy/asyncpg not installed — falling back to no-DB mode")
        return None

    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url:
        logger.info("DATABASE_URL not set — falling back to no-DB mode")
        return None

    # Ensure asyncpg scheme
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif not url.startswith("postgresql+asyncpg://"):
        url = f"postgresql+asyncpg://{url}"

    try:
        _engine = create_async_engine(
            url,
            pool_size=10,
            max_overflow=5,
            pool_timeout=10,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=False,
        )
        _session_factory = sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        # Verify connectivity
        async with _engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connected: %s", _mask_url(url))
        return _engine
    except Exception as e:
        logger.warning("PostgreSQL unavailable (%s) — falling back to no-DB mode", e)
        _engine = None
        _session_factory = None
        return None


def get_engine() -> Any | None:
    """Return the shared async engine, or ``None`` if unavailable."""
    return _engine


def get_session_factory() -> Any | None:
    """Return the async session factory, or ``None`` if unavailable."""
    return _session_factory


async def get_session() -> Any | None:
    """Create and return a new async session, or ``None`` if DB unavailable.

    Callers MUST handle the ``None`` case gracefully. Usage::

        session = await get_session()
        if session is None:
            return  # fall back
        async with session:
            await session.execute(...)
            await session.commit()
    """
    if _session_factory is None:
        return None
    try:
        return _session_factory()
    except Exception as e:
        logger.warning("Failed to create DB session: %s", e)
        return None


async def close_db() -> None:
    """Dispose the engine and connection pool. Call during shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception as e:
            logger.warning("DB engine dispose error: %s", e)
        _engine = None
        _session_factory = None


async def db_health() -> dict[str, Any]:
    """Health check for PostgreSQL connectivity.

    Returns a dict suitable for inclusion in the /health endpoint.
    """
    if _engine is None:
        return {"status": "unavailable", "reason": "not initialized"}
    try:
        start = time.perf_counter()
        async with _engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.close()
        latency = (time.perf_counter() - start) * 1000
        return {
            "status": "connected",
            "latency_ms": round(latency, 2),
        }
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}


def _mask_url(url: str) -> str:
    """Mask password in database URL for logging."""
    if "@" in url and ":" in url.split("@")[0]:
        scheme_auth, rest = url.rsplit("@", 1)
        parts = scheme_auth.rsplit(":", 1)
        if len(parts) == 2:
            return f"{parts[0]}:***@{rest}"
    return url
