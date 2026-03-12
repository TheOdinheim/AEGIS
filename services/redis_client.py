"""
Redis Client — Singleton async connection pool with graceful degradation.

Provides a shared Redis connection across the application. If Redis is
unavailable (not configured, unreachable, or connection lost), all callers
get ``None`` and must fall back to in-memory implementations.

AEGIS must NEVER crash because Redis is down. Redis is a performance
optimization and durability enhancement, not a hard dependency.

ASSUMED-BREACH POSTURE: A compromised Redis instance could poison rate
limit state, inject false events into the event bus, or exfiltrate data
via MONITOR commands. In production, Redis should use TLS, AUTH, and
network-level isolation (private subnet, security group).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Optional import — redis package may not be installed in dev/test
try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    HAS_REDIS = False

_pool: Any | None = None
_client: Any | None = None


async def init_redis(redis_url: str | None = None) -> Any | None:
    """Initialize the singleton Redis connection.

    Call once at application startup (in lifespan). Returns the Redis
    client if connection succeeds, or ``None`` if Redis is unavailable.
    """
    global _pool, _client

    if not HAS_REDIS:
        logger.warning("redis package not installed — falling back to in-memory")
        return None

    url = redis_url or os.environ.get("REDIS_URL", "")
    if not url:
        logger.info("REDIS_URL not set — falling back to in-memory")
        return None

    try:
        _client = aioredis.from_url(
            url,
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        # Verify connectivity
        await _client.ping()
        logger.info("Redis connected: %s", _mask_url(url))
        return _client
    except Exception as e:
        logger.warning("Redis unavailable (%s) — falling back to in-memory", e)
        _client = None
        return None


async def get_redis() -> Any | None:
    """Return the shared Redis client, or ``None`` if unavailable.

    Callers MUST handle the ``None`` case by falling back to in-memory
    implementations. Never raise on Redis unavailability.
    """
    if _client is None:
        return None
    try:
        await _client.ping()
        return _client
    except Exception:
        logger.warning("Redis ping failed — connection lost")
        return None


async def close_redis() -> None:
    """Close the Redis connection pool. Call during shutdown."""
    global _client, _pool
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as e:
            logger.warning("Redis close error: %s", e)
        _client = None
        _pool = None


async def redis_health() -> dict[str, Any]:
    """Health check for Redis connectivity.

    Returns a dict suitable for inclusion in the /health endpoint:
    ``{"redis": {"status": "connected", "latency_ms": 0.5}}``
    or ``{"redis": {"status": "unavailable"}}``
    """
    import time

    if _client is None:
        return {"status": "unavailable", "reason": "not initialized"}
    try:
        start = time.perf_counter()
        await _client.ping()
        latency = (time.perf_counter() - start) * 1000
        info = await _client.info("memory")
        return {
            "status": "connected",
            "latency_ms": round(latency, 2),
            "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 1),
        }
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}


def _mask_url(url: str) -> str:
    """Mask password in Redis URL for logging."""
    # redis://:password@host:port/db -> redis://:***@host:port/db
    if "@" in url and ":" in url.split("@")[0]:
        scheme_auth, rest = url.rsplit("@", 1)
        parts = scheme_auth.rsplit(":", 1)
        if len(parts) == 2:
            return f"{parts[0]}:***@{rest}"
    return url
