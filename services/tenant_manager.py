"""
Tenant Manager — Multi-tenant configuration with in-memory caching.

Loads tenant configurations from the ``tenants`` PostgreSQL table on startup,
caches them in-memory keyed by ``api_key_hash`` (SHA-256), and refreshes the
cache periodically (every 60 seconds). Individual tenant lookups are cached
for 5 minutes to avoid DB round-trips on every request.

Graceful degradation: if PostgreSQL is unavailable, cached tenants remain
usable. If no cache exists (first startup without DB), falls back to a
default tenant derived from environment-variable config. AEGIS must NEVER
crash because PostgreSQL is down.

ASSUMED-BREACH POSTURE: A compromised PostgreSQL could inject malicious
tenant configs (e.g., widened rate limits, permissive thresholds). The
TenantManager treats DB-sourced config as advisory — global safety limits
(L6 Tier 1 policies) override tenant config. In production, tenant rows
should be protected with row-level security and config changes should
require multi-party approval with audit trail.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Cache entry TTL: 5 minutes
_CACHE_TTL_SECONDS = 300

# Background refresh interval: 60 seconds
_REFRESH_INTERVAL_SECONDS = 60


@dataclass
class TenantConfig:
    """Per-tenant configuration mirroring the ``tenants`` table schema."""
    tenant_id: str
    name: str
    api_key_hash: str = ""
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    max_tokens_per_request: int = 128_000
    block_threshold: float = 0.85
    alert_threshold: float = 0.50
    allowed_models: list[str] = field(default_factory=list)
    policy_tier: str = "standard"  # standard | strict | permissive | custom
    custom_policy: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class _CacheEntry:
    """Wraps a TenantConfig with expiry time for per-lookup caching."""
    tenant: TenantConfig
    expires_at: float  # monotonic time


class TenantManager:
    """Multi-tenant configuration manager with PostgreSQL-backed cache.

    Usage::

        tm = TenantManager(session_factory, default_config=env_defaults)
        await tm.load_all()       # Startup: bulk load from DB
        tm.start_refresh()        # Background refresh every 60s

        tenant = await tm.resolve_tenant("sha256-of-api-key")
        if tenant is None:
            # Unknown key — reject or use default
    """

    def __init__(
        self,
        session_factory: Any | None = None,
        default_config: TenantConfig | None = None,
    ):
        self._session_factory = session_factory
        # Primary cache: api_key_hash → _CacheEntry
        self._cache: dict[str, _CacheEntry] = {}
        # Bulk cache: tenant_id → TenantConfig (from full DB load)
        self._tenants_by_id: dict[str, TenantConfig] = {}
        # Fallback when no DB and no cache
        self._default = default_config or TenantConfig(
            tenant_id="default",
            name="default",
        )
        self._refresh_task: asyncio.Task | None = None
        self._last_refresh: float = 0.0
        self._total_resolved: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    @property
    def session_factory(self) -> Any | None:
        return self._session_factory

    @session_factory.setter
    def session_factory(self, value: Any | None) -> None:
        self._session_factory = value

    @property
    def cache_size(self) -> int:
        """Number of tenants in the lookup cache."""
        return len(self._cache)

    @property
    def tenant_count(self) -> int:
        """Number of tenants loaded from the DB."""
        return len(self._tenants_by_id)

    @property
    def last_refresh(self) -> float:
        """Monotonic timestamp of last successful cache refresh."""
        return self._last_refresh

    @property
    def default_config(self) -> TenantConfig:
        return self._default

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics for /health."""
        return {
            "cache_size": self.cache_size,
            "tenant_count": self.tenant_count,
            "last_refresh": self._last_refresh,
            "total_resolved": self._total_resolved,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "db_connected": self._session_factory is not None,
        }

    async def load_all(self) -> int:
        """Bulk-load all enabled tenants from PostgreSQL into cache.

        Returns the number of tenants loaded, or 0 if DB unavailable.
        Called on startup and by the periodic refresh task.
        """
        if self._session_factory is None:
            return 0

        try:
            async with self._session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(
                    text("""
                        SELECT tenant_id, name, api_key_hash,
                               rate_limit_rpm, rate_limit_burst,
                               block_threshold, alert_threshold,
                               allowed_models, policy_tier, custom_policy,
                               enabled
                        FROM tenants
                        WHERE enabled = TRUE
                    """)
                )
                rows = result.fetchall()

            now = time.monotonic()
            loaded = 0
            new_by_id: dict[str, TenantConfig] = {}

            for row in rows:
                tc = TenantConfig(
                    tenant_id=str(row[0]),
                    name=row[1] or "",
                    api_key_hash=row[2] or "",
                    rate_limit_rpm=row[3] if row[3] is not None else 60,
                    rate_limit_burst=row[4] if row[4] is not None else 10,
                    block_threshold=row[5] if row[5] is not None else 0.85,
                    alert_threshold=row[6] if row[6] is not None else 0.50,
                    allowed_models=list(row[7]) if row[7] else [],
                    policy_tier=row[8] or "standard",
                    custom_policy=dict(row[9]) if row[9] else {},
                    enabled=bool(row[10]),
                )
                new_by_id[tc.tenant_id] = tc
                # Also index by api_key_hash for fast lookup
                if tc.api_key_hash:
                    self._cache[tc.api_key_hash] = _CacheEntry(
                        tenant=tc,
                        expires_at=now + _CACHE_TTL_SECONDS,
                    )
                loaded += 1

            self._tenants_by_id = new_by_id
            self._last_refresh = now
            logger.info("Loaded %d tenants from PostgreSQL", loaded)
            return loaded

        except Exception as e:
            logger.warning("Failed to load tenants from DB (%s) — using cached data", e)
            return 0

    async def resolve_tenant(self, api_key_hash: str) -> TenantConfig | None:
        """Resolve a tenant configuration by API key hash.

        Checks the in-memory cache first (5-minute TTL). On cache miss,
        queries PostgreSQL. Returns None if the key is not found.
        Falls back to default config if DB is unavailable and key matches
        no cached tenant.
        """
        self._total_resolved += 1
        now = time.monotonic()

        # Check cache (fast path)
        entry = self._cache.get(api_key_hash)
        if entry and entry.expires_at > now:
            self._cache_hits += 1
            return entry.tenant

        # Cache miss — try DB lookup
        self._cache_misses += 1
        tenant = await self._lookup_from_db(api_key_hash)
        if tenant:
            self._cache[api_key_hash] = _CacheEntry(
                tenant=tenant,
                expires_at=now + _CACHE_TTL_SECONDS,
            )
            return tenant

        # Expired cache entry is better than nothing
        if entry:
            return entry.tenant

        return None

    async def _lookup_from_db(self, api_key_hash: str) -> TenantConfig | None:
        """Query PostgreSQL for a tenant by API key hash."""
        if self._session_factory is None:
            return None

        try:
            async with self._session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(
                    text("""
                        SELECT tenant_id, name, api_key_hash,
                               rate_limit_rpm, rate_limit_burst,
                               block_threshold, alert_threshold,
                               allowed_models, policy_tier, custom_policy,
                               enabled
                        FROM tenants
                        WHERE api_key_hash = :hash AND enabled = TRUE
                        LIMIT 1
                    """),
                    {"hash": api_key_hash},
                )
                row = result.fetchone()

            if not row:
                return None

            tc = TenantConfig(
                tenant_id=str(row[0]),
                name=row[1] or "",
                api_key_hash=row[2] or "",
                rate_limit_rpm=row[3] if row[3] is not None else 60,
                rate_limit_burst=row[4] if row[4] is not None else 10,
                block_threshold=row[5] if row[5] is not None else 0.85,
                alert_threshold=row[6] if row[6] is not None else 0.50,
                allowed_models=list(row[7]) if row[7] else [],
                policy_tier=row[8] or "standard",
                custom_policy=dict(row[9]) if row[9] else {},
                enabled=bool(row[10]),
            )
            self._tenants_by_id[tc.tenant_id] = tc
            return tc

        except Exception as e:
            logger.warning("Tenant lookup failed (%s) — using cached/default", e)
            return None

    def get_tenant_by_id(self, tenant_id: str) -> TenantConfig | None:
        """Look up a tenant by ID from the in-memory cache (sync)."""
        return self._tenants_by_id.get(tenant_id)

    async def refresh_cache(self) -> int:
        """Refresh the tenant cache from PostgreSQL.

        Called periodically by the background task and can be called
        manually for immediate refresh.
        """
        return await self.load_all()

    def start_refresh(self) -> None:
        """Start the background cache refresh task (every 60s)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._refresh_task = loop.create_task(self._refresh_loop())
        except RuntimeError:
            pass

    def stop_refresh(self) -> None:
        """Stop the background refresh task."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            self._refresh_task = None

    async def _refresh_loop(self) -> None:
        """Periodic background refresh of the tenant cache."""
        while True:
            await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)
            try:
                await self.refresh_cache()
            except Exception as e:
                logger.warning("Tenant cache refresh failed: %s", e)

    def evict_expired(self) -> int:
        """Remove expired cache entries. Returns number evicted."""
        now = time.monotonic()
        expired = [k for k, v in self._cache.items() if v.expires_at <= now]
        for k in expired:
            self._cache.pop(k, None)
        return len(expired)


def hash_api_key(api_key: str) -> str:
    """Compute SHA-256 hash of an API key for tenant lookup."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
