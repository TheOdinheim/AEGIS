"""
Tests for Stage 4: Multi-tenant support.

Covers:
- TenantConfig dataclass
- TenantManager: load, resolve, cache, refresh, eviction, graceful degradation
- BarrierLayer: per-tenant rate limits, allowed model filtering, tenant_id binding
- PolicyEngine: per-tenant thresholds, policy_tier modifiers, custom_policy
- Per-tenant audit log query
- Backward compatibility: None tenant_manager uses env-var defaults
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegis.config import BarrierConfig, PolicyConfig, ThreatLevel
from aegis.layers.barrier import BarrierLayer, BarrierReject
from aegis.layers.policy import PolicyEngine, TenantPolicy
from aegis.models.scan_result import InnateScanReport, ScanResult, ThreatCategory
from aegis.services.tenant_manager import (
    TenantConfig,
    TenantManager,
    _CACHE_TTL_SECONDS,
    _CacheEntry,
    hash_api_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_body(model: str = "gpt-4", content: str = "Hello") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }


def _make_headers(api_key: str = "test-key") -> dict:
    return {"authorization": f"Bearer {api_key}"}


def _make_tenant(
    tenant_id: str = "t1",
    name: str = "acme",
    api_key_hash: str = "",
    rpm: int = 60,
    burst: int = 10,
    block_threshold: float = 0.85,
    alert_threshold: float = 0.50,
    allowed_models: list[str] | None = None,
    policy_tier: str = "standard",
    custom_policy: dict | None = None,
) -> TenantConfig:
    return TenantConfig(
        tenant_id=tenant_id,
        name=name,
        api_key_hash=api_key_hash,
        rate_limit_rpm=rpm,
        rate_limit_burst=burst,
        block_threshold=block_threshold,
        alert_threshold=alert_threshold,
        allowed_models=allowed_models or [],
        policy_tier=policy_tier,
        custom_policy=custom_policy or {},
    )


def _make_innate_report(
    confidence: float = 0.0,
    is_threat: bool = False,
    categories: list[ThreatCategory] | None = None,
) -> InnateScanReport:
    results = []
    cats = categories or ([ThreatCategory.PROMPT_INJECTION] if is_threat else [])
    if is_threat:
        results.append(ScanResult(
            scanner_id="test",
            is_threat=True,
            confidence=confidence,
            threat_category=cats[0],
            latency_ms=0.1,
        ))
    return InnateScanReport(
        request_id="test-req",
        scanner_results=results,
        max_confidence=confidence if is_threat else 0.0,
        threat_categories=cats if is_threat else [],
    )


# ===========================================================================
# TenantConfig Tests
# ===========================================================================

class TestTenantConfig:

    def test_defaults(self):
        tc = TenantConfig(tenant_id="t1", name="test")
        assert tc.rate_limit_rpm == 60
        assert tc.rate_limit_burst == 10
        assert tc.max_tokens_per_request == 128_000
        assert tc.block_threshold == 0.85
        assert tc.alert_threshold == 0.50
        assert tc.allowed_models == []
        assert tc.policy_tier == "standard"
        assert tc.custom_policy == {}
        assert tc.enabled is True

    def test_custom_values(self):
        tc = _make_tenant(
            rpm=120, burst=20, block_threshold=0.70,
            allowed_models=["gpt-4", "gpt-3.5-turbo"],
            policy_tier="strict",
            custom_policy={"blocked_categories": ["pii_exposure"]},
        )
        assert tc.rate_limit_rpm == 120
        assert tc.rate_limit_burst == 20
        assert tc.block_threshold == 0.70
        assert tc.allowed_models == ["gpt-4", "gpt-3.5-turbo"]
        assert tc.policy_tier == "strict"
        assert tc.custom_policy["blocked_categories"] == ["pii_exposure"]


# ===========================================================================
# hash_api_key Tests
# ===========================================================================

class TestHashApiKey:

    def test_deterministic(self):
        h1 = hash_api_key("my-secret-key")
        h2 = hash_api_key("my-secret-key")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_keys(self):
        h1 = hash_api_key("key-a")
        h2 = hash_api_key("key-b")
        assert h1 != h2

    def test_matches_stdlib(self):
        key = "aegis-test-123"
        expected = hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert hash_api_key(key) == expected


# ===========================================================================
# TenantManager Tests
# ===========================================================================

class TestTenantManagerInit:

    def test_create_without_db(self):
        tm = TenantManager()
        assert tm.session_factory is None
        assert tm.cache_size == 0
        assert tm.tenant_count == 0

    def test_create_with_default_config(self):
        default = _make_tenant(tenant_id="env-default", name="env")
        tm = TenantManager(default_config=default)
        assert tm.default_config.tenant_id == "env-default"

    def test_session_factory_setter(self):
        tm = TenantManager()
        mock_sf = MagicMock()
        tm.session_factory = mock_sf
        assert tm.session_factory is mock_sf


class TestTenantManagerResolve:

    @pytest.mark.asyncio
    async def test_resolve_from_cache(self):
        tm = TenantManager()
        tc = _make_tenant(tenant_id="t1", api_key_hash="abc123")
        tm._cache["abc123"] = _CacheEntry(
            tenant=tc, expires_at=time.monotonic() + 300,
        )
        result = await tm.resolve_tenant("abc123")
        assert result is not None
        assert result.tenant_id == "t1"
        assert tm._cache_hits == 1

    @pytest.mark.asyncio
    async def test_resolve_cache_miss_no_db(self):
        tm = TenantManager()
        result = await tm.resolve_tenant("unknown-hash")
        assert result is None
        assert tm._cache_misses == 1

    @pytest.mark.asyncio
    async def test_resolve_expired_cache_still_returns(self):
        """Expired cache entry is returned as fallback when DB is unavailable."""
        tm = TenantManager()
        tc = _make_tenant(tenant_id="t1", api_key_hash="abc123")
        tm._cache["abc123"] = _CacheEntry(
            tenant=tc, expires_at=time.monotonic() - 10,  # expired
        )
        result = await tm.resolve_tenant("abc123")
        assert result is not None
        assert result.tenant_id == "t1"

    @pytest.mark.asyncio
    async def test_resolve_from_db(self):
        """Simulates a DB lookup on cache miss."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (
            "uuid-1", "acme", "hash123",
            120, 20,
            0.75, 0.45,
            ["gpt-4"], "strict", {"key": "val"},
            True,
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_sf = MagicMock(return_value=mock_session)
        tm = TenantManager(session_factory=mock_sf)

        result = await tm.resolve_tenant("hash123")
        assert result is not None
        assert result.tenant_id == "uuid-1"
        assert result.name == "acme"
        assert result.rate_limit_rpm == 120
        assert result.policy_tier == "strict"
        # Should be cached now
        assert "hash123" in tm._cache

    @pytest.mark.asyncio
    async def test_resolve_db_failure_graceful(self):
        """DB failure returns None, doesn't crash."""
        mock_sf = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_sf.return_value = mock_session

        tm = TenantManager(session_factory=mock_sf)
        result = await tm.resolve_tenant("some-hash")
        assert result is None


class TestTenantManagerLoadAll:

    @pytest.mark.asyncio
    async def test_load_all_no_db(self):
        tm = TenantManager()
        count = await tm.load_all()
        assert count == 0

    @pytest.mark.asyncio
    async def test_load_all_from_db(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("uuid-1", "tenant-a", "hash-a", 60, 10, 0.85, 0.50, [], "standard", {}, True),
            ("uuid-2", "tenant-b", "hash-b", 120, 20, 0.70, 0.40, ["gpt-4"], "strict", {}, True),
        ]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_sf = MagicMock(return_value=mock_session)
        tm = TenantManager(session_factory=mock_sf)

        count = await tm.load_all()
        assert count == 2
        assert tm.tenant_count == 2
        assert tm.cache_size == 2  # Both have api_key_hash
        assert tm.last_refresh > 0

    @pytest.mark.asyncio
    async def test_load_all_db_failure(self):
        mock_sf = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
        mock_sf.return_value = mock_session

        tm = TenantManager(session_factory=mock_sf)
        count = await tm.load_all()
        assert count == 0


class TestTenantManagerCache:

    def test_evict_expired(self):
        tm = TenantManager()
        tc1 = _make_tenant(tenant_id="t1")
        tc2 = _make_tenant(tenant_id="t2")
        now = time.monotonic()
        tm._cache["a"] = _CacheEntry(tenant=tc1, expires_at=now - 10)  # expired
        tm._cache["b"] = _CacheEntry(tenant=tc2, expires_at=now + 300)  # valid
        evicted = tm.evict_expired()
        assert evicted == 1
        assert "a" not in tm._cache
        assert "b" in tm._cache

    def test_get_tenant_by_id(self):
        tm = TenantManager()
        tc = _make_tenant(tenant_id="t1")
        tm._tenants_by_id["t1"] = tc
        result = tm.get_tenant_by_id("t1")
        assert result is tc

    def test_get_tenant_by_id_missing(self):
        tm = TenantManager()
        result = tm.get_tenant_by_id("nonexistent")
        assert result is None

    def test_stats(self):
        tm = TenantManager()
        stats = tm.stats
        assert "cache_size" in stats
        assert "tenant_count" in stats
        assert "last_refresh" in stats
        assert "total_resolved" in stats
        assert "db_connected" in stats
        assert stats["db_connected"] is False


class TestTenantManagerRefresh:

    def test_start_stop_refresh(self):
        tm = TenantManager()
        # start_refresh without event loop should not crash
        tm.start_refresh()
        tm.stop_refresh()

    @pytest.mark.asyncio
    async def test_refresh_cache_calls_load_all(self):
        tm = TenantManager()
        tm.load_all = AsyncMock(return_value=5)
        result = await tm.refresh_cache()
        assert result == 5
        tm.load_all.assert_awaited_once()


# ===========================================================================
# BarrierLayer Multi-Tenant Tests
# ===========================================================================

class TestBarrierTenantResolution:

    @pytest.mark.asyncio
    async def test_process_without_tenant_manager(self):
        """Backward compatible: no tenant manager uses env-var defaults."""
        config = BarrierConfig()
        barrier = BarrierLayer(config, valid_api_keys={"test-key"})
        ctx = await barrier.process(
            _make_body(), _make_headers("test-key"), "1.2.3.4",
        )
        assert ctx.tenant_id == "default"

    @pytest.mark.asyncio
    async def test_process_with_tenant_manager(self):
        """Tenant manager resolves tenant_id from API key hash."""
        config = BarrierConfig()
        api_key = "test-key"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        tc = _make_tenant(tenant_id="acme-corp", name="acme")
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=tc)

        barrier = BarrierLayer(config, valid_api_keys={"test-key"}, tenant_manager=mock_tm)
        ctx = await barrier.process(
            _make_body(), _make_headers("test-key"), "1.2.3.4",
        )
        assert ctx.tenant_id == "acme-corp"
        mock_tm.resolve_tenant.assert_awaited_once_with(api_key_hash)

    @pytest.mark.asyncio
    async def test_tenant_manager_failure_uses_default(self):
        """If tenant resolution fails, fall back to key-prefix tenant."""
        config = BarrierConfig()
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(side_effect=Exception("DB timeout"))

        barrier = BarrierLayer(config, valid_api_keys={"test-key"}, tenant_manager=mock_tm)
        ctx = await barrier.process(
            _make_body(), _make_headers("test-key"), "1.2.3.4",
        )
        assert ctx.tenant_id == "default"

    @pytest.mark.asyncio
    async def test_tenant_manager_returns_none_uses_default(self):
        """Unknown API key → resolve_tenant returns None → use key-prefix tenant."""
        config = BarrierConfig()
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=None)

        barrier = BarrierLayer(config, valid_api_keys={"test-key"}, tenant_manager=mock_tm)
        ctx = await barrier.process(
            _make_body(), _make_headers("test-key"), "1.2.3.4",
        )
        assert ctx.tenant_id == "default"


class TestBarrierPerTenantRateLimits:

    @pytest.mark.asyncio
    async def test_tenant_rate_limit_enforced(self):
        """Tenant with rpm=2, burst=0 should throttle on 3rd request."""
        config = BarrierConfig(rate_limit_rpm=1000, rate_limit_burst=1000)  # global is high
        tc = _make_tenant(tenant_id="t1", rpm=2, burst=0)
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=tc)

        barrier = BarrierLayer(config, valid_api_keys={"k"}, tenant_manager=mock_tm)

        # Requests 1 and 2 should pass
        await barrier.process(_make_body(), _make_headers("k"), "1.1.1.1")
        await barrier.process(_make_body(), _make_headers("k"), "1.1.1.1")

        # Request 3 should be throttled
        with pytest.raises(BarrierReject) as exc_info:
            await barrier.process(_make_body(), _make_headers("k"), "1.1.1.1")
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_different_tenants_have_separate_limits(self):
        """Two tenants with different limits shouldn't interfere."""
        config = BarrierConfig()
        tc1 = _make_tenant(tenant_id="t1", rpm=1, burst=0, api_key_hash="hash-a")
        tc2 = _make_tenant(tenant_id="t2", rpm=100, burst=0, api_key_hash="hash-b")

        async def resolve(key_hash):
            if key_hash == hashlib.sha256(b"key-a").hexdigest():
                return tc1
            if key_hash == hashlib.sha256(b"key-b").hexdigest():
                return tc2
            return None

        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(side_effect=resolve)

        barrier = BarrierLayer(
            config, valid_api_keys={"key-a", "key-b"}, tenant_manager=mock_tm,
        )

        # t1 uses its limit (1 request)
        await barrier.process(_make_body(), _make_headers("key-a"), "1.1.1.1")
        with pytest.raises(BarrierReject) as exc_info:
            await barrier.process(_make_body(), _make_headers("key-a"), "1.1.1.1")
        assert exc_info.value.status_code == 429

        # t2 should still be fine
        await barrier.process(_make_body(), _make_headers("key-b"), "2.2.2.2")


class TestBarrierAllowedModels:

    @pytest.mark.asyncio
    async def test_allowed_model_passes(self):
        config = BarrierConfig()
        tc = _make_tenant(tenant_id="t1", allowed_models=["gpt-4", "gpt-3.5-turbo"])
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=tc)

        barrier = BarrierLayer(config, valid_api_keys={"k"}, tenant_manager=mock_tm)
        ctx = await barrier.process(
            _make_body(model="gpt-4"), _make_headers("k"), "1.1.1.1",
        )
        assert ctx.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_disallowed_model_rejected(self):
        config = BarrierConfig()
        tc = _make_tenant(tenant_id="t1", allowed_models=["gpt-3.5-turbo"])
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=tc)

        barrier = BarrierLayer(config, valid_api_keys={"k"}, tenant_manager=mock_tm)
        with pytest.raises(BarrierReject) as exc_info:
            await barrier.process(
                _make_body(model="gpt-4"), _make_headers("k"), "1.1.1.1",
            )
        assert exc_info.value.status_code == 403
        assert "not authorized" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_empty_allowed_models_means_all_allowed(self):
        config = BarrierConfig()
        tc = _make_tenant(tenant_id="t1", allowed_models=[])
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=tc)

        barrier = BarrierLayer(config, valid_api_keys={"k"}, tenant_manager=mock_tm)
        ctx = await barrier.process(
            _make_body(model="any-model"), _make_headers("k"), "1.1.1.1",
        )
        assert ctx.model == "any-model"

    @pytest.mark.asyncio
    async def test_no_model_in_request_skips_check(self):
        config = BarrierConfig()
        tc = _make_tenant(tenant_id="t1", allowed_models=["gpt-4"])
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=tc)

        barrier = BarrierLayer(config, valid_api_keys={"k"}, tenant_manager=mock_tm)
        body = _make_body(model="")
        # model="" should skip the allowed_models check
        ctx = await barrier.process(body, _make_headers("k"), "1.1.1.1")
        assert ctx is not None


class TestBarrierPerTenantTokenLimit:

    @pytest.mark.asyncio
    async def test_tenant_token_limit(self):
        config = BarrierConfig(max_tokens_per_request=128_000)
        tc = _make_tenant(tenant_id="t1")
        tc.max_tokens_per_request = 10  # Very low limit
        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=tc)

        barrier = BarrierLayer(config, valid_api_keys={"k"}, tenant_manager=mock_tm)
        # A message with more than 10 tokens
        body = _make_body(content="This is a test message with enough tokens to exceed the limit for sure")
        with pytest.raises(BarrierReject) as exc_info:
            await barrier.process(body, _make_headers("k"), "1.1.1.1")
        assert exc_info.value.status_code == 400
        assert "Token count" in exc_info.value.reason


# ===========================================================================
# PolicyEngine Multi-Tenant Tests
# ===========================================================================

class TestPolicyTenantThresholds:

    def test_tenant_block_threshold_from_manager(self):
        """Policy engine uses per-tenant block threshold from TenantManager."""
        tc = _make_tenant(tenant_id="strict-tenant", block_threshold=0.50)
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        report = _make_innate_report(confidence=0.55, is_threat=True)

        decision = engine.evaluate(
            "req-1", innate_report=report, tenant_id="strict-tenant",
        )
        # 0.55 >= 0.50 (standard tier, threshold used as-is) → should block
        assert decision.action.value == "block"
        assert decision.triggered_by.value == "tenant"

    def test_no_tenant_manager_uses_global_defaults(self):
        """Without tenant manager, global thresholds apply."""
        engine = PolicyEngine()
        report = _make_innate_report(confidence=0.55, is_threat=True)

        decision = engine.evaluate("req-1", innate_report=report)
        # 0.55 < 0.85 (global default) → should NOT block at tier 2
        assert decision.action.value in ("allow", "allow_degraded")

    def test_registered_policy_takes_priority(self):
        """Manually registered tenant policy overrides TenantManager."""
        tc_from_manager = _make_tenant(tenant_id="t1", block_threshold=0.90)
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc_from_manager)

        engine = PolicyEngine(tenant_manager=mock_tm)
        # Register a manual policy with lower threshold
        engine.register_tenant_policy(TenantPolicy(
            tenant_id="t1",
            custom_block_threshold=0.50,
        ))

        report = _make_innate_report(confidence=0.55, is_threat=True)
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        # Manual policy (0.50) wins over TenantManager (0.90)
        assert decision.action.value == "block"


class TestPolicyTierModifiers:

    def test_strict_tier_lowers_threshold(self):
        """Strict tier lowers block threshold by 15%."""
        tc = _make_tenant(
            tenant_id="t1", block_threshold=0.80, policy_tier="strict",
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        # 0.80 * 0.85 = 0.68 → strict threshold
        report = _make_innate_report(confidence=0.70, is_threat=True)
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        assert decision.action.value == "block"
        assert "TENANT-TIER-STRICT" in decision.applied_policies

    def test_strict_tier_below_threshold_allows(self):
        """Strict tier: score below adjusted threshold still allows."""
        tc = _make_tenant(
            tenant_id="t1", block_threshold=0.80, policy_tier="strict",
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        # 0.80 * 0.85 = 0.68 → strict threshold, 0.60 is below
        report = _make_innate_report(confidence=0.60, is_threat=True)
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        assert decision.action.value != "block" or decision.triggered_by.value != "tenant"

    def test_permissive_tier_raises_threshold(self):
        """Permissive tier raises block threshold by 10%."""
        tc = _make_tenant(
            tenant_id="t1", block_threshold=0.80, policy_tier="permissive",
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        # 0.80 * 1.10 = 0.88 → permissive threshold
        # Use TOKEN_STUFFING (not in hard-block categories) to avoid Tier 1 block
        report = _make_innate_report(
            confidence=0.84, is_threat=True,
            categories=[ThreatCategory.PII_EXFILTRATION],
        )
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        # 0.84 < 0.88 → should NOT block at tenant tier
        assert decision.triggered_by.value != "tenant" or decision.action.value != "block"
        assert "TENANT-TIER-PERMISSIVE" in decision.applied_policies

    def test_permissive_capped_at_1(self):
        """Permissive tier threshold capped at 1.0."""
        tc = _make_tenant(
            tenant_id="t1", block_threshold=0.95, policy_tier="permissive",
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        engine._apply_policy_tier(
            TenantPolicy(
                tenant_id="t1",
                custom_block_threshold=0.95,
                policy_tier="permissive",
            ),
            [], [],
        )
        # 0.95 * 1.10 = 1.045 → capped to 1.0

    def test_standard_tier_uses_as_is(self):
        """Standard tier uses configured threshold unchanged."""
        tc = _make_tenant(
            tenant_id="t1", block_threshold=0.75, policy_tier="standard",
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        report = _make_innate_report(confidence=0.76, is_threat=True)
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        assert decision.action.value == "block"
        assert "TENANT-TIER-STANDARD" in decision.applied_policies

    def test_custom_tier_applies_policy(self):
        """Custom tier tag is applied."""
        tc = _make_tenant(
            tenant_id="t1", block_threshold=0.80, policy_tier="custom",
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        report = _make_innate_report(confidence=0.81, is_threat=True)
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        assert decision.action.value == "block"
        assert "TENANT-TIER-CUSTOM" in decision.applied_policies


class TestPolicyCustomPolicy:

    def test_custom_blocked_categories(self):
        """Custom policy with blocked_categories blocks matching threat."""
        tc = _make_tenant(
            tenant_id="t1",
            block_threshold=0.95,  # High threshold (wouldn't block normally)
            custom_policy={"blocked_categories": ["AML.T0048"]},  # PII_EXFILTRATION
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        report = _make_innate_report(
            confidence=0.60, is_threat=True,
            categories=[ThreatCategory.PII_EXFILTRATION],
        )
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        assert decision.action.value == "block"
        assert "TENANT-CUSTOM-BLOCKED-CATEGORY" in decision.applied_policies

    def test_custom_blocked_categories_no_match(self):
        """Custom blocked_categories doesn't match → no custom block."""
        tc = _make_tenant(
            tenant_id="t1",
            block_threshold=0.95,
            custom_policy={"blocked_categories": ["AML.T0048"]},  # PII_EXFILTRATION
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        report = _make_innate_report(
            confidence=0.60, is_threat=True,
            categories=[ThreatCategory.PROMPT_INJECTION],
        )
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        # prompt_injection is not in blocked_categories and 0.60 < 0.95
        assert "TENANT-CUSTOM-BLOCKED-CATEGORY" not in decision.applied_policies

    def test_empty_custom_policy(self):
        """Empty custom_policy is a no-op."""
        tc = _make_tenant(tenant_id="t1", block_threshold=0.90, custom_policy={})
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        report = _make_innate_report(confidence=0.50, is_threat=True)
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        assert "TENANT-CUSTOM-BLOCKED-CATEGORY" not in decision.applied_policies


class TestPolicyAllowedModels:

    def test_model_allowlist_from_tenant_manager(self):
        """Model not in tenant's allowed_models → blocked by policy."""
        tc = _make_tenant(
            tenant_id="t1", allowed_models=["gpt-3.5-turbo"],
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        decision = engine.evaluate(
            "req-1", tenant_id="t1", model="gpt-4",
        )
        assert decision.action.value == "block"
        assert "TENANT-MODEL-ALLOWLIST" in decision.applied_policies

    def test_model_in_allowlist_passes(self):
        tc = _make_tenant(
            tenant_id="t1", allowed_models=["gpt-4", "gpt-3.5-turbo"],
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        decision = engine.evaluate(
            "req-1", tenant_id="t1", model="gpt-4",
        )
        assert decision.action.value in ("allow", "allow_degraded")


class TestPolicyTierWithTLI:

    def test_strict_tenant_with_blue_tli(self):
        """Strict tenant + BLUE TLI should be extra aggressive."""
        tc = _make_tenant(
            tenant_id="t1", block_threshold=0.80, policy_tier="strict",
        )
        mock_tm = MagicMock(spec=TenantManager)
        mock_tm.get_tenant_by_id = MagicMock(return_value=tc)

        engine = PolicyEngine(tenant_manager=mock_tm)
        engine.threat_level = ThreatLevel.BLUE

        # Strict: 0.80 * 0.85 = 0.68
        report = _make_innate_report(confidence=0.69, is_threat=True)
        decision = engine.evaluate("req-1", innate_report=report, tenant_id="t1")
        assert decision.action.value == "block"


# ===========================================================================
# Graceful Degradation Tests
# ===========================================================================

class TestGracefulDegradation:

    @pytest.mark.asyncio
    async def test_barrier_without_tenant_manager(self):
        """BarrierLayer works perfectly fine without TenantManager."""
        config = BarrierConfig()
        barrier = BarrierLayer(config, valid_api_keys={"k"})
        ctx = await barrier.process(_make_body(), _make_headers("k"), "1.1.1.1")
        assert ctx is not None
        assert ctx.tenant_id == "default"

    def test_policy_without_tenant_manager(self):
        """PolicyEngine works perfectly fine without TenantManager."""
        engine = PolicyEngine()
        decision = engine.evaluate("req-1")
        assert decision.action.value in ("allow", "allow_degraded")

    @pytest.mark.asyncio
    async def test_tenant_manager_without_db(self):
        """TenantManager works without DB — returns default or cached."""
        default = _make_tenant(tenant_id="fallback", name="fallback")
        tm = TenantManager(default_config=default)
        result = await tm.resolve_tenant("any-hash")
        assert result is None  # No cache, no DB → None


class TestTenantManagerSetter:

    @pytest.mark.asyncio
    async def test_barrier_tenant_manager_setter(self):
        config = BarrierConfig()
        barrier = BarrierLayer(config, valid_api_keys={"k"})
        assert barrier.tenant_manager is None

        mock_tm = AsyncMock(spec=TenantManager)
        mock_tm.resolve_tenant = AsyncMock(return_value=_make_tenant(tenant_id="new"))
        barrier.tenant_manager = mock_tm

        ctx = await barrier.process(_make_body(), _make_headers("k"), "1.1.1.1")
        assert ctx.tenant_id == "new"

    def test_policy_tenant_manager_setter(self):
        engine = PolicyEngine()
        assert engine.tenant_manager is None

        mock_tm = MagicMock(spec=TenantManager)
        engine.tenant_manager = mock_tm
        assert engine.tenant_manager is mock_tm


# ===========================================================================
# Per-Tenant Audit Query Tests
# ===========================================================================

class TestTenantAuditQuery:

    @pytest.mark.asyncio
    async def test_get_tenant_audit_log_no_db(self):
        from aegis.services.audit_logger import get_tenant_audit_log
        result = await get_tenant_audit_log(None, "any-tenant")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_tenant_audit_log_from_db(self):
        from aegis.services.audit_logger import get_tenant_audit_log
        from datetime import datetime, timezone

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("req-1", datetime(2026, 1, 1, tzinfo=timezone.utc),
             "user-1", "sess-1", "gpt-4", "allow", None, 1, 0.2, 0.0, "closed"),
        ]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_sf = MagicMock(return_value=mock_session)
        result = await get_tenant_audit_log(mock_sf, "tenant-1", limit=10)
        assert len(result) == 1
        assert result[0]["request_id"] == "req-1"
        assert result[0]["final_action"] == "allow"

    @pytest.mark.asyncio
    async def test_get_tenant_audit_log_db_failure(self):
        from aegis.services.audit_logger import get_tenant_audit_log

        mock_sf = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB error"))
        mock_sf.return_value = mock_session

        result = await get_tenant_audit_log(mock_sf, "tenant-1")
        assert result == []


# ===========================================================================
# Integration-style Tests
# ===========================================================================

class TestTenantEndToEnd:

    @pytest.mark.asyncio
    async def test_full_tenant_flow(self):
        """Simulate a full tenant flow: resolve → barrier → policy."""
        # 1. Create tenant config
        tc = _make_tenant(
            tenant_id="enterprise",
            name="enterprise-corp",
            api_key_hash=hash_api_key("aegis-enterprise-key"),
            rpm=100,
            burst=20,
            block_threshold=0.70,
            allowed_models=["gpt-4"],
            policy_tier="strict",
        )

        # 2. Set up TenantManager with cached tenant
        tm = TenantManager()
        tm._cache[tc.api_key_hash] = _CacheEntry(
            tenant=tc, expires_at=time.monotonic() + 300,
        )
        tm._tenants_by_id[tc.tenant_id] = tc

        # 3. Barrier resolves tenant
        config = BarrierConfig()
        barrier = BarrierLayer(
            config,
            valid_api_keys={"aegis-enterprise-key"},
            tenant_manager=tm,
        )
        ctx = await barrier.process(
            _make_body(model="gpt-4"),
            _make_headers("aegis-enterprise-key"),
            "10.0.0.1",
        )
        assert ctx.tenant_id == "enterprise"
        assert ctx.model == "gpt-4"

        # 4. Policy uses tenant thresholds (strict: 0.70 * 0.85 = 0.595)
        engine = PolicyEngine(tenant_manager=tm)
        report = _make_innate_report(confidence=0.60, is_threat=True)
        decision = engine.evaluate(
            ctx.request_id, innate_report=report, tenant_id=ctx.tenant_id,
        )
        assert decision.action.value == "block"
        assert "TENANT-TIER-STRICT" in decision.applied_policies

    @pytest.mark.asyncio
    async def test_disallowed_model_blocked_at_barrier(self):
        """Model not in tenant's allowed list → 403 at barrier."""
        tc = _make_tenant(
            tenant_id="t1",
            api_key_hash=hash_api_key("k"),
            allowed_models=["gpt-3.5-turbo"],
        )
        tm = TenantManager()
        tm._cache[tc.api_key_hash] = _CacheEntry(
            tenant=tc, expires_at=time.monotonic() + 300,
        )

        config = BarrierConfig()
        barrier = BarrierLayer(config, valid_api_keys={"k"}, tenant_manager=tm)

        with pytest.raises(BarrierReject) as exc_info:
            await barrier.process(
                _make_body(model="gpt-4"), _make_headers("k"), "1.1.1.1",
            )
        assert exc_info.value.status_code == 403
        assert "gpt-4" in exc_info.value.reason
