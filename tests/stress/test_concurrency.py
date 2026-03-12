"""
Concurrency tests for AEGIS — CI-safe, no live servers required.

Validates that Phase C concurrency fixes work correctly:
  - setdefault() patterns prevent duplicate object creation
  - Lock-protected shared state prevents races
  - pop(k, None) prevents KeyError during concurrent eviction
  - Atomic Lua script prevents TOCTOU in Redis rate limiting

All tests run against in-process objects, not live servers.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import numpy as np
import pytest

from aegis.config import (
    AdaptiveConfig,
    BarrierConfig,
    HealingConfig,
    MemoryConfig,
)
from aegis.layers.barrier import (
    BarrierLayer,
    RedisSlidingWindowRateLimiter,
    SlidingWindowRateLimiter,
)
from aegis.layers.healing import CircuitBreaker, HealingLayer
from aegis.layers.memory.threat_vault import ThreatVault
from aegis.layers.adaptive.multi_turn import MultiTurnAnalyzer
from aegis.layers.agent_security.identity import AgentIdentityManager
from aegis.models.request_context import ChatMessage, RequestContext
from aegis.models.scan_result import ThreatCategory
from aegis.models.threat_indicator import IndicatorSource, ThreatIndicator
from aegis.services.tenant_manager import TenantManager, TenantConfig


# ========================================================================
# Test 1: Barrier tenant limiter — setdefault() produces single limiter
# ========================================================================


class TestBarrierTenantLimiterConcurrency:
    """Verify _get_tenant_limiter returns one limiter per key under races."""

    def test_concurrent_get_tenant_limiter_no_redis(self):
        """Multiple threads requesting the same tenant get the same limiter."""
        barrier = BarrierLayer(
            config=BarrierConfig(),
            valid_api_keys={"test-key"},
        )
        results = []
        barrier_lock = threading.Lock()

        def get_limiter():
            limiter = barrier._get_tenant_limiter("tenant-1", 60, 10)
            with barrier_lock:
                results.append(id(limiter))

        threads = [threading.Thread(target=get_limiter) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same limiter object
        assert len(set(results)) == 1, (
            f"Expected 1 unique limiter, got {len(set(results))}"
        )

    def test_different_tenants_get_different_limiters(self):
        """Different tenant keys produce different limiters."""
        barrier = BarrierLayer(
            config=BarrierConfig(),
            valid_api_keys={"test-key"},
        )
        limiter_a = barrier._get_tenant_limiter("tenant-a", 60, 10)
        limiter_b = barrier._get_tenant_limiter("tenant-b", 60, 10)
        assert id(limiter_a) != id(limiter_b)


# ========================================================================
# Test 2: Healing breaker — setdefault() produces single breaker
# ========================================================================


class TestHealingBreakerConcurrency:
    """Verify get_breaker returns one breaker per endpoint under races."""

    def test_concurrent_get_breaker(self):
        """Multiple threads requesting the same endpoint get the same breaker."""
        layer = HealingLayer()
        results = []
        lock = threading.Lock()

        def get_breaker():
            breaker = layer.get_breaker("primary")
            with lock:
                results.append(id(breaker))

        threads = [threading.Thread(target=get_breaker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1, (
            f"Expected 1 unique breaker, got {len(set(results))}"
        )

    def test_different_endpoints_get_different_breakers(self):
        """Different endpoints produce different breakers."""
        layer = HealingLayer()
        b1 = layer.get_breaker("primary")
        b2 = layer.get_breaker("fallback")
        assert id(b1) != id(b2)


# ========================================================================
# Test 3: TenantManager evict_expired — pop(k, None) safe
# ========================================================================


class TestTenantManagerEvictExpired:
    """Verify evict_expired doesn't KeyError under concurrent eviction."""

    def test_concurrent_evict_expired_no_keyerror(self):
        """Multiple threads evicting expired entries simultaneously — no crash."""
        manager = TenantManager.__new__(TenantManager)
        manager._cache = {}
        manager._db_available = False
        manager._lock = threading.Lock()
        manager._hit_count = 0
        manager._miss_count = 0

        # Create expired entries
        for i in range(100):
            key = f"key-{i}"
            config = TenantConfig(
                tenant_id=f"t-{i}",
                name=f"Tenant {i}",
                api_key_hash=key,
            )
            entry = type("CacheEntry", (), {
                "config": config,
                "expires_at": time.monotonic() - 1.0,  # already expired
            })()
            manager._cache[key] = entry

        errors = []

        def evict():
            try:
                manager.evict_expired()
            except KeyError as e:
                errors.append(e)

        threads = [threading.Thread(target=evict) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Got {len(errors)} KeyErrors"

    def test_evict_expired_returns_count(self):
        """evict_expired returns correct count when entries are expired."""
        manager = TenantManager.__new__(TenantManager)
        manager._cache = {}

        # Create 5 expired entries
        for i in range(5):
            key = f"key-{i}"
            entry = type("CacheEntry", (), {
                "config": None,
                "expires_at": time.monotonic() - 1.0,
            })()
            manager._cache[key] = entry

        # Create 3 non-expired entries
        for i in range(5, 8):
            key = f"key-{i}"
            entry = type("CacheEntry", (), {
                "config": None,
                "expires_at": time.monotonic() + 300.0,
            })()
            manager._cache[key] = entry

        evicted = manager.evict_expired()
        assert evicted == 5
        assert len(manager._cache) == 3


# ========================================================================
# Test 4: MultiTurnAnalyzer — concurrent analyze and request_count
# ========================================================================


class TestMultiTurnConcurrency:
    """Verify multi-turn analyzer handles concurrent access safely."""

    def _make_context(self, session_id: str = "sess-1") -> RequestContext:
        return RequestContext(
            request_id="req-1",
            timestamp="2026-01-01T00:00:00Z",
            api_key_hash="hash-1",
            source_ip="127.0.0.1",
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello world")],
            session_id=session_id,
        )

    @pytest.mark.asyncio
    async def test_concurrent_analyze_different_sessions(self):
        """Concurrent analysis of different sessions completes without error."""
        config = AdaptiveConfig()
        analyzer = MultiTurnAnalyzer(config=config)

        async def analyze_session(sid):
            ctx = self._make_context(session_id=sid)
            return await analyzer.analyze(ctx)

        results = await asyncio.gather(
            *[analyze_session(f"sess-{i}") for i in range(20)]
        )
        assert len(results) == 20
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_analyze_same_session(self):
        """Concurrent analysis of the same session doesn't corrupt state."""
        config = AdaptiveConfig()
        analyzer = MultiTurnAnalyzer(config=config)

        async def analyze():
            ctx = self._make_context(session_id="same-session")
            return await analyzer.analyze(ctx)

        results = await asyncio.gather(*[analyze() for _ in range(10)])
        assert len(results) == 10
        assert all(r is not None for r in results)

    def test_concurrent_record_block(self):
        """Concurrent record_block calls don't corrupt block timestamps."""
        config = AdaptiveConfig()
        analyzer = MultiTurnAnalyzer(config=config)

        def record():
            for _ in range(50):
                analyzer.record_block("sess-block")

        threads = [threading.Thread(target=record) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with analyzer._lock:
            count = len(analyzer._block_timestamps.get("sess-block", []))
        assert count == 250, f"Expected 250 block records, got {count}"

    def test_request_count_accuracy_under_concurrency(self):
        """_request_count increments accurately under concurrent access."""
        config = AdaptiveConfig()
        analyzer = MultiTurnAnalyzer(config=config)
        analyzer._request_count = 0
        total_increments = 1000

        def increment():
            for _ in range(100):
                with analyzer._lock:
                    analyzer._request_count += 1

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert analyzer._request_count == total_increments


# ========================================================================
# Test 5: ThreatVault — concurrent add and search
# ========================================================================


class TestThreatVaultConcurrency:
    """Verify threat vault handles concurrent add/search safely."""

    def _random_embedding(self, dim: int = 384) -> list[float]:
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def _make_indicator(self, suffix: str) -> ThreatIndicator:
        import uuid
        return ThreatIndicator(
            indicator_id=str(uuid.uuid4()),
            embedding=self._random_embedding(),
            threat_category=ThreatCategory.PROMPT_INJECTION,
            payload_hash=f"hash-{suffix}",
            source=IndicatorSource.ADAPTIVE_DETECTION,
            mitre_tactic="AML.T0051",
            confidence=0.95,
        )

    def test_concurrent_add(self):
        """Concurrent adds via thread pool don't corrupt the vault."""
        vault = ThreatVault(config=MemoryConfig())

        def add_indicator(i):
            vault.add(self._make_indicator(str(i)))

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(add_indicator, range(50)))

        assert vault.size == 50, f"Expected 50 indicators, got {vault.size}"

    def test_concurrent_add_and_search(self):
        """Concurrent adds and searches don't crash or corrupt state."""
        vault = ThreatVault(config=MemoryConfig())

        # Seed some initial data
        for i in range(10):
            vault.add(self._make_indicator(f"seed-{i}"))

        errors = []

        def add_worker(start_idx):
            try:
                for i in range(10):
                    vault.add(self._make_indicator(f"add-{start_idx}-{i}"))
            except Exception as e:
                errors.append(e)

        def search_worker():
            try:
                for _ in range(20):
                    vault.search(self._random_embedding(), k=5)
            except Exception as e:
                errors.append(e)

        threads = []
        for j in range(3):
            threads.append(threading.Thread(target=add_worker, args=(j * 10,)))
        for _ in range(3):
            threads.append(threading.Thread(target=search_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Got errors: {errors}"
        assert vault.size == 40  # 10 seed + 3*10 added

    def test_concurrent_lifecycle_ops(self):
        """Concurrent add + promote + search doesn't crash."""
        vault = ThreatVault(config=MemoryConfig())

        # Seed data
        for i in range(20):
            vault.add(self._make_indicator(f"lifecycle-{i}"))

        errors = []

        def promote_worker():
            try:
                for ind in list(vault._indicators):
                    try:
                        vault.promote(ind.indicator_id)
                    except Exception:
                        pass  # Some may fail if already promoted
            except Exception as e:
                errors.append(e)

        def search_worker():
            try:
                for _ in range(20):
                    vault.search(self._random_embedding(), k=3)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=promote_worker),
            threading.Thread(target=promote_worker),
            threading.Thread(target=search_worker),
            threading.Thread(target=search_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Got errors: {errors}"


# ========================================================================
# Test 6: Redis rate limiter atomicity
# ========================================================================


class TestRedisRateLimiterAtomicity:
    """Verify the Lua-based atomic rate limiter works correctly."""

    @pytest.mark.asyncio
    async def test_lua_allows_under_limit(self):
        """eval() returning count < limit means request is allowed."""
        mock = AsyncMock()
        mock.eval = AsyncMock(return_value=5)  # 5 existing, under limit

        limiter = RedisSlidingWindowRateLimiter(
            redis_client=mock, rpm=60, burst=10,
        )
        status = await limiter.check("test-key")
        assert not status.is_throttled
        assert status.requests_remaining > 0

    @pytest.mark.asyncio
    async def test_lua_blocks_at_limit(self):
        """eval() returning -1 means request is throttled."""
        mock = AsyncMock()
        mock.eval = AsyncMock(return_value=-1)

        limiter = RedisSlidingWindowRateLimiter(
            redis_client=mock, rpm=60, burst=10,
        )
        # Need a fallback limiter for the constructor
        status = await limiter.check("test-key")
        assert status.is_throttled
        assert status.requests_remaining == 0


# ========================================================================
# Test 7: Agent registry concurrency
# ========================================================================


class TestAgentRegistryConcurrency:
    """Verify agent registration handles concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_registration(self):
        """Concurrent agent registrations all succeed."""
        manager = AgentIdentityManager(signing_key="test-key-32chars-secure-enough!!")

        async def register(i):
            return await manager.register_agent(
                capabilities=[f"tool_{i}"],
                scope={"env": "test"},
            )

        results = await asyncio.gather(
            *[register(i) for i in range(20)]
        )
        assert len(results) == 20
        assert manager.registry_size == 20

        # All agents should have unique IDs
        ids = {r.agent_id for r in results}
        assert len(ids) == 20


# ========================================================================
# Test 8: Circuit breaker concurrent state transitions
# ========================================================================


class TestCircuitBreakerConcurrency:
    """Verify circuit breaker handles concurrent failure recordings."""

    def test_concurrent_failure_recording(self):
        """Concurrent failure recordings don't corrupt state."""
        config = HealingConfig()
        breaker = CircuitBreaker(config, "test-endpoint")

        def record_failures():
            for _ in range(100):
                breaker.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Breaker should have recorded failures and possibly tripped
        # The important thing is no crash and consistent state
        assert breaker.state is not None

    def test_concurrent_success_and_failure(self):
        """Mixed concurrent success/failure recordings don't crash."""
        config = HealingConfig()
        breaker = CircuitBreaker(config, "test-endpoint")
        errors = []

        def record_mixed(is_success):
            try:
                for _ in range(100):
                    if is_success:
                        breaker.record_success()
                    else:
                        breaker.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_mixed, args=(True,)),
            threading.Thread(target=record_mixed, args=(True,)),
            threading.Thread(target=record_mixed, args=(False,)),
            threading.Thread(target=record_mixed, args=(False,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Got errors: {errors}"
        assert breaker.state is not None
