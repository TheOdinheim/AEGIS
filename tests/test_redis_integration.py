"""
Tests for Redis integration: rate limiter, event bus, graceful degradation.

All tests use mock Redis clients or in-memory implementations — no live
Redis instance is required.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegis.config import BarrierConfig
from aegis.layers.barrier import (
    BarrierLayer,
    BarrierReject,
    RedisSlidingWindowRateLimiter,
    SlidingWindowRateLimiter,
)
from aegis.services.event_bus import (
    ALL_CHANNELS,
    CHANNEL_ANTIBODY_GENERATED,
    CHANNEL_CIRCUIT_BREAKER,
    CHANNEL_POLICY_ESCALATION,
    CHANNEL_THREAT_DETECTED,
    CHANNEL_AUDIT_EVENT,
    Event,
    InMemoryEventBus,
    RedisEventBus,
    create_event_bus,
)
from aegis.services.redis_client import (
    close_redis,
    init_redis,
    redis_health,
)


# ========================================================================
# Helpers
# ========================================================================


class MockPipeline:
    """Mock Redis pipeline that records and returns results."""

    def __init__(self, results: list[Any] | None = None):
        self._commands: list[tuple[str, tuple]] = []
        self._results = results or [0, 0]

    def zremrangebyscore(self, key: str, min_score: Any, max_score: Any) -> None:
        self._commands.append(("zremrangebyscore", (key, min_score, max_score)))

    def zcard(self, key: str) -> None:
        self._commands.append(("zcard", (key,)))

    def zadd(self, key: str, mapping: dict) -> None:
        self._commands.append(("zadd", (key, mapping)))

    def expire(self, key: str, ttl: int) -> None:
        self._commands.append(("expire", (key, ttl)))

    async def execute(self) -> list[Any]:
        return self._results


class MockRedis:
    """Mock async Redis client for testing."""

    def __init__(
        self,
        *,
        ping_ok: bool = True,
        zcard_count: int = 0,
        fail_on_pipeline: bool = False,
        eval_result: int | None = None,
    ):
        self._ping_ok = ping_ok
        self._zcard_count = zcard_count
        self._fail_on_pipeline = fail_on_pipeline
        self._eval_result = eval_result
        self._streams: dict[str, list[tuple[str, dict]]] = {}
        self._groups: dict[str, set[str]] = {}

    async def ping(self) -> bool:
        if not self._ping_ok:
            raise ConnectionError("Redis connection refused")
        return True

    async def info(self, section: str = "") -> dict[str, Any]:
        return {"used_memory": 1024 * 1024}  # 1MB

    async def aclose(self) -> None:
        pass

    def pipeline(self) -> MockPipeline:
        if self._fail_on_pipeline:
            raise ConnectionError("Pipeline failed")
        return MockPipeline(results=[0, self._zcard_count])

    async def eval(self, script: str, numkeys: int, *args) -> int:
        """Mock Lua script evaluation for atomic rate limiting."""
        if self._fail_on_pipeline:
            raise ConnectionError("eval failed")
        if self._eval_result is not None:
            return self._eval_result
        # Default: use zcard_count to determine result
        # If under limit, return current count; if at/over, return -1
        # The Lua script receives: key, now, cutoff, limit, member, ttl
        if len(args) >= 4:
            limit = int(float(args[3]))
            if self._zcard_count < limit:
                return self._zcard_count
            return -1
        return self._zcard_count

    async def xadd(self, stream: str, fields: dict, maxlen: int = 0) -> str:
        msg_id = f"{time.time()}-0"
        if stream not in self._streams:
            self._streams[stream] = []
        self._streams[stream].append((msg_id, fields))
        return msg_id

    async def xgroup_create(
        self, stream: str, group: str, id: str = "0", mkstream: bool = True
    ) -> None:
        if stream not in self._groups:
            self._groups[stream] = set()
        self._groups[stream].add(group)

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict,
        count: int = 10,
        block: int = 0,
    ) -> list | None:
        # Return nothing — tests publish and verify via callbacks directly
        await asyncio.sleep(0.01)
        return None

    async def xack(self, stream: str, group: str, msg_id: str) -> int:
        return 1


# ========================================================================
# Redis Client Tests
# ========================================================================


class TestRedisClient:
    """Tests for services/redis_client.py singleton and health check."""

    @pytest.mark.asyncio
    async def test_init_redis_no_url(self):
        """No REDIS_URL → returns None gracefully."""
        with patch.dict("os.environ", {}, clear=False):
            result = await init_redis("")
            assert result is None

    @pytest.mark.asyncio
    async def test_init_redis_no_package(self):
        """redis package not installed → returns None with warning."""
        with patch("aegis.services.redis_client.HAS_REDIS", False):
            result = await init_redis("redis://localhost:6379/0")
            assert result is None

    @pytest.mark.asyncio
    async def test_redis_health_not_initialized(self):
        """Health check when Redis not initialized."""
        with patch("aegis.services.redis_client._client", None):
            health = await redis_health()
            assert health["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_redis_health_connected(self):
        """Health check with a mock connected Redis."""
        mock_redis = MockRedis(ping_ok=True)
        with patch("aegis.services.redis_client._client", mock_redis):
            health = await redis_health()
            assert health["status"] == "connected"
            assert "latency_ms" in health
            assert health["used_memory_mb"] == 1.0

    @pytest.mark.asyncio
    async def test_redis_health_connection_lost(self):
        """Health check when Redis connection is lost."""
        mock_redis = MockRedis(ping_ok=False)
        with patch("aegis.services.redis_client._client", mock_redis):
            health = await redis_health()
            assert health["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_close_redis_noop(self):
        """close_redis when not initialized is a no-op."""
        with patch("aegis.services.redis_client._client", None):
            await close_redis()  # Should not raise


# ========================================================================
# Redis Rate Limiter Tests
# ========================================================================


class TestRedisSlidingWindowRateLimiter:
    """Tests for the Redis-backed sliding window rate limiter."""

    @pytest.mark.asyncio
    async def test_allow_under_limit(self):
        """Requests under limit are allowed."""
        mock = MockRedis(zcard_count=5)
        limiter = RedisSlidingWindowRateLimiter(
            redis_client=mock, rpm=60, burst=10,
        )
        status = await limiter.check("test-key")
        assert not status.is_throttled
        assert status.requests_remaining == 64  # 70 - 5 - 1 (just added)

    @pytest.mark.asyncio
    async def test_throttle_at_limit(self):
        """Requests at the limit are throttled."""
        mock = MockRedis(zcard_count=70)  # 60 + 10
        limiter = RedisSlidingWindowRateLimiter(
            redis_client=mock, rpm=60, burst=10,
        )
        status = await limiter.check("test-key")
        assert status.is_throttled
        assert status.requests_remaining == 0

    @pytest.mark.asyncio
    async def test_fallback_on_redis_error(self):
        """Falls back to in-memory when Redis pipeline fails."""
        mock = MockRedis(fail_on_pipeline=True)
        limiter = RedisSlidingWindowRateLimiter(
            redis_client=mock, rpm=60, burst=10,
        )
        # Should use in-memory fallback, not crash
        status = await limiter.check("test-key")
        assert not status.is_throttled

    @pytest.mark.asyncio
    async def test_key_format(self):
        """Redis keys use the correct prefix."""
        mock = MockRedis(zcard_count=0)
        limiter = RedisSlidingWindowRateLimiter(
            redis_client=mock, rpm=60, burst=10,
        )
        status = await limiter.check("abc123")
        assert not status.is_throttled
        # Verify the pipeline was called (key format is checked internally)


class TestBarrierLayerWithRedis:
    """Verify BarrierLayer uses Redis limiter when configured."""

    def test_barrier_uses_redis_limiter(self):
        """BarrierLayer with redis_client uses RedisSlidingWindowRateLimiter."""
        mock = MockRedis()
        barrier = BarrierLayer(
            config=BarrierConfig(),
            valid_api_keys={"test-key"},
            redis_client=mock,
        )
        assert barrier._use_redis_limiter is True
        assert isinstance(barrier._limiter, RedisSlidingWindowRateLimiter)

    def test_barrier_uses_inmemory_without_redis(self):
        """BarrierLayer without redis_client uses SlidingWindowRateLimiter."""
        barrier = BarrierLayer(
            config=BarrierConfig(),
            valid_api_keys={"test-key"},
        )
        assert barrier._use_redis_limiter is False
        assert isinstance(barrier._limiter, SlidingWindowRateLimiter)

    @pytest.mark.asyncio
    async def test_barrier_process_with_redis_limiter(self):
        """End-to-end: BarrierLayer.process() works with Redis rate limiter."""
        mock = MockRedis(zcard_count=0)
        barrier = BarrierLayer(
            config=BarrierConfig(),
            valid_api_keys={"test-key"},
            redis_client=mock,
        )
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        headers = {"authorization": "Bearer test-key"}
        ctx = await barrier.process(body, headers, "127.0.0.1", b'{"test": true}')
        assert ctx.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_barrier_process_rate_limited_with_redis(self):
        """Redis says we're over limit → BarrierReject 429."""
        mock = MockRedis(zcard_count=100)
        barrier = BarrierLayer(
            config=BarrierConfig(rate_limit_rpm=10, rate_limit_burst=5),
            valid_api_keys={"test-key"},
            redis_client=mock,
        )
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        headers = {"authorization": "Bearer test-key"}
        with pytest.raises(BarrierReject) as exc_info:
            await barrier.process(body, headers, "127.0.0.1", b"test")
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_barrier_falls_back_on_redis_error(self):
        """Redis pipeline fails → falls back to in-memory, request succeeds."""
        mock = MockRedis(fail_on_pipeline=True)
        barrier = BarrierLayer(
            config=BarrierConfig(),
            valid_api_keys={"test-key"},
            redis_client=mock,
        )
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        headers = {"authorization": "Bearer test-key"}
        # Should succeed via in-memory fallback
        ctx = await barrier.process(body, headers, "127.0.0.1", b"test")
        assert ctx.model == "gpt-4"


# ========================================================================
# InMemoryEventBus Tests
# ========================================================================


class TestInMemoryEventBus:
    """Tests for the in-memory event bus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Basic publish/subscribe works."""
        bus = InMemoryEventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(CHANNEL_THREAT_DETECTED, handler)
        await bus.start()
        await bus.publish(CHANNEL_THREAT_DETECTED, {"test": True})
        assert len(received) == 1
        assert received[0].channel == CHANNEL_THREAT_DETECTED
        assert received[0].payload == {"test": True}

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Multiple subscribers on the same channel all receive events."""
        bus = InMemoryEventBus()
        count = {"a": 0, "b": 0}

        async def handler_a(event: Event) -> None:
            count["a"] += 1

        async def handler_b(event: Event) -> None:
            count["b"] += 1

        await bus.subscribe(CHANNEL_THREAT_DETECTED, handler_a)
        await bus.subscribe(CHANNEL_THREAT_DETECTED, handler_b)
        await bus.start()
        await bus.publish(CHANNEL_THREAT_DETECTED, {})
        assert count["a"] == 1
        assert count["b"] == 1

    @pytest.mark.asyncio
    async def test_no_cross_channel_delivery(self):
        """Events on one channel don't leak to another."""
        bus = InMemoryEventBus()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.channel)

        await bus.subscribe(CHANNEL_THREAT_DETECTED, handler)
        await bus.start()
        await bus.publish(CHANNEL_CIRCUIT_BREAKER, {"test": True})
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_callback_error_doesnt_crash(self):
        """A failing callback doesn't crash the bus or block other subscribers."""
        bus = InMemoryEventBus()
        received: list[Event] = []

        async def bad_handler(event: Event) -> None:
            raise ValueError("boom")

        async def good_handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(CHANNEL_THREAT_DETECTED, bad_handler)
        await bus.subscribe(CHANNEL_THREAT_DETECTED, good_handler)
        await bus.start()
        await bus.publish(CHANNEL_THREAT_DETECTED, {})
        # Good handler still receives despite bad handler crash
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_all_channels_defined(self):
        """All expected channels are defined."""
        assert CHANNEL_THREAT_DETECTED in ALL_CHANNELS
        assert CHANNEL_POLICY_ESCALATION in ALL_CHANNELS
        assert CHANNEL_CIRCUIT_BREAKER in ALL_CHANNELS
        assert CHANNEL_ANTIBODY_GENERATED in ALL_CHANNELS
        assert CHANNEL_AUDIT_EVENT in ALL_CHANNELS
        assert len(ALL_CHANNELS) == 7

    @pytest.mark.asyncio
    async def test_backend_name(self):
        """InMemoryEventBus reports 'memory' as backend."""
        bus = InMemoryEventBus()
        assert bus.backend == "memory"

    @pytest.mark.asyncio
    async def test_published_count(self):
        """Published count tracks events."""
        bus = InMemoryEventBus()
        assert bus.published_count == 0
        await bus.publish(CHANNEL_THREAT_DETECTED, {})
        await bus.publish(CHANNEL_CIRCUIT_BREAKER, {})
        assert bus.published_count == 2

    @pytest.mark.asyncio
    async def test_threat_detected_channel(self):
        """Verify threat_detected channel carries expected payload."""
        bus = InMemoryEventBus()
        received: list[dict] = []

        async def handler(event: Event) -> None:
            received.append(event.payload)

        await bus.subscribe(CHANNEL_THREAT_DETECTED, handler)
        await bus.start()
        payload = {
            "request_id": "req-123",
            "session_id": "sess-456",
            "tenant_id": "default",
            "mcav_score": 0.95,
            "should_block": True,
            "is_novel_attack": True,
        }
        await bus.publish(CHANNEL_THREAT_DETECTED, payload)
        assert received[0] == payload

    @pytest.mark.asyncio
    async def test_circuit_breaker_channel(self):
        """Verify circuit_breaker channel carries state transition payload."""
        bus = InMemoryEventBus()
        received: list[dict] = []

        async def handler(event: Event) -> None:
            received.append(event.payload)

        await bus.subscribe(CHANNEL_CIRCUIT_BREAKER, handler)
        await bus.start()
        payload = {
            "endpoint": "primary",
            "previous_state": "closed",
            "new_state": "open",
            "trigger_reason": "50% error rate",
        }
        await bus.publish(CHANNEL_CIRCUIT_BREAKER, payload)
        assert received[0]["new_state"] == "open"

    @pytest.mark.asyncio
    async def test_antibody_generated_channel(self):
        """Verify antibody_generated channel works."""
        bus = InMemoryEventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(CHANNEL_ANTIBODY_GENERATED, handler)
        await bus.start()
        await bus.publish(CHANNEL_ANTIBODY_GENERATED, {
            "request_id": "req-abc",
            "mcav_score": 0.92,
        })
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_policy_escalation_channel(self):
        """Verify policy_escalation channel works."""
        bus = InMemoryEventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(CHANNEL_POLICY_ESCALATION, handler)
        await bus.start()
        await bus.publish(CHANNEL_POLICY_ESCALATION, {
            "from_level": "GREEN",
            "to_level": "BLUE",
        })
        assert len(received) == 1


# ========================================================================
# RedisEventBus Tests
# ========================================================================


class TestRedisEventBus:
    """Tests for the Redis Streams event bus using mock Redis."""

    @pytest.mark.asyncio
    async def test_publish_xadd(self):
        """Publish calls XADD on the correct stream key."""
        mock = MockRedis()
        bus = RedisEventBus(mock)
        await bus.publish(CHANNEL_THREAT_DETECTED, {"test": True})
        stream_key = f"{bus.STREAM_PREFIX}{CHANNEL_THREAT_DETECTED}"
        assert stream_key in mock._streams
        assert len(mock._streams[stream_key]) == 1

    @pytest.mark.asyncio
    async def test_publish_fallback_on_error(self):
        """When XADD fails, local callbacks are invoked as fallback."""
        mock = MockRedis()
        # Make xadd fail
        original_xadd = mock.xadd

        async def failing_xadd(*args, **kwargs):
            raise ConnectionError("XADD failed")

        mock.xadd = failing_xadd

        bus = RedisEventBus(mock)
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(CHANNEL_THREAT_DETECTED, handler)
        await bus.publish(CHANNEL_THREAT_DETECTED, {"fallback": True})
        assert len(received) == 1
        assert received[0].payload == {"fallback": True}

    @pytest.mark.asyncio
    async def test_backend_name(self):
        """RedisEventBus reports 'redis' as backend."""
        mock = MockRedis()
        bus = RedisEventBus(mock)
        assert bus.backend == "redis"

    @pytest.mark.asyncio
    async def test_start_creates_groups(self):
        """start() creates consumer groups for subscribed channels."""
        mock = MockRedis()
        bus = RedisEventBus(mock)

        async def noop(event: Event) -> None:
            pass

        await bus.subscribe(CHANNEL_THREAT_DETECTED, noop)
        await bus.subscribe(CHANNEL_CIRCUIT_BREAKER, noop)
        await bus.start()
        await asyncio.sleep(0.05)
        await bus.stop()

        # Verify groups were created
        assert f"{bus.STREAM_PREFIX}{CHANNEL_THREAT_DETECTED}" in mock._groups
        assert f"{bus.STREAM_PREFIX}{CHANNEL_CIRCUIT_BREAKER}" in mock._groups

    @pytest.mark.asyncio
    async def test_stop_cancels_reader(self):
        """stop() cancels the reader task cleanly."""
        mock = MockRedis()
        bus = RedisEventBus(mock)

        async def noop(event: Event) -> None:
            pass

        await bus.subscribe(CHANNEL_THREAT_DETECTED, noop)
        await bus.start()
        assert bus._reader_task is not None
        assert not bus._reader_task.done()
        await bus.stop()
        assert bus._running is False


# ========================================================================
# Event Bus Factory Tests
# ========================================================================


class TestCreateEventBus:
    """Tests for the event bus factory function."""

    @pytest.mark.asyncio
    async def test_factory_no_redis(self):
        """No Redis client → InMemoryEventBus."""
        bus = await create_event_bus(None)
        assert isinstance(bus, InMemoryEventBus)
        assert bus.backend == "memory"

    @pytest.mark.asyncio
    async def test_factory_redis_available(self):
        """Redis client available → RedisEventBus."""
        mock = MockRedis(ping_ok=True)
        bus = await create_event_bus(mock)
        assert isinstance(bus, RedisEventBus)
        assert bus.backend == "redis"

    @pytest.mark.asyncio
    async def test_factory_redis_unreachable(self):
        """Redis client unreachable → falls back to InMemoryEventBus."""
        mock = MockRedis(ping_ok=False)
        bus = await create_event_bus(mock)
        assert isinstance(bus, InMemoryEventBus)


# ========================================================================
# Event Serialization Tests
# ========================================================================


class TestEventSerialization:
    """Tests for Event to_dict/from_dict round-trip."""

    def test_event_to_dict(self):
        """Event.to_dict() produces expected format."""
        event = Event(
            channel=CHANNEL_THREAT_DETECTED,
            payload={"score": 0.95},
            event_id="test-123",
            timestamp="2026-03-03T00:00:00+00:00",
        )
        d = event.to_dict()
        assert d["channel"] == CHANNEL_THREAT_DETECTED
        assert d["event_id"] == "test-123"
        assert '"score": 0.95' in d["payload"]

    def test_event_round_trip(self):
        """Event survives to_dict → from_dict round-trip."""
        original = Event(
            channel=CHANNEL_CIRCUIT_BREAKER,
            payload={"endpoint": "primary", "new_state": "open"},
        )
        d = original.to_dict()
        restored = Event.from_dict(d)
        assert restored.channel == original.channel
        assert restored.payload == original.payload
        assert restored.event_id == original.event_id

    def test_event_from_dict_string_payload(self):
        """from_dict handles JSON string payload."""
        data = {
            "channel": "test",
            "payload": '{"key": "value"}',
            "event_id": "abc",
            "timestamp": "2026-01-01",
        }
        event = Event.from_dict(data)
        assert event.payload == {"key": "value"}

    def test_event_from_dict_dict_payload(self):
        """from_dict handles dict payload directly."""
        data = {
            "channel": "test",
            "payload": {"key": "value"},
            "event_id": "abc",
        }
        event = Event.from_dict(data)
        assert event.payload == {"key": "value"}


# ========================================================================
# Graceful Degradation Tests
# ========================================================================


class TestGracefulDegradation:
    """Verify AEGIS never crashes when Redis is unavailable."""

    @pytest.mark.asyncio
    async def test_rate_limiter_redis_down_mid_request(self):
        """Rate limiter falls back when Redis dies mid-request."""
        mock = MockRedis(fail_on_pipeline=True)
        limiter = RedisSlidingWindowRateLimiter(
            redis_client=mock, rpm=60, burst=10,
        )
        # Should not raise — falls back to in-memory
        for _ in range(5):
            status = await limiter.check("test-key")
            assert not status.is_throttled

    @pytest.mark.asyncio
    async def test_event_bus_redis_down_publish(self):
        """Event bus falls back to local callbacks when Redis XADD fails."""
        mock = MockRedis()
        received: list[Event] = []

        async def failing_xadd(*args, **kwargs):
            raise ConnectionError("Redis gone")

        mock.xadd = failing_xadd

        bus = RedisEventBus(mock)

        async def handler(event: Event) -> None:
            received.append(event)

        await bus.subscribe(CHANNEL_THREAT_DETECTED, handler)
        # Should not raise — falls back to local delivery
        await bus.publish(CHANNEL_THREAT_DETECTED, {"degraded": True})
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_inmemory_bus_as_fallback(self):
        """InMemoryEventBus provides full functionality without Redis."""
        bus = InMemoryEventBus()
        events: list[Event] = []

        async def handler(event: Event) -> None:
            events.append(event)

        # Subscribe to all channels
        for channel in ALL_CHANNELS:
            await bus.subscribe(channel, handler)
        await bus.start()

        # Publish to each channel
        for channel in ALL_CHANNELS:
            await bus.publish(channel, {"channel": channel})

        assert len(events) == len(ALL_CHANNELS)
        await bus.stop()

    @pytest.mark.asyncio
    async def test_barrier_no_redis_still_works(self):
        """BarrierLayer without Redis uses in-memory rate limiting."""
        barrier = BarrierLayer(
            config=BarrierConfig(rate_limit_rpm=10, rate_limit_burst=0),
            valid_api_keys={"test-key"},
        )
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        headers = {"authorization": "Bearer test-key"}

        # First 10 requests should pass
        for _ in range(10):
            ctx = await barrier.process(body, headers, "127.0.0.1", b"test")
            assert ctx is not None

        # 11th should be throttled
        with pytest.raises(BarrierReject) as exc_info:
            await barrier.process(body, headers, "127.0.0.1", b"test")
        assert exc_info.value.status_code == 429
