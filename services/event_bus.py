"""
Event Bus — The immune system's cytokine signaling network.

Two implementations:
    - **RedisEventBus**: Redis Streams (XADD/XREAD with consumer groups).
      Durable, ordered, multi-consumer. Production-grade.
    - **InMemoryEventBus**: asyncio.Queue per channel. Zero dependencies.
      Used when Redis is unavailable or in tests.

Factory function ``create_event_bus()`` picks the right implementation
based on Redis availability.

Channels:
    - ``threat_detected``    — L3 publishes when a novel attack is caught
    - ``policy_escalation``  — L6 publishes when TLI changes
    - ``circuit_breaker``    — L7 publishes on state transitions
    - ``antibody_generated`` — L3 publishes when a new antibody is stored
    - ``audit_event``        — Cross-layer audit trail

ASSUMED-BREACH POSTURE: A compromised event bus could suppress threat
signals (dropping threat_detected events) or inject false signals
(fabricating circuit_breaker events to cause denial of service). Each
layer must maintain independent state — event bus notifications are
advisory, not authoritative. Layers that receive bus events validate
them against their own observations before acting.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Well-known channel names
CHANNEL_THREAT_DETECTED = "threat_detected"
CHANNEL_POLICY_ESCALATION = "policy_escalation"
CHANNEL_CIRCUIT_BREAKER = "circuit_breaker"
CHANNEL_ANTIBODY_GENERATED = "antibody_generated"
CHANNEL_AUDIT_EVENT = "audit_event"
CHANNEL_TOOL_VIOLATION = "tool_violation"

ALL_CHANNELS = [
    CHANNEL_THREAT_DETECTED,
    CHANNEL_POLICY_ESCALATION,
    CHANNEL_CIRCUIT_BREAKER,
    CHANNEL_ANTIBODY_GENERATED,
    CHANNEL_AUDIT_EVENT,
    CHANNEL_TOOL_VIOLATION,
]


@dataclass
class Event:
    """A single event on the bus."""
    channel: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "channel": self.channel,
            "timestamp": self.timestamp,
            "payload": json.dumps(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        payload = data.get("payload", "{}")
        if isinstance(payload, str):
            payload = json.loads(payload)
        return cls(
            channel=data.get("channel", ""),
            payload=payload,
            event_id=data.get("event_id", ""),
            timestamp=data.get("timestamp", ""),
        )


# Callback signature: async def handler(event: Event) -> None
EventCallback = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus(abc.ABC):
    """Abstract event bus interface."""

    @abc.abstractmethod
    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        """Publish an event to a channel."""

    @abc.abstractmethod
    async def subscribe(self, channel: str, callback: EventCallback) -> None:
        """Register an async callback for events on a channel."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Start consuming events (call after all subscriptions)."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop consuming events and clean up."""

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """Return backend name for health/diagnostics."""


class InMemoryEventBus(EventBus):
    """In-memory event bus using asyncio.Queue.

    Suitable for single-process development/testing. Events are lost on
    process restart. Each subscriber gets its own copy of every event.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._running = False
        self._published_count = 0

    @property
    def backend(self) -> str:
        return "memory"

    @property
    def published_count(self) -> int:
        return self._published_count

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        event = Event(channel=channel, payload=payload)
        self._published_count += 1
        callbacks = self._subscribers.get(channel, [])
        for cb in callbacks:
            try:
                await cb(event)
            except Exception as e:
                logger.error("Event callback error on %s: %s", channel, e)

    async def subscribe(self, channel: str, callback: EventCallback) -> None:
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False


class RedisEventBus(EventBus):
    """Redis Streams event bus using XADD/XREAD with consumer groups.

    Each channel maps to a Redis Stream key ``aegis:events:{channel}``.
    A single consumer group ``aegis-workers`` is created per stream.
    Each AEGIS instance has a unique consumer name.

    Events are trimmed to the last 10,000 entries per stream to prevent
    unbounded memory growth.
    """

    STREAM_PREFIX = "aegis:events:"
    GROUP_NAME = "aegis-workers"
    MAX_STREAM_LEN = 10_000

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._consumer_name = f"aegis-{uuid.uuid4().hex[:8]}"
        self._running = False
        self._reader_task: asyncio.Task | None = None

    @property
    def backend(self) -> str:
        return "redis"

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        event = Event(channel=channel, payload=payload)
        stream_key = f"{self.STREAM_PREFIX}{channel}"
        try:
            await self._redis.xadd(
                stream_key,
                event.to_dict(),
                maxlen=self.MAX_STREAM_LEN,
            )
        except Exception as e:
            logger.error("Redis XADD failed on %s: %s", channel, e)
            # Fire local callbacks as fallback
            for cb in self._subscribers.get(channel, []):
                try:
                    await cb(event)
                except Exception as cb_err:
                    logger.error("Fallback callback error: %s", cb_err)

    async def subscribe(self, channel: str, callback: EventCallback) -> None:
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)

    async def start(self) -> None:
        """Create consumer groups and start the reader loop."""
        for channel in self._subscribers:
            stream_key = f"{self.STREAM_PREFIX}{channel}"
            try:
                await self._redis.xgroup_create(
                    stream_key, self.GROUP_NAME, id="0", mkstream=True
                )
            except Exception:
                # Group may already exist — that's fine
                pass

        self._running = True
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        self._running = False
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    async def _read_loop(self) -> None:
        """Continuously read from subscribed streams."""
        streams = {
            f"{self.STREAM_PREFIX}{ch}": ">"
            for ch in self._subscribers
        }
        if not streams:
            return

        while self._running:
            try:
                results = await self._redis.xreadgroup(
                    self.GROUP_NAME,
                    self._consumer_name,
                    streams,
                    count=10,
                    block=1000,  # 1 second block
                )
                if not results:
                    continue

                for stream_key, messages in results:
                    channel = stream_key.replace(self.STREAM_PREFIX, "")
                    callbacks = self._subscribers.get(channel, [])
                    for msg_id, data in messages:
                        event = Event.from_dict(data)
                        for cb in callbacks:
                            try:
                                await cb(event)
                            except Exception as e:
                                logger.error(
                                    "Event callback error on %s: %s",
                                    channel, e,
                                )
                        # Acknowledge the message
                        try:
                            await self._redis.xack(
                                stream_key, self.GROUP_NAME, msg_id,
                            )
                        except Exception:
                            pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Redis XREAD error: %s", e)
                await asyncio.sleep(1)


async def create_event_bus(redis_client: Any | None = None) -> EventBus:
    """Factory: create the appropriate event bus implementation.

    If a Redis client is provided and reachable, returns RedisEventBus.
    Otherwise returns InMemoryEventBus.
    """
    if redis_client is not None:
        try:
            await redis_client.ping()
            logger.info("Event bus: Redis Streams")
            return RedisEventBus(redis_client)
        except Exception as e:
            logger.warning("Redis not reachable for event bus (%s) — using in-memory", e)

    logger.info("Event bus: in-memory (asyncio.Queue)")
    return InMemoryEventBus()
