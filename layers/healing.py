"""
L7 — Self-Healing (Wound Healing / Tissue Repair)

Circuit breaker (Closed/Open/Half-Open) per model endpoint. Four automated
remediation capabilities: prompt sanitization, model fallback routing,
session quarantine, and model rollback. Recovery telemetry feeds the
adaptive policy engine and compliance dashboard.

Circuit breaker triggers: 50% error rate over 60s sliding window, or any
confirmed active exploit. Cooldown: 30s default, exponential backoff to
max 5 minutes. Probe: 5 requests in Half-Open state.

ASSUMED-BREACH POSTURE: This layer assumes the primary model endpoint is
actively compromised. Elevated error rates are not treated as transient
infrastructure issues — they are treated as potential indicators of model
compromise, data poisoning, or active attack. The healing layer does not
trust that the model will "recover on its own." Fallback models may also
be compromised (supply chain attack), so fallback responses receive the
same full-pipeline validation as primary responses. Session quarantine
assumes the adversary controls the session — quarantined sessions may be
routed to honeypots for intelligence collection.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from aegis.config import CircuitBreakerState, HealingConfig

logger = logging.getLogger(__name__)


@dataclass
class RecoveryEvent:
    """Telemetry record for a healing action."""
    timestamp: datetime
    action: str  # "circuit_open", "circuit_half_open", "circuit_closed", "fallback", "quarantine"
    trigger: str
    duration_ms: float = 0.0
    outcome: str = ""  # "success", "failure", "pending"
    details: dict = field(default_factory=dict)


@dataclass
class ProbeResult:
    """Result of a single probe request."""
    success: bool
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CircuitBreaker:
    """Circuit breaker state machine for a single model endpoint.

    States:
        CLOSED  — Normal operation. Errors tracked in sliding window.
        OPEN    — Tripped. All requests routed to fallback.
                  Cooldown before probing.
        HALF_OPEN — Probing primary with limited requests.
                    All pass -> CLOSED. Any fail -> OPEN with backoff.

    Trigger: error_rate >= threshold (50%) over window (60s),
             or force_open() for confirmed exploit.

    Cooldown: starts at 30s, exponential backoff (2x) on repeated failures,
              capped at max_cooldown (5 min).
    """

    def __init__(self, config: HealingConfig | None = None, endpoint: str = "primary"):
        self._config = config or HealingConfig()
        self._endpoint = endpoint
        self._state = CircuitBreakerState.CLOSED
        self._window: deque[tuple[float, bool]] = deque()  # (timestamp, is_failure)
        self._open_time: float = 0.0
        self._current_cooldown: float = self._config.cooldown_seconds
        self._probe_results: list[ProbeResult] = []
        self._consecutive_trips: int = 0
        self._recovery_events: list[RecoveryEvent] = []

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def current_cooldown(self) -> float:
        return self._current_cooldown

    @property
    def recovery_events(self) -> list[RecoveryEvent]:
        return list(self._recovery_events)

    @property
    def consecutive_trips(self) -> int:
        return self._consecutive_trips

    def record_success(self) -> None:
        """Record a successful request."""
        now = time.monotonic()
        self._window.append((now, False))
        self._prune_window(now)

    def record_failure(self) -> None:
        """Record a failed request. May trip the breaker."""
        now = time.monotonic()
        self._window.append((now, True))
        self._prune_window(now)

        if self._state == CircuitBreakerState.CLOSED:
            rate = self._failure_rate()
            if rate >= self._config.circuit_breaker_threshold:
                self._trip(f"Error rate {rate:.0%} >= threshold {self._config.circuit_breaker_threshold:.0%}")

        elif self._state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open -> back to open with backoff
            self._trip("Probe failure in HALF_OPEN state")

    def force_open(self, reason: str = "Confirmed active exploit") -> None:
        """Force the circuit breaker open (confirmed exploit)."""
        self._trip(reason)

    def should_allow_request(self) -> bool:
        """Determine if a request should be sent to primary or fallback.

        Returns True if the request should go to primary.
        Returns False if it should go to fallback.
        """
        if self._state == CircuitBreakerState.CLOSED:
            return True

        if self._state == CircuitBreakerState.OPEN:
            elapsed = time.monotonic() - self._open_time
            if elapsed >= self._current_cooldown:
                # Transition to half-open, allow probe
                self._transition_to_half_open()
                return True
            return False

        # HALF_OPEN: allow limited probes
        if self._state == CircuitBreakerState.HALF_OPEN:
            if len(self._probe_results) < self._config.probe_count:
                return True
            # All probes done — check results
            return False

        return True

    def record_probe_result(self, success: bool, latency_ms: float = 0.0) -> None:
        """Record the result of a probe request in HALF_OPEN state."""
        self._probe_results.append(ProbeResult(
            success=success,
            latency_ms=latency_ms,
        ))

        if not success:
            # Immediate re-trip with exponential backoff
            self._trip("Probe failure")
            return

        # Check if all probes passed
        if len(self._probe_results) >= self._config.probe_count:
            if all(p.success for p in self._probe_results):
                self._close()

    def get_fallback_endpoint(self) -> str | None:
        """Get the next fallback endpoint from the priority list."""
        if not self._config.fallback_models:
            return None
        # Simple round-robin for bootstrap; production uses health-aware selection
        return self._config.fallback_models[0] if self._config.fallback_models else None

    def _trip(self, reason: str) -> None:
        """Trip the circuit breaker to OPEN state."""
        old_state = self._state
        self._state = CircuitBreakerState.OPEN
        self._open_time = time.monotonic()
        self._probe_results = []
        self._consecutive_trips += 1

        # Exponential backoff on cooldown
        if self._consecutive_trips > 1:
            self._current_cooldown = min(
                self._config.cooldown_seconds * (2 ** (self._consecutive_trips - 1)),
                self._config.max_cooldown_seconds,
            )
        else:
            self._current_cooldown = self._config.cooldown_seconds

        event = RecoveryEvent(
            timestamp=datetime.now(timezone.utc),
            action="circuit_open",
            trigger=reason,
            details={
                "from_state": old_state.value,
                "cooldown_seconds": self._current_cooldown,
                "consecutive_trips": self._consecutive_trips,
            },
        )
        self._recovery_events.append(event)
        logger.warning(
            "Circuit breaker OPEN [%s]: %s (cooldown=%.0fs, trips=%d)",
            self._endpoint, reason, self._current_cooldown, self._consecutive_trips,
        )

    def _transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN for probing."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._probe_results = []
        event = RecoveryEvent(
            timestamp=datetime.now(timezone.utc),
            action="circuit_half_open",
            trigger="Cooldown elapsed",
            details={"probe_count": self._config.probe_count},
        )
        self._recovery_events.append(event)
        logger.info("Circuit breaker HALF_OPEN [%s]: probing primary", self._endpoint)

    def _close(self) -> None:
        """Transition to CLOSED state (healthy)."""
        self._state = CircuitBreakerState.CLOSED
        self._consecutive_trips = 0
        self._current_cooldown = self._config.cooldown_seconds
        self._probe_results = []
        event = RecoveryEvent(
            timestamp=datetime.now(timezone.utc),
            action="circuit_closed",
            trigger="All probes passed",
            outcome="success",
        )
        self._recovery_events.append(event)
        logger.info("Circuit breaker CLOSED [%s]: primary recovered", self._endpoint)

    def _prune_window(self, now: float) -> None:
        """Remove entries older than the sliding window."""
        cutoff = now - self._config.circuit_breaker_window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _failure_rate(self) -> float:
        """Calculate failure rate in the current window."""
        if not self._window:
            return 0.0
        failures = sum(1 for _, is_failure in self._window if is_failure)
        return failures / len(self._window)


class SessionQuarantine:
    """Tracks adversarial sessions and quarantines repeat offenders.

    After quarantine_threshold adversarial events in a session, the
    session is quarantined (maximum scrutiny or honeypot routing).
    """

    def __init__(self, config: HealingConfig | None = None):
        self._config = config or HealingConfig()
        self._session_strikes: dict[str, int] = {}
        self._quarantined: set[str] = set()
        # Source-level tracking (RT-009): tracks strikes by API key/IP
        self._source_strikes: dict[str, int] = {}
        self._quarantined_sources: set[str] = set()

    def record_adversarial_event(
        self, session_id: str, *, source_id: str | None = None,
    ) -> bool:
        """Record an adversarial event for a session.

        Args:
            session_id: Session identifier.
            source_id: Optional API key or IP for source-level tracking.

        Returns True if the session or source is now quarantined.
        """
        quarantined = False
        self._session_strikes[session_id] = self._session_strikes.get(session_id, 0) + 1
        if self._session_strikes[session_id] >= self._config.quarantine_threshold:
            self._quarantined.add(session_id)
            logger.warning("Session quarantined: %s (strikes=%d)",
                         session_id, self._session_strikes[session_id])
            quarantined = True

        # Source-level tracking: catches session rotation attacks
        if source_id:
            self._source_strikes[source_id] = self._source_strikes.get(source_id, 0) + 1
            if self._source_strikes[source_id] >= self._config.quarantine_threshold:
                self._quarantined_sources.add(source_id)
                logger.warning("Source quarantined: %s (strikes=%d)",
                             source_id, self._source_strikes[source_id])
                quarantined = True

        return quarantined

    def is_quarantined(self, session_id: str) -> bool:
        """Check if a session is quarantined."""
        return session_id in self._quarantined

    def is_source_quarantined(self, source_id: str) -> bool:
        """Check if a source (API key/IP) is quarantined."""
        return source_id in self._quarantined_sources

    @property
    def quarantined_sessions(self) -> set[str]:
        return set(self._quarantined)

    @property
    def quarantined_sources(self) -> set[str]:
        return set(self._quarantined_sources)

    def release(self, session_id: str) -> None:
        """Release a session from quarantine."""
        self._quarantined.discard(session_id)
        self._session_strikes.pop(session_id, None)

    def release_source(self, source_id: str) -> None:
        """Release a source from quarantine."""
        self._quarantined_sources.discard(source_id)
        self._source_strikes.pop(source_id, None)


class HealingLayer:
    """L7 Self-Healing — orchestrates circuit breakers and remediation.

    Manages per-endpoint circuit breakers, session quarantine, and
    recovery telemetry.
    """

    def __init__(self, config: HealingConfig | None = None):
        self._config = config or HealingConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._quarantine = SessionQuarantine(self._config)

    def get_breaker(self, endpoint: str = "primary") -> CircuitBreaker:
        """Get or create a circuit breaker for an endpoint.

        Uses setdefault() to avoid check-then-act race under concurrent access.
        """
        if endpoint in self._breakers:
            return self._breakers[endpoint]
        new_breaker = CircuitBreaker(self._config, endpoint)
        return self._breakers.setdefault(endpoint, new_breaker)

    @property
    def quarantine(self) -> SessionQuarantine:
        return self._quarantine

    def record_adversarial_event(
        self, session_id: str, *, source_id: str | None = None,
    ) -> bool:
        """Convenience: delegate to quarantine with source tracking."""
        return self._quarantine.record_adversarial_event(
            session_id, source_id=source_id,
        )

    def is_source_quarantined(self, source_id: str) -> bool:
        """Check if a source (API key/IP) is quarantined."""
        return self._quarantine.is_source_quarantined(source_id)

    def should_route_to_primary(self, endpoint: str = "primary") -> bool:
        """Check if requests should go to primary or fallback."""
        breaker = self.get_breaker(endpoint)
        return breaker.should_allow_request()

    def get_fallback(self, endpoint: str = "primary") -> str | None:
        """Get fallback endpoint when primary is unavailable."""
        return self.get_breaker(endpoint).get_fallback_endpoint()

    def all_recovery_events(self) -> list[RecoveryEvent]:
        """Collect all recovery events across all breakers."""
        events = []
        for breaker in self._breakers.values():
            events.extend(breaker.recovery_events)
        events.sort(key=lambda e: e.timestamp)
        return events
