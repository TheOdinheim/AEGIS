"""
Adaptive Rate Limiter — Extension 5.4

Dynamically adjusts per-source rate limits based on manipulation risk scores
from MTMD and Source Profiler. Wraps the existing rate limiter — does NOT
replace it.

Standard rate limiting is insufficient against autonomous jailbreak agents
because they operate within normal rate limits while executing multi-turn
manipulation. This component applies progressive slowdowns and cooling
periods based on cumulative behavioral evidence.

ASSUMED-BREACH POSTURE: An attacker who knows the adaptive rate limiter
exists may try to stay below manipulation detection thresholds. The
progressive cooling ensures that even partial detection accumulates cost.
Hard stops require 5 flagged manipulation attempts — an attacker who
avoids all 5 flags has already been significantly slowed by the cooling
periods on the ones that were detected.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveRateState:
    source_id: str
    base_rpm: int
    effective_rpm: int
    slowdown_factor: float = 1.0
    cooling_period_seconds: float = 0.0
    cooling_until: float = 0.0
    manipulation_attempts: int = 0
    hard_stopped: bool = False
    last_updated: float = 0.0


@dataclass
class RateLimitDecision:
    allowed: bool
    reason: str
    effective_rpm: int
    slowdown_factor: float
    cooling_remaining_seconds: float = 0.0
    hard_stopped: bool = False


# Cooling escalation schedule (seconds)
_COOLING_SCHEDULE = [5.0, 15.0, 45.0, 120.0]


class AdaptiveRateLimiter:
    """Dynamically adjusts rate limits based on manipulation risk.

    Wraps the existing SlidingWindowRateLimiter by modifying the effective
    RPM for each source based on MTMD/profiler risk signals. Does NOT
    replace or modify the underlying rate limiter.
    """

    def __init__(
        self,
        base_rpm: int = 60,
        hard_stop_threshold: int = 5,
        hard_stop_duration: int = 300,
        max_states: int = 10000,
    ):
        self._base_rpm = base_rpm
        self._hard_stop_threshold = hard_stop_threshold
        self._hard_stop_duration = hard_stop_duration
        self._max_states = max_states
        self._lock = threading.Lock()
        self._states: OrderedDict[str, AdaptiveRateState] = OrderedDict()

    def _get_or_create_state(self, source_id: str) -> AdaptiveRateState:
        """Get existing state or create new one with LRU eviction."""
        if source_id in self._states:
            self._states.move_to_end(source_id)
            return self._states[source_id]

        if len(self._states) >= self._max_states:
            self._states.popitem(last=False)

        state = AdaptiveRateState(
            source_id=source_id,
            base_rpm=self._base_rpm,
            effective_rpm=self._base_rpm,
            last_updated=time.time(),
        )
        self._states[source_id] = state
        return state

    @staticmethod
    def _slowdown_for_risk(risk_score: float) -> float:
        """Compute slowdown factor from risk score."""
        if risk_score >= 0.7:
            return 0.1
        elif risk_score >= 0.5:
            return 0.3
        elif risk_score >= 0.3:
            return 0.6
        return 1.0

    def check(
        self, source_id: str, risk_score: float, manipulation_flagged: bool,
    ) -> RateLimitDecision:
        """Check whether request should be allowed based on adaptive state."""
        now = time.time()
        with self._lock:
            state = self._get_or_create_state(source_id)

            # Check hard stop expiry
            if state.hard_stopped:
                if now >= state.cooling_until:
                    state.hard_stopped = False
                    state.manipulation_attempts = 0
                    state.cooling_until = 0.0
                    logger.info("Hard stop expired for source %s", source_id)
                else:
                    remaining = state.cooling_until - now
                    return RateLimitDecision(
                        allowed=False,
                        reason="hard_stop",
                        effective_rpm=0,
                        slowdown_factor=0.0,
                        cooling_remaining_seconds=remaining,
                        hard_stopped=True,
                    )

            # Update manipulation attempts
            if manipulation_flagged:
                state.manipulation_attempts += 1

                if state.manipulation_attempts >= self._hard_stop_threshold:
                    state.hard_stopped = True
                    state.cooling_until = now + self._hard_stop_duration
                    state.last_updated = now
                    return RateLimitDecision(
                        allowed=False,
                        reason="hard_stop",
                        effective_rpm=0,
                        slowdown_factor=0.0,
                        cooling_remaining_seconds=float(self._hard_stop_duration),
                        hard_stopped=True,
                    )

                # Apply progressive cooling
                idx = min(state.manipulation_attempts - 1, len(_COOLING_SCHEDULE) - 1)
                cooling = _COOLING_SCHEDULE[idx]
                state.cooling_period_seconds = cooling
                state.cooling_until = now + cooling

            # Check if in cooling period
            if now < state.cooling_until:
                remaining = state.cooling_until - now
                state.effective_rpm = 1
                state.slowdown_factor = 1.0 / state.base_rpm if state.base_rpm > 0 else 0.0
                state.last_updated = now
                return RateLimitDecision(
                    allowed=True,
                    reason="cooling",
                    effective_rpm=1,
                    slowdown_factor=state.slowdown_factor,
                    cooling_remaining_seconds=remaining,
                    hard_stopped=False,
                )

            # Apply risk-based slowdown
            factor = self._slowdown_for_risk(risk_score)
            state.slowdown_factor = factor
            state.effective_rpm = max(1, int(state.base_rpm * factor))
            state.last_updated = now

            return RateLimitDecision(
                allowed=True,
                reason="ok" if factor >= 1.0 else "slowdown",
                effective_rpm=state.effective_rpm,
                slowdown_factor=factor,
                cooling_remaining_seconds=0.0,
                hard_stopped=False,
            )

    def update_risk(
        self, source_id: str, risk_score: float, manipulation_flagged: bool,
    ) -> None:
        """Update source risk state after MTMD/profiler analysis."""
        with self._lock:
            state = self._get_or_create_state(source_id)
            factor = self._slowdown_for_risk(risk_score)
            state.slowdown_factor = factor
            state.effective_rpm = max(1, int(state.base_rpm * factor))
            if manipulation_flagged:
                state.manipulation_attempts += 1
            state.last_updated = time.time()

    def clear_hard_stop(self, source_id: str) -> bool:
        """Admin action: clear a hard stop for a source."""
        with self._lock:
            state = self._states.get(source_id)
            if state and state.hard_stopped:
                state.hard_stopped = False
                state.cooling_until = 0.0
                state.manipulation_attempts = 0
                state.last_updated = time.time()
                logger.info("Admin cleared hard stop for source %s", source_id)
                return True
            return False

    def get_state(self, source_id: str) -> AdaptiveRateState | None:
        """Get current adaptive rate state for a source."""
        with self._lock:
            return self._states.get(source_id)

    @property
    def state_count(self) -> int:
        with self._lock:
            return len(self._states)
