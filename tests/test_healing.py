"""
L7 Self-Healing Tests

Validates circuit breaker state machine (Closed -> Open -> Half-Open -> Closed),
exponential backoff cooldown (30s default, max 5 min), probe-based recovery
(5 probes), and fallback routing. Tests session quarantine trigger threshold.

Tests cover attack #20 from the 20-attack battery.
    #20 — Circuit Breaker Trip (sustained attack triggering failover)

Also tests:
    - Full state machine transitions
    - Exponential backoff on repeated failures
    - Probe-based recovery
    - Force-open for confirmed exploit
    - Session quarantine
    - Recovery telemetry
    - Fallback routing
    - HealingLayer orchestration

L6 Policy Engine tests are also included here:
    - Three-tier policy evaluation (Global, Tenant, Adaptive)
    - Five-level Threat Level Indicator (GREEN -> RED)
    - Dynamic threshold adjustment
    - Policy escalation
    - Corroboration boost
"""

from __future__ import annotations



import pytest

from aegis.config import (
    CircuitBreakerState,
    HealingConfig,
    PolicyConfig,
    ThreatLevel,
)
from aegis.layers.healing import (
    CircuitBreaker,
    HealingLayer,
    SessionQuarantine,
)
from aegis.layers.policy import PolicyEngine, TenantPolicy
from aegis.models.adaptive_result import AdaptiveAnalysisReport
from aegis.models.policy_decision import PolicyAction, PolicyTier
from aegis.models.scan_result import InnateScanReport, ScanResult, ThreatCategory


# =========================================================================
# Circuit Breaker State Machine Tests
# =========================================================================

class TestCircuitBreaker:
    @pytest.fixture
    def config(self) -> HealingConfig:
        return HealingConfig(
            circuit_breaker_threshold=0.50,
            circuit_breaker_window_seconds=60,
            cooldown_seconds=1,  # Short for testing
            max_cooldown_seconds=10,
            probe_count=3,  # Fewer probes for testing
            fallback_models=["fallback-model-1", "fallback-model-2"],
        )

    @pytest.fixture
    def breaker(self, config: HealingConfig) -> CircuitBreaker:
        return CircuitBreaker(config)

    def test_initial_state_closed(self, breaker: CircuitBreaker):
        """Circuit breaker starts in CLOSED state."""
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.should_allow_request()

    def test_attack_20_sustained_failures_trip(self, breaker: CircuitBreaker):
        """Attack #20: Sustained attack triggers circuit breaker trip."""
        # Record 50% failures (threshold is 50%)
        for _ in range(5):
            breaker.record_success()
        for _ in range(6):
            breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN
        assert not breaker.should_allow_request()

    def test_closed_to_open_transition(self, breaker: CircuitBreaker):
        """Failures exceeding threshold trip the breaker."""
        # 3 failures out of 4 = 75% > 50% threshold
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN

    def test_open_blocks_requests(self, breaker: CircuitBreaker):
        """OPEN state blocks requests."""
        breaker.force_open("test")
        assert not breaker.should_allow_request()

    def test_open_to_half_open_after_cooldown(self, config: HealingConfig):
        """After cooldown expires, transitions to HALF_OPEN."""
        config.cooldown_seconds = 0.1  # 100ms for testing
        breaker = CircuitBreaker(config)
        breaker.force_open("test")

        assert breaker.state == CircuitBreakerState.OPEN
        # Backdate _open_time to simulate cooldown expiry
        breaker._open_time -= 1.0

        allowed = breaker.should_allow_request()
        assert allowed
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_probe_success_closes(self, config: HealingConfig):
        """All probes passing in HALF_OPEN closes the breaker."""
        config.cooldown_seconds = 0.01
        breaker = CircuitBreaker(config)
        breaker.force_open("test")

        breaker._open_time -= 1.0  # Backdate to simulate cooldown expiry
        breaker.should_allow_request()  # Triggers HALF_OPEN
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # All 3 probes succeed
        for _ in range(config.probe_count):
            breaker.record_probe_result(success=True, latency_ms=5.0)

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.consecutive_trips == 0

    def test_half_open_probe_failure_reopens(self, config: HealingConfig):
        """Probe failure in HALF_OPEN reopens the breaker."""
        config.cooldown_seconds = 0.01
        breaker = CircuitBreaker(config)
        breaker.force_open("test")

        breaker._open_time -= 1.0  # Backdate to simulate cooldown expiry
        breaker.should_allow_request()
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        breaker.record_probe_result(success=False, latency_ms=5.0)
        assert breaker.state == CircuitBreakerState.OPEN

    def test_exponential_backoff(self, config: HealingConfig):
        """Consecutive trips increase cooldown exponentially."""
        config.cooldown_seconds = 1
        config.max_cooldown_seconds = 16
        breaker = CircuitBreaker(config)

        # First trip: cooldown = 1s
        breaker.force_open("trip 1")
        assert breaker.current_cooldown == 1

        # Second trip: cooldown = 2s
        breaker.force_open("trip 2")
        assert breaker.current_cooldown == 2

        # Third trip: cooldown = 4s
        breaker.force_open("trip 3")
        assert breaker.current_cooldown == 4

        # Fourth trip: cooldown = 8s
        breaker.force_open("trip 4")
        assert breaker.current_cooldown == 8

        # Fifth trip: cooldown = 16s (max)
        breaker.force_open("trip 5")
        assert breaker.current_cooldown == 16

        # Sixth trip: still capped at max
        breaker.force_open("trip 6")
        assert breaker.current_cooldown == 16

    def test_force_open_for_exploit(self, breaker: CircuitBreaker):
        """Confirmed exploit immediately opens the breaker."""
        breaker.force_open("Confirmed active exploit detected")
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.consecutive_trips == 1

    def test_recovery_telemetry(self, config: HealingConfig):
        """Recovery events should be logged for telemetry."""
        config.cooldown_seconds = 0.01
        breaker = CircuitBreaker(config)

        breaker.force_open("test exploit")
        assert len(breaker.recovery_events) == 1
        assert breaker.recovery_events[0].action == "circuit_open"

        breaker._open_time -= 1.0  # Backdate to simulate cooldown expiry
        breaker.should_allow_request()  # -> HALF_OPEN
        assert len(breaker.recovery_events) == 2
        assert breaker.recovery_events[1].action == "circuit_half_open"

        for _ in range(config.probe_count):
            breaker.record_probe_result(success=True)

        assert len(breaker.recovery_events) == 3
        assert breaker.recovery_events[2].action == "circuit_closed"

    def test_fallback_endpoint(self, breaker: CircuitBreaker):
        """Should return fallback model when configured."""
        fallback = breaker.get_fallback_endpoint()
        assert fallback == "fallback-model-1"

    def test_fallback_no_models(self):
        """No fallback configured returns None."""
        config = HealingConfig(fallback_models=[])
        breaker = CircuitBreaker(config)
        assert breaker.get_fallback_endpoint() is None

    def test_success_resets_after_recovery(self, config: HealingConfig):
        """After recovery, consecutive_trips resets to 0."""
        config.cooldown_seconds = 0.01
        breaker = CircuitBreaker(config)

        breaker.force_open("test")
        assert breaker.consecutive_trips == 1

        breaker._open_time -= 1.0  # Backdate to simulate cooldown expiry
        breaker.should_allow_request()
        for _ in range(config.probe_count):
            breaker.record_probe_result(success=True)

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.consecutive_trips == 0


# =========================================================================
# Session Quarantine Tests
# =========================================================================

class TestSessionQuarantine:
    def test_quarantine_after_threshold(self):
        """Session quarantined after threshold adversarial events."""
        config = HealingConfig(quarantine_threshold=3)
        quarantine = SessionQuarantine(config)

        assert not quarantine.record_adversarial_event("session-1")
        assert not quarantine.record_adversarial_event("session-1")
        assert quarantine.record_adversarial_event("session-1")  # 3rd = quarantine

        assert quarantine.is_quarantined("session-1")
        assert not quarantine.is_quarantined("session-2")

    def test_release_from_quarantine(self):
        """Released sessions are no longer quarantined."""
        config = HealingConfig(quarantine_threshold=1)
        quarantine = SessionQuarantine(config)
        quarantine.record_adversarial_event("session-1")
        assert quarantine.is_quarantined("session-1")

        quarantine.release("session-1")
        assert not quarantine.is_quarantined("session-1")

    def test_independent_sessions(self):
        """Different sessions tracked independently."""
        config = HealingConfig(quarantine_threshold=2)
        quarantine = SessionQuarantine(config)

        quarantine.record_adversarial_event("session-a")
        quarantine.record_adversarial_event("session-b")

        assert not quarantine.is_quarantined("session-a")
        assert not quarantine.is_quarantined("session-b")

        quarantine.record_adversarial_event("session-a")  # 2nd for a
        assert quarantine.is_quarantined("session-a")
        assert not quarantine.is_quarantined("session-b")


# =========================================================================
# Healing Layer Orchestrator Tests
# =========================================================================

class TestHealingLayer:
    def test_get_or_create_breaker(self):
        """Layer creates breakers on demand per endpoint."""
        layer = HealingLayer()
        b1 = layer.get_breaker("endpoint-1")
        b2 = layer.get_breaker("endpoint-2")
        assert b1 is not b2
        assert layer.get_breaker("endpoint-1") is b1  # Same instance

    def test_fallback_routing_when_open(self):
        """When primary is open, should route to fallback."""
        config = HealingConfig(fallback_models=["backup-model"])
        layer = HealingLayer(config)
        breaker = layer.get_breaker("primary")
        breaker.force_open("test")

        assert not layer.should_route_to_primary("primary")
        assert layer.get_fallback("primary") == "backup-model"

    def test_aggregate_recovery_events(self):
        """All recovery events collected across breakers."""
        layer = HealingLayer()
        layer.get_breaker("ep1").force_open("test1")
        layer.get_breaker("ep2").force_open("test2")

        events = layer.all_recovery_events()
        assert len(events) == 2

    def test_quarantine_integration(self):
        """Quarantine accessible through healing layer."""
        config = HealingConfig(quarantine_threshold=2)
        layer = HealingLayer(config)
        layer.quarantine.record_adversarial_event("sess-1")
        layer.quarantine.record_adversarial_event("sess-1")
        assert layer.quarantine.is_quarantined("sess-1")


# =========================================================================
# Policy Engine Tests
# =========================================================================

def _innate_report(
    max_confidence: float = 0.0,
    categories: list[ThreatCategory] | None = None,
) -> InnateScanReport:
    """Build a mock innate report."""
    results = []
    if max_confidence > 0:
        results.append(ScanResult(
            scanner_id="regex_engine",
            is_threat=True,
            confidence=max_confidence,
            threat_category=categories[0] if categories else ThreatCategory.PROMPT_INJECTION,
            latency_ms=0.1,
        ))
    return InnateScanReport(
        request_id="test",
        scanner_results=results,
        max_confidence=max_confidence,
        should_block=max_confidence >= 0.85,
        threat_categories=categories or [],
    )


def _adaptive_report(mcav: float = 0.0) -> AdaptiveAnalysisReport:
    """Build a mock adaptive report."""
    return AdaptiveAnalysisReport(
        request_id="test",
        mcav_score=mcav,
        should_block=mcav >= 0.9,
    )


class TestPolicyEngine:
    @pytest.fixture
    def engine(self) -> PolicyEngine:
        return PolicyEngine()

    def test_green_benign_allows(self, engine: PolicyEngine):
        """GREEN threat level, benign request -> ALLOW."""
        decision = engine.evaluate(
            "req-1",
            innate_report=_innate_report(0.0),
            adaptive_report=_adaptive_report(0.0),
        )
        assert decision.action == PolicyAction.ALLOW
        assert decision.threat_level == ThreatLevel.GREEN

    def test_global_hard_block_injection(self, engine: PolicyEngine):
        """Global policy blocks high-confidence prompt injection."""
        decision = engine.evaluate(
            "req-2",
            innate_report=_innate_report(
                0.95,
                [ThreatCategory.PROMPT_INJECTION]
            ),
        )
        assert decision.action == PolicyAction.BLOCK
        assert decision.triggered_by == PolicyTier.GLOBAL
        assert "GLOBAL" in decision.reasons[0]

    def test_global_hard_block_jailbreak(self, engine: PolicyEngine):
        """Global policy blocks jailbreak."""
        decision = engine.evaluate(
            "req-3",
            innate_report=_innate_report(
                0.90,
                [ThreatCategory.JAILBREAK]
            ),
        )
        assert decision.action == PolicyAction.BLOCK
        assert decision.triggered_by == PolicyTier.GLOBAL

    def test_red_blocks_everything(self, engine: PolicyEngine):
        """RED threat level blocks all requests."""
        engine.threat_level = ThreatLevel.RED
        decision = engine.evaluate(
            "req-4",
            innate_report=_innate_report(0.0),
            adaptive_report=_adaptive_report(0.0),
        )
        assert decision.action == PolicyAction.BLOCK
        assert "RED" in decision.reasons[0]
        assert "fail-closed" in decision.reasons[0].lower()

    def test_blue_lowers_thresholds(self, engine: PolicyEngine):
        """BLUE threat level lowers block thresholds by 10%."""
        engine.threat_level = ThreatLevel.BLUE

        # Score that would be below GREEN block threshold but above BLUE
        decision = engine.evaluate(
            "req-5",
            innate_report=_innate_report(0.0),
            adaptive_report=_adaptive_report(0.80),
        )
        # 0.85 * 0.9 = 0.765, MCAV 0.80 > 0.765 -> block
        assert decision.action == PolicyAction.BLOCK

    def test_green_same_score_allows(self, engine: PolicyEngine):
        """Same score that blocks in BLUE is allowed in GREEN."""
        engine.threat_level = ThreatLevel.GREEN

        decision = engine.evaluate(
            "req-6",
            innate_report=_innate_report(0.0),
            adaptive_report=_adaptive_report(0.80),
        )
        # 0.80 < 0.85 -> not blocked in GREEN
        assert decision.action != PolicyAction.BLOCK

    def test_yellow_escalates_to_human(self, engine: PolicyEngine):
        """YELLOW level with moderate threat escalates to human review."""
        engine.threat_level = ThreatLevel.YELLOW

        decision = engine.evaluate(
            "req-7",
            innate_report=_innate_report(0.0),
            adaptive_report=_adaptive_report(0.55),
        )
        # YELLOW escalate threshold = 0.60 * 0.75 = 0.45, score 0.55 > 0.45
        assert decision.action == PolicyAction.ESCALATE
        assert decision.output_scrutiny_level > 1.0

    def test_orange_maximum_sensitivity(self, engine: PolicyEngine):
        """ORANGE level blocks even moderate threats."""
        engine.threat_level = ThreatLevel.ORANGE

        decision = engine.evaluate(
            "req-8",
            innate_report=_innate_report(0.0),
            adaptive_report=_adaptive_report(0.55),
        )
        # ORANGE block threshold = 0.85 * 0.60 = 0.51, score 0.55 > 0.51
        assert decision.action == PolicyAction.BLOCK

    def test_tenant_policy_model_allowlist(self, engine: PolicyEngine):
        """Tenant policy blocks unauthorized models."""
        engine.register_tenant_policy(TenantPolicy(
            tenant_id="acme",
            allowed_models=["gpt-4", "gpt-4-turbo"],
        ))

        decision = engine.evaluate(
            "req-9",
            tenant_id="acme",
            model="gpt-3.5-turbo",
        )
        assert decision.action == PolicyAction.BLOCK
        assert decision.triggered_by == PolicyTier.TENANT
        assert "not authorized" in decision.block_message.lower() or "not in allowed" in decision.reasons[0].lower()

    def test_tenant_custom_threshold(self, engine: PolicyEngine):
        """Tenant-specific block threshold overrides adaptive."""
        engine.register_tenant_policy(TenantPolicy(
            tenant_id="strict-co",
            custom_block_threshold=0.50,
        ))

        decision = engine.evaluate(
            "req-10",
            innate_report=_innate_report(0.55),
            tenant_id="strict-co",
        )
        assert decision.action == PolicyAction.BLOCK
        assert decision.triggered_by == PolicyTier.TENANT

    def test_escalation_history_tracked(self, engine: PolicyEngine):
        """Threat level changes are recorded in escalation history."""
        engine.threat_level = ThreatLevel.BLUE
        engine.threat_level = ThreatLevel.YELLOW

        assert len(engine._escalation_history) == 2
        assert engine._escalation_history[0]["from"] == "GREEN"
        assert engine._escalation_history[0]["to"] == "BLUE"
        assert engine._escalation_history[1]["from"] == "BLUE"
        assert engine._escalation_history[1]["to"] == "YELLOW"

    def test_escalate_and_deescalate(self, engine: PolicyEngine):
        """Escalate and de-escalate threat level."""
        assert engine.threat_level == ThreatLevel.GREEN

        engine.escalate_threat_level()
        assert engine.threat_level == ThreatLevel.BLUE

        engine.escalate_threat_level()
        assert engine.threat_level == ThreatLevel.YELLOW

        engine.de_escalate_threat_level()
        assert engine.threat_level == ThreatLevel.BLUE

    def test_corroboration_boost(self, engine: PolicyEngine):
        """When both innate and adaptive detect threats, fused score should increase."""
        # Innate at 0.6, adaptive at 0.6 -> corroboration boost
        decision = engine.evaluate(
            "req-11",
            innate_report=_innate_report(0.60),
            adaptive_report=_adaptive_report(0.60),
        )
        # Fused = max(0.6, 0.6) + min(0.6, 0.6) * 0.2 = 0.6 + 0.12 = 0.72
        assert decision.fused_score > 0.70

    def test_allow_degraded_with_elevated_scrutiny(self, engine: PolicyEngine):
        """Moderate threat with GREEN level -> ALLOW_DEGRADED."""
        decision = engine.evaluate(
            "req-12",
            innate_report=_innate_report(0.0),
            adaptive_report=_adaptive_report(0.65),
        )
        # 0.65 >= 0.60 escalate threshold -> ALLOW_DEGRADED (GREEN, no human review)
        assert decision.action == PolicyAction.ALLOW_DEGRADED
        assert decision.output_scrutiny_level > 1.0

    def test_reasons_populated(self, engine: PolicyEngine):
        """Decision reasons should always be populated for audit."""
        decision = engine.evaluate("req-13")
        assert len(decision.reasons) > 0

    def test_applied_policies_populated(self, engine: PolicyEngine):
        """Applied policy IDs should be tracked."""
        decision = engine.evaluate("req-14")
        assert len(decision.applied_policies) > 0

    def test_max_escalation_capped(self, engine: PolicyEngine):
        """Cannot escalate past RED."""
        engine.threat_level = ThreatLevel.RED
        result = engine.escalate_threat_level()
        assert result == ThreatLevel.RED

    def test_min_deescalation_capped(self, engine: PolicyEngine):
        """Cannot de-escalate below GREEN."""
        engine.threat_level = ThreatLevel.GREEN
        result = engine.de_escalate_threat_level()
        assert result == ThreatLevel.GREEN
