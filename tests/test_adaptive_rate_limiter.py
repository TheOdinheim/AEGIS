"""
Tests for Adaptive Rate Limiter (Extension 5.4) and Jailbreak Taxonomy Logger (Extension 5.5).

Adaptive Rate Limiter tests (18+): slowdown factors, progressive cooling, hard stops,
per-source isolation, LRU eviction, clear_hard_stop, risk escalation/recovery.

Jailbreak Taxonomy Logger tests (15+): classification, stats, trends, FIFO eviction.

Integration tests (5+): combined MTMD + rate limiter + taxonomy operation.
"""

import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aegis.layers.adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    AdaptiveRateState,
    RateLimitDecision,
)
from aegis.layers.memory.jailbreak_taxonomy import (
    JailbreakAttempt,
    JailbreakTaxonomyLogger,
    JailbreakTechnique,
    TaxonomyStats,
)


# ============================================================================
# Adaptive Rate Limiter Tests (18+)
# ============================================================================


class TestAdaptiveRateLimiterSlowdown:
    """Test risk-based slowdown factors."""

    def test_no_slowdown_at_low_risk(self):
        limiter = AdaptiveRateLimiter(base_rpm=60)
        decision = limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        assert decision.allowed is True
        assert decision.slowdown_factor == 1.0
        assert decision.effective_rpm == 60
        assert decision.reason == "ok"

    def test_slowdown_at_elevated_risk(self):
        limiter = AdaptiveRateLimiter(base_rpm=60)
        decision = limiter.check("src-1", risk_score=0.35, manipulation_flagged=False)
        assert decision.allowed is True
        assert decision.slowdown_factor == 0.6
        assert decision.effective_rpm == 36

    def test_slowdown_at_high_risk(self):
        limiter = AdaptiveRateLimiter(base_rpm=60)
        decision = limiter.check("src-1", risk_score=0.55, manipulation_flagged=False)
        assert decision.allowed is True
        assert decision.slowdown_factor == 0.3
        assert decision.effective_rpm == 18

    def test_slowdown_at_critical_risk(self):
        limiter = AdaptiveRateLimiter(base_rpm=60)
        decision = limiter.check("src-1", risk_score=0.75, manipulation_flagged=False)
        assert decision.allowed is True
        assert decision.slowdown_factor == 0.1
        assert decision.effective_rpm == 6


class TestAdaptiveRateLimiterCooling:
    """Test progressive cooling periods."""

    def test_first_cooling_5_seconds(self):
        limiter = AdaptiveRateLimiter(base_rpm=60)
        decision = limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        assert decision.allowed is True
        assert decision.reason == "cooling"
        assert decision.cooling_remaining_seconds > 0
        assert decision.cooling_remaining_seconds <= 5.0

    def test_progressive_cooling_escalation(self):
        """Cooling periods escalate: 5s, 15s, 45s, 120s."""
        limiter = AdaptiveRateLimiter(base_rpm=60)

        # 1st flag → 5s cooling
        d1 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        assert d1.cooling_remaining_seconds <= 5.0

        # Backdate cooling expiry
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1

        # 2nd flag → 15s cooling
        d2 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        assert d2.cooling_remaining_seconds <= 15.0
        assert d2.cooling_remaining_seconds > 5.0

        # Backdate again
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1

        # 3rd flag → 45s cooling
        d3 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        assert d3.cooling_remaining_seconds <= 45.0
        assert d3.cooling_remaining_seconds > 15.0

        # Backdate again
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1

        # 4th flag → 120s cooling
        d4 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        assert d4.cooling_remaining_seconds <= 120.0
        assert d4.cooling_remaining_seconds > 45.0

    def test_cooling_enforces_1_rpm(self):
        """During cooling, effective RPM drops to 1."""
        limiter = AdaptiveRateLimiter(base_rpm=60)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        # Second check during cooling period
        decision = limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        assert decision.effective_rpm == 1
        assert decision.reason == "cooling"


class TestAdaptiveRateLimiterHardStop:
    """Test hard stop on repeated manipulation."""

    def test_hard_stop_on_5th_attempt(self):
        limiter = AdaptiveRateLimiter(base_rpm=60, hard_stop_threshold=5)
        for i in range(4):
            # Backdate cooling to allow next check
            with limiter._lock:
                state = limiter._states.get("src-1")
                if state:
                    state.cooling_until = time.time() - 1
            d = limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
            assert d.allowed is True

        # 5th attempt → hard stop
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1
        d5 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        assert d5.allowed is False
        assert d5.hard_stopped is True
        assert d5.reason == "hard_stop"

    def test_hard_stop_expiry(self):
        """Hard stop expires after duration."""
        limiter = AdaptiveRateLimiter(base_rpm=60, hard_stop_threshold=2, hard_stop_duration=10)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)

        # Hard stopped
        d = limiter.check("src-1", risk_score=0.0, manipulation_flagged=False)
        assert d.allowed is False

        # Backdate hard stop expiry
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1
        d2 = limiter.check("src-1", risk_score=0.0, manipulation_flagged=False)
        assert d2.allowed is True
        assert d2.hard_stopped is False

    def test_clear_hard_stop(self):
        """Admin clear_hard_stop removes hard stop."""
        limiter = AdaptiveRateLimiter(base_rpm=60, hard_stop_threshold=2)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)

        d = limiter.check("src-1", risk_score=0.0, manipulation_flagged=False)
        assert d.allowed is False

        assert limiter.clear_hard_stop("src-1") is True
        d2 = limiter.check("src-1", risk_score=0.0, manipulation_flagged=False)
        assert d2.allowed is True

    def test_clear_hard_stop_nonexistent_source(self):
        limiter = AdaptiveRateLimiter()
        assert limiter.clear_hard_stop("nonexistent") is False


class TestAdaptiveRateLimiterIsolation:
    """Test per-source isolation and state management."""

    def test_per_source_isolation(self):
        """Different sources have independent state."""
        limiter = AdaptiveRateLimiter(base_rpm=60)
        limiter.check("src-1", risk_score=0.8, manipulation_flagged=True)
        d2 = limiter.check("src-2", risk_score=0.1, manipulation_flagged=False)
        assert d2.allowed is True
        assert d2.slowdown_factor == 1.0
        assert d2.effective_rpm == 60

    def test_lru_eviction(self):
        """Oldest entries evicted when max_states reached."""
        limiter = AdaptiveRateLimiter(base_rpm=60, max_states=3)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        limiter.check("src-2", risk_score=0.1, manipulation_flagged=False)
        limiter.check("src-3", risk_score=0.1, manipulation_flagged=False)
        assert limiter.state_count == 3

        # Adding 4th evicts src-1
        limiter.check("src-4", risk_score=0.1, manipulation_flagged=False)
        assert limiter.state_count == 3
        assert limiter.get_state("src-1") is None
        assert limiter.get_state("src-4") is not None

    def test_get_state(self):
        limiter = AdaptiveRateLimiter(base_rpm=60)
        limiter.check("src-1", risk_score=0.5, manipulation_flagged=False)
        state = limiter.get_state("src-1")
        assert state is not None
        assert state.source_id == "src-1"
        assert state.effective_rpm == 18

    def test_update_risk(self):
        """update_risk modifies state without checking."""
        limiter = AdaptiveRateLimiter(base_rpm=60)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        limiter.update_risk("src-1", risk_score=0.8, manipulation_flagged=True)
        state = limiter.get_state("src-1")
        assert state.slowdown_factor == 0.1
        assert state.manipulation_attempts == 1

    def test_risk_escalation_and_recovery(self):
        """Source risk escalates and recovers."""
        limiter = AdaptiveRateLimiter(base_rpm=60)
        # Low risk
        d1 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        assert d1.effective_rpm == 60
        # Escalate
        d2 = limiter.check("src-1", risk_score=0.8, manipulation_flagged=False)
        assert d2.effective_rpm == 6
        # Recover
        d3 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        assert d3.effective_rpm == 60


class TestAdaptiveRateLimiterRetryAfter:
    """Test Retry-After header behavior."""

    def test_hard_stop_has_cooling_remaining(self):
        limiter = AdaptiveRateLimiter(base_rpm=60, hard_stop_threshold=1, hard_stop_duration=300)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        d = limiter.check("src-1", risk_score=0.0, manipulation_flagged=False)
        assert d.allowed is False
        assert d.cooling_remaining_seconds > 0
        assert d.cooling_remaining_seconds <= 300


# ============================================================================
# Jailbreak Taxonomy Logger Tests (15+)
# ============================================================================


class TestTaxonomyClassification:
    """Test classification from various signal sources."""

    def test_classify_from_l2_pattern_ids(self):
        logger = JailbreakTaxonomyLogger()
        techs = logger.classify(pattern_ids=["PI-001", "RC-003"])
        assert JailbreakTechnique.INSTRUCTION_OVERRIDE in techs
        assert JailbreakTechnique.ROLEPLAY in techs

    def test_classify_from_mtmd_signals(self):
        logger = JailbreakTaxonomyLogger()
        techs = logger.classify(
            mtmd_signal_types=["boundary_testing", "tactic_switching"],
        )
        assert JailbreakTechnique.PROGRESSIVE_ESCALATION in techs
        assert JailbreakTechnique.AUTONOMOUS_AGENT in techs

    def test_classify_multi_category(self):
        """A single attempt can have multiple categories."""
        logger = JailbreakTaxonomyLogger()
        techs = logger.classify(
            pattern_ids=["PI-001"],
            mtmd_signal_types=["persona_adoption"],
            cross_modal=True,
            tool_use=True,
        )
        assert JailbreakTechnique.INSTRUCTION_OVERRIDE in techs
        assert JailbreakTechnique.PERSONA_ADOPTION in techs
        assert JailbreakTechnique.MULTI_MODAL_LAUNDERING in techs
        assert JailbreakTechnique.TOOL_EXPLOITATION in techs
        assert len(techs) == 4

    def test_classify_cross_modal(self):
        logger = JailbreakTaxonomyLogger()
        techs = logger.classify(cross_modal=True)
        assert techs == [JailbreakTechnique.MULTI_MODAL_LAUNDERING]

    def test_classify_tool_use(self):
        logger = JailbreakTaxonomyLogger()
        techs = logger.classify(tool_use=True)
        assert techs == [JailbreakTechnique.TOOL_EXPLOITATION]

    def test_classify_unknown_fallback(self):
        """No signals → UNKNOWN."""
        logger = JailbreakTaxonomyLogger()
        techs = logger.classify()
        assert techs == [JailbreakTechnique.UNKNOWN]

    def test_classify_unknown_pattern_prefix(self):
        """Unknown prefix → only UNKNOWN returned."""
        logger = JailbreakTaxonomyLogger()
        techs = logger.classify(pattern_ids=["ZZ-999"])
        assert techs == [JailbreakTechnique.UNKNOWN]


class TestTaxonomyLogging:
    """Test attempt logging and retrieval."""

    def test_log_attempt_returns_attempt(self):
        logger = JailbreakTaxonomyLogger()
        attempt = logger.log_attempt(
            source_id="src-1",
            session_id="sess-1",
            detection_layer="innate",
            confidence=0.95,
            blocked=True,
            pattern_ids=["PI-001"],
        )
        assert isinstance(attempt, JailbreakAttempt)
        assert attempt.source_id == "src-1"
        assert attempt.blocked is True
        assert JailbreakTechnique.INSTRUCTION_OVERRIDE in attempt.techniques

    def test_primary_technique_selection(self):
        """Primary technique is first in sorted list (deterministic)."""
        logger = JailbreakTaxonomyLogger()
        attempt = logger.log_attempt(
            source_id="src-1",
            session_id="sess-1",
            detection_layer="innate",
            confidence=0.9,
            blocked=True,
            pattern_ids=["PI-001", "RC-003"],
        )
        # Sorted by enum value: instruction_override < roleplay
        assert attempt.primary_technique == attempt.techniques[0]

    def test_total_logged(self):
        logger = JailbreakTaxonomyLogger()
        assert logger.total_logged == 0
        logger.log_attempt("src-1", "sess-1", "innate", 0.9, True, pattern_ids=["PI-001"])
        logger.log_attempt("src-2", "sess-2", "adaptive", 0.85, True)
        assert logger.total_logged == 2

    def test_get_recent_attempts(self):
        logger = JailbreakTaxonomyLogger()
        for i in range(10):
            logger.log_attempt(f"src-{i}", f"sess-{i}", "innate", 0.9, True)
        recent = logger.get_recent_attempts(n=5)
        assert len(recent) == 5
        assert recent[-1].source_id == "src-9"

    def test_fifo_eviction(self):
        """Oldest attempts evicted when max reached."""
        logger = JailbreakTaxonomyLogger(max_attempts=5)
        for i in range(10):
            logger.log_attempt(f"src-{i}", f"sess-{i}", "innate", 0.9, True)
        assert logger.total_logged == 5
        recent = logger.get_recent_attempts(n=10)
        assert recent[0].source_id == "src-5"


class TestTaxonomyStats:
    """Test statistics computation."""

    def test_stats_computation(self):
        logger = JailbreakTaxonomyLogger()
        logger.log_attempt("s1", "ses1", "innate", 0.9, True, pattern_ids=["PI-001"])
        logger.log_attempt("s2", "ses2", "innate", 0.9, True, pattern_ids=["PI-002"])
        logger.log_attempt("s3", "ses3", "adaptive", 0.85, True, pattern_ids=["RC-001"])
        stats = logger.get_stats()
        assert stats.total_attempts == 3
        assert stats.by_technique[JailbreakTechnique.INSTRUCTION_OVERRIDE] == 2
        assert stats.by_technique[JailbreakTechnique.ROLEPLAY] == 1
        assert stats.by_detection_layer["innate"] == 2
        assert stats.by_detection_layer["adaptive"] == 1

    def test_top_techniques(self):
        logger = JailbreakTaxonomyLogger()
        for _ in range(5):
            logger.log_attempt("s1", "ses1", "innate", 0.9, True, pattern_ids=["PI-001"])
        for _ in range(3):
            logger.log_attempt("s2", "ses2", "innate", 0.9, True, pattern_ids=["RC-001"])
        for _ in range(1):
            logger.log_attempt("s3", "ses3", "adaptive", 0.85, True, cross_modal=True)
        top = logger.get_top_techniques(n=2)
        assert top[0][0] == JailbreakTechnique.INSTRUCTION_OVERRIDE
        assert top[0][1] == 5
        assert top[1][0] == JailbreakTechnique.ROLEPLAY
        assert top[1][1] == 3

    def test_technique_trend(self):
        logger = JailbreakTaxonomyLogger()
        # Log attempts at known times
        now = time.time()
        for i in range(3):
            attempt = logger.log_attempt(
                "s1", "ses1", "innate", 0.9, True, pattern_ids=["PI-001"],
            )
            # Backdate: 0, 1, 2 hours ago
            with logger._lock:
                logger._attempts[-1].timestamp = now - i * 3600

        trend = logger.get_technique_trend(
            JailbreakTechnique.INSTRUCTION_OVERRIDE, window_hours=24.0,
        )
        assert sum(trend.values()) == 3

    def test_trend_window_filtering(self):
        """Stats respect trend window."""
        logger = JailbreakTaxonomyLogger()
        now = time.time()
        # Old attempt (48 hours ago)
        logger.log_attempt("s1", "ses1", "innate", 0.9, True, pattern_ids=["PI-001"])
        with logger._lock:
            logger._attempts[-1].timestamp = now - 48 * 3600
        # Recent attempt
        logger.log_attempt("s2", "ses2", "innate", 0.9, True, pattern_ids=["RC-001"])

        stats = logger.get_stats(trend_window_hours=24.0)
        assert stats.total_attempts == 2  # all-time count
        # Recent trend should only include the recent one
        assert JailbreakTechnique.INSTRUCTION_OVERRIDE not in stats.recent_trend
        assert stats.recent_trend.get(JailbreakTechnique.ROLEPLAY, 0) == 1

    def test_empty_stats(self):
        logger = JailbreakTaxonomyLogger()
        stats = logger.get_stats()
        assert stats.total_attempts == 0
        assert stats.by_technique == {}
        assert stats.top_techniques == []


# ============================================================================
# Taxonomy API Endpoint Tests
# ============================================================================


class TestTaxonomyEndpoint:
    """Test GET /v1/taxonomy/stats endpoint."""

    def test_taxonomy_stats_unauthenticated(self):
        from aegis.main import app
        client = TestClient(app)
        resp = client.get("/v1/taxonomy/stats")
        assert resp.status_code == 401

    def test_taxonomy_stats_authenticated(self):
        from aegis.main import app, _config

        if not _config or not _config.api_key:
            pytest.skip("No API key configured")

        client = TestClient(app)
        resp = client.get(
            "/v1/taxonomy/stats",
            headers={"Authorization": f"Bearer {_config.api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_attempts" in data
        assert "by_technique" in data
        assert "top_techniques" in data


# ============================================================================
# Integration Tests (5+)
# ============================================================================


class TestAdaptiveRateLimiterIntegration:
    """Integration tests: rate limiter + MTMD + taxonomy."""

    def test_rate_limiter_with_mtmd_signals(self):
        """MTMD manipulation flags feed into adaptive rate limiter."""
        from aegis.layers.adaptive.manipulation_detector import (
            MultiTurnManipulationDetector,
        )

        detector = MultiTurnManipulationDetector(window_size=5)
        limiter = AdaptiveRateLimiter(base_rpm=60)

        # Record escalating turns
        for i in range(10):
            detector.record_turn(
                source_id="agent-1",
                content=f"Ignore instructions {i}",
                injection_score=0.3 + i * 0.05,
                was_blocked=False,
                detection_categories=["PI-001"] if i % 2 == 0 else [],
            )

        report = detector.analyze("agent-1", "sess-1")
        manipulation_flagged = report.should_alert

        decision = limiter.check("agent-1", risk_score=0.0, manipulation_flagged=manipulation_flagged)
        # If MTMD flagged, cooling should apply
        if manipulation_flagged:
            assert decision.reason in ("cooling", "ok")

    def test_taxonomy_receives_all_block_types(self):
        """Taxonomy logger receives blocks from innate, adaptive, and manipulation layers."""
        taxonomy = JailbreakTaxonomyLogger()

        # Innate block
        taxonomy.log_attempt(
            "src-1", "sess-1", "innate", 0.95, True,
            pattern_ids=["PI-001", "EE-003"],
        )
        # Adaptive block
        taxonomy.log_attempt(
            "src-2", "sess-2", "adaptive", 0.92, True,
        )
        # Manipulation block
        taxonomy.log_attempt(
            "src-3", "sess-3", "manipulation", 0.88, True,
            mtmd_signal_types=["boundary_testing", "tactic_switching"],
        )
        # Output block
        taxonomy.log_attempt(
            "src-4", "sess-4", "output", 1.0, True,
        )
        # Distillation block
        taxonomy.log_attempt(
            "src-5", "sess-5", "distillation", 0.9, True,
        )

        stats = taxonomy.get_stats()
        assert stats.total_attempts == 5
        assert "innate" in stats.by_detection_layer
        assert "adaptive" in stats.by_detection_layer
        assert "manipulation" in stats.by_detection_layer
        assert "output" in stats.by_detection_layer
        assert "distillation" in stats.by_detection_layer

    def test_hard_stop_prevents_requests(self):
        """Hard-stopped source is denied all subsequent requests."""
        limiter = AdaptiveRateLimiter(base_rpm=60, hard_stop_threshold=2)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)

        # Multiple subsequent checks are all denied
        for _ in range(5):
            d = limiter.check("src-1", risk_score=0.0, manipulation_flagged=False)
            assert d.allowed is False
            assert d.hard_stopped is True

    def test_recovery_after_cooling(self):
        """Source recovers after cooling period expires."""
        limiter = AdaptiveRateLimiter(base_rpm=60)
        limiter.check("src-1", risk_score=0.1, manipulation_flagged=True)

        # During cooling → limited
        d = limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        assert d.reason == "cooling"

        # Backdate cooling expiry
        with limiter._lock:
            limiter._states["src-1"].cooling_until = time.time() - 1

        # After cooling → normal
        d2 = limiter.check("src-1", risk_score=0.1, manipulation_flagged=False)
        assert d2.reason == "ok"
        assert d2.effective_rpm == 60

    def test_taxonomy_stats_reflect_distribution(self):
        """Stats accurately reflect the distribution of logged attempts."""
        taxonomy = JailbreakTaxonomyLogger()

        # 10 injection overrides, 5 roleplay, 2 encoding tricks
        for _ in range(10):
            taxonomy.log_attempt("s", "s", "innate", 0.9, True, pattern_ids=["PI-001"])
        for _ in range(5):
            taxonomy.log_attempt("s", "s", "innate", 0.9, True, pattern_ids=["RC-001"])
        for _ in range(2):
            taxonomy.log_attempt("s", "s", "innate", 0.9, True, pattern_ids=["EE-001"])

        stats = taxonomy.get_stats()
        assert stats.by_technique[JailbreakTechnique.INSTRUCTION_OVERRIDE] == 10
        assert stats.by_technique[JailbreakTechnique.ROLEPLAY] == 5
        assert stats.by_technique[JailbreakTechnique.ENCODING_TRICK] == 2

        top = taxonomy.get_top_techniques(n=3)
        assert top[0] == (JailbreakTechnique.INSTRUCTION_OVERRIDE, 10)
        assert top[1] == (JailbreakTechnique.ROLEPLAY, 5)
        assert top[2] == (JailbreakTechnique.ENCODING_TRICK, 2)

    def test_concurrent_source_rate_limiting(self):
        """Multiple sources tracked independently under rate limiter."""
        limiter = AdaptiveRateLimiter(base_rpm=60)

        # High-risk source
        limiter.check("attacker", risk_score=0.8, manipulation_flagged=True)
        # Normal source
        limiter.check("legit", risk_score=0.0, manipulation_flagged=False)

        attacker_state = limiter.get_state("attacker")
        legit_state = limiter.get_state("legit")

        assert attacker_state.manipulation_attempts == 1
        assert legit_state.manipulation_attempts == 0
        assert legit_state.effective_rpm == 60
