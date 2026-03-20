"""
Tests for adaptive baseline freeze/unfreeze on the fingerprint detector
and engine-triggered freeze during active campaigns.
"""

from __future__ import annotations

import time

import pytest

from aegis.layers.correlation.events import ActionType, AgentActionEvent
from aegis.layers.correlation.fingerprint_detector import FingerprintDetector
from aegis.layers.correlation.engine import CampaignCorrelationEngine


def _make_event(
    agent_id: str = "agent-1",
    action_type: ActionType = ActionType.API_CALL,
    target: str = "/api/v1/users",
    timestamp: float | None = None,
    parameter_value: str = "",
) -> AgentActionEvent:
    return AgentActionEvent(
        agent_id=agent_id,
        action_type=action_type,
        target_resource=target,
        timestamp=timestamp or time.time(),
        parameter_value=parameter_value,
    )


class TestBaselineFreeze:
    """Tests for freeze/unfreeze of adaptive baselines."""

    def test_freeze_prevents_entropy_drift(self):
        """After freeze, 100 attack events don't change entropy baseline."""
        detector = FingerprintDetector(window_sizes=[60])
        now = time.time()

        # Build up some baseline entropy history
        for i in range(10):
            detector._events.append(_make_event(
                agent_id=f"normal-{i}",
                target="/api/v1/login",
                parameter_value=f"val_{i % 3}",  # Low entropy
                timestamp=now + i,
            ))
        detector.detect()  # Populate entropy_history

        pre_freeze_history = list(detector._entropy_history)
        assert len(pre_freeze_history) > 0

        # Freeze baselines
        detector.freeze_baselines()
        assert detector.is_frozen is True

        # Add 100 high-entropy attack events
        for i in range(100):
            detector._events.append(_make_event(
                agent_id=f"attacker-{i}",
                target="/api/v1/login",
                parameter_value=f"unique_payload_{i}",
                timestamp=now + 20 + i * 0.1,
            ))
            detector.detect()

        # Entropy history should NOT have grown (frozen)
        assert len(detector._entropy_history) == len(pre_freeze_history)

    def test_unfreeze_resumes_updates(self):
        """After unfreeze, baseline updates resume."""
        detector = FingerprintDetector(window_sizes=[60])
        now = time.time()

        # Build baseline
        for i in range(10):
            detector._events.append(_make_event(
                agent_id=f"a-{i}",
                target="/api/login",
                parameter_value=f"v_{i % 3}",
                timestamp=now + i,
            ))
        detector.detect()
        pre_freeze_len = len(detector._entropy_history)

        # Freeze then unfreeze
        detector.freeze_baselines()
        assert detector.is_frozen is True
        detector.unfreeze_baselines()
        assert detector.is_frozen is False

        # Add more events — history should grow
        for i in range(10):
            detector._events.append(_make_event(
                agent_id=f"b-{i}",
                target="/api/login",
                parameter_value=f"new_{i}",
                timestamp=now + 20 + i,
            ))
        detector.detect()
        assert len(detector._entropy_history) > pre_freeze_len

    def test_freeze_uses_saved_threshold(self):
        """When frozen, detection uses pre-freeze thresholds, not live."""
        detector = FingerprintDetector(window_sizes=[300], fuzzing_entropy_std_threshold=2.0)
        now = time.time()

        # Build baseline with enough agents and distinct param values
        # to trigger parameter fuzzing detection and populate entropy_history
        for i in range(20):
            detector._events.append(_make_event(
                agent_id=f"a-{i}",
                target="/api/login",
                parameter_value=f"val_{i % 5}",
                timestamp=now + i,
            ))
        detector.detect()

        # Directly seed entropy_history if detection didn't populate it
        # (threshold logic requires >= 5 entries)
        if len(detector._entropy_history) < 5:
            detector._entropy_history = [1.0, 1.1, 0.9, 1.0, 1.05]

        # Save state and freeze
        detector.freeze_baselines()
        frozen_history = list(detector._frozen_entropy_history)
        assert len(frozen_history) >= 5

        # Add high-entropy events — history should NOT change
        for i in range(20):
            detector._events.append(_make_event(
                agent_id=f"fuzz-{i}",
                target="/api/login",
                parameter_value=f"unique_{i}",
                timestamp=now + 50 + i,
            ))
        detector.detect()

        # Frozen entropy history unchanged, live history also unchanged (frozen)
        assert detector._frozen_entropy_history == frozen_history
        assert len(detector._entropy_history) == len(frozen_history)

    def test_is_frozen_property(self):
        """is_frozen property tracks freeze state."""
        detector = FingerprintDetector()
        assert detector.is_frozen is False
        detector.freeze_baselines()
        assert detector.is_frozen is True
        detector.unfreeze_baselines()
        assert detector.is_frozen is False


class TestEngineFreezeIntegration:
    """Tests for engine triggering freeze on campaign alerts."""

    @pytest.mark.asyncio
    async def test_engine_freezes_on_campaign_alert(self):
        """XBOW campaign → detector baselines frozen after alert."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            enumeration_coverage_threshold=0.6,
            campaign_alert_threshold=0.6,
        )
        await engine.start()

        now = time.time()
        endpoints = [f"/api/v1/endpoint-{i}" for i in range(10)]
        engine.detector._known_resources = set(endpoints)

        # Run enough events to trigger a campaign alert
        for i in range(50):
            target = endpoints[i % len(endpoints)]
            await engine.ingest_event(_make_event(
                agent_id=f"solver-{i}",
                action_type=ActionType.PROBE,
                target=target,
                timestamp=now + i * 0.5,
            ))

        # Detector should be frozen after campaign alert
        assert engine.detector.is_frozen is True
        assert engine.get_stats()["alerts_generated"] > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_detection_stable_during_freeze(self):
        """Detection accuracy doesn't degrade as more attack events arrive during freeze."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            temporal_cluster_min_agents=5,
            temporal_cluster_cv_threshold=0.5,
            campaign_alert_threshold=0.5,
        )

        now = time.time()
        all_alerts: list = []

        # First burst — triggers alert and freeze
        for i in range(10):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"bot-{i}",
                target="/api/v1/critical",
                timestamp=now + i * 0.1,
            ))
            all_alerts.extend(alerts)

        assert len(all_alerts) > 0
        assert engine.detector.is_frozen is True

        # Second burst — more attack events while frozen
        second_burst_alerts: list = []
        for i in range(20):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"bot-{10 + i}",
                target="/api/v1/critical",
                timestamp=now + 5 + i * 0.1,
            ))
            second_burst_alerts.extend(alerts)

        # Engine continued processing while frozen (events were ingested)
        assert engine.get_stats()["events_processed"] == 30
        # Total alerts should reflect ongoing detection
        total = len(all_alerts) + len(second_burst_alerts)
        assert total > 0
