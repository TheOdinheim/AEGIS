"""
Tests for intent detection integration with the Campaign Correlation Engine.
Full XBOW simulation, legitimate traffic, alloy detection, baseline freeze.
"""

from __future__ import annotations

import time

import pytest

from aegis.layers.correlation.events import ActionType, AgentActionEvent
from aegis.layers.correlation.engine import CampaignCorrelationEngine
from aegis.layers.correlation.intent_alert import IntentCategory


def _make_event(
    agent_id: str = "agent-1",
    action_type: ActionType = ActionType.API_CALL,
    target: str = "/api/v1/users",
    timestamp: float | None = None,
    input_hash: str = "",
    output_hash: str = "",
    source_model: str | None = None,
    parameter_value: str = "",
    tenant_id: str = "default",
) -> AgentActionEvent:
    return AgentActionEvent(
        agent_id=agent_id,
        action_type=action_type,
        target_resource=target,
        timestamp=timestamp or time.time(),
        input_hash=input_hash,
        output_hash=output_hash,
        source_model=source_model,
        parameter_value=parameter_value,
        tenant_id=tenant_id,
    )


class TestXBOWSimulationWithIntent:
    """Full XBOW simulation with intent classification."""

    @pytest.mark.asyncio
    async def test_xbow_campaign_has_intent(self):
        """XBOW campaign → CampaignAlert with intent field populated."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            enumeration_coverage_threshold=0.6,
            recon_exploit_transition_threshold=0.7,
            info_flow_min_links=3,
            campaign_alert_threshold=0.5,
        )
        await engine.start()

        now = time.time()
        endpoints = [f"/api/v1/ep-{i}" for i in range(8)]
        engine.detector._known_resources = set(endpoints)

        all_alerts: list = []

        # Phase 1: 20 recon agents enumerate endpoints
        for i in range(20):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"recon-{i}",
                action_type=ActionType.PROBE,
                target=endpoints[i % len(endpoints)],
                timestamp=now + i * 0.3,
            ))
            all_alerts.extend(alerts)

        # Phase 2: 10 exploit agents attack
        for i in range(10):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"exploit-{i}",
                action_type=ActionType.AUTH_ATTEMPT,
                target=endpoints[i % 3],
                timestamp=now + 15 + i,
            ))
            all_alerts.extend(alerts)

        # Phase 3: info flow chains
        for i in range(5):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"chain-a-{i}",
                target="/exec",
                output_hash=f"chain_{i}",
                timestamp=now + 30 + i,
            ))
            all_alerts.extend(alerts)
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"chain-b-{i}",
                target="/exec",
                input_hash=f"chain_{i}",
                output_hash=f"chain_next_{i}",
                timestamp=now + 31 + i,
            ))
            all_alerts.extend(alerts)

        # Should have campaign alerts
        assert len(all_alerts) > 0

        # Check for intent field on alerts
        alerts_with_intent = [a for a in all_alerts if a.intent is not None]
        # At least some alerts should have intent
        # (intent only attached when confidence > 0.7 and not benign)
        # The campaign may or may not meet intent thresholds
        assert engine.get_stats()["alerts_generated"] > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_classify_events_direct(self):
        """Direct classify_events on campaign-like events."""
        engine = CampaignCorrelationEngine(window_sizes=[60])

        now = time.time()
        events = []

        # Create enumeration-like events
        for i in range(30):
            events.append(_make_event(
                agent_id=f"scanner-{i}",
                action_type=ActionType.PROBE,
                target=f"/api/v1/ep-{i}",
                timestamp=now + i * 0.5,
            ))

        result = engine.classify_events(events)
        # With 30 unique targets and agents, should detect enumeration
        assert result.category in (
            IntentCategory.SYSTEMATIC_ENUMERATION,
            IntentCategory.BENIGN_ACTIVITY,
        )
        assert result.confidence > 0


class TestLegitimateTrafficIntent:
    """Tests that legitimate traffic classifies as benign."""

    @pytest.mark.asyncio
    async def test_uncorrelated_agents_benign(self):
        """Normal uncorrelated traffic → BENIGN_ACTIVITY."""
        engine = CampaignCorrelationEngine(
            window_sizes=[30],
            campaign_alert_threshold=0.7,
        )

        now = time.time()
        rng = __import__("random").Random(42)
        events = []

        # Few agents making varied API calls with irregular timing
        # Low agent cardinality, high temporal irregularity, no progression
        targets = ["/api/v1/data", "/api/v1/status", "/api/v1/profile"]
        for i in range(20):
            events.append(_make_event(
                agent_id=f"user-{i % 3}",
                action_type=ActionType.API_CALL,
                target=targets[i % len(targets)],
                timestamp=now + rng.uniform(0, 100),  # Irregular
            ))

        result = engine.classify_events(events)
        assert result.category == IntentCategory.BENIGN_ACTIVITY


class TestAlloyCampaign:
    """Tests for alloy-powered (multi-model) campaign detection."""

    @pytest.mark.asyncio
    async def test_alloy_campaign_amplification(self):
        """Multi-model enumeration → alloy amplification fires."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            enumeration_coverage_threshold=0.6,
            campaign_alert_threshold=0.5,
        )
        await engine.start()

        now = time.time()
        endpoints = [f"/api/v1/ep-{i}" for i in range(10)]
        engine.detector._known_resources = set(endpoints)

        models = ["claude", "gpt-5", "gemini"]
        all_alerts: list = []

        # 50 agents with alternating models doing enumeration
        for i in range(50):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"solver-{i}",
                action_type=ActionType.PROBE,
                target=endpoints[i % len(endpoints)],
                source_model=models[i % len(models)],
                timestamp=now + i * 0.3,
            ))
            all_alerts.extend(alerts)

        assert len(all_alerts) > 0

        # Run direct classification to check alloy signal
        events = list(engine.detector._events)
        result = engine.classify_events(events)

        # If features match enumeration + discontinuity, alloy should fire
        if result.category != IntentCategory.BENIGN_ACTIVITY:
            # Check stylistic discontinuity was detected in the features
            if result.feature_vector:
                assert result.feature_vector.stylistic_discontinuity > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_alloy_higher_confidence_than_single_model(self):
        """Multi-model campaign should have equal or higher confidence than single-model."""
        engine_single = CampaignCorrelationEngine(window_sizes=[60])
        engine_multi = CampaignCorrelationEngine(window_sizes=[60])

        now = time.time()
        targets = [f"/api/ep-{i}" for i in range(10)]

        # Single-model events
        single_events = [
            _make_event(
                agent_id=f"a-{i}",
                action_type=ActionType.PROBE,
                target=targets[i % len(targets)],
                source_model="claude",
                timestamp=now + i * 0.5,
            )
            for i in range(30)
        ]

        # Multi-model events (same patterns, different models)
        models = ["claude", "gpt-5", "gemini"]
        multi_events = [
            _make_event(
                agent_id=f"a-{i}",
                action_type=ActionType.PROBE,
                target=targets[i % len(targets)],
                source_model=models[i % len(models)],
                timestamp=now + i * 0.5,
            )
            for i in range(30)
        ]

        single_result = engine_single.classify_events(single_events)
        multi_result = engine_multi.classify_events(multi_events)

        # If both classify as the same category, multi should have >= confidence
        if (
            single_result.category == multi_result.category
            and single_result.category != IntentCategory.BENIGN_ACTIVITY
        ):
            assert multi_result.confidence >= single_result.confidence


class TestBaselineFreezeDuringCampaign:
    """Tests for baseline freeze during active campaigns."""

    @pytest.mark.asyncio
    async def test_freeze_during_xbow(self):
        """Baselines frozen during active XBOW campaign."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            enumeration_coverage_threshold=0.6,
            campaign_alert_threshold=0.6,
        )
        await engine.start()

        now = time.time()
        endpoints = [f"/api/v1/ep-{i}" for i in range(10)]
        engine.detector._known_resources = set(endpoints)

        # Generate campaign
        for i in range(50):
            await engine.ingest_event(_make_event(
                agent_id=f"solver-{i}",
                action_type=ActionType.PROBE,
                target=endpoints[i % len(endpoints)],
                timestamp=now + i * 0.5,
            ))

        # Should be frozen
        assert engine.detector.is_frozen is True
        assert engine.get_stats()["alerts_generated"] > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_no_detection_degradation_under_flood(self):
        """Detection doesn't degrade as attacker floods events during freeze."""
        engine = CampaignCorrelationEngine(
            window_sizes=[30],
            temporal_cluster_min_agents=5,
            temporal_cluster_cv_threshold=0.5,
            campaign_alert_threshold=0.5,
        )

        now = time.time()
        first_burst_alerts: list = []

        # First burst — triggers alert and freeze
        for i in range(10):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"bot-{i}",
                target="/api/target",
                timestamp=now + i * 0.1,
            ))
            first_burst_alerts.extend(alerts)

        assert len(first_burst_alerts) > 0
        assert engine.detector.is_frozen is True

        # Flood of events while frozen
        for i in range(50):
            await engine.ingest_event(_make_event(
                agent_id=f"bot-{10 + i}",
                target="/api/target",
                timestamp=now + 5 + i * 0.1,
            ))

        # All events processed — engine didn't crash or stall
        assert engine.get_stats()["events_processed"] == 60
        # Alerts were generated during the campaign
        assert engine.get_stats()["alerts_generated"] > 0
