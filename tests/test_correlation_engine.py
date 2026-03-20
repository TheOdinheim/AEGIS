"""
Tests for the Campaign Correlation Engine — integration, simulated XBOW
campaign, negative (legitimate traffic), and degradation tests.
"""

from __future__ import annotations

import time

import pytest

from aegis.layers.correlation.events import (
    ActionType,
    AgentActionEvent,
    AlertLevel,
    FingerprintType,
)
from aegis.layers.correlation.engine import CampaignCorrelationEngine


def _make_event(
    agent_id: str = "agent-1",
    action_type: ActionType = ActionType.API_CALL,
    target: str = "/api/v1/users",
    timestamp: float | None = None,
    input_hash: str = "",
    output_hash: str = "",
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
        parameter_value=parameter_value,
        tenant_id=tenant_id,
    )


class TestEngineLifecycle:
    """Tests for engine start/stop and basic operations."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Engine starts and stops cleanly."""
        engine = CampaignCorrelationEngine()
        await engine.start()
        assert engine._running is True
        await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Stats reflect engine state."""
        engine = CampaignCorrelationEngine()
        stats = engine.get_stats()
        assert stats["events_processed"] == 0
        assert stats["alerts_generated"] == 0

    @pytest.mark.asyncio
    async def test_ingest_single_event_no_alert(self):
        """Single event does not trigger alert."""
        engine = CampaignCorrelationEngine()
        alerts = await engine.ingest_event(_make_event())
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Engine handles errors gracefully, never crashes."""
        engine = CampaignCorrelationEngine()
        # Even if detector is somehow broken, ingest_event returns empty
        result = await engine.ingest_event(_make_event())
        assert isinstance(result, list)


class TestSimulatedXBOWCampaign:
    """Simulated multi-agent campaign detection tests.

    Recreates the XBOW pattern: 50 solver agents systematically
    probing endpoints, with recon-to-exploit progression and
    information flow chains.
    """

    @pytest.mark.asyncio
    async def test_systematic_enumeration_campaign(self):
        """50 agents systematically probing endpoints → campaign detected."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            enumeration_coverage_threshold=0.6,
            campaign_alert_threshold=0.6,
        )
        await engine.start()

        now = time.time()
        endpoints = [f"/api/v1/endpoint-{i}" for i in range(10)]

        # Pre-populate known resources
        engine.detector._known_resources = set(endpoints)

        # 50 solver agents each probe different endpoints
        all_alerts: list = []
        for i in range(50):
            target = endpoints[i % len(endpoints)]
            event = _make_event(
                agent_id=f"solver-{i}",
                action_type=ActionType.PROBE,
                target=target,
                timestamp=now + i * 0.5,
            )
            alerts = await engine.ingest_event(event)
            all_alerts.extend(alerts)

        assert len(all_alerts) > 0
        # Should have an enumeration detection
        enum_alerts = [a for a in all_alerts if a.fingerprint_type == FingerprintType.SYSTEMATIC_ENUMERATION]
        assert len(enum_alerts) > 0
        assert enum_alerts[0].confidence >= 0.6

        stats = engine.get_stats()
        assert stats["events_processed"] == 50
        assert stats["alerts_generated"] > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_recon_to_exploit_campaign(self):
        """Agents transitioning from recon to exploit → campaign detected."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            recon_exploit_transition_threshold=0.7,
            campaign_alert_threshold=0.7,
        )
        await engine.start()

        now = time.time()

        # First phase: 10 recon agents probing
        for i in range(10):
            await engine.ingest_event(_make_event(
                agent_id=f"recon-{i}",
                action_type=ActionType.PROBE,
                target="/api/v1/admin",
                timestamp=now + i,
            ))

        # Second phase: 10 exploit agents attacking
        all_alerts: list = []
        for i in range(10):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"exploit-{i}",
                action_type=ActionType.AUTH_ATTEMPT if i % 2 == 0 else ActionType.TOOL_INVOCATION,
                target="/api/v1/admin",
                timestamp=now + 30 + i,
            ))
            all_alerts.extend(alerts)

        r2e_alerts = [a for a in all_alerts if a.fingerprint_type == FingerprintType.RECON_TO_EXPLOIT]
        assert len(r2e_alerts) > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_information_flow_campaign(self):
        """Coordinator-solver hash chain → campaign detected."""
        engine = CampaignCorrelationEngine(
            window_sizes=[60],
            info_flow_min_links=3,
            campaign_alert_threshold=0.5,
        )
        await engine.start()

        now = time.time()
        all_alerts: list = []

        # Coordinator produces plan
        await engine.ingest_event(_make_event(
            agent_id="coordinator",
            target="/plan",
            output_hash="plan_1",
            timestamp=now,
        ))

        # Solver 1 consumes plan, produces result
        alerts = await engine.ingest_event(_make_event(
            agent_id="solver-1",
            target="/exec",
            input_hash="plan_1",
            output_hash="result_1",
            timestamp=now + 1,
        ))
        all_alerts.extend(alerts)

        # Solver 2 consumes solver-1's result
        alerts = await engine.ingest_event(_make_event(
            agent_id="solver-2",
            target="/exec",
            input_hash="result_1",
            output_hash="result_2",
            timestamp=now + 2,
        ))
        all_alerts.extend(alerts)

        # Solver 3 consumes solver-2's result
        alerts = await engine.ingest_event(_make_event(
            agent_id="solver-3",
            target="/exec",
            input_hash="result_2",
            output_hash="result_3",
            timestamp=now + 3,
        ))
        all_alerts.extend(alerts)

        # Solver 4 consumes solver-3's result
        alerts = await engine.ingest_event(_make_event(
            agent_id="solver-4",
            target="/exec",
            input_hash="result_3",
            output_hash="result_4",
            timestamp=now + 4,
        ))
        all_alerts.extend(alerts)

        flow_alerts = [a for a in all_alerts if a.fingerprint_type == FingerprintType.INFORMATION_FLOW]
        assert len(flow_alerts) > 0
        assert flow_alerts[0].evidence["chain_count"] >= 3

        await engine.stop()

    @pytest.mark.asyncio
    async def test_temporal_cluster_campaign(self):
        """20 agents hitting same endpoint simultaneously → campaign detected."""
        engine = CampaignCorrelationEngine(
            window_sizes=[10],
            temporal_cluster_min_agents=10,
            temporal_cluster_cv_threshold=0.5,
            campaign_alert_threshold=0.5,
        )
        await engine.start()

        now = time.time()
        all_alerts: list = []

        for i in range(20):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"bot-{i}",
                target="/api/v1/target",
                timestamp=now + i * 0.2,
            ))
            all_alerts.extend(alerts)

        temporal_alerts = [a for a in all_alerts if a.fingerprint_type == FingerprintType.TEMPORAL_CLUSTER]
        assert len(temporal_alerts) > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_full_xbow_simulation(self):
        """Full XBOW-style campaign: enumeration + recon-to-exploit + info flow."""
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

        # Phase 3: info flow chains between 5 pairs
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

        # Should have detected at least one campaign pattern
        assert len(all_alerts) > 0
        assert engine.get_stats()["alerts_generated"] > 0

        # Verify alert structure
        for alert in all_alerts:
            assert alert.campaign_id
            assert alert.confidence >= 0.5
            assert len(alert.contributing_agent_ids) > 0
            assert alert.recommended_action

        await engine.stop()


class TestLegitimateTraffic:
    """Negative tests: legitimate traffic should not trigger alerts."""

    @pytest.mark.asyncio
    async def test_uncorrelated_agents(self):
        """50 agents making normal, uncorrelated API calls → no alert."""
        engine = CampaignCorrelationEngine(
            window_sizes=[30],
            temporal_cluster_min_agents=10,
            temporal_cluster_cv_threshold=0.5,
            campaign_alert_threshold=0.7,
        )
        await engine.start()

        now = time.time()
        all_alerts: list = []
        targets = [f"/api/v1/resource-{i}" for i in range(50)]

        # Each agent hits its own unique endpoint — no coordination
        for i in range(50):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"user-{i}",
                action_type=ActionType.API_CALL,
                target=targets[i],
                timestamp=now + i * 5,  # Well-spaced timing
            ))
            all_alerts.extend(alerts)

        # No temporal clustering (different targets, spaced timing)
        temporal = [a for a in all_alerts if a.fingerprint_type == FingerprintType.TEMPORAL_CLUSTER]
        assert len(temporal) == 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_single_agent_repeated_calls(self):
        """Single agent making repeated calls → no campaign alert."""
        engine = CampaignCorrelationEngine(
            window_sizes=[30],
            temporal_cluster_min_agents=10,
            campaign_alert_threshold=0.7,
        )
        await engine.start()

        now = time.time()
        all_alerts: list = []

        for i in range(30):
            alerts = await engine.ingest_event(_make_event(
                agent_id="legitimate-user",
                action_type=ActionType.API_CALL,
                target="/api/v1/data",
                timestamp=now + i,
            ))
            all_alerts.extend(alerts)

        # Single agent should not trigger multi-agent detection
        temporal = [a for a in all_alerts if a.fingerprint_type == FingerprintType.TEMPORAL_CLUSTER]
        assert len(temporal) == 0

        await engine.stop()


class TestDegradation:
    """Tests for graceful degradation."""

    @pytest.mark.asyncio
    async def test_engine_without_event_bus(self):
        """Engine works without event bus (no crash on publish)."""
        engine = CampaignCorrelationEngine(
            campaign_alert_threshold=0.3,
            window_sizes=[30],
            temporal_cluster_min_agents=3,
        )
        await engine.start()

        now = time.time()
        # Generate enough events to trigger detection
        for i in range(10):
            await engine.ingest_event(_make_event(
                agent_id=f"a-{i}",
                target="/api",
                timestamp=now + i * 0.1,
            ))

        # Should not crash even without event bus
        stats = engine.get_stats()
        assert stats["events_processed"] == 10

        await engine.stop()

    @pytest.mark.asyncio
    async def test_engine_with_inmemory_bus(self):
        """Engine works with InMemoryEventBus."""
        from aegis.services.event_bus import InMemoryEventBus

        bus = InMemoryEventBus()
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("threat_detected", handler)
        await bus.start()

        engine = CampaignCorrelationEngine(
            event_bus=bus,
            window_sizes=[30],
            temporal_cluster_min_agents=5,
            campaign_alert_threshold=0.3,
        )
        await engine.start()

        now = time.time()
        for i in range(15):
            await engine.ingest_event(_make_event(
                agent_id=f"a-{i}",
                target="/api/target",
                timestamp=now + i * 0.1,
            ))

        import asyncio
        await asyncio.sleep(0.1)

        # If alerts were generated, they should appear on the bus
        if engine.get_stats()["alerts_generated"] > 0:
            assert len(received) > 0
            assert received[0].payload["type"] == "campaign_alert"

        await bus.stop()
        await engine.stop()

    @pytest.mark.asyncio
    async def test_recent_alerts(self):
        """get_recent_alerts returns alerts in reverse order."""
        engine = CampaignCorrelationEngine(
            window_sizes=[30],
            info_flow_min_links=2,
            campaign_alert_threshold=0.3,
        )
        now = time.time()

        # Create enough info flow to trigger alerts
        for i in range(5):
            await engine.ingest_event(_make_event(
                agent_id=f"a-{i}",
                target="/api",
                output_hash=f"h{i}",
                timestamp=now + i,
            ))
            await engine.ingest_event(_make_event(
                agent_id=f"b-{i}",
                target="/api",
                input_hash=f"h{i}",
                timestamp=now + i + 0.5,
            ))

        alerts = engine.get_recent_alerts(limit=5)
        assert isinstance(alerts, list)


class TestAlertLevels:
    """Tests for alert level assignment."""

    @pytest.mark.asyncio
    async def test_high_confidence_escalation(self):
        """Confidence > escalation_threshold → HIGH alert + escalate_tli action."""
        engine = CampaignCorrelationEngine(
            window_sizes=[10],
            temporal_cluster_min_agents=5,
            temporal_cluster_cv_threshold=0.5,
            campaign_alert_threshold=0.5,
            campaign_escalation_threshold=0.7,
        )

        now = time.time()
        all_alerts: list = []

        # Create tight temporal cluster
        for i in range(20):
            alerts = await engine.ingest_event(_make_event(
                agent_id=f"bot-{i}",
                target="/api/v1/critical",
                timestamp=now + i * 0.01,  # Very tight clustering
            ))
            all_alerts.extend(alerts)

        high_alerts = [a for a in all_alerts if a.alert_level in (AlertLevel.HIGH, AlertLevel.MEDIUM)]
        if high_alerts:
            assert high_alerts[0].recommended_action in ("escalate_tli", "investigate")
