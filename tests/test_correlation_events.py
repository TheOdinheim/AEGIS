"""
Tests for campaign correlation event schema and event bus integration.
"""

from __future__ import annotations

import time

import pytest

from aegis.layers.correlation.events import (
    ActionType,
    AgentActionEvent,
    AlertLevel,
    CampaignAlert,
    FingerprintType,
)


class TestAgentActionEvent:
    """Tests for AgentActionEvent model."""

    def test_create_minimal(self):
        """Create event with minimal required fields."""
        event = AgentActionEvent(
            agent_id="agent-1",
            action_type=ActionType.API_CALL,
            target_resource="/api/v1/users",
        )
        assert event.agent_id == "agent-1"
        assert event.action_type == ActionType.API_CALL
        assert event.target_resource == "/api/v1/users"
        assert event.event_id  # Auto-generated
        assert event.timestamp > 0
        assert event.tenant_id == "default"

    def test_create_full(self):
        """Create event with all fields."""
        event = AgentActionEvent(
            agent_id="solver-42",
            timestamp=1000.0,
            action_type=ActionType.TOOL_INVOCATION,
            target_resource="/api/v1/exec",
            input_hash="abc123",
            output_hash="def456",
            session_id="sess-1",
            tenant_id="tenant-a",
            source_model="gpt-4",
            latency_ms=50.0,
            metadata={"key": "value"},
            parameter_value="test_param",
        )
        assert event.input_hash == "abc123"
        assert event.output_hash == "def456"
        assert event.source_model == "gpt-4"
        assert event.parameter_value == "test_param"

    def test_serialization_roundtrip(self):
        """Event serializes and deserializes correctly."""
        event = AgentActionEvent(
            agent_id="agent-1",
            action_type=ActionType.PROBE,
            target_resource="/api/v1/info",
            metadata={"scan": True},
        )
        data = event.model_dump()
        restored = AgentActionEvent.model_validate(data)
        assert restored.agent_id == event.agent_id
        assert restored.action_type == event.action_type
        assert restored.metadata == event.metadata

    def test_action_types_complete(self):
        """All action types are valid enum values."""
        expected = {
            "api_call", "data_access", "tool_invocation",
            "skill_execution", "output_generation", "probe", "auth_attempt",
        }
        actual = {a.value for a in ActionType}
        assert actual == expected

    def test_json_serialization(self):
        """Event serializes to JSON string."""
        event = AgentActionEvent(
            agent_id="a1",
            action_type=ActionType.DATA_ACCESS,
            target_resource="/data",
        )
        json_str = event.model_dump_json()
        assert "a1" in json_str
        assert "data_access" in json_str


class TestCampaignAlert:
    """Tests for CampaignAlert model."""

    def test_create_alert(self):
        """Create campaign alert with required fields."""
        alert = CampaignAlert(
            confidence=0.85,
            alert_level=AlertLevel.HIGH,
            contributing_agent_ids=["agent-1", "agent-2", "agent-3"],
            fingerprint_type=FingerprintType.TEMPORAL_CLUSTER,
            recommended_action="escalate_tli",
        )
        assert alert.campaign_id
        assert alert.confidence == 0.85
        assert alert.alert_level == AlertLevel.HIGH
        assert len(alert.contributing_agent_ids) == 3
        assert alert.fingerprint_type == FingerprintType.TEMPORAL_CLUSTER

    def test_alert_serialization(self):
        """Alert serializes correctly."""
        alert = CampaignAlert(
            confidence=0.9,
            alert_level=AlertLevel.CRITICAL,
            contributing_agent_ids=["a1"],
            fingerprint_type=FingerprintType.INFORMATION_FLOW,
            evidence={"chain_count": 5},
        )
        data = alert.model_dump()
        assert data["confidence"] == 0.9
        assert data["evidence"]["chain_count"] == 5

    def test_fingerprint_types_complete(self):
        """All fingerprint types are valid."""
        expected = {
            "temporal_cluster", "systematic_enumeration",
            "parameter_fuzzing", "recon_to_exploit", "information_flow",
        }
        actual = {f.value for f in FingerprintType}
        assert actual == expected

    def test_alert_levels(self):
        """Alert levels map correctly."""
        assert AlertLevel.LOW.value == "low"
        assert AlertLevel.MEDIUM.value == "medium"
        assert AlertLevel.HIGH.value == "high"
        assert AlertLevel.CRITICAL.value == "critical"


class TestEventBusIntegration:
    """Tests for event bus publish/subscribe with campaign events."""

    @pytest.mark.asyncio
    async def test_publish_subscribe_inmemory(self):
        """InMemoryEventBus publish/subscribe works with campaign events."""
        from aegis.services.event_bus import InMemoryEventBus

        bus = InMemoryEventBus()
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("threat_detected", handler)
        await bus.start()

        await bus.publish("threat_detected", {
            "type": "campaign_alert",
            "campaign_id": "camp-1",
            "fingerprint_type": "temporal_cluster",
            "confidence": 0.85,
        })

        # Allow event to be processed
        import asyncio
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].payload["type"] == "campaign_alert"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_event_bus_graceful_without_redis(self):
        """InMemoryEventBus works when Redis is unavailable (graceful fallback)."""
        from aegis.services.event_bus import InMemoryEventBus

        bus = InMemoryEventBus()
        await bus.start()

        # Should not raise
        await bus.publish("threat_detected", {"type": "test"})
        assert bus.backend == "memory"

        await bus.stop()
