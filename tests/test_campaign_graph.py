"""
Tests for campaign graph construction, connected components,
edge classification, pruning, and serialization.
"""

from __future__ import annotations

import time

import pytest

from aegis.layers.correlation.events import ActionType, AgentActionEvent
from aegis.layers.correlation.campaign_graph import CampaignGraph, EdgeType


def _make_event(
    agent_id: str = "agent-1",
    action_type: ActionType = ActionType.API_CALL,
    target: str = "/api/v1/users",
    timestamp: float | None = None,
    input_hash: str = "",
    output_hash: str = "",
) -> AgentActionEvent:
    return AgentActionEvent(
        agent_id=agent_id,
        action_type=action_type,
        target_resource=target,
        timestamp=timestamp or time.time(),
        input_hash=input_hash,
        output_hash=output_hash,
    )


class TestGraphConstruction:
    """Tests for graph node and edge creation."""

    def test_add_single_event(self):
        """Adding one event creates a node with no edges."""
        graph = CampaignGraph()
        event = _make_event()
        edges = graph.add_event(event)
        assert graph.node_count() == 1
        assert len(edges) == 0

    def test_target_correlation_edge(self):
        """Two events from different agents on same target create TARGET_CORRELATION edge."""
        graph = CampaignGraph()
        now = time.time()
        e1 = _make_event(agent_id="a1", target="/api", timestamp=now)
        e2 = _make_event(agent_id="a2", target="/api", timestamp=now + 1)

        graph.add_event(e1)
        edges = graph.add_event(e2)

        target_edges = [e for e in edges if e.edge_type == EdgeType.TARGET_CORRELATION]
        assert len(target_edges) >= 1

    def test_temporal_edge(self):
        """Events on same target within window create TEMPORAL edge."""
        graph = CampaignGraph(temporal_window_seconds=10.0)
        now = time.time()
        e1 = _make_event(agent_id="a1", target="/api", timestamp=now)
        e2 = _make_event(agent_id="a2", target="/api", timestamp=now + 5)

        graph.add_event(e1)
        edges = graph.add_event(e2)

        temporal_edges = [e for e in edges if e.edge_type == EdgeType.TEMPORAL]
        assert len(temporal_edges) >= 1

    def test_information_flow_edge(self):
        """Output hash matching input hash creates INFORMATION_FLOW edge."""
        graph = CampaignGraph()
        now = time.time()
        e1 = _make_event(agent_id="a1", output_hash="hash_x", timestamp=now)
        e2 = _make_event(agent_id="a2", input_hash="hash_x", timestamp=now + 1)

        graph.add_event(e1)
        edges = graph.add_event(e2)

        flow_edges = [e for e in edges if e.edge_type == EdgeType.INFORMATION_FLOW]
        assert len(flow_edges) == 1

    def test_technique_progression_edge(self):
        """PROBE → AUTH_ATTEMPT on same target creates TECHNIQUE_PROGRESSION edge."""
        graph = CampaignGraph()
        now = time.time()
        e1 = _make_event(
            agent_id="recon", action_type=ActionType.PROBE,
            target="/api", timestamp=now,
        )
        e2 = _make_event(
            agent_id="exploit", action_type=ActionType.AUTH_ATTEMPT,
            target="/api", timestamp=now + 1,
        )

        graph.add_event(e1)
        edges = graph.add_event(e2)

        prog_edges = [e for e in edges if e.edge_type == EdgeType.TECHNIQUE_PROGRESSION]
        assert len(prog_edges) >= 1

    def test_no_self_edges(self):
        """Same agent hitting same target should not create target_correlation edges."""
        graph = CampaignGraph()
        now = time.time()
        e1 = _make_event(agent_id="a1", target="/api", timestamp=now)
        e2 = _make_event(agent_id="a1", target="/api", timestamp=now + 1)

        graph.add_event(e1)
        edges = graph.add_event(e2)

        # No edges to self
        target_edges = [e for e in edges if e.edge_type == EdgeType.TARGET_CORRELATION]
        assert len(target_edges) == 0


class TestCampaignSubgraphs:
    """Tests for connected component extraction."""

    def test_single_campaign(self):
        """Connected events form one campaign."""
        graph = CampaignGraph()
        now = time.time()

        for i in range(5):
            graph.add_event(_make_event(
                agent_id=f"a-{i}", target="/api", timestamp=now,
            ))

        components = graph.get_campaign_subgraphs()
        assert len(components) == 1
        assert len(components[0]) == 5

    def test_disjoint_campaigns(self):
        """Events on different targets form separate campaigns."""
        graph = CampaignGraph(temporal_window_seconds=5.0)
        now = time.time()

        # Campaign 1: agents hitting /api/a
        for i in range(3):
            graph.add_event(_make_event(
                agent_id=f"camp1-{i}", target="/api/a", timestamp=now,
            ))

        # Campaign 2: agents hitting /api/b (different target, no overlap)
        for i in range(3):
            graph.add_event(_make_event(
                agent_id=f"camp2-{i}", target="/api/b", timestamp=now,
            ))

        components = graph.get_campaign_subgraphs()
        assert len(components) == 2

    def test_no_campaigns_single_events(self):
        """Isolated events (no edges) form no campaigns."""
        graph = CampaignGraph()
        now = time.time()

        # Each event on a different target from the same agent
        for i in range(3):
            graph.add_event(_make_event(
                agent_id="solo", target=f"/api/{i}", timestamp=now + i,
            ))

        components = graph.get_campaign_subgraphs()
        # No multi-node components (each node is isolated)
        assert len(components) == 0

    def test_campaign_summary(self):
        """Campaign summary contains correct metadata."""
        graph = CampaignGraph()
        now = time.time()

        events = []
        for i in range(4):
            e = _make_event(agent_id=f"a-{i}", target="/api/v1/users", timestamp=now)
            graph.add_event(e)
            events.append(e)

        components = graph.get_campaign_subgraphs()
        assert len(components) == 1
        summary = graph.get_campaign_summary(components[0])
        assert summary["agent_count"] == 4
        assert summary["target_count"] == 1
        assert summary["node_count"] == 4


class TestGraphPruning:
    """Tests for event pruning and retention."""

    def test_prune_old_events(self):
        """Events older than retention window are pruned."""
        graph = CampaignGraph(retention_seconds=10)
        old_time = time.time() - 20  # 20 seconds ago

        e1 = _make_event(agent_id="old", target="/api", timestamp=old_time)
        graph.add_event(e1)
        assert graph.node_count() == 1

        # Adding a new event triggers pruning
        e2 = _make_event(agent_id="new", target="/api2")
        graph.add_event(e2)

        # Old event should be pruned
        assert graph.get_node(e1.event_id) is None
        assert graph.node_count() == 1

    def test_recent_events_not_pruned(self):
        """Events within retention window survive pruning."""
        graph = CampaignGraph(retention_seconds=300)

        e1 = _make_event(agent_id="a1", target="/api")
        graph.add_event(e1)
        e2 = _make_event(agent_id="a2", target="/api")
        graph.add_event(e2)

        assert graph.node_count() == 2
        assert graph.get_node(e1.event_id) is not None


class TestGraphSerialization:
    """Tests for graph serialization."""

    def test_to_serializable(self):
        """to_serializable produces a JSON-compatible dict."""
        graph = CampaignGraph()
        now = time.time()

        for i in range(3):
            graph.add_event(_make_event(
                agent_id=f"a-{i}", target="/api", timestamp=now,
            ))

        data = graph.to_serializable()
        assert "node_count" in data
        assert "edge_count" in data
        assert "campaigns" in data
        assert data["node_count"] == 3

    def test_empty_graph_serializable(self):
        """Empty graph serializes without error."""
        graph = CampaignGraph()
        data = graph.to_serializable()
        assert data["node_count"] == 0
        assert data["campaigns"] == []
