"""
Campaign Graph — Directed graph reconstructing causal relationships
between agent actions across agents.

Lightweight adjacency-list implementation. No external graph library
dependency. Nodes are AgentActionEvents, edges represent causal or
informational relationships.

ASSUMED-BREACH POSTURE: The graph is ephemeral (in-memory, pruned after
retention window). Alerts generated from graph analysis are persisted to
the audit log independently. A compromised graph that drops nodes would
reduce campaign visibility — the fingerprint detector provides independent
statistical detection that does not depend on graph completeness.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.correlation.events import AgentActionEvent, ActionType

logger = logging.getLogger(__name__)

# Recon action types (information gathering)
_RECON_ACTIONS = {ActionType.PROBE, ActionType.API_CALL, ActionType.DATA_ACCESS}
# Exploit action types (active exploitation)
_EXPLOIT_ACTIONS = {ActionType.AUTH_ATTEMPT, ActionType.TOOL_INVOCATION, ActionType.SKILL_EXECUTION}


class EdgeType(str, enum.Enum):
    """Types of relationships between agent actions."""
    TEMPORAL = "temporal"
    TARGET_CORRELATION = "target_correlation"
    INFORMATION_FLOW = "information_flow"
    TECHNIQUE_PROGRESSION = "technique_progression"


@dataclass
class GraphEdge:
    """An edge in the campaign graph."""
    source_event_id: str
    target_event_id: str
    edge_type: EdgeType
    weight: float = 1.0


class CampaignGraph:
    """Directed graph of agent actions with causal/informational edges.

    Nodes are AgentActionEvents. Edges are classified by type:
    - TEMPORAL: same target within time window
    - TARGET_CORRELATION: same resource from different agents
    - INFORMATION_FLOW: output_hash → input_hash chain
    - TECHNIQUE_PROGRESSION: recon → exploit sequence

    Pruning removes events older than retention_seconds to bound growth.
    """

    def __init__(
        self,
        *,
        temporal_window_seconds: float = 30.0,
        retention_seconds: int = 300,
    ) -> None:
        self._temporal_window = temporal_window_seconds
        self._retention_seconds = retention_seconds

        # Node storage
        self._nodes: dict[str, AgentActionEvent] = {}
        # Adjacency list: event_id -> list of edges
        self._edges: dict[str, list[GraphEdge]] = {}
        # Reverse adjacency for traversal
        self._reverse_edges: dict[str, list[GraphEdge]] = {}
        # Index: output_hash -> event_id (for info flow detection)
        self._output_index: dict[str, str] = {}
        # Index: target_resource -> list of event_ids (recent)
        self._target_index: dict[str, list[str]] = {}

    def add_event(self, event: AgentActionEvent) -> list[GraphEdge]:
        """Add an event to the graph and create edges. Returns new edges."""
        self._prune()

        eid = event.event_id
        self._nodes[eid] = event
        self._edges.setdefault(eid, [])
        self._reverse_edges.setdefault(eid, [])

        new_edges: list[GraphEdge] = []

        # Information flow: does this event's input_hash match a prior output?
        if event.input_hash and event.input_hash in self._output_index:
            source_id = self._output_index[event.input_hash]
            if source_id in self._nodes and source_id != eid:
                edge = GraphEdge(
                    source_event_id=source_id,
                    target_event_id=eid,
                    edge_type=EdgeType.INFORMATION_FLOW,
                    weight=1.5,
                )
                self._edges[source_id].append(edge)
                self._reverse_edges[eid].append(edge)
                new_edges.append(edge)

        # Register output hash for future info flow detection
        if event.output_hash:
            self._output_index[event.output_hash] = eid

        # Target correlation: same resource from different agents
        target = event.target_resource
        target_events = self._target_index.get(target, [])
        for prev_eid in target_events:
            if prev_eid not in self._nodes:
                continue
            prev = self._nodes[prev_eid]
            if prev.agent_id == event.agent_id:
                continue

            time_diff = abs(event.timestamp - prev.timestamp)

            # Temporal edge: same target within window
            if time_diff <= self._temporal_window:
                edge = GraphEdge(
                    source_event_id=prev_eid,
                    target_event_id=eid,
                    edge_type=EdgeType.TEMPORAL,
                )
                self._edges[prev_eid].append(edge)
                self._reverse_edges[eid].append(edge)
                new_edges.append(edge)

            # Target correlation
            edge = GraphEdge(
                source_event_id=prev_eid,
                target_event_id=eid,
                edge_type=EdgeType.TARGET_CORRELATION,
            )
            self._edges[prev_eid].append(edge)
            self._reverse_edges[eid].append(edge)
            new_edges.append(edge)

            # Technique progression: recon -> exploit on same target
            if (prev.action_type in _RECON_ACTIONS
                    and event.action_type in _EXPLOIT_ACTIONS
                    and prev.timestamp <= event.timestamp):
                edge = GraphEdge(
                    source_event_id=prev_eid,
                    target_event_id=eid,
                    edge_type=EdgeType.TECHNIQUE_PROGRESSION,
                    weight=2.0,
                )
                self._edges[prev_eid].append(edge)
                self._reverse_edges[eid].append(edge)
                new_edges.append(edge)

        self._target_index.setdefault(target, []).append(eid)

        return new_edges

    def get_campaign_subgraphs(self) -> list[list[str]]:
        """Get connected components as lists of event IDs (campaigns)."""
        visited: set[str] = set()
        components: list[list[str]] = []

        for node_id in self._nodes:
            if node_id in visited:
                continue
            component: list[str] = []
            stack = [node_id]
            while stack:
                nid = stack.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                component.append(nid)
                # Follow forward edges
                for edge in self._edges.get(nid, []):
                    if edge.target_event_id not in visited:
                        stack.append(edge.target_event_id)
                # Follow reverse edges
                for edge in self._reverse_edges.get(nid, []):
                    if edge.source_event_id not in visited:
                        stack.append(edge.source_event_id)
            if len(component) > 1:
                components.append(component)

        return components

    def get_campaign_summary(self, event_ids: list[str]) -> dict[str, Any]:
        """Get summary of a campaign subgraph."""
        agents: set[str] = set()
        targets: set[str] = set()
        edge_types: dict[str, int] = {}
        action_types: dict[str, int] = {}
        timestamps: list[float] = []

        for eid in event_ids:
            node = self._nodes.get(eid)
            if not node:
                continue
            agents.add(node.agent_id)
            targets.add(node.target_resource)
            action_types[node.action_type.value] = action_types.get(node.action_type.value, 0) + 1
            timestamps.append(node.timestamp)

        for eid in event_ids:
            for edge in self._edges.get(eid, []):
                if edge.target_event_id in event_ids:
                    edge_types[edge.edge_type.value] = edge_types.get(edge.edge_type.value, 0) + 1

        return {
            "node_count": len(event_ids),
            "agent_count": len(agents),
            "agents": sorted(agents),
            "target_count": len(targets),
            "targets": sorted(targets),
            "edge_types": edge_types,
            "action_types": action_types,
            "time_span_seconds": (max(timestamps) - min(timestamps)) if timestamps else 0,
        }

    def to_serializable(self) -> dict[str, Any]:
        """Serialize graph state for CampaignAlert inclusion."""
        return {
            "node_count": len(self._nodes),
            "edge_count": sum(len(edges) for edges in self._edges.values()),
            "campaigns": [
                self.get_campaign_summary(component)
                for component in self.get_campaign_subgraphs()
            ],
        }

    def get_node(self, event_id: str) -> AgentActionEvent | None:
        """Get a node by event ID."""
        return self._nodes.get(event_id)

    def get_edges(self, event_id: str) -> list[GraphEdge]:
        """Get forward edges from a node."""
        return self._edges.get(event_id, [])

    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    def _prune(self) -> None:
        """Remove events older than retention window."""
        cutoff = time.time() - self._retention_seconds
        expired = [
            eid for eid, event in self._nodes.items()
            if event.timestamp < cutoff
        ]
        for eid in expired:
            del self._nodes[eid]
            # Clean forward edges
            if eid in self._edges:
                del self._edges[eid]
            # Clean reverse edges
            if eid in self._reverse_edges:
                del self._reverse_edges[eid]
            # Clean from other nodes' edge lists
            for other_edges in self._edges.values():
                other_edges[:] = [e for e in other_edges if e.target_event_id != eid]
            for other_edges in self._reverse_edges.values():
                other_edges[:] = [e for e in other_edges if e.source_event_id != eid]
            # Clean from target index
            for target, eids in self._target_index.items():
                if eid in eids:
                    eids.remove(eid)
            # Clean from output index
            self._output_index = {
                h: e for h, e in self._output_index.items() if e != eid
            }
