"""
Campaign Fingerprint Detector — Statistical/heuristic detection of
coordinated multi-agent campaigns from aggregate event populations.

Five detection signatures:
1. Temporal Clustering: CV < threshold on inter-arrival times, N > min_agents
2. Systematic Enumeration: monotonic coverage of resource space
3. Parameter Fuzzing: Shannon entropy of parameter distributions > baseline + 2σ
4. Recon-to-Exploit: phase transition in action_type distribution
5. Information Flow: output_hash → input_hash chains across agents

Key architectural distinction: detection operates on aggregate statistics
over populations of agent behaviors, not on individual agent analysis.

ASSUMED-BREACH POSTURE: A sophisticated adversary could add jitter to
defeat temporal clustering or randomize enumeration order. The five
signatures provide defense-in-depth — defeating all five simultaneously
while maintaining campaign effectiveness is significantly harder.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.correlation.events import (
    ActionType,
    AgentActionEvent,
    FingerprintType,
)

logger = logging.getLogger(__name__)

# Recon action types
_RECON_ACTIONS = {ActionType.PROBE, ActionType.API_CALL, ActionType.DATA_ACCESS}
# Exploit action types
_EXPLOIT_ACTIONS = {ActionType.AUTH_ATTEMPT, ActionType.TOOL_INVOCATION, ActionType.SKILL_EXECUTION}


@dataclass
class FingerprintMatch:
    """A matched campaign fingerprint."""
    signature_type: FingerprintType
    confidence: float  # 0.0-1.0
    contributing_events: list[str]  # event IDs
    evidence: dict[str, Any] = field(default_factory=dict)


class FingerprintDetector:
    """Sliding-window statistical analysis on the event stream.

    Maintains overlapping time windows and runs all five detection
    signatures on the event population within each window.
    """

    def __init__(
        self,
        *,
        window_sizes: list[int] | None = None,
        temporal_cluster_cv_threshold: float = 0.5,
        temporal_cluster_min_agents: int = 10,
        enumeration_coverage_threshold: float = 0.6,
        fuzzing_entropy_std_threshold: float = 2.0,
        recon_exploit_transition_threshold: float = 0.7,
        info_flow_min_links: int = 3,
    ) -> None:
        self._window_sizes = window_sizes or [10, 30, 60]
        self._cv_threshold = temporal_cluster_cv_threshold
        self._min_agents = temporal_cluster_min_agents
        self._enum_coverage_threshold = enumeration_coverage_threshold
        self._entropy_std_threshold = fuzzing_entropy_std_threshold
        self._recon_exploit_threshold = recon_exploit_transition_threshold
        self._info_flow_min_links = info_flow_min_links

        # Event buffer (pruned periodically)
        self._events: list[AgentActionEvent] = []
        self._max_buffer = 10000

        # Known resource space (grows as resources are seen)
        self._known_resources: set[str] = set()

        # Entropy baseline tracking
        self._entropy_history: list[float] = []
        self._entropy_max_history = 100

    def add_event(self, event: AgentActionEvent) -> list[FingerprintMatch]:
        """Add an event and check all fingerprint signatures. Returns matches."""
        self._events.append(event)
        self._known_resources.add(event.target_resource)

        # Prune old events beyond max retention
        if len(self._events) > self._max_buffer:
            self._events = self._events[-self._max_buffer:]

        return self.detect()

    def detect(self) -> list[FingerprintMatch]:
        """Run all detection signatures on current event buffer."""
        now = time.time()
        matches: list[FingerprintMatch] = []

        for window_size in self._window_sizes:
            cutoff = now - window_size
            window_events = [e for e in self._events if e.timestamp >= cutoff]
            if len(window_events) < 3:
                continue

            match = self._detect_temporal_clustering(window_events, window_size)
            if match:
                matches.append(match)

            match = self._detect_systematic_enumeration(window_events, window_size)
            if match:
                matches.append(match)

            match = self._detect_parameter_fuzzing(window_events, window_size)
            if match:
                matches.append(match)

            match = self._detect_recon_to_exploit(window_events, window_size)
            if match:
                matches.append(match)

            match = self._detect_information_flow(window_events, window_size)
            if match:
                matches.append(match)

        # Deduplicate by signature type (keep highest confidence)
        best: dict[FingerprintType, FingerprintMatch] = {}
        for m in matches:
            if m.signature_type not in best or m.confidence > best[m.signature_type].confidence:
                best[m.signature_type] = m

        return list(best.values())

    def _detect_temporal_clustering(
        self, events: list[AgentActionEvent], window_size: int
    ) -> FingerprintMatch | None:
        """Detect many distinct agents targeting the same resource in a narrow window.

        Uses coefficient of variation on inter-arrival times.
        CV < threshold with N > min_agents = coordinated.
        """
        # Group by target resource
        by_target: dict[str, list[AgentActionEvent]] = defaultdict(list)
        for e in events:
            by_target[e.target_resource].append(e)

        for target, target_events in by_target.items():
            distinct_agents = {e.agent_id for e in target_events}
            if len(distinct_agents) < self._min_agents:
                continue

            # Sort by timestamp and compute inter-arrival times
            sorted_events = sorted(target_events, key=lambda e: e.timestamp)
            inter_arrivals = [
                sorted_events[i + 1].timestamp - sorted_events[i].timestamp
                for i in range(len(sorted_events) - 1)
            ]

            if not inter_arrivals:
                continue

            mean_ia = sum(inter_arrivals) / len(inter_arrivals)
            if mean_ia < 1e-9:
                # All simultaneous — clearly coordinated
                cv = 0.0
            else:
                std_ia = (sum((x - mean_ia) ** 2 for x in inter_arrivals) / len(inter_arrivals)) ** 0.5
                cv = std_ia / mean_ia

            if cv < self._cv_threshold:
                confidence = min(1.0, (1.0 - cv / self._cv_threshold) * 0.5 + 0.5)
                return FingerprintMatch(
                    signature_type=FingerprintType.TEMPORAL_CLUSTER,
                    confidence=confidence,
                    contributing_events=[e.event_id for e in target_events],
                    evidence={
                        "target": target,
                        "distinct_agents": len(distinct_agents),
                        "cv": round(cv, 4),
                        "mean_inter_arrival_seconds": round(mean_ia, 4),
                        "window_size": window_size,
                    },
                )
        return None

    def _detect_systematic_enumeration(
        self, events: list[AgentActionEvent], window_size: int
    ) -> FingerprintMatch | None:
        """Detect sequential probing of resources covering the known resource space.

        Monotonically increasing coverage from distinct agents.
        """
        if not self._known_resources or len(self._known_resources) < 3:
            return None

        # Resources accessed by distinct agents in this window
        agents_per_resource: dict[str, set[str]] = defaultdict(set)
        for e in events:
            agents_per_resource[e.target_resource].add(e.agent_id)

        # Coverage: fraction of known resources hit by at least 1 distinct agent
        resources_hit = set(agents_per_resource.keys())
        coverage = len(resources_hit & self._known_resources) / len(self._known_resources)

        # Need at least a few distinct agents doing the scanning
        all_agents = set()
        for agents in agents_per_resource.values():
            all_agents.update(agents)

        if coverage >= self._enum_coverage_threshold and len(all_agents) >= 3:
            confidence = min(1.0, coverage)
            return FingerprintMatch(
                signature_type=FingerprintType.SYSTEMATIC_ENUMERATION,
                confidence=confidence,
                contributing_events=[e.event_id for e in events],
                evidence={
                    "coverage": round(coverage, 4),
                    "resources_hit": len(resources_hit),
                    "known_resources": len(self._known_resources),
                    "distinct_agents": len(all_agents),
                    "window_size": window_size,
                },
            )
        return None

    def _detect_parameter_fuzzing(
        self, events: list[AgentActionEvent], window_size: int
    ) -> FingerprintMatch | None:
        """Detect multiple agents testing different parameter values on same endpoint.

        Shannon entropy of parameter distributions > baseline + 2σ.
        """
        # Group by target resource
        by_target: dict[str, list[AgentActionEvent]] = defaultdict(list)
        for e in events:
            if e.parameter_value:
                by_target[e.target_resource].append(e)

        for target, target_events in by_target.items():
            distinct_agents = {e.agent_id for e in target_events}
            if len(distinct_agents) < 3:
                continue

            # Get distinct parameter values
            param_values = [e.parameter_value for e in target_events]
            distinct_values = set(param_values)

            if len(distinct_values) < 3:
                continue

            # Compute Shannon entropy
            entropy = self._shannon_entropy(param_values)

            # Check against baseline
            self._entropy_history.append(entropy)
            if len(self._entropy_history) > self._entropy_max_history:
                self._entropy_history = self._entropy_history[-self._entropy_max_history:]

            if len(self._entropy_history) >= 5:
                mean_entropy = sum(self._entropy_history) / len(self._entropy_history)
                std_entropy = (
                    sum((x - mean_entropy) ** 2 for x in self._entropy_history)
                    / len(self._entropy_history)
                ) ** 0.5
                threshold = mean_entropy + self._entropy_std_threshold * max(std_entropy, 0.1)
            else:
                # Not enough history — use absolute threshold
                threshold = 2.0

            if entropy > threshold and len(distinct_values) >= len(distinct_agents):
                confidence = min(1.0, 0.6 + 0.1 * len(distinct_agents))
                return FingerprintMatch(
                    signature_type=FingerprintType.PARAMETER_FUZZING,
                    confidence=confidence,
                    contributing_events=[e.event_id for e in target_events],
                    evidence={
                        "target": target,
                        "entropy": round(entropy, 4),
                        "threshold": round(threshold, 4),
                        "distinct_values": len(distinct_values),
                        "distinct_agents": len(distinct_agents),
                        "window_size": window_size,
                    },
                )
        return None

    def _detect_recon_to_exploit(
        self, events: list[AgentActionEvent], window_size: int
    ) -> FingerprintMatch | None:
        """Detect phase transition from recon to exploit actions.

        Agents transitioning from information-gathering to exploitation
        in a coordinated cadence.
        """
        if len(events) < 5:
            return None

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        midpoint = len(sorted_events) // 2

        first_half = sorted_events[:midpoint]
        second_half = sorted_events[midpoint:]

        # Compute recon fraction in each half
        recon_first = sum(1 for e in first_half if e.action_type in _RECON_ACTIONS)
        exploit_second = sum(1 for e in second_half if e.action_type in _EXPLOIT_ACTIONS)

        recon_ratio_first = recon_first / len(first_half) if first_half else 0
        exploit_ratio_second = exploit_second / len(second_half) if second_half else 0

        # Phase transition: high recon in first half, high exploit in second
        transition_score = (recon_ratio_first + exploit_ratio_second) / 2.0

        if transition_score >= self._recon_exploit_threshold:
            # Need multiple distinct agents involved
            recon_agents = {e.agent_id for e in first_half if e.action_type in _RECON_ACTIONS}
            exploit_agents = {e.agent_id for e in second_half if e.action_type in _EXPLOIT_ACTIONS}

            if len(recon_agents) >= 2 and len(exploit_agents) >= 2:
                confidence = min(1.0, transition_score)
                return FingerprintMatch(
                    signature_type=FingerprintType.RECON_TO_EXPLOIT,
                    confidence=confidence,
                    contributing_events=[e.event_id for e in sorted_events],
                    evidence={
                        "recon_ratio_first_half": round(recon_ratio_first, 4),
                        "exploit_ratio_second_half": round(exploit_ratio_second, 4),
                        "transition_score": round(transition_score, 4),
                        "recon_agents": len(recon_agents),
                        "exploit_agents": len(exploit_agents),
                        "window_size": window_size,
                    },
                )
        return None

    def _detect_information_flow(
        self, events: list[AgentActionEvent], window_size: int
    ) -> FingerprintMatch | None:
        """Detect output→input hash chains across agents (coordinator-solver pattern)."""
        # Build hash chains
        output_map: dict[str, AgentActionEvent] = {}
        chains: list[tuple[str, str]] = []  # (source_event_id, target_event_id)

        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for e in sorted_events:
            if e.output_hash:
                output_map[e.output_hash] = e
            if e.input_hash and e.input_hash in output_map:
                source = output_map[e.input_hash]
                if source.agent_id != e.agent_id:
                    chains.append((source.event_id, e.event_id))

        if len(chains) >= self._info_flow_min_links:
            # Collect all contributing events
            contributing = set()
            for src, tgt in chains:
                contributing.add(src)
                contributing.add(tgt)

            # Count distinct agent pairs
            agent_pairs = set()
            for src, tgt in chains:
                src_event = next((e for e in events if e.event_id == src), None)
                tgt_event = next((e for e in events if e.event_id == tgt), None)
                if src_event and tgt_event:
                    agent_pairs.add((src_event.agent_id, tgt_event.agent_id))

            confidence = min(1.0, 0.5 + 0.1 * len(chains))
            return FingerprintMatch(
                signature_type=FingerprintType.INFORMATION_FLOW,
                confidence=confidence,
                contributing_events=sorted(contributing),
                evidence={
                    "chain_count": len(chains),
                    "agent_pairs": len(agent_pairs),
                    "distinct_events": len(contributing),
                    "window_size": window_size,
                },
            )
        return None

    @staticmethod
    def _shannon_entropy(values: list[str]) -> float:
        """Compute Shannon entropy of a list of values."""
        if not values:
            return 0.0
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        total = len(values)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
