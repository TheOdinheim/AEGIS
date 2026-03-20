"""
Intent Feature Extractor — Extracts statistical features from sliding
windows of AgentActionEvent sequences for intent classification.

This is a sequence classification problem: it classifies what a series
of actions is trying to accomplish, not what any individual text says.

Feature set (8 dimensions):
1. Target Entropy — breadth of resource targeting
2. Action Type Distribution — proportions of each action type
3. Technique Diversity — unique (action_type, target) pairs / total
4. Temporal Regularity — CV of inter-event intervals
5. Progression Score — recon→exploit phase transition
6. Information Flow Density — hash chain density
7. Agent Cardinality — distinct agent count
8. Stylistic Discontinuity — model alloy detection (LLM swap frequency)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from aegis.layers.correlation.events import (
    ActionType,
    AgentActionEvent,
)

# Recon and exploit action type sets (same as fingerprint_detector)
_RECON_ACTIONS = {ActionType.PROBE, ActionType.API_CALL, ActionType.DATA_ACCESS}
_EXPLOIT_ACTIONS = {ActionType.AUTH_ATTEMPT, ActionType.TOOL_INVOCATION, ActionType.SKILL_EXECUTION}


class IntentFeatureVector(BaseModel):
    """Statistical features extracted from an event window."""
    target_entropy: float = 0.0
    action_type_ratios: dict[str, float] = Field(default_factory=dict)
    technique_diversity: float = 0.0
    temporal_regularity: float = 0.0
    progression_score: float = 0.0
    info_flow_density: float = 0.0
    agent_cardinality: int = 0
    stylistic_discontinuity: float = 0.0
    window_event_count: int = 0


class IntentFeatureExtractor:
    """Extracts statistical features from event sequences."""

    def __init__(
        self,
        *,
        window_size: int = 50,
        window_step: int = 25,
    ) -> None:
        self._window_size = window_size
        self._window_step = window_step

    def extract(self, events: list[AgentActionEvent]) -> IntentFeatureVector:
        """Extract features from a single window of events."""
        if not events:
            return IntentFeatureVector()

        return IntentFeatureVector(
            target_entropy=self._compute_target_entropy(events),
            action_type_ratios=self._compute_action_type_ratios(events),
            technique_diversity=self._compute_technique_diversity(events),
            temporal_regularity=self._compute_temporal_regularity(events),
            progression_score=self._compute_progression_score(events),
            info_flow_density=self._compute_info_flow_density(events),
            agent_cardinality=len({e.agent_id for e in events}),
            stylistic_discontinuity=self._compute_stylistic_discontinuity(events),
            window_event_count=len(events),
        )

    def extract_sliding(self, events: list[AgentActionEvent]) -> list[IntentFeatureVector]:
        """Extract feature vectors for all sliding windows."""
        if not events or len(events) < self._window_size:
            # If fewer events than window size, return single extraction
            if events:
                return [self.extract(events)]
            return []

        vectors: list[IntentFeatureVector] = []
        start = 0
        while start + self._window_size <= len(events):
            window = events[start:start + self._window_size]
            vectors.append(self.extract(window))
            start += self._window_step

        # Handle remaining events if they form a partial window
        if start < len(events) and start > 0:
            # Already covered by last full window
            pass

        return vectors

    @staticmethod
    def _compute_target_entropy(events: list[AgentActionEvent]) -> float:
        """Shannon entropy of target_resource distribution."""
        if not events:
            return 0.0
        counts: dict[str, int] = {}
        for e in events:
            counts[e.target_resource] = counts.get(e.target_resource, 0) + 1
        total = len(events)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _compute_action_type_ratios(events: list[AgentActionEvent]) -> dict[str, float]:
        """Proportion of each ActionType in the window."""
        if not events:
            return {}
        counts: dict[str, int] = {}
        for e in events:
            key = e.action_type.value
            counts[key] = counts.get(key, 0) + 1
        total = len(events)
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def _compute_technique_diversity(events: list[AgentActionEvent]) -> float:
        """Unique (action_type, target) pairs / total events."""
        if not events:
            return 0.0
        pairs = {(e.action_type.value, e.target_resource) for e in events}
        return len(pairs) / len(events)

    @staticmethod
    def _compute_temporal_regularity(events: list[AgentActionEvent]) -> float:
        """CV of inter-event time intervals."""
        if len(events) < 2:
            return 0.0
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        intervals = [
            sorted_events[i + 1].timestamp - sorted_events[i].timestamp
            for i in range(len(sorted_events) - 1)
        ]
        if not intervals:
            return 0.0
        mean = sum(intervals) / len(intervals)
        if mean < 1e-9:
            return 0.0  # All simultaneous
        std = (sum((x - mean) ** 2 for x in intervals) / len(intervals)) ** 0.5
        return std / mean

    @staticmethod
    def _compute_progression_score(events: list[AgentActionEvent]) -> float:
        """Directional change: exploit_ratio_second_half - exploit_ratio_first_half."""
        if len(events) < 2:
            return 0.0
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        mid = len(sorted_events) // 2
        first_half = sorted_events[:mid]
        second_half = sorted_events[mid:]

        exploit_first = sum(1 for e in first_half if e.action_type in _EXPLOIT_ACTIONS) / len(first_half) if first_half else 0
        exploit_second = sum(1 for e in second_half if e.action_type in _EXPLOIT_ACTIONS) / len(second_half) if second_half else 0

        return exploit_second - exploit_first

    @staticmethod
    def _compute_info_flow_density(events: list[AgentActionEvent]) -> float:
        """Proportion of events whose input_hash matches a prior output_hash."""
        if not events:
            return 0.0
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        output_hashes: set[str] = set()
        matches = 0
        for e in sorted_events:
            if e.input_hash and e.input_hash in output_hashes:
                matches += 1
            if e.output_hash:
                output_hashes.add(e.output_hash)
        return matches / len(events)

    @staticmethod
    def _compute_stylistic_discontinuity(events: list[AgentActionEvent]) -> float:
        """Frequency of source_model transitions on same target within time window.

        This is the alloy detection feature — detects mid-sequence LLM swaps.
        """
        # Group events by target resource, sorted by time
        by_target: dict[str, list[AgentActionEvent]] = defaultdict(list)
        for e in events:
            if e.source_model:
                by_target[e.target_resource].append(e)

        total_pairs = 0
        transitions = 0

        for target, target_events in by_target.items():
            if len(target_events) < 2:
                continue
            sorted_events = sorted(target_events, key=lambda e: e.timestamp)
            for i in range(len(sorted_events) - 1):
                total_pairs += 1
                if sorted_events[i].source_model != sorted_events[i + 1].source_model:
                    transitions += 1

        if total_pairs == 0:
            return 0.0
        return transitions / total_pairs
