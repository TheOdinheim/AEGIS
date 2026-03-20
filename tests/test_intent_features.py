"""
Tests for intent feature extraction from agent action event sequences.
"""

from __future__ import annotations

import math
import random
import time

import pytest

from aegis.layers.correlation.events import ActionType, AgentActionEvent
from aegis.layers.correlation.intent_features import IntentFeatureExtractor, IntentFeatureVector


def _make_event(
    agent_id: str = "agent-1",
    action_type: ActionType = ActionType.API_CALL,
    target: str = "/api/v1/users",
    timestamp: float | None = None,
    input_hash: str = "",
    output_hash: str = "",
    source_model: str | None = None,
    parameter_value: str = "",
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
    )


class TestTargetEntropy:
    """Tests for target entropy feature."""

    def test_uniform_distribution(self):
        """10 unique targets uniformly → entropy ≈ log2(10)."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", target=f"/api/ep-{i}", timestamp=now + i)
            for i in range(10)
        ]
        features = extractor.extract(events)
        assert abs(features.target_entropy - math.log2(10)) < 0.01

    def test_single_target(self):
        """All events to one target → entropy ≈ 0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", target="/api/single", timestamp=now + i)
            for i in range(10)
        ]
        features = extractor.extract(events)
        assert features.target_entropy == pytest.approx(0.0, abs=0.01)


class TestActionTypeDistribution:
    """Tests for action type ratio feature."""

    def test_known_proportions(self):
        """Generate events with known proportions, verify ratios."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = []
        # 6 PROBE, 4 AUTH_ATTEMPT = 60% probe, 40% auth
        for i in range(6):
            events.append(_make_event(agent_id=f"a-{i}", action_type=ActionType.PROBE, timestamp=now + i))
        for i in range(4):
            events.append(_make_event(agent_id=f"b-{i}", action_type=ActionType.AUTH_ATTEMPT, timestamp=now + 10 + i))

        features = extractor.extract(events)
        assert features.action_type_ratios["probe"] == pytest.approx(0.6, abs=0.01)
        assert features.action_type_ratios["auth_attempt"] == pytest.approx(0.4, abs=0.01)


class TestTechniqueDiversity:
    """Tests for technique diversity feature."""

    def test_all_unique_pairs(self):
        """20 events with 20 unique (action, target) → diversity = 1.0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        actions = list(ActionType)
        events = [
            _make_event(
                agent_id=f"a-{i}",
                action_type=actions[i % len(actions)],
                target=f"/api/ep-{i}",
                timestamp=now + i,
            )
            for i in range(20)
        ]
        features = extractor.extract(events)
        assert features.technique_diversity == pytest.approx(1.0, abs=0.01)

    def test_all_same_pair(self):
        """20 events all same (action, target) → diversity = 1/20."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", target="/api/same", timestamp=now + i)
            for i in range(20)
        ]
        features = extractor.extract(events)
        assert features.technique_diversity == pytest.approx(1 / 20, abs=0.01)


class TestTemporalRegularity:
    """Tests for temporal regularity (CV) feature."""

    def test_regular_intervals(self):
        """Events at perfectly regular intervals → CV ≈ 0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", timestamp=now + i * 1.0)
            for i in range(20)
        ]
        features = extractor.extract(events)
        assert features.temporal_regularity == pytest.approx(0.0, abs=0.01)

    def test_irregular_intervals(self):
        """Events with random intervals → CV > 0.5."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        rng = random.Random(42)
        timestamps = sorted([now + rng.uniform(0, 100) for _ in range(20)])
        events = [
            _make_event(agent_id=f"a-{i}", timestamp=timestamps[i])
            for i in range(20)
        ]
        features = extractor.extract(events)
        assert features.temporal_regularity > 0.5


class TestProgressionScore:
    """Tests for progression score feature."""

    def test_full_escalation(self):
        """First half PROBE, second half TOOL_INVOCATION → score ≈ 1.0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = []
        for i in range(10):
            events.append(_make_event(
                agent_id=f"a-{i}", action_type=ActionType.PROBE, timestamp=now + i,
            ))
        for i in range(10):
            events.append(_make_event(
                agent_id=f"b-{i}", action_type=ActionType.TOOL_INVOCATION, timestamp=now + 20 + i,
            ))
        features = extractor.extract(events)
        assert features.progression_score == pytest.approx(1.0, abs=0.01)

    def test_uniform_mix(self):
        """Uniform action type mix → score ≈ 0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = []
        for i in range(20):
            at = ActionType.PROBE if i % 2 == 0 else ActionType.TOOL_INVOCATION
            events.append(_make_event(agent_id=f"a-{i}", action_type=at, timestamp=now + i))
        features = extractor.extract(events)
        assert abs(features.progression_score) < 0.2


class TestInfoFlowDensity:
    """Tests for information flow density feature."""

    def test_full_chain(self):
        """Every event's input matches prior output → high density."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = []
        for i in range(10):
            events.append(_make_event(
                agent_id=f"a-{i}",
                target="/exec",
                input_hash=f"h{i - 1}" if i > 0 else "",
                output_hash=f"h{i}",
                timestamp=now + i,
            ))
        features = extractor.extract(events)
        # 9 out of 10 have matching input
        assert features.info_flow_density == pytest.approx(0.9, abs=0.01)

    def test_no_chain(self):
        """No hash matches → density = 0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", timestamp=now + i)
            for i in range(10)
        ]
        features = extractor.extract(events)
        assert features.info_flow_density == 0.0


class TestAgentCardinality:
    """Tests for agent cardinality feature."""

    def test_distinct_agents(self):
        """5 distinct agents → cardinality = 5."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id=f"agent-{i}", timestamp=now + i)
            for i in range(5)
        ]
        features = extractor.extract(events)
        assert features.agent_cardinality == 5

    def test_same_agent(self):
        """Same agent repeated → cardinality = 1."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id="solo", timestamp=now + i)
            for i in range(10)
        ]
        features = extractor.extract(events)
        assert features.agent_cardinality == 1


class TestStylisticDiscontinuity:
    """Tests for stylistic discontinuity (alloy detection) feature."""

    def test_alternating_models(self):
        """Alternating claude/gpt-5 on same target → discontinuity > 0.8."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(
                agent_id=f"a-{i}",
                target="/api/target",
                source_model="claude" if i % 2 == 0 else "gpt-5",
                timestamp=now + i,
            )
            for i in range(20)
        ]
        features = extractor.extract(events)
        assert features.stylistic_discontinuity > 0.8

    def test_single_model(self):
        """All same model → discontinuity = 0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(
                agent_id=f"a-{i}", target="/api/target",
                source_model="claude", timestamp=now + i,
            )
            for i in range(20)
        ]
        features = extractor.extract(events)
        assert features.stylistic_discontinuity == 0.0

    def test_no_model_populated(self):
        """No source_model → discontinuity = 0."""
        extractor = IntentFeatureExtractor()
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", timestamp=now + i)
            for i in range(20)
        ]
        features = extractor.extract(events)
        assert features.stylistic_discontinuity == 0.0


class TestSlidingWindow:
    """Tests for sliding window extraction."""

    def test_correct_window_count(self):
        """100 events with window=50, step=25 → 3 windows."""
        extractor = IntentFeatureExtractor(window_size=50, window_step=25)
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", timestamp=now + i)
            for i in range(100)
        ]
        vectors = extractor.extract_sliding(events)
        # Windows: [0:50], [25:75], [50:100] = 3
        assert len(vectors) == 3

    def test_fewer_than_window_size(self):
        """Fewer events than window → single extraction."""
        extractor = IntentFeatureExtractor(window_size=50, window_step=25)
        now = time.time()
        events = [
            _make_event(agent_id=f"a-{i}", timestamp=now + i)
            for i in range(10)
        ]
        vectors = extractor.extract_sliding(events)
        assert len(vectors) == 1

    def test_empty_events(self):
        """Empty events → empty list."""
        extractor = IntentFeatureExtractor()
        vectors = extractor.extract_sliding([])
        assert vectors == []
