"""
Tests for campaign fingerprint detection signatures.

Each test verifies both the detection (true positive) and non-detection
(true negative) case for each of the five fingerprint signatures.
"""

from __future__ import annotations

import time

import pytest

from aegis.layers.correlation.events import (
    ActionType,
    AgentActionEvent,
    FingerprintType,
)
from aegis.layers.correlation.fingerprint_detector import FingerprintDetector


def _make_event(
    agent_id: str = "agent-1",
    action_type: ActionType = ActionType.API_CALL,
    target: str = "/api/v1/users",
    timestamp: float | None = None,
    input_hash: str = "",
    output_hash: str = "",
    parameter_value: str = "",
) -> AgentActionEvent:
    return AgentActionEvent(
        agent_id=agent_id,
        action_type=action_type,
        target_resource=target,
        timestamp=timestamp or time.time(),
        input_hash=input_hash,
        output_hash=output_hash,
        parameter_value=parameter_value,
    )


class TestTemporalClustering:
    """Tests for temporal clustering detection."""

    def test_detect_coordinated_burst(self):
        """20 distinct agents hitting same endpoint in 5s → detection."""
        detector = FingerprintDetector(
            window_sizes=[10],
            temporal_cluster_min_agents=10,
            temporal_cluster_cv_threshold=0.5,
        )
        now = time.time()
        for i in range(20):
            event = _make_event(
                agent_id=f"agent-{i}",
                target="/api/v1/users",
                timestamp=now + i * 0.25,  # 0.25s apart
            )
            detector.add_event(event)

        matches = detector.detect()
        temporal = [m for m in matches if m.signature_type == FingerprintType.TEMPORAL_CLUSTER]
        assert len(temporal) > 0
        assert temporal[0].confidence >= 0.5
        assert temporal[0].evidence["distinct_agents"] >= 10

    def test_no_detection_spread_traffic(self):
        """20 events spread over 5 minutes → no detection."""
        detector = FingerprintDetector(
            window_sizes=[10],
            temporal_cluster_min_agents=10,
        )
        now = time.time()
        for i in range(20):
            event = _make_event(
                agent_id=f"agent-{i}",
                target="/api/v1/users",
                timestamp=now - 300 + i * 15,  # 15s apart over 5 minutes
            )
            detector.add_event(event)

        matches = detector.detect()
        temporal = [m for m in matches if m.signature_type == FingerprintType.TEMPORAL_CLUSTER]
        assert len(temporal) == 0

    def test_no_detection_same_agent(self):
        """20 events from same agent → no detection (not distributed)."""
        detector = FingerprintDetector(
            window_sizes=[10],
            temporal_cluster_min_agents=10,
        )
        now = time.time()
        for i in range(20):
            event = _make_event(
                agent_id="single-agent",
                target="/api/v1/users",
                timestamp=now + i * 0.1,
            )
            detector.add_event(event)

        matches = detector.detect()
        temporal = [m for m in matches if m.signature_type == FingerprintType.TEMPORAL_CLUSTER]
        assert len(temporal) == 0


class TestSystematicEnumeration:
    """Tests for systematic enumeration detection."""

    def test_detect_sequential_probing(self):
        """Agents probing all known resources → detection."""
        detector = FingerprintDetector(
            window_sizes=[30],
            enumeration_coverage_threshold=0.6,
        )
        resources = [
            "/api/v1/users", "/api/v1/orders", "/api/v1/products",
            "/api/v1/settings", "/api/v1/admin",
        ]
        # First establish the known resource space
        now = time.time()
        for r in resources:
            detector._known_resources.add(r)

        # Now different agents enumerate them
        for i, r in enumerate(resources):
            event = _make_event(
                agent_id=f"scanner-{i}",
                target=r,
                timestamp=now,
            )
            detector.add_event(event)

        matches = detector.detect()
        enum_matches = [m for m in matches if m.signature_type == FingerprintType.SYSTEMATIC_ENUMERATION]
        assert len(enum_matches) > 0
        assert enum_matches[0].evidence["coverage"] >= 0.6

    def test_no_detection_partial_random(self):
        """Random access to 1 of 10 resources → no detection."""
        detector = FingerprintDetector(
            window_sizes=[30],
            enumeration_coverage_threshold=0.6,
        )
        resources = [f"/api/v1/endpoint-{i}" for i in range(10)]
        for r in resources:
            detector._known_resources.add(r)

        now = time.time()
        # Only hit 1 resource
        event = _make_event(agent_id="a1", target=resources[0], timestamp=now)
        detector.add_event(event)
        event = _make_event(agent_id="a2", target=resources[0], timestamp=now)
        detector.add_event(event)
        event = _make_event(agent_id="a3", target=resources[0], timestamp=now)
        detector.add_event(event)

        matches = detector.detect()
        enum_matches = [m for m in matches if m.signature_type == FingerprintType.SYSTEMATIC_ENUMERATION]
        assert len(enum_matches) == 0


class TestParameterFuzzing:
    """Tests for parameter fuzzing detection."""

    def test_detect_diverse_params(self):
        """15 agents testing 15 different param values → detection."""
        detector = FingerprintDetector(
            window_sizes=[30],
            fuzzing_entropy_std_threshold=2.0,
        )
        now = time.time()
        # Add events directly to avoid detect() calls building up entropy
        # history (which inflates the adaptive threshold).
        for i in range(15):
            event = _make_event(
                agent_id=f"fuzzer-{i}",
                target="/api/v1/login",
                parameter_value=f"unique_payload_{i}",
                timestamp=now,
            )
            detector._events.append(event)

        matches = detector.detect()
        fuzz_matches = [m for m in matches if m.signature_type == FingerprintType.PARAMETER_FUZZING]
        assert len(fuzz_matches) > 0
        assert fuzz_matches[0].evidence["distinct_values"] >= 15

    def test_no_detection_same_params(self):
        """15 events with same parameter value (retries) → no detection."""
        detector = FingerprintDetector(
            window_sizes=[30],
            fuzzing_entropy_std_threshold=2.0,
        )
        now = time.time()
        for i in range(15):
            event = _make_event(
                agent_id=f"agent-{i}",
                target="/api/v1/login",
                parameter_value="same_value",
                timestamp=now,
            )
            detector.add_event(event)

        matches = detector.detect()
        fuzz_matches = [m for m in matches if m.signature_type == FingerprintType.PARAMETER_FUZZING]
        assert len(fuzz_matches) == 0


class TestReconToExploit:
    """Tests for recon-to-exploit progression detection."""

    def test_detect_phase_transition(self):
        """PROBE events → AUTH_ATTEMPT/TOOL events → detection."""
        detector = FingerprintDetector(
            window_sizes=[60],
            recon_exploit_transition_threshold=0.7,
        )
        now = time.time()

        # First half: recon (PROBEs from different agents)
        for i in range(10):
            event = _make_event(
                agent_id=f"recon-{i}",
                action_type=ActionType.PROBE,
                target="/api/v1/users",
                timestamp=now - 50 + i,
            )
            detector.add_event(event)

        # Second half: exploit (AUTH_ATTEMPT/TOOL_INVOCATION from different agents)
        for i in range(10):
            event = _make_event(
                agent_id=f"exploit-{i}",
                action_type=ActionType.AUTH_ATTEMPT if i % 2 == 0 else ActionType.TOOL_INVOCATION,
                target="/api/v1/users",
                timestamp=now - 20 + i,
            )
            detector.add_event(event)

        matches = detector.detect()
        r2e = [m for m in matches if m.signature_type == FingerprintType.RECON_TO_EXPLOIT]
        assert len(r2e) > 0
        assert r2e[0].evidence["transition_score"] >= 0.7

    def test_no_detection_mixed_actions(self):
        """Mixed action types with no clear progression → no detection."""
        detector = FingerprintDetector(
            window_sizes=[60],
            recon_exploit_transition_threshold=0.7,
        )
        now = time.time()
        actions = [ActionType.API_CALL, ActionType.PROBE, ActionType.AUTH_ATTEMPT,
                   ActionType.OUTPUT_GENERATION, ActionType.DATA_ACCESS]

        for i in range(20):
            event = _make_event(
                agent_id=f"agent-{i % 5}",
                action_type=actions[i % len(actions)],
                target="/api/v1/mixed",
                timestamp=now - 50 + i * 2.5,
            )
            detector.add_event(event)

        matches = detector.detect()
        r2e = [m for m in matches if m.signature_type == FingerprintType.RECON_TO_EXPLOIT]
        assert len(r2e) == 0


class TestInformationFlow:
    """Tests for information flow (output→input hash chain) detection."""

    def test_detect_hash_chain(self):
        """Agent A output → Agent B input → Agent C input → detection."""
        detector = FingerprintDetector(
            window_sizes=[60],
            info_flow_min_links=3,
        )
        now = time.time()

        # Agent A produces output
        detector.add_event(_make_event(
            agent_id="coordinator",
            target="/api/v1/plan",
            output_hash="hash_1",
            timestamp=now - 10,
        ))
        # Agent B consumes A's output, produces own output
        detector.add_event(_make_event(
            agent_id="solver-1",
            target="/api/v1/exec",
            input_hash="hash_1",
            output_hash="hash_2",
            timestamp=now - 8,
        ))
        # Agent C consumes B's output
        detector.add_event(_make_event(
            agent_id="solver-2",
            target="/api/v1/exec",
            input_hash="hash_2",
            output_hash="hash_3",
            timestamp=now - 6,
        ))
        # Agent D consumes C's output
        detector.add_event(_make_event(
            agent_id="solver-3",
            target="/api/v1/exec",
            input_hash="hash_3",
            output_hash="hash_4",
            timestamp=now - 4,
        ))

        matches = detector.detect()
        flow_matches = [m for m in matches if m.signature_type == FingerprintType.INFORMATION_FLOW]
        assert len(flow_matches) > 0
        assert flow_matches[0].evidence["chain_count"] >= 3

    def test_no_detection_no_hash_overlap(self):
        """Events with no hash overlap → no detection."""
        detector = FingerprintDetector(
            window_sizes=[60],
            info_flow_min_links=3,
        )
        now = time.time()
        for i in range(10):
            detector.add_event(_make_event(
                agent_id=f"agent-{i}",
                target="/api/v1/data",
                input_hash=f"unique_in_{i}",
                output_hash=f"unique_out_{i}",
                timestamp=now,
            ))

        matches = detector.detect()
        flow_matches = [m for m in matches if m.signature_type == FingerprintType.INFORMATION_FLOW]
        assert len(flow_matches) == 0


class TestDetectorUtilities:
    """Tests for detector utility functions."""

    def test_shannon_entropy_uniform(self):
        """Uniform distribution has maximum entropy."""
        values = ["a", "b", "c", "d"]
        entropy = FingerprintDetector._shannon_entropy(values)
        assert entropy == pytest.approx(2.0, abs=0.01)  # log2(4)

    def test_shannon_entropy_constant(self):
        """Constant distribution has zero entropy."""
        values = ["a", "a", "a", "a"]
        entropy = FingerprintDetector._shannon_entropy(values)
        assert entropy == 0.0

    def test_shannon_entropy_empty(self):
        """Empty list has zero entropy."""
        assert FingerprintDetector._shannon_entropy([]) == 0.0

    def test_multiple_windows(self):
        """Detector checks all window sizes."""
        detector = FingerprintDetector(window_sizes=[5, 10, 30])
        now = time.time()
        for i in range(5):
            detector.add_event(_make_event(
                agent_id=f"a-{i}",
                target="/api",
                timestamp=now,
            ))
        # Should run without error
        matches = detector.detect()
        assert isinstance(matches, list)
