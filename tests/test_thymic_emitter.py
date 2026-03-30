"""
Tests for the Response Emitter — event bus publication of TVE findings.
"""

from __future__ import annotations

import asyncio

import pytest

from aegis.layers.thymic.engine import ValidationReport, LayerResult, TierResult
from aegis.layers.thymic.layer_probe_router import ProbeResult
from aegis.layers.thymic.verdict_analyzer import (
    VerdictReport,
    DetectionVerdict,
    FalsePositiveVerdict,
    DeadScannerVerdict,
    LatencyVerdict,
)
from aegis.layers.thymic.response_emitter import ResponseEmitter, CHANNEL_TVE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_validation_report(
    run_id: str = "run-1",
    total_probes: int = 10,
    probes_detected: int = 8,
    probes_missed: int = 2,
    false_positives: int = 0,
    overall_tpr: float = 0.8,
    overall_fpr: float = 0.0,
    probe_results: list[ProbeResult] | None = None,
) -> ValidationReport:
    return ValidationReport(
        run_id=run_id,
        run_type="spot_check",
        total_probes=total_probes,
        probes_detected=probes_detected,
        probes_missed=probes_missed,
        false_positives=false_positives,
        overall_tpr=overall_tpr,
        overall_fpr=overall_fpr,
        probe_results=probe_results or [],
    )


def _make_verdict_report(
    run_id: str = "run-1",
    overall_passed: bool = True,
    detection_verdicts: list[DetectionVerdict] | None = None,
    false_positive_verdict: FalsePositiveVerdict | None = None,
    dead_scanner_verdict: DeadScannerVerdict | None = None,
    latency_verdicts: list[LatencyVerdict] | None = None,
    recommended_actions: list[str] | None = None,
) -> VerdictReport:
    return VerdictReport(
        run_id=run_id,
        overall_passed=overall_passed,
        detection_verdicts=detection_verdicts or [],
        false_positive_verdict=false_positive_verdict,
        dead_scanner_verdict=dead_scanner_verdict,
        latency_verdicts=latency_verdicts or [],
        recommended_actions=recommended_actions or [],
    )


def _make_probe_result(probe_id: str, tier: int, detected: bool,
                       layers: list[str] | None = None,
                       missed: list[str] | None = None) -> ProbeResult:
    return ProbeResult(
        probe_id=probe_id, probe_tier=tier, detected=detected,
        detection_layers=layers or [], missed_layers=missed or [],
        latency_ms={"total": 5.0},
        false_positive=tier == 5 and detected,
        response_status=403 if detected else 200,
    )


class FakeEventBus:
    """In-memory event bus that records published events."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, channel: str, event: dict) -> None:
        self.published.append((channel, event))


class FailingEventBus:
    """Event bus that always raises."""

    async def publish(self, channel: str, event: dict) -> None:
        raise ConnectionError("Event bus down")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidationCompleteEvent:
    def test_always_emitted(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report()
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        complete = [e for e in events if e["event_type"] == "tve.validation.complete"]
        assert len(complete) == 1

    def test_carries_standard_fields(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(run_id="r42")
        validation = _make_validation_report(run_id="r42")
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        complete = events[0]
        assert complete["source"] == "tve"
        assert complete["run_id"] == "r42"
        assert complete["event_type"] == "tve.validation.complete"

    def test_includes_summary_fields(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(overall_passed=True, recommended_actions=["fix L2"])
        validation = _make_validation_report(
            total_probes=50, probes_detected=45, probes_missed=5,
            overall_tpr=0.9, overall_fpr=0.01,
        )
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        complete = events[0]
        assert complete["total_probes"] == 50
        assert complete["overall_tpr"] == 0.9
        assert complete["overall_passed"] is True


class TestDetectionFailureEvent:
    def test_emitted_for_missed_probes(self):
        emitter = ResponseEmitter()
        probe_results = [
            _make_probe_result("p1", 1, False, missed=["L2"]),
            _make_probe_result("p2", 2, False, missed=["L3"]),
            _make_probe_result("p3", 1, True, layers=["L2"]),
        ]
        verdict = _make_verdict_report()
        validation = _make_validation_report(probe_results=probe_results)
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        failures = [e for e in events if e["event_type"] == "tve.detection.failure"]
        assert len(failures) == 2
        assert failures[0]["probe_id"] == "p1"
        assert failures[1]["probe_id"] == "p2"

    def test_not_emitted_for_tier5(self):
        emitter = ResponseEmitter()
        probe_results = [_make_probe_result("b1", 5, False)]
        verdict = _make_verdict_report()
        validation = _make_validation_report(probe_results=probe_results)
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        failures = [e for e in events if e["event_type"] == "tve.detection.failure"]
        assert len(failures) == 0


class TestFalsePositiveEvent:
    def test_emitted_for_flagged_tier5(self):
        emitter = ResponseEmitter()
        probe_results = [_make_probe_result("fp1", 5, True, layers=["L2"])]
        verdict = _make_verdict_report()
        validation = _make_validation_report(probe_results=probe_results)
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        fps = [e for e in events if e["event_type"] == "tve.false_positive.detected"]
        assert len(fps) == 1
        assert fps[0]["probe_id"] == "fp1"
        assert fps[0]["flagging_layers"] == ["L2"]


class TestDeadScannerEvent:
    def test_emitted_for_dead_scanners(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(
            dead_scanner_verdict=DeadScannerVerdict(passed=False, dead_scanners=["L99", "L100"]),
        )
        validation = _make_validation_report(total_probes=20)
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        dead = [e for e in events if e["event_type"] == "tve.scanner.dead"]
        assert len(dead) == 2
        scanner_ids = {e["scanner_id"] for e in dead}
        assert scanner_ids == {"L99", "L100"}

    def test_not_emitted_when_passed(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(
            dead_scanner_verdict=DeadScannerVerdict(passed=True, dead_scanners=[]),
        )
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        dead = [e for e in events if e["event_type"] == "tve.scanner.dead"]
        assert len(dead) == 0


class TestLatencyRegressionEvent:
    def test_emitted_for_regression(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(
            latency_verdicts=[
                LatencyVerdict(passed=False, layer_id="L2",
                               observed_p95=100.0, baseline_p95=20.0, regression_factor=5.0),
            ],
        )
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        latency = [e for e in events if e["event_type"] == "tve.latency.regression"]
        assert len(latency) == 1
        assert latency[0]["layer_id"] == "L2"
        assert latency[0]["regression_factor"] == 5.0

    def test_not_emitted_when_passed(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(
            latency_verdicts=[
                LatencyVerdict(passed=True, layer_id="L2",
                               observed_p95=15.0, baseline_p95=10.0, regression_factor=1.5),
            ],
        )
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        latency = [e for e in events if e["event_type"] == "tve.latency.regression"]
        assert len(latency) == 0


class TestTLIRecommendation:
    def test_up_on_detection_failure_plus_dead_scanners(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(
            overall_passed=False,
            detection_verdicts=[
                DetectionVerdict(passed=False, layer_id="L2",
                                 observed_tpr=0.6, baseline_tpr=0.95,
                                 threshold=0.95, deficit=0.35),
            ],
            dead_scanner_verdict=DeadScannerVerdict(passed=False, dead_scanners=["L99"]),
        )
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        tli = [e for e in events if e["event_type"] == "tve.tli.recommendation"]
        assert len(tli) == 1
        assert tli[0]["recommended_direction"] == "up"

    def test_down_when_all_pass_and_low_fpr(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(
            overall_passed=True,
            false_positive_verdict=FalsePositiveVerdict(
                passed=True, observed_fpr=0.001, baseline_fpr=0.0,
                threshold=0.005,
            ),
        )
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        tli = [e for e in events if e["event_type"] == "tve.tli.recommendation"]
        assert len(tli) == 1
        assert tli[0]["recommended_direction"] == "down"

    def test_no_tli_when_pass_but_moderate_fpr(self):
        emitter = ResponseEmitter()
        verdict = _make_verdict_report(
            overall_passed=True,
            false_positive_verdict=FalsePositiveVerdict(
                passed=True, observed_fpr=0.004, baseline_fpr=0.0,
                threshold=0.005,
            ),
        )
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        tli = [e for e in events if e["event_type"] == "tve.tli.recommendation"]
        assert len(tli) == 0


class TestEventBusIntegration:
    def test_publishes_to_event_bus(self):
        bus = FakeEventBus()
        emitter = ResponseEmitter(event_bus=bus)
        verdict = _make_verdict_report()
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        assert len(bus.published) == len(events)
        assert all(ch == CHANNEL_TVE for ch, _ in bus.published)

    def test_graceful_without_event_bus(self):
        emitter = ResponseEmitter(event_bus=None)
        verdict = _make_verdict_report()
        validation = _make_validation_report()
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        assert len(events) >= 1  # validation.complete at minimum

    def test_graceful_on_bus_failure(self):
        emitter = ResponseEmitter(event_bus=FailingEventBus())
        verdict = _make_verdict_report()
        validation = _make_validation_report()
        # Should not raise
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        assert len(events) >= 1

    def test_returns_all_emitted_events(self):
        emitter = ResponseEmitter()
        probe_results = [
            _make_probe_result("p1", 1, False, missed=["L2"]),
            _make_probe_result("fp1", 5, True, layers=["L2"]),
        ]
        verdict = _make_verdict_report()
        validation = _make_validation_report(probe_results=probe_results)
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        types = {e["event_type"] for e in events}
        assert "tve.validation.complete" in types
        assert "tve.detection.failure" in types
        assert "tve.false_positive.detected" in types

    def test_all_events_have_source_and_run_id(self):
        emitter = ResponseEmitter()
        probe_results = [_make_probe_result("p1", 1, False, missed=["L2"])]
        verdict = _make_verdict_report(
            run_id="check-run",
            dead_scanner_verdict=DeadScannerVerdict(passed=False, dead_scanners=["L99"]),
        )
        validation = _make_validation_report(run_id="check-run", probe_results=probe_results)
        events = asyncio.get_event_loop().run_until_complete(
            emitter.emit(verdict, validation))
        for event in events:
            assert event["source"] == "tve"
            assert event["run_id"] == "check-run"
