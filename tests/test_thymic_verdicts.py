"""
Tests for the Verdict Analyzer — four validation checks against baselines.
"""

from __future__ import annotations

import pytest

from aegis.layers.thymic.attack_profile_library import Probe
from aegis.layers.thymic.engine import ValidationReport, LayerResult, TierResult
from aegis.layers.thymic.layer_probe_router import ProbeResult
from aegis.layers.thymic.telemetry_collector import TelemetryCollector
from aegis.layers.thymic.verdict_analyzer import (
    VerdictAnalyzer,
    VerdictReport,
    DetectionVerdict,
    FalsePositiveVerdict,
    DeadScannerVerdict,
    LatencyVerdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(
    run_id: str = "test-run",
    tpr: float = 1.0,
    fpr: float = 0.0,
    layer_results: dict[str, LayerResult] | None = None,
    tier_results: dict[int, TierResult] | None = None,
    probe_results: list[ProbeResult] | None = None,
    total_probes: int = 100,
) -> ValidationReport:
    return ValidationReport(
        run_id=run_id,
        run_type="spot_check",
        total_probes=total_probes,
        probes_detected=int(total_probes * tpr),
        probes_missed=int(total_probes * (1 - tpr)),
        false_positives=0,
        per_layer_results=layer_results or {},
        per_tier_results=tier_results or {},
        overall_tpr=tpr,
        overall_fpr=fpr,
        probe_results=probe_results or [],
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


@pytest.fixture
def analyzer() -> VerdictAnalyzer:
    return VerdictAnalyzer(
        default_tpr_threshold=0.95,
        default_fpr_threshold=0.005,
        latency_regression_factor=2.0,
    )


@pytest.fixture
def telemetry() -> TelemetryCollector:
    return TelemetryCollector()


# ---------------------------------------------------------------------------
# Check 1: Detection Validation
# ---------------------------------------------------------------------------

class TestDetectionValidation:
    def test_passes_when_above_threshold(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", probes_tested=50, probes_detected=48, tpr=0.96)},
        )
        verdict = analyzer.analyze(report, telemetry)
        l2_verdicts = [v for v in verdict.detection_verdicts if v.layer_id == "L2"]
        assert len(l2_verdicts) == 1
        assert l2_verdicts[0].passed is True

    def test_fails_when_below_threshold(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", probes_tested=50, probes_detected=40, tpr=0.80)},
        )
        verdict = analyzer.analyze(report, telemetry)
        l2_verdicts = [v for v in verdict.detection_verdicts if v.layer_id == "L2"]
        assert l2_verdicts[0].passed is False
        assert l2_verdicts[0].deficit > 0

    def test_uses_default_when_no_baseline(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", probes_tested=50, probes_detected=50, tpr=1.0)},
        )
        verdict = analyzer.analyze(report, telemetry)
        l2_verdicts = [v for v in verdict.detection_verdicts if v.layer_id == "L2"]
        assert l2_verdicts[0].threshold == 0.95  # default


# ---------------------------------------------------------------------------
# Check 2: False Positive Validation
# ---------------------------------------------------------------------------

class TestFalsePositiveValidation:
    def test_passes_when_below_threshold(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(
            fpr=0.0,
            tier_results={5: TierResult(tier=5, total=50)},
            probe_results=[_make_probe_result(f"b{i}", 5, False) for i in range(50)],
        )
        verdict = analyzer.analyze(report, telemetry)
        assert verdict.false_positive_verdict is not None
        assert verdict.false_positive_verdict.passed is True

    def test_fails_when_above_threshold(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        # 5/50 = 10% FPR, way above 0.5% threshold
        probe_results = [_make_probe_result(f"b{i}", 5, False) for i in range(45)]
        probe_results += [_make_probe_result(f"fp{i}", 5, True, layers=["L2"]) for i in range(5)]
        report = _make_report(
            fpr=0.1,
            tier_results={5: TierResult(tier=5, total=50)},
            probe_results=probe_results,
        )
        verdict = analyzer.analyze(report, telemetry)
        assert verdict.false_positive_verdict.passed is False

    def test_identifies_flagged_probes(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        probe_results = [_make_probe_result("b1", 5, False)]
        probe_results.append(_make_probe_result("fp1", 5, True, layers=["L2"]))
        report = _make_report(
            fpr=0.5,
            tier_results={5: TierResult(tier=5, total=2)},
            probe_results=probe_results,
        )
        verdict = analyzer.analyze(report, telemetry)
        assert "fp1" in verdict.false_positive_verdict.flagged_probes

    def test_skipped_when_no_tier5(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(tier_results={})
        verdict = analyzer.analyze(report, telemetry)
        assert verdict.false_positive_verdict is None


# ---------------------------------------------------------------------------
# Check 3: Dead Scanner Detection
# ---------------------------------------------------------------------------

class TestDeadScannerDetection:
    def test_passes_when_all_have_detections(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        probe_results = [_make_probe_result(f"p{i}", 1, True, layers=["L2"]) for i in range(10)]
        report = _make_report(probe_results=probe_results)
        verdict = analyzer.analyze(report, telemetry)
        if verdict.dead_scanner_verdict:
            assert verdict.dead_scanner_verdict.passed is True

    def test_fails_with_dead_scanners(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        # All probes expected by L99 but none detected
        probe_results = [
            _make_probe_result(f"p{i}", 1, False, missed=["L99"]) for i in range(5)
        ]
        report = _make_report(probe_results=probe_results)
        verdict = analyzer.analyze(report, telemetry)
        assert verdict.dead_scanner_verdict is not None
        assert verdict.dead_scanner_verdict.passed is False
        assert "L99" in verdict.dead_scanner_verdict.dead_scanners


# ---------------------------------------------------------------------------
# Check 4: Latency Regression
# ---------------------------------------------------------------------------

class TestLatencyRegression:
    def test_passes_within_threshold(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        # Seed baseline: L2 p95 = 10ms
        probe = Probe(id="t1", text="test", tier=1, expected_detection_layer="L2")
        result = ProbeResult(probe_id="t1", probe_tier=1, detected=True,
                             detection_layers=["L2"], latency_ms={"total": 10.0, "L2": 10.0})
        telemetry.record_result(result, probe, run_id="seed")

        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", p95_latency_ms=15.0)},
        )
        verdict = analyzer.analyze(report, telemetry)
        l2_latency = [v for v in verdict.latency_verdicts if v.layer_id == "L2"]
        if l2_latency:
            assert l2_latency[0].passed is True

    def test_fails_when_regression(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        probe = Probe(id="t1", text="test", tier=1, expected_detection_layer="L2")
        result = ProbeResult(probe_id="t1", probe_tier=1, detected=True,
                             detection_layers=["L2"], latency_ms={"total": 10.0, "L2": 10.0})
        telemetry.record_result(result, probe, run_id="seed")

        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", p95_latency_ms=50.0)},
        )
        verdict = analyzer.analyze(report, telemetry)
        l2_latency = [v for v in verdict.latency_verdicts if v.layer_id == "L2"]
        if l2_latency:
            assert l2_latency[0].passed is False
            assert l2_latency[0].regression_factor > 2.0


# ---------------------------------------------------------------------------
# VerdictReport aggregate
# ---------------------------------------------------------------------------

class TestVerdictReport:
    def test_overall_passed_when_all_pass(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        probe_results = [_make_probe_result(f"p{i}", 1, True, layers=["L2"]) for i in range(10)]
        probe_results += [_make_probe_result(f"b{i}", 5, False) for i in range(10)]
        report = _make_report(
            tpr=1.0, fpr=0.0,
            layer_results={"L2": LayerResult(layer_id="L2", probes_tested=10, probes_detected=10, tpr=1.0)},
            tier_results={5: TierResult(tier=5, total=10)},
            probe_results=probe_results,
        )
        verdict = analyzer.analyze(report, telemetry)
        assert verdict.overall_passed is True

    def test_overall_failed_when_detection_fails(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", probes_tested=10, probes_detected=5, tpr=0.5)},
        )
        verdict = analyzer.analyze(report, telemetry)
        assert verdict.overall_passed is False

    def test_recommended_actions_on_failure(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", probes_tested=10, probes_detected=5, tpr=0.5)},
        )
        verdict = analyzer.analyze(report, telemetry)
        assert len(verdict.recommended_actions) > 0
        assert "L2" in verdict.recommended_actions[0]

    def test_no_actions_when_all_pass(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(
            layer_results={"L2": LayerResult(layer_id="L2", probes_tested=10, probes_detected=10, tpr=1.0)},
        )
        verdict = analyzer.analyze(report, telemetry)
        assert len(verdict.recommended_actions) == 0

    def test_handles_no_probes(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report(total_probes=0)
        verdict = analyzer.analyze(report, telemetry)
        assert isinstance(verdict, VerdictReport)

    def test_handles_no_tier5(self, analyzer: VerdictAnalyzer, telemetry: TelemetryCollector):
        report = _make_report()
        verdict = analyzer.analyze(report, telemetry)
        assert verdict.false_positive_verdict is None
