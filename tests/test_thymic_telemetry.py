"""
Tests for the Telemetry Collector — per-probe result capture and baseline computation.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from aegis.layers.thymic.attack_profile_library import Probe
from aegis.layers.thymic.layer_probe_router import ProbeResult
from aegis.layers.thymic.telemetry_collector import (
    TelemetryCollector,
    TelemetryRecord,
    LayerBaseline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collector() -> TelemetryCollector:
    return TelemetryCollector(max_records=1000)


def _make_probe(probe_id: str, tier: int = 1, category: str = "direct_injection",
                expected_layer: str = "L2") -> Probe:
    return Probe(
        id=probe_id, text="test attack", tier=tier, category=category,
        mitre_tactic="AML.T0051", expected_detection_layer=expected_layer,
        expected_result="block" if tier != 5 else "pass",
    )


def _make_result(probe_id: str, tier: int = 1, detected: bool = True,
                 layers: list[str] | None = None, missed: list[str] | None = None,
                 latency: float = 10.0) -> ProbeResult:
    return ProbeResult(
        probe_id=probe_id, probe_tier=tier, detected=detected,
        detection_layers=layers or (["L2"] if detected else []),
        missed_layers=missed or [],
        latency_ms={"total": latency},
        false_positive=tier == 5 and detected,
        response_status=403 if detected else 200,
    )


# ---------------------------------------------------------------------------
# record_result tests
# ---------------------------------------------------------------------------

class TestRecordResult:
    def test_stores_record(self, collector: TelemetryCollector):
        probe = _make_probe("p1")
        result = _make_result("p1")
        collector.record_result(result, probe, run_id="run1")
        assert collector.record_count() == 1

    def test_record_has_all_fields(self, collector: TelemetryCollector):
        probe = _make_probe("p1", category="jailbreak")
        result = _make_result("p1", layers=["L2", "L3"])
        collector.record_result(result, probe, run_id="run1")
        records = collector.get_run_results("run1")
        r = records[0]
        assert r.probe_id == "p1"
        assert r.probe_tier == 1
        assert r.run_id == "run1"
        assert r.detected is True
        assert r.detection_layers == ["L2", "L3"]
        assert r.category == "jailbreak"
        assert r.expected_detection_layer == "L2"


class TestGetRunResults:
    def test_returns_only_matching_run(self, collector: TelemetryCollector):
        for i in range(3):
            collector.record_result(
                _make_result(f"p{i}"), _make_probe(f"p{i}"), run_id="run1")
        collector.record_result(
            _make_result("p99"), _make_probe("p99"), run_id="run2")
        assert len(collector.get_run_results("run1")) == 3
        assert len(collector.get_run_results("run2")) == 1

    def test_empty_run(self, collector: TelemetryCollector):
        assert collector.get_run_results("nonexistent") == []


class TestGetLayerHistory:
    def test_filters_by_layer(self, collector: TelemetryCollector):
        collector.record_result(
            _make_result("p1", layers=["L2"]), _make_probe("p1", expected_layer="L2"), run_id="r1")
        collector.record_result(
            _make_result("p2", layers=["L3"]), _make_probe("p2", expected_layer="L3"), run_id="r1")
        history = collector.get_layer_history("L2", window_hours=24)
        assert len(history) == 1
        assert history[0].probe_id == "p1"

    def test_filters_by_time_window(self, collector: TelemetryCollector):
        collector.record_result(
            _make_result("p1", layers=["L2"]), _make_probe("p1"), run_id="r1")
        # Manually backdate
        collector._records[-1].timestamp = datetime.now(timezone.utc) - timedelta(hours=48)
        collector.record_result(
            _make_result("p2", layers=["L2"]), _make_probe("p2"), run_id="r2")
        history = collector.get_layer_history("L2", window_hours=24)
        assert len(history) == 1
        assert history[0].probe_id == "p2"


class TestGetBaseline:
    def test_computes_tpr(self, collector: TelemetryCollector):
        # 8 detected, 2 missed = TPR 0.8
        for i in range(8):
            collector.record_result(
                _make_result(f"d{i}", detected=True, layers=["L2"]),
                _make_probe(f"d{i}", expected_layer="L2"), run_id="r1")
        for i in range(2):
            collector.record_result(
                _make_result(f"m{i}", detected=False, missed=["L2"]),
                _make_probe(f"m{i}", expected_layer="L2"), run_id="r1")
        baseline = collector.get_baseline("L2")
        assert baseline.layer_id == "L2"
        assert 0.79 <= baseline.tpr <= 0.81

    def test_computes_fpr(self, collector: TelemetryCollector):
        # Add 10 benign, 1 false positive
        for i in range(9):
            collector.record_result(
                _make_result(f"b{i}", tier=5, detected=False),
                _make_probe(f"b{i}", tier=5, expected_layer=None), run_id="r1")
        collector.record_result(
            _make_result("fp1", tier=5, detected=True, layers=["L2"]),
            _make_probe("fp1", tier=5, expected_layer=None), run_id="r1")
        baseline = collector.get_baseline("L2")
        assert 0.09 <= baseline.fpr <= 0.11  # 1/10

    def test_empty_returns_defaults(self, collector: TelemetryCollector):
        baseline = collector.get_baseline("L2")
        assert baseline.tpr == 0.0
        assert baseline.fpr == 0.0
        assert baseline.sample_count == 0


class TestComputeBaselines:
    def test_returns_all_layers(self, collector: TelemetryCollector):
        collector.record_result(
            _make_result("p1", layers=["L2"]), _make_probe("p1", expected_layer="L2"), run_id="r1")
        collector.record_result(
            _make_result("p2", layers=["L3"]), _make_probe("p2", expected_layer="L3"), run_id="r1")
        baselines = collector.compute_baselines()
        assert "L2" in baselines
        assert "L3" in baselines


class TestPrune:
    def test_removes_old_records(self, collector: TelemetryCollector):
        collector.record_result(
            _make_result("old"), _make_probe("old"), run_id="r1")
        collector._records[-1].timestamp = datetime.now(timezone.utc) - timedelta(days=100)
        collector.record_result(
            _make_result("new"), _make_probe("new"), run_id="r2")
        removed = collector.prune(retention_days=90)
        assert removed == 1
        assert collector.record_count() == 1

    def test_preserves_recent(self, collector: TelemetryCollector):
        for i in range(5):
            collector.record_result(
                _make_result(f"p{i}"), _make_probe(f"p{i}"), run_id="r1")
        removed = collector.prune(retention_days=90)
        assert removed == 0
        assert collector.record_count() == 5


class TestRollingBuffer:
    def test_fifo_eviction(self):
        collector = TelemetryCollector(max_records=5)
        for i in range(10):
            collector.record_result(
                _make_result(f"p{i}"), _make_probe(f"p{i}"), run_id="r1")
        assert collector.record_count() == 5
        # Should have the last 5
        ids = [r.probe_id for r in collector._records]
        assert ids == ["p5", "p6", "p7", "p8", "p9"]


class TestRecordCount:
    def test_accurate(self, collector: TelemetryCollector):
        assert collector.record_count() == 0
        collector.record_result(
            _make_result("p1"), _make_probe("p1"), run_id="r1")
        assert collector.record_count() == 1
