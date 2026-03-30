"""
Tests for the Compliance Reporter — compliance evidence from TVE results.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aegis.layers.thymic.attack_profile_library import AttackProfileLibrary, Probe
from aegis.layers.thymic.engine import (
    ThymicValidationEngine,
    ValidationReport,
    LayerResult,
    TierResult,
)
from aegis.layers.thymic.layer_probe_router import LayerProbeRouter, ProbeResult
from aegis.layers.thymic.telemetry_collector import TelemetryCollector
from aegis.layers.thymic.verdict_analyzer import (
    VerdictAnalyzer,
    VerdictReport,
    DetectionVerdict,
    FalsePositiveVerdict,
    DeadScannerVerdict,
)
from aegis.layers.thymic.response_emitter import ResponseEmitter
from aegis.layers.thymic.compliance_reporter import (
    ComplianceReporter,
    ComplianceEvidence,
    ComplianceSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(
    run_id: str = "test-run",
    run_type: str = "spot_check",
    total_probes: int = 100,
    probes_detected: int = 95,
    probes_missed: int = 5,
    false_positives: int = 0,
    overall_tpr: float = 0.95,
    overall_fpr: float = 0.0,
    verdict: VerdictReport | None = None,
) -> ValidationReport:
    report = ValidationReport(
        run_id=run_id,
        run_type=run_type,
        total_probes=total_probes,
        probes_detected=probes_detected,
        probes_missed=probes_missed,
        false_positives=false_positives,
        overall_tpr=overall_tpr,
        overall_fpr=overall_fpr,
    )
    report.verdict = verdict
    return report


def _make_passing_verdict(run_id: str = "test-run") -> VerdictReport:
    return VerdictReport(
        run_id=run_id,
        overall_passed=True,
        detection_verdicts=[
            DetectionVerdict(passed=True, layer_id="L2",
                             observed_tpr=0.98, baseline_tpr=0.95, threshold=0.95),
        ],
        false_positive_verdict=FalsePositiveVerdict(
            passed=True, observed_fpr=0.0, baseline_fpr=0.0, threshold=0.005),
        dead_scanner_verdict=DeadScannerVerdict(passed=True),
    )


def _make_failing_verdict(run_id: str = "test-run") -> VerdictReport:
    return VerdictReport(
        run_id=run_id,
        overall_passed=False,
        detection_verdicts=[
            DetectionVerdict(passed=False, layer_id="L2",
                             observed_tpr=0.60, baseline_tpr=0.95,
                             threshold=0.95, deficit=0.35),
        ],
        dead_scanner_verdict=DeadScannerVerdict(
            passed=False, dead_scanners=["L99"]),
    )


@pytest.fixture
def reporter() -> ComplianceReporter:
    return ComplianceReporter()


# ---------------------------------------------------------------------------
# generate_evidence
# ---------------------------------------------------------------------------

class TestGenerateEvidence:
    def test_produces_all_required_fields(self, reporter: ComplianceReporter):
        verdict = _make_passing_verdict()
        report = _make_report(verdict=verdict)
        evidence = reporter.generate_evidence(report)
        assert evidence.run_id == "test-run"
        assert evidence.run_type == "spot_check"
        assert evidence.timestamp is not None

    def test_includes_all_six_frameworks(self, reporter: ComplianceReporter):
        verdict = _make_passing_verdict()
        report = _make_report(verdict=verdict)
        evidence = reporter.generate_evidence(report)
        frameworks = {m.framework for m in evidence.framework_mappings}
        assert "NIST AI RMF" in frameworks
        assert "ISO 42001" in frameworks
        assert "EU AI Act" in frameworks
        assert "CMMC 2.0" in frameworks
        assert "SOC 2" in frameworks
        assert "FedRAMP" in frameworks

    def test_each_mapping_has_required_fields(self, reporter: ComplianceReporter):
        verdict = _make_passing_verdict()
        report = _make_report(verdict=verdict)
        evidence = reporter.generate_evidence(report)
        for m in evidence.framework_mappings:
            assert m.framework
            assert m.requirement_id
            assert m.requirement_description
            assert m.evidence_summary
            assert isinstance(m.passed, bool)

    def test_pass_reflects_verdict(self, reporter: ComplianceReporter):
        verdict = _make_passing_verdict()
        report = _make_report(verdict=verdict)
        evidence = reporter.generate_evidence(report)
        assert all(m.passed for m in evidence.framework_mappings)

    def test_detection_failure_maps_to_nist(self, reporter: ComplianceReporter):
        verdict = _make_failing_verdict()
        report = _make_report(verdict=verdict)
        evidence = reporter.generate_evidence(report)
        nist = [m for m in evidence.framework_mappings if m.framework == "NIST AI RMF"]
        assert len(nist) == 1
        assert nist[0].passed is False

    def test_dead_scanner_maps_to_cmmc(self, reporter: ComplianceReporter):
        verdict = _make_failing_verdict()
        report = _make_report(verdict=verdict)
        evidence = reporter.generate_evidence(report)
        cmmc = [m for m in evidence.framework_mappings if m.framework == "CMMC 2.0"]
        assert len(cmmc) == 1
        assert cmmc[0].passed is False
        assert "dead scanner" in cmmc[0].evidence_summary.lower()

    def test_to_dict_serialization(self, reporter: ComplianceReporter):
        verdict = _make_passing_verdict()
        report = _make_report(verdict=verdict)
        evidence = reporter.generate_evidence(report)
        d = evidence.to_dict()
        assert "run_id" in d
        assert "framework_mappings" in d
        assert len(d["framework_mappings"]) == 6


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    def test_aggregates_multiple_reports(self, reporter: ComplianceReporter):
        reports = [
            _make_report(run_id=f"r{i}", overall_tpr=0.9 + i * 0.01,
                         verdict=_make_passing_verdict(f"r{i}"))
            for i in range(5)
        ]
        summary = reporter.generate_summary(reports, period_days=7)
        assert summary.total_runs == 5
        assert summary.period_days == 7

    def test_average_tpr_correct(self, reporter: ComplianceReporter):
        reports = [
            _make_report(run_id="r1", overall_tpr=0.9, verdict=_make_passing_verdict("r1")),
            _make_report(run_id="r2", overall_tpr=1.0, verdict=_make_passing_verdict("r2")),
        ]
        summary = reporter.generate_summary(reports)
        assert abs(summary.average_tpr - 0.95) < 0.01

    def test_average_fpr_correct(self, reporter: ComplianceReporter):
        reports = [
            _make_report(run_id="r1", overall_fpr=0.01, verdict=_make_passing_verdict("r1")),
            _make_report(run_id="r2", overall_fpr=0.03, verdict=_make_passing_verdict("r2")),
        ]
        summary = reporter.generate_summary(reports)
        assert abs(summary.average_fpr - 0.02) < 0.01

    def test_failed_check_timestamps(self, reporter: ComplianceReporter):
        reports = [
            _make_report(run_id="r1", verdict=_make_passing_verdict("r1")),
            _make_report(run_id="r2", verdict=_make_failing_verdict("r2")),
        ]
        summary = reporter.generate_summary(reports)
        assert len(summary.failed_checks) == 1
        assert summary.failed_checks[0]["run_id"] == "r2"

    def test_uptime_percentage(self, reporter: ComplianceReporter):
        reports = [
            _make_report(run_id="r1", verdict=_make_passing_verdict("r1")),
            _make_report(run_id="r2", verdict=_make_passing_verdict("r2")),
            _make_report(run_id="r3", verdict=_make_failing_verdict("r3")),
        ]
        summary = reporter.generate_summary(reports)
        # 2 passed / 3 total = 66.67%
        assert abs(summary.uptime_percentage - 66.67) < 1.0

    def test_empty_reports(self, reporter: ComplianceReporter):
        summary = reporter.generate_summary([])
        assert summary.total_runs == 0
        assert summary.average_tpr == 0.0
        assert summary.uptime_percentage == 0.0

    def test_single_report(self, reporter: ComplianceReporter):
        reports = [_make_report(run_id="r1", overall_tpr=0.97,
                                verdict=_make_passing_verdict("r1"))]
        summary = reporter.generate_summary(reports)
        assert summary.total_runs == 1
        assert abs(summary.average_tpr - 0.97) < 0.01
        assert summary.uptime_percentage == 100.0

    def test_to_dict_serialization(self, reporter: ComplianceReporter):
        reports = [_make_report(run_id="r1", verdict=_make_passing_verdict("r1"))]
        summary = reporter.generate_summary(reports)
        d = summary.to_dict()
        assert "total_runs" in d
        assert "uptime_percentage" in d


# ---------------------------------------------------------------------------
# Dashboard endpoint
# ---------------------------------------------------------------------------

class TestDashboardEndpoint:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        import aegis.main as main_module
        from aegis.config import get_config

        self._saved_engine = main_module._tve_engine
        self._saved_config = main_module._config
        config = get_config()
        if not config.api_key:
            config.api_key = "test-tve-api-key"
        main_module._config = config
        self.headers = {"Authorization": f"Bearer {config.api_key}"}
        self.client = TestClient(main_module.app)
        yield
        main_module._tve_engine = self._saved_engine
        main_module._config = self._saved_config

    def test_health_returns_no_data_without_tve(self):
        import aegis.main as main_module
        main_module._tve_engine = None
        resp = self.client.get("/v1/tve/health", headers=self.headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_data"

    def test_health_returns_200_with_no_runs(self):
        import aegis.main as main_module
        engine = ThymicValidationEngine.__new__(ThymicValidationEngine)
        engine._last_report = None
        main_module._tve_engine = engine
        resp = self.client.get("/v1/tve/health", headers=self.headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_data"
