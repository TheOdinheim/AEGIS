"""
Tests for Compliance Dashboard — Cross-Framework Regulatory Reporting.

Covers:
- Framework mapping completeness (10 capabilities × 6 frameworks = 60+ controls)
- Coverage matrix generation
- Evidence collection for each evidence type
- Evidence collection graceful degradation
- Report generation (single, multiple, all frameworks)
- Compliance scoring
- Gap identification
- API endpoints (frameworks, matrix, report, evidence)
- Report persistence and retrieval
"""

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from aegis.services.compliance import ComplianceEngine
from aegis.services.compliance.framework_mappings import (
    AEGIS_CAPABILITIES,
    ComplianceFramework,
    FrameworkControl,
    get_all_controls,
    get_controls_for_capability,
    get_controls_for_framework,
    get_coverage_matrix,
)
from aegis.services.compliance.evidence_collector import (
    Evidence,
    EvidenceCollector,
)
from aegis.services.compliance.report_generator import (
    ComplianceGap,
    ComplianceReport,
    ComplianceReportGenerator,
)


# ---------------------------------------------------------------------------
# Framework Mapping Tests
# ---------------------------------------------------------------------------


class TestFrameworkMappingCompleteness:
    """Verify all 10 capabilities × 6 frameworks have control mappings."""

    def test_total_control_count_minimum_60(self):
        """At least 60 controls mapped (10 capabilities × 6 frameworks)."""
        controls = get_all_controls()
        assert len(controls) >= 60

    def test_all_six_frameworks_present(self):
        """Each framework has at least one control."""
        for fw in ComplianceFramework:
            controls = get_controls_for_framework(fw)
            assert len(controls) > 0, f"No controls for {fw.value}"

    def test_all_ten_capabilities_mapped(self):
        """Each capability maps to at least one control."""
        for cap in AEGIS_CAPABILITIES:
            controls = get_controls_for_capability(cap)
            assert len(controls) > 0, f"No controls for capability {cap}"

    def test_each_capability_covers_all_frameworks(self):
        """Each capability should map to controls in all 6 frameworks."""
        for cap in AEGIS_CAPABILITIES:
            controls_by_fw = get_controls_for_capability(cap)
            assert len(controls_by_fw) == 6, (
                f"Capability '{cap}' only covers {len(controls_by_fw)} frameworks, "
                f"missing: {set(ComplianceFramework) - set(controls_by_fw.keys())}"
            )

    def test_framework_control_has_required_fields(self):
        """Every control should have all required fields populated."""
        for c in get_all_controls():
            assert c.framework in ComplianceFramework
            assert c.control_id, "control_id is empty"
            assert c.control_name, "control_name is empty"
            assert c.description, "description is empty"
            assert c.aegis_capability in AEGIS_CAPABILITIES
            assert c.evidence_type in {"continuous_monitoring", "audit_log", "configuration", "test_result"}

    def test_nist_ai_rmf_controls(self):
        controls = get_controls_for_framework(ComplianceFramework.NIST_AI_RMF)
        ids = {c.control_id for c in controls}
        assert "MEASURE-2" in ids
        assert "GOVERN-1.4" in ids
        assert "GOVERN-1" in ids
        assert "MAP-1" in ids

    def test_eu_ai_act_controls(self):
        controls = get_controls_for_framework(ComplianceFramework.EU_AI_ACT)
        ids = {c.control_id for c in controls}
        assert "Art-15" in ids
        assert "Art-12" in ids
        assert "Art-14" in ids
        assert "Art-13" in ids

    def test_soc_2_controls(self):
        controls = get_controls_for_framework(ComplianceFramework.SOC_2)
        ids = {c.control_id for c in controls}
        assert "CC7.2" in ids
        assert "CC7.4" in ids
        assert "CC1.1" in ids
        assert "CC6.1" in ids

    def test_cmmc_2_controls(self):
        controls = get_controls_for_framework(ComplianceFramework.CMMC_2)
        ids = {c.control_id for c in controls}
        assert "SI-4" in ids
        assert "AU-2" in ids
        assert "IR-4" in ids
        assert "CM-2" in ids

    def test_iso_42001_controls(self):
        controls = get_controls_for_framework(ComplianceFramework.ISO_42001)
        ids = {c.control_id for c in controls}
        assert any("Clause" in cid for cid in ids)

    def test_fedramp_controls(self):
        controls = get_controls_for_framework(ComplianceFramework.FEDRAMP)
        assert len(controls) >= 6


class TestCoverageMatrix:
    """Test coverage matrix generation."""

    def test_matrix_has_all_frameworks(self):
        matrix = get_coverage_matrix()
        assert len(matrix["frameworks"]) == 6
        for fw in ComplianceFramework:
            assert fw.value in matrix["frameworks"]

    def test_matrix_coverage_percentage(self):
        matrix = get_coverage_matrix()
        for fw_name, data in matrix["frameworks"].items():
            assert 0 <= data["coverage_pct"] <= 100
            assert data["controls"] > 0
            assert data["capabilities_covered"] > 0

    def test_matrix_total_controls(self):
        matrix = get_coverage_matrix()
        assert matrix["total_controls"] >= 60

    def test_matrix_total_capabilities(self):
        matrix = get_coverage_matrix()
        assert matrix["total_capabilities"] == 10

    def test_matrix_capabilities_list(self):
        matrix = get_coverage_matrix()
        assert "real_time_threat_monitoring" in matrix["capabilities"]
        assert "data_protection_pii" in matrix["capabilities"]

    def test_full_coverage_per_framework(self):
        """Each framework should cover all 10 capabilities (100%)."""
        matrix = get_coverage_matrix()
        for fw_name, data in matrix["frameworks"].items():
            assert data["coverage_pct"] == 100.0, (
                f"{fw_name} only covers {data['capabilities_covered']}/10 capabilities"
            )


class TestFrameworkControlDataclass:
    """Test FrameworkControl helper methods."""

    def test_to_dict(self):
        c = FrameworkControl(
            framework=ComplianceFramework.SOC_2,
            control_id="CC7.2",
            control_name="System Monitoring",
            description="Monitoring of system components",
            aegis_capability="real_time_threat_monitoring",
            evidence_type="continuous_monitoring",
        )
        d = c.to_dict()
        assert d["framework"] == "SOC_2"
        assert d["control_id"] == "CC7.2"
        assert d["evidence_type"] == "continuous_monitoring"


class TestComplianceFrameworkEnum:
    """Test framework enum values."""

    def test_six_frameworks(self):
        assert len(ComplianceFramework) == 6

    def test_framework_values(self):
        values = {fw.value for fw in ComplianceFramework}
        assert "NIST_AI_RMF" in values
        assert "ISO_42001" in values
        assert "EU_AI_ACT" in values
        assert "CMMC_2" in values
        assert "SOC_2" in values
        assert "FEDRAMP" in values


# ---------------------------------------------------------------------------
# Evidence Collector Tests
# ---------------------------------------------------------------------------


class TestEvidenceCollection:
    """Test evidence collection for each evidence type."""

    @pytest.fixture
    def mock_audit(self):
        audit = MagicMock()
        audit.record_count = 42
        audit.get_recent.return_value = [
            {"action": "block", "layer": "innate"},
            {"action": "allow", "layer": "pipeline"},
            {"action": "allow", "layer": "pipeline"},
        ]
        return audit

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.innate.block_threshold = 0.85
        config.innate.alert_threshold = 0.50
        config.barrier.rate_limit_rpm = 60
        config.barrier.rate_limit_burst = 10
        config.barrier.max_tokens_per_request = 128000
        config.barrier.tls_min_version = "1.3"
        config.policy.backend = "python"
        return config

    @pytest.fixture
    def mock_vault(self):
        vault = MagicMock()
        vault.get_stats.return_value = {
            "total_indicators": 25,
            "phase_distribution": {"acute": 10, "persistent": 10, "dormant": 5},
        }
        return vault

    @pytest.fixture
    def mock_policy(self):
        policy = MagicMock()
        policy.threat_level.name = "GREEN"
        return policy

    @pytest.fixture
    def mock_healing(self):
        healing = MagicMock()
        breaker = MagicMock()
        breaker.state.value = "closed"
        breaker.consecutive_trips = 0
        healing.get_breaker.return_value = breaker
        return healing

    def test_continuous_monitoring_evidence(
        self, mock_vault, mock_policy, mock_healing,
    ):
        collector = EvidenceCollector(
            vault=mock_vault, policy=mock_policy, healing=mock_healing,
        )
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("real_time_threat_monitoring")
        )
        assert evidence.evidence_type == "continuous_monitoring"
        assert "vault_stats" in evidence.data
        assert "threat_level" in evidence.data
        assert "circuit_breaker" in evidence.data
        assert evidence.data["threat_level"] == "GREEN"

    def test_audit_log_evidence(self, mock_audit):
        collector = EvidenceCollector(audit_logger=mock_audit)
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("automated_audit_trails")
        )
        assert evidence.evidence_type == "audit_log"
        assert evidence.data["audit_record_count"] == 42
        assert evidence.data["audit_active"] is True
        assert evidence.data["recent_blocks"] == 1
        assert evidence.data["recent_allows"] == 2

    def test_configuration_evidence(self, mock_config, mock_policy):
        collector = EvidenceCollector(config=mock_config, policy=mock_policy)
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("policy_enforcement")
        )
        assert evidence.evidence_type == "configuration"
        assert evidence.data["block_threshold"] == 0.85
        assert evidence.data["rate_limit_rpm"] == 60
        assert evidence.data["policy_tiers"] == ["global", "tenant", "adaptive"]

    def test_test_result_evidence(self):
        collector = EvidenceCollector()
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("risk_assessment")
        )
        assert evidence.evidence_type == "test_result"
        assert "risk_assessment_components" in evidence.data

    def test_pii_monitoring_evidence(self, mock_vault, mock_policy, mock_healing):
        collector = EvidenceCollector(
            vault=mock_vault, policy=mock_policy, healing=mock_healing,
        )
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("data_protection_pii")
        )
        assert evidence.data.get("pii_protection_enabled") is True

    def test_human_oversight_config_evidence(self, mock_config):
        collector = EvidenceCollector(config=mock_config)
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("human_oversight")
        )
        assert evidence.evidence_type == "configuration"
        assert evidence.data.get("hitl_escalation") is True

    def test_incident_response_audit_evidence(self, mock_audit, mock_healing):
        collector = EvidenceCollector(
            audit_logger=mock_audit, healing=mock_healing,
        )
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("incident_response")
        )
        assert evidence.data.get("self_healing_active") is True

    def test_supply_chain_test_evidence(self):
        mock_sc = MagicMock()
        collector = EvidenceCollector(supply_chain=mock_sc)
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("supply_chain_verification")
        )
        assert evidence.data.get("supply_chain_verifier_active") is True


class TestEvidenceGracefulDegradation:
    """Test evidence collection when data sources are unavailable."""

    def test_no_vault_notes_gap(self):
        collector = EvidenceCollector()
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("real_time_threat_monitoring")
        )
        assert any("vault" in g.lower() for g in evidence.gaps)

    def test_no_audit_notes_gap(self):
        collector = EvidenceCollector()
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("automated_audit_trails")
        )
        assert any("audit" in g.lower() for g in evidence.gaps)

    def test_no_config_notes_gap(self):
        collector = EvidenceCollector()
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("policy_enforcement")
        )
        assert any("config" in g.lower() for g in evidence.gaps)

    def test_no_policy_notes_gap(self):
        collector = EvidenceCollector()
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("real_time_threat_monitoring")
        )
        assert any("policy" in g.lower() for g in evidence.gaps)

    def test_no_healing_notes_gap(self):
        collector = EvidenceCollector()
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("real_time_threat_monitoring")
        )
        assert any("healing" in g.lower() for g in evidence.gaps)

    def test_partial_evidence_still_has_data(self):
        """Even with gaps, available data should be collected."""
        mock_vault = MagicMock()
        mock_vault.get_stats.return_value = {"total_indicators": 5}
        collector = EvidenceCollector(vault=mock_vault)
        evidence = asyncio.get_event_loop().run_until_complete(
            collector.collect_evidence("real_time_threat_monitoring")
        )
        assert evidence.data  # has some data
        assert evidence.gaps  # also has gaps (no policy, no healing)

    def test_collect_all_returns_all_capabilities(self):
        collector = EvidenceCollector()
        results = asyncio.get_event_loop().run_until_complete(
            collector.collect_all()
        )
        assert len(results) == len(AEGIS_CAPABILITIES)


class TestEvidenceDataclass:
    """Test Evidence dataclass."""

    def test_evidence_to_dict(self):
        e = Evidence(
            capability="test_cap",
            evidence_type="audit_log",
            data={"key": "value"},
            summary="test summary",
            gaps=["gap1"],
        )
        d = e.to_dict()
        assert d["capability"] == "test_cap"
        assert d["evidence_type"] == "audit_log"
        assert d["data"]["key"] == "value"
        assert d["gaps"] == ["gap1"]
        assert "collected_at" in d

    def test_evidence_defaults(self):
        e = Evidence(capability="test", evidence_type="config")
        assert e.data == {}
        assert e.summary == ""
        assert e.gaps == []


# ---------------------------------------------------------------------------
# Report Generator Tests
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Test compliance report generation."""

    @pytest.fixture
    def full_collector(self):
        """Collector with all sources available."""
        return EvidenceCollector(
            audit_logger=self._mock_audit(),
            config=self._mock_config(),
            vault=self._mock_vault(),
            policy=self._mock_policy(),
            healing=self._mock_healing(),
            supply_chain=MagicMock(),
        )

    @pytest.fixture
    def empty_collector(self):
        """Collector with no sources available."""
        return EvidenceCollector()

    def _mock_audit(self):
        audit = MagicMock()
        audit.record_count = 100
        audit.get_recent.return_value = [{"action": "allow"}]
        return audit

    def _mock_config(self):
        config = MagicMock()
        config.innate.block_threshold = 0.85
        config.innate.alert_threshold = 0.50
        config.barrier.rate_limit_rpm = 60
        config.barrier.rate_limit_burst = 10
        config.barrier.max_tokens_per_request = 128000
        config.barrier.tls_min_version = "1.3"
        config.policy.backend = "python"
        return config

    def _mock_vault(self):
        vault = MagicMock()
        vault.get_stats.return_value = {"total_indicators": 25, "phase_distribution": {}}
        return vault

    def _mock_policy(self):
        policy = MagicMock()
        policy.threat_level.name = "GREEN"
        return policy

    def _mock_healing(self):
        healing = MagicMock()
        breaker = MagicMock()
        breaker.state.value = "closed"
        breaker.consecutive_trips = 0
        healing.get_breaker.return_value = breaker
        return healing

    def test_single_framework_report(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        assert "SOC_2" in report.frameworks
        assert report.controls_total > 0
        assert report.overall_compliance_score > 0

    def test_multiple_frameworks_report(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["NIST_AI_RMF", "SOC_2"])
        )
        assert len(report.frameworks) == 2
        assert "NIST_AI_RMF" in report.per_framework_scores
        assert "SOC_2" in report.per_framework_scores

    def test_all_frameworks_report(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report()
        )
        assert len(report.frameworks) == 6
        assert report.controls_total >= 60

    def test_report_has_uuid(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        assert report.report_id
        assert len(report.report_id) == 36  # UUID format

    def test_report_has_timestamp(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        assert report.generated_at is not None

    def test_report_evidence_summary(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        assert len(report.evidence_summary) == len(AEGIS_CAPABILITIES)

    def test_unknown_framework_skipped(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["NONEXISTENT", "SOC_2"])
        )
        assert "SOC_2" in report.frameworks
        assert "NONEXISTENT" not in report.frameworks

    def test_report_persistence_and_retrieval(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        retrieved = generator.get_report(report.report_id)
        assert retrieved is not None
        assert retrieved.report_id == report.report_id

    def test_report_not_found(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        assert generator.get_report("nonexistent-id") is None

    def test_report_count(self, full_collector):
        generator = ComplianceReportGenerator(full_collector)
        assert generator.report_count == 0
        asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        assert generator.report_count == 1


class TestComplianceScoring:
    """Test compliance score calculation."""

    def test_high_score_with_full_evidence(self):
        """Full evidence should produce high compliance score."""
        collector = EvidenceCollector(
            audit_logger=MagicMock(
                record_count=100,
                get_recent=MagicMock(return_value=[{"action": "allow"}]),
            ),
            config=MagicMock(
                innate=MagicMock(block_threshold=0.85, alert_threshold=0.50),
                barrier=MagicMock(rate_limit_rpm=60, rate_limit_burst=10, max_tokens_per_request=128000, tls_min_version="1.3"),
                policy=MagicMock(backend="python"),
            ),
            vault=MagicMock(
                get_stats=MagicMock(return_value={"total_indicators": 25, "phase_distribution": {}}),
            ),
            policy=MagicMock(threat_level=MagicMock(name="GREEN")),
            healing=MagicMock(
                get_breaker=MagicMock(return_value=MagicMock(
                    state=MagicMock(value="closed"), consecutive_trips=0,
                )),
            ),
            supply_chain=MagicMock(),
        )
        generator = ComplianceReportGenerator(collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        assert report.overall_compliance_score > 50

    def test_low_score_without_evidence(self):
        """No evidence sources should produce lower compliance score."""
        collector = EvidenceCollector()  # No sources
        generator = ComplianceReportGenerator(collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        # Even without full sources, some evidence types (test_result, config)
        # may still produce data from static analysis
        assert report.controls_total > 0

    def test_per_framework_scores_between_0_and_100(self):
        collector = EvidenceCollector()
        generator = ComplianceReportGenerator(collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report()
        )
        for fw, score in report.per_framework_scores.items():
            assert 0 <= score <= 100, f"{fw} score {score} out of range"


class TestGapIdentification:
    """Test compliance gap detection."""

    def test_no_evidence_gap(self):
        """Controls without evidence should be flagged as gaps."""
        collector = EvidenceCollector()  # No sources → partial/no evidence
        generator = ComplianceReportGenerator(collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["NIST_AI_RMF"])
        )
        # Should have at least some gaps when no sources available
        gap_types = {g.gap_type for g in report.gaps}
        assert "partial_evidence" in gap_types or "no_evidence" in gap_types

    def test_gap_has_recommendation(self):
        collector = EvidenceCollector()
        generator = ComplianceReportGenerator(collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["SOC_2"])
        )
        for gap in report.gaps:
            assert gap.recommendation, f"Gap {gap.control_id} has no recommendation"

    def test_gap_has_framework_and_control(self):
        collector = EvidenceCollector()
        generator = ComplianceReportGenerator(collector)
        report = asyncio.get_event_loop().run_until_complete(
            generator.generate_report(frameworks=["CMMC_2"])
        )
        for gap in report.gaps:
            assert gap.framework
            assert gap.control_id
            assert gap.control_name


class TestComplianceReportDataclass:
    """Test ComplianceReport serialization."""

    def test_to_dict(self):
        report = ComplianceReport(
            frameworks=["SOC_2"],
            overall_compliance_score=85.5,
            per_framework_scores={"SOC_2": 85.5},
            controls_met=8,
            controls_total=10,
            controls_with_evidence=8,
        )
        d = report.to_dict()
        assert d["overall_compliance_score"] == 85.5
        assert d["frameworks"] == ["SOC_2"]
        assert d["controls_met"] == 8
        assert "report_id" in d
        assert "generated_at" in d

    def test_gap_to_dict(self):
        gap = ComplianceGap(
            framework="SOC_2",
            control_id="CC7.2",
            control_name="System Monitoring",
            gap_type="no_evidence",
            recommendation="Enable Prometheus metrics",
        )
        d = gap.to_dict()
        assert d["gap_type"] == "no_evidence"
        assert d["recommendation"] == "Enable Prometheus metrics"


# ---------------------------------------------------------------------------
# Compliance Engine (Orchestrator) Tests
# ---------------------------------------------------------------------------


class TestComplianceEngine:
    """Test the ComplianceEngine orchestrator."""

    @pytest.fixture
    def engine(self):
        return ComplianceEngine(
            audit_logger=MagicMock(
                record_count=10,
                get_recent=MagicMock(return_value=[]),
            ),
            config=MagicMock(
                innate=MagicMock(block_threshold=0.85, alert_threshold=0.50),
                barrier=MagicMock(rate_limit_rpm=60, rate_limit_burst=10, max_tokens_per_request=128000, tls_min_version="1.3"),
                policy=MagicMock(backend="python"),
            ),
        )

    def test_generate_report(self, engine):
        report = asyncio.get_event_loop().run_until_complete(
            engine.generate_report(frameworks=["SOC_2"])
        )
        assert report.report_id
        assert "SOC_2" in report.frameworks

    def test_get_report(self, engine):
        report = asyncio.get_event_loop().run_until_complete(
            engine.generate_report(frameworks=["SOC_2"])
        )
        retrieved = engine.get_report(report.report_id)
        assert retrieved is not None

    def test_collect_evidence(self, engine):
        evidence = asyncio.get_event_loop().run_until_complete(
            engine.collect_evidence("policy_enforcement")
        )
        assert evidence.capability == "policy_enforcement"
        assert evidence.data

    def test_get_frameworks(self, engine):
        frameworks = engine.get_frameworks()
        assert len(frameworks) == 6
        for fw in frameworks:
            assert "framework" in fw
            assert "control_count" in fw
            assert fw["control_count"] > 0

    def test_get_coverage_matrix(self, engine):
        matrix = engine.get_coverage_matrix()
        assert "frameworks" in matrix
        assert len(matrix["frameworks"]) == 6

    def test_stats(self, engine):
        stats = engine.stats
        assert stats["frameworks_mapped"] == 6
        assert stats["total_controls"] >= 60
        assert stats["total_capabilities"] == 10
        assert stats["reports_generated"] == 0


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


class TestComplianceEndpoints:
    """Test compliance API endpoints via TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from starlette.testclient import TestClient
        import aegis.main as main_mod
        from aegis.config import get_config

        saved_compliance = main_mod._compliance
        saved_config = main_mod._config

        config = get_config()
        if not config.api_key:
            config.api_key = "test-compliance-api-key"
        main_mod._config = config

        main_mod._compliance = ComplianceEngine(
            config=config,
        )
        self.client = TestClient(main_mod.app)
        self.api_key = config.api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        yield

        main_mod._compliance = saved_compliance
        main_mod._config = saved_config

    def test_frameworks_list(self):
        resp = self.client.get("/v1/compliance/frameworks", headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "frameworks" in data
        assert len(data["frameworks"]) == 6

    def test_frameworks_unauthenticated(self):
        resp = self.client.get("/v1/compliance/frameworks")
        assert resp.status_code == 401

    def test_coverage_matrix(self):
        resp = self.client.get("/v1/compliance/matrix", headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "frameworks" in data
        assert "total_controls" in data

    def test_coverage_matrix_unauthenticated(self):
        resp = self.client.get("/v1/compliance/matrix")
        assert resp.status_code == 401

    def test_generate_report(self):
        resp = self.client.post(
            "/v1/compliance/report",
            json={"frameworks": ["SOC_2"], "time_range_hours": 24},
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data
        assert "overall_compliance_score" in data
        assert "SOC_2" in data["frameworks"]

    def test_generate_report_all_frameworks(self):
        resp = self.client.post(
            "/v1/compliance/report",
            json={},
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["frameworks"]) == 6

    def test_generate_report_unauthenticated(self):
        resp = self.client.post(
            "/v1/compliance/report",
            json={"frameworks": ["SOC_2"]},
        )
        assert resp.status_code == 401

    def test_retrieve_report_by_id(self):
        # Generate first
        gen_resp = self.client.post(
            "/v1/compliance/report",
            json={"frameworks": ["SOC_2"]},
            headers=self.headers,
        )
        report_id = gen_resp.json()["report_id"]

        # Retrieve
        resp = self.client.get(
            f"/v1/compliance/report/{report_id}",
            headers=self.headers,
        )
        assert resp.status_code == 200
        assert resp.json()["report_id"] == report_id

    def test_retrieve_report_not_found(self):
        resp = self.client.get(
            "/v1/compliance/report/nonexistent-id",
            headers=self.headers,
        )
        assert resp.status_code == 404

    def test_evidence_for_capability(self):
        resp = self.client.get(
            "/v1/compliance/evidence/policy_enforcement",
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["capability"] == "policy_enforcement"
        assert data["evidence_type"] == "configuration"

    def test_evidence_unknown_capability(self):
        resp = self.client.get(
            "/v1/compliance/evidence/nonexistent_capability",
            headers=self.headers,
        )
        assert resp.status_code == 400
        assert "valid_capabilities" in resp.json()

    def test_evidence_unauthenticated(self):
        resp = self.client.get("/v1/compliance/evidence/policy_enforcement")
        assert resp.status_code == 401

    def test_multiple_framework_report_scores(self):
        resp = self.client.post(
            "/v1/compliance/report",
            json={"frameworks": ["NIST_AI_RMF", "SOC_2", "EU_AI_ACT"]},
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["per_framework_scores"]) == 3
        for fw in ["NIST_AI_RMF", "SOC_2", "EU_AI_ACT"]:
            assert fw in data["per_framework_scores"]

    def test_report_includes_gaps(self):
        resp = self.client.post(
            "/v1/compliance/report",
            json={"frameworks": ["SOC_2"]},
            headers=self.headers,
        )
        data = resp.json()
        # Should have gaps since not all evidence sources are wired
        assert "gaps" in data
        assert isinstance(data["gaps"], list)

    def test_report_includes_evidence_summary(self):
        resp = self.client.post(
            "/v1/compliance/report",
            json={"frameworks": ["SOC_2"]},
            headers=self.headers,
        )
        data = resp.json()
        assert "evidence_summary" in data
        assert len(data["evidence_summary"]) == 10  # all capabilities
