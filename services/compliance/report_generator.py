"""
Compliance Report Generator — Cross-Framework Compliance Reporting.

Generates compliance reports that map AEGIS capabilities to controls
across multiple regulatory frameworks simultaneously. Reports include
compliance scores, evidence summaries, and gap analysis.

A single dashboard shows compliance status across NIST AI RMF, ISO 42001,
EU AI Act, CMMC 2.0, SOC 2, and FedRAMP — reducing compliance costs 40-60%.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aegis.services.compliance.evidence_collector import Evidence, EvidenceCollector
from aegis.services.compliance.framework_mappings import (
    ComplianceFramework,
    FrameworkControl,
    get_all_controls,
    get_controls_for_framework,
)

logger = logging.getLogger(__name__)


@dataclass
class ComplianceGap:
    """A gap in compliance coverage."""
    framework: str
    control_id: str
    control_name: str
    gap_type: str  # "no_evidence", "partial_evidence", "not_implemented"
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "control_id": self.control_id,
            "control_name": self.control_name,
            "gap_type": self.gap_type,
            "recommendation": self.recommendation,
        }


@dataclass
class ComplianceReport:
    """Complete cross-framework compliance report."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frameworks: list[str] = field(default_factory=list)
    overall_compliance_score: float = 0.0
    per_framework_scores: dict[str, float] = field(default_factory=dict)
    controls_met: int = 0
    controls_total: int = 0
    controls_with_evidence: int = 0
    gaps: list[ComplianceGap] = field(default_factory=list)
    evidence_summary: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "frameworks": self.frameworks,
            "overall_compliance_score": round(self.overall_compliance_score, 1),
            "per_framework_scores": {
                k: round(v, 1) for k, v in self.per_framework_scores.items()
            },
            "controls_met": self.controls_met,
            "controls_total": self.controls_total,
            "controls_with_evidence": self.controls_with_evidence,
            "gaps": [g.to_dict() for g in self.gaps],
            "evidence_summary": [e.to_dict() for e in self.evidence_summary],
        }


# Recommendations for common gap types
_GAP_RECOMMENDATIONS: dict[str, str] = {
    "continuous_monitoring": "Enable Prometheus metrics collection and configure alerting rules",
    "audit_log": "Enable PostgreSQL audit logging or verify JSONL audit logger is active",
    "configuration": "Review and document current AEGIS configuration settings",
    "test_result": "Run the 20-attack simulation battery and benchmark suite",
}


class ComplianceReportGenerator:
    """Generates compliance reports from evidence and framework mappings."""

    def __init__(self, evidence_collector: EvidenceCollector) -> None:
        self._collector = evidence_collector
        self._reports: dict[str, ComplianceReport] = {}

    async def generate_report(
        self,
        frameworks: list[str] | None = None,
        time_range_hours: int = 24,
    ) -> ComplianceReport:
        """Generate a compliance report for specified frameworks.

        Args:
            frameworks: List of framework names (e.g., ["NIST_AI_RMF", "SOC_2"]).
                       If None, generates for all frameworks.
            time_range_hours: How far back to collect evidence.

        Returns:
            ComplianceReport with scores, evidence, and gap analysis.
        """
        # Resolve frameworks
        if frameworks:
            selected = []
            for name in frameworks:
                try:
                    selected.append(ComplianceFramework(name))
                except ValueError:
                    logger.warning("Unknown framework: %s (skipping)", name)
        else:
            selected = list(ComplianceFramework)

        report = ComplianceReport(
            frameworks=[fw.value for fw in selected],
        )

        # Collect evidence for all capabilities
        all_evidence = await self._collector.collect_all(time_range_hours)
        evidence_by_capability = {e.capability: e for e in all_evidence}
        report.evidence_summary = all_evidence

        # Evaluate each framework
        total_controls = 0
        total_with_evidence = 0

        for fw in selected:
            controls = get_controls_for_framework(fw)
            fw_total = len(controls)
            fw_with_evidence = 0

            for control in controls:
                total_controls += 1
                evidence = evidence_by_capability.get(control.aegis_capability)

                if evidence and evidence.data and not evidence.gaps:
                    # Full evidence available
                    fw_with_evidence += 1
                    total_with_evidence += 1
                elif evidence and evidence.data and evidence.gaps:
                    # Partial evidence
                    fw_with_evidence += 1
                    total_with_evidence += 1
                    report.gaps.append(ComplianceGap(
                        framework=fw.value,
                        control_id=control.control_id,
                        control_name=control.control_name,
                        gap_type="partial_evidence",
                        recommendation=f"Address gaps: {'; '.join(evidence.gaps)}",
                    ))
                else:
                    # No evidence
                    report.gaps.append(ComplianceGap(
                        framework=fw.value,
                        control_id=control.control_id,
                        control_name=control.control_name,
                        gap_type="no_evidence",
                        recommendation=_GAP_RECOMMENDATIONS.get(
                            control.evidence_type,
                            "Collect evidence for this control",
                        ),
                    ))

            # Per-framework score
            fw_score = (fw_with_evidence / fw_total * 100) if fw_total > 0 else 0
            report.per_framework_scores[fw.value] = fw_score

        report.controls_total = total_controls
        report.controls_with_evidence = total_with_evidence
        report.controls_met = total_with_evidence  # met = has evidence
        report.overall_compliance_score = (
            (total_with_evidence / total_controls * 100) if total_controls > 0 else 0
        )

        # Store for retrieval
        self._reports[report.report_id] = report
        logger.info(
            "Compliance report %s generated: %.1f%% overall (%d/%d controls)",
            report.report_id,
            report.overall_compliance_score,
            total_with_evidence,
            total_controls,
        )

        return report

    def get_report(self, report_id: str) -> ComplianceReport | None:
        """Retrieve a previously generated report by ID."""
        return self._reports.get(report_id)

    @property
    def report_count(self) -> int:
        """Number of reports stored."""
        return len(self._reports)
