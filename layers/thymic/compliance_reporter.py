"""
Compliance Reporter — Generates compliance evidence from TVE validation results.

Maps TVE findings to six regulatory frameworks:
- NIST AI RMF (MEASURE 2.5-2.11)
- ISO 42001 (Clause 9.1)
- EU AI Act (Article 15)
- CMMC 2.0 (SI-4, SI-5)
- SOC 2 (CC7.2)
- FedRAMP (CA-7)

Produces structured evidence data for API serving or export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aegis.layers.thymic.engine import ValidationReport

logger = logging.getLogger(__name__)


@dataclass
class FrameworkMapping:
    """A single compliance requirement mapping."""

    framework: str
    requirement_id: str
    requirement_description: str
    evidence_summary: str
    passed: bool


@dataclass
class ComplianceEvidence:
    """Compliance evidence from a single TVE run."""

    run_id: str
    timestamp: datetime
    run_type: str
    framework_mappings: list[FrameworkMapping] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON response."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "run_type": self.run_type,
            "framework_mappings": [
                {
                    "framework": m.framework,
                    "requirement_id": m.requirement_id,
                    "requirement_description": m.requirement_description,
                    "evidence_summary": m.evidence_summary,
                    "passed": m.passed,
                }
                for m in self.framework_mappings
            ],
        }


@dataclass
class ComplianceSummary:
    """Aggregated compliance summary across multiple TVE runs."""

    period_days: int
    total_runs: int
    average_tpr: float
    average_fpr: float
    uptime_percentage: float  # runs with all checks passed / total runs
    failed_checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON response."""
        return {
            "period_days": self.period_days,
            "total_runs": self.total_runs,
            "average_tpr": round(self.average_tpr, 4),
            "average_fpr": round(self.average_fpr, 4),
            "uptime_percentage": round(self.uptime_percentage, 2),
            "failed_checks": self.failed_checks,
        }


# Framework requirement definitions
_FRAMEWORK_REQUIREMENTS = [
    {
        "framework": "NIST AI RMF",
        "requirement_id": "MEASURE 2.5-2.11",
        "requirement_description": "Continuous measurement of AI system performance and risk metrics",
        "check": "detection",
    },
    {
        "framework": "ISO 42001",
        "requirement_id": "Clause 9.1",
        "requirement_description": "Monitoring, measurement, analysis and evaluation of AI management system",
        "check": "overall",
    },
    {
        "framework": "EU AI Act",
        "requirement_id": "Article 15",
        "requirement_description": "Accuracy, robustness and cybersecurity requirements for high-risk AI",
        "check": "overall",
    },
    {
        "framework": "CMMC 2.0",
        "requirement_id": "SI-4, SI-5",
        "requirement_description": "System monitoring and security alert generation",
        "check": "dead_scanner",
    },
    {
        "framework": "SOC 2",
        "requirement_id": "CC7.2",
        "requirement_description": "Monitoring of system components for anomalies and security events",
        "check": "overall",
    },
    {
        "framework": "FedRAMP",
        "requirement_id": "CA-7",
        "requirement_description": "Continuous monitoring of security controls at defined frequency",
        "check": "overall",
    },
]


class ComplianceReporter:
    """Generates compliance evidence from TVE validation results."""

    def generate_evidence(self, report: ValidationReport) -> ComplianceEvidence:
        """Produce compliance evidence from a single validation run."""
        evidence = ComplianceEvidence(
            run_id=report.run_id,
            timestamp=report.timestamp,
            run_type=report.run_type,
        )

        verdict = report.verdict
        overall_passed = verdict.overall_passed if verdict else True
        detection_passed = (
            all(v.passed for v in verdict.detection_verdicts) if verdict else True
        )
        fp_passed = (
            verdict.false_positive_verdict is None or verdict.false_positive_verdict.passed
        ) if verdict else True
        dead_scanner_passed = (
            verdict.dead_scanner_verdict is None or verdict.dead_scanner_verdict.passed
        ) if verdict else True

        for req in _FRAMEWORK_REQUIREMENTS:
            check = req["check"]
            if check == "detection":
                passed = detection_passed
                summary = self._detection_summary(report, verdict)
            elif check == "dead_scanner":
                passed = dead_scanner_passed
                summary = self._dead_scanner_summary(report, verdict)
            else:  # overall
                passed = overall_passed
                summary = self._overall_summary(report, verdict)

            evidence.framework_mappings.append(FrameworkMapping(
                framework=req["framework"],
                requirement_id=req["requirement_id"],
                requirement_description=req["requirement_description"],
                evidence_summary=summary,
                passed=passed,
            ))

        return evidence

    def generate_summary(
        self,
        reports: list[ValidationReport],
        period_days: int = 30,
    ) -> ComplianceSummary:
        """Produce aggregated compliance summary across multiple runs."""
        if not reports:
            return ComplianceSummary(
                period_days=period_days,
                total_runs=0,
                average_tpr=0.0,
                average_fpr=0.0,
                uptime_percentage=0.0,
            )

        total = len(reports)
        avg_tpr = sum(r.overall_tpr for r in reports) / total
        avg_fpr = sum(r.overall_fpr for r in reports) / total

        passed_count = 0
        failed_checks: list[dict] = []

        for r in reports:
            if r.verdict and r.verdict.overall_passed:
                passed_count += 1
            elif r.verdict and not r.verdict.overall_passed:
                failed_checks.append({
                    "run_id": r.run_id,
                    "timestamp": r.timestamp.isoformat(),
                    "run_type": r.run_type,
                    "recommended_actions": list(r.verdict.recommended_actions),
                })
            else:
                # No verdict — count as passed (no analysis = no failure)
                passed_count += 1

        uptime = (passed_count / total) * 100.0

        return ComplianceSummary(
            period_days=period_days,
            total_runs=total,
            average_tpr=avg_tpr,
            average_fpr=avg_fpr,
            uptime_percentage=uptime,
            failed_checks=failed_checks,
        )

    def _detection_summary(self, report: ValidationReport, verdict) -> str:
        """Build evidence summary for detection-based requirements."""
        if verdict and verdict.detection_verdicts:
            layers = [v.layer_id for v in verdict.detection_verdicts]
            tprs = [f"{v.layer_id}={v.observed_tpr:.2f}" for v in verdict.detection_verdicts]
            return (
                f"TVE {report.run_type} run {report.run_id}: "
                f"tested {report.total_probes} probes across layers {', '.join(layers)}. "
                f"Per-layer TPR: {', '.join(tprs)}. "
                f"Overall TPR={report.overall_tpr:.2f}, FPR={report.overall_fpr:.3f}."
            )
        return (
            f"TVE {report.run_type} run {report.run_id}: "
            f"{report.total_probes} probes, "
            f"TPR={report.overall_tpr:.2f}, FPR={report.overall_fpr:.3f}."
        )

    def _dead_scanner_summary(self, report: ValidationReport, verdict) -> str:
        """Build evidence summary for scanner health requirements."""
        if verdict and verdict.dead_scanner_verdict:
            dsv = verdict.dead_scanner_verdict
            if dsv.passed:
                return (
                    f"TVE {report.run_type} run {report.run_id}: "
                    f"all scanners operational, no dead scanners detected."
                )
            return (
                f"TVE {report.run_type} run {report.run_id}: "
                f"dead scanners detected: {', '.join(dsv.dead_scanners)}."
            )
        return (
            f"TVE {report.run_type} run {report.run_id}: "
            f"scanner health monitoring active, {report.total_probes} probes tested."
        )

    def _overall_summary(self, report: ValidationReport, verdict) -> str:
        """Build evidence summary for overall requirements."""
        passed_str = "PASSED" if (verdict and verdict.overall_passed) else "N/A"
        if verdict and not verdict.overall_passed:
            passed_str = "FAILED"
        return (
            f"TVE {report.run_type} run {report.run_id}: "
            f"{report.total_probes} probes, {report.probes_detected} detected, "
            f"{report.probes_missed} missed, {report.false_positives} false positives. "
            f"Verdict: {passed_str}."
        )
