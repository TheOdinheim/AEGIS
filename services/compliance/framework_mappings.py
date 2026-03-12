"""
Framework Mappings — Cross-Framework Control Matrix.

Maps AEGIS capabilities to controls across six regulatory frameworks:
NIST AI RMF, ISO 42001, EU AI Act, CMMC 2.0, SOC 2, and FedRAMP.

A single AEGIS capability satisfies controls across 5-6 frameworks
simultaneously, reducing compliance costs 40-60%.

Reference: CLAUDE.md Section 14 — Compliance Mapping Matrix.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ComplianceFramework(str, enum.Enum):
    """Supported regulatory frameworks."""
    NIST_AI_RMF = "NIST_AI_RMF"
    ISO_42001 = "ISO_42001"
    EU_AI_ACT = "EU_AI_ACT"
    CMMC_2 = "CMMC_2"
    SOC_2 = "SOC_2"
    FEDRAMP = "FEDRAMP"


# Evidence collection method for each control
EVIDENCE_TYPES = {
    "continuous_monitoring",
    "audit_log",
    "configuration",
    "test_result",
}


@dataclass
class FrameworkControl:
    """A single control mapping from AEGIS capability to framework requirement."""
    framework: ComplianceFramework
    control_id: str
    control_name: str
    description: str
    aegis_capability: str
    evidence_type: str  # one of EVIDENCE_TYPES

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework.value,
            "control_id": self.control_id,
            "control_name": self.control_name,
            "description": self.description,
            "aegis_capability": self.aegis_capability,
            "evidence_type": self.evidence_type,
        }


# ---------------------------------------------------------------------------
# Complete cross-framework control mappings (Section 14)
# ---------------------------------------------------------------------------

# Each AEGIS capability maps to controls across all six frameworks.
# 10 capabilities × 6 frameworks = 60 control mappings.

_CONTROL_MAPPINGS: list[FrameworkControl] = [
    # --- Real-time threat monitoring ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MEASURE-2",
        control_name="AI Risk Measurement",
        description="Continuous monitoring of AI system threats and anomalies",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MANAGE-4",
        control_name="AI Risk Management",
        description="Ongoing threat management and response for AI systems",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-9",
        control_name="Performance Evaluation",
        description="Monitoring and measurement of AI management system performance",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-15",
        control_name="Accuracy, Robustness, Cybersecurity",
        description="Real-time monitoring of high-risk AI system performance",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="SI-4",
        control_name="System Monitoring",
        description="Information system monitoring for threats and anomalies",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="SI-5",
        control_name="Security Alerts and Advisories",
        description="Receipt and response to security alerts",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC7.2",
        control_name="System Monitoring",
        description="Monitoring of system components for anomalies and threats",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="SI-4",
        control_name="Information System Monitoring",
        description="Federal system monitoring for unauthorized access and threats",
        aegis_capability="real_time_threat_monitoring",
        evidence_type="continuous_monitoring",
    ),

    # --- Automated audit trails ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="GOVERN-1.4",
        control_name="AI Governance Documentation",
        description="Automated documentation and audit trail of AI system decisions",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-9.1",
        control_name="Monitoring, Measurement, Analysis",
        description="Documented evidence of AI system monitoring and measurement",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-12",
        control_name="Record-Keeping",
        description="Automatic logging of high-risk AI system events",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-19",
        control_name="Quality Management System",
        description="Documentation requirements for AI quality management",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="AU-2",
        control_name="Audit Events",
        description="Definition and logging of auditable events",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="AU-6",
        control_name="Audit Review, Analysis, Reporting",
        description="Review and analysis of audit records",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC7.2-AUDIT",
        control_name="System Operation Monitoring",
        description="Monitoring and logging of system operations",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC7.3",
        control_name="Change Detection",
        description="Detection and logging of changes to system configuration",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="AU-2",
        control_name="Audit Events",
        description="Federal audit event identification and logging",
        aegis_capability="automated_audit_trails",
        evidence_type="audit_log",
    ),

    # --- Anomaly / drift detection ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MEASURE-2.6",
        control_name="AI Performance Measurement",
        description="Detection of performance anomalies and distribution drift",
        aegis_capability="anomaly_drift_detection",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-8.2",
        control_name="AI Risk Assessment",
        description="Ongoing risk assessment including anomaly detection",
        aegis_capability="anomaly_drift_detection",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-9",
        control_name="Risk Management System",
        description="Continuous risk management including anomaly detection",
        aegis_capability="anomaly_drift_detection",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="SI-4-DRIFT",
        control_name="System Monitoring — Anomaly Detection",
        description="Detection of anomalous patterns and behavioral drift",
        aegis_capability="anomaly_drift_detection",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="PI1.3",
        control_name="Processing Integrity Monitoring",
        description="Monitoring for processing integrity anomalies",
        aegis_capability="anomaly_drift_detection",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="SI-4-ANOMALY",
        control_name="Information System Monitoring — Anomaly Detection",
        description="Federal system anomaly detection and drift monitoring",
        aegis_capability="anomaly_drift_detection",
        evidence_type="continuous_monitoring",
    ),

    # --- Policy enforcement ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="GOVERN-1",
        control_name="AI Governance Policies",
        description="Enforcement of AI governance policies and controls",
        aegis_capability="policy_enforcement",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-5",
        control_name="Leadership and Commitment",
        description="Policy framework for AI management system",
        aegis_capability="policy_enforcement",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-17",
        control_name="Quality Management System",
        description="Policy enforcement within quality management framework",
        aegis_capability="policy_enforcement",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="CM-2",
        control_name="Baseline Configuration",
        description="Configuration baseline and enforcement",
        aegis_capability="policy_enforcement",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="CM-6",
        control_name="Configuration Settings",
        description="Security configuration settings enforcement",
        aegis_capability="policy_enforcement",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC1.1",
        control_name="Control Environment",
        description="Organizational control environment and policy enforcement",
        aegis_capability="policy_enforcement",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="CM-2",
        control_name="Baseline Configuration",
        description="Federal baseline configuration and enforcement",
        aegis_capability="policy_enforcement",
        evidence_type="configuration",
    ),

    # --- Incident response ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MANAGE-1",
        control_name="AI Risk Management Process",
        description="Automated incident detection and response for AI systems",
        aegis_capability="incident_response",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-10",
        control_name="Improvement",
        description="Incident handling and corrective action processes",
        aegis_capability="incident_response",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-20",
        control_name="Corrective Actions",
        description="Automated corrective actions for AI system incidents",
        aegis_capability="incident_response",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="IR-4",
        control_name="Incident Handling",
        description="Incident detection, analysis, containment, and recovery",
        aegis_capability="incident_response",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="IR-5",
        control_name="Incident Monitoring",
        description="Tracking and documenting security incidents",
        aegis_capability="incident_response",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC7.4",
        control_name="Incident Response",
        description="Incident response procedures and execution",
        aegis_capability="incident_response",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="IR-4",
        control_name="Incident Handling",
        description="Federal incident handling and response procedures",
        aegis_capability="incident_response",
        evidence_type="audit_log",
    ),

    # --- Supply chain verification ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MAP-3",
        control_name="AI Context Mapping",
        description="AI supply chain risk identification and mapping",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MANAGE-3",
        control_name="AI Third-Party Risk",
        description="Third-party AI component risk management",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-8.3",
        control_name="AI Risk Treatment",
        description="Treatment of AI supply chain risks",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-15-SC",
        control_name="Accuracy — Supply Chain",
        description="Verification of third-party AI components",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="SA-9",
        control_name="External System Services",
        description="Security controls for external AI services",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="SR-3",
        control_name="Supply Chain Controls",
        description="Supply chain risk management controls",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC9.2",
        control_name="Vendor and Third-Party Risk",
        description="Management of vendor and third-party risks",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="SA-9",
        control_name="External Information System Services",
        description="Federal supply chain security controls",
        aegis_capability="supply_chain_verification",
        evidence_type="test_result",
    ),

    # --- Data protection / PII ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="GOVERN-5",
        control_name="AI Data Governance",
        description="Data protection and PII handling in AI systems",
        aegis_capability="data_protection_pii",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-8.4",
        control_name="AI System Data Management",
        description="Data management and protection in AI systems",
        aegis_capability="data_protection_pii",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-10",
        control_name="Data and Data Governance",
        description="Data governance requirements for high-risk AI",
        aegis_capability="data_protection_pii",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="SC-7",
        control_name="Boundary Protection",
        description="Protection of data at system boundaries",
        aegis_capability="data_protection_pii",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="SC-8",
        control_name="Transmission Confidentiality",
        description="Protection of data during transmission",
        aegis_capability="data_protection_pii",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC6.1",
        control_name="Logical and Physical Access",
        description="Access controls for sensitive data protection",
        aegis_capability="data_protection_pii",
        evidence_type="continuous_monitoring",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="SC-7",
        control_name="Boundary Protection",
        description="Federal boundary protection and data controls",
        aegis_capability="data_protection_pii",
        evidence_type="continuous_monitoring",
    ),

    # --- Human oversight ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="GOVERN-1.3",
        control_name="Human Oversight of AI",
        description="Human-in-the-loop oversight for AI system decisions",
        aegis_capability="human_oversight",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-5.1",
        control_name="Leadership",
        description="Leadership and human oversight of AI management",
        aegis_capability="human_oversight",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-14",
        control_name="Human Oversight",
        description="Human oversight measures for high-risk AI systems",
        aegis_capability="human_oversight",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="PL-4",
        control_name="Rules of Behavior",
        description="Human behavioral rules and oversight policies",
        aegis_capability="human_oversight",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC1.3",
        control_name="Management Oversight",
        description="Board and management oversight responsibilities",
        aegis_capability="human_oversight",
        evidence_type="configuration",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="PL-4",
        control_name="Rules of Behavior",
        description="Federal rules of behavior and human oversight",
        aegis_capability="human_oversight",
        evidence_type="configuration",
    ),

    # --- Risk assessment ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MAP-1",
        control_name="AI Risk Context",
        description="Risk assessment and context mapping for AI systems",
        aegis_capability="risk_assessment",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-6",
        control_name="Planning",
        description="Planning and risk assessment for AI management system",
        aegis_capability="risk_assessment",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-9-RA",
        control_name="Risk Management — Assessment",
        description="Risk identification and assessment for high-risk AI",
        aegis_capability="risk_assessment",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="RA-3",
        control_name="Risk Assessment",
        description="Organization-wide risk assessment",
        aegis_capability="risk_assessment",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="RA-5",
        control_name="Vulnerability Monitoring and Scanning",
        description="Continuous vulnerability monitoring and scanning",
        aegis_capability="risk_assessment",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC3.2",
        control_name="Risk Assessment Process",
        description="Risk identification and assessment process",
        aegis_capability="risk_assessment",
        evidence_type="test_result",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="RA-3",
        control_name="Risk Assessment",
        description="Federal risk assessment requirements",
        aegis_capability="risk_assessment",
        evidence_type="test_result",
    ),

    # --- Transparency / explainability ---
    FrameworkControl(
        framework=ComplianceFramework.NIST_AI_RMF,
        control_id="MEASURE-2.5",
        control_name="AI Explainability",
        description="Transparency and explainability of AI decisions",
        aegis_capability="transparency_explainability",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.ISO_42001,
        control_id="Clause-9.3",
        control_name="Management Review",
        description="Management review and transparency requirements",
        aegis_capability="transparency_explainability",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art-13",
        control_name="Transparency and Information",
        description="Transparency requirements for AI system users",
        aegis_capability="transparency_explainability",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.CMMC_2,
        control_id="AU-3",
        control_name="Content of Audit Records",
        description="Detailed audit records for transparency",
        aegis_capability="transparency_explainability",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.SOC_2,
        control_id="CC2.2",
        control_name="Internal Communication",
        description="Transparent internal communication of security information",
        aegis_capability="transparency_explainability",
        evidence_type="audit_log",
    ),
    FrameworkControl(
        framework=ComplianceFramework.FEDRAMP,
        control_id="AU-3",
        control_name="Content of Audit Records",
        description="Federal audit record content and transparency",
        aegis_capability="transparency_explainability",
        evidence_type="audit_log",
    ),
]


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

# All 10 AEGIS capabilities
AEGIS_CAPABILITIES = [
    "real_time_threat_monitoring",
    "automated_audit_trails",
    "anomaly_drift_detection",
    "policy_enforcement",
    "incident_response",
    "supply_chain_verification",
    "data_protection_pii",
    "human_oversight",
    "risk_assessment",
    "transparency_explainability",
]


def get_all_controls() -> list[FrameworkControl]:
    """Return all control mappings."""
    return list(_CONTROL_MAPPINGS)


def get_controls_for_framework(
    framework: ComplianceFramework,
) -> list[FrameworkControl]:
    """Get all controls for a specific framework."""
    return [c for c in _CONTROL_MAPPINGS if c.framework == framework]


def get_controls_for_capability(
    capability: str,
) -> dict[ComplianceFramework, list[FrameworkControl]]:
    """Get controls across all frameworks for a specific AEGIS capability.

    Shows which frameworks a single capability satisfies.
    """
    result: dict[ComplianceFramework, list[FrameworkControl]] = {}
    for c in _CONTROL_MAPPINGS:
        if c.aegis_capability == capability:
            result.setdefault(c.framework, []).append(c)
    return result


def get_coverage_matrix() -> dict[str, Any]:
    """Get coverage percentage per framework.

    Returns:
        {
            "frameworks": {
                "NIST_AI_RMF": {"controls": 12, "capabilities_covered": 10, "coverage_pct": 100.0},
                ...
            },
            "total_controls": 68,
            "total_capabilities": 10,
            "capabilities": ["real_time_threat_monitoring", ...],
        }
    """
    frameworks: dict[str, dict[str, Any]] = {}

    for fw in ComplianceFramework:
        controls = get_controls_for_framework(fw)
        # Count unique capabilities covered by this framework
        capabilities_covered = len({c.aegis_capability for c in controls})
        coverage_pct = (capabilities_covered / len(AEGIS_CAPABILITIES)) * 100 if AEGIS_CAPABILITIES else 0

        frameworks[fw.value] = {
            "controls": len(controls),
            "capabilities_covered": capabilities_covered,
            "coverage_pct": round(coverage_pct, 1),
        }

    return {
        "frameworks": frameworks,
        "total_controls": len(_CONTROL_MAPPINGS),
        "total_capabilities": len(AEGIS_CAPABILITIES),
        "capabilities": list(AEGIS_CAPABILITIES),
    }
