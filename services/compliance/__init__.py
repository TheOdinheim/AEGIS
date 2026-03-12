"""
Compliance Engine — Cross-Framework Regulatory Compliance Dashboard.

Biological Analog: Immune system documentation — the body maintains records
of every immune response, antibody generated, and pathogen encountered.
Medical records prove vaccination status across multiple health frameworks.

AEGIS maps its security controls to six regulatory frameworks simultaneously:
NIST AI RMF, ISO 42001, EU AI Act, CMMC 2.0, SOC 2, and FedRAMP.
A single AEGIS capability satisfies controls across 5-6 frameworks,
reducing compliance costs 40-60%.

Components:
- FrameworkMappings: Static mapping of AEGIS capabilities to framework controls
- EvidenceCollector: Automated evidence collection from AEGIS telemetry
- ReportGenerator: Cross-framework compliance report generation
"""

from __future__ import annotations

import logging
from typing import Any

from aegis.services.compliance.evidence_collector import Evidence, EvidenceCollector
from aegis.services.compliance.framework_mappings import (
    AEGIS_CAPABILITIES,
    ComplianceFramework,
    FrameworkControl,
    get_all_controls,
    get_controls_for_capability,
    get_controls_for_framework,
    get_coverage_matrix,
)
from aegis.services.compliance.report_generator import (
    ComplianceGap,
    ComplianceReport,
    ComplianceReportGenerator,
)

logger = logging.getLogger(__name__)


class ComplianceEngine:
    """Orchestrator for the compliance reporting system.

    Combines framework mappings, evidence collection, and report generation
    into a single interface. Initialized during AEGIS startup with references
    to the layers needed for evidence collection.
    """

    def __init__(
        self,
        *,
        audit_logger: Any | None = None,
        config: Any | None = None,
        vault: Any | None = None,
        policy: Any | None = None,
        healing: Any | None = None,
        supply_chain: Any | None = None,
    ) -> None:
        self._collector = EvidenceCollector(
            audit_logger=audit_logger,
            config=config,
            vault=vault,
            policy=policy,
            healing=healing,
            supply_chain=supply_chain,
        )
        self._generator = ComplianceReportGenerator(self._collector)

    async def generate_report(
        self,
        frameworks: list[str] | None = None,
        time_range_hours: int = 24,
    ) -> ComplianceReport:
        """Generate a compliance report."""
        return await self._generator.generate_report(frameworks, time_range_hours)

    def get_report(self, report_id: str) -> ComplianceReport | None:
        """Retrieve a previously generated report."""
        return self._generator.get_report(report_id)

    async def collect_evidence(
        self, capability: str, time_range_hours: int = 24,
    ) -> Evidence:
        """Collect evidence for a specific capability."""
        return await self._collector.collect_evidence(capability, time_range_hours)

    def get_frameworks(self) -> list[dict[str, Any]]:
        """List available frameworks with control counts."""
        result = []
        for fw in ComplianceFramework:
            controls = get_controls_for_framework(fw)
            result.append({
                "framework": fw.value,
                "control_count": len(controls),
                "capabilities_covered": len({c.aegis_capability for c in controls}),
            })
        return result

    def get_coverage_matrix(self) -> dict[str, Any]:
        """Get the full cross-framework coverage matrix."""
        return get_coverage_matrix()

    @property
    def stats(self) -> dict[str, Any]:
        """Compliance engine statistics."""
        return {
            "frameworks_mapped": len(ComplianceFramework),
            "total_controls": len(get_all_controls()),
            "total_capabilities": len(AEGIS_CAPABILITIES),
            "reports_generated": self._generator.report_count,
        }
