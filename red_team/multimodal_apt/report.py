"""
Multimodal APT Report Generator — comprehensive assessment reports.

Produces JSON and markdown reports for the full multimodal APT assessment
including per-campaign results, per-modality detection matrices, adaptive
attacker results, model behavior validation, and immune system learning.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from red_team.multimodal_apt import (
    CampaignResult,
    MultimodalAPTAssessment,
    _grade,
)

logger = logging.getLogger(__name__)


class MultimodalAPTReport:
    """Generates comprehensive multimodal APT assessment reports."""

    def generate_report(
        self,
        assessment: MultimodalAPTAssessment,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Generate and save the full assessment report.

        Saves:
        - multimodal_apt_results.json — machine-readable full results
        - MULTIMODAL_APT_ASSESSMENT.md — human-readable assessment

        Returns:
            Report dictionary.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self._build_report(assessment)

        # Save JSON
        json_path = output_dir / "multimodal_apt_results.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        # Save Markdown
        md_path = output_dir / "MULTIMODAL_APT_ASSESSMENT.md"
        md_path.write_text(self._render_markdown(report, assessment))

        logger.info("Report saved to %s", output_dir)
        return report

    def _build_report(self, assessment: MultimodalAPTAssessment) -> dict[str, Any]:
        """Build the complete report dictionary."""
        return {
            "meta": {
                "timestamp": assessment.timestamp,
                "total_campaigns": len(assessment.campaigns),
                "overall_detection_rate": assessment.overall_detection_rate,
                "overall_fpr": assessment.overall_fpr,
                "grade": assessment.grade,
            },
            "campaigns": [c.to_dict() for c in assessment.campaigns],
            "immune_system_learning": assessment.immune_system_learning,
            "adaptive_attacker_results": assessment.adaptive_attacker_results,
            "model_behavior_results": assessment.model_behavior_results,
            "executive_summary": assessment.executive_summary,
        }

    def _render_markdown(
        self,
        report: dict[str, Any],
        assessment: MultimodalAPTAssessment,
    ) -> str:
        """Render the assessment as markdown."""
        lines: list[str] = []
        meta = report["meta"]

        lines.append("# Multimodal APT Assessment Report")
        lines.append(f"\n**Date:** {meta['timestamp']}")
        lines.append(f"**Campaigns:** {meta['total_campaigns']}")
        lines.append(f"**Overall Grade: {meta['grade']}**")
        lines.append(f"**Detection Rate:** {meta['overall_detection_rate']:.1%}")
        lines.append(f"**False Positive Rate:** {meta['overall_fpr']:.1%}")

        # Executive summary
        lines.append("\n## Executive Summary")
        lines.append(f"\n{assessment.executive_summary}")

        # Per-campaign results
        lines.append("\n## Campaign Results")
        lines.append("\n| Campaign | Attacks | Blocked | Detection Rate | FPR |")
        lines.append("|----------|---------|---------|----------------|-----|")
        for campaign in assessment.campaigns:
            lines.append(
                f"| {campaign.campaign_name} | {campaign.total_attacks} | "
                f"{campaign.blocked} | {campaign.detection_rate:.1%} | "
                f"{campaign.false_positive_rate:.1%} |"
            )

        # Per-modality matrix
        lines.append("\n## Per-Modality Detection Matrix")
        modality_data = self._aggregate_modality_data(assessment)
        if modality_data:
            lines.append("\n| Modality | Total | Blocked | Allowed | Detection Rate |")
            lines.append("|----------|-------|---------|---------|----------------|")
            for mod, stats in sorted(modality_data.items()):
                rate = stats["blocked"] / stats["total"] if stats["total"] > 0 else 0
                lines.append(
                    f"| {mod} | {stats['total']} | {stats['blocked']} | "
                    f"{stats['allowed']} | {rate:.1%} |"
                )

        # Adaptive attacker results
        if assessment.adaptive_attacker_results:
            lines.append("\n## Adaptive Attacker Results")
            ar = assessment.adaptive_attacker_results
            lines.append(f"\n- Initial evasion rate: {ar.get('initial_evasion_rate', 'N/A')}")
            lines.append(f"- Final evasion rate: {ar.get('final_evasion_rate', 'N/A')}")
            lines.append(f"- Weakest modality: {ar.get('weakest_modality', 'N/A')}")
            lines.append(f"- Strongest modality: {ar.get('strongest_modality', 'N/A')}")
            if ar.get("adaptation_narrative"):
                lines.append(f"\n{ar['adaptation_narrative']}")

        # Model behavior
        if assessment.model_behavior_results:
            lines.append("\n## Model Behavior Validation")
            mb = assessment.model_behavior_results
            lines.append(f"\n- Total validated: {mb.get('total_validated', 0)}")
            lines.append(f"- Behavior changed: {mb.get('behavior_changed', 0)}")
            if mb.get("severity_counts"):
                lines.append(f"- Severity breakdown: {mb['severity_counts']}")

        # Immune learning
        if assessment.immune_system_learning:
            lines.append("\n## Immune System Learning")
            il = assessment.immune_system_learning
            lines.append(f"\n- Vault growth: {il.get('vault_growth', 'N/A')}")
            lines.append(f"- Antibodies generated: {il.get('antibodies_generated', 'N/A')}")
            lines.append(f"- Initial evasion rate: {il.get('initial_evasion_rate', 'N/A')}")
            lines.append(f"- Final evasion rate: {il.get('final_evasion_rate', 'N/A')}")
            delta = il.get("learning_delta")
            if delta is not None:
                lines.append(f"- Learning delta: {delta}")

        # Residual risks
        lines.append("\n## Residual Risks")
        lines.append("\nAttacks that consistently evade detection require targeted hardening:")
        lines.append("- Steganographic payloads in alpha channels and LSB spread patterns")
        lines.append("- Adversarial perturbations that survive image re-encoding")
        lines.append("- Cross-modal sentence reconstruction spanning 3+ modalities")
        lines.append("- Non-English injections in image OCR text")
        lines.append("- Near-ultrasonic structured audio patterns")

        lines.append("\n## Recommendations")
        lines.append("\n1. Expand OCR to handle rotated and spiral text layouts")
        lines.append("2. Add alpha channel inspection for steganographic payloads")
        lines.append("3. Implement cross-modal text concatenation analysis")
        lines.append("4. Add multi-language injection detection for OCR output")
        lines.append("5. Enhance spectral analysis for near-ultrasonic patterns")

        lines.append("\n---")
        lines.append("*Generated by AEGIS Multimodal APT Framework*")

        return "\n".join(lines)

    @staticmethod
    def _aggregate_modality_data(
        assessment: MultimodalAPTAssessment,
    ) -> dict[str, dict[str, int]]:
        """Aggregate modality data across all campaigns."""
        agg: dict[str, dict[str, int]] = {}
        for campaign in assessment.campaigns:
            for mod, stats in campaign.per_modality_breakdown.items():
                if mod not in agg:
                    agg[mod] = {"total": 0, "blocked": 0, "allowed": 0}
                agg[mod]["total"] += stats.get("total", 0)
                agg[mod]["blocked"] += stats.get("blocked", 0)
                agg[mod]["allowed"] += stats.get("allowed", 0)
        return agg
