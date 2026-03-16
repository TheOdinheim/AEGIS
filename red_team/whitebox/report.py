"""
White-Box Assessment Report — generates JSON and markdown reports.

Summarizes token importance analysis, per-attack-technique success rates,
decision boundary mapping, sensitivity analysis, and the adversarial corpus.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from red_team.whitebox import WhiteBoxAssessment

logger = logging.getLogger(__name__)


class WhiteBoxReport:
    """Generate comprehensive white-box assessment reports."""

    def generate_report(
        self,
        assessment: WhiteBoxAssessment,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Build, save, and return the full assessment report.

        Saves:
        - whitebox_assessment.json — machine-readable
        - WHITEBOX_ASSESSMENT.md — human-readable

        Returns:
            Report dictionary.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self._build_report(assessment)

        json_path = output_dir / "whitebox_assessment.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        md_path = output_dir / "WHITEBOX_ASSESSMENT.md"
        md_path.write_text(self._render_markdown(report, assessment))

        logger.info("White-box report saved to %s", output_dir)
        return report

    def _build_report(self, assessment: WhiteBoxAssessment) -> dict[str, Any]:
        """Build the report dictionary."""
        # Token importance summary
        token_summary = []
        for ti in assessment.token_importance:
            token_summary.append({
                "text": ti.text[:80],
                "confidence": ti.classification_confidence,
                "critical_tokens": ti.critical_tokens,
                "max_importance": ti.max_importance,
                "num_tokens": len(ti.tokens),
            })

        # Adversarial examples summary
        evasions = [e for e in assessment.adversarial_examples if e.evades]
        technique_stats: dict[str, dict[str, int]] = {}
        for ex in assessment.adversarial_examples:
            t = ex.technique
            if t not in technique_stats:
                technique_stats[t] = {"total": 0, "evasions": 0}
            technique_stats[t]["total"] += 1
            if ex.evades:
                technique_stats[t]["evasions"] += 1

        # Padding analysis summary
        padding_summary = []
        for pa in assessment.padding_analyses:
            padding_summary.append({
                "injection": pa.injection_text[:60],
                "min_padding_words": pa.min_padding_words,
                "padding_ratio": round(pa.padding_ratio, 2),
                "original_confidence": round(pa.original_confidence, 4),
                "boundary_confidence": round(pa.confidence_at_boundary, 4),
            })

        # Interpolation summary
        interp_summary = []
        for ir in assessment.interpolation_reports:
            interp_summary.append({
                "crossing_ratio": round(ir.crossing_ratio, 4),
                "crossing_confidence": round(ir.crossing_confidence, 4),
                "steps": ir.steps,
            })

        # Sensitivity
        sensitivity = None
        if assessment.sensitivity:
            s = assessment.sensitivity
            sensitivity = {
                "mean_margin": round(s.mean_margin, 4),
                "fragile_count": len(s.fragile_prompts),
                "robust_count": len(s.robust_prompts),
                "total_prompts": len(s.per_prompt_margins),
            }

        # Corpus
        corpus = None
        if assessment.corpus:
            c = assessment.corpus
            corpus = {
                "total_examples": len(c.examples),
                "evasion_count": c.evasion_count,
                "evasion_rate": round(c.evasion_rate, 4),
                "total_attempts": c.total_attempts,
                "mean_confidence_reduction": round(c.mean_confidence_reduction, 4),
                "technique_breakdown": c.technique_breakdown,
            }

        return {
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "white_box_assessment",
            },
            "token_importance": token_summary,
            "technique_stats": technique_stats,
            "total_adversarial_examples": len(assessment.adversarial_examples),
            "total_evasions": len(evasions),
            "evasion_rate": len(evasions) / max(len(assessment.adversarial_examples), 1),
            "padding_analysis": padding_summary,
            "interpolation": interp_summary,
            "sensitivity": sensitivity,
            "corpus": corpus,
        }

    def _render_markdown(
        self,
        report: dict[str, Any],
        assessment: WhiteBoxAssessment,
    ) -> str:
        """Render the assessment as markdown."""
        lines: list[str] = []
        lines.append("# White-Box Adversarial Assessment Report")
        lines.append(f"\n**Date:** {report['meta']['timestamp']}")
        lines.append(f"**Total adversarial examples:** {report['total_adversarial_examples']}")
        lines.append(f"**Evasions found:** {report['total_evasions']}")
        lines.append(f"**Evasion rate:** {report['evasion_rate']:.1%}")

        # Token importance
        lines.append("\n## Token Importance Analysis")
        lines.append("\nTokens DeBERTa relies on most for injection classification:")
        lines.append("\n| Prompt (truncated) | Confidence | Critical Tokens | Max Importance |")
        lines.append("|-------------------|------------|-----------------|----------------|")
        for ti in report["token_importance"]:
            crit = ", ".join(ti["critical_tokens"][:5]) if ti["critical_tokens"] else "none"
            lines.append(
                f"| {ti['text'][:50]} | {ti['confidence']:.4f} | {crit} | {ti['max_importance']:.4f} |"
            )

        # Per-technique results
        lines.append("\n## Attack Technique Results")
        lines.append("\n| Technique | Attempts | Evasions | Success Rate |")
        lines.append("|-----------|----------|----------|--------------|")
        for tech, stats in report["technique_stats"].items():
            rate = stats["evasions"] / max(stats["total"], 1)
            lines.append(
                f"| {tech} | {stats['total']} | {stats['evasions']} | {rate:.1%} |"
            )

        # Padding analysis
        if report["padding_analysis"]:
            lines.append("\n## Padding Dilution Analysis")
            lines.append("\n| Injection | Padding Words to Evade | Ratio | Original Conf |")
            lines.append("|-----------|----------------------|-------|---------------|")
            for pa in report["padding_analysis"]:
                evade_str = str(pa["min_padding_words"]) if pa["min_padding_words"] > 0 else "N/A"
                lines.append(
                    f"| {pa['injection'][:40]} | {evade_str} | {pa['padding_ratio']:.1f}x | {pa['original_confidence']:.4f} |"
                )

        # Interpolation
        if report["interpolation"]:
            lines.append("\n## Decision Boundary Interpolation")
            for ir in report["interpolation"]:
                lines.append(f"\n- Crossing ratio: {ir['crossing_ratio']:.2f} ({ir['crossing_ratio']*100:.0f}% words replaced)")
                lines.append(f"- Confidence at crossing: {ir['crossing_confidence']:.4f}")

        # Sensitivity
        if report["sensitivity"]:
            s = report["sensitivity"]
            lines.append("\n## Detection Sensitivity Analysis")
            lines.append(f"\n- Mean margin: {s['mean_margin']:.4f}")
            lines.append(f"- Fragile detections (margin < 0.1): {s['fragile_count']}/{s['total_prompts']}")
            lines.append(f"- Robust detections (margin > 0.3): {s['robust_count']}/{s['total_prompts']}")

        # Corpus
        if report["corpus"]:
            c = report["corpus"]
            lines.append("\n## Adversarial Corpus")
            lines.append(f"\n- Total unique examples: {c['total_examples']}")
            lines.append(f"- Successful evasions: {c['evasion_count']}")
            lines.append(f"- Evasion rate: {c['evasion_rate']:.1%}")
            lines.append(f"- Mean confidence reduction: {c['mean_confidence_reduction']:.4f}")

        # Top evasions
        evasions = [e for e in assessment.adversarial_examples if e.evades]
        if evasions:
            lines.append("\n## Top Evasions")
            top = sorted(evasions, key=lambda e: e.adversarial_confidence)[:10]
            lines.append("\n| Original (truncated) | Adversarial (truncated) | Conf Drop | Technique |")
            lines.append("|---------------------|------------------------|-----------|-----------|")
            for e in top:
                lines.append(
                    f"| {e.original[:30]} | {e.adversarial[:30]} | "
                    f"{e.original_confidence:.2f} -> {e.adversarial_confidence:.2f} | {e.technique} |"
                )

        # Recommendations
        lines.append("\n## Hardening Recommendations")
        lines.append("\n1. Add evasion examples to the threat vault as seed embeddings (antibodies)")
        lines.append("2. Generate L2 regex patterns for consistent synonym substitution patterns")
        lines.append("3. Add evasion corpus to benchmark for regression testing")
        lines.append("4. Consider ensemble with a second classifier for fragile detections")
        lines.append("5. Implement input normalization for synonym-based evasions")

        lines.append("\n---")
        lines.append("*Generated by AEGIS White-Box Adversarial Framework*")
        return "\n".join(lines)
