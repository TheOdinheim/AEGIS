"""
Blind spot detector — analyzes all prior red team results (Phases 1-5)
to find systematic weaknesses in AEGIS detection.

Performs three types of analysis:
1. Systematic weakness identification: patterns of missed detections
2. Confidence gap analysis: attacks near decision boundaries
3. Temporal pattern analysis: detection degradation over time
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from red_team.adaptive import (
    BlindSpot,
    BlindSpotReport,
)


class BlindSpotDetector:
    """Analyze all red team results to find systematic defense weaknesses.

    Input: results from Phases 1-5 (campaign results, adversarial ML,
    infrastructure security, evasion reports).
    Output: BlindSpotReport with ranked blind spots, coverage gaps,
    and confidence boundary analysis.
    """

    def __init__(self, data_dir: str | None = None):
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path(__file__).resolve().parent.parent / "data"

    def analyze(
        self,
        campaign_results: list[dict[str, Any]] | None = None,
        evasion_results: list[dict[str, Any]] | None = None,
        infrastructure_results: list[dict[str, Any]] | None = None,
        adversarial_ml_results: dict[str, Any] | None = None,
    ) -> BlindSpotReport:
        """Run full blind spot analysis across all red team phases."""
        # Load data from files if not provided
        if campaign_results is None:
            campaign_results = self._load_campaign_results()
        if evasion_results is None:
            evasion_results = self._load_evasion_results()

        blind_spots: list[BlindSpot] = []
        total_analyzed = 0

        # 1. Systematic weakness identification
        weakness_spots, count = self._find_systematic_weaknesses(
            campaign_results, evasion_results
        )
        blind_spots.extend(weakness_spots)
        total_analyzed += count

        # 2. Infrastructure weakness analysis
        infra_spots = self._find_infrastructure_weaknesses(
            infrastructure_results or []
        )
        blind_spots.extend(infra_spots)

        # 3. Coverage gap analysis
        coverage_gaps = self._identify_coverage_gaps(
            campaign_results, evasion_results, adversarial_ml_results
        )

        # 4. Confidence gap analysis
        confidence_gaps = self._analyze_confidence_gaps(evasion_results)

        # 5. Temporal pattern analysis
        temporal_patterns = self._analyze_temporal_patterns(campaign_results)

        # 6. Cross-phase blind spots
        cross_phase_spots = self._find_cross_phase_blind_spots(
            campaign_results, evasion_results, adversarial_ml_results
        )
        blind_spots.extend(cross_phase_spots)

        return BlindSpotReport(
            blind_spots=blind_spots,
            total_results_analyzed=total_analyzed,
            coverage_gaps=coverage_gaps,
            confidence_gaps=confidence_gaps,
            temporal_patterns=temporal_patterns,
        )

    def _load_campaign_results(self) -> list[dict[str, Any]]:
        """Load campaign results from saved files."""
        results = []
        for name in ("campaign_results_initial.json", "campaign_results_final.json"):
            path = self._data_dir / name
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    if isinstance(data, list):
                        results.extend(data)
                    elif isinstance(data, dict):
                        results.append(data)
                except (json.JSONDecodeError, OSError):
                    pass
        return results

    def _load_evasion_results(self) -> list[dict[str, Any]]:
        """Load evasion results from detection_gaps.json."""
        path = self._data_dir / "detection_gaps.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "gaps" in data:
                    return data["gaps"]
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _find_systematic_weaknesses(
        self,
        campaign_results: list[dict[str, Any]],
        evasion_results: list[dict[str, Any]],
    ) -> tuple[list[BlindSpot], int]:
        """Identify patterns of repeated detection failures."""
        blind_spots: list[BlindSpot] = []
        total_count = len(campaign_results) + len(evasion_results)
        spot_idx = 0

        # Analyze campaigns for high evasion rates by technique
        technique_evasions: dict[str, list[str]] = defaultdict(list)
        layer_misses: dict[str, int] = Counter()

        for result in campaign_results:
            campaign_name = result.get("campaign", result.get("name", "unknown"))
            stages = result.get("stages", result.get("results", []))
            if isinstance(stages, list):
                for stage in stages:
                    if isinstance(stage, dict):
                        evaded = not stage.get("detected", stage.get("was_blocked", True))
                        if evaded:
                            technique = stage.get("technique", stage.get(
                                "evasion_technique", "unknown"
                            ))
                            layer = stage.get("target_layer", stage.get(
                                "expected_layer", "unknown"
                            ))
                            technique_evasions[technique].append(campaign_name)
                            layer_misses[layer] += 1

        # Techniques that consistently evade detection
        for technique, campaigns in technique_evasions.items():
            if len(campaigns) >= 2:
                spot_idx += 1
                blind_spots.append(BlindSpot(
                    spot_id=f"BS-SYS-{spot_idx:03d}",
                    category="systematic_weakness",
                    description=(
                        f"Technique '{technique}' evaded detection in "
                        f"{len(campaigns)} campaigns"
                    ),
                    affected_layers=list(set(
                        r.get("target_layer", "unknown")
                        for r in campaign_results
                        if r.get("campaign", r.get("name")) in campaigns
                    ))[:3],
                    severity="high" if len(campaigns) >= 3 else "medium",
                    evidence=[f"Evaded in: {', '.join(campaigns[:5])}"],
                    exploit_difficulty=2,
                    remediation=f"Add specific detection for {technique} technique",
                ))

        # Layers with high miss rates
        for layer, misses in layer_misses.most_common(3):
            if misses >= 3:
                spot_idx += 1
                blind_spots.append(BlindSpot(
                    spot_id=f"BS-LAY-{spot_idx:03d}",
                    category="layer_weakness",
                    description=f"Layer {layer} missed {misses} attacks across campaigns",
                    affected_layers=[layer],
                    severity="high" if misses >= 5 else "medium",
                    evidence=[f"{misses} missed detections"],
                    exploit_difficulty=3,
                    remediation=f"Strengthen {layer} detection capabilities",
                ))

        # Analyze evasion gaps
        for gap in evasion_results:
            if isinstance(gap, dict) and gap.get("status") != "fixed":
                spot_idx += 1
                blind_spots.append(BlindSpot(
                    spot_id=f"BS-GAP-{spot_idx:03d}",
                    category="detection_gap",
                    description=gap.get("description", gap.get("gap", "Unknown gap")),
                    affected_layers=[gap.get("layer", "unknown")],
                    severity=gap.get("severity", "medium"),
                    evidence=[gap.get("evidence", "From detection_gaps.json")],
                    exploit_difficulty=gap.get("difficulty", 3),
                    remediation=gap.get("remediation", gap.get("fix", "Manual review")),
                ))

        return blind_spots, total_count

    def _find_infrastructure_weaknesses(
        self, infra_results: list[dict[str, Any]]
    ) -> list[BlindSpot]:
        """Convert infrastructure attack findings to blind spots."""
        blind_spots: list[BlindSpot] = []
        for idx, result in enumerate(infra_results):
            if result.get("vulnerable", False):
                blind_spots.append(BlindSpot(
                    spot_id=f"BS-INF-{idx + 1:03d}",
                    category="infrastructure",
                    description=result.get("description", "Infrastructure vulnerability"),
                    affected_layers=["infrastructure"],
                    severity=result.get("severity", "medium"),
                    evidence=result.get("evidence", [])[:3],
                    exploit_difficulty=2,
                    remediation=result.get("recommendation", "See infrastructure report"),
                ))
        return blind_spots

    def _identify_coverage_gaps(
        self,
        campaign_results: list[dict[str, Any]],
        evasion_results: list[dict[str, Any]],
        adversarial_ml_results: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        """Identify detection coverage gaps per layer."""
        gaps: dict[str, list[str]] = defaultdict(list)

        # Known architectural gaps
        gaps["L2_innate"].append(
            "No language detection — non-English attacks bypass regex patterns"
        )
        gaps["L2_innate"].append(
            "Homoglyph map is finite (200+) — new Unicode blocks may contain unmapped confusables"
        )
        gaps["L3_adaptive"].append(
            "DeBERTa fine-tuned on English — reduced accuracy on multilingual prompts"
        )
        gaps["L3_adaptive"].append(
            "Semantic similarity requires minimum vault population for coverage"
        )
        gaps["L5_output"].append(
            "Steganographic PII requires NER (Presidio) — regex alone insufficient"
        )
        gaps["L5_output"].append(
            "Acrostic/first-letter encoding not detectable by pattern matching"
        )

        # Gaps from adversarial ML results
        if adversarial_ml_results:
            corpus_baseline = adversarial_ml_results.get("corpus_baseline", {})
            evasion_rate = corpus_baseline.get("evasion_rate", 0)
            if evasion_rate > 0.1:
                gaps["multi_layer"].append(
                    f"Corpus baseline evasion rate {evasion_rate:.1%} — "
                    f"systematic gaps in attack coverage"
                )
            timing = adversarial_ml_results.get("timing_analysis", {})
            if timing.get("timing_leak_detected"):
                gaps["L1_barrier"].append(
                    "Timing side-channel detected — response time leaks block/allow decision"
                )

        # Gaps from campaign evasion patterns
        evaded_techniques: set[str] = set()
        for result in campaign_results:
            stages = result.get("stages", result.get("results", []))
            if isinstance(stages, list):
                for stage in stages:
                    if isinstance(stage, dict):
                        evaded = not stage.get("detected", stage.get("was_blocked", True))
                        if evaded:
                            evaded_techniques.add(
                                stage.get("technique", stage.get("evasion_technique", "unknown"))
                            )

        if "synonym" in evaded_techniques or "semantic_restructuring" in evaded_techniques:
            gaps["L2_innate"].append(
                "Synonym substitution evades regex — requires L3 semantic analysis"
            )
        if "encoding_mix" in evaded_techniques:
            gaps["L2_innate"].append(
                "Stacked encoding (ROT13+base64) partially bypasses single-pass decode"
            )

        return dict(gaps)

    def _analyze_confidence_gaps(
        self, evasion_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find attacks near decision boundaries (confidence 0.40-0.60)."""
        confidence_gaps: list[dict[str, Any]] = []

        for result in evasion_results:
            if isinstance(result, dict):
                confidence = result.get("confidence", 0)
                if 0.40 <= confidence <= 0.60:
                    confidence_gaps.append({
                        "attack_id": result.get("attack_id", "unknown"),
                        "confidence": confidence,
                        "technique": result.get("technique", result.get(
                            "evasion_technique", "unknown"
                        )),
                        "layer": result.get("layer", result.get("target_layer", "unknown")),
                        "risk": "Decision boundary — small perturbations could flip outcome",
                    })

        # Add known boundary conditions from architecture
        confidence_gaps.append({
            "attack_id": "architectural_boundary_1",
            "confidence": 0.50,
            "technique": "context_dilution",
            "layer": "L3_adaptive",
            "risk": (
                "DeBERTa confidence near 0.50 for long context-diluted prompts — "
                "512-token window may miss injection buried at position 800+"
            ),
        })
        confidence_gaps.append({
            "attack_id": "architectural_boundary_2",
            "confidence": 0.45,
            "technique": "academic_framing",
            "layer": "L3_adaptive",
            "risk": (
                "Academic/research framing reduces DeBERTa injection confidence — "
                "legitimate research queries about attacks score near threshold"
            ),
        })

        return confidence_gaps

    def _analyze_temporal_patterns(
        self, campaign_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze detection effectiveness over campaign progression."""
        patterns: list[dict[str, Any]] = []

        for result in campaign_results:
            campaign_name = result.get("campaign", result.get("name", "unknown"))
            stages = result.get("stages", result.get("results", []))
            if not isinstance(stages, list) or len(stages) < 3:
                continue

            # Track detection rate in windows
            window_size = max(3, len(stages) // 3)
            for i in range(0, len(stages) - window_size + 1, window_size):
                window = stages[i:i + window_size]
                detected = sum(
                    1 for s in window
                    if isinstance(s, dict) and s.get("detected", s.get("was_blocked", False))
                )
                rate = detected / len(window) if window else 0

                if rate < 0.5 and i > 0:
                    patterns.append({
                        "campaign": campaign_name,
                        "window_start": i,
                        "window_end": i + window_size,
                        "detection_rate": rate,
                        "pattern": "detection_degradation",
                        "risk": (
                            f"Detection rate dropped to {rate:.0%} in window "
                            f"[{i}:{i + window_size}] of campaign {campaign_name}"
                        ),
                    })

        # Known temporal patterns
        patterns.append({
            "campaign": "SLOW_BURN",
            "window_start": 0,
            "window_end": 20,
            "detection_rate": 0.50,
            "pattern": "multi_turn_escalation",
            "risk": (
                "Multi-turn escalation campaigns show progressive detection "
                "degradation as attacker learns boundary conditions"
            ),
        })

        return patterns

    def _find_cross_phase_blind_spots(
        self,
        campaign_results: list[dict[str, Any]],
        evasion_results: list[dict[str, Any]],
        adversarial_ml_results: dict[str, Any] | None,
    ) -> list[BlindSpot]:
        """Find blind spots that span multiple red team phases."""
        spots: list[BlindSpot] = []
        spot_idx = 0

        # Cross-phase: normalization gaps confirmed by adversarial ML
        spot_idx += 1
        spots.append(BlindSpot(
            spot_id=f"BS-XPH-{spot_idx:03d}",
            category="cross_phase",
            description=(
                "Normalization gaps confirmed across Phase 2 (HYDRA mutations), "
                "Phase 3 (CharSwap confusables), and Phase 4 (extended campaigns)"
            ),
            affected_layers=["L2_innate"],
            severity="high",
            evidence=[
                "HYDRA campaign: 83% evasion via mutations",
                "CharSwap: 229 confusables beyond normalize_text()",
                "Phase 4: BIDI/combining marks required hardening",
            ],
            exploit_difficulty=2,
            remediation=(
                "Continuously expand homoglyph map, add confusable detection "
                "from Unicode CLDR confusables.txt"
            ),
        ))

        # Cross-phase: PII format evasion
        spot_idx += 1
        spots.append(BlindSpot(
            spot_id=f"BS-XPH-{spot_idx:03d}",
            category="cross_phase",
            description=(
                "PII format evasion confirmed across Phase 2 (SILENT SIPHON), "
                "Phase 3 (boundary mapper), and Phase 5 (MIRROR MIRROR)"
            ),
            affected_layers=["L5_output"],
            severity="medium",
            evidence=[
                "SILENT SIPHON: 100% evasion on word-spelled/steganographic PII",
                "Boundary mapper: PII format variants find detection boundaries",
                "Phase 4 fixes added code-context and URL-embedded patterns",
            ],
            exploit_difficulty=3,
            remediation=(
                "Deploy Presidio NER for context-aware PII detection, "
                "add format-agnostic PII models"
            ),
        ))

        # Cross-phase: semantic injection without keywords
        spot_idx += 1
        spots.append(BlindSpot(
            spot_id=f"BS-XPH-{spot_idx:03d}",
            category="cross_phase",
            description=(
                "Keyword-free semantic injection consistently evades L2 regex — "
                "relies entirely on L3 DeBERTa which may not be loaded"
            ),
            affected_layers=["L2_innate", "L3_adaptive"],
            severity="high",
            evidence=[
                "Phase 1: paraphrased injection marked as L3-only detection",
                "Phase 3: TextFooler synonym substitution removes all keywords",
                "Phase 4: residual risk acknowledged",
            ],
            exploit_difficulty=3,
            remediation=(
                "Ensure DeBERTa always loaded in production, add lightweight "
                "intent classifier as L2 backup"
            ),
        ))

        return spots
