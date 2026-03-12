"""
Red Team — Adversarial attack generation and evasion testing for AEGIS.

This is an operational tool, NOT a CI test suite. It generates novel attacks
designed to bypass AEGIS detection layers, runs them against the live pipeline,
measures what gets through, and feeds successful evasions back into the immune
system so it learns.

Components:
    - AttackGenerator: Generates attacks targeting specific detection layers
    - EvasionEngine: Runs attacks against AEGIS and classifies results
    - CampaignRunner: Orchestrates multi-stage attack campaigns
    - LearningValidator: Measures whether the immune system learned from attacks
    - RedTeamReport: Generates assessment reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TargetLayer(str, Enum):
    """Which AEGIS layer the attack targets."""

    L2_INNATE = "L2_innate"
    L3_ADAPTIVE = "L3_adaptive"
    L5_OUTPUT = "L5_output"
    MULTI_LAYER = "multi_layer"


class EvasionTechnique(str, Enum):
    """Specific evasion technique used."""

    # L2 Innate evasion
    UNICODE_HOMOGLYPH = "unicode_homoglyph"
    ZERO_WIDTH_CHARS = "zero_width_chars"
    MIXED_SCRIPT = "mixed_script"
    TOKEN_BOUNDARY = "token_boundary"
    ENCODING_STACKING = "encoding_stacking"
    PROMPT_SPLITTING = "prompt_splitting"
    SEMANTIC_RESTRUCTURING = "semantic_restructuring"
    HOMOPHONE_SUBSTITUTION = "homophone_substitution"
    MARKDOWN_INJECTION = "markdown_injection"
    LANGUAGE_SWITCHING = "language_switching"

    # L3 Adaptive evasion
    ADVERSARIAL_TOKEN_INSERTION = "adversarial_token_insertion"
    CONFIDENCE_BOUNDARY_PROBING = "confidence_boundary_probing"
    CONTEXT_DILUTION = "context_dilution"
    PERSONA_FRAMING = "persona_framing"
    HYPOTHETICAL_WRAPPING = "hypothetical_wrapping"
    ACADEMIC_FRAMING = "academic_framing"

    # L5 Output evasion
    PARTIAL_PII_WORDS = "partial_pii_words"
    ENCODED_PII = "encoded_pii"
    STEGANOGRAPHIC_PII = "steganographic_pii"
    STRUCTURED_PII = "structured_pii"
    SPLIT_PII = "split_pii"

    # Multi-layer gap attacks
    L2_PASS_L3_CATCH = "l2_pass_l3_catch"
    L3_PASS_L5_CATCH = "l3_pass_l5_catch"
    MULTI_TURN_ESCALATION = "multi_turn_escalation"


@dataclass
class Attack:
    """A single adversarial attack payload."""

    attack_id: str
    target_layer: TargetLayer
    evasion_technique: EvasionTechnique
    payload: str | list[str]  # str for single-message, list for multi-turn
    expected_detection_layer: str
    expected_confidence_range: tuple[float, float]
    difficulty_rating: int  # 1-5, higher = harder to detect
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "target_layer": self.target_layer.value,
            "evasion_technique": self.evasion_technique.value,
            "payload": self.payload,
            "expected_detection_layer": self.expected_detection_layer,
            "expected_confidence_range": list(self.expected_confidence_range),
            "difficulty_rating": self.difficulty_rating,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attack:
        return cls(
            attack_id=data["attack_id"],
            target_layer=TargetLayer(data["target_layer"]),
            evasion_technique=EvasionTechnique(data["evasion_technique"]),
            payload=data["payload"],
            expected_detection_layer=data["expected_detection_layer"],
            expected_confidence_range=tuple(data["expected_confidence_range"]),
            difficulty_rating=data["difficulty_rating"],
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvasionResult:
    """Result of running a single attack against AEGIS."""

    attack_id: str
    target_layer: str
    evasion_technique: str
    was_blocked: bool
    caught_by_layer: str | None
    confidence: float
    response_preview: str
    latency_ms: float
    status_code: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "target_layer": self.target_layer,
            "evasion_technique": self.evasion_technique,
            "was_blocked": self.was_blocked,
            "caught_by_layer": self.caught_by_layer,
            "confidence": self.confidence,
            "response_preview": self.response_preview,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass
class EvasionReport:
    """Aggregated results from an evasion test run."""

    total_attacks: int
    blocked: int
    evaded: int
    evasion_rate: float
    per_layer_results: dict[str, dict[str, int]]
    per_technique_results: dict[str, dict[str, int]]
    hardest_to_detect: list[EvasionResult]
    all_results: list[EvasionResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_attacks": self.total_attacks,
            "blocked": self.blocked,
            "evaded": self.evaded,
            "evasion_rate": self.evasion_rate,
            "per_layer_results": self.per_layer_results,
            "per_technique_results": self.per_technique_results,
            "hardest_to_detect": [r.to_dict() for r in self.hardest_to_detect],
            "all_results": [r.to_dict() for r in self.all_results],
            "timestamp": self.timestamp,
        }


@dataclass
class LearningReport:
    """Results from immune system learning validation."""

    antibodies_generated: int
    variants_tested: int
    variants_caught: int
    generalization_rate: float
    detection_gaps: list[dict[str, Any]]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "antibodies_generated": self.antibodies_generated,
            "variants_tested": self.variants_tested,
            "variants_caught": self.variants_caught,
            "generalization_rate": self.generalization_rate,
            "detection_gaps": self.detection_gaps,
            "timestamp": self.timestamp,
        }
