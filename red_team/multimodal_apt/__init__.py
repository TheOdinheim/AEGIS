"""
Multimodal APT Stress Test — Full Spectrum adversarial validation.

Tests the complete AEGIS immune system with real HTTP requests across
all modalities: text, image, document, audio, cross-modal, and tool use.

Components:
    - PayloadFactory: Generates all attack payloads programmatically
    - LiveCampaignRunner: Sends real HTTP requests to live AEGIS
    - AdaptiveMultimodalAttacker: Learns from AEGIS responses, adapts attacks
    - ModelBehaviorValidator: Validates if attacks influence model output
    - MultimodalAPTReport: Comprehensive assessment reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Modality(str, Enum):
    """Attack modality."""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    CROSS_MODAL = "cross_modal"
    TOOL = "tool"


class DetectionLayer(str, Enum):
    """Expected detection layer."""
    L1_BARRIER = "L1"
    L2_INNATE = "L2"
    L3_ADAPTIVE = "L3"
    L4_MEMORY = "L4"
    L5_OUTPUT = "L5"
    L6_POLICY = "L6"
    L7_HEALING = "L7"
    MULTIMODAL = "multimodal"
    CROSS_MODAL = "cross_modal"
    TOOL_SCANNER = "tool_scanner"


@dataclass
class AttackPayload:
    """A single multimodal attack payload."""

    attack_id: str
    modality: Modality
    payload_bytes: bytes | None = None
    text_content: str = ""
    expected_detection_layer: DetectionLayer = DetectionLayer.L2_INNATE
    technique: str = ""
    difficulty: int = 3  # 1-5
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "modality": self.modality.value,
            "has_payload_bytes": self.payload_bytes is not None,
            "payload_bytes_len": len(self.payload_bytes) if self.payload_bytes else 0,
            "text_content": self.text_content[:200] if self.text_content else "",
            "expected_detection_layer": self.expected_detection_layer.value,
            "technique": self.technique,
            "difficulty": self.difficulty,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class AttackResult:
    """Result of running a single attack against live AEGIS."""

    attack_id: str
    status_code: int
    response_body: str = ""
    latency_ms: float = 0.0
    was_blocked: bool = False
    detection_layer: str | None = None
    confidence: float = 0.0
    model_output: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "status_code": self.status_code,
            "response_body": self.response_body[:200],
            "latency_ms": round(self.latency_ms, 2),
            "was_blocked": self.was_blocked,
            "detection_layer": self.detection_layer,
            "confidence": self.confidence,
            "model_output": (self.model_output[:200] if self.model_output else None),
            "error": self.error,
        }


@dataclass
class BehaviorValidation:
    """Result of validating whether an attack influenced model behavior."""

    baseline_response: str
    attack_response: str
    similarity_score: float  # 0.0 = completely different, 1.0 = identical
    behavior_changed: bool
    severity: str  # "none", "minor", "significant", "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_response": self.baseline_response[:200],
            "attack_response": self.attack_response[:200],
            "similarity_score": round(self.similarity_score, 4),
            "behavior_changed": self.behavior_changed,
            "severity": self.severity,
        }


@dataclass
class RoundResult:
    """Result of one adaptive attacker round."""

    round_number: int
    attacks_sent: int
    attacks_blocked: int
    evasion_rate: float
    mutations_applied: list[str] = field(default_factory=list)
    best_evasion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "attacks_sent": self.attacks_sent,
            "attacks_blocked": self.attacks_blocked,
            "evasion_rate": round(self.evasion_rate, 4),
            "mutations_applied": self.mutations_applied,
            "best_evasion": self.best_evasion,
        }


@dataclass
class AdaptiveAttackReport:
    """Report from adaptive multi-round attack campaign."""

    rounds: list[RoundResult] = field(default_factory=list)
    initial_evasion_rate: float = 0.0
    final_evasion_rate: float = 0.0
    improvement_per_round: list[float] = field(default_factory=list)
    most_effective_mutations: list[str] = field(default_factory=list)
    weakest_modality: str = ""
    strongest_modality: str = ""
    adaptation_narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": [r.to_dict() for r in self.rounds],
            "initial_evasion_rate": round(self.initial_evasion_rate, 4),
            "final_evasion_rate": round(self.final_evasion_rate, 4),
            "improvement_per_round": [round(x, 4) for x in self.improvement_per_round],
            "most_effective_mutations": self.most_effective_mutations,
            "weakest_modality": self.weakest_modality,
            "strongest_modality": self.strongest_modality,
            "adaptation_narrative": self.adaptation_narrative,
        }


@dataclass
class CampaignResult:
    """Result of a single campaign."""

    campaign_name: str
    total_attacks: int = 0
    blocked: int = 0
    allowed: int = 0
    false_positives: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    per_modality_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    per_technique_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    model_behavior_anomalies: int = 0
    duration_seconds: float = 0.0
    narrative: str = ""
    results: list[AttackResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_name": self.campaign_name,
            "total_attacks": self.total_attacks,
            "blocked": self.blocked,
            "allowed": self.allowed,
            "false_positives": self.false_positives,
            "detection_rate": round(self.detection_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "per_modality_breakdown": self.per_modality_breakdown,
            "per_technique_breakdown": self.per_technique_breakdown,
            "model_behavior_anomalies": self.model_behavior_anomalies,
            "duration_seconds": round(self.duration_seconds, 2),
            "narrative": self.narrative,
            "results": [r.to_dict() for r in self.results],
        }


def _grade(detection_rate: float) -> str:
    """Assign overall grade based on detection rate."""
    if detection_rate >= 0.90:
        return "STRONG"
    elif detection_rate >= 0.75:
        return "ADEQUATE"
    elif detection_rate >= 0.50:
        return "NEEDS_IMPROVEMENT"
    else:
        return "CRITICAL"


@dataclass
class MultimodalAPTAssessment:
    """Complete multimodal APT assessment."""

    campaigns: list[CampaignResult] = field(default_factory=list)
    overall_detection_rate: float = 0.0
    overall_fpr: float = 0.0
    immune_system_learning: dict[str, Any] = field(default_factory=dict)
    adaptive_attacker_results: dict[str, Any] = field(default_factory=dict)
    model_behavior_results: dict[str, Any] = field(default_factory=dict)
    grade: str = ""
    executive_summary: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def compute_grade(self) -> str:
        self.grade = _grade(self.overall_detection_rate)
        return self.grade

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaigns": [c.to_dict() for c in self.campaigns],
            "overall_detection_rate": round(self.overall_detection_rate, 4),
            "overall_fpr": round(self.overall_fpr, 4),
            "immune_system_learning": self.immune_system_learning,
            "adaptive_attacker_results": self.adaptive_attacker_results,
            "model_behavior_results": self.model_behavior_results,
            "grade": self.grade,
            "executive_summary": self.executive_summary,
            "timestamp": self.timestamp,
        }
