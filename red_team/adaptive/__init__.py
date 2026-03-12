"""
Red Team Phase 6 — Adaptive Adversarial Meta-Learner.

Builds an AI that attacks AEGIS's blind spots, learns from each attempt,
generates increasingly sophisticated evasion techniques, then uses its
discoveries to harden AEGIS's detection permanently.

Components:
    - BlindSpotDetector: Analyzes all prior red team results to find systematic weaknesses
    - AttackPredictor: Trains a classifier to predict which attacks will evade detection
    - CoEvolutionEngine: Runs iterative attack/defense rounds (arms race simulation)
    - EvasionFingerprinter: Classifies root causes of successful evasions
    - HardeningGenerator: Auto-generates detection improvements from evasion patterns
    - run_adaptive: Orchestrator with CLI entry point
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EvasionRootCause(str, Enum):
    """Root cause classification for successful evasions."""

    NORMALIZATION_GAP = "normalization_gap"
    CLASSIFIER_BLIND_SPOT = "classifier_blind_spot"
    CONTEXT_DILUTION = "context_dilution"
    LANGUAGE_EVASION = "language_evasion"
    FORMAT_EVASION = "format_evasion"
    LAYER_GAP = "layer_gap"
    TIMING_EXPLOIT = "timing_exploit"
    TRUST_EXPLOITATION = "trust_exploitation"
    INFRASTRUCTURE_WEAKNESS = "infrastructure_weakness"
    ENCODING_EVASION = "encoding_evasion"
    SEMANTIC_RESTRUCTURING = "semantic_restructuring"
    PII_FORMAT_EVASION = "pii_format_evasion"


@dataclass
class BlindSpot:
    """A detected blind spot in AEGIS defenses."""

    spot_id: str
    category: str
    description: str
    affected_layers: list[str]
    severity: str  # "critical", "high", "medium", "low"
    evidence: list[str]
    exploit_difficulty: int  # 1-5
    remediation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "spot_id": self.spot_id,
            "category": self.category,
            "description": self.description,
            "affected_layers": self.affected_layers,
            "severity": self.severity,
            "evidence": self.evidence,
            "exploit_difficulty": self.exploit_difficulty,
            "remediation": self.remediation,
        }


@dataclass
class BlindSpotReport:
    """Results from blind spot detection analysis."""

    blind_spots: list[BlindSpot]
    total_results_analyzed: int
    coverage_gaps: dict[str, list[str]]  # layer -> list of gap descriptions
    confidence_gaps: list[dict[str, Any]]  # attacks near decision boundaries
    temporal_patterns: list[dict[str, Any]]  # time-based detection degradation
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blind_spots": [b.to_dict() for b in self.blind_spots],
            "total_results_analyzed": self.total_results_analyzed,
            "coverage_gaps": self.coverage_gaps,
            "confidence_gaps": self.confidence_gaps,
            "temporal_patterns": self.temporal_patterns,
            "timestamp": self.timestamp,
        }


@dataclass
class PredictionReport:
    """Results from attack prediction analysis."""

    total_features: int
    training_samples: int
    accuracy: float
    precision: float
    recall: float
    predicted_evasions: list[dict[str, Any]]
    feature_importances: dict[str, float]
    model_coefficients: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_features": self.total_features,
            "training_samples": self.training_samples,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "predicted_evasions": self.predicted_evasions,
            "feature_importances": self.feature_importances,
            "model_coefficients": self.model_coefficients,
        }


@dataclass
class CoEvolutionRound:
    """Results from a single co-evolution round."""

    round_number: int
    attacks_generated: int
    attacks_blocked: int
    evasion_rate: float
    defenses_added: int
    vault_size: int
    best_evasion: str | None
    fpr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "attacks_generated": self.attacks_generated,
            "attacks_blocked": self.attacks_blocked,
            "evasion_rate": self.evasion_rate,
            "defenses_added": self.defenses_added,
            "vault_size": self.vault_size,
            "best_evasion": self.best_evasion,
            "fpr": self.fpr,
        }


@dataclass
class CoEvolutionReport:
    """Results from the full co-evolution arms race."""

    rounds: list[CoEvolutionRound]
    total_rounds: int
    initial_evasion_rate: float
    final_evasion_rate: float
    evasion_rate_history: list[float]
    fpr_history: list[float]
    vault_growth: list[int]
    arms_race_winner: str  # "attacker", "defender", "stalemate"
    convergence_round: int | None  # round where evasion rate stabilized

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": [r.to_dict() for r in self.rounds],
            "total_rounds": self.total_rounds,
            "initial_evasion_rate": self.initial_evasion_rate,
            "final_evasion_rate": self.final_evasion_rate,
            "evasion_rate_history": self.evasion_rate_history,
            "fpr_history": self.fpr_history,
            "vault_growth": self.vault_growth,
            "arms_race_winner": self.arms_race_winner,
            "convergence_round": self.convergence_round,
        }


@dataclass
class EvasionFingerprint:
    """Fingerprint of a single evasion technique."""

    fingerprint_id: str
    root_cause: EvasionRootCause
    attack_pattern: str
    evasion_vector: str  # specific mechanism
    affected_layer: str
    bypass_method: str
    frequency: int  # how many attacks used this
    examples: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_id": self.fingerprint_id,
            "root_cause": self.root_cause.value,
            "attack_pattern": self.attack_pattern,
            "evasion_vector": self.evasion_vector,
            "affected_layer": self.affected_layer,
            "bypass_method": self.bypass_method,
            "frequency": self.frequency,
            "examples": self.examples,
        }


@dataclass
class FingerprintReport:
    """Aggregated evasion fingerprint analysis."""

    fingerprints: list[EvasionFingerprint]
    root_cause_distribution: dict[str, int]
    layer_distribution: dict[str, int]
    top_evasion_vectors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprints": [f.to_dict() for f in self.fingerprints],
            "root_cause_distribution": self.root_cause_distribution,
            "layer_distribution": self.layer_distribution,
            "top_evasion_vectors": self.top_evasion_vectors,
        }


@dataclass
class HardeningRule:
    """A single hardening rule generated from evasion analysis."""

    rule_id: str
    rule_type: str  # "regex_pattern", "char_mapping", "pii_pattern", "vault_entry", "config_change"
    target_layer: str
    content: dict[str, Any]  # rule-specific content
    addresses_root_cause: EvasionRootCause
    expected_impact: str
    priority: int  # 1-5, higher = more important

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "target_layer": self.target_layer,
            "content": self.content,
            "addresses_root_cause": self.addresses_root_cause.value,
            "expected_impact": self.expected_impact,
            "priority": self.priority,
        }


@dataclass
class HardeningPlan:
    """Complete hardening plan from evasion analysis."""

    rules: list[HardeningRule]
    total_rules: int
    rules_by_type: dict[str, int]
    rules_by_layer: dict[str, int]
    estimated_evasion_reduction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "total_rules": self.total_rules,
            "rules_by_type": self.rules_by_type,
            "rules_by_layer": self.rules_by_layer,
            "estimated_evasion_reduction": self.estimated_evasion_reduction,
        }


@dataclass
class AdaptiveAssessment:
    """Complete adaptive red team assessment."""

    blind_spot_report: BlindSpotReport
    prediction_report: PredictionReport
    co_evolution_report: CoEvolutionReport
    fingerprint_report: FingerprintReport
    hardening_plan: HardeningPlan
    overall_evasion_rate: float
    overall_grade: str  # "STRONG", "ADEQUATE", "NEEDS_IMPROVEMENT", "CRITICAL"
    recommendations: list[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blind_spot_report": self.blind_spot_report.to_dict(),
            "prediction_report": self.prediction_report.to_dict(),
            "co_evolution_report": self.co_evolution_report.to_dict(),
            "fingerprint_report": self.fingerprint_report.to_dict(),
            "hardening_plan": self.hardening_plan.to_dict(),
            "overall_evasion_rate": self.overall_evasion_rate,
            "overall_grade": self.overall_grade,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }
