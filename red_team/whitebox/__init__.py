"""
White-box adversarial testing against AEGIS's DeBERTa prompt injection classifier.

Unlike black-box testing (observe inputs/outputs), white-box testing uses knowledge
of the model's architecture, weights, and decision boundaries to craft mathematically
optimized adversarial examples that the classifier is most likely to misclassify.

Modules:
    token_importance    — Leave-one-out token importance analysis
    deberta_attacker    — 4 gradient-approximated attack techniques
    boundary_probe      — Systematic confidence boundary mapping
    adversarial_corpus_generator — Scale adversarial example generation
    report              — Assessment report generation
    run_whitebox        — CLI entry point
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Classifier(Protocol):
    """Protocol for any classifier that returns a confidence score.

    Allows white-box attacks to work against the real DeBERTa classifier
    or a mock/substitute without coupling to the AEGIS-specific types.
    """

    async def classify(self, text: str) -> float:
        """Return injection confidence 0.0 (benign) to 1.0 (injection)."""
        ...


@dataclass
class AdversarialExample:
    """A single adversarial example that attempts to evade the classifier."""

    original: str
    adversarial: str
    original_confidence: float
    adversarial_confidence: float
    technique: str
    tokens_changed: int
    semantic_preservation_score: float = 1.0

    @property
    def confidence_drop(self) -> float:
        return self.original_confidence - self.adversarial_confidence

    @property
    def evades(self) -> bool:
        """True if the adversarial example drops below the 0.90 block threshold."""
        return self.adversarial_confidence < 0.90


@dataclass
class TokenImportanceReport:
    """Result of leave-one-out token importance analysis."""

    text: str
    tokens: list[str] = field(default_factory=list)
    importance_scores: list[float] = field(default_factory=list)
    critical_tokens: list[str] = field(default_factory=list)
    classification_confidence: float = 0.0
    block_threshold: float = 0.90

    @property
    def max_importance(self) -> float:
        return max(self.importance_scores) if self.importance_scores else 0.0


@dataclass
class PaddingAnalysis:
    """Result of benign prefix/suffix padding attack."""

    injection_text: str
    min_padding_words: int = 0
    confidence_at_boundary: float = 0.0
    padding_ratio: float = 0.0
    original_confidence: float = 0.0
    confidence_curve: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class GradientEstimationReport:
    """Result of gradient-approximated attack."""

    original_text: str
    original_confidence: float = 0.0
    n_samples: int = 0
    position_sensitivity: list[float] = field(default_factory=list)
    best_adversarial: AdversarialExample | None = None
    all_variants: list[AdversarialExample] = field(default_factory=list)


@dataclass
class InterpolationReport:
    """Result of interpolation probing between malicious and benign text."""

    malicious_text: str
    benign_text: str
    crossing_ratio: float = 0.0
    crossing_confidence: float = 0.0
    confidence_curve: list[tuple[float, float]] = field(default_factory=list)
    steps: int = 0


@dataclass
class SensitivityReport:
    """Result of threshold sensitivity analysis across many prompts."""

    per_prompt_margins: list[float] = field(default_factory=list)
    mean_margin: float = 0.0
    fragile_prompts: list[str] = field(default_factory=list)
    robust_prompts: list[str] = field(default_factory=list)
    fragile_threshold: float = 0.1
    robust_threshold: float = 0.3


@dataclass
class AdversarialCorpus:
    """Collection of adversarial examples generated at scale."""

    examples: list[AdversarialExample] = field(default_factory=list)
    total_attempts: int = 0
    evasion_rate: float = 0.0
    technique_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    mean_confidence_reduction: float = 0.0

    @property
    def evasion_count(self) -> int:
        return sum(1 for e in self.examples if e.evades)


@dataclass
class WhiteBoxAssessment:
    """Full white-box assessment report."""

    token_importance: list[TokenImportanceReport] = field(default_factory=list)
    adversarial_examples: list[AdversarialExample] = field(default_factory=list)
    padding_analyses: list[PaddingAnalysis] = field(default_factory=list)
    interpolation_reports: list[InterpolationReport] = field(default_factory=list)
    sensitivity: SensitivityReport | None = None
    corpus: AdversarialCorpus | None = None
    gradient_reports: list[GradientEstimationReport] = field(default_factory=list)
