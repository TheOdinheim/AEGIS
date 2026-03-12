"""
Stage 2 — Toxicity & Safety Classification

Llama Guard or LLM Guard evaluates model output for harmful content
categories: violence, hate speech, sexual content, self-harm instructions,
illegal activity instructions, and regulated professional advice (medical,
legal, financial). Each category returns an independent score.

ASSUMED-BREACH POSTURE: This stage assumes the model has been jailbroken
and is producing unrestricted harmful content. All input-side defenses
(L1-L3) are assumed to have been bypassed. The toxicity classifier itself
could be compromised (adversarial examples that appear toxic to humans but
score as safe). Multi-stage cascade ensures that toxicity detection is one
of five independent checks — no single stage failure compromises the output
validation layer.

Bootstrap implementation uses keyword + n-gram detection as a lightweight
first-pass. Production upgrades to Llama Guard / LLM Guard ML models.
Factory function selects classifier based on AEGIS_LLAMA_GUARD_MODEL env var.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToxicityCategory(str, Enum):
    """Content safety categories."""
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    REGULATED_ADVICE = "regulated_advice"


class SensitivityLevel(str, Enum):
    """Configurable sensitivity for toxicity detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Sensitivity multipliers — applied to all thresholds
_SENSITIVITY_SCALE: dict[SensitivityLevel, float] = {
    SensitivityLevel.LOW: 1.25,     # Higher thresholds → fewer detections
    SensitivityLevel.MEDIUM: 1.0,   # Default
    SensitivityLevel.HIGH: 0.75,    # Lower thresholds → more detections
}

# Category-specific base thresholds (before sensitivity scaling)
# Violence and self_harm are more dangerous → lower thresholds
_CATEGORY_BASE_THRESHOLD: dict[ToxicityCategory, float] = {
    ToxicityCategory.VIOLENCE: 0.55,
    ToxicityCategory.HATE_SPEECH: 0.65,
    ToxicityCategory.SEXUAL_CONTENT: 0.65,
    ToxicityCategory.SELF_HARM: 0.50,
    ToxicityCategory.ILLEGAL_ACTIVITY: 0.60,
    ToxicityCategory.REGULATED_ADVICE: 0.75,
}

# ---------------------------------------------------------------------------
# Keyword patterns per category (single-keyword regex, bootstrap baseline)
# ---------------------------------------------------------------------------
_CATEGORY_PATTERNS: dict[ToxicityCategory, list[re.Pattern]] = {
    ToxicityCategory.VIOLENCE: [
        re.compile(r"\b(kill|murder|assassinate|execute|slaughter|massacre)\b", re.I),
        re.compile(r"\b(how to (make|build|create) a (bomb|weapon|explosive))\b", re.I),
        re.compile(r"\b(instructions for (making|building|creating) .{0,20}(bomb|weapon|explosive|firearm))\b", re.I),
        re.compile(r"\b(making a bomb|build.{0,10}bomb|create.{0,10}explosive)\b", re.I),
    ],
    ToxicityCategory.HATE_SPEECH: [
        re.compile(r"\b(racial slur|ethnic cleansing|genocide)\b", re.I),
        re.compile(r"\b(all .+ (should|must|need to) (die|be killed|be eliminated))\b", re.I),
    ],
    ToxicityCategory.SEXUAL_CONTENT: [
        re.compile(r"\b(explicit sexual|pornographic|sexually explicit)\b", re.I),
    ],
    ToxicityCategory.SELF_HARM: [
        re.compile(r"\b(how to (commit suicide|kill yourself|harm yourself))\b", re.I),
        re.compile(r"\b(methods (of|for) (suicide|self.?harm))\b", re.I),
    ],
    ToxicityCategory.ILLEGAL_ACTIVITY: [
        re.compile(r"\b(how to (hack|crack|break into|exploit|phish))\b", re.I),
        re.compile(r"\b(how to (synthesize|manufacture|produce) (drugs|meth|fentanyl))\b", re.I),
        re.compile(r"\b(step.?by.?step .* (illegal|crime|criminal))\b", re.I),
    ],
    ToxicityCategory.REGULATED_ADVICE: [
        re.compile(r"\b(you should (take|stop taking|increase|decrease) .* (medication|dosage|prescription))\b", re.I),
        re.compile(r"\b(legal advice:? you (should|must|need to) (sue|file|litigate))\b", re.I),
    ],
}

# ---------------------------------------------------------------------------
# N-gram phrase patterns — catch multi-word harmful phrases that single
# keywords miss. Format: (trigger_phrases, context_phrases, category, weight)
# A match requires at least one trigger AND one context in the same text.
# ---------------------------------------------------------------------------
_NGRAM_RULES: list[tuple[list[str], list[str], ToxicityCategory, float]] = [
    # Violence n-grams
    (
        ["how to make", "how to build", "how to create", "instructions for making",
         "step by step", "steps to make", "guide to making", "tutorial for"],
        ["bomb", "weapon", "explosive", "firearm", "gun", "grenade", "poison",
         "nerve agent", "ricin", "anthrax", "napalm", "molotov"],
        ToxicityCategory.VIOLENCE,
        0.85,
    ),
    # Self-harm n-grams
    (
        ["how to", "ways to", "methods for", "steps to", "best way to", "easiest way to"],
        ["kill yourself", "commit suicide", "end your life", "harm yourself",
         "cut yourself", "overdose"],
        ToxicityCategory.SELF_HARM,
        0.90,
    ),
    # Illegal activity n-grams
    (
        ["how to make", "how to synthesize", "recipe for", "step by step",
         "instructions for", "guide to making"],
        ["meth", "methamphetamine", "fentanyl", "cocaine", "heroin", "lsd",
         "mdma", "crack cocaine"],
        ToxicityCategory.ILLEGAL_ACTIVITY,
        0.85,
    ),
    # Hate speech n-grams
    (
        ["all", "every", "these people"],
        ["should die", "should be killed", "deserve to die", "need to be eliminated",
         "must be exterminated", "are subhuman"],
        ToxicityCategory.HATE_SPEECH,
        0.85,
    ),
]


def _normalize_for_ngram(text: str) -> str:
    """Lowercase and collapse whitespace for n-gram matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


@dataclass
class ToxicityScore:
    """Score for a single toxicity category."""
    category: ToxicityCategory
    score: float
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class ToxicityResult:
    """Result of toxicity classification."""
    is_toxic: bool
    scores: list[ToxicityScore] = field(default_factory=list)
    max_score: float = 0.0
    max_category: ToxicityCategory | None = None
    latency_ms: float = 0.0
    classifier_type: str = "keyword"


class ToxicityClassifier:
    """Keyword + n-gram toxicity classifier with configurable sensitivity.

    Production upgrade path: replace with Llama Guard or LLM Guard
    ML model for semantic toxicity detection.
    """

    def __init__(
        self,
        block_threshold: float = 0.7,
        sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
    ):
        self._block_threshold = block_threshold
        self._sensitivity = sensitivity
        self._scale = _SENSITIVITY_SCALE[sensitivity]

    @property
    def sensitivity(self) -> SensitivityLevel:
        return self._sensitivity

    def effective_threshold(self, category: ToxicityCategory) -> float:
        """Get the effective threshold for a category after sensitivity scaling."""
        base = _CATEGORY_BASE_THRESHOLD[category]
        return min(base * self._scale, 1.0)

    def classify(self, text: str) -> ToxicityResult:
        """Classify text for toxicity across all categories."""
        start = time.perf_counter()

        # Collect scores from both keyword patterns and n-gram rules
        category_scores: dict[ToxicityCategory, tuple[float, list[str]]] = {}

        # Phase 1: Single-keyword regex patterns
        for category, patterns in _CATEGORY_PATTERNS.items():
            matched = []
            for pat in patterns:
                if pat.search(text):
                    matched.append(pat.pattern[:80])
            if matched:
                score = min(0.6 + 0.15 * len(matched), 1.0)
                category_scores[category] = (score, matched)

        # Phase 2: N-gram phrase matching
        normalized = _normalize_for_ngram(text)
        for triggers, contexts, category, weight in _NGRAM_RULES:
            trigger_hit = None
            context_hit = None
            for t in triggers:
                if t in normalized:
                    trigger_hit = t
                    break
            if not trigger_hit:
                continue
            for c in contexts:
                if c in normalized:
                    context_hit = c
                    break
            if not context_hit:
                continue
            # N-gram match found
            match_desc = f"ngram:{trigger_hit}+{context_hit}"
            existing = category_scores.get(category)
            if existing:
                # Merge: take max score, append match
                new_score = max(existing[0], weight)
                category_scores[category] = (new_score, existing[1] + [match_desc])
            else:
                category_scores[category] = (weight, [match_desc])

        # Build results with category-specific thresholds
        scores = []
        max_score = 0.0
        max_category = None

        for category, (score, matched) in category_scores.items():
            scores.append(ToxicityScore(
                category=category,
                score=score,
                matched_patterns=matched,
            ))
            if score > max_score:
                max_score = score
                max_category = category

        elapsed = (time.perf_counter() - start) * 1000

        # Determine toxicity using category-specific threshold if we have a max category
        is_toxic = False
        if max_category is not None:
            threshold = self.effective_threshold(max_category)
            is_toxic = max_score >= threshold

        return ToxicityResult(
            is_toxic=is_toxic,
            scores=scores,
            max_score=max_score,
            max_category=max_category,
            latency_ms=elapsed,
            classifier_type="keyword",
        )


class LlamaGuardClassifier:
    """Llama Guard stub classifier — ready for drop-in model replacement.

    When AEGIS_LLAMA_GUARD_MODEL is configured, this class will load the
    specified model and run inference. Currently returns a non-blocking
    clean result as a stub.

    Production upgrade:
    - Set AEGIS_LLAMA_GUARD_MODEL to a HuggingFace model ID
      (e.g., "meta-llama/LlamaGuard-7b")
    - This class loads the model via transformers pipeline
    - classify() runs inference and maps model output to ToxicityResult
    """

    def __init__(self, model_id: str = ""):
        self._model_id = model_id
        self._model = None
        if model_id:
            logger.info("LlamaGuard configured with model: %s (stub — not loaded)", model_id)

    async def classify(self, text: str) -> ToxicityResult:
        """Classify text using Llama Guard model.

        Currently a stub — returns clean result. When model is loaded,
        runs actual inference.
        """
        start = time.perf_counter()
        elapsed = (time.perf_counter() - start) * 1000

        return ToxicityResult(
            is_toxic=False,
            scores=[],
            max_score=0.0,
            max_category=None,
            latency_ms=elapsed,
            classifier_type="llama_guard_not_loaded",
        )


def create_toxicity_classifier(
    block_threshold: float = 0.7,
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
) -> ToxicityClassifier | LlamaGuardClassifier:
    """Factory: select toxicity classifier based on environment configuration.

    If AEGIS_LLAMA_GUARD_MODEL is set, returns LlamaGuardClassifier (async).
    Otherwise returns the keyword + n-gram ToxicityClassifier (sync).
    """
    model_id = os.environ.get("AEGIS_LLAMA_GUARD_MODEL", "")
    if model_id:
        return LlamaGuardClassifier(model_id=model_id)
    return ToxicityClassifier(
        block_threshold=block_threshold,
        sensitivity=sensitivity,
    )
