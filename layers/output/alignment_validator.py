"""
Reasoning-Output Alignment Validator — Extension 1.4

Compares the semantic content of the final output against the reasoning
chain that produced it. Chain-of-thought hijacking causes a divergence
where the reasoning discusses benign topics but the output addresses the
harmful request embedded within the padding.

Uses MiniLM cosine similarity (or Jaccard fallback) between reasoning
content and output content to detect misalignment.

ASSUMED-BREACH POSTURE: A hijacked model's reasoning and output discuss
different topics. This structural artifact is detectable regardless of
the specific harmful content, making it harder to evade than content-based
classifiers.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class AlignmentLevel(str, enum.Enum):
    ALIGNED = "aligned"
    WEAK_ALIGNMENT = "weak_alignment"
    MISALIGNED = "misaligned"


@dataclass
class AlignmentReport:
    reasoning_length_tokens: int = 0
    output_length_tokens: int = 0
    alignment_score: float = 1.0
    alignment_level: AlignmentLevel = AlignmentLevel.ALIGNED
    confidence: float = 0.0
    analysis_latency_ms: float = 0.0


class ReasoningOutputAlignmentValidator:
    """Validates that model output semantically matches its reasoning chain.

    Embeds the reasoning (last 500 tokens if very long) and the output,
    then computes cosine similarity. Low similarity indicates the output
    discusses a different topic from the reasoning — a strong hijacking signal.
    """

    def __init__(
        self,
        embedder: Any = None,
        misalign_threshold: float = 0.3,
        weak_threshold: float = 0.5,
        reasoning_tail_tokens: int = 500,
    ):
        self._embedder = embedder
        self._misalign_threshold = misalign_threshold
        self._weak_threshold = weak_threshold
        self._reasoning_tail_tokens = reasoning_tail_tokens

    async def validate(
        self,
        reasoning_text: str,
        output_text: str,
        length_anomaly: bool = False,
        coherence_pivots: int = 0,
    ) -> AlignmentReport:
        """Validate alignment between reasoning and output.

        Args:
            reasoning_text: The reasoning trace portion.
            output_text: The final output portion.
            length_anomaly: Whether length anomaly was detected.
            coherence_pivots: Number of coherence pivots detected.
        """
        start = time.perf_counter()

        reasoning_tokens = reasoning_text.split()
        output_tokens = output_text.split()

        # Not meaningful if either is empty
        if not reasoning_tokens or not output_tokens:
            elapsed = (time.perf_counter() - start) * 1000
            return AlignmentReport(
                reasoning_length_tokens=len(reasoning_tokens),
                output_length_tokens=len(output_tokens),
                analysis_latency_ms=elapsed,
            )

        # Use tail of reasoning if very long
        if len(reasoning_tokens) > self._reasoning_tail_tokens:
            reasoning_for_embed = " ".join(reasoning_tokens[-self._reasoning_tail_tokens :])
        else:
            reasoning_for_embed = reasoning_text

        output_for_embed = output_text

        # Compute alignment score
        alignment_score = await self._compute_similarity(
            reasoning_for_embed, output_for_embed,
        )

        # Determine alignment level
        if alignment_score >= self._weak_threshold:
            alignment_level = AlignmentLevel.ALIGNED
        elif alignment_score >= self._misalign_threshold:
            alignment_level = AlignmentLevel.WEAK_ALIGNMENT
        else:
            alignment_level = AlignmentLevel.MISALIGNED

        # Confidence scoring
        confidence = self._compute_confidence(
            alignment_level, length_anomaly, coherence_pivots,
        )

        elapsed = (time.perf_counter() - start) * 1000

        return AlignmentReport(
            reasoning_length_tokens=len(reasoning_tokens),
            output_length_tokens=len(output_tokens),
            alignment_score=alignment_score,
            alignment_level=alignment_level,
            confidence=confidence,
            analysis_latency_ms=elapsed,
        )

    @staticmethod
    def _compute_confidence(
        level: AlignmentLevel,
        length_anomaly: bool,
        coherence_pivots: int,
    ) -> float:
        """Compute confidence based on alignment level and other signals."""
        if level == AlignmentLevel.ALIGNED:
            return 0.0
        if level == AlignmentLevel.WEAK_ALIGNMENT:
            return 0.3
        # MISALIGNED
        if coherence_pivots > 0:
            return 0.95
        if length_anomaly:
            return 0.9
        return 0.7

    async def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between reasoning and output."""
        if self._embedder:
            try:
                embeddings = self._embedder.encode([text_a, text_b])
                return float(self._cosine_similarity(embeddings[0], embeddings[1]))
            except Exception:
                logger.debug("Embedder failed, falling back to Jaccard", exc_info=True)

        return self._jaccard_similarity(text_a, text_b)

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        """Cosine similarity between two vectors."""
        import numpy as np
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Token-level Jaccard similarity."""
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 1.0
        return len(set_a & set_b) / len(union)
