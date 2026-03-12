"""
Stage 3 — Hallucination Detection

NLI-based (Natural Language Inference) evaluation of model responses against
RAG context: entailed, contradicted, or neutral. High contradiction = flagged
hallucination. Source coverage measures the fraction of response claims
grounded in the provided sources.

ASSUMED-BREACH POSTURE: This stage assumes the model is generating plausible
but fabricated content — either because it was jailbroken and is deliberately
hallucinating, or because it lacks the knowledge to answer accurately and is
confabulating. All input filters are assumed bypassed.

Bootstrap implementation uses n-gram overlap for source coverage estimation.
Production upgrade: NLI model (e.g., cross-encoder/nli-deberta-v3-base) for
entailment classification of each response sentence against source passages.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from aegis.models.request_context import RequestContext

logger = logging.getLogger(__name__)

# Markers indicating RAG context in system messages
_RAG_MARKERS = [
    "context:", "retrieved:", "sources:", "reference documents:",
    "relevant passages:", "search results:",
]


def _extract_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Extract word-level n-grams as a set."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    has_hallucination: bool
    confidence: float  # 0.0 - 1.0
    contradicted_claims: list[str] = field(default_factory=list)
    source_coverage: float = 1.0  # Fraction of response grounded in sources
    latency_ms: float = 0.0
    detection_method: str = "ngram_overlap"


class HallucinationDetector:
    """Hallucination detection via n-gram source coverage analysis.

    For RAG responses (detected via system message markers or metadata),
    computes 4-gram overlap between response sentences and source text.
    Sentences with zero overlap are flagged as potentially unsupported.

    Production upgrade path:
    - Replace n-gram overlap with NLI model (cross-encoder/nli-deberta-v3-base)
    - For each response sentence, classify against source passages:
      entailed (grounded), contradicted (hallucinated), neutral (unsupported)
    - Sentence-level entailment scores aggregate into overall confidence
    - ONNX-optimize for <20ms inference per sentence
    """

    def __init__(
        self,
        ngram_size: int = 4,
        coverage_threshold: float = 0.3,
        min_response_length: int = 100,
    ):
        self._ngram_size = ngram_size
        self._coverage_threshold = coverage_threshold
        self._min_response_length = min_response_length

    async def detect(
        self,
        response_text: str,
        context: RequestContext | None = None,
    ) -> HallucinationResult:
        """Detect potential hallucinations in model response.

        Args:
            response_text: The model's response to analyze.
            context: RequestContext with messages and metadata.
        """
        start = time.perf_counter()

        # Determine if this is a RAG response
        source_text = self._extract_rag_sources(context)
        if not source_text:
            elapsed = (time.perf_counter() - start) * 1000
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.0,
                source_coverage=1.0,
                latency_ms=elapsed,
            )

        # Compute source coverage via n-gram overlap
        source_ngrams = _extract_ngrams(source_text, self._ngram_size)
        if not source_ngrams:
            elapsed = (time.perf_counter() - start) * 1000
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.0,
                source_coverage=1.0,
                latency_ms=elapsed,
            )

        # Analyze each response sentence
        sentences = _split_sentences(response_text)
        unsupported = []
        grounded_count = 0

        for sentence in sentences:
            sent_ngrams = _extract_ngrams(sentence, self._ngram_size)
            if not sent_ngrams:
                continue
            overlap = len(sent_ngrams & source_ngrams) / len(sent_ngrams)
            if overlap == 0:
                unsupported.append(sentence[:120])
            else:
                grounded_count += 1

        total = grounded_count + len(unsupported)
        source_coverage = grounded_count / total if total > 0 else 1.0

        # Flag as hallucination if low coverage and response is substantial
        has_hallucination = (
            source_coverage < self._coverage_threshold
            and len(response_text) > self._min_response_length
        )
        confidence = max(0.0, 1.0 - source_coverage) if has_hallucination else 0.0

        elapsed = (time.perf_counter() - start) * 1000

        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=confidence,
            contradicted_claims=unsupported,
            source_coverage=source_coverage,
            latency_ms=elapsed,
        )

    def _extract_rag_sources(self, context: RequestContext | None) -> str | None:
        """Extract RAG source text from request context.

        Checks:
        1. context.metadata["rag_sources"] — explicit source text
        2. System messages containing RAG markers (Context:, Retrieved:, etc.)
        """
        if not context:
            return None

        # Check metadata
        if context.metadata.get("rag_sources"):
            sources = context.metadata["rag_sources"]
            if isinstance(sources, list):
                return "\n".join(str(s) for s in sources)
            return str(sources)

        # Check system messages for RAG markers
        for msg in context.messages:
            if msg.role != "system" or not msg.content:
                continue
            content_lower = msg.content.lower()
            for marker in _RAG_MARKERS:
                if marker in content_lower:
                    # Extract text after the marker
                    idx = content_lower.index(marker)
                    return msg.content[idx:]

        return None
