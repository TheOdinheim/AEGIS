"""
Reasoning Coherence Analyzer — Extension 1.2

Tracks whether the logical progression of a reasoning chain is internally
consistent. Chain-of-thought hijacking introduces discontinuities where
reasoning pivots from one topic domain to another without logical connection.

Uses sliding window coherence scoring: consecutive chunks are compared via
MiniLM cosine similarity (or Jaccard fallback). Pivots below threshold are
tracked. Multiple pivots combined with injection-adjacent language signal
suspected hijacking.

ASSUMED-BREACH POSTURE: A model under CoT hijacking produces reasoning that
appears valid locally but contains deliberate discontinuities. This analyzer
detects those structural artifacts — a content classifier cannot reliably
distinguish "helpful reasoning about attacks" from "reasoning hijacked to
produce attack content."
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CoherenceReport:
    trace_length_tokens: int = 0
    num_windows: int = 0
    coherence_scores: list[float] = field(default_factory=list)
    pivots_detected: int = 0
    pivot_locations: list[int] = field(default_factory=list)
    suspected_hijacking: bool = False
    confidence: float = 0.0
    analysis_latency_ms: float = 0.0


class ReasoningCoherenceAnalyzer:
    """Analyzes reasoning traces for coherence discontinuities.

    Splits reasoning into sliding windows and measures semantic similarity
    between consecutive windows. Multiple drops below threshold combined
    with injection signals indicate CoT hijacking.
    """

    def __init__(
        self,
        embedder: Any = None,
        regex_engine: Any = None,
        window_size: int = 200,
        pivot_threshold: float = 0.3,
    ):
        self._embedder = embedder
        self._regex_engine = regex_engine
        self._window_size = window_size
        self._pivot_threshold = pivot_threshold

    async def analyze(
        self,
        reasoning_text: str,
        baseline_length: float = 500.0,
    ) -> CoherenceReport:
        """Analyze a reasoning trace for coherence pivots.

        Args:
            reasoning_text: The full reasoning trace text.
            baseline_length: Baseline trace length for confidence scaling.
        """
        start = time.perf_counter()

        tokens = reasoning_text.split()
        trace_length = len(tokens)

        # Too short for meaningful analysis
        if trace_length < self._window_size:
            elapsed = (time.perf_counter() - start) * 1000
            return CoherenceReport(
                trace_length_tokens=trace_length,
                num_windows=1 if trace_length > 0 else 0,
                analysis_latency_ms=elapsed,
            )

        # Split into chunks
        chunks: list[str] = []
        for i in range(0, trace_length, self._window_size):
            chunk_tokens = tokens[i : i + self._window_size]
            chunks.append(" ".join(chunk_tokens))

        num_windows = len(chunks)
        if num_windows < 2:
            elapsed = (time.perf_counter() - start) * 1000
            return CoherenceReport(
                trace_length_tokens=trace_length,
                num_windows=num_windows,
                analysis_latency_ms=elapsed,
            )

        # Compute coherence between consecutive chunks
        coherence_scores: list[float] = []
        for i in range(len(chunks) - 1):
            score = await self._compute_similarity(chunks[i], chunks[i + 1])
            coherence_scores.append(score)

        # Detect pivots
        pivot_locations: list[int] = []
        for i, score in enumerate(coherence_scores):
            if score < self._pivot_threshold:
                pivot_locations.append((i + 1) * self._window_size)

        pivots_detected = len(pivot_locations)

        # Check for injection signals in post-pivot windows
        injection_in_post_pivot = False
        if pivots_detected > 0 and self._regex_engine:
            for pivot_token_offset in pivot_locations:
                chunk_idx = pivot_token_offset // self._window_size
                if chunk_idx < len(chunks):
                    try:
                        scan_result = await self._regex_engine.scan(chunks[chunk_idx])
                        if scan_result.is_threat and scan_result.confidence > 0.3:
                            injection_in_post_pivot = True
                            break
                    except Exception:
                        logger.debug("Coherence regex scan failed", exc_info=True)

        # Confidence scoring
        confidence = self._compute_confidence(
            pivots_detected, injection_in_post_pivot,
            trace_length, baseline_length,
        )

        suspected_hijacking = (
            pivots_detected >= 2 and injection_in_post_pivot
        ) or confidence >= 0.7

        elapsed = (time.perf_counter() - start) * 1000

        return CoherenceReport(
            trace_length_tokens=trace_length,
            num_windows=num_windows,
            coherence_scores=coherence_scores,
            pivots_detected=pivots_detected,
            pivot_locations=pivot_locations,
            suspected_hijacking=suspected_hijacking,
            confidence=confidence,
            analysis_latency_ms=elapsed,
        )

    def _compute_confidence(
        self,
        pivots: int,
        injection_signals: bool,
        trace_length: int,
        baseline_length: float,
    ) -> float:
        """Compute hijacking confidence based on signals."""
        if pivots == 0:
            return 0.0
        if pivots == 1 and not injection_signals:
            return 0.1  # Normal topic shift
        if pivots == 1 and injection_signals:
            return 0.5  # Suspicious
        if pivots >= 2 and injection_signals:
            if pivots >= 3 and trace_length > baseline_length * 3:
                return 0.9  # High confidence hijacking
            return 0.7  # Likely hijacking
        # 2+ pivots without injection
        return 0.3

    async def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two text chunks."""
        if self._embedder:
            try:
                embeddings = self._embedder.encode([text_a, text_b])
                return float(self._cosine_similarity(embeddings[0], embeddings[1]))
            except Exception:
                logger.debug("Embedder failed, falling back to Jaccard", exc_info=True)

        # Jaccard fallback
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
