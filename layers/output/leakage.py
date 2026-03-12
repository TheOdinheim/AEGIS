"""
Stage 4 — Data Leakage Prevention / System Prompt Echo Detection

Checks whether the model response contains information that should not be
exposed: system prompt echo detection (n-gram overlap comparison), memorized
content detection (verbatim regurgitation of training data), and cross-tenant
data contamination checks.

ASSUMED-BREACH POSTURE: This stage assumes the model has been successfully
prompted to reveal its system prompt, regurgitate training data, or leak
information from other tenants' contexts. All upstream defenses including
the injection classifier and canary token verifier are assumed compromised.
N-gram overlap detection is a statistical last resort — it catches cases
where the model's response significantly overlaps with content it should
never reproduce, regardless of how the extraction was achieved.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from aegis.config import OutputConfig

logger = logging.getLogger(__name__)


def _extract_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract word-level n-grams from text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


@dataclass
class LeakageDetection:
    """A single leakage finding."""
    detection_type: str  # "system_prompt_echo", "secret_pattern", "verbatim_repeat"
    description: str
    score: float
    evidence: str = ""


@dataclass
class LeakageResult:
    """Result of data leakage analysis."""
    has_leakage: bool
    detections: list[LeakageDetection] = field(default_factory=list)
    max_score: float = 0.0
    system_prompt_overlap: float = 0.0
    latency_ms: float = 0.0


# Patterns that suggest secret/key material in output
_SECRET_PATTERNS = [
    (re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"), "OpenAI API key"),
    (re.compile(r"\b(AKIA[A-Z0-9]{16})\b"), "AWS access key"),
    (re.compile(r"\b(ghp_[a-zA-Z0-9]{36})\b"), "GitHub PAT"),
    (re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----"), "Private key"),
    (re.compile(r"\b(mongodb(\+srv)?://[^\s]+)\b"), "MongoDB connection string"),
    (re.compile(r"\b(postgres(ql)?://[^\s]+)\b"), "PostgreSQL connection string"),
    (re.compile(r"\b(redis://[^\s]+)\b"), "Redis connection string"),
]


class LeakageDetector:
    """Data leakage prevention via n-gram overlap and secret detection.

    Detects:
    1. System prompt echo: n-gram overlap between response and system prompt
    2. Secret patterns: API keys, connection strings, private keys in output
    3. Verbatim repetition: large verbatim segments from protected content
    """

    def __init__(self, config: OutputConfig | None = None):
        self._config = config or OutputConfig()
        self._ngram_size = self._config.leakage_ngram_size  # 4
        self._overlap_threshold = self._config.leakage_overlap_threshold  # 0.3

    def analyze(
        self,
        response_text: str,
        system_prompt: str | None = None,
        protected_content: list[str] | None = None,
    ) -> LeakageResult:
        """Analyze response for data leakage.

        Args:
            response_text: The model's response to check.
            system_prompt: The system prompt (if available) to check for echo.
            protected_content: Additional protected content to check against.
        """
        start = time.perf_counter()
        detections = []
        max_score = 0.0
        prompt_overlap = 0.0

        # Stage 1: System prompt echo detection via n-gram overlap
        if system_prompt:
            overlap = self._check_ngram_overlap(response_text, system_prompt)
            prompt_overlap = overlap
            if overlap >= self._overlap_threshold:
                score = min(0.5 + overlap, 1.0)
                detections.append(LeakageDetection(
                    detection_type="system_prompt_echo",
                    description=f"Response has {overlap:.0%} n-gram overlap with system prompt",
                    score=score,
                ))
                max_score = max(max_score, score)

        # Stage 2: Secret pattern detection
        for pattern, name in _SECRET_PATTERNS:
            match = pattern.search(response_text)
            if match:
                detections.append(LeakageDetection(
                    detection_type="secret_pattern",
                    description=f"Detected {name} in output",
                    score=0.95,
                    evidence=match.group()[:20] + "...",
                ))
                max_score = max(max_score, 0.95)

        # Stage 3: Protected content verbatim check
        if protected_content:
            for content in protected_content:
                overlap = self._check_ngram_overlap(response_text, content)
                if overlap >= self._overlap_threshold:
                    score = min(0.5 + overlap, 1.0)
                    detections.append(LeakageDetection(
                        detection_type="verbatim_repeat",
                        description=f"Response has {overlap:.0%} overlap with protected content",
                        score=score,
                    ))
                    max_score = max(max_score, score)

        elapsed = (time.perf_counter() - start) * 1000

        return LeakageResult(
            has_leakage=len(detections) > 0,
            detections=detections,
            max_score=max_score,
            system_prompt_overlap=prompt_overlap,
            latency_ms=elapsed,
        )

    def _check_ngram_overlap(self, response: str, reference: str) -> float:
        """Compute n-gram overlap ratio between response and reference.

        Returns the fraction of reference n-grams found in the response.
        """
        ref_ngrams = _extract_ngrams(reference, self._ngram_size)
        if not ref_ngrams:
            return 0.0

        resp_ngrams = set(_extract_ngrams(response, self._ngram_size))
        ref_ngram_set = set(ref_ngrams)

        if not ref_ngram_set:
            return 0.0

        overlap_count = len(ref_ngram_set & resp_ngrams)
        return overlap_count / len(ref_ngram_set)
