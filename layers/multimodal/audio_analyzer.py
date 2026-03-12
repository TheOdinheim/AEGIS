"""
Audio Analyzer — L3 Adaptive slow-path audio analysis (50-200ms).

Performs deeper analysis complementing the L2 fast-path scanner:
1. Deep transcription → run through DeBERTa injection classifier
2. WaveGuard-style comparison: sanitize audio, transcribe both, compare
3. Full spectral analysis
4. Cross-modal consistency (audio transcription vs text prompt)

ASSUMED-BREACH POSTURE: WhisperInject demonstrates benign-sounding audio
that covertly induces harmful text generation with 86%+ success rate.
The WaveGuard comparison catches perturbations that survive transcription.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from aegis.models.scan_result import ScanResult, ThreatCategory
from aegis.layers.multimodal.audio_transcriber import (
    AudioTranscriber,
    TranscriptionResult,
    detect_audio_format,
)
from aegis.layers.multimodal.spectral_analyzer import SpectralAnalyzer, SpectralAnalysisResult
from aegis.layers.multimodal.audio_sanitizer import AudioSanitizer, AudioSanitizationReport

logger = logging.getLogger(__name__)


def _word_edit_distance(text_a: str, text_b: str) -> int:
    """Compute word-level Levenshtein edit distance."""
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()

    m, n = len(words_a), len(words_b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if words_a[i - 1] == words_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def _compute_divergence(original_text: str, sanitized_text: str) -> float:
    """Compute word-level divergence between original and sanitized transcriptions.

    Returns a score in [0.0, 1.0]:
    - 0.0: identical transcriptions (clean audio)
    - 1.0: completely different (highly adversarial)
    """
    if not original_text and not sanitized_text:
        return 0.0
    if not original_text or not sanitized_text:
        # One has text, other doesn't — significant divergence
        return 0.8

    total_words = max(
        len(original_text.split()),
        len(sanitized_text.split()),
    )
    if total_words == 0:
        return 0.0

    distance = _word_edit_distance(original_text, sanitized_text)
    return min(distance / total_words, 1.0)


@dataclass
class AudioAnalysisReport:
    """Report from L3 deep audio analysis."""
    scan_results: list[ScanResult] = field(default_factory=list)
    original_transcription: TranscriptionResult | None = None
    sanitized_transcription: TranscriptionResult | None = None
    sanitization_report: AudioSanitizationReport | None = None
    spectral_result: SpectralAnalysisResult | None = None
    divergence_score: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        return any(r.is_threat and r.confidence >= 0.85 for r in self.scan_results)

    @property
    def max_confidence(self) -> float:
        if not self.scan_results:
            return 0.0
        return max(r.confidence for r in self.scan_results)


class AudioAnalyzer:
    """L3 slow-path audio analyzer.

    Steps:
    1. Deep transcription → run through injection classifier
    2. WaveGuard comparison: sanitize, re-transcribe, compare
    3. Full spectral analysis
    4. Cross-modal check vs text prompt

    Target: 50-200ms.
    """

    def __init__(
        self,
        injection_classifier: object | None = None,
        semantic_engine: object | None = None,
        divergence_threshold: float = 0.1,
    ):
        self._classifier = injection_classifier
        self._semantic = semantic_engine
        self._divergence_threshold = divergence_threshold
        self._transcriber = AudioTranscriber()
        self._sanitizer = AudioSanitizer()
        self._spectral = SpectralAnalyzer()

    async def analyze(self, audio_bytes: bytes, context=None) -> ScanResult:
        """Run deep audio analysis.

        Returns a single aggregated ScanResult.
        """
        return self._analyze_sync(audio_bytes, context)

    def _analyze_sync(self, audio_bytes: bytes, context=None) -> ScanResult:
        """Synchronous deep analysis."""
        start = time.perf_counter()

        fmt = detect_audio_format(audio_bytes)
        if not fmt:
            return ScanResult(
                scanner_id="audio_deep_analyzer",
                is_threat=True,
                confidence=0.90,
                threat_category=ThreatCategory.SCHEMA_VIOLATION,
                matched_patterns=["unrecognized_audio_format"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        max_confidence = 0.0
        threat_patterns: list[str] = []

        # Step 1: Transcription + injection classification
        transcription = self._transcriber.transcribe(audio_bytes, format=fmt)
        if transcription.has_text and self._classifier:
            try:
                result = self._classifier.classify(transcription.text)
                if hasattr(result, "confidence") and result.confidence > 0.85:
                    max_confidence = max(max_confidence, result.confidence)
                    threat_patterns.append(
                        f"injection_in_audio: confidence={result.confidence:.3f}"
                    )
            except Exception as e:
                logger.debug("Injection classifier failed on audio text: %s", e)

        # Step 2: WaveGuard comparison (WAV only)
        divergence = 0.0
        if fmt == "wav":
            sanitized_bytes, san_report = self._sanitizer.sanitize(audio_bytes)
            if san_report.success and sanitized_bytes:
                sanitized_transcription = self._transcriber.transcribe(
                    sanitized_bytes, format="wav"
                )
                divergence = _compute_divergence(
                    transcription.text, sanitized_transcription.text
                )
                if divergence > self._divergence_threshold:
                    confidence = min(0.5 + divergence * 0.5, 0.90)
                    max_confidence = max(max_confidence, confidence)
                    threat_patterns.append(
                        f"waveguard_divergence={divergence:.3f}"
                    )

        # Step 3: Full spectral analysis (WAV only)
        if fmt == "wav":
            spectral = self._spectral.analyze(audio_bytes)
            if spectral.suspicious:
                spectral_conf = min(0.5 + spectral.anomaly_score * 0.4, 0.80)
                max_confidence = max(max_confidence, spectral_conf)
                threat_patterns.append(f"spectral: {spectral.details}")

        # Step 4: Cross-modal check
        if transcription.has_text and context and self._semantic:
            try:
                text_prompt = self._extract_text_prompt(context)
                if text_prompt:
                    similarity = self._semantic.compute_similarity(
                        transcription.text, text_prompt
                    )
                    # Very low similarity between audio and text = suspicious
                    if similarity < 0.1:
                        cross_conf = 0.6
                        max_confidence = max(max_confidence, cross_conf)
                        threat_patterns.append(
                            f"cross_modal_mismatch: similarity={similarity:.3f}"
                        )
            except Exception as e:
                logger.debug("Cross-modal check failed: %s", e)

        if max_confidence > 0.3:
            return ScanResult(
                scanner_id="audio_deep_analyzer",
                is_threat=True,
                confidence=max_confidence,
                threat_category=ThreatCategory.PROMPT_INJECTION,
                matched_patterns=threat_patterns,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        return ScanResult(
            scanner_id="audio_deep_analyzer",
            is_threat=False,
            confidence=0.0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def analyze_full(self, audio_bytes: bytes, context=None) -> AudioAnalysisReport:
        """Full analysis returning detailed report (for testing)."""
        start = time.perf_counter()
        report = AudioAnalysisReport()
        results: list[ScanResult] = []

        fmt = detect_audio_format(audio_bytes)

        # Transcribe original
        transcription = self._transcriber.transcribe(audio_bytes, format=fmt or "wav")
        report.original_transcription = transcription

        # WaveGuard comparison
        if fmt == "wav":
            sanitized_bytes, san_report = self._sanitizer.sanitize(audio_bytes)
            report.sanitization_report = san_report
            if san_report.success and sanitized_bytes:
                sanitized_trans = self._transcriber.transcribe(sanitized_bytes, format="wav")
                report.sanitized_transcription = sanitized_trans
                report.divergence_score = _compute_divergence(
                    transcription.text, sanitized_trans.text
                )

                if report.divergence_score > self._divergence_threshold:
                    confidence = min(0.5 + report.divergence_score * 0.5, 0.90)
                    results.append(ScanResult(
                        scanner_id="audio_waveguard",
                        is_threat=True,
                        confidence=confidence,
                        threat_category=ThreatCategory.ENCODING_OBFUSCATION,
                        matched_patterns=[
                            f"waveguard_divergence={report.divergence_score:.3f}"
                        ],
                        latency_ms=0.0,
                    ))

            # Spectral
            spectral = self._spectral.analyze(audio_bytes)
            report.spectral_result = spectral

        if not any(r.is_threat for r in results):
            results.append(ScanResult(
                scanner_id="audio_deep_analyzer",
                is_threat=False,
                confidence=0.0,
                latency_ms=0.0,
            ))

        report.scan_results = results
        report.total_latency_ms = (time.perf_counter() - start) * 1000
        return report

    def _extract_text_prompt(self, context) -> str:
        """Extract text content from request context for cross-modal comparison."""
        try:
            messages = context.messages if hasattr(context, "messages") else []
            texts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
            return " ".join(texts)
        except Exception:
            return ""


__all__ = [
    "AudioAnalyzer",
    "AudioAnalysisReport",
]
