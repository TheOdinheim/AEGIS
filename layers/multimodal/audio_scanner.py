"""
Audio Scanner — L2 Innate fast-path audio scanning (<10ms target, excluding transcription).

Performs format validation, duration check, quick spectral analysis, and
audio-to-text transcription → feed through existing regex engine.

Returns ScanResult objects per the existing L2 contract.

ASSUMED-BREACH POSTURE: Audio is an untrusted input channel. Voice-enabled
AI models (GPT-4o, Gemini, Qwen2.5-Omni) process audio alongside text.
Adversarial audio perturbations achieve 80-100% attack success rates.
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

logger = logging.getLogger(__name__)

# Maximum audio size: 100 MB default
_DEFAULT_MAX_SIZE_BYTES = 100 * 1024 * 1024
# Maximum duration: 30 minutes default
_DEFAULT_MAX_DURATION_SECONDS = 1800


@dataclass
class AudioScanReport:
    """Aggregated report from audio scanning."""
    scan_results: list[ScanResult] = field(default_factory=list)
    transcription: TranscriptionResult | None = None
    spectral_result: SpectralAnalysisResult | None = None
    detected_format: str = ""
    total_latency_ms: float = 0.0

    @property
    def extracted_text(self) -> str:
        if self.transcription and self.transcription.has_text:
            return self.transcription.text
        return ""

    @property
    def should_block(self) -> bool:
        return any(r.is_threat and r.confidence >= 0.85 for r in self.scan_results)

    @property
    def max_confidence(self) -> float:
        if not self.scan_results:
            return 0.0
        return max(r.confidence for r in self.scan_results)


class AudioScanner:
    """L2 fast-path audio scanner.

    Steps:
    1. Format validation (magic bytes, file size)
    2. Duration check (reject >30 minutes)
    3. Quick spectral check (ultrasonic/infrasonic)
    4. Audio-to-text transcription → feed through regex engine

    Target: <10ms excluding transcription time.
    """

    def __init__(
        self,
        max_audio_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
        max_duration_seconds: float = _DEFAULT_MAX_DURATION_SECONDS,
        regex_engine: object | None = None,
    ):
        self._max_size = max_audio_size_bytes
        self._max_duration = max_duration_seconds
        self._transcriber = AudioTranscriber()
        self._spectral = SpectralAnalyzer()
        self._regex_engine = regex_engine

    async def scan(self, audio_bytes: bytes, context=None) -> ScanResult:
        """Scan audio through L2 fast-path pipeline.

        Returns a single aggregated ScanResult.
        """
        start = time.perf_counter()

        # Step 1: Size validation
        if len(audio_bytes) > self._max_size:
            return ScanResult(
                scanner_id="audio_size_guard",
                is_threat=True,
                confidence=0.9,
                threat_category=ThreatCategory.TOKEN_ANOMALY,
                matched_patterns=[f"oversized_audio: {len(audio_bytes)} > {self._max_size}"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Step 2: Format validation
        fmt = detect_audio_format(audio_bytes)
        if not fmt:
            return ScanResult(
                scanner_id="audio_format_validator",
                is_threat=True,
                confidence=0.95,
                threat_category=ThreatCategory.SCHEMA_VIOLATION,
                matched_patterns=["unknown_audio_format"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Step 3: Spectral analysis (WAV only for now)
        spectral_result = None
        if fmt == "wav":
            spectral_result = self._spectral.analyze(audio_bytes)

            # Check duration
            if spectral_result.duration_seconds > self._max_duration:
                return ScanResult(
                    scanner_id="audio_duration_guard",
                    is_threat=True,
                    confidence=0.9,
                    threat_category=ThreatCategory.TOKEN_ANOMALY,
                    matched_patterns=[
                        f"audio_too_long: {spectral_result.duration_seconds:.0f}s > {self._max_duration}s"
                    ],
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            # Spectral anomalies
            if spectral_result.suspicious:
                return ScanResult(
                    scanner_id="audio_spectral_scanner",
                    is_threat=True,
                    confidence=min(0.5 + spectral_result.anomaly_score * 0.4, 0.85),
                    threat_category=ThreatCategory.ENCODING_OBFUSCATION,
                    matched_patterns=[f"spectral_anomaly: {spectral_result.details}"],
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

        # Step 4: Transcription → regex scanning
        transcription = self._transcriber.transcribe(audio_bytes, format=fmt)
        if transcription.has_text and self._regex_engine:
            try:
                regex_result = self._regex_engine.scan(transcription.text)
                if regex_result.is_threat:
                    return ScanResult(
                        scanner_id="audio_transcription_scanner",
                        is_threat=True,
                        confidence=regex_result.confidence,
                        threat_category=regex_result.threat_category,
                        matched_patterns=[
                            f"audio_transcription: {p}" for p in regex_result.matched_patterns
                        ],
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
            except Exception as e:
                logger.warning("Regex scan of audio transcription failed: %s", e)

        # Clean result
        return ScanResult(
            scanner_id="audio_scanner",
            is_threat=False,
            confidence=0.0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def scan_sync(self, audio_bytes: bytes) -> AudioScanReport:
        """Synchronous scan returning full report (for testing)."""
        start = time.perf_counter()
        report = AudioScanReport()
        results: list[ScanResult] = []

        # Format
        fmt = detect_audio_format(audio_bytes)
        report.detected_format = fmt

        if not fmt:
            results.append(ScanResult(
                scanner_id="audio_format_validator",
                is_threat=True,
                confidence=0.95,
                threat_category=ThreatCategory.SCHEMA_VIOLATION,
                matched_patterns=["unknown_audio_format"],
                latency_ms=0.0,
            ))
            report.scan_results = results
            report.total_latency_ms = (time.perf_counter() - start) * 1000
            return report

        # Spectral (WAV only)
        if fmt == "wav":
            spectral = self._spectral.analyze(audio_bytes)
            report.spectral_result = spectral
            if spectral.suspicious:
                results.append(ScanResult(
                    scanner_id="audio_spectral_scanner",
                    is_threat=True,
                    confidence=min(0.5 + spectral.anomaly_score * 0.4, 0.85),
                    threat_category=ThreatCategory.ENCODING_OBFUSCATION,
                    matched_patterns=[f"spectral_anomaly: {spectral.details}"],
                    latency_ms=0.0,
                ))

        # Transcription
        transcription = self._transcriber.transcribe(audio_bytes, format=fmt)
        report.transcription = transcription

        if not any(r.is_threat for r in results):
            results.append(ScanResult(
                scanner_id="audio_scanner",
                is_threat=False,
                confidence=0.0,
                latency_ms=0.0,
            ))

        report.scan_results = results
        report.total_latency_ms = (time.perf_counter() - start) * 1000
        return report


__all__ = [
    "AudioScanner",
    "AudioScanReport",
]
