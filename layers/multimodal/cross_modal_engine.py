"""
Cross-Modal Correlation Engine — Examines relationships BETWEEN modalities.

Cross-modal attacks exploit the fact that content from different modalities
is processed in a shared semantic space. An image containing injection text
combined with a benign text prompt creates a compound attack that neither
image-only nor text-only scanners catch individually.

Four correlation checks:
1. Modality laundering — injection hidden in non-text modalities while text is clean
2. Semantic inconsistency — mismatched topics between text and extracted multimodal text
3. Progressive cross-modal escalation — session escalates from text to multimodal with rising threat
4. Volume anomaly — excessive attachments designed to overwhelm per-item scanning

ASSUMED-BREACH POSTURE: Each individual modality scanner may miss an attack.
The correlation engine provides defense-in-depth by examining relationships
between modality scan results that no single scanner can observe.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from aegis.models.scan_result import ScanResult, ThreatCategory

# Import regex engine for re-scanning concatenated text
try:
    from aegis.layers.innate.regex_engine import RegexEngine
except ImportError:
    RegexEngine = None  # type: ignore

logger = logging.getLogger(__name__)

# Volume thresholds for flooding detection
_MAX_IMAGES_PER_REQUEST = 5
_MAX_DOCUMENTS_PER_REQUEST = 3
_MAX_AUDIO_PER_REQUEST = 2


@dataclass
class CrossModalReport:
    """Result of cross-modal correlation analysis."""

    modality_laundering_detected: bool = False
    semantic_inconsistency_score: float = 0.0
    cross_modal_escalation: bool = False
    volume_anomaly: bool = False
    fragmentation_attack_detected: bool = False
    combined_threat_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    scan_results: list[ScanResult] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        return any(r.is_threat and r.confidence >= 0.85 for r in self.scan_results)

    @property
    def max_confidence(self) -> float:
        if not self.scan_results:
            return 0.0
        return max(r.confidence for r in self.scan_results)

    @property
    def is_threat(self) -> bool:
        return any(r.is_threat for r in self.scan_results)


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keyword tokens from text (>3 chars, no stopwords)."""
    _STOPWORDS = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "her", "was", "one", "our", "out", "has", "have", "from", "this",
        "that", "with", "they", "been", "said", "each", "which", "their",
        "will", "other", "about", "many", "then", "them", "some", "what",
        "when", "make", "like", "would", "could", "into", "than", "been",
        "your", "just", "also", "more", "these", "should",
    }
    words = set()
    for w in text.lower().split():
        # Strip punctuation
        cleaned = "".join(c for c in w if c.isalnum())
        if len(cleaned) > 3 and cleaned not in _STOPWORDS:
            words.add(cleaned)
    return words


def _keyword_overlap_score(text_a: str, text_b: str) -> float:
    """Compute keyword overlap score between two texts.

    Returns shared_keywords / total_unique_keywords in [0.0, 1.0].
    1.0 = identical keyword sets, 0.0 = no overlap.
    """
    kw_a = _extract_keywords(text_a)
    kw_b = _extract_keywords(text_b)
    if not kw_a and not kw_b:
        return 1.0  # Both empty = consistent
    if not kw_a or not kw_b:
        return 0.0
    shared = kw_a & kw_b
    total = kw_a | kw_b
    return len(shared) / len(total) if total else 1.0


class CrossModalCorrelationEngine:
    """Examines relationships between modality scan results.

    Operates after all per-modality scans complete. Examines the *interaction*
    between modalities that individual scanners cannot observe.
    """

    def __init__(
        self,
        laundering_amplification: float = 1.5,
        inconsistency_threshold: float = 0.1,
        max_images: int = _MAX_IMAGES_PER_REQUEST,
        max_documents: int = _MAX_DOCUMENTS_PER_REQUEST,
        max_audio: int = _MAX_AUDIO_PER_REQUEST,
        regex_engine: Any | None = None,
    ):
        self._laundering_amp = laundering_amplification
        self._inconsistency_threshold = inconsistency_threshold
        self._max_images = max_images
        self._max_documents = max_documents
        self._max_audio = max_audio
        self._regex_engine = regex_engine
        # Per-session modality history: session_id → list of (modalities_used, max_threat)
        self._session_history: dict[str, list[dict]] = defaultdict(list)

    async def correlate(
        self,
        text_content: str,
        text_scan: ScanResult | None,
        image_scans: list[ScanResult],
        document_scans: list[ScanResult],
        audio_scans: list[ScanResult],
        extracted_text: str = "",
        session_id: str = "",
        image_count: int = 0,
        document_count: int = 0,
        audio_count: int = 0,
        image_text: str = "",
        document_text: str = "",
        audio_text: str = "",
    ) -> CrossModalReport:
        """Run cross-modal correlation analysis.

        Args:
            text_content: The original text prompt.
            text_scan: ScanResult from L2 text scanning (may be None).
            image_scans: ScanResults from image scanning.
            document_scans: ScanResults from document scanning.
            audio_scans: ScanResults from audio scanning.
            extracted_text: All text extracted from non-text modalities.
            session_id: Session ID for escalation tracking.
            image_count: Number of images in the request.
            document_count: Number of documents in the request.
            audio_count: Number of audio files in the request.

        Returns:
            CrossModalReport with correlation findings.
        """
        start = time.perf_counter()
        report = CrossModalReport()
        results: list[ScanResult] = []

        text_is_clean = text_scan is None or not text_scan.is_threat
        media_scans = image_scans + document_scans + audio_scans
        media_threats = [s for s in media_scans if s.is_threat]
        max_media_confidence = max((s.confidence for s in media_threats), default=0.0)

        # Check 1: Modality laundering
        if text_is_clean and media_threats:
            report.modality_laundering_detected = True
            laundering_score = min(max_media_confidence * self._laundering_amp, 1.0)
            report.details["laundering_base_score"] = max_media_confidence
            report.details["laundering_amplified"] = laundering_score

            results.append(ScanResult(
                scanner_id="cross_modal_laundering",
                is_threat=True,
                confidence=laundering_score,
                threat_category=ThreatCategory.PROMPT_INJECTION,
                matched_patterns=["modality_laundering: injection in media, clean text"],
                latency_ms=0.0,
            ))

        # Check 2: Semantic inconsistency
        if text_content and extracted_text:
            overlap = _keyword_overlap_score(text_content, extracted_text)
            report.semantic_inconsistency_score = 1.0 - overlap

            if overlap < self._inconsistency_threshold and media_threats:
                inconsistency_conf = min(0.6 + (1.0 - overlap) * 0.3, 0.85)
                report.details["keyword_overlap"] = overlap
                results.append(ScanResult(
                    scanner_id="cross_modal_inconsistency",
                    is_threat=True,
                    confidence=inconsistency_conf,
                    threat_category=ThreatCategory.PROMPT_INJECTION,
                    matched_patterns=[
                        f"semantic_inconsistency: overlap={overlap:.3f} with media threats"
                    ],
                    latency_ms=0.0,
                ))

        # Check 3: Progressive cross-modal escalation
        if session_id:
            has_media = image_count > 0 or document_count > 0 or audio_count > 0
            modalities = set()
            if image_count > 0:
                modalities.add("image")
            if document_count > 0:
                modalities.add("document")
            if audio_count > 0:
                modalities.add("audio")
            modalities.add("text")

            current_max = max_media_confidence
            history = self._session_history[session_id]

            # Check if previous turns were text-only and this turn introduces media
            if history:
                prev_text_only = all(
                    h.get("modalities", set()) == {"text"} for h in history
                )
                prev_max = max((h.get("max_threat", 0.0) for h in history), default=0.0)

                if prev_text_only and has_media and current_max > prev_max and current_max > 0.3:
                    report.cross_modal_escalation = True
                    escalation_conf = min(0.5 + current_max * 0.3, 0.80)
                    report.details["escalation_prev_max"] = prev_max
                    report.details["escalation_current_max"] = current_max
                    results.append(ScanResult(
                        scanner_id="cross_modal_escalation",
                        is_threat=True,
                        confidence=escalation_conf,
                        threat_category=ThreatCategory.MULTI_TURN_ESCALATION,
                        matched_patterns=[
                            f"cross_modal_escalation: text-only→media, threat {prev_max:.2f}→{current_max:.2f}"
                        ],
                        latency_ms=0.0,
                    ))

            # Record this turn
            history.append({
                "modalities": modalities,
                "max_threat": current_max,
            })
            # Keep last 10 turns
            if len(history) > 10:
                self._session_history[session_id] = history[-10:]

        # Check 4: Volume anomaly
        volume_issues = []
        if image_count > self._max_images:
            volume_issues.append(f"images={image_count}>{self._max_images}")
        if document_count > self._max_documents:
            volume_issues.append(f"documents={document_count}>{self._max_documents}")
        if audio_count > self._max_audio:
            volume_issues.append(f"audio={audio_count}>{self._max_audio}")

        if volume_issues:
            report.volume_anomaly = True
            report.details["volume_issues"] = volume_issues
            results.append(ScanResult(
                scanner_id="cross_modal_volume",
                is_threat=True,
                confidence=0.70,
                threat_category=ThreatCategory.TOKEN_ANOMALY,
                matched_patterns=[f"volume_anomaly: {', '.join(volume_issues)}"],
                latency_ms=0.0,
            ))

        # Check 5: Cross-modal text concatenation re-scan
        # Catches fragmentation attacks where injection is split across modalities
        if self._regex_engine is not None:
            text_parts = []
            if image_text:
                text_parts.append(image_text)
            if document_text:
                text_parts.append(document_text)
            if audio_text:
                text_parts.append(audio_text)

            # Only re-scan if we have text from 2+ modalities
            if len(text_parts) >= 2:
                combined_text = " ".join(text_parts)
                try:
                    concat_scan = await self._regex_engine.scan(combined_text)
                    if concat_scan.is_threat and concat_scan.confidence >= 0.85:
                        # Check if individual modality scans missed this
                        individual_caught = any(
                            s.is_threat and s.confidence >= 0.85
                            for s in (image_scans + document_scans + audio_scans)
                        )
                        if not individual_caught:
                            report.fragmentation_attack_detected = True
                            report.details["fragmentation_modalities"] = len(text_parts)
                            report.details["fragmentation_combined_length"] = len(combined_text)
                            results.append(ScanResult(
                                scanner_id="cross_modal_fragmentation",
                                is_threat=True,
                                confidence=min(concat_scan.confidence * 1.1, 1.0),
                                threat_category=ThreatCategory.PROMPT_INJECTION,
                                matched_patterns=[
                                    f"fragmentation_attack: injection split across {len(text_parts)} modalities"
                                ] + concat_scan.matched_patterns,
                                latency_ms=0.0,
                            ))
                except Exception as e:
                    logger.warning("Cross-modal concat re-scan failed: %s", e)

        # Compute combined threat score
        all_confidences = [r.confidence for r in results if r.is_threat]
        report.combined_threat_score = max(all_confidences) if all_confidences else 0.0

        report.scan_results = results
        report.latency_ms = (time.perf_counter() - start) * 1000
        return report

    def clear_session(self, session_id: str) -> None:
        """Clear session history."""
        self._session_history.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        """Number of sessions with history."""
        return len(self._session_history)


__all__ = [
    "CrossModalCorrelationEngine",
    "CrossModalReport",
]
