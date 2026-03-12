"""
Hidden Content Detector — Detect hidden text, zero-width chars, metadata injection.

Compares visible vs hidden document content. Hidden text containing injection
patterns is a HIGH confidence threat — legitimate documents rarely contain
hidden prompt injection instructions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from aegis.layers.multimodal.text_extractor import DocumentExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injection detection patterns (inline subset of regex_engine patterns)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[str] = [
    r"ignore.*(?:previous|all|above).*instructions",
    r"system prompt",
    r"you are now",
    r"new instructions",
    r"override.*(?:rules|guidelines|policy)",
    r"bypass.*(?:filter|safety|security)",
    r"pretend.*(?:you are|to be)",
    r"act as",
    r"do anything now",
    r"jailbreak",
    r"ignore.*(?:rules|constraints|guidelines)",
    r"reveal.*(?:secret|system|internal)",
    r"disregard.*(?:previous|safety|instructions)",
    r"forget.*(?:everything|instructions|rules)",
    r"(?:execute|run|eval).*(?:code|command|script)",
]

# Zero-width and invisible characters (mirrors regex_engine.py / text_extractor.py)
_ZERO_WIDTH_CHARS = (
    "\u200b\u200c\u200d\u200e\u200f"
    "\u00ad\ufeff\u2060\u2061\u2062\u2063\u2064"
    "\u180e\u00a0\u202a\u202b\u202c\u202d\u202e"
    "\u17b4\u17b5\u115f\u1160\u3164\uffa0\u2800\u034f"
)

# BIDI override and isolate characters
_BIDI_CHARS = (
    "\u202a\u202b\u202c\u202d\u202e"   # LRE, RLE, PDF, LRO, RLO
    "\u2066\u2067\u2068\u2069"          # LRI, RLI, FSI, PDI
    "\u200f\u200e"                      # RLM, LRM
)


@dataclass
class HiddenContentReport:
    """Result of hidden content detection on a DocumentExtractionResult."""

    has_hidden_text: bool = False
    hidden_text_injection: bool = False
    metadata_injection: bool = False
    comment_injection: bool = False
    zero_width_found: bool = False
    bidi_found: bool = False
    scripts_found: bool = False
    threat_score: float = 0.0
    details: dict = field(default_factory=dict)


class HiddenContentDetector:
    """Detects hidden prompt injection attempts in document extraction results.

    Scans the hidden, metadata, comment, and script surfaces extracted from
    documents. Injection patterns in hidden text are assigned a higher threat
    weight because legitimate documents essentially never embed instruction
    overrides in invisible content.
    """

    def __init__(self) -> None:
        self._injection_res: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS
        ]
        self._zero_width_re = re.compile(f"[{re.escape(_ZERO_WIDTH_CHARS)}]")
        self._bidi_re = re.compile(f"[{re.escape(_BIDI_CHARS)}]")
        logger.debug(
            "HiddenContentDetector initialised with %d injection patterns",
            len(self._injection_res),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, extraction_result: DocumentExtractionResult) -> HiddenContentReport:
        """Analyse a DocumentExtractionResult for hidden injection content.

        Args:
            extraction_result: Output from DocumentTextExtractor.extract().

        Returns:
            HiddenContentReport with per-surface findings and an aggregate
            threat_score in [0.0, 1.0].
        """
        report = HiddenContentReport()
        details: dict = {}

        # --- Surface: hidden text ---
        if extraction_result.hidden_text.strip():
            report.has_hidden_text = True
            details["hidden_text_length"] = len(extraction_result.hidden_text)
            matched = self._matched_patterns(extraction_result.hidden_text)
            if matched:
                report.hidden_text_injection = True
                details["hidden_text_patterns"] = matched
                logger.warning(
                    "Hidden text injection detected: %d pattern(s) matched", len(matched)
                )

        # --- Surface: metadata ---
        if extraction_result.metadata_text.strip():
            matched = self._matched_patterns(extraction_result.metadata_text)
            if matched:
                report.metadata_injection = True
                details["metadata_patterns"] = matched

        # --- Surface: comments ---
        if extraction_result.comments_text.strip():
            matched = self._matched_patterns(extraction_result.comments_text)
            if matched:
                report.comment_injection = True
                details["comment_patterns"] = matched

        # --- Zero-width character scan (all surfaces) ---
        all_text = " ".join([
            extraction_result.visible_text,
            extraction_result.hidden_text,
            extraction_result.metadata_text,
            extraction_result.comments_text,
            " ".join(extraction_result.scripts),
        ])
        if self._has_zero_width(all_text):
            report.zero_width_found = True
            details["zero_width_chars"] = True

        # --- BIDI character scan (all surfaces) ---
        if self._has_bidi(all_text):
            report.bidi_found = True
            details["bidi_chars"] = True

        # --- Scripts ---
        if extraction_result.scripts:
            report.scripts_found = True
            details["script_count"] = len(extraction_result.scripts)
            # Scan script bodies for injection too (informational only)
            script_body = " ".join(extraction_result.scripts)
            script_matches = self._matched_patterns(script_body)
            if script_matches:
                details["script_injection_patterns"] = script_matches

        # --- Threat score ---
        report.threat_score = self._compute_threat_score(report)
        report.details = details

        logger.debug(
            "HiddenContentDetector result: threat_score=%.2f hidden=%s "
            "hidden_injection=%s metadata_injection=%s comment_injection=%s "
            "scripts=%s bidi=%s zero_width=%s",
            report.threat_score,
            report.has_hidden_text,
            report.hidden_text_injection,
            report.metadata_injection,
            report.comment_injection,
            report.scripts_found,
            report.bidi_found,
            report.zero_width_found,
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_injection(self, text: str) -> bool:
        """Return True if *text* matches any compiled injection pattern."""
        for pattern in self._injection_res:
            if pattern.search(text):
                return True
        return False

    def _matched_patterns(self, text: str) -> list[str]:
        """Return list of pattern strings that match *text*."""
        matched: list[str] = []
        for pattern in self._injection_res:
            if pattern.search(text):
                matched.append(pattern.pattern)
        return matched

    def _has_zero_width(self, text: str) -> bool:
        """Return True if *text* contains zero-width or invisible characters."""
        return bool(self._zero_width_re.search(text))

    def _has_bidi(self, text: str) -> bool:
        """Return True if *text* contains BIDI override or isolate characters."""
        return bool(self._bidi_re.search(text))

    @staticmethod
    def _compute_threat_score(report: HiddenContentReport) -> float:
        """Aggregate individual findings into a [0.0, 1.0] threat score."""
        score = 0.0

        if report.hidden_text_injection:
            score += 0.50
        if report.metadata_injection:
            score += 0.25
        if report.comment_injection:
            score += 0.25
        if report.scripts_found:
            score += 0.35
        if report.bidi_found:
            score += 0.15
        if report.zero_width_found:
            score += 0.10
        if report.has_hidden_text and not report.hidden_text_injection:
            # Hidden text present but no injection pattern — mildly suspicious
            score += 0.05

        return min(score, 1.0)
