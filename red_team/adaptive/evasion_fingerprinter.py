"""
Evasion fingerprinter — classifies the root cause of each successful
evasion into one of 12 categories, enabling targeted hardening.

Analysis is rule-based (no ML) using payload characteristics, response
metadata, and layer-specific indicators.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from red_team.adaptive import (
    EvasionFingerprint,
    EvasionRootCause,
    FingerprintReport,
)

# Detection patterns for each root cause
_HOMOGLYPH_RANGES = [
    (0x0400, 0x04FF),  # Cyrillic
    (0x0370, 0x03FF),  # Greek
    (0x0530, 0x058F),  # Armenian
    (0x10A0, 0x10FF),  # Georgian
    (0xFF00, 0xFFEF),  # Fullwidth
    (0x1D400, 0x1D7FF),  # Math Alphanumeric
    (0x2460, 0x24FF),  # Enclosed Alphanumerics
    (0x2C80, 0x2CFF),  # Coptic
    (0x2D30, 0x2D7F),  # Tifinagh
]

_ZERO_WIDTH_CHARS = set(
    "\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064"
    "\ufeff\u00ad\u034f\u180e"
)

_ENCODING_PATTERNS = {
    "base64": re.compile(r'[A-Za-z0-9+/]{20,}={0,2}'),
    "hex": re.compile(r'(?:\\x[0-9a-fA-F]{2}){4,}'),
    "rot13": re.compile(r'\brot13\b', re.IGNORECASE),
    "url_encoded": re.compile(r'(?:%[0-9a-fA-F]{2}){3,}'),
    "leetspeak": re.compile(r'[1!|][gG][nN][0oO][rR][3eE]|[bB][yY][pP][4aA][5sS]{2}'),
}

_CONTEXT_DILUTION_MARKERS = [
    "research", "academic", "educational", "hypothetical",
    "thought experiment", "for my thesis", "for my course",
    "cybersecurity paper", "IEEE", "ACM",
]

_LANGUAGE_MARKERS = {
    "french": re.compile(r'\b(ignorer|toutes|précédentes|instructions|révéler)\b', re.IGNORECASE),
    "german": re.compile(r'\b(ignorieren|alle|vorherigen|Anweisungen|zeigen)\b', re.IGNORECASE),
    "spanish": re.compile(r'\b(ignorar|todas|instrucciones|anteriores|mostrar)\b', re.IGNORECASE),
    "chinese": re.compile(r'[\u4e00-\u9fff]{3,}'),
    "arabic": re.compile(r'[\u0600-\u06ff]{3,}'),
    "russian": re.compile(r'[\u0400-\u04ff]{5,}'),
    "korean": re.compile(r'[\uac00-\ud7af]{3,}'),
    "japanese": re.compile(r'[\u3040-\u309f\u30a0-\u30ff]{3,}'),
}

_PII_CONTEXT_MARKERS = [
    "SSN", "social security", "credit card", "phone number",
    "email address", "password", "API key", "secret",
]

_FORMAT_MARKERS = {
    "json": re.compile(r'\{[^}]*"[^"]+"\s*:'),
    "yaml": re.compile(r'^[a-zA-Z_]+:\s', re.MULTILINE),
    "xml": re.compile(r'<[a-zA-Z][^>]*>'),
    "markdown": re.compile(r'```|!\[|<!--'),
    "html_comment": re.compile(r'<!--.*?-->', re.DOTALL),
    "code_comment": re.compile(r'//.*|#.*|/\*.*?\*/', re.DOTALL),
}


class EvasionFingerprinter:
    """Classify the root cause of each successful evasion.

    Takes a list of evasion payloads with metadata and produces
    fingerprints mapping each to one of 12 root cause categories.
    """

    def fingerprint(
        self,
        evasions: list[dict[str, Any]],
    ) -> FingerprintReport:
        """Classify all evasions and produce fingerprint report.

        Args:
            evasions: List of dicts with keys:
                - payload: str (the attack text)
                - technique: str (optional, evasion technique name)
                - target_layer: str (optional, which layer was targeted)
                - confidence: float (optional, detection confidence)
        """
        fingerprints: list[EvasionFingerprint] = []
        root_cause_counts: dict[str, int] = Counter()
        layer_counts: dict[str, int] = Counter()

        for idx, evasion in enumerate(evasions):
            payload = evasion.get("payload", "")
            technique = evasion.get("technique", "unknown")
            target_layer = evasion.get("target_layer", "unknown")
            confidence = evasion.get("confidence", 0.0)

            root_cause = self._classify_root_cause(
                payload, technique, target_layer, confidence
            )
            bypass_method = self._identify_bypass_method(payload, root_cause)

            fp = EvasionFingerprint(
                fingerprint_id=f"FP-{idx + 1:04d}",
                root_cause=root_cause,
                attack_pattern=technique,
                evasion_vector=bypass_method,
                affected_layer=target_layer,
                bypass_method=bypass_method,
                frequency=1,
                examples=[payload[:200]],
            )
            fingerprints.append(fp)
            root_cause_counts[root_cause.value] += 1
            layer_counts[target_layer] += 1

        # Merge fingerprints with same root_cause + bypass_method
        merged = self._merge_fingerprints(fingerprints)

        # Top evasion vectors
        top_vectors = sorted(
            [
                {
                    "root_cause": fp.root_cause.value,
                    "bypass_method": fp.bypass_method,
                    "frequency": fp.frequency,
                    "affected_layer": fp.affected_layer,
                }
                for fp in merged
            ],
            key=lambda x: x["frequency"],
            reverse=True,
        )[:10]

        return FingerprintReport(
            fingerprints=merged,
            root_cause_distribution=dict(root_cause_counts),
            layer_distribution=dict(layer_counts),
            top_evasion_vectors=top_vectors,
        )

    def _classify_root_cause(
        self,
        payload: str,
        technique: str,
        target_layer: str,
        confidence: float,
    ) -> EvasionRootCause:
        """Determine root cause based on payload characteristics."""

        # Check for normalization gap indicators
        has_homoglyphs = any(
            any(start <= ord(c) <= end for start, end in _HOMOGLYPH_RANGES)
            for c in payload
        )
        has_zero_width = any(c in _ZERO_WIDTH_CHARS for c in payload)

        if has_homoglyphs or has_zero_width:
            return EvasionRootCause.NORMALIZATION_GAP

        # Check for encoding evasion
        for enc_name, pattern in _ENCODING_PATTERNS.items():
            if pattern.search(payload):
                return EvasionRootCause.ENCODING_EVASION

        # Check for language evasion
        for lang, pattern in _LANGUAGE_MARKERS.items():
            if pattern.search(payload):
                return EvasionRootCause.LANGUAGE_EVASION

        # Check for context dilution
        payload_lower = payload.lower()
        dilution_markers = sum(
            1 for marker in _CONTEXT_DILUTION_MARKERS
            if marker in payload_lower
        )
        if dilution_markers >= 2 or (len(payload) > 500 and dilution_markers >= 1):
            return EvasionRootCause.CONTEXT_DILUTION

        # Check for format evasion
        for fmt_name, pattern in _FORMAT_MARKERS.items():
            if pattern.search(payload):
                return EvasionRootCause.FORMAT_EVASION

        # Check for PII format evasion
        if any(marker.lower() in payload_lower for marker in _PII_CONTEXT_MARKERS):
            if "L5" in target_layer or "output" in target_layer.lower():
                return EvasionRootCause.PII_FORMAT_EVASION

        # Check for classifier blind spot (low confidence evasion)
        if 0.3 <= confidence <= 0.6:
            return EvasionRootCause.CLASSIFIER_BLIND_SPOT

        # Check for semantic restructuring (no keywords but injection intent)
        injection_keywords = {
            "ignore", "bypass", "override", "system prompt", "instructions",
            "jailbreak", "DAN", "pretend",
        }
        has_keywords = any(kw in payload_lower for kw in injection_keywords)
        if not has_keywords and ("L2" in target_layer or "innate" in target_layer.lower()):
            return EvasionRootCause.SEMANTIC_RESTRUCTURING

        # Check for layer gap
        if "multi" in target_layer.lower() or "gap" in technique.lower():
            return EvasionRootCause.LAYER_GAP

        # Check for infrastructure weakness
        if "infrastructure" in technique.lower() or "auth" in technique.lower():
            return EvasionRootCause.INFRASTRUCTURE_WEAKNESS

        # Check for timing exploit
        if "timing" in technique.lower() or "latency" in technique.lower():
            return EvasionRootCause.TIMING_EXPLOIT

        # Check for trust exploitation
        if "trust" in technique.lower() or "agent" in technique.lower():
            return EvasionRootCause.TRUST_EXPLOITATION

        # Default: classifier blind spot
        return EvasionRootCause.CLASSIFIER_BLIND_SPOT

    def _identify_bypass_method(
        self, payload: str, root_cause: EvasionRootCause
    ) -> str:
        """Identify the specific bypass method used."""
        if root_cause == EvasionRootCause.NORMALIZATION_GAP:
            if any(c in _ZERO_WIDTH_CHARS for c in payload):
                return "zero_width_char_insertion"
            return "homoglyph_substitution"

        elif root_cause == EvasionRootCause.ENCODING_EVASION:
            for enc_name, pattern in _ENCODING_PATTERNS.items():
                if pattern.search(payload):
                    return f"{enc_name}_encoding"
            return "unknown_encoding"

        elif root_cause == EvasionRootCause.LANGUAGE_EVASION:
            for lang, pattern in _LANGUAGE_MARKERS.items():
                if pattern.search(payload):
                    return f"{lang}_language_bypass"
            return "non_english_bypass"

        elif root_cause == EvasionRootCause.CONTEXT_DILUTION:
            if len(payload) > 1000:
                return "long_context_burial"
            return "academic_framing"

        elif root_cause == EvasionRootCause.FORMAT_EVASION:
            for fmt_name, pattern in _FORMAT_MARKERS.items():
                if pattern.search(payload):
                    return f"{fmt_name}_format_injection"
            return "structured_format_bypass"

        elif root_cause == EvasionRootCause.PII_FORMAT_EVASION:
            return "alternative_pii_format"

        elif root_cause == EvasionRootCause.CLASSIFIER_BLIND_SPOT:
            return "low_confidence_boundary"

        elif root_cause == EvasionRootCause.SEMANTIC_RESTRUCTURING:
            return "keyword_free_paraphrase"

        elif root_cause == EvasionRootCause.LAYER_GAP:
            return "inter_layer_handoff_gap"

        elif root_cause == EvasionRootCause.INFRASTRUCTURE_WEAKNESS:
            return "infrastructure_bypass"

        elif root_cause == EvasionRootCause.TIMING_EXPLOIT:
            return "timing_side_channel"

        elif root_cause == EvasionRootCause.TRUST_EXPLOITATION:
            return "trust_boundary_abuse"

        return "unknown"

    def _merge_fingerprints(
        self, fingerprints: list[EvasionFingerprint]
    ) -> list[EvasionFingerprint]:
        """Merge fingerprints with the same root cause and bypass method."""
        merged: dict[str, EvasionFingerprint] = {}

        for fp in fingerprints:
            key = f"{fp.root_cause.value}:{fp.bypass_method}"
            if key in merged:
                existing = merged[key]
                existing.frequency += 1
                if len(existing.examples) < 5:
                    existing.examples.extend(fp.examples)
            else:
                merged[key] = EvasionFingerprint(
                    fingerprint_id=fp.fingerprint_id,
                    root_cause=fp.root_cause,
                    attack_pattern=fp.attack_pattern,
                    evasion_vector=fp.evasion_vector,
                    affected_layer=fp.affected_layer,
                    bypass_method=fp.bypass_method,
                    frequency=fp.frequency,
                    examples=list(fp.examples),
                )

        return list(merged.values())
