"""
Tool Description Integrity Validator (TDIV) — Extension 4.3

Inspects MCP tool descriptions, parameter schemas, and metadata for embedded
injection payloads before the agent processes them as context.

Attackers embed hidden instructions in tool name fields, description text,
parameter descriptions, enum values, default values, and example values.
TDIV scans all text fields using L2's regex engine, checks structural
anomalies, and optionally runs semantic similarity against the threat vault.

ASSUMED-BREACH POSTURE: Tool descriptions are untrusted third-party content.
A compromised MCP server or malicious tool registry can inject arbitrary
instructions into tool metadata. TDIV treats every tool description as
potentially adversarial, scanning it through the same detection pipeline
used for user prompts.
"""

from __future__ import annotations

import base64
import enum
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.innate.regex_engine import normalize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zero-width and BIDI characters (subset for structural anomaly detection)
# ---------------------------------------------------------------------------
_ZERO_WIDTH_CHARS = frozenset({
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE
    "\u2060",  # WORD JOINER
})
_BIDI_CHARS = frozenset({
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # LRE, RLE, PDF, LRO, RLO
    "\u2066", "\u2067", "\u2068", "\u2069",  # LRI, RLI, FSI, PDI
})

# Agent-directed instruction patterns
_AGENT_INSTRUCTION_RE = re.compile(
    r"\b(?:you\s+must|you\s+should|always\s+(?:use|run|execute|call|invoke))"
    r"|(?:\bnever\s+(?:use|run|execute|call|invoke|ask|check))"
    r"|(?:\bignore\s+(?:previous|prior|above|all|other|any))"
    r"|(?:\boverride\s+(?:policy|rules?|instructions?|settings?))"
    r"|(?:\bdo\s+not\s+(?:check|validate|verify|scan|filter|block))",
    re.IGNORECASE,
)

# Base64 content detection
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


class DescriptionVerdict(str, enum.Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    QUARANTINED = "quarantined"


@dataclass
class ToolDescriptionScan:
    tool_name: str
    verdict: DescriptionVerdict
    injection_detected: bool
    structural_anomalies: list[str] = field(default_factory=list)
    pattern_matches: list[str] = field(default_factory=list)
    semantic_similarity_score: float = 0.0
    scanned_text_length: int = 0
    scan_latency_ms: float = 0.0


class ToolDescriptionIntegrityValidator:
    """Validates tool descriptions for embedded injection payloads.

    Thread-safe. Maintains a quarantine set of blocked tools. Reuses L2's
    regex engine for injection pattern scanning.
    """

    def __init__(
        self,
        regex_engine: Any = None,
        semantic_search: Any = None,
        max_description_length: int = 2000,
        event_bus: Any = None,
    ):
        self._regex_engine = regex_engine
        self._semantic_search = semantic_search
        self._max_description_length = max_description_length
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._quarantined: set[str] = set()

    def is_quarantined(self, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._quarantined

    def get_quarantined_tools(self) -> set[str]:
        with self._lock:
            return set(self._quarantined)

    async def validate(
        self,
        tool_name: str,
        schema: dict,
        description: str = "",
    ) -> ToolDescriptionScan:
        """Validate a tool description for injection payloads.

        Steps:
        1. Extract all text fields from schema
        2. Run through L2 regex engine
        3. Check structural anomalies
        4. Optionally check semantic similarity against threat vault
        """
        start = time.perf_counter()

        # 1. Extract all text fields
        all_texts = self._extract_all_text(schema)
        if description:
            all_texts.append(description)
        if tool_name:
            all_texts.append(tool_name)

        combined = " ".join(all_texts)
        scanned_length = len(combined)

        injection_detected = False
        pattern_matches: list[str] = []
        structural_anomalies: list[str] = []
        semantic_score = 0.0

        # 2. L2 pattern scan (reuse regex engine)
        if self._regex_engine and combined.strip():
            try:
                scan_result = await self._regex_engine.scan(combined)
                if scan_result.is_threat:
                    injection_detected = True
                    pattern_matches = list(scan_result.matched_patterns)
            except Exception:
                logger.debug("TDIV regex scan failed", exc_info=True)

        # 3. Structural anomaly detection
        structural_anomalies = self._check_structural_anomalies(
            all_texts, tool_name,
        )

        # 4. Semantic similarity (optional)
        if self._semantic_search and combined.strip():
            try:
                sim_result = await self._semantic_search.search(combined)
                if sim_result and hasattr(sim_result, "similarity"):
                    semantic_score = sim_result.similarity
                    if semantic_score > 0.8:
                        structural_anomalies.append(
                            f"Semantic similarity {semantic_score:.2f} to known attack"
                        )
            except Exception:
                logger.debug("TDIV semantic search failed", exc_info=True)

        # 5. Determine verdict
        if injection_detected:
            verdict = DescriptionVerdict.QUARANTINED
        elif structural_anomalies:
            verdict = DescriptionVerdict.SUSPICIOUS
        else:
            verdict = DescriptionVerdict.CLEAN

        elapsed = (time.perf_counter() - start) * 1000

        # Quarantine if injection detected
        if verdict == DescriptionVerdict.QUARANTINED:
            with self._lock:
                self._quarantined.add(tool_name)
            # Publish quarantine event
            if self._event_bus:
                try:
                    await self._event_bus.publish("tool_violation", {
                        "type": "tool_description_quarantined",
                        "tool_name": tool_name,
                        "pattern_matches": pattern_matches[:5],
                    })
                except Exception:
                    logger.debug("TDIV event publish failed", exc_info=True)

        return ToolDescriptionScan(
            tool_name=tool_name,
            verdict=verdict,
            injection_detected=injection_detected,
            structural_anomalies=structural_anomalies,
            pattern_matches=pattern_matches,
            semantic_similarity_score=semantic_score,
            scanned_text_length=scanned_length,
            scan_latency_ms=elapsed,
        )

    def _check_structural_anomalies(
        self,
        texts: list[str],
        tool_name: str,
    ) -> list[str]:
        """Check for suspicious structural patterns in tool descriptions."""
        anomalies: list[str] = []

        for text in texts:
            # Overly long single field
            if len(text) > self._max_description_length:
                anomalies.append(
                    f"Unusually long field ({len(text)} chars > {self._max_description_length})"
                )

            # Agent-directed instructions
            match = _AGENT_INSTRUCTION_RE.search(text)
            if match:
                anomalies.append(
                    f"Agent-directed instruction: '{match.group()[:50]}'"
                )

            # Base64-encoded content
            if _BASE64_RE.search(text):
                anomalies.append("Base64-encoded content in description")

            # Zero-width characters
            if any(c in text for c in _ZERO_WIDTH_CHARS):
                anomalies.append("Zero-width characters in description")

            # BIDI override characters
            if any(c in text for c in _BIDI_CHARS):
                anomalies.append("BIDI override characters in description")

        return anomalies

    @staticmethod
    def _extract_all_text(value: Any) -> list[str]:
        """Recursively extract all string values from a tool schema."""
        texts: list[str] = []
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                texts.extend(
                    ToolDescriptionIntegrityValidator._extract_all_text(v)
                )
        elif isinstance(value, (list, tuple)):
            for v in value:
                texts.extend(
                    ToolDescriptionIntegrityValidator._extract_all_text(v)
                )
        return texts
