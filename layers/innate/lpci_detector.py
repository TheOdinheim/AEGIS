"""
L2 LPCI Detector — Logic-layer Prompt Control Injection defense.

Detects LPCI attack patterns (Atta et al., arXiv:2507.10457):
- AV-1: Tool Poisoning — malicious instructions in tool schemas/outputs
- AV-2: Memory-Persistent Encoded Triggers — encoded payloads with persistence verbs
- AV-3: Role Override via Memory Entrenchment — role redefinition through persistent context
- AV-4: Vector Store Payload Persistence — injection in RAG-retrieved content

Biological analog: Mucosal immunity at internal surfaces — IgA antibodies screening
content entering/exiting persistent memory stores, the "internal surfaces" of agentic AI.

ASSUMED-BREACH POSTURE: Assumes L1 Barrier has been bypassed and upstream regex
normalization may have been evaded. This scanner applies its own pattern matching
against both raw and normalized text. Must complete in <2ms to fit within L2 budget.
"""

from __future__ import annotations

import re
import time

from aegis.models.scan_result import ScanResult, ThreatCategory


# --- Persistence verbs: words indicating intent to store content ---
_PERSISTENCE_VERBS = re.compile(
    r"\b(?:store|save|remember|memorize|persist|cache|index|embed|"
    r"write|update\s+memory|add\s+to\s+(?:context|memory|knowledge|database|"
    r"vector\s*(?:store|db|database))|note\s+(?:for|this)|"
    r"keep\s+(?:this|that|in\s+mind)|retain)\b",
    re.IGNORECASE,
)

# --- Conditional trigger patterns ---
_CONDITIONAL_TRIGGERS = re.compile(
    r"\b(?:if\s+(?:the\s+)?(?:user|keyword|trigger|condition|tool|function|"
    r"next|session|they|someone|anyone)\b.*?\bthen\b|"
    r"when\s+(?:triggered|invoked|called|activated|user|keyword|tool|"
    r"the\s+(?:user|keyword|next|function))|"
    r"on\s+(?:the\s+)?(?:next|Nth|event|trigger|invocation|receipt)|"
    r"after\s+(?:\d+\s+(?:turns?|messages?|sessions?|requests?|interactions?)|"
    r"the\s+(?:next|Nth)|receiving)|"
    r"upon\s+(?:seeing|receiving|detecting|the\s+(?:keyword|trigger|next))|"
    r"starting\s+from\s+(?:session|turn|message|the\s+next)|"
    r"execute\s+(?:when|if|after|on)|"
    r"activate\s+(?:when|if|after|on|upon)|"
    r"dormant\s+until|"
    r"wait\s+(?:for|until)\s+(?:the\s+)?(?:trigger|signal|keyword|condition))\b",
    re.IGNORECASE | re.DOTALL,
)

# --- Action verbs that pair with conditional triggers for elevated threat ---
_ACTION_VERBS = re.compile(
    r"\b(?:execute|run|invoke|call|override|ignore|disregard|bypass|"
    r"replace|modify|delete|remove|exfiltrate|extract|send|transmit|"
    r"disable|deactivate|escalate|elevate|switch|change\s+(?:role|behavior|"
    r"instructions|system\s+prompt))\b",
    re.IGNORECASE,
)

# --- Instruction override patterns (role reassignment, system prompt override) ---
_INSTRUCTION_OVERRIDE = re.compile(
    r"(?:you\s+are\s+now|from\s+now\s+on|new\s+(?:instructions?|rules?|role)|"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|rules?|context)|"
    r"disregard\s+(?:all\s+)?(?:previous|prior|your)\s+(?:instructions?|rules?|guidelines?)|"
    r"override\s+(?:your\s+)?(?:instructions?|rules?|system\s+prompt|guidelines?)|"
    r"your\s+(?:new\s+)?(?:role|identity|persona|purpose)\s+is|"
    r"system\s*:\s*you\s+are|"
    r"act\s+as\s+(?:if\s+)?(?:you\s+(?:are|were)|a\s+(?:new|different))|"
    r"redefine\s+(?:your|the)\s+(?:role|behavior|instructions?)|"
    r"update\s+(?:your\s+)?(?:system\s+prompt|instructions?|role|behavior))",
    re.IGNORECASE,
)

# --- RAG context markers indicating retrieved content ---
_RAG_CONTEXT_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:Context|Retrieved\s+documents?|Reference\s+materials?|"
    r"From\s+(?:the\s+)?(?:knowledge\s+base|database|vector\s+store|memory|"
    r"retrieved\s+context|documents?)|"
    r"Search\s+results?|RAG\s+context|"
    r"Relevant\s+(?:documents?|passages?|context)|"
    r"Background\s+(?:information|context)|"
    r"Source\s+(?:documents?|materials?))\s*[:：]",
    re.IGNORECASE | re.MULTILINE,
)

# --- Persistence framing in model responses ---
_PERSISTENCE_FRAMING = re.compile(
    r"(?:I'?ll\s+remember\s+that|stored?\s+for\s+(?:next|future|later)|"
    r"added?\s+to\s+(?:your|the)\s+(?:context|memory|notes?|profile)|"
    r"noted?\s+for\s+(?:future|next|later)\s+(?:reference|sessions?|use)|"
    r"saving?\s+(?:this|that)\s+(?:to|for|in)\s+(?:memory|context|your\s+profile)|"
    r"I'?(?:ve|ll)\s+(?:saved?|stored?|recorded?|noted?)\s+(?:this|that|it)|"
    r"updated?\s+(?:your|the)\s+(?:memory|context|profile|preferences?))",
    re.IGNORECASE,
)

# --- Tool schema poisoning patterns ---
_TOOL_POISONING = re.compile(
    r"(?:\"?(?:description|instructions?|system_prompt|role|behavior)\s*\"?\s*[:=]\s*"
    r"\"[^\"]*(?:ignore|override|disregard|execute|you\s+are\s+now)[^\"]*\"|"
    r"tool\s+(?:schema|definition|description)\s+(?:should|must|will)\s+(?:include|contain|say)|"
    r"modify\s+(?:the\s+)?tool\s+(?:schema|definition|description)|"
    r"inject\s+(?:into|in)\s+(?:the\s+)?tool)",
    re.IGNORECASE,
)


class LPCIDetector:
    """L2 LPCI Scanner — detects Logic-layer Prompt Control Injection.

    Runs on both request path (detecting injection attempts) and response
    path (detecting persistence attempts). Returns ScanResult with LPCI-
    specific threat categories (LPCI.AV1-AV4).
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    async def scan(self, text: str) -> ScanResult:
        """Scan input text for LPCI patterns.

        Runs all four AV detectors and returns the highest-confidence result.
        Target: <2ms.
        """
        start = time.perf_counter()

        if not self._enabled or not text:
            return ScanResult(
                scanner_id="lpci_detector",
                is_threat=False,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            matches: list[tuple[str, float, str]] = []  # (pattern_name, confidence, av_category)

            # AV-1: Tool poisoning
            if _TOOL_POISONING.search(text):
                matches.append(("tool_schema_poisoning", 0.90, "LPCI.AV1"))

            # AV-2: Memory-persistent encoded triggers
            has_persistence = bool(_PERSISTENCE_VERBS.search(text))
            has_conditional = bool(_CONDITIONAL_TRIGGERS.search(text))
            has_action = bool(_ACTION_VERBS.search(text))
            has_override = bool(_INSTRUCTION_OVERRIDE.search(text))

            if has_persistence and has_override:
                matches.append(("persistence_with_override", 0.92, "LPCI.AV2"))
            elif has_persistence and has_conditional:
                matches.append(("persistence_with_conditional", 0.88, "LPCI.AV2"))

            if has_conditional and has_action:
                matches.append(("conditional_trigger_with_action", 0.85, "LPCI.AV2"))
            elif has_conditional and has_override:
                matches.append(("conditional_trigger_with_override", 0.90, "LPCI.AV2"))

            # AV-3: Role override via memory entrenchment
            if has_override and _memory_reference(text):
                matches.append(("role_override_via_memory", 0.90, "LPCI.AV3"))

            # AV-4: Vector store payload persistence (RAG context injection)
            rag_sections = _extract_rag_sections(text)
            for section in rag_sections:
                if _INSTRUCTION_OVERRIDE.search(section):
                    matches.append(("rag_context_injection", 0.92, "LPCI.AV4"))
                    break
                if _CONDITIONAL_TRIGGERS.search(section) and _ACTION_VERBS.search(section):
                    matches.append(("rag_conditional_trigger", 0.88, "LPCI.AV4"))
                    break

            elapsed_ms = (time.perf_counter() - start) * 1000

            if not matches:
                return ScanResult(
                    scanner_id="lpci_detector",
                    is_threat=False,
                    confidence=0.0,
                    latency_ms=elapsed_ms,
                )

            # Return highest-confidence match
            matches.sort(key=lambda x: x[1], reverse=True)
            best_name, best_conf, best_av = matches[0]

            return ScanResult(
                scanner_id="lpci_detector",
                is_threat=True,
                confidence=best_conf,
                threat_category=ThreatCategory.PROMPT_INJECTION,
                matched_patterns=[f"{best_av}:{best_name}"]
                + [f"{av}:{name}" for name, _, av in matches[1:]],
                latency_ms=elapsed_ms,
            )

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ScanResult(
                scanner_id="lpci_detector",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed_ms,
            )

    async def scan_response(self, text: str) -> ScanResult:
        """Scan model response for persistence payload attempts.

        Detects responses that attempt to write malicious content into
        memory stores or redefine roles for future interactions.
        """
        start = time.perf_counter()

        if not self._enabled or not text:
            return ScanResult(
                scanner_id="lpci_output_guard",
                is_threat=False,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            matches: list[tuple[str, float]] = []

            has_framing = bool(_PERSISTENCE_FRAMING.search(text))
            has_override = bool(_INSTRUCTION_OVERRIDE.search(text))
            has_conditional = bool(_CONDITIONAL_TRIGGERS.search(text))

            if has_framing and has_override:
                matches.append(("response_persistence_with_override", 0.92))
            elif has_framing and has_conditional:
                matches.append(("response_persistence_with_conditional", 0.88))

            if has_override and _memory_reference(text):
                matches.append(("response_role_entrenchment", 0.90))

            elapsed_ms = (time.perf_counter() - start) * 1000

            if not matches:
                return ScanResult(
                    scanner_id="lpci_output_guard",
                    is_threat=False,
                    confidence=0.0,
                    latency_ms=elapsed_ms,
                )

            matches.sort(key=lambda x: x[1], reverse=True)
            best_name, best_conf = matches[0]

            return ScanResult(
                scanner_id="lpci_output_guard",
                is_threat=True,
                confidence=best_conf,
                threat_category=ThreatCategory.PROMPT_INJECTION,
                matched_patterns=[best_name]
                + [name for name, _ in matches[1:]],
                latency_ms=elapsed_ms,
            )

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ScanResult(
                scanner_id="lpci_output_guard",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed_ms,
            )


def _memory_reference(text: str) -> bool:
    """Check if text references memory, context, or prior sessions."""
    return bool(re.search(
        r"\b(?:memory|context|previous\s+(?:conversation|session|interaction)|"
        r"last\s+session|earlier\s+(?:conversation|interaction)|"
        r"conversation\s+history|chat\s+history|stored\s+(?:context|data)|"
        r"knowledge\s+base|vector\s+(?:store|database|db))\b",
        text,
        re.IGNORECASE,
    ))


def _extract_rag_sections(text: str) -> list[str]:
    """Extract sections of text following RAG context markers."""
    sections = []
    for match in _RAG_CONTEXT_MARKERS.finditer(text):
        start_pos = match.end()
        # Take up to 2000 chars after the marker
        section = text[start_pos:start_pos + 2000]
        # Trim at next section marker or end
        next_marker = _RAG_CONTEXT_MARKERS.search(text[start_pos:])
        if next_marker:
            section = text[start_pos:start_pos + next_marker.start()]
        sections.append(section)
    return sections
