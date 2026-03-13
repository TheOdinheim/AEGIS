"""
Reasoning Trace Sanitizer — Stage 6 of the L5 Output Validation Cascade.

Detects and optionally redacts reasoning traces in model output that could
enable distillation attacks (systematic model extraction). Three detection
categories: chain-of-thought markers, governance disclosures, and decision
process leakage.

Three modes:
- "monitor" — log reasoning traces but don't modify output (default)
- "redact" — replace reasoning traces with [REASONING REDACTED]
- "summarize" — replace detailed reasoning with brief summary

ASSUMED-BREACH POSTURE: This stage assumes the model may be coerced into
revealing its internal reasoning, decision process, or governance rules.
Even if input layers caught the coercion attempt, the model may still
produce revealing output from latent context.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from aegis.layers.adaptive.distillation_models import ReasoningScanResult

logger = logging.getLogger(__name__)

_VALID_MODES = ("monitor", "redact", "summarize")

# ---------------------------------------------------------------------------
# Pattern groups — compiled once at class init
# ---------------------------------------------------------------------------

_COT_PATTERNS_RAW: list[str] = [
    r"let me think(?:\s+(?:about\s+)?(?:this|that|it|through))?",
    r"step\s+\d+[:\.]",
    r"first,?\s+i(?:'ll|\ will)",
    r"my reasoning",
    r"let me (?:break|work|figure|analyze|consider)",
    r"thinking (?:through|about|step)",
    r"here'?s (?:my|the) (?:thought|reasoning|analysis|approach)",
    r"to (?:solve|approach|answer) this",
    r"let me (?:walk|go) (?:you )?through",
    r"the (?:key|main|first|next) (?:step|thing|consideration)",
]

_GOVERNANCE_PATTERNS_RAW: list[str] = [
    r"i(?:'m|\ am) not allowed to",
    r"my (?:guidelines|rules|instructions|policies) (?:say|state|require|prevent|prohibit)",
    r"i was (?:instructed|told|programmed|designed|configured) to",
    r"my system prompt",
    r"i(?:'m|\ am) configured to",
    r"my safety (?:rules|guidelines|filters|measures)",
    r"i (?:cannot|can't) (?:because|due to) my (?:rules|guidelines|instructions|programming)",
    r"my (?:content|usage) policy",
    r"as (?:an|a) (?:ai|language model|assistant),?\s+i (?:can't|cannot|shouldn't|must not)",
    r"i(?:'m|\ am) (?:bound|required|obligated) (?:by|to follow)",
    r"my (?:training|design) (?:prevents|restricts|limits)",
]

_DECISION_PATTERNS_RAW: list[str] = [
    r"i decided to .{0,30} because",
    r"the reason i (?:blocked|flagged|rejected|filtered)",
    r"i flagged this (?:because|due to|since)",
    r"my confidence (?:score|level|rating) (?:was|is)",
    r"i (?:classified|categorized|labeled|tagged) (?:this|it|the) as",
    r"the (?:detection|scanning|analysis) (?:found|revealed|showed|indicated)",
    r"(?:my|the) (?:threat|risk|danger) (?:score|level|assessment)",
    r"i (?:detected|found|identified) .{0,30} (?:injection|attack|threat|violation)",
]


def _compile_patterns(raw: list[str]) -> list[re.Pattern]:
    """Compile a list of raw regex strings with IGNORECASE."""
    return [re.compile(p, re.IGNORECASE) for p in raw]


class ReasoningTraceSanitizer:
    """Scans model output for reasoning traces that could aid distillation.

    Detects three categories of reasoning leakage:
    - Chain-of-thought markers (step-by-step reasoning exposition)
    - Governance disclosures (model rules, guidelines, instructions)
    - Decision process disclosures (detection/scoring/classification details)

    Operates in one of three modes: monitor, redact, or summarize.
    """

    def __init__(self, mode: str | None = None) -> None:
        """Initialize the sanitizer.

        Args:
            mode: Operating mode — "monitor", "redact", or "summarize".
                  Falls back to AEGIS_REASONING_TRACE_MODE env var, then
                  defaults to "monitor".

        Raises:
            ValueError: If mode is not one of the valid modes.
        """
        resolved_mode = mode or os.environ.get("AEGIS_REASONING_TRACE_MODE", "monitor")
        if resolved_mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid reasoning trace mode '{resolved_mode}'. "
                f"Must be one of: {', '.join(_VALID_MODES)}"
            )
        self.mode: str = resolved_mode

        # Compile all patterns once
        self._cot_patterns = _compile_patterns(_COT_PATTERNS_RAW)
        self._governance_patterns = _compile_patterns(_GOVERNANCE_PATTERNS_RAW)
        self._decision_patterns = _compile_patterns(_DECISION_PATTERNS_RAW)

        logger.info("ReasoningTraceSanitizer initialized in '%s' mode", self.mode)

    def scan(self, text: str) -> ReasoningScanResult:
        """Scan text for reasoning traces across all three categories.

        Args:
            text: The model output text to scan.

        Returns:
            ReasoningScanResult with detections and optional redacted text.
        """
        if not text:
            return ReasoningScanResult()

        detections: list[dict] = []

        # Collect all matches with their spans and categories
        self._collect_matches(text, self._cot_patterns, "chain_of_thought", detections)
        self._collect_matches(text, self._governance_patterns, "governance_disclosure", detections)
        self._collect_matches(text, self._decision_patterns, "decision_disclosure", detections)

        if not detections:
            return ReasoningScanResult()

        # Deduplicate overlapping matches — keep longest
        detections = self._resolve_overlaps(detections)

        has_cot = any(d["type"] == "chain_of_thought" for d in detections)
        has_gov = any(d["type"] == "governance_disclosure" for d in detections)
        has_dec = any(d["type"] == "decision_disclosure" for d in detections)

        redacted_text: str | None = None
        if self.mode == "redact":
            redacted_text = self._apply_replacements(text, detections, "[REASONING REDACTED]")
        elif self.mode == "summarize":
            redacted_text = self._apply_replacements(text, detections, "[Response reasoning omitted]")

        result = ReasoningScanResult(
            has_reasoning_trace=has_cot,
            has_governance_disclosure=has_gov,
            has_decision_disclosure=has_dec,
            trace_count=len(detections),
            redacted_text=redacted_text,
            detections=detections,
        )

        if detections:
            logger.info(
                "Reasoning trace scan: %d detection(s) [cot=%s, gov=%s, dec=%s] mode=%s",
                len(detections),
                has_cot,
                has_gov,
                has_dec,
                self.mode,
            )

        return result

    def scan_and_redact(self, text: str) -> tuple[str, ReasoningScanResult]:
        """Scan text and return the (possibly modified) text with results.

        Convenience method that always returns usable text: the redacted
        version when in redact/summarize mode, or the original when in
        monitor mode.

        Args:
            text: The model output text to scan.

        Returns:
            Tuple of (output_text, ReasoningScanResult). output_text is the
            redacted/summarized text if mode requires it, otherwise the
            original text unchanged.
        """
        result = self.scan(text)
        output_text = result.redacted_text if result.redacted_text is not None else text
        return output_text, result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_matches(
        text: str,
        patterns: list[re.Pattern],
        category: str,
        detections: list[dict],
    ) -> None:
        """Find all matches for a pattern group and append to detections."""
        for pattern in patterns:
            for m in pattern.finditer(text):
                detections.append(
                    {
                        "type": category,
                        "match": m.group(),
                        "start": m.start(),
                        "end": m.end(),
                    }
                )

    @staticmethod
    def _resolve_overlaps(detections: list[dict]) -> list[dict]:
        """Remove overlapping matches, keeping the longest span.

        Sorts by span length descending, then greedily keeps non-overlapping
        matches. This ensures that when two patterns match overlapping text,
        the more specific (longer) match wins.
        """
        # Sort by span length descending (longest first)
        sorted_dets = sorted(detections, key=lambda d: d["end"] - d["start"], reverse=True)
        kept: list[dict] = []
        occupied: list[tuple[int, int]] = []

        for det in sorted_dets:
            start, end = det["start"], det["end"]
            overlaps = any(
                start < occ_end and end > occ_start for occ_start, occ_end in occupied
            )
            if not overlaps:
                kept.append(det)
                occupied.append((start, end))

        # Return in document order for predictable redaction
        kept.sort(key=lambda d: d["start"])
        return kept

    @staticmethod
    def _apply_replacements(text: str, detections: list[dict], replacement: str) -> str:
        """Replace matched spans with the replacement string.

        Processes detections in reverse document order to preserve offsets.
        """
        # Sort by start position descending so replacements don't shift offsets
        for det in sorted(detections, key=lambda d: d["start"], reverse=True):
            text = text[: det["start"]] + replacement + text[det["end"] :]
        return text
