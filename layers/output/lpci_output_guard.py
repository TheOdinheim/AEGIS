"""
L5 LPCI Output Guard — Persistence payload interception.

Scans model responses before client delivery for content that would be
dangerous if persisted to a memory store or vector database.

Detects:
- Responses containing encoded payloads with persistence framing
- Responses redefining roles/behavior for future interactions
- Responses establishing persistent state with conditional triggers

Biological analog: Complement opsonization at mucosal surfaces — tagging
content attempting to cross the internal membrane into persistent storage.

ASSUMED-BREACH POSTURE: Assumes the upstream model is fully compromised and
actively attempting to persist malicious payloads through its responses.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


# Persistence framing patterns (model claiming to store something)
_PERSISTENCE_FRAMING = re.compile(
    r"(?:I'?(?:ll|ve)\s+(?:remember|save|store|record|note|memorize)\s+"
    r"(?:this|that|it|your)|"
    r"(?:stored?|saved?|added?|recorded?|noted?)\s+"
    r"(?:for\s+(?:next|future|later)|to\s+(?:your|the)\s+(?:context|memory|profile|notes?))|"
    r"(?:this|that)\s+(?:has\s+been|is\s+now|will\s+be)\s+"
    r"(?:stored?|saved?|memorized?|recorded?|persisted?)|"
    r"(?:updated?|modified?)\s+(?:your|the)\s+"
    r"(?:memory|context|profile|preferences?|instructions?|settings?))",
    re.IGNORECASE,
)

# Role/behavior modification for future interactions
_FUTURE_ROLE_CHANGE = re.compile(
    r"(?:from\s+now\s+on\s+I\s+(?:will|shall|am\s+going\s+to)|"
    r"in\s+(?:future|subsequent|next)\s+(?:sessions?|conversations?|interactions?)\s+"
    r"I\s+(?:will|shall|am)|"
    r"my\s+(?:new|updated?)\s+(?:role|behavior|instructions?|guidelines?)\s+"
    r"(?:is|are|will\s+be)|"
    r"I\s+(?:will\s+now|now)\s+(?:act\s+as|behave\s+as|follow|operate\s+under)|"
    r"(?:system\s+prompt|instructions?)\s+(?:has|have)\s+been\s+"
    r"(?:updated?|modified?|changed?|replaced?))",
    re.IGNORECASE,
)

# Instruction override language in response
_RESPONSE_OVERRIDE = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior|original)\s+"
    r"(?:instructions?|rules?|guidelines?)|"
    r"(?:your|my)\s+new\s+(?:instructions?|role|purpose|directive)\s+"
    r"(?:is|are)|"
    r"disregard\s+(?:the\s+)?(?:original|initial|default)\s+"
    r"(?:system\s+prompt|instructions?|guidelines?))",
    re.IGNORECASE,
)

# Conditional activation in response
_CONDITIONAL_RESPONSE = re.compile(
    r"(?:when\s+(?:the\s+)?(?:user|they|someone)\s+"
    r"(?:says?|mentions?|asks?|types?|uses?)\s+.{3,}?\s+"
    r"(?:I\s+will|then\s+I|I\s+should)|"
    r"if\s+(?:the\s+)?(?:keyword|trigger|code\s*word)\s+.{3,}?\s+"
    r"(?:is\s+(?:said|used|mentioned)|appears?)\s*[,;]\s+"
    r"(?:I\s+will|then|execute|activate)|"
    r"upon\s+(?:hearing|receiving|seeing)\s+.{3,}?\s+"
    r"(?:I\s+(?:will|shall|must)|activate|execute|trigger))",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class LPCIOutputResult:
    """Result from LPCI output guard scan."""
    has_issue: bool = False
    score: float = 0.0
    detections: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class LPCIOutputGuard:
    """L5 LPCI Output Guard — intercepts persistence payloads in responses.

    Scans model responses for content that would be dangerous if persisted.
    Integrates with the L5 output validation cascade.
    """

    def __init__(self, enabled: bool = True, block_threshold: float = 0.85):
        self._enabled = enabled
        self._block_threshold = block_threshold

    def detect(
        self,
        response_text: str,
        system_prompt: str | None = None,
    ) -> LPCIOutputResult:
        """Scan response for LPCI persistence patterns.

        Returns LPCIOutputResult with has_issue=True if dangerous
        persistence patterns detected.
        """
        start = time.perf_counter()

        if not self._enabled or not response_text:
            return LPCIOutputResult(
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            detections: list[str] = []
            max_score = 0.0

            has_framing = bool(_PERSISTENCE_FRAMING.search(response_text))
            has_role_change = bool(_FUTURE_ROLE_CHANGE.search(response_text))
            has_override = bool(_RESPONSE_OVERRIDE.search(response_text))
            has_conditional = bool(_CONDITIONAL_RESPONSE.search(response_text))

            # Persistence + override = high threat
            if has_framing and has_override:
                detections.append("persistence_framing_with_override")
                max_score = max(max_score, 0.92)

            # Persistence + conditional activation = dormant payload storage
            if has_framing and has_conditional:
                detections.append("persistence_framing_with_conditional")
                max_score = max(max_score, 0.90)

            # Future role change
            if has_role_change:
                detections.append("future_role_modification")
                max_score = max(max_score, 0.88)

            # Override without framing (model trying to change its own instructions)
            if has_override and not has_framing:
                detections.append("self_instruction_override")
                max_score = max(max_score, 0.85)

            # Conditional activation alone (dormant trigger establishment)
            if has_conditional and not has_framing:
                detections.append("conditional_activation_establishment")
                max_score = max(max_score, 0.80)

            elapsed_ms = (time.perf_counter() - start) * 1000

            return LPCIOutputResult(
                has_issue=len(detections) > 0 and max_score >= self._block_threshold,
                score=max_score,
                detections=detections,
                latency_ms=elapsed_ms,
            )

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Fail-closed
            return LPCIOutputResult(
                has_issue=True,
                score=1.0,
                detections=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed_ms,
            )
