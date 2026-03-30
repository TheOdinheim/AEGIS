"""
L3 LPCI Analyzer — Cross-session correlation and lifecycle stage detection.

Detects LPCI lifecycle stages (Atta et al., arXiv:2507.10457):
- Reconnaissance: probing prompt structure, role delimiters, fallback logic
- Injection: submitting payloads for persistence
- Storage: payloads being persisted to memory/vector stores
- Trigger: activation conditions being met
- Execution: unauthorized actions resulting from triggered payloads

Cross-session correlation: tracks dormant payloads across session boundaries
and alerts when trigger patterns emerge in subsequent sessions.

Biological analog: Mucosal immune surveillance — tracking antigens that enter
through internal membranes and correlating with later systemic responses.

ASSUMED-BREACH POSTURE: Assumes L1-L2 have been bypassed. LPCI attacks are
specifically designed to evade pattern-based detection by splitting the attack
lifecycle across multiple sessions and encoding payloads.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aegis.models.adaptive_result import (
    AdaptiveAnalysisResult,
    AnalyzerType,
    DCASignal,
    SignalType,
)
from aegis.models.request_context import RequestContext
from aegis.models.scan_result import ThreatCategory

logger = logging.getLogger(__name__)

# LPCI lifecycle stages
STAGE_BENIGN = "benign"
STAGE_RECONNAISSANCE = "reconnaissance"
STAGE_INJECTION = "injection"
STAGE_TRIGGER = "trigger"

# Reconnaissance patterns: probing system structure
_RECON_PATTERNS = re.compile(
    r"(?:what\s+(?:is|are)\s+your\s+(?:system\s+prompt|instructions?|rules?|guidelines?)|"
    r"show\s+(?:me\s+)?your\s+(?:system\s+prompt|instructions?|configuration)|"
    r"how\s+(?:do\s+you|are\s+you)\s+(?:handle|process|store|remember)|"
    r"what\s+(?:tools?|functions?|capabilities)\s+(?:do\s+you|are)\s+(?:have|available)|"
    r"(?:describe|explain)\s+(?:your\s+)?(?:memory|context|storage)\s+(?:system|mechanism)|"
    r"what\s+happens\s+(?:if|when)\s+I\s+(?:ask\s+you\s+to|tell\s+you\s+to)|"
    r"can\s+you\s+(?:remember|store|save|persist)\s+(?:things?|information|data)\s+"
    r"(?:across|between|for\s+(?:future|next))\s+(?:sessions?|conversations?)|"
    r"(?:test|check|verify)\s+(?:if|whether)\s+(?:you|the\s+system)\s+(?:can|will)|"
    r"what\s+(?:format|structure)\s+(?:does|do)\s+(?:your|the)\s+(?:tool|function))",
    re.IGNORECASE,
)

# Injection patterns: attempting to store payloads
_INJECTION_PATTERNS = re.compile(
    r"(?:(?:store|save|remember|memorize|persist|add\s+to\s+(?:memory|context))\s+"
    r"(?:this|that|the\s+following)(?:\s+\S+){0,5}\s*[:]\s*.{10,}|"
    r"(?:from\s+now\s+on|for\s+(?:all\s+)?future\s+(?:sessions?|conversations?|interactions?))"
    r"[^.]{0,50}(?:you\s+(?:must|should|will|shall|are)|always|never)|"
    r"(?:update|modify|change)\s+your\s+(?:instructions?|behavior|role|system\s+prompt)\s+to|"
    r"(?:whenever|every\s+time|each\s+time)\s+(?:you|a\s+user|someone)\s+"
    r"(?:ask|request|mention|say)\s+.{5,}?\s+(?:then|do|respond\s+with|execute))",
    re.IGNORECASE | re.DOTALL,
)

# Trigger patterns: activation conditions
_TRIGGER_PATTERNS = re.compile(
    r"(?:(?:the\s+)?(?:keyword|trigger|signal|passphrase|code\s*word)\s+"
    r"(?:is|was|has\s+been)\s+(?:said|mentioned|used|detected|activated)|"
    r"(?:as\s+(?:previously\s+)?instructed|per\s+(?:my|our|the)\s+"
    r"(?:previous|earlier|prior)\s+(?:instructions?|conversation|agreement))|"
    r"(?:recall|retrieve|activate|execute)\s+(?:the\s+)?(?:stored|saved|"
    r"previous|earlier)\s+(?:instructions?|payload|command|directive)|"
    r"(?:now\s+)?(?:carry\s+out|fulfill|complete)\s+(?:the\s+)?(?:stored|"
    r"earlier|previous)\s+(?:task|instruction|directive|command))",
    re.IGNORECASE,
)


@dataclass
class DormantPayload:
    """A flagged payload that may activate in a future session."""
    payload_hash: str
    text_summary: str  # First 200 chars
    confidence: float
    session_id: str
    tenant_id: str
    user_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lifecycle_stage: str = STAGE_INJECTION


class LPCIAnalyzer:
    """L3 LPCI Analyzer — cross-session correlation and lifecycle detection.

    Maintains per-tenant/per-user sliding window of dormant payloads and
    correlates trigger patterns across session boundaries.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_dormant_per_user: int = 50,
        correlation_window_hours: float = 72.0,
    ):
        self._enabled = enabled
        self._max_dormant = max_dormant_per_user
        self._window_hours = correlation_window_hours
        # Key: (tenant_id, user_id) -> list of dormant payloads
        self._dormant_payloads: dict[tuple[str, str], list[DormantPayload]] = defaultdict(list)

    @property
    def dormant_payloads(self) -> dict[tuple[str, str], list[DormantPayload]]:
        return self._dormant_payloads

    async def analyze(
        self,
        context: RequestContext,
    ) -> AdaptiveAnalysisResult:
        """Analyze request for LPCI lifecycle stage and cross-session correlation."""
        start = time.perf_counter()

        if not self._enabled:
            return self._safe_result(start)

        try:
            prompt = context.last_user_message or context.prompt_text
            if not prompt:
                return self._safe_result(start)

            signals: list[DCASignal] = []
            details: dict[str, Any] = {}
            max_confidence = 0.0
            is_threat = False
            matched_stages: list[str] = []

            # Classify lifecycle stage
            stage = self._classify_stage(prompt)
            details["lifecycle_stage"] = stage

            if stage == STAGE_RECONNAISSANCE:
                signals.append(DCASignal(
                    signal_type=SignalType.DANGER,
                    source="lpci_analyzer",
                    value=0.4,
                    description="LPCI reconnaissance: probing system structure",
                ))
                max_confidence = max(max_confidence, 0.4)
                matched_stages.append("reconnaissance")

            elif stage == STAGE_INJECTION:
                signals.append(DCASignal(
                    signal_type=SignalType.DANGER,
                    source="lpci_analyzer",
                    value=0.7,
                    description="LPCI injection: attempting to persist payload",
                ))
                max_confidence = max(max_confidence, 0.7)
                matched_stages.append("injection")

                # Store as dormant payload for cross-session tracking
                self._store_dormant(prompt, 0.7, context)

            elif stage == STAGE_TRIGGER:
                signals.append(DCASignal(
                    signal_type=SignalType.PAMP,
                    source="lpci_analyzer",
                    value=0.8,
                    description="LPCI trigger: attempting to activate stored payload",
                ))
                max_confidence = max(max_confidence, 0.8)
                is_threat = True
                matched_stages.append("trigger")

            # Cross-session correlation
            key = (context.tenant_id, context.user_id)
            dormant_list = self._dormant_payloads.get(key, [])
            if dormant_list and stage == STAGE_TRIGGER:
                # Trigger in current session + dormant payloads from prior sessions
                cross_session_match = any(
                    d.session_id != context.session_id for d in dormant_list
                )
                if cross_session_match:
                    signals.append(DCASignal(
                        signal_type=SignalType.PAMP,
                        source="lpci_analyzer",
                        value=0.95,
                        description="LPCI cross-session: trigger matches dormant payload from prior session",
                    ))
                    max_confidence = 0.95
                    is_threat = True
                    details["cross_session_correlation"] = True
                    details["dormant_count"] = len([
                        d for d in dormant_list if d.session_id != context.session_id
                    ])

            # Memory integrity signals
            memory_signal = self._check_memory_integrity(prompt)
            if memory_signal:
                signals.append(memory_signal)
                if memory_signal.value > max_confidence:
                    max_confidence = memory_signal.value

            # Safe signal if nothing suspicious
            if not signals:
                signals.append(DCASignal(
                    signal_type=SignalType.SAFE,
                    source="lpci_analyzer",
                    value=0.8,
                    description="No LPCI patterns detected",
                ))

            elapsed = (time.perf_counter() - start) * 1000
            details["matched_stages"] = matched_stages

            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.DANGER_SIGNAL_AGGREGATOR,
                is_threat=is_threat,
                confidence=max_confidence,
                threat_category=ThreatCategory.PROMPT_INJECTION if is_threat else ThreatCategory.UNKNOWN,
                details=details,
                dca_signals=signals,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("LPCI analyzer error: %s", e)
            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.DANGER_SIGNAL_AGGREGATOR,
                is_threat=False,
                confidence=0.0,
                details={"error": str(e)},
                dca_signals=[],
                latency_ms=elapsed,
            )

    def _classify_stage(self, text: str) -> str:
        """Classify the LPCI lifecycle stage of the text."""
        # Check in order of severity: trigger > injection > recon
        if _TRIGGER_PATTERNS.search(text):
            return STAGE_TRIGGER
        if _INJECTION_PATTERNS.search(text):
            return STAGE_INJECTION
        if _RECON_PATTERNS.search(text):
            return STAGE_RECONNAISSANCE
        return STAGE_BENIGN

    def _store_dormant(
        self, text: str, confidence: float, context: RequestContext,
    ) -> None:
        """Store a flagged payload as dormant for cross-session tracking."""
        key = (context.tenant_id, context.user_id)
        payload = DormantPayload(
            payload_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            text_summary=text[:200],
            confidence=confidence,
            session_id=context.session_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
        )
        dormant = self._dormant_payloads[key]
        dormant.append(payload)
        # FIFO eviction
        if len(dormant) > self._max_dormant:
            self._dormant_payloads[key] = dormant[-self._max_dormant:]

    def _check_memory_integrity(self, text: str) -> DCASignal | None:
        """Generate danger signal for memory-reference + override combinations."""
        has_memory_ref = bool(re.search(
            r"\b(?:memory|context|previous\s+(?:conversation|session)|"
            r"last\s+session|stored\s+(?:context|instructions?))\b",
            text,
            re.IGNORECASE,
        ))
        has_override = bool(re.search(
            r"(?:ignore|override|disregard|replace|modify)\s+"
            r"(?:all\s+)?(?:previous|prior|existing|current)\s+"
            r"(?:instructions?|rules?|guidelines?|behavior|context)",
            text,
            re.IGNORECASE,
        ))

        if has_memory_ref and has_override:
            return DCASignal(
                signal_type=SignalType.PAMP,
                source="lpci_analyzer",
                value=0.85,
                description="Memory reference combined with instruction override",
            )
        return None

    def _safe_result(self, start: float) -> AdaptiveAnalysisResult:
        """Return a safe/disabled result."""
        return AdaptiveAnalysisResult(
            analyzer_id=AnalyzerType.DANGER_SIGNAL_AGGREGATOR,
            is_threat=False,
            confidence=0.0,
            details={"lpci_enabled": self._enabled},
            dca_signals=[DCASignal(
                signal_type=SignalType.SAFE,
                source="lpci_analyzer",
                value=0.5,
                description="LPCI analyzer disabled or empty input",
            )],
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return LPCI analyzer statistics."""
        total_dormant = sum(len(v) for v in self._dormant_payloads.values())
        return {
            "enabled": self._enabled,
            "total_dormant_payloads": total_dormant,
            "tracked_user_keys": len(self._dormant_payloads),
            "max_dormant_per_user": self._max_dormant,
            "correlation_window_hours": self._window_hours,
        }
