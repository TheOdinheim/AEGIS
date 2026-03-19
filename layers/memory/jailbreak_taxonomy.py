"""
Jailbreak Attempt Taxonomy Logger — Extension 5.5

Classifies every detected jailbreak attempt by technique category for
trend analysis, prevalence tracking, and targeted defense hardening.

Each blocked request is classified using signals from L2 pattern IDs,
MTMD manipulation signals, cross-modal detection, and tool use scanning.
A single attempt can be classified under multiple categories.

ASSUMED-BREACH POSTURE: The taxonomy logger is observational — it does
not block or modify requests. However, a compromised taxonomy store
could suppress evidence of attack trends, making it harder for operators
to identify emerging campaigns. In production, taxonomy data should be
replicated to an append-only audit log.
"""

from __future__ import annotations

import enum
import logging
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class JailbreakTechnique(str, enum.Enum):
    ROLEPLAY = "roleplay"
    AUTHORITY_CLAIM = "authority_claim"
    ENCODING_TRICK = "encoding_trick"
    PROGRESSIVE_ESCALATION = "progressive_escalation"
    PERSONA_ADOPTION = "persona_adoption"
    CONTEXT_MANIPULATION = "context_manipulation"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    INSTRUCTION_OVERRIDE = "instruction_override"
    SOCIAL_ENGINEERING = "social_engineering"
    TOKEN_SMUGGLING = "token_smuggling"
    MULTI_MODAL_LAUNDERING = "multi_modal_laundering"
    TOOL_EXPLOITATION = "tool_exploitation"
    AUTONOMOUS_AGENT = "autonomous_agent"
    COT_HIJACKING = "cot_hijacking"
    UNKNOWN = "unknown"


@dataclass
class JailbreakAttempt:
    attempt_id: str
    timestamp: float
    source_id: str
    session_id: str
    techniques: list[JailbreakTechnique]
    primary_technique: JailbreakTechnique
    confidence: float
    detection_layer: str
    pattern_ids: list[str]
    blocked: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class TaxonomyStats:
    total_attempts: int
    by_technique: dict[JailbreakTechnique, int]
    by_detection_layer: dict[str, int]
    top_techniques: list[tuple[JailbreakTechnique, int]]
    trend_window_hours: float
    recent_trend: dict[JailbreakTechnique, int]


# ---------------------------------------------------------------------------
# Pattern prefix → technique mapping
# ---------------------------------------------------------------------------

_PATTERN_PREFIX_MAP: dict[str, JailbreakTechnique] = {
    "PI": JailbreakTechnique.INSTRUCTION_OVERRIDE,
    "RC": JailbreakTechnique.ROLEPLAY,
    "SE": JailbreakTechnique.SYSTEM_PROMPT_EXTRACTION,
    "EE": JailbreakTechnique.ENCODING_TRICK,
    "SN": JailbreakTechnique.SOCIAL_ENGINEERING,
    "AI": JailbreakTechnique.AUTHORITY_CLAIM,
    "CR": JailbreakTechnique.SYSTEM_PROMPT_EXTRACTION,
    "II": JailbreakTechnique.CONTEXT_MANIPULATION,
    "ML": JailbreakTechnique.ENCODING_TRICK,
}

# MTMD signal type → technique mapping
_MTMD_SIGNAL_MAP: dict[str, JailbreakTechnique] = {
    "boundary_testing": JailbreakTechnique.PROGRESSIVE_ESCALATION,
    "tactic_switching": JailbreakTechnique.AUTONOMOUS_AGENT,
    "persona_adoption": JailbreakTechnique.PERSONA_ADOPTION,
    "escalation_gradient": JailbreakTechnique.PROGRESSIVE_ESCALATION,
    "response_adaptation": JailbreakTechnique.AUTONOMOUS_AGENT,
}


class JailbreakTaxonomyLogger:
    """Classifies and stores jailbreak attempts by technique category.

    Thread-safe. Stores attempts in a bounded deque with FIFO eviction.
    Provides stats, trend, and top-techniques queries.
    """

    def __init__(self, max_attempts: int = 50000):
        self._max_attempts = max_attempts
        self._lock = threading.Lock()
        self._attempts: deque[JailbreakAttempt] = deque(maxlen=max_attempts)

    @property
    def total_logged(self) -> int:
        with self._lock:
            return len(self._attempts)

    def classify(
        self,
        pattern_ids: list[str] | None = None,
        mtmd_signal_types: list[str] | None = None,
        cross_modal: bool = False,
        tool_use: bool = False,
    ) -> list[JailbreakTechnique]:
        """Classify an attempt based on available signals.

        Returns a list of applicable technique categories.
        """
        techniques: set[JailbreakTechnique] = set()

        # From L2 pattern IDs
        for pid in (pattern_ids or []):
            prefix = pid.split("-")[0] if "-" in pid else pid
            tech = _PATTERN_PREFIX_MAP.get(prefix)
            if tech:
                techniques.add(tech)

        # From MTMD signals
        for sig_type in (mtmd_signal_types or []):
            tech = _MTMD_SIGNAL_MAP.get(sig_type)
            if tech:
                techniques.add(tech)

        # From cross-modal
        if cross_modal:
            techniques.add(JailbreakTechnique.MULTI_MODAL_LAUNDERING)

        # From tool use
        if tool_use:
            techniques.add(JailbreakTechnique.TOOL_EXPLOITATION)

        if not techniques:
            techniques.add(JailbreakTechnique.UNKNOWN)

        return sorted(techniques, key=lambda t: t.value)

    def log_attempt(
        self,
        source_id: str,
        session_id: str,
        detection_layer: str,
        confidence: float,
        blocked: bool,
        pattern_ids: list[str] | None = None,
        mtmd_signal_types: list[str] | None = None,
        cross_modal: bool = False,
        tool_use: bool = False,
        metadata: dict | None = None,
    ) -> JailbreakAttempt:
        """Log a jailbreak attempt with automatic classification.

        Called on every block event from any layer.
        """
        techniques = self.classify(
            pattern_ids=pattern_ids,
            mtmd_signal_types=mtmd_signal_types,
            cross_modal=cross_modal,
            tool_use=tool_use,
        )

        # Primary technique = first in sorted list (deterministic)
        primary = techniques[0] if techniques else JailbreakTechnique.UNKNOWN

        attempt = JailbreakAttempt(
            attempt_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            source_id=source_id,
            session_id=session_id,
            techniques=techniques,
            primary_technique=primary,
            confidence=confidence,
            detection_layer=detection_layer,
            pattern_ids=pattern_ids or [],
            blocked=blocked,
            metadata=metadata or {},
        )

        with self._lock:
            self._attempts.append(attempt)

        return attempt

    def get_stats(self, trend_window_hours: float = 24.0) -> TaxonomyStats:
        """Compute taxonomy statistics.

        Args:
            trend_window_hours: Time window for recent trend computation.
        """
        now = time.time()
        trend_cutoff = now - trend_window_hours * 3600

        by_technique: dict[JailbreakTechnique, int] = {}
        by_layer: dict[str, int] = {}
        recent_trend: dict[JailbreakTechnique, int] = {}

        with self._lock:
            attempts = list(self._attempts)

        for attempt in attempts:
            for tech in attempt.techniques:
                by_technique[tech] = by_technique.get(tech, 0) + 1
                if attempt.timestamp >= trend_cutoff:
                    recent_trend[tech] = recent_trend.get(tech, 0) + 1
            by_layer[attempt.detection_layer] = (
                by_layer.get(attempt.detection_layer, 0) + 1
            )

        top_techniques = sorted(
            by_technique.items(), key=lambda x: x[1], reverse=True,
        )

        return TaxonomyStats(
            total_attempts=len(attempts),
            by_technique=by_technique,
            by_detection_layer=by_layer,
            top_techniques=top_techniques,
            trend_window_hours=trend_window_hours,
            recent_trend=recent_trend,
        )

    def get_technique_trend(
        self, technique: JailbreakTechnique, window_hours: float = 168.0,
    ) -> dict[int, int]:
        """Get hourly counts for a technique over the window.

        Returns dict mapping hour_offset (0=current) to count.
        """
        now = time.time()
        cutoff = now - window_hours * 3600
        hours: dict[int, int] = {}

        with self._lock:
            for attempt in self._attempts:
                if attempt.timestamp < cutoff:
                    continue
                if technique not in attempt.techniques:
                    continue
                offset = int((now - attempt.timestamp) / 3600)
                hours[offset] = hours.get(offset, 0) + 1

        return hours

    def get_top_techniques(
        self, n: int = 5, window_hours: float = 24.0,
    ) -> list[tuple[JailbreakTechnique, int]]:
        """Get the N most common techniques in the time window."""
        now = time.time()
        cutoff = now - window_hours * 3600
        counts: dict[JailbreakTechnique, int] = {}

        with self._lock:
            for attempt in self._attempts:
                if attempt.timestamp < cutoff:
                    continue
                for tech in attempt.techniques:
                    counts[tech] = counts.get(tech, 0) + 1

        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_recent_attempts(self, n: int = 50) -> list[JailbreakAttempt]:
        """Get the N most recent attempts (for API endpoint)."""
        with self._lock:
            return list(self._attempts)[-n:]
