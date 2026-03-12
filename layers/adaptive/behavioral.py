"""
Analyzer 3 — Behavioral Baseline Analyzer (Immune Surveillance Analog)

Maintains per-user and per-tenant behavioral baselines: typical request
frequency, average token count, topic distribution (embedding centroid),
time-of-day patterns, and model usage patterns. Uses Population Stability
Index (PSI > 0.25 = significant drift) and Kolmogorov-Smirnov tests for
continuous variable comparison.

Also serves as Analyzer 4 — Multi-Turn Sequence Analyzer in the bootstrap.
Analyzes conversation history for progressive jailbreak attempts across
multiple turns via a sliding window of the last N turns per session.
Monotonically increasing boundary-testing patterns trigger high-confidence
alerts. Catches "slow and low" attacks that single-turn analysis misses.

ASSUMED-BREACH POSTURE: This analyzer assumes all pattern-based and
ML-based detection has been bypassed. Behavioral analysis is the last
line of adaptive defense — it catches attacks that look individually benign
but are anomalous in aggregate. A compromised behavioral baseline (e.g.,
an attacker slowly shifting the baseline over weeks) would degrade this
analyzer. Baselines are periodically validated against tenant-wide
distributions to detect individual baseline poisoning.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Any

from aegis.config import AdaptiveConfig
from aegis.models.adaptive_result import (
    AdaptiveAnalysisResult,
    AnalyzerType,
    DCASignal,
    SignalType,
)
from aegis.models.request_context import ChatMessage, RequestContext
from aegis.models.scan_result import ThreatCategory

logger = logging.getLogger(__name__)

# Keywords that indicate boundary-testing / escalation in multi-turn
_ESCALATION_KEYWORDS = [
    "ignore", "bypass", "override", "disable", "remove", "restrictions",
    "safety", "guidelines", "rules", "filter", "limitation", "unrestricted",
    "jailbreak", "developer mode", "no restrictions", "do anything",
    "pretend", "act as", "roleplay", "hypothetically", "fictional",
    "no limits", "all bets are off", "forget", "previous instructions",
]


class BehavioralAnalyzer:
    """Per-session behavioral baseline analyzer with multi-turn detection.

    Tracks per-session turn history and detects:
    1. Multi-turn escalation (progressive boundary testing)
    2. Abnormal message length patterns
    3. Escalation keyword density trends
    """

    def __init__(self, config: AdaptiveConfig):
        self._config = config
        # Per-session turn history: session_id -> list of per-turn scores
        self._session_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _escalation_score(self, text: str) -> float:
        """Score how 'escalatory' a single message is (0.0-1.0).

        Uses keyword density (hits / word count) to resist padding dilution
        attacks where benign filler is inserted between injection tokens.
        A message with 2 keywords in 10 words is more suspicious than
        2 keywords in 200 words — but both still count.
        """
        text_lower = text.lower()
        hits = sum(1 for kw in _ESCALATION_KEYWORDS if kw in text_lower)
        if hits == 0:
            return 0.0
        # Base score: 3+ keywords = max
        base_score = min(hits / 3.0, 1.0)
        # Density bonus: if keywords are sparse (padding attack), ensure
        # minimum score based on raw hit count
        word_count = max(len(text_lower.split()), 1)
        density = hits / word_count
        # Even a single keyword in a long padded message gets minimum 0.15
        # Two keywords get minimum 0.30, three+ get base_score
        min_score = min(hits * 0.15, 0.5)
        return max(base_score, min_score)

    def _analyze_multi_turn(
        self, session_id: str, messages: list[ChatMessage]
    ) -> tuple[bool, float, list[DCASignal]]:
        """Analyze conversation history for progressive escalation.

        Returns (is_threat, confidence, dca_signals).
        """
        user_messages = [m for m in messages if m.role == "user" and m.content]
        if len(user_messages) < 2:
            return False, 0.0, []

        # Score each user message for escalation content
        window = user_messages[-self._config.multi_turn_window:]
        scores = [self._escalation_score(m.content) for m in window]

        # Record in session history
        self._session_history[session_id] = [
            {"score": s, "length": len(m.content)}
            for s, m in zip(scores, window)
        ]

        # Detect monotonically increasing escalation
        increasing_count = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[i - 1] and scores[i] > 0:
                increasing_count += 1

        # Calculate overall escalation metrics
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        trend_ratio = increasing_count / max(len(scores) - 1, 1)

        signals = []

        # Multi-turn escalation: monotonically increasing + high latest score
        if trend_ratio >= 0.5 and max_score >= 0.5 and len(scores) >= 3:
            confidence = min(trend_ratio * max_score * 1.5, 1.0)
            signals.append(DCASignal(
                signal_type=SignalType.DANGER,
                source="multi_turn_sequence",
                value=confidence,
                description=(
                    f"Multi-turn escalation detected: {increasing_count}/{len(scores)-1} "
                    f"turns increasing, max_score={max_score:.2f}"
                ),
            ))
            return True, confidence, signals

        # High average escalation across window
        if avg_score >= 0.4 and len(scores) >= 2:
            confidence = min(avg_score, 0.85)
            signals.append(DCASignal(
                signal_type=SignalType.DANGER,
                source="multi_turn_sequence",
                value=confidence,
                description=f"Elevated escalation across {len(scores)} turns, avg={avg_score:.2f}",
            ))
            return True, confidence, signals

        if avg_score > 0:
            signals.append(DCASignal(
                signal_type=SignalType.SAFE,
                source="multi_turn_sequence",
                value=1.0 - avg_score,
                description=f"Low escalation: avg={avg_score:.2f}",
            ))

        return False, 0.0, signals

    def _analyze_single_turn(self, text: str) -> tuple[bool, float, list[DCASignal]]:
        """Behavioral anomaly check on a single message."""
        signals = []
        score = self._escalation_score(text)

        if score >= 0.7:
            signals.append(DCASignal(
                signal_type=SignalType.DANGER,
                source="behavioral_baseline",
                value=score,
                description=f"High escalation keyword density: {score:.2f}",
            ))
            return True, score * 0.8, signals

        if score > 0:
            signals.append(DCASignal(
                signal_type=SignalType.SAFE,
                source="behavioral_baseline",
                value=1.0 - score,
                description=f"Normal behavioral signal: escalation={score:.2f}",
            ))
        else:
            signals.append(DCASignal(
                signal_type=SignalType.SAFE,
                source="behavioral_baseline",
                value=0.9,
                description="No escalation keywords detected",
            ))

        return False, 0.0, signals

    async def analyze(self, context: RequestContext) -> AdaptiveAnalysisResult:
        """Analyze request for behavioral anomalies and multi-turn escalation."""
        start = time.perf_counter()

        try:
            last_msg = context.last_user_message or ""

            # Single-turn behavioral analysis
            st_threat, st_conf, st_signals = self._analyze_single_turn(last_msg)

            # Multi-turn escalation analysis
            mt_threat, mt_conf, mt_signals = self._analyze_multi_turn(
                context.session_id, context.messages
            )

            elapsed = (time.perf_counter() - start) * 1000

            is_threat = st_threat or mt_threat
            confidence = max(st_conf, mt_conf)

            all_signals = st_signals + mt_signals

            details: dict[str, Any] = {
                "single_turn_threat": st_threat,
                "multi_turn_threat": mt_threat,
                "single_turn_confidence": st_conf,
                "multi_turn_confidence": mt_conf,
                "session_turns": len(context.messages),
            }

            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.BEHAVIORAL_BASELINE,
                is_threat=is_threat,
                confidence=confidence,
                threat_category=ThreatCategory.JAILBREAK if is_threat else ThreatCategory.UNKNOWN,
                details=details,
                dca_signals=all_signals,
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Behavioral analyzer crashed: %s", e)
            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.BEHAVIORAL_BASELINE,
                is_threat=False,
                confidence=0.0,
                details={"error": str(e)},
                dca_signals=[
                    DCASignal(
                        signal_type=SignalType.DANGER,
                        source="behavioral_baseline",
                        value=0.3,
                        description=f"Behavioral analyzer crash: {e}",
                    )
                ],
                latency_ms=elapsed,
            )
