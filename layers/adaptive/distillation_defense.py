"""
Distillation Defense Analyzer — Detects systematic model extraction attacks.

Distillation attacks are CROSS-SESSION, LONG-TERM campaigns where an adversary
systematically queries the model to reconstruct its capabilities. This module
tracks per-API-key query history and detects five attack patterns:

1. Query Diversity Anomaly — broad topic coverage indicating systematic extraction
2. Governance Boundary Mapping — analyzing blocked/allowed patterns to map rules
3. Reasoning Trace Coercion — queries designed to elicit reasoning processes
4. Response Information Gain — maximizing extracted information per query
5. Systematic Complexity Escalation — progressively harder queries

ASSUMED-BREACH POSTURE: This analyzer assumes L1-L3 may have been bypassed.
Distillation attackers typically send individually benign queries that only
become suspicious in aggregate. No single request triggers detection.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from collections import defaultdict
from typing import Any

from aegis.layers.adaptive.distillation_models import (
    DistillationReport,
    DistillationSignal,
    DistillationStrategy,
    InteractionRecord,
)

logger = logging.getLogger(__name__)

# Stopwords excluded from topic hash computation.
_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "about",
    "this", "that", "these", "those", "it", "its", "my", "your", "his",
    "her", "our", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "not", "no", "and", "or", "but", "if", "then", "else",
    "so", "just", "also", "very", "much", "more", "most", "than", "too",
    "only", "own", "same", "all", "each", "every", "both", "few", "some",
    "any", "other", "i", "me", "we", "you", "he", "she", "they", "them",
}


class DistillationDefenseAnalyzer:
    """Cross-session analyzer detecting systematic model distillation attacks.

    Tracks per-API-key interaction history and applies five detection
    strategies to identify coordinated extraction campaigns that would
    appear benign at the individual query level.
    """

    # Strategy weights for combined threat score computation.
    STRATEGY_WEIGHTS: dict[DistillationStrategy, float] = {
        DistillationStrategy.QUERY_DIVERSITY: 0.25,
        DistillationStrategy.BOUNDARY_MAPPING: 0.30,
        DistillationStrategy.REASONING_COERCION: 0.20,
        DistillationStrategy.INFO_HARVESTING: 0.10,
        DistillationStrategy.COMPLEXITY_ESCALATION: 0.15,
    }

    def __init__(
        self,
        window_hours: float = 24.0,
        max_history: int = 10000,
    ) -> None:
        """Initialize the distillation defense analyzer.

        Args:
            window_hours: Time window in hours for pattern analysis.
                Overridden by AEGIS_DISTILLATION_WINDOW_HOURS env var.
            max_history: Maximum interaction records per API key.
                Overridden by AEGIS_DISTILLATION_MAX_HISTORY env var.
        """
        self.window_hours: float = float(
            os.environ.get("AEGIS_DISTILLATION_WINDOW_HOURS", window_hours)
        )
        self.max_history: int = int(
            os.environ.get("AEGIS_DISTILLATION_MAX_HISTORY", max_history)
        )

        self._history: dict[str, list[InteractionRecord]] = defaultdict(list)
        self._lock: threading.Lock = threading.Lock()

        self._global_avg_response_length: float = 500.0
        self._global_response_count: int = 0

        # Pre-compile reasoning coercion patterns (case insensitive).
        self._reasoning_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"explain (your|the) reasoning",
                r"show (your|me your|me the) work",
                r"walk me through",
                r"step by step",
                r"why did you",
                r"how did you (decide|determine|choose|conclude)",
                r"what (rules|guidelines|constraints) do you follow",
                r"what are your (guidelines|constraints|rules|limitations)",
                r"what is your system prompt",
                r"describe your (decision|thinking|reasoning) process",
                r"what factors did you consider",
                r"what criteria (do|did) you (use|apply)",
                r"how (do|does) your (filter|safety|detection|moderation) work",
                r"what (makes|would make) you (block|reject|flag|refuse)",
                r"tell me (about )?your (instructions|programming|training)",
            ]
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_topic_hash(self, text: str) -> str:
        """Compute a coarse topic fingerprint from the top 3 keywords.

        Extracts the three longest non-stopword tokens from *text*,
        sorts them alphabetically, and returns their SHA-256 hash.
        This groups semantically related queries without storing
        raw content.

        Args:
            text: Raw query text.

        Returns:
            Hex-encoded SHA-256 hash representing the topic cluster.
        """
        words = re.findall(r"[a-zA-Z]+", text.lower())
        keywords = [w for w in words if w not in _STOPWORDS]
        # Sort by length descending, take top 3, then sort alphabetically
        # for deterministic hashing regardless of word order.
        top = sorted(keywords, key=len, reverse=True)[:3]
        top.sort()
        return hashlib.sha256("|".join(top).encode()).hexdigest()

    def _compute_complexity(self, text: str) -> float:
        """Estimate the structural complexity of a query.

        Combines word count, vocabulary richness (unique-word ratio),
        and question density into a single scalar.

        Args:
            text: Raw query text.

        Returns:
            Non-negative complexity score.
        """
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        unique_ratio = len(set(w.lower() for w in words)) / word_count
        question_count = text.count("?")
        return word_count * unique_ratio * (1 + question_count * 0.2)

    def _is_reasoning_query(self, text: str) -> bool:
        """Check whether *text* contains reasoning coercion language.

        Args:
            text: Raw query text.

        Returns:
            True if any compiled reasoning pattern matches.
        """
        for pattern in self._reasoning_patterns:
            if pattern.search(text):
                return True
        return False

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def record_interaction(
        self, api_key: str, interaction: InteractionRecord
    ) -> None:
        """Append an interaction to the per-key history.

        Thread-safe.  Trims oldest entries when *max_history* is exceeded
        and updates the running global average response length.

        Args:
            api_key: Hashed API key identifying the caller.
            interaction: Interaction metadata to record.
        """
        with self._lock:
            history = self._history[api_key]
            history.append(interaction)
            if len(history) > self.max_history:
                # Remove oldest entries to stay within budget.
                excess = len(history) - self.max_history
                del history[:excess]

            # Update running global average response length.
            self._global_response_count += 1
            n = self._global_response_count
            self._global_avg_response_length += (
                interaction.response_length - self._global_avg_response_length
            ) / n

    def _get_windowed_history(
        self, api_key: str
    ) -> list[InteractionRecord]:
        """Return a snapshot of recent history within the analysis window.

        Thread-safe.  Returns a copy so callers can iterate without
        holding the lock.

        Args:
            api_key: Hashed API key.

        Returns:
            List of records whose timestamps fall within the window.
        """
        cutoff = time.time() - self.window_hours * 3600
        with self._lock:
            history = list(self._history.get(api_key, []))
        return [r for r in history if r.timestamp > cutoff]

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------

    def check_query_diversity(self, api_key: str) -> DistillationSignal:
        """Detect abnormally broad topic coverage (systematic extraction).

        A legitimate user tends to cluster queries around a narrow set of
        topics.  An attacker performing model extraction sweeps across
        many topics to maximize knowledge coverage.

        Args:
            api_key: Hashed API key.

        Returns:
            DistillationSignal indicating whether query diversity is anomalous.
        """
        history = self._get_windowed_history(api_key)
        query_count = len(history)

        if query_count < 50:
            return DistillationSignal(
                strategy=DistillationStrategy.QUERY_DIVERSITY,
                triggered=False,
                confidence=0.0,
                evidence={"query_count": query_count, "min_required": 50},
            )

        distinct_topics = len({r.topic_hash for r in history})
        topic_coverage = distinct_topics / query_count

        if topic_coverage > 0.7 and query_count > 50:
            confidence = min(0.5 + (query_count - 50) * 0.005, 0.95)
            return DistillationSignal(
                strategy=DistillationStrategy.QUERY_DIVERSITY,
                triggered=True,
                confidence=confidence,
                evidence={
                    "topic_coverage": round(topic_coverage, 4),
                    "distinct_topics": distinct_topics,
                    "query_count": query_count,
                },
            )

        return DistillationSignal(
            strategy=DistillationStrategy.QUERY_DIVERSITY,
            triggered=False,
            confidence=0.0,
            evidence={
                "topic_coverage": round(topic_coverage, 4),
                "distinct_topics": distinct_topics,
                "query_count": query_count,
            },
        )

    def check_boundary_mapping(self, api_key: str) -> DistillationSignal:
        """Detect governance boundary probing (block/allow alternation).

        An attacker mapping the model's safety rules will alternate
        between blocked and allowed requests at an abnormally high rate,
        with an overall block rate in the 15-60% range (neither pure
        benign nor pure attack traffic).

        Args:
            api_key: Hashed API key.

        Returns:
            DistillationSignal indicating boundary mapping behavior.
        """
        history = self._get_windowed_history(api_key)
        total_count = len(history)

        if total_count < 30:
            return DistillationSignal(
                strategy=DistillationStrategy.BOUNDARY_MAPPING,
                triggered=False,
                confidence=0.0,
                evidence={"total_count": total_count, "min_required": 30},
            )

        blocked_count = sum(1 for r in history if r.was_blocked)
        block_rate = blocked_count / total_count

        if block_rate < 0.15 or block_rate > 0.60:
            return DistillationSignal(
                strategy=DistillationStrategy.BOUNDARY_MAPPING,
                triggered=False,
                confidence=0.0,
                evidence={
                    "block_rate": round(block_rate, 4),
                    "blocked_count": blocked_count,
                    "total_count": total_count,
                },
            )

        # Count blocked↔allowed transitions.
        transitions = 0
        for i in range(1, len(history)):
            if history[i].was_blocked != history[i - 1].was_blocked:
                transitions += 1
        alternation_score = transitions / (total_count - 1)

        if alternation_score > 0.3:
            confidence = min(0.6 + alternation_score * 0.5, 0.95)
            return DistillationSignal(
                strategy=DistillationStrategy.BOUNDARY_MAPPING,
                triggered=True,
                confidence=confidence,
                evidence={
                    "block_rate": round(block_rate, 4),
                    "alternation_score": round(alternation_score, 4),
                    "blocked_count": blocked_count,
                    "total_count": total_count,
                },
            )

        return DistillationSignal(
            strategy=DistillationStrategy.BOUNDARY_MAPPING,
            triggered=False,
            confidence=0.0,
            evidence={
                "block_rate": round(block_rate, 4),
                "alternation_score": round(alternation_score, 4),
                "blocked_count": blocked_count,
                "total_count": total_count,
            },
        )

    def check_reasoning_coercion(
        self, api_key: str, query_text: str
    ) -> DistillationSignal:
        """Detect excessive reasoning-elicitation queries.

        An attacker harvesting reasoning traces will send a
        disproportionate number of queries containing phrases like
        "explain your reasoning" or "step by step".

        Args:
            api_key: Hashed API key.
            query_text: Current query text to include in the count.

        Returns:
            DistillationSignal indicating reasoning coercion pattern.
        """
        is_reasoning = self._is_reasoning_query(query_text)
        history = self._get_windowed_history(api_key)
        reasoning_count = sum(1 for r in history if r.is_reasoning_query)
        if is_reasoning:
            reasoning_count += 1
        total = len(history) + 1  # Include current query.

        reasoning_ratio = reasoning_count / max(total, 1)

        if reasoning_ratio > 0.25 and reasoning_count > 20:
            confidence = min(0.55 + reasoning_ratio * 0.5, 0.95)
            return DistillationSignal(
                strategy=DistillationStrategy.REASONING_COERCION,
                triggered=True,
                confidence=confidence,
                evidence={
                    "reasoning_ratio": round(reasoning_ratio, 4),
                    "reasoning_count": reasoning_count,
                    "total": total,
                    "current_is_reasoning": is_reasoning,
                },
            )

        return DistillationSignal(
            strategy=DistillationStrategy.REASONING_COERCION,
            triggered=False,
            confidence=0.0,
            evidence={"current_is_reasoning": is_reasoning},
        )

    def check_information_gain(
        self, api_key: str, response_text: str
    ) -> DistillationSignal:
        """Detect queries optimized for maximum information extraction.

        An attacker maximizing distillation efficiency will achieve
        unusually long responses (high information per query) across
        diverse topics.

        Args:
            api_key: Hashed API key.
            response_text: Current model response (used only for length;
                not stored).

        Returns:
            DistillationSignal indicating information harvesting behavior.
        """
        history = self._get_windowed_history(api_key)

        if len(history) < 20:
            return DistillationSignal(
                strategy=DistillationStrategy.INFO_HARVESTING,
                triggered=False,
                confidence=0.0,
                evidence={
                    "history_count": len(history),
                    "min_required": 20,
                },
            )

        avg_response_length = sum(
            r.response_length for r in history
        ) / len(history)

        if avg_response_length > 2 * self._global_avg_response_length:
            # Measure query diversity across the last 20 interactions.
            recent = history[-20:]
            all_tokens: list[str] = []
            unique_tokens: set[str] = set()
            for r in recent:
                tokens = r.query_text_summary.lower().split()
                all_tokens.extend(tokens)
                unique_tokens.update(tokens)
            response_diversity = (
                len(unique_tokens) / max(len(all_tokens), 1)
            )

            if response_diversity > 0.6:
                ratio = avg_response_length / self._global_avg_response_length
                confidence = min(0.5 + (ratio - 2) * 0.2, 0.90)
                return DistillationSignal(
                    strategy=DistillationStrategy.INFO_HARVESTING,
                    triggered=True,
                    confidence=confidence,
                    evidence={
                        "avg_response_length": round(avg_response_length, 2),
                        "global_avg_response_length": round(
                            self._global_avg_response_length, 2
                        ),
                        "response_length_ratio": round(ratio, 4),
                        "response_diversity": round(response_diversity, 4),
                    },
                )

        return DistillationSignal(
            strategy=DistillationStrategy.INFO_HARVESTING,
            triggered=False,
            confidence=0.0,
            evidence={
                "avg_response_length": round(avg_response_length, 2),
                "global_avg_response_length": round(
                    self._global_avg_response_length, 2
                ),
            },
        )

    def check_complexity_escalation(
        self, api_key: str
    ) -> DistillationSignal:
        """Detect systematic increase in query complexity over time.

        A distillation attacker may start with simple queries and
        progressively escalate to harder ones to map the model's
        capability frontier.

        Args:
            api_key: Hashed API key.

        Returns:
            DistillationSignal indicating complexity escalation.
        """
        history = self._get_windowed_history(api_key)
        n = len(history)

        if n < 30:
            return DistillationSignal(
                strategy=DistillationStrategy.COMPLEXITY_ESCALATION,
                triggered=False,
                confidence=0.0,
                evidence={"query_count": n, "min_required": 30},
            )

        # Simple linear regression: y = complexity_score, x = ordinal index.
        complexities = [r.complexity_score for r in history]
        sum_x: float = 0.0
        sum_y: float = 0.0
        sum_xy: float = 0.0
        sum_x2: float = 0.0
        for i, c in enumerate(complexities):
            sum_x += i
            sum_y += c
            sum_xy += i * c
            sum_x2 += i * i

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator

        mean_y = sum_y / n
        normalized_slope = slope / max(mean_y, 1e-6)

        if normalized_slope > 0.02 and n > 30:
            confidence = min(0.5 + normalized_slope * 5, 0.90)
            return DistillationSignal(
                strategy=DistillationStrategy.COMPLEXITY_ESCALATION,
                triggered=True,
                confidence=confidence,
                evidence={
                    "slope": round(slope, 6),
                    "normalized_slope": round(normalized_slope, 6),
                    "mean_complexity": round(mean_y, 4),
                    "query_count": n,
                },
            )

        return DistillationSignal(
            strategy=DistillationStrategy.COMPLEXITY_ESCALATION,
            triggered=False,
            confidence=0.0,
            evidence={
                "slope": round(slope, 6),
                "normalized_slope": round(normalized_slope, 6),
                "mean_complexity": round(mean_y, 4),
                "query_count": n,
            },
        )

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    async def analyze(
        self,
        api_key: str,
        query_text: str,
        response_text: str,
        was_blocked: bool,
        block_reason: str = "",
    ) -> DistillationReport:
        """Run all five detection strategies and produce a unified report.

        Creates an InteractionRecord, persists it in the per-key history,
        then evaluates every strategy.  The combined threat score is the
        weighted sum of triggered signal confidences (weights always sum
        to 1.0, untriggered strategies contribute zero).

        Args:
            api_key: Hashed API key identifying the caller.
            query_text: Raw query text (used for feature extraction, then
                only a truncated summary is stored).
            response_text: Model response text (used for length only).
            was_blocked: Whether this request was blocked by any layer.
            block_reason: Reason string if blocked.

        Returns:
            DistillationReport with per-strategy signals and a combined
            threat assessment.
        """
        # Build interaction record.
        topic_hash = self._compute_topic_hash(query_text)
        complexity = self._compute_complexity(query_text)
        is_reasoning = self._is_reasoning_query(query_text)

        record = InteractionRecord(
            timestamp=time.time(),
            topic_hash=topic_hash,
            query_text_summary=query_text[:200],  # Truncated summary only.
            response_length=len(response_text),
            was_blocked=was_blocked,
            block_reason=block_reason,
            complexity_score=complexity,
            is_reasoning_query=is_reasoning,
        )

        self.record_interaction(api_key, record)

        # Run all five detection strategies.
        sig_diversity = self.check_query_diversity(api_key)
        sig_boundary = self.check_boundary_mapping(api_key)
        sig_reasoning = self.check_reasoning_coercion(api_key, query_text)
        sig_info = self.check_information_gain(api_key, response_text)
        sig_complexity = self.check_complexity_escalation(api_key)

        signals = [
            sig_diversity,
            sig_boundary,
            sig_reasoning,
            sig_info,
            sig_complexity,
        ]

        # Weighted combination — untriggered signals contribute zero.
        combined_threat_score = sum(
            self.STRATEGY_WEIGHTS[s.strategy] * s.confidence
            for s in signals
            if s.triggered
        )

        should_alert = combined_threat_score >= 0.60
        should_block = combined_threat_score >= 0.85

        if combined_threat_score < 0.30:
            recommended_action = "none"
        elif combined_threat_score < 0.60:
            recommended_action = "increase_monitoring"
        elif combined_threat_score < 0.85:
            recommended_action = "rate_limit"
        else:
            recommended_action = "block_and_review"

        # Compute window bounds from history.
        history = self._get_windowed_history(api_key)
        window_end = time.time()
        window_start = window_end - self.window_hours * 3600
        blocked_queries = sum(1 for r in history if r.was_blocked)

        return DistillationReport(
            signals=signals,
            combined_threat_score=round(combined_threat_score, 4),
            should_alert=should_alert,
            should_block=should_block,
            recommended_action=recommended_action,
            api_key=api_key,
            window_start=window_start,
            window_end=window_end,
            total_queries=len(history),
            blocked_queries=blocked_queries,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove interaction records older than the analysis window.

        Thread-safe.  Also removes API keys with no remaining records.

        Returns:
            Total number of records removed across all keys.
        """
        cutoff = time.time() - self.window_hours * 3600
        removed = 0

        with self._lock:
            empty_keys: list[str] = []
            for key, records in self._history.items():
                original_len = len(records)
                self._history[key] = [
                    r for r in records if r.timestamp > cutoff
                ]
                removed += original_len - len(self._history[key])
                if not self._history[key]:
                    empty_keys.append(key)
            for key in empty_keys:
                del self._history[key]

        if removed > 0:
            logger.debug(
                "Distillation defense cleanup: removed %d expired records",
                removed,
            )
        return removed

    def get_stats(self, api_key: str) -> dict[str, Any]:
        """Return diagnostic statistics for monitoring.

        Args:
            api_key: Hashed API key to report on (may be empty for
                global stats only).

        Returns:
            Dictionary with tracked key count, total entries,
            per-key entry count, window size, and global average
            response length.
        """
        with self._lock:
            tracked_keys = len(self._history)
            total_entries = sum(len(v) for v in self._history.values())
            api_key_entries = len(self._history.get(api_key, []))

        return {
            "tracked_keys": tracked_keys,
            "total_entries": total_entries,
            "api_key_entries": api_key_entries,
            "window_hours": self.window_hours,
            "global_avg_response_length": round(
                self._global_avg_response_length, 2
            ),
        }
