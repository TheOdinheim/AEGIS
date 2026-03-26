"""
Indicator Reputation Scoring — Trust-Before-Distribute.

Every indicator submitted to the federation hub gets a trust score before
being distributed. Indicators must earn their way into the network.

Four-factor weighted scoring:
  1. Source trust (40%): Node trust score of the submitting node
  2. Consistency (25%): How many other nodes report similar indicators
  3. Pattern validity (20%): Structural quality checks on the indicator
  4. Temporal coherence (15%): Timing anomaly detection (burst/stale)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReputationScore:
    """Computed reputation for an indicator."""
    indicator_id: str
    overall: float  # 0.0–1.0
    source_trust: float
    consistency: float
    pattern_validity: float
    temporal_coherence: float
    accepted: bool
    reasons: list[str] = field(default_factory=list)


class IndicatorReputationScorer:
    """Score indicators before accepting them into the federation.

    Parameters:
        accept_threshold: Minimum overall score to accept (default 0.4).
        quarantine_threshold: Below this, indicator is quarantined (default 0.2).
        source_weight: Weight for source trust factor (default 0.40).
        consistency_weight: Weight for consistency factor (default 0.25).
        pattern_weight: Weight for pattern validity factor (default 0.20).
        temporal_weight: Weight for temporal coherence factor (default 0.15).
        burst_window_seconds: Window for burst detection (default 60).
        burst_max_count: Max indicators from one source in burst window (default 10).
    """

    def __init__(
        self,
        *,
        accept_threshold: float = 0.4,
        quarantine_threshold: float = 0.2,
        source_weight: float = 0.40,
        consistency_weight: float = 0.25,
        pattern_weight: float = 0.20,
        temporal_weight: float = 0.15,
        burst_window_seconds: float = 60.0,
        burst_max_count: int = 10,
    ) -> None:
        self._accept_threshold = accept_threshold
        self._quarantine_threshold = quarantine_threshold
        self._weights = {
            "source": source_weight,
            "consistency": consistency_weight,
            "pattern": pattern_weight,
            "temporal": temporal_weight,
        }
        self._burst_window = burst_window_seconds
        self._burst_max = burst_max_count

        # Track submissions per source for burst detection
        self._submission_times: dict[str, list[float]] = {}
        # Track embeddings for consistency checking
        self._recent_embeddings: list[tuple[str, np.ndarray, float]] = []  # (source, emb, time)
        self._max_recent = 500

        # Quarantine store
        self._quarantined: dict[str, dict[str, Any]] = {}

        # Stats
        self._total_scored: int = 0
        self._total_accepted: int = 0
        self._total_quarantined: int = 0

    def score(
        self,
        indicator: dict[str, Any],
        source_node_id: str,
        source_trust: float = 0.3,
    ) -> ReputationScore:
        """Score an indicator's reputation.

        Args:
            indicator: STIX indicator dict.
            source_node_id: Node that submitted it.
            source_trust: Trust score of the source node (0.0-1.0).
        """
        self._total_scored += 1
        ind_id = indicator.get("id", "unknown")
        reasons: list[str] = []

        # Factor 1: Source trust (direct passthrough)
        f_source = max(0.0, min(1.0, source_trust))

        # Factor 2: Consistency — how many other sources report similar
        f_consistency = self._score_consistency(indicator, source_node_id)

        # Factor 3: Pattern validity — structural checks
        f_pattern = self._score_pattern_validity(indicator, reasons)

        # Factor 4: Temporal coherence — burst/stale detection
        f_temporal = self._score_temporal_coherence(source_node_id, reasons)

        overall = (
            self._weights["source"] * f_source
            + self._weights["consistency"] * f_consistency
            + self._weights["pattern"] * f_pattern
            + self._weights["temporal"] * f_temporal
        )
        overall = round(max(0.0, min(1.0, overall)), 4)

        accepted = overall >= self._accept_threshold

        if overall < self._quarantine_threshold:
            self._quarantined[ind_id] = {
                "indicator": indicator,
                "source": source_node_id,
                "score": overall,
                "reasons": list(reasons),
                "quarantined_at": time.time(),
            }
            self._total_quarantined += 1
            reasons.append(f"quarantined (score {overall:.3f} < {self._quarantine_threshold})")
        elif accepted:
            self._total_accepted += 1

        # Record embedding for future consistency checks
        emb = self._extract_embedding(indicator)
        if emb is not None:
            self._recent_embeddings.append((source_node_id, emb, time.time()))
            if len(self._recent_embeddings) > self._max_recent:
                self._recent_embeddings = self._recent_embeddings[-self._max_recent:]

        # Record submission time for burst detection
        now = time.time()
        if source_node_id not in self._submission_times:
            self._submission_times[source_node_id] = []
        self._submission_times[source_node_id].append(now)

        score = ReputationScore(
            indicator_id=ind_id,
            overall=overall,
            source_trust=round(f_source, 4),
            consistency=round(f_consistency, 4),
            pattern_validity=round(f_pattern, 4),
            temporal_coherence=round(f_temporal, 4),
            accepted=accepted,
            reasons=reasons,
        )

        if not accepted:
            logger.warning(
                "Indicator %s from %s rejected (score=%.3f): %s",
                ind_id, source_node_id, overall, "; ".join(reasons),
            )

        return score

    def _score_consistency(
        self, indicator: dict[str, Any], source_node_id: str,
    ) -> float:
        """How many other sources report similar indicators."""
        emb = self._extract_embedding(indicator)
        if emb is None:
            return 0.5  # No embedding → neutral score

        if not self._recent_embeddings:
            return 0.5  # First indicator → neutral

        # Count unique sources with similar embeddings
        similar_sources: set[str] = set()
        for src, ref_emb, _ in self._recent_embeddings:
            if src == source_node_id:
                continue
            if len(ref_emb) != len(emb):
                continue
            norm_a = np.linalg.norm(emb)
            norm_b = np.linalg.norm(ref_emb)
            if norm_a == 0 or norm_b == 0:
                continue
            sim = float(np.dot(emb, ref_emb) / (norm_a * norm_b))
            if sim >= 0.8:
                similar_sources.add(src)

        if not similar_sources:
            return 0.3  # No corroboration

        # More sources = higher consistency (cap at 1.0)
        return min(1.0, 0.5 + 0.25 * len(similar_sources))

    def _score_pattern_validity(
        self, indicator: dict[str, Any], reasons: list[str],
    ) -> float:
        """Structural quality checks on the indicator."""
        score = 1.0

        # Must have valid STIX type
        if indicator.get("type") != "indicator":
            score -= 0.3
            reasons.append("invalid STIX type")

        # Must have valid ID format
        ind_id = indicator.get("id", "")
        if not ind_id.startswith("indicator--"):
            score -= 0.2
            reasons.append("invalid ID format")

        # Must have pattern field
        pattern_str = indicator.get("pattern", "")
        if not pattern_str:
            score -= 0.4
            reasons.append("missing pattern")
            return max(0.0, score)

        # Parse pattern JSON
        try:
            pattern = json.loads(pattern_str)
        except (json.JSONDecodeError, TypeError):
            score -= 0.3
            reasons.append("invalid pattern JSON")
            return max(0.0, score)

        # Should have MITRE tactic
        if not pattern.get("mitre_tactic"):
            score -= 0.1

        # Should have embedding
        emb = pattern.get("threat_embedding")
        if not emb or not isinstance(emb, list):
            score -= 0.2
            reasons.append("missing embedding")
        elif len(emb) < 10:
            score -= 0.1
            reasons.append("embedding too short")

        # Should have confidence
        conf = pattern.get("confidence")
        if conf is not None and (conf < 0 or conf > 1):
            score -= 0.1
            reasons.append("confidence out of range")

        return max(0.0, score)

    def _score_temporal_coherence(
        self, source_node_id: str, reasons: list[str],
    ) -> float:
        """Detect burst submissions or suspicious timing."""
        now = time.time()
        times = self._submission_times.get(source_node_id, [])

        if not times:
            return 0.8  # First submission → slightly positive

        # Count recent submissions within burst window
        recent = [t for t in times if now - t < self._burst_window]
        if len(recent) >= self._burst_max:
            reasons.append(
                f"burst detected ({len(recent)} in {self._burst_window}s)"
            )
            return 0.1  # Flooding

        # Moderate rate → slight penalty
        if len(recent) >= self._burst_max // 2:
            return 0.5

        return 0.9  # Normal rate

    def get_quarantined(self) -> list[dict[str, Any]]:
        """Return quarantined indicators."""
        result = []
        for ind_id, info in self._quarantined.items():
            result.append({
                "indicator_id": ind_id,
                "source": info["source"],
                "score": info["score"],
                "reasons": info["reasons"],
                "quarantined_at": info["quarantined_at"],
            })
        return result

    def release_from_quarantine(self, indicator_id: str) -> dict[str, Any] | None:
        """Release an indicator from quarantine. Returns the indicator or None."""
        info = self._quarantined.pop(indicator_id, None)
        if info:
            return info["indicator"]
        return None

    @staticmethod
    def _extract_embedding(indicator: dict[str, Any]) -> np.ndarray | None:
        """Extract embedding from STIX indicator pattern field."""
        pattern_str = indicator.get("pattern", "")
        if not pattern_str:
            return None
        try:
            parsed = json.loads(pattern_str)
            emb = parsed.get("threat_embedding")
            if emb is not None:
                return np.array(emb, dtype=np.float64)
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_scored": self._total_scored,
            "total_accepted": self._total_accepted,
            "total_quarantined": self._total_quarantined,
            "acceptance_rate": round(
                self._total_accepted / max(1, self._total_scored), 3
            ),
            "quarantine_size": len(self._quarantined),
        }
