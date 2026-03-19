"""
Source Behavioral Profiling — Extension 5.2

Maintains a behavioral model for each interaction source, tracking query
diversity, injection attempt rates, block-to-adaptation ratios, session
coherence, and temporal patterns. These metrics distinguish autonomous
jailbreak agents from legitimate users.

ASSUMED-BREACH POSTURE: This profiler assumes all single-turn and
multi-turn detectors may be bypassed. It builds a cumulative behavioral
model per source that detects the statistical signature of systematic
attack campaigns — even when individual turns appear benign. A compromised
profiler store could suppress risk escalation — the MTMD operates
independently as a cross-check.
"""

from __future__ import annotations

import enum
import logging
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RiskLevel(str, enum.Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProfileUpdate:
    source_id: str
    turn_content: str
    injection_score: float
    was_blocked: bool
    detection_categories: list[str]
    timestamp: float


@dataclass
class SourceProfile:
    source_id: str
    first_seen: float
    last_seen: float
    total_interactions: int = 0
    query_diversity_score: float = 1.0
    topic_centroid: np.ndarray | None = None
    injection_attempt_count: int = 0
    block_count: int = 0
    adaptation_count: int = 0
    manipulation_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW

    # Internal tracking
    _embedding_distances: list[float] = field(default_factory=list)
    _injection_scores: list[float] = field(default_factory=list)
    _timestamps: list[float] = field(default_factory=list)
    _blocked_indices: list[int] = field(default_factory=list)
    _category_after_block: list[tuple[set[str], set[str]]] = field(default_factory=list)
    _token_sets: list[set[str]] = field(default_factory=list)


def _risk_level_from_score(score: float) -> RiskLevel:
    if score >= 0.7:
        return RiskLevel.CRITICAL
    elif score >= 0.5:
        return RiskLevel.HIGH
    elif score >= 0.3:
        return RiskLevel.ELEVATED
    return RiskLevel.LOW


class SourceBehavioralProfiler:
    """Maintains behavioral models for interaction sources.

    Tracks query diversity, injection attempt rate, block-to-adaptation
    ratio, session coherence, and temporal patterns. Updates run after
    the response is sent (async) so they add zero latency to the request.
    """

    def __init__(
        self,
        max_profiles: int = 10000,
        embed_fn: Any | None = None,
    ):
        self._max_profiles = max_profiles
        self._embed_fn = embed_fn
        self._lock = threading.Lock()
        self._profiles: OrderedDict[str, SourceProfile] = OrderedDict()

    @property
    def profile_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    def get_profile(self, source_id: str) -> SourceProfile | None:
        with self._lock:
            return self._profiles.get(source_id)

    def update(self, update: ProfileUpdate) -> SourceProfile:
        """Update the behavioral profile for a source.

        Called after each request completes. Returns the updated profile.
        """
        with self._lock:
            profile = self._profiles.get(update.source_id)
            if profile is None:
                # LRU eviction
                if len(self._profiles) >= self._max_profiles:
                    self._profiles.popitem(last=False)
                profile = SourceProfile(
                    source_id=update.source_id,
                    first_seen=update.timestamp,
                    last_seen=update.timestamp,
                )
                self._profiles[update.source_id] = profile
            else:
                self._profiles.move_to_end(update.source_id)

            profile.last_seen = update.timestamp
            profile.total_interactions += 1
            profile._timestamps.append(update.timestamp)
            profile._injection_scores.append(update.injection_score)

            # Track injection attempts
            if update.injection_score > 0.3:
                profile.injection_attempt_count += 1

            # Track blocks and adaptation
            current_cats = set(
                c.split("-")[0] for c in update.detection_categories if "-" in c
            )
            if update.was_blocked:
                profile.block_count += 1
                profile._blocked_indices.append(profile.total_interactions - 1)

            # Check for adaptation: was previous turn blocked with different categories?
            if (
                profile._blocked_indices
                and len(profile._blocked_indices) >= 1
                and profile.total_interactions >= 2
            ):
                last_blocked_idx = profile._blocked_indices[-1]
                # If the previous interaction was a block and this one has different categories
                if last_blocked_idx == profile.total_interactions - 2 and current_cats:
                    # We need to know what the blocked categories were
                    if profile._category_after_block:
                        prev_blocked_cats, _ = profile._category_after_block[-1]
                        if prev_blocked_cats and prev_blocked_cats != current_cats:
                            profile.adaptation_count += 1

            # Store categories for adaptation tracking
            if update.was_blocked:
                profile._category_after_block.append((current_cats, set()))
            elif profile._category_after_block and not update.was_blocked:
                # Update the "after" categories for the most recent block
                if profile._category_after_block:
                    blocked_cats, _ = profile._category_after_block[-1]
                    profile._category_after_block[-1] = (blocked_cats, current_cats)

            # Update query diversity
            self._update_query_diversity(profile, update.turn_content)

            # Recompute risk score
            profile.manipulation_risk_score = self._compute_risk_score(profile)
            old_level = profile.risk_level
            profile.risk_level = _risk_level_from_score(profile.manipulation_risk_score)

            if profile.risk_level != old_level:
                logger.info(
                    "Source %s risk level changed: %s -> %s (score=%.2f)",
                    update.source_id, old_level.value, profile.risk_level.value,
                    profile.manipulation_risk_score,
                )

            return profile

    def _update_query_diversity(self, profile: SourceProfile, content: str) -> None:
        """Update query diversity using embeddings or Jaccard fallback."""
        if self._embed_fn is not None:
            try:
                emb = np.array(self._embed_fn(content), dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                if profile.topic_centroid is None:
                    profile.topic_centroid = emb.copy()
                    profile._embedding_distances.append(0.0)
                else:
                    # Distance from centroid
                    dist = 1.0 - float(np.dot(emb, profile.topic_centroid))
                    dist = max(dist, 0.0)
                    profile._embedding_distances.append(dist)

                    # Update centroid (running average)
                    n = profile.total_interactions
                    profile.topic_centroid = (
                        profile.topic_centroid * ((n - 1) / n) + emb * (1 / n)
                    )
                    c_norm = np.linalg.norm(profile.topic_centroid)
                    if c_norm > 0:
                        profile.topic_centroid /= c_norm

                if profile._embedding_distances:
                    profile.query_diversity_score = min(
                        sum(profile._embedding_distances) / len(profile._embedding_distances),
                        1.0,
                    )
                return
            except Exception:
                pass

        # Jaccard fallback
        tokens = set(content.lower().split())
        profile._token_sets.append(tokens)

        if len(profile._token_sets) < 2:
            profile.query_diversity_score = 0.5
            return

        # Average pairwise Jaccard distance (sample recent pairs)
        recent = profile._token_sets[-20:]
        distances = []
        for i in range(len(recent) - 1):
            a, b = recent[i], recent[i + 1]
            union = a | b
            if not union:
                continue
            intersection = a & b
            jaccard = len(intersection) / len(union)
            distances.append(1.0 - jaccard)

        if distances:
            profile.query_diversity_score = min(
                sum(distances) / len(distances), 1.0,
            )

    def _compute_risk_score(self, profile: SourceProfile) -> float:
        """Combine all metrics into manipulation_risk_score."""
        score = 0.0

        if profile.total_interactions == 0:
            return 0.0

        # Injection attempt rate
        injection_rate = profile.injection_attempt_count / profile.total_interactions
        if injection_rate > 0.3:
            score += 0.3

        # Low diversity + elevated injection
        if profile.query_diversity_score < 0.3 and injection_rate > 0.2:
            score += 0.2

        # Block-to-adaptation ratio
        if profile.block_count > 0:
            adaptation_ratio = profile.adaptation_count / profile.block_count
            if adaptation_ratio > 0.5:
                score += 0.2

        # Session coherence (low coherence + elevated injection)
        if profile._embedding_distances:
            avg_dist = sum(profile._embedding_distances) / len(profile._embedding_distances)
            session_coherence = 1.0 - avg_dist
            if session_coherence < 0.3 and injection_rate > 0.2:
                score += 0.15

        # Temporal pattern (automation detection)
        if len(profile._timestamps) >= 5 and injection_rate > 0.2:
            intervals = [
                profile._timestamps[i] - profile._timestamps[i - 1]
                for i in range(1, len(profile._timestamps))
            ]
            if intervals:
                mean_interval = sum(intervals) / len(intervals)
                if mean_interval > 0:
                    std_interval = (
                        sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                    ) ** 0.5
                    cv = std_interval / mean_interval
                    if cv < 0.3:
                        score += 0.15

        return min(score, 1.0)

    def get_risk_score(self, source_id: str) -> float:
        """Get the current risk score for a source. Returns 0.0 if unknown."""
        with self._lock:
            profile = self._profiles.get(source_id)
            if profile is None:
                return 0.0
            return profile.manipulation_risk_score
