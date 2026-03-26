"""
Indicator Registry — In-Memory STIX Indicator Store.

Stores STIX 2.1 indicators with deduplication by embedding similarity,
lifecycle management (active → dormant after 180 days), and query support.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DORMANT_AGE_SECONDS = 180 * 86400  # 180 days


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _extract_embedding(indicator: dict[str, Any]) -> np.ndarray | None:
    """Extract embedding from STIX indicator pattern field."""
    pattern = indicator.get("pattern", "")
    if not pattern:
        return None
    try:
        parsed = json.loads(pattern)
        emb = parsed.get("threat_embedding")
        if emb is not None:
            return np.array(emb, dtype=np.float64)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


class IndicatorRegistry:
    """In-memory STIX indicator store with deduplication and lifecycle.

    Parameters:
        similarity_threshold: Cosine similarity threshold for dedup (default 0.95).
        dormant_age_seconds: Seconds before inactive indicators become dormant.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.95,
        dormant_age_seconds: int = _DORMANT_AGE_SECONDS,
    ) -> None:
        self._indicators: dict[str, dict[str, Any]] = {}
        self._hit_counts: Counter[str] = Counter()
        self._similarity_threshold = similarity_threshold
        self._dormant_age = dormant_age_seconds

    def add_indicator(self, indicator: dict[str, Any]) -> str:
        """Store a STIX indicator. Returns its ID.

        If an indicator with similar embedding (cosine > threshold) exists,
        merges by incrementing hit count instead of creating a duplicate.
        """
        ind_id = indicator.get("id", "")
        if not ind_id:
            raise ValueError("Indicator must have an 'id' field")

        # Check for duplicates by embedding similarity
        new_emb = _extract_embedding(indicator)
        if new_emb is not None:
            for existing_id, existing in self._indicators.items():
                if existing.get("_status") == "dormant":
                    continue
                existing_emb = _extract_embedding(existing)
                if existing_emb is not None and len(existing_emb) == len(new_emb):
                    sim = _cosine_similarity(new_emb, existing_emb)
                    if sim >= self._similarity_threshold:
                        self._hit_counts[existing_id] += 1
                        logger.debug(
                            "Merged indicator %s into %s (similarity=%.3f)",
                            ind_id, existing_id, sim,
                        )
                        return existing_id

        # Store new indicator
        indicator = dict(indicator)  # copy
        indicator["_status"] = "active"
        indicator["_added_at"] = time.time()
        self._indicators[ind_id] = indicator
        self._hit_counts[ind_id] = 1

        logger.info("Registered indicator %s", ind_id)
        return ind_id

    def get_indicator(self, indicator_id: str) -> dict[str, Any] | None:
        """Retrieve a single indicator by ID."""
        ind = self._indicators.get(indicator_id)
        if ind is None:
            return None
        result = {k: v for k, v in ind.items() if not k.startswith("_")}
        result["hit_count"] = self._hit_counts.get(indicator_id, 0)
        result["status"] = ind.get("_status", "active")
        return result

    def get_indicators_since(self, timestamp: str) -> list[dict[str, Any]]:
        """Get active indicators created after the given ISO timestamp."""
        results = []
        for ind_id, ind in self._indicators.items():
            if ind.get("_status") == "dormant":
                continue
            created = ind.get("created", "")
            if created > timestamp:
                result = {k: v for k, v in ind.items() if not k.startswith("_")}
                result["hit_count"] = self._hit_counts.get(ind_id, 0)
                results.append(result)
        return results

    def get_all_indicators(self, include_dormant: bool = False) -> list[dict[str, Any]]:
        """Get all indicators."""
        results = []
        for ind_id, ind in self._indicators.items():
            if not include_dormant and ind.get("_status") == "dormant":
                continue
            result = {k: v for k, v in ind.items() if not k.startswith("_")}
            result["hit_count"] = self._hit_counts.get(ind_id, 0)
            result["status"] = ind.get("_status", "active")
            results.append(result)
        return results

    def age_indicators(self) -> int:
        """Move old indicators to dormant status. Returns count aged."""
        now = time.time()
        aged = 0
        for ind_id, ind in self._indicators.items():
            if ind.get("_status") != "active":
                continue
            added = ind.get("_added_at", now)
            if now - added >= self._dormant_age:
                ind["_status"] = "dormant"
                aged += 1
        if aged:
            logger.info("Aged %d indicators to dormant", aged)
        return aged

    def reactivate(self, indicator_id: str) -> bool:
        """Reactivate a dormant indicator. Returns True if reactivated."""
        ind = self._indicators.get(indicator_id)
        if ind and ind.get("_status") == "dormant":
            ind["_status"] = "active"
            ind["_added_at"] = time.time()
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Registry statistics."""
        active = sum(1 for i in self._indicators.values() if i.get("_status") == "active")
        dormant = sum(1 for i in self._indicators.values() if i.get("_status") == "dormant")

        # Top tactics
        tactic_counts: Counter[str] = Counter()
        for ind in self._indicators.values():
            if ind.get("_status") == "dormant":
                continue
            try:
                pattern = json.loads(ind.get("pattern", "{}"))
                tactic = pattern.get("mitre_tactic", "")
                if tactic:
                    tactic_counts[tactic] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        timestamps = [
            ind.get("created", "")
            for ind in self._indicators.values()
            if ind.get("created")
        ]

        return {
            "total": len(self._indicators),
            "active": active,
            "dormant": dormant,
            "newest": max(timestamps) if timestamps else None,
            "oldest": min(timestamps) if timestamps else None,
            "top_tactics": dict(tactic_counts.most_common(10)),
        }
