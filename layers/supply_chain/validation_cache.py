"""
Supply Chain Validation Cache — Extension 3.4

Stores all supply chain validation results (MPV, SPA, DCA) in L4's immune
memory with LRU eviction, TTL-based expiry, and content-hash-based
change detection.

Enables:
- Fast-path cache hits for unchanged components
- Retroactive threat checking against cached approvals
- Re-validation scheduling via get_all_approved()

ASSUMED-BREACH POSTURE: A compromised cache could suppress detection of
newly-discovered threats in previously-approved components. The cache
is read-through — expired or invalidated entries force re-validation.
Retroactive checks ensure new threat intelligence is applied to all
cached approvals.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CachedValidation:
    component_id: str
    component_type: str  # "model", "skill", "dependency"
    validation_result: Any
    validated_at: float
    valid_until: float
    status: str  # "approved", "flagged", "rejected"
    content_hash: str
    last_accessed: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.valid_until


class SupplyChainValidationCache:
    """LRU cache for supply chain validation results.

    Thread-safe. Supports TTL-based expiry, content-hash change detection,
    retroactive threat checking, and LRU eviction.
    """

    def __init__(self, max_entries: int = 10000, default_ttl: int = 604800) -> None:
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, CachedValidation] = OrderedDict()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def cache_result(
        self,
        component_id: str,
        component_type: str,
        result: Any,
        content_hash: str,
        status: str = "approved",
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache a validation result with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        now = time.time()

        entry = CachedValidation(
            component_id=component_id,
            component_type=component_type,
            validation_result=result,
            validated_at=now,
            valid_until=now + ttl,
            status=status,
            content_hash=content_hash,
            last_accessed=now,
        )

        with self._lock:
            # Remove old entry if exists (to refresh LRU position)
            self._cache.pop(component_id, None)
            self._cache[component_id] = entry

            # LRU eviction
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def get_cached(
        self,
        component_id: str,
        content_hash: str | None = None,
    ) -> CachedValidation | None:
        """Get cached result. Returns None if not cached, expired, or hash changed."""
        with self._lock:
            entry = self._cache.get(component_id)
            if entry is None:
                return None

            # Expired
            if entry.is_expired:
                return None

            # Content changed
            if content_hash is not None and entry.content_hash != content_hash:
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(component_id)
            entry.last_accessed = time.time()
            return entry

    def invalidate(self, component_id: str) -> bool:
        """Manually invalidate a cached result."""
        with self._lock:
            return self._cache.pop(component_id, None) is not None

    def invalidate_by_type(self, component_type: str) -> int:
        """Invalidate all cached results of a type."""
        with self._lock:
            to_remove = [
                cid for cid, entry in self._cache.items()
                if entry.component_type == component_type
            ]
            for cid in to_remove:
                del self._cache[cid]
            return len(to_remove)

    def get_all_approved(self, component_type: str | None = None) -> list[CachedValidation]:
        """Get all currently approved components."""
        with self._lock:
            results = []
            for entry in self._cache.values():
                if entry.status != "approved":
                    continue
                if component_type is not None and entry.component_type != component_type:
                    continue
                results.append(entry)
            return results

    def get_all_entries(self) -> list[CachedValidation]:
        """Get all cached entries (for admin/debug)."""
        with self._lock:
            return list(self._cache.values())

    def retroactive_check(self, threat_signature: str) -> list[CachedValidation]:
        """Check all cached approvals against a new threat signature.

        Performs a simple substring match against the component_id and
        the string representation of the validation result. In production,
        this would use semantic similarity or pattern matching.
        """
        sig_lower = threat_signature.lower()
        matches: list[CachedValidation] = []

        with self._lock:
            for entry in self._cache.values():
                if entry.status != "approved":
                    continue
                # Check component_id
                if sig_lower in entry.component_id.lower():
                    matches.append(entry)
                    continue
                # Check result string representation
                result_str = str(entry.validation_result).lower()
                if sig_lower in result_str:
                    matches.append(entry)

        return matches

    def update_status(self, component_id: str, new_status: str) -> bool:
        """Update the status of a cached entry."""
        with self._lock:
            entry = self._cache.get(component_id)
            if entry is None:
                return False
            entry.status = new_status
            return True


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()
