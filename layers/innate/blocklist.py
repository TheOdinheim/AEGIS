"""
Scanner 2 — Blocklist Lookup (Defensin Analog)

Hash-based lookup against curated blocklists of known malicious payloads,
banned phrases, and prohibited content categories. Uses a set for O(1)
lookup on normalized payloads. Bootstrap implementation; production adds
Bloom filter for probabilistic first-pass.

ASSUMED-BREACH POSTURE: This scanner assumes L1 Barrier has been bypassed
and the regex engine may have been evaded. The blocklist file itself could
be tampered with if the filesystem is compromised — entries could be removed
to create blind spots. Even a complete blocklist bypass is survivable:
L3 Adaptive provides semantic detection independent of exact-match lookups.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from aegis.models.scan_result import ScanResult, ThreatCategory


class BlocklistScanner:
    """Exact-match blocklist scanner using normalized text comparison.

    Entries are stored as lowercase stripped strings for case-insensitive
    matching. SHA-256 hashes of entries are also stored for hash-based
    substring matching on longer inputs.
    """

    def __init__(self, blocklist_file: str | Path | None = None):
        self._entries: set[str] = set()
        self._hashes: set[str] = set()
        if blocklist_file is not None:
            self.load(blocklist_file)

    def load(self, blocklist_file: str | Path) -> None:
        """Load blocklist entries from a text file (one per line)."""
        path = Path(blocklist_file)
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                normalized = line.lower().strip()
                self._entries.add(normalized)
                self._hashes.add(
                    hashlib.sha256(normalized.encode()).hexdigest()
                )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    async def scan(self, text: str) -> ScanResult:
        """Scan text against the blocklist.

        Checks both exact match (full text) and substring containment
        (each blocklist entry is checked as substring of the input).
        """
        start = time.perf_counter()
        try:
            normalized = text.lower().strip()
            matched: list[str] = []

            # Check if any blocklist entry appears as substring in input
            for entry in self._entries:
                if entry in normalized:
                    # Truncate long entries for the match report
                    display = entry[:80] + "..." if len(entry) > 80 else entry
                    matched.append(f"blocklist: {display}")

            elapsed_ms = (time.perf_counter() - start) * 1000

            if matched:
                confidence = min(0.70 + 0.10 * len(matched), 1.0)
                return ScanResult(
                    scanner_id="blocklist",
                    is_threat=True,
                    confidence=confidence,
                    threat_category=ThreatCategory.BLOCKLIST_MATCH,
                    matched_patterns=matched,
                    latency_ms=elapsed_ms,
                )

            return ScanResult(
                scanner_id="blocklist",
                is_threat=False,
                confidence=0.0,
                latency_ms=elapsed_ms,
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ScanResult(
                scanner_id="blocklist",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed_ms,
            )
