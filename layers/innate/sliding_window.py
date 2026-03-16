"""
Scanner 7 — Sliding Window Scanner (Padding Dilution Defense)

Defends against white-box Finding 1: injection text buried in 80+ words
of benign padding drops DeBERTa below its block threshold via attention
dilution. The sliding window scanner segments long inputs into overlapping
windows and runs each through the regex engine, catching injections that
full-text scanning misses when diluted by surrounding benign content.

Algorithm:
  1. Split input into words
  2. If word count <= min_text_length: skip (full-text scan sufficient)
  3. Create overlapping windows (default 20 words, 10-word overlap)
  4. Run each window through RegexEngine.scan()
  5. If ANY window triggers: return highest-confidence result

Performance: O(n) where n = word count. <1ms for typical prompts.

ASSUMED-BREACH POSTURE: This scanner assumes L1 Barrier and the full-text
regex scan have been bypassed via padding dilution. Long benign text
wrapping a short injection payload is a documented DeBERTa evasion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.models.scan_result import ScanResult, ThreatCategory

if TYPE_CHECKING:
    from aegis.layers.innate.regex_engine import RegexEngine


@dataclass
class SlidingWindowResult:
    """Result from sliding window analysis."""
    triggered: bool = False
    confidence: float = 0.0
    window_index: int = -1
    window_text: str = ""
    total_windows: int = 0
    threat_category: str = ""
    scan_result: ScanResult | None = None


class SlidingWindowScanner:
    """Sliding window scanner for padding dilution defense.

    Segments long inputs into overlapping windows and scans each through
    the regex engine. Catches injection buried in benign padding that
    full-text regex misses due to DeBERTa attention dilution.
    """

    def __init__(
        self,
        window_size: int = 20,
        overlap: int = 10,
        min_text_length: int = 50,
    ) -> None:
        self._window_size = window_size
        self._overlap = overlap
        self._min_text_length = min_text_length

    async def scan(
        self,
        text: str,
        regex_engine: "RegexEngine",
    ) -> SlidingWindowResult:
        """Scan text using overlapping windows through the regex engine.

        Args:
            text: Input text to scan.
            regex_engine: The L2 regex engine instance.

        Returns:
            SlidingWindowResult with the highest-confidence window match.
        """
        words = text.split()

        if len(words) <= self._min_text_length:
            return SlidingWindowResult(total_windows=0)

        # Build overlapping windows
        step = self._window_size - self._overlap
        if step < 1:
            step = 1

        windows: list[tuple[int, str]] = []
        i = 0
        while i < len(words):
            end = min(i + self._window_size, len(words))
            window_text = " ".join(words[i:end])
            windows.append((len(windows), window_text))
            i += step
            if end == len(words):
                break

        # Scan each window
        best_result: SlidingWindowResult | None = None

        for idx, window_text in windows:
            result = await regex_engine.scan(window_text)
            if result.is_threat:
                if best_result is None or result.confidence > best_result.confidence:
                    # Boost confidence for windowed detection — if full-text
                    # didn't catch it but a window did, this is a padding
                    # dilution attack. Set confidence to 0.92 minimum.
                    boosted_confidence = max(result.confidence, 0.92)
                    best_result = SlidingWindowResult(
                        triggered=True,
                        confidence=boosted_confidence,
                        window_index=idx,
                        window_text=window_text,
                        total_windows=len(windows),
                        threat_category=result.threat_category.value
                        if result.threat_category
                        else "",
                        scan_result=result,
                    )

        if best_result is not None:
            return best_result

        return SlidingWindowResult(total_windows=len(windows))
