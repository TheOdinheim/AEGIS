"""
L2 — Innate Detection (Pattern Recognition Receptors / Natural Killer Cells)

Synchronous fast-path scanning. Must complete in <5ms. Five scanners execute
in parallel via asyncio.gather: regex engine, blocklist, token guard, PII
regex, and canary token verifier (NK cell analog).

ASSUMED-BREACH POSTURE: This layer assumes L1 Barrier has been bypassed.
Malformed requests, missing auth tokens, exceeded rate limits, and invalid
schemas may all be present. Every scanner independently validates the raw
input regardless of what L1 claims to have checked. A compromised L1 could
forge RequestContext metadata — innate scanners examine the actual payload.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from aegis.config import CanaryConfig, InnateConfig
from aegis.models.request_context import RequestContext
from aegis.models.scan_result import InnateScanReport, ScanResult, ThreatCategory

from aegis.layers.innate.regex_engine import RegexEngine
from aegis.layers.innate.blocklist import BlocklistScanner
from aegis.layers.innate.token_guard import TokenGuard
from aegis.layers.innate.pii_regex import PIIRegexScanner
from aegis.layers.innate.canary_verifier import CanaryVerifier
from aegis.layers.innate.sliding_window import SlidingWindowScanner


class InnateDetectionLayer:
    """L2 Innate Detection — orchestrates all scanners in parallel.

    Runs regex engine, blocklist, token guard, PII regex, and canary
    verifier concurrently via asyncio.gather. Aggregates results into
    an InnateScanReport with block/alert decisions based on configured
    thresholds.
    """

    def __init__(
        self,
        config: InnateConfig,
        data_dir: str | Path | None = None,
        canary_config: CanaryConfig | None = None,
    ):
        self._config = config
        data_path = Path(data_dir) if data_dir else Path("data")

        self._regex_engine = RegexEngine(data_path / config.regex_pattern_file.split("/")[-1])
        self._blocklist = BlocklistScanner(data_path / config.blocklist_file.split("/")[-1])
        self._token_guard = TokenGuard(max_tokens=128_000)
        self._pii_regex = PIIRegexScanner()
        self._canary_verifier = CanaryVerifier(canary_config)
        self._sliding_window = SlidingWindowScanner()

    @property
    def regex_engine(self) -> RegexEngine:
        return self._regex_engine

    @property
    def blocklist(self) -> BlocklistScanner:
        return self._blocklist

    @property
    def canary_verifier(self) -> CanaryVerifier:
        return self._canary_verifier

    async def scan(self, context: RequestContext) -> InnateScanReport:
        """Run all innate scanners in parallel and produce an aggregated report.

        Target: <5ms total. All scanners run concurrently via asyncio.gather.
        """
        start = time.perf_counter()
        prompt = context.prompt_text

        # Get canary token from context metadata (injected by main.py)
        expected_canary = context.metadata.get("canary_token")

        # Extract system prompt for canary verification
        system_prompt = ""
        for msg in context.messages:
            if msg.role == "system" and msg.content:
                system_prompt = msg.content
                break

        # Run all scanners concurrently (including canary verifier)
        results: list[ScanResult] = await asyncio.gather(
            self._regex_engine.scan(prompt),
            self._blocklist.scan(prompt),
            self._token_guard.scan(context.messages),
            self._pii_regex.scan(prompt),
            self._canary_verifier.scan_input(system_prompt, expected_canary)
            if self._canary_verifier.enabled and system_prompt
            else _noop_scan("canary_verifier"),
        )

        # Run sliding window scanner on long inputs for padding dilution defense
        # This catches injections buried in benign padding that full-text regex misses
        regex_caught = any(
            r.scanner_id == "regex_engine" and r.is_threat for r in results
        )
        sw_result = await self._sliding_window.scan(prompt, self._regex_engine)
        if sw_result.triggered and not regex_caught and sw_result.scan_result:
            # Padding dilution detected: window found injection that full-text missed
            results.append(ScanResult(
                scanner_id="sliding_window",
                is_threat=True,
                confidence=sw_result.confidence,
                threat_category=sw_result.scan_result.threat_category,
                matched_patterns=[
                    f"sliding_window[{sw_result.window_index}/{sw_result.total_windows}]: "
                    f"padding dilution detected"
                ] + sw_result.scan_result.matched_patterns,
                latency_ms=0.0,
            ))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Aggregate
        max_confidence = 0.0
        threat_categories: list[ThreatCategory] = []
        should_block = False

        for result in results:
            if result.is_threat:
                max_confidence = max(max_confidence, result.confidence)
                if result.threat_category not in threat_categories:
                    threat_categories.append(result.threat_category)
                if result.confidence >= self._config.block_threshold:
                    should_block = True

        return InnateScanReport(
            request_id=context.request_id,
            scanner_results=results,
            should_block=should_block,
            max_confidence=max_confidence,
            total_latency_ms=elapsed_ms,
            threat_categories=threat_categories,
        )


async def _noop_scan(scanner_id: str) -> ScanResult:
    """No-op scan result for disabled scanners."""
    return ScanResult(
        scanner_id=scanner_id,
        is_threat=False,
        confidence=0.0,
        latency_ms=0.0,
    )
