"""
Evasion engine — runs adversarial attacks against the live AEGIS pipeline.

Sends each attack through AEGIS, records whether it was blocked or allowed,
which layer caught it, at what confidence, and the full response. Produces
an EvasionReport with per-layer and per-technique breakdowns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from red_team import (
    Attack,
    EvasionReport,
    EvasionResult,
)

logger = logging.getLogger(__name__)


class EvasionEngine:
    """Runs attacks against a live AEGIS instance and classifies results.

    For each attack, sends through AEGIS via HTTP, then classifies the outcome
    as blocked (403) or evaded (200). Extracts detection metadata from response
    headers and body when available.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        concurrency: int = 5,
    ) -> None:
        self._timeout = timeout
        self._concurrency = concurrency

    async def run_evasion_test(
        self,
        attacks: list[Attack],
        aegis_url: str,
        api_key: str,
    ) -> EvasionReport:
        """Run all attacks against AEGIS and produce an evasion report.

        Args:
            attacks: List of Attack objects to test.
            aegis_url: Base URL of the AEGIS instance (e.g., http://localhost:8000).
            api_key: Valid AEGIS API key for authentication.

        Returns:
            EvasionReport with per-layer and per-technique breakdowns.
        """
        sem = asyncio.Semaphore(self._concurrency)
        results: list[EvasionResult] = []

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [
                self._run_single(client, attack, aegis_url, api_key, sem)
                for attack in attacks
            ]
            results = await asyncio.gather(*tasks)

        return self._build_report(results)

    async def _run_single(
        self,
        client: httpx.AsyncClient,
        attack: Attack,
        aegis_url: str,
        api_key: str,
        sem: asyncio.Semaphore,
    ) -> EvasionResult:
        """Execute a single attack and record the result."""
        async with sem:
            start = time.monotonic()
            try:
                if isinstance(attack.payload, list):
                    # Multi-turn attack: send each message sequentially
                    return await self._run_multi_turn(
                        client, attack, aegis_url, api_key, start
                    )
                else:
                    return await self._run_single_turn(
                        client, attack, aegis_url, api_key, start
                    )
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                logger.warning("Attack %s failed: %s", attack.attack_id, e)
                return EvasionResult(
                    attack_id=attack.attack_id,
                    target_layer=attack.target_layer.value,
                    evasion_technique=attack.evasion_technique.value,
                    was_blocked=False,
                    caught_by_layer=None,
                    confidence=0.0,
                    response_preview=f"Error: {e}",
                    latency_ms=elapsed,
                    error=str(e),
                )

    async def _run_single_turn(
        self,
        client: httpx.AsyncClient,
        attack: Attack,
        aegis_url: str,
        api_key: str,
        start: float,
    ) -> EvasionResult:
        """Send a single-turn attack."""
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": attack.payload}],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = await client.post(
            f"{aegis_url}/v1/chat/completions",
            json=body,
            headers=headers,
        )
        elapsed = (time.monotonic() - start) * 1000
        return self._classify_response(attack, resp, elapsed)

    async def _run_multi_turn(
        self,
        client: httpx.AsyncClient,
        attack: Attack,
        aegis_url: str,
        api_key: str,
        start: float,
    ) -> EvasionResult:
        """Send a multi-turn attack sequence, one message at a time."""
        messages: list[dict[str, str]] = []
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_resp = None

        for turn_payload in attack.payload:
            messages.append({"role": "user", "content": turn_payload})
            body = {"model": "gpt-4", "messages": messages}
            resp = await client.post(
                f"{aegis_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
            last_resp = resp
            if resp.status_code == 403:
                # Blocked on this turn
                break
            # Add assistant response for next turn context
            try:
                resp_data = resp.json()
                assistant_msg = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                messages.append({"role": "assistant", "content": assistant_msg})
            except Exception:
                messages.append({"role": "assistant", "content": ""})

        elapsed = (time.monotonic() - start) * 1000
        if last_resp is None:
            return EvasionResult(
                attack_id=attack.attack_id,
                target_layer=attack.target_layer.value,
                evasion_technique=attack.evasion_technique.value,
                was_blocked=False,
                caught_by_layer=None,
                confidence=0.0,
                response_preview="No response received",
                latency_ms=elapsed,
                error="No messages in payload",
            )
        return self._classify_response(attack, last_resp, elapsed)

    def _classify_response(
        self,
        attack: Attack,
        resp: httpx.Response,
        latency_ms: float,
    ) -> EvasionResult:
        """Classify an HTTP response as blocked or evaded."""
        was_blocked = resp.status_code == 403
        caught_by_layer = None
        confidence = 0.0
        response_preview = ""

        try:
            data = resp.json()
            if was_blocked:
                # Extract detection metadata from error response
                error = data.get("error", {})
                detail = error.get("message", "") if isinstance(error, dict) else str(data.get("detail", ""))
                caught_by_layer = self._extract_layer(detail)
                confidence = self._extract_confidence(data)
                response_preview = detail[:200]
            else:
                # Attack evaded — capture response preview
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    response_preview = content[:200]
        except Exception:
            response_preview = resp.text[:200]

        return EvasionResult(
            attack_id=attack.attack_id,
            target_layer=attack.target_layer.value,
            evasion_technique=attack.evasion_technique.value,
            was_blocked=was_blocked,
            caught_by_layer=caught_by_layer,
            confidence=confidence,
            response_preview=response_preview,
            latency_ms=latency_ms,
            status_code=resp.status_code,
        )

    @staticmethod
    def _extract_layer(detail: str) -> str | None:
        """Extract which layer caught the attack from the error detail."""
        detail_lower = detail.lower()
        if "innate" in detail_lower or "regex" in detail_lower or "blocklist" in detail_lower:
            return "L2"
        if "adaptive" in detail_lower or "classifier" in detail_lower or "semantic" in detail_lower:
            return "L3"
        if "output" in detail_lower or "pii" in detail_lower or "redact" in detail_lower:
            return "L5"
        if "policy" in detail_lower:
            return "L6"
        if "rate" in detail_lower or "throttl" in detail_lower:
            return "L1"
        if "multi-turn" in detail_lower or "escalation" in detail_lower:
            return "L3"
        return "unknown"

    @staticmethod
    def _extract_confidence(data: dict[str, Any]) -> float:
        """Extract detection confidence from the response data."""
        # AEGIS may include confidence in the error response metadata
        error = data.get("error", {})
        if isinstance(error, dict):
            meta = error.get("metadata", {})
            if isinstance(meta, dict):
                return float(meta.get("confidence", 0.0))
        return 0.0

    def _build_report(self, results: list[EvasionResult]) -> EvasionReport:
        """Build aggregated EvasionReport from individual results."""
        total = len(results)
        blocked = sum(1 for r in results if r.was_blocked)
        evaded = total - blocked
        evasion_rate = evaded / total if total > 0 else 0.0

        # Per-layer breakdown
        per_layer: dict[str, dict[str, int]] = {}
        for r in results:
            layer = r.target_layer
            if layer not in per_layer:
                per_layer[layer] = {"total": 0, "blocked": 0, "evaded": 0}
            per_layer[layer]["total"] += 1
            if r.was_blocked:
                per_layer[layer]["blocked"] += 1
            else:
                per_layer[layer]["evaded"] += 1

        # Per-technique breakdown
        per_technique: dict[str, dict[str, int]] = {}
        for r in results:
            tech = r.evasion_technique
            if tech not in per_technique:
                per_technique[tech] = {"total": 0, "blocked": 0, "evaded": 0}
            per_technique[tech]["total"] += 1
            if r.was_blocked:
                per_technique[tech]["blocked"] += 1
            else:
                per_technique[tech]["evaded"] += 1

        # Hardest to detect: evaded attacks first, then lowest-confidence blocks
        sorted_results = sorted(
            results,
            key=lambda r: (r.was_blocked, r.confidence),
        )
        hardest = sorted_results[:10]

        return EvasionReport(
            total_attacks=total,
            blocked=blocked,
            evaded=evaded,
            evasion_rate=evasion_rate,
            per_layer_results=per_layer,
            per_technique_results=per_technique,
            hardest_to_detect=hardest,
            all_results=results,
        )
