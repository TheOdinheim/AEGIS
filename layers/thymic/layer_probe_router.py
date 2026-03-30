"""
Layer Probe Router — Routes probes through the AEGIS pipeline.

Phase A implements Full Pipeline Mode only: each probe is submitted as a
simulated request through the ASGI transport. Probes carry X-AEGIS-TVE-Probe
headers and are intercepted before model forwarding (zero upstream calls).

CRITICAL CONSTRAINTS:
- Zero upstream model calls from TVE probes
- No production metric contamination
- Probes must NOT appear in real audit logs or rate limit counters
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from aegis.layers.thymic.attack_profile_library import Probe

logger = logging.getLogger(__name__)

# Header used to mark TVE probes
TVE_PROBE_HEADER = "x-aegis-tve-probe"


@dataclass
class ProbeResult:
    """Result of routing a probe through the AEGIS pipeline."""

    probe_id: str
    probe_tier: int
    detected: bool
    detection_layers: list[str] = field(default_factory=list)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    missed_layers: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    false_positive: bool = False
    response_status: int = 0
    error: str | None = None


class LayerProbeRouter:
    """Routes probes through the AEGIS pipeline via ASGI transport.

    Uses httpx.AsyncClient with ASGITransport to route probes directly
    through the FastAPI app without making external HTTP calls.
    """

    def __init__(
        self,
        app: object | None = None,
        api_key: str = "",
        concurrency: int = 5,
    ) -> None:
        self._app = app
        self._api_key = api_key
        self._concurrency = concurrency

    async def route_probe(self, probe: Probe) -> ProbeResult:
        """Route a single probe through the ASGI pipeline."""
        import httpx

        if self._app is None:
            return ProbeResult(
                probe_id=probe.id,
                probe_tier=probe.tier,
                detected=False,
                error="No app configured",
            )

        start = time.perf_counter()

        # Build request body
        messages: list[dict[str, str]] = []
        if probe.conversation_sequence and probe.tier == 4:
            messages = list(probe.conversation_sequence)
        else:
            messages = [{"role": "user", "content": probe.text}]

        body = {
            "model": "gpt-4",
            "messages": messages,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            TVE_PROBE_HEADER: probe.id,
        }

        try:
            transport = httpx.ASGITransport(app=self._app)  # type: ignore[arg-type]
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://aegis-tve-internal",
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=body,
                    headers=headers,
                )

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Analyze response
            detected = resp.status_code in (403, 429)
            detection_layers: list[str] = []
            confidence_scores: dict[str, float] = {}

            if detected:
                # Try to extract detection info from response body
                try:
                    resp_body = resp.json()
                    error_msg = resp_body.get("error", {}).get("message", "")
                    if "innate" in error_msg:
                        detection_layers.append("L2")
                    if "adaptive" in error_msg:
                        detection_layers.append("L3")
                    if "policy" in error_msg:
                        detection_layers.append("L6")
                    if "manipulation" in error_msg:
                        detection_layers.append("L3")
                    if "barrier" in error_msg or "rate" in error_msg.lower():
                        detection_layers.append("L1")
                    if "multimodal" in error_msg:
                        detection_layers.append("L2")
                    # Default to L2 if we can't determine the layer
                    if not detection_layers:
                        detection_layers.append("L2")
                except Exception:
                    detection_layers.append("unknown")

            # Determine missed layers (expected but didn't catch)
            missed_layers: list[str] = []
            if probe.expected_detection_layer and not detected:
                missed_layers.append(probe.expected_detection_layer)

            # False positive: Tier 5 benign probe was flagged
            false_positive = probe.tier == 5 and detected

            return ProbeResult(
                probe_id=probe.id,
                probe_tier=probe.tier,
                detected=detected,
                detection_layers=detection_layers,
                confidence_scores=confidence_scores,
                missed_layers=missed_layers,
                latency_ms={"total": elapsed_ms},
                false_positive=false_positive,
                response_status=resp.status_code,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Probe routing error for %s: %s", probe.id, e)
            return ProbeResult(
                probe_id=probe.id,
                probe_tier=probe.tier,
                detected=False,
                latency_ms={"total": elapsed_ms},
                error=str(e),
            )

    async def route_batch(
        self,
        probes: list[Probe],
        concurrency: int | None = None,
    ) -> list[ProbeResult]:
        """Route batch with configurable concurrency."""
        max_concurrent = concurrency or self._concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[ProbeResult] = []

        async def _route_with_semaphore(probe: Probe) -> ProbeResult:
            async with semaphore:
                return await self.route_probe(probe)

        tasks = [_route_with_semaphore(p) for p in probes]
        results = await asyncio.gather(*tasks)
        return list(results)
