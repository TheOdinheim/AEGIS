"""Server-Sent Events stream for dashboard real-time updates.

Subscribes to the AEGIS event bus and forwards relevant events:
- threat_detected → detection events
- circuit_breaker → breaker state changes
- Periodic heartbeat every 5s with TLI and RPS
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard-sse"])


def _is_sse_authenticated(request: Request) -> bool:
    """Check API key for SSE endpoint."""
    import main as m
    if not m._config or not m._config.api_key:
        return False
    auth = request.headers.get("authorization", "")
    token = ""
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    if not token:
        token = (request.headers.get("x-api-key")
                 or request.headers.get("X-Api-Key") or "").strip()
    # Also accept query param for EventSource (which can't set headers easily)
    if not token:
        token = request.query_params.get("token", "").strip()
    if not token:
        return False
    return hmac.compare_digest(token, m._config.api_key)


async def _event_generator(request: Request) -> AsyncGenerator[dict, None]:
    """Generate SSE events from the event bus + heartbeat."""
    import main as m

    event_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    # Subscribe to event bus channels
    subscribed = False
    if m._event_bus:
        try:
            async def _on_threat(event):
                try:
                    event_queue.put_nowait({
                        "event": "detection",
                        "data": json.dumps({
                            "type": "detection",
                            "data": event.payload if hasattr(event, "payload") else {},
                        }),
                    })
                except asyncio.QueueFull:
                    pass

            async def _on_breaker(event):
                try:
                    event_queue.put_nowait({
                        "event": "breaker",
                        "data": json.dumps({
                            "type": "breaker",
                            "data": event.payload if hasattr(event, "payload") else {},
                        }),
                    })
                except asyncio.QueueFull:
                    pass

            await m._event_bus.subscribe("threat_detected", _on_threat)
            await m._event_bus.subscribe("circuit_breaker", _on_breaker)
            subscribed = True
        except Exception as e:
            logger.debug("SSE event bus subscription failed: %s", e)

    heartbeat_interval = 5.0
    last_heartbeat = 0.0

    try:
        while True:
            if await request.is_disconnected():
                break

            # Drain queued events
            try:
                event = event_queue.get_nowait()
                yield event
                continue
            except asyncio.QueueEmpty:
                pass

            # Heartbeat
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now

                tli = "GREEN"
                rps = 0.0
                if m._policy:
                    tli_val = m._policy.threat_level.value
                    tli_names = {1: "GREEN", 2: "BLUE", 3: "YELLOW", 4: "ORANGE", 5: "RED"}
                    tli = tli_names.get(tli_val, "GREEN")

                if m._dashboard_metrics_buffer:
                    ts = m._dashboard_metrics_buffer.get_timeseries("requests_total", "1h")
                    if ts:
                        rps = ts[-1]["value"]

                yield {
                    "event": "heartbeat",
                    "data": json.dumps({
                        "type": "heartbeat",
                        "data": {"tli": tli, "rps": round(rps, 2)},
                    }),
                }

            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass


@router.get("/dashboard/events")
async def dashboard_sse(request: Request):
    """Server-Sent Events stream for real-time dashboard updates."""
    if not _is_sse_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"},
        )

    return EventSourceResponse(_event_generator(request))
