"""
Request Enrichment Middleware — Builds RequestContext from raw HTTP request.

Extracts and validates identity metadata, parses the request body, computes
API key hash, resolves source IP (handling X-Forwarded-For), and initializes
rate limit state.

ASSUMED-BREACH POSTURE: Every field extracted from the HTTP request is
adversarial. X-Forwarded-For can be spoofed. Content-Type can lie. The
request body may be malformed JSON designed to crash the parser. API keys
may be stolen. This middleware performs minimal validation — enough to
construct a RequestContext — and leaves deep validation to L1 Barrier and
subsequent layers. If request parsing fails, the middleware returns 400
without forwarding to any layer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from starlette.requests import Request

logger = logging.getLogger(__name__)


def extract_source_ip(request: Request) -> str:
    """Extract the real client IP, handling X-Forwarded-For.

    Only trusts the last hop in XFF to mitigate spoofing.
    Falls back to request.client.host.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # Take the rightmost (closest proxy) IP — leftmost is client-controlled
        ips = [ip.strip() for ip in xff.split(",")]
        if ips:
            return ips[-1]
    if request.client:
        return request.client.host
    return "0.0.0.0"


def extract_headers(request: Request) -> dict[str, str]:
    """Extract relevant headers as a flat dict."""
    headers = {
        "authorization": request.headers.get("authorization", ""),
        "x-api-key": request.headers.get("x-api-key", ""),
        "x-session-id": request.headers.get("x-session-id", ""),
        "content-type": request.headers.get("content-type", ""),
        "x-forwarded-for": request.headers.get("x-forwarded-for", ""),
    }
    # Optional headers — only include if present
    for key in ("x-aegis-agent-id", "x-aegis-tve-probe"):
        val = request.headers.get(key)
        if val:
            headers[key] = val
    return headers


async def parse_request_body(request: Request) -> dict[str, Any]:
    """Parse the JSON request body. Returns empty dict on parse failure."""
    try:
        raw = await request.body()
        if not raw:
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Failed to parse request body: %s", e)
        return {}


async def get_raw_body(request: Request) -> bytes:
    """Get the raw request body bytes."""
    try:
        return await request.body()
    except Exception:
        return b""
