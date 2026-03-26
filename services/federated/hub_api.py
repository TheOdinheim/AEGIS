"""
Federation Hub API — Distribution endpoints for federated threat intelligence.

Provides STIX indicator submission, retrieval, node heartbeat, and network
stats. All endpoints require API key authentication.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/federation", tags=["federation"])


def _get_auth_and_components(request: Request) -> tuple[JSONResponse | None, Any, Any, Any]:
    """Authenticate and return (error_response, registry, node_registry, federated).

    Returns (None, registry, nodes, fed) on success, or (401_response, ...) on failure.
    """
    from aegis.dashboard import _get_main

    m = _get_main()
    if not m._config or not m._config.api_key:
        return JSONResponse(status_code=401, content={"error": "Authentication required"}), None, None, None

    auth = request.headers.get("authorization", "")
    token = ""
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    if not token:
        token = (request.headers.get("x-api-key") or request.headers.get("X-Api-Key") or "").strip()
    if not token:
        return JSONResponse(status_code=401, content={"error": "Authentication required"}), None, None, None

    import hmac
    if not hmac.compare_digest(token, m._config.api_key):
        return JSONResponse(status_code=401, content={"error": "Authentication required"}), None, None, None

    indicator_registry = getattr(m, "_indicator_registry", None)
    node_registry = getattr(m, "_node_registry", None)
    federated = getattr(m, "_federated", None)

    return None, indicator_registry, node_registry, federated


# ---------------------------------------------------------------------------
# POST /v1/federation/indicators
# ---------------------------------------------------------------------------

@router.post("/indicators")
async def submit_indicator(request: Request) -> JSONResponse:
    """Submit a STIX indicator to the federation hub."""
    err, registry, node_registry, federated = _get_auth_and_components(request)
    if err:
        return err

    if not registry:
        return JSONResponse(status_code=503, content={"error": "Federation not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    # Validate required STIX fields
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "Body must be a JSON object"})
    if body.get("type") != "indicator":
        return JSONResponse(status_code=400, content={"error": "Object type must be 'indicator'"})
    if not body.get("id", "").startswith("indicator--"):
        return JSONResponse(status_code=400, content={"error": "Invalid indicator ID format"})
    if not body.get("pattern"):
        return JSONResponse(status_code=400, content={"error": "Indicator must have a pattern"})

    # Screen indicator through immune response if available
    from aegis.dashboard import _get_main
    m = _get_main()
    immune = getattr(m, "_federation_immune", None)
    source_node = body.get("x_aegis_source_node", "unknown")
    if immune:
        rep_score = immune.screen_indicator(body, source_node)
        if not rep_score.accepted:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "rejected",
                    "score": rep_score.overall,
                    "reasons": rep_score.reasons,
                },
            )

    try:
        ind_id = registry.add_indicator(body)
    except Exception as e:
        logger.error("Failed to add indicator: %s", e)
        return JSONResponse(status_code=500, content={"error": "Failed to store indicator"})

    # Publish event
    if m._event_bus:
        try:
            from aegis.services.event_bus import Event
            import asyncio
            asyncio.ensure_future(m._event_bus.publish(
                "indicator_generated",
                Event(channel="indicator_generated", payload={"indicator_id": ind_id}),
            ))
        except Exception:
            pass

    return JSONResponse(
        status_code=201,
        content={"status": "accepted", "indicator_id": ind_id},
    )


# ---------------------------------------------------------------------------
# GET /v1/federation/indicators
# ---------------------------------------------------------------------------

@router.get("/indicators")
async def list_indicators(
    request: Request,
    since: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> JSONResponse:
    """List STIX indicators, optionally filtered by timestamp."""
    err, registry, _, _ = _get_auth_and_components(request)
    if err:
        return err

    if not registry:
        return JSONResponse(content={"indicators": [], "total": 0})

    if since:
        indicators = registry.get_indicators_since(since)
    else:
        indicators = registry.get_all_indicators()

    # Strip embedding vectors from response (they're large)
    clean = []
    for ind in indicators[:limit]:
        clean_ind = {k: v for k, v in ind.items() if k != "pattern"}
        # Include pattern metadata without the raw embedding
        try:
            import json
            pattern = json.loads(ind.get("pattern", "{}"))
            clean_ind["pattern_meta"] = {
                "mitre_tactic": pattern.get("mitre_tactic", ""),
                "detection_layer": pattern.get("detection_layer", ""),
                "confidence": pattern.get("confidence", 0),
                "attack_category": pattern.get("attack_category", ""),
                "embedding_dim": len(pattern.get("threat_embedding", [])),
            }
        except (json.JSONDecodeError, TypeError):
            pass
        clean.append(clean_ind)

    return JSONResponse(content={"indicators": clean, "total": len(indicators)})


# ---------------------------------------------------------------------------
# GET /v1/federation/indicators/{indicator_id}
# ---------------------------------------------------------------------------

@router.get("/indicators/{indicator_id}")
async def get_indicator(request: Request, indicator_id: str) -> JSONResponse:
    """Retrieve a single STIX indicator by ID."""
    err, registry, _, _ = _get_auth_and_components(request)
    if err:
        return err

    if not registry:
        return JSONResponse(status_code=503, content={"error": "Federation not initialized"})

    ind = registry.get_indicator(indicator_id)
    if ind is None:
        return JSONResponse(status_code=404, content={"error": "Indicator not found"})

    return JSONResponse(content=ind)


# ---------------------------------------------------------------------------
# GET /v1/federation/stats
# ---------------------------------------------------------------------------

@router.get("/stats")
async def federation_stats(request: Request) -> JSONResponse:
    """Network-wide federation statistics."""
    err, registry, node_registry, federated = _get_auth_and_components(request)
    if err:
        return err

    result: dict[str, Any] = {
        "indicators": registry.get_stats() if registry else {},
        "network": node_registry.get_network_stats() if node_registry else {},
    }

    if federated:
        result["privacy"] = federated.dp_engine.stats

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /v1/federation/heartbeat
# ---------------------------------------------------------------------------

@router.post("/heartbeat")
async def node_heartbeat(request: Request) -> JSONResponse:
    """Register a node heartbeat."""
    err, registry, node_registry, _ = _get_auth_and_components(request)
    if err:
        return err

    if not node_registry:
        return JSONResponse(status_code=503, content={"error": "Federation not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    node_id = body.get("node_id", "")
    if not node_id:
        return JSONResponse(status_code=400, content={"error": "node_id is required"})

    metadata = {
        "indicators_ingested": body.get("indicators_ingested", 0),
        "last_attack_seen": body.get("last_attack_seen", ""),
    }
    node_registry.heartbeat(node_id, metadata)

    stats = node_registry.get_network_stats()
    indicator_count = registry.get_stats()["total"] if registry else 0

    # Record heartbeat in trust scorer
    from aegis.dashboard import _get_main
    m = _get_main()
    immune = getattr(m, "_federation_immune", None)
    if immune:
        immune.trust.record_positive(node_id, "heartbeat")

    return JSONResponse(content={
        "status": "ok",
        "network_size": stats["active_nodes"],
        "indicators_available": indicator_count,
    })


# ---------------------------------------------------------------------------
# GET /v1/federation/quarantined
# ---------------------------------------------------------------------------

@router.get("/quarantined")
async def list_quarantined(request: Request) -> JSONResponse:
    """List quarantined indicators."""
    err, registry, _, _ = _get_auth_and_components(request)
    if err:
        return err

    from aegis.dashboard import _get_main
    m = _get_main()
    immune = getattr(m, "_federation_immune", None)
    if not immune:
        return JSONResponse(content={"quarantined": [], "total": 0})

    items = immune.reputation.get_quarantined()
    return JSONResponse(content={"quarantined": items, "total": len(items)})


# ---------------------------------------------------------------------------
# GET /v1/federation/trust
# ---------------------------------------------------------------------------

@router.get("/trust")
async def node_trust_scores(request: Request) -> JSONResponse:
    """Get node trust scores."""
    err, _, _, _ = _get_auth_and_components(request)
    if err:
        return err

    from aegis.dashboard import _get_main
    m = _get_main()
    immune = getattr(m, "_federation_immune", None)
    if not immune:
        return JSONResponse(content={"scores": {}, "stats": {}})

    return JSONResponse(content={
        "scores": immune.trust.get_all_scores(),
        "stats": immune.trust.stats,
    })


# ---------------------------------------------------------------------------
# GET /v1/federation/immune/stats
# ---------------------------------------------------------------------------

@router.get("/immune/stats")
async def immune_stats(request: Request) -> JSONResponse:
    """Federation immune response statistics."""
    err, _, _, _ = _get_auth_and_components(request)
    if err:
        return err

    from aegis.dashboard import _get_main
    m = _get_main()
    immune = getattr(m, "_federation_immune", None)
    if not immune:
        return JSONResponse(content={"status": "not_initialized"})

    return JSONResponse(content=immune.stats)
