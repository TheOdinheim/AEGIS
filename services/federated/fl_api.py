"""
FL API — Federated Learning HTTP Endpoints.

FastAPI router mounted at /v1/federation/fl/ for federated model training.
All endpoints require API key authentication.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/federation/fl", tags=["federation-fl"])


def _get_auth_and_fl(request: Request) -> tuple[JSONResponse | None, Any, Any, Any]:
    """Authenticate and return (error, fl_server, fl_client, fl_scheduler)."""
    from aegis.dashboard import _get_main
    import hmac

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

    if not hmac.compare_digest(token, m._config.api_key):
        return JSONResponse(status_code=401, content={"error": "Authentication required"}), None, None, None

    fl_server = getattr(m, "_fl_server", None)
    fl_client = getattr(m, "_fl_client", None)
    fl_scheduler = getattr(m, "_fl_scheduler", None)

    return None, fl_server, fl_client, fl_scheduler


# ---------------------------------------------------------------------------
# GET /v1/federation/fl/model
# ---------------------------------------------------------------------------

@router.get("/model")
async def get_model(request: Request) -> JSONResponse:
    """Return current global model weights."""
    err, fl_server, _, _ = _get_auth_and_fl(request)
    if err:
        return err
    if not fl_server:
        return JSONResponse(status_code=503, content={"error": "FL server not initialized"})

    return JSONResponse(content=fl_server.get_global_model())


# ---------------------------------------------------------------------------
# POST /v1/federation/fl/update
# ---------------------------------------------------------------------------

@router.post("/update")
async def submit_update(request: Request) -> JSONResponse:
    """Submit local training update."""
    err, fl_server, _, _ = _get_auth_and_fl(request)
    if err:
        return err
    if not fl_server:
        return JSONResponse(status_code=503, content={"error": "FL server not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    node_id = body.get("node_id", "")
    if not node_id:
        return JSONResponse(status_code=400, content={"error": "node_id is required"})

    weights = body.get("weights")
    if not weights or not isinstance(weights, list):
        return JSONResponse(status_code=400, content={"error": "weights must be a list"})

    num_samples = body.get("num_samples", 0)
    metrics = body.get("metrics", {})

    try:
        result = fl_server.submit_update(
            node_id=node_id,
            weights=weights,
            num_samples=num_samples,
            metrics=metrics,
        )
    except Exception as e:
        logger.error("FL update submission failed: %s", e)
        return JSONResponse(status_code=500, content={"error": "Update submission failed"})

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /v1/federation/fl/trigger-round
# ---------------------------------------------------------------------------

@router.post("/trigger-round")
async def trigger_round(request: Request) -> JSONResponse:
    """Manually trigger aggregation (for testing/demos)."""
    err, fl_server, fl_client, fl_scheduler = _get_auth_and_fl(request)
    if err:
        return err
    if not fl_server:
        return JSONResponse(status_code=503, content={"error": "FL server not initialized"})

    # If we have a local client with data, train and submit first
    if fl_client:
        from aegis.dashboard import _get_main
        m = _get_main()
        training_buffer = getattr(m, "_fl_training_buffer", None)
        if training_buffer and training_buffer.size >= 2:
            train_result = fl_client.train_local()
            if train_result.get("status") == "trained":
                update = fl_client.get_update()
                update["metrics"]["loss"] = train_result.get("loss", 0.0)
                fl_server.submit_update(
                    node_id=update["node_id"],
                    weights=update["weights"],
                    num_samples=update["num_samples"],
                    metrics=update["metrics"],
                )

    result = fl_server.aggregate()
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# GET /v1/federation/fl/status
# ---------------------------------------------------------------------------

@router.get("/status")
async def fl_status(request: Request) -> JSONResponse:
    """Current FL server status."""
    err, fl_server, fl_client, fl_scheduler = _get_auth_and_fl(request)
    if err:
        return err
    if not fl_server:
        return JSONResponse(status_code=503, content={"error": "FL server not initialized"})

    status = fl_server.get_status()

    if fl_scheduler:
        status["scheduler"] = fl_scheduler.stats

    return JSONResponse(content=status)
