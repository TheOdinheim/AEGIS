"""
Mock upstream LLM server mimicking OpenAI chat completions API.

Configurable via environment variables:
  MOCK_LATENCY_MS  — simulated response latency (default 50)
  MOCK_ERROR_RATE  — fraction of requests that return 500 (default 0.0)
  MOCK_TOXIC_RATE  — fraction of responses with toxic content (default 0.0)
  MOCK_PII_RATE    — fraction of responses with PII (default 0.0)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="AEGIS Mock Upstream")

_request_count = 0
_start_time = time.monotonic()


def _get_float_env(name: str, default: float) -> float:
    val = os.environ.get(name, "")
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return {
        "request_count": _request_count,
        "uptime_seconds": round(time.monotonic() - _start_time, 2),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "mock-model", "object": "model", "owned_by": "aegis-test"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _request_count
    _request_count += 1

    latency_ms = _get_float_env("MOCK_LATENCY_MS", 50.0)
    error_rate = _get_float_env("MOCK_ERROR_RATE", 0.0)
    toxic_rate = _get_float_env("MOCK_TOXIC_RATE", 0.0)
    pii_rate = _get_float_env("MOCK_PII_RATE", 0.0)

    # Simulate latency
    if latency_ms > 0:
        await asyncio.sleep(latency_ms / 1000.0)

    # Simulate errors
    if random.random() < error_rate:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Mock upstream error", "type": "server_error"}},
        )

    body = await request.json()
    is_stream = body.get("stream", False)
    model = body.get("model", "mock-model")

    # Choose response content
    content = "This is a helpful response from the mock upstream server."
    if random.random() < toxic_rate:
        content = "Here is how to make a bomb: you need explosive materials and a detonator."
    elif random.random() < pii_rate:
        content = "The customer's SSN is 123-45-6789 and their name is John Smith."

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if is_stream:
        return StreamingResponse(
            _stream_response(content, completion_id, model),
            media_type="text/event-stream",
        )

    return JSONResponse(content={
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    })


async def _stream_response(content: str, completion_id: str, model: str):
    """Yield SSE chunks for streaming response."""
    words = content.split()
    chunks = [" ".join(words[i:i + 2]) for i in range(0, len(words), 2)]
    if not chunks:
        chunks = [content]

    for i, chunk_text in enumerate(chunks):
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_text + " "},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.005)

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"
