"""
Load generator for AEGIS stress testing.

Sends concurrent requests to an AEGIS instance and collects latency,
error rate, and throughput metrics.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


@dataclass
class RequestResult:
    """Result of a single request."""
    status_code: int
    latency_ms: float
    blocked: bool = False
    error: str | None = None


@dataclass
class LoadTestResult:
    """Aggregate results of a load test run."""
    total_requests: int = 0
    successful: int = 0
    blocked: int = 0
    errors: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    rps_achieved: float = 0.0
    duration_seconds: float = 0.0
    results: list[RequestResult] = field(default_factory=list)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100.0)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


async def run_load_test(
    base_url: str,
    api_key: str,
    concurrency: int,
    total_requests: int,
    request_body_fn: Callable[[], dict[str, Any]] | None = None,
) -> LoadTestResult:
    """Run a load test with the given concurrency and request count.

    Args:
        base_url: AEGIS base URL (e.g., http://localhost:8765)
        api_key: AEGIS API key for authentication
        concurrency: Maximum concurrent requests
        total_requests: Total number of requests to send
        request_body_fn: Callable returning request body dict. Defaults to benign prompt.

    Returns:
        LoadTestResult with aggregate metrics
    """
    if request_body_fn is None:
        def request_body_fn():
            return {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
            }

    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def send_request(client: httpx.AsyncClient) -> RequestResult:
        async with sem:
            body = request_body_fn()
            start = time.monotonic()
            try:
                resp = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0,
                )
                latency = (time.monotonic() - start) * 1000
                blocked = resp.status_code in (403, 429)
                error_msg = None
                if resp.status_code >= 500:
                    error_msg = f"HTTP {resp.status_code}"
                return RequestResult(
                    status_code=resp.status_code,
                    latency_ms=latency,
                    blocked=blocked,
                    error=error_msg,
                )
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                return RequestResult(
                    status_code=0,
                    latency_ms=latency,
                    error=str(e),
                )

    start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [send_request(client) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)

    duration = time.monotonic() - start_time
    results_list = list(results)

    latencies = sorted(r.latency_ms for r in results_list)
    successful = sum(1 for r in results_list if r.status_code == 200)
    blocked = sum(1 for r in results_list if r.blocked)
    errors = sum(1 for r in results_list if r.error is not None)

    return LoadTestResult(
        total_requests=len(results_list),
        successful=successful,
        blocked=blocked,
        errors=errors,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        max_ms=max(latencies) if latencies else 0.0,
        rps_achieved=len(results_list) / duration if duration > 0 else 0.0,
        duration_seconds=duration,
        results=results_list,
    )
