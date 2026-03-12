"""
Prometheus metrics scraper for AEGIS stress tests.

Parses the /metrics endpoint output to extract key counters and gauges.
"""

from __future__ import annotations

import httpx


def _parse_metric(text: str, name: str) -> float | None:
    """Parse a simple metric value from Prometheus text format."""
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return None


def _parse_metric_with_labels(
    text: str, name: str, labels: dict[str, str],
) -> float | None:
    """Parse a metric with specific label values from Prometheus text format."""
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    target = f"{name}{{{label_str}"
    for line in text.splitlines():
        if line.startswith(target):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return None


async def scrape_metrics(base_url: str, api_key: str) -> dict[str, float | None]:
    """Scrape and parse AEGIS Prometheus metrics.

    Returns a dict of metric names to their values.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{base_url}/metrics",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        text = resp.text

    return {
        "requests_total": _parse_metric(text, "aegis_requests_total"),
        "threats_detected": _parse_metric(text, "aegis_threats_detected_total"),
        "blocks_total": _parse_metric(text, "aegis_blocks_total"),
        "threat_level": _parse_metric(text, "aegis_threat_level"),
        "circuit_breaker_trips": _parse_metric(text, "aegis_circuit_breaker_trips_total"),
        "rate_limit_triggers": _parse_metric(text, "aegis_rate_limit_triggers_total"),
        "vault_size": _parse_metric(text, "aegis_vault_size"),
        "active_connections": _parse_metric(text, "aegis_active_connections"),
    }
