"""In-memory rolling time-series buffer for dashboard charts.

Samples key Prometheus metrics every 5 seconds and retains 1 hour of data.
Ephemeral — starts empty when AEGIS starts. Not a time-series database.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 1 hour at 5-second intervals = 720 samples
_MAX_SAMPLES = 720
_SAMPLE_INTERVAL = 5.0


class MetricsSample:
    """Single point-in-time metric sample."""

    __slots__ = ("timestamp", "values")

    def __init__(self, timestamp: float, values: dict[str, float]) -> None:
        self.timestamp = timestamp
        self.values = values


class MetricsBuffer:
    """Rolling in-memory buffer for dashboard time-series charts.

    Tracks: requests_total, blocks_total, threat_level, p95_latency_ms.
    Samples from Prometheus counters every 5 seconds.
    """

    def __init__(self, max_samples: int = _MAX_SAMPLES) -> None:
        self._samples: deque[MetricsSample] = deque(maxlen=max_samples)
        self._task: asyncio.Task | None = None
        self._running = False
        self._prev_requests: float = 0.0
        self._prev_blocks: float = 0.0

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def _read_metrics(self) -> dict[str, float]:
        """Read current values from Prometheus collectors."""
        try:
            from aegis.middleware.metrics import (
                REQUESTS_TOTAL,
                BLOCKS_TOTAL,
                THREAT_LEVEL,
                REQUEST_LATENCY,
            )

            # Sum all label combinations for counters
            req_total = 0.0
            for metric in REQUESTS_TOTAL.collect():
                for sample in metric.samples:
                    if sample.name == "aegis_requests_total_total":
                        req_total += sample.value

            blk_total = 0.0
            for metric in BLOCKS_TOTAL.collect():
                for sample in metric.samples:
                    if sample.name == "aegis_blocks_total_total":
                        blk_total += sample.value

            tli = 0.0
            for metric in THREAT_LEVEL.collect():
                for sample in metric.samples:
                    if sample.name == "aegis_threat_level":
                        tli = sample.value

            # Compute delta (rate) since last sample
            req_delta = max(0.0, req_total - self._prev_requests)
            blk_delta = max(0.0, blk_total - self._prev_blocks)
            self._prev_requests = req_total
            self._prev_blocks = blk_total

            # P95 from histogram — approximate from quantile buckets
            p95 = 0.0
            for metric in REQUEST_LATENCY.collect():
                for sample in metric.samples:
                    if "_sum" in sample.name:
                        count_name = sample.name.replace("_sum", "_count")
                        for s2 in metric.samples:
                            if s2.name == count_name and s2.labels == sample.labels:
                                if s2.value > 0:
                                    p95 = (sample.value / s2.value) * 1000  # avg as proxy
                                break

            return {
                "requests_total": req_delta / _SAMPLE_INTERVAL,  # RPS
                "blocks_total": blk_delta / _SAMPLE_INTERVAL,  # blocks/sec
                "threat_level": tli,
                "p95_latency_ms": round(p95, 2),
            }
        except Exception as e:
            logger.debug("Metrics read failed: %s", e)
            return {
                "requests_total": 0.0,
                "blocks_total": 0.0,
                "threat_level": 0.0,
                "p95_latency_ms": 0.0,
            }

    def _sample(self) -> None:
        """Take a single sample."""
        values = self._read_metrics()
        self._samples.append(MetricsSample(
            timestamp=time.time(),
            values=values,
        ))

    async def _run_loop(self) -> None:
        """Background sampling loop."""
        while self._running:
            try:
                self._sample()
            except Exception as e:
                logger.debug("Metrics buffer sample failed: %s", e)
            await asyncio.sleep(_SAMPLE_INTERVAL)

    async def start(self) -> None:
        """Start the background sampling task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background sampling task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def get_timeseries(
        self,
        metric: str = "requests_total",
        period: str = "1h",
    ) -> list[dict[str, Any]]:
        """Get time-bucketed metric data for charts.

        Args:
            metric: One of requests_total, blocks_total, threat_level, p95_latency_ms
            period: Time period — currently only '1h' supported

        Returns:
            List of {timestamp, value} dicts
        """
        now = time.time()
        cutoff = now - 3600  # 1h

        buckets = []
        for sample in self._samples:
            if sample.timestamp >= cutoff:
                buckets.append({
                    "timestamp": datetime.fromtimestamp(
                        sample.timestamp, tz=timezone.utc
                    ).isoformat(),
                    "value": round(sample.values.get(metric, 0.0), 4),
                })
        return buckets

    def prune(self, max_age_seconds: float = 3600.0) -> int:
        """Remove samples older than max_age_seconds. Returns count removed."""
        cutoff = time.time() - max_age_seconds
        removed = 0
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
            removed += 1
        return removed
