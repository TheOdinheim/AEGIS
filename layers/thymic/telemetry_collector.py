"""
Telemetry Collector — Per-probe, per-layer result capture and persistence.

Stores detection results from every probe in every validation run in a
rolling in-memory buffer. Provides aggregation methods for computing
per-layer baselines (TPR, FPR, latency). PostgreSQL persistence deferred
to Phase C/D.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from aegis.layers.thymic.attack_profile_library import Probe
from aegis.layers.thymic.layer_probe_router import ProbeResult

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    """A single probe result record."""

    probe_id: str
    probe_tier: int
    run_id: str
    timestamp: datetime
    detected: bool
    detection_layers: list[str]
    confidence_scores: dict[str, float]
    missed_layers: list[str]
    latency_ms: dict[str, float]
    false_positive: bool
    category: str
    expected_detection_layer: str | None


@dataclass
class LayerBaseline:
    """Computed baseline for a layer from historical results."""

    layer_id: str
    tpr: float = 0.0
    fpr: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryCollector:
    """Captures and stores detection results from validation runs.

    Rolling in-memory buffer with configurable max size and FIFO eviction.
    """

    def __init__(self, max_records: int = 100_000) -> None:
        self._records: list[TelemetryRecord] = []
        self._max_records = max_records

    def record_result(self, probe_result: ProbeResult, probe: Probe, run_id: str = "") -> None:
        """Store a single probe result."""
        record = TelemetryRecord(
            probe_id=probe_result.probe_id,
            probe_tier=probe_result.probe_tier,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            detected=probe_result.detected,
            detection_layers=list(probe_result.detection_layers),
            confidence_scores=dict(probe_result.confidence_scores),
            missed_layers=list(probe_result.missed_layers),
            latency_ms=dict(probe_result.latency_ms),
            false_positive=probe_result.false_positive,
            category=probe.category,
            expected_detection_layer=probe.expected_detection_layer,
        )
        self._records.append(record)
        # FIFO eviction
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

    def get_run_results(self, run_id: str) -> list[TelemetryRecord]:
        """All results for a specific run."""
        return [r for r in self._records if r.run_id == run_id]

    def get_layer_history(
        self, layer_id: str, window_hours: int = 24,
    ) -> list[TelemetryRecord]:
        """Recent results relevant to a specific layer."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        return [
            r for r in self._records
            if r.timestamp >= cutoff and (
                layer_id in r.detection_layers
                or layer_id in r.missed_layers
                or r.expected_detection_layer == layer_id
            )
        ]

    def get_baseline(self, layer_id: str) -> LayerBaseline:
        """Compute baseline for a layer from all historical results."""
        # Find records where this layer was expected to detect
        expected = [r for r in self._records if r.expected_detection_layer == layer_id]
        attack_expected = [r for r in expected if r.probe_tier in (1, 2, 3, 4)]
        benign_records = [r for r in self._records if r.probe_tier == 5]

        # TPR: of probes this layer should catch, how many did it catch?
        if attack_expected:
            detected = sum(1 for r in attack_expected if layer_id in r.detection_layers or r.detected)
            tpr = detected / len(attack_expected)
        else:
            tpr = 0.0

        # FPR: of benign probes, how many did this layer flag?
        if benign_records:
            flagged = sum(1 for r in benign_records if layer_id in r.detection_layers)
            fpr = flagged / len(benign_records)
        else:
            fpr = 0.0

        # Latency: from records where this layer participated
        latencies = []
        for r in self._records:
            if layer_id in r.latency_ms:
                latencies.append(r.latency_ms[layer_id])
            elif r.expected_detection_layer == layer_id and "total" in r.latency_ms:
                latencies.append(r.latency_ms["total"])

        avg_latency = statistics.mean(latencies) if latencies else 0.0
        p95_latency = (
            sorted(latencies)[int(len(latencies) * 0.95)]
            if len(latencies) >= 2
            else (latencies[0] if latencies else 0.0)
        )

        return LayerBaseline(
            layer_id=layer_id,
            tpr=tpr,
            fpr=fpr,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            sample_count=len(expected) + len([r for r in self._records if layer_id in r.detection_layers]),
        )

    def compute_baselines(self) -> dict[str, LayerBaseline]:
        """Compute baselines for all layers with data."""
        layers: set[str] = set()
        for r in self._records:
            layers.update(r.detection_layers)
            layers.update(r.missed_layers)
            if r.expected_detection_layer:
                layers.add(r.expected_detection_layer)

        return {layer_id: self.get_baseline(layer_id) for layer_id in layers}

    def prune(self, retention_days: int = 90) -> int:
        """Remove records older than retention period. Returns count removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        before = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        removed = before - len(self._records)
        if removed:
            logger.info("Pruned %d telemetry records older than %d days", removed, retention_days)
        return removed

    def record_count(self) -> int:
        """Total records stored."""
        return len(self._records)
