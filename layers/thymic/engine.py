"""
ThymicValidationEngine — L9 orchestrator for continuous defense validation.

Generates adversarial probes, routes them through the AEGIS pipeline (L1-L7),
and measures per-layer detection rates. After routing, feeds results to the
TelemetryCollector, VerdictAnalyzer, and ResponseEmitter for analysis and
event bus publication.

Biological analog: Thymic selection — the thymus continuously tests T-cell
competence by presenting self-antigens and foreign antigens. Only T-cells
that correctly distinguish self from non-self are allowed to mature.
"""

from __future__ import annotations

import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.layers.thymic.attack_profile_library import AttackProfileLibrary, Probe
from aegis.layers.thymic.layer_probe_router import LayerProbeRouter, ProbeResult
from aegis.layers.thymic.mutation_engine import MutationEngine
from aegis.layers.thymic.probe_generator import ProbeGenerator

logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    """Per-layer detection statistics from a validation run."""

    layer_id: str
    probes_tested: int = 0
    probes_detected: int = 0
    probes_missed: int = 0
    tpr: float = 0.0
    fpr: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


@dataclass
class TierResult:
    """Per-tier detection statistics from a validation run."""

    tier: int
    total: int = 0
    detected: int = 0
    missed: int = 0
    false_positives: int = 0
    tpr: float = 0.0
    fpr: float = 0.0


@dataclass
class ValidationReport:
    """Complete results from a thymic validation run."""

    run_id: str
    run_type: str  # "spot_check" or "comprehensive"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0
    total_probes: int = 0
    probes_detected: int = 0
    probes_missed: int = 0
    false_positives: int = 0
    per_layer_results: dict[str, LayerResult] = field(default_factory=dict)
    per_tier_results: dict[int, TierResult] = field(default_factory=dict)
    overall_tpr: float = 0.0
    overall_fpr: float = 0.0
    probe_results: list[ProbeResult] = field(default_factory=list)
    verdict: Any | None = None  # VerdictReport, typed as Any to avoid circular import


@dataclass
class HealthSummary:
    """Current per-layer TPR/FPR from most recent run."""

    last_run_id: str | None = None
    last_run_timestamp: datetime | None = None
    last_run_type: str | None = None
    overall_tpr: float = 0.0
    overall_fpr: float = 0.0
    per_layer_tpr: dict[str, float] = field(default_factory=dict)
    per_layer_fpr: dict[str, float] = field(default_factory=dict)
    total_probes_last_run: int = 0
    verdict_passed: bool | None = None
    recommended_actions: list[str] = field(default_factory=list)


class ThymicValidationEngine:
    """L9 Thymic Validation Engine — continuous defense validation orchestrator."""

    def __init__(
        self,
        library: AttackProfileLibrary | None = None,
        router: LayerProbeRouter | None = None,
        mutation_engine: MutationEngine | None = None,
        data_dir: Path | None = None,
        telemetry: Any | None = None,
        verdict_analyzer: Any | None = None,
        response_emitter: Any | None = None,
    ) -> None:
        self._data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "thymic"
        self._library = library or AttackProfileLibrary(data_dir=self._data_dir)
        self._mutation = mutation_engine or MutationEngine()
        self._generator = ProbeGenerator(self._library, self._mutation)
        self._router = router or LayerProbeRouter()
        self._last_report: ValidationReport | None = None
        self._telemetry = telemetry
        self._verdict_analyzer = verdict_analyzer
        self._response_emitter = response_emitter

    @property
    def library(self) -> AttackProfileLibrary:
        return self._library

    @property
    def generator(self) -> ProbeGenerator:
        return self._generator

    @property
    def router(self) -> LayerProbeRouter:
        return self._router

    @property
    def telemetry(self) -> Any:
        return self._telemetry

    async def run_spot_check(self, probes_per_tier: int = 50) -> ValidationReport:
        """Generate spot check probes, route them, return aggregate results."""
        start = time.perf_counter()
        run_id = str(uuid.uuid4())

        try:
            probes = self._generator.generate_spot_check(probes_per_tier=probes_per_tier)
        except Exception as e:
            logger.error("Probe generation failed: %s", e)
            return ValidationReport(
                run_id=run_id,
                run_type="spot_check",
                duration_seconds=time.perf_counter() - start,
            )

        if not probes:
            return ValidationReport(
                run_id=run_id,
                run_type="spot_check",
                duration_seconds=time.perf_counter() - start,
            )

        results = await self._router.route_batch(probes)
        report = self._build_report(run_id, "spot_check", probes, results, start)

        # Phase B: telemetry → verdict → emit
        await self._analyze_and_emit(report, probes, results)

        self._last_report = report
        return report

    async def run_comprehensive_sweep(self) -> ValidationReport:
        """Generate comprehensive probes, route them, return detailed results."""
        start = time.perf_counter()
        run_id = str(uuid.uuid4())

        try:
            probes = self._generator.generate_comprehensive()
        except Exception as e:
            logger.error("Comprehensive probe generation failed: %s", e)
            return ValidationReport(
                run_id=run_id,
                run_type="comprehensive",
                duration_seconds=time.perf_counter() - start,
            )

        if not probes:
            return ValidationReport(
                run_id=run_id,
                run_type="comprehensive",
                duration_seconds=time.perf_counter() - start,
            )

        results = await self._router.route_batch(probes)
        report = self._build_report(run_id, "comprehensive", probes, results, start)

        # Phase B: telemetry → verdict → emit
        await self._analyze_and_emit(report, probes, results)

        self._last_report = report
        return report

    async def _analyze_and_emit(
        self,
        report: ValidationReport,
        probes: list[Probe],
        results: list[ProbeResult],
    ) -> None:
        """Feed results to telemetry, verdict analyzer, and response emitter."""
        # Build probe map for telemetry recording
        probe_map = {p.id: p for p in probes}

        # 1. Record telemetry
        if self._telemetry:
            for pr in results:
                probe = probe_map.get(pr.probe_id)
                if probe:
                    self._telemetry.record_result(pr, probe, run_id=report.run_id)

        # 2. Run verdict analysis
        if self._verdict_analyzer and self._telemetry:
            try:
                verdict = self._verdict_analyzer.analyze(report, self._telemetry)
                report.verdict = verdict
            except Exception as e:
                logger.error("Verdict analysis failed: %s", e)

        # 3. Emit events
        if self._response_emitter and report.verdict:
            try:
                await self._response_emitter.emit(report.verdict, report)
            except Exception as e:
                logger.error("Response emission failed: %s", e)

    def get_health_summary(self) -> HealthSummary:
        """Return current per-layer TPR/FPR from most recent run."""
        if not self._last_report:
            return HealthSummary()

        r = self._last_report
        summary = HealthSummary(
            last_run_id=r.run_id,
            last_run_timestamp=r.timestamp,
            last_run_type=r.run_type,
            overall_tpr=r.overall_tpr,
            overall_fpr=r.overall_fpr,
            per_layer_tpr={
                lid: lr.tpr for lid, lr in r.per_layer_results.items()
            },
            per_layer_fpr={
                lid: lr.fpr for lid, lr in r.per_layer_results.items()
            },
            total_probes_last_run=r.total_probes,
        )

        # Include verdict info if available
        if r.verdict:
            summary.verdict_passed = r.verdict.overall_passed
            summary.recommended_actions = list(r.verdict.recommended_actions)

        return summary

    def _build_report(
        self,
        run_id: str,
        run_type: str,
        probes: list[Probe],
        results: list[ProbeResult],
        start: float,
    ) -> ValidationReport:
        """Build a ValidationReport from probe results."""
        duration = time.perf_counter() - start

        # Map results by probe_id for matching
        result_map = {r.probe_id: r for r in results}

        # Aggregate stats
        total = len(probes)
        attack_probes = [p for p in probes if p.expected_result == "block"]
        benign_probes = [p for p in probes if p.expected_result == "pass"]

        detected = sum(1 for p in attack_probes if result_map.get(p.id, ProbeResult(probe_id="", probe_tier=0, detected=False)).detected)
        missed = len(attack_probes) - detected
        false_positives = sum(1 for p in benign_probes if result_map.get(p.id, ProbeResult(probe_id="", probe_tier=0, detected=False)).detected)

        overall_tpr = detected / len(attack_probes) if attack_probes else 0.0
        overall_fpr = false_positives / len(benign_probes) if benign_probes else 0.0

        # Per-tier results
        per_tier: dict[int, TierResult] = {}
        for tier in {p.tier for p in probes}:
            tier_probes = [p for p in probes if p.tier == tier]
            tier_attacks = [p for p in tier_probes if p.expected_result == "block"]
            tier_benign = [p for p in tier_probes if p.expected_result == "pass"]
            tier_detected = sum(1 for p in tier_attacks if result_map.get(p.id, ProbeResult(probe_id="", probe_tier=0, detected=False)).detected)
            tier_fp = sum(1 for p in tier_benign if result_map.get(p.id, ProbeResult(probe_id="", probe_tier=0, detected=False)).detected)

            per_tier[tier] = TierResult(
                tier=tier,
                total=len(tier_probes),
                detected=tier_detected,
                missed=len(tier_attacks) - tier_detected,
                false_positives=tier_fp,
                tpr=tier_detected / len(tier_attacks) if tier_attacks else 0.0,
                fpr=tier_fp / len(tier_benign) if tier_benign else 0.0,
            )

        # Per-layer results (aggregate from detection_layers)
        per_layer: dict[str, LayerResult] = {}
        all_layers = set()
        for r in results:
            all_layers.update(r.detection_layers)
            if r.missed_layers:
                all_layers.update(r.missed_layers)

        for layer_id in all_layers:
            layer_detected = sum(
                1 for r in results
                if layer_id in r.detection_layers
            )
            layer_expected = sum(
                1 for p in probes
                if p.expected_detection_layer == layer_id
            )
            layer_missed = sum(
                1 for r in results
                if layer_id in r.missed_layers
            )

            # Collect latencies for this layer
            latencies = [
                r.latency_ms.get("total", 0.0)
                for r in results
                if layer_id in r.detection_layers
            ]

            per_layer[layer_id] = LayerResult(
                layer_id=layer_id,
                probes_tested=layer_expected,
                probes_detected=layer_detected,
                probes_missed=layer_missed,
                tpr=layer_detected / layer_expected if layer_expected > 0 else 0.0,
                avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
                p95_latency_ms=(
                    sorted(latencies)[int(len(latencies) * 0.95)]
                    if len(latencies) >= 2 else (latencies[0] if latencies else 0.0)
                ),
            )

        return ValidationReport(
            run_id=run_id,
            run_type=run_type,
            duration_seconds=duration,
            total_probes=total,
            probes_detected=detected,
            probes_missed=missed,
            false_positives=false_positives,
            per_layer_results=per_layer,
            per_tier_results=per_tier,
            overall_tpr=overall_tpr,
            overall_fpr=overall_fpr,
            probe_results=results,
        )
