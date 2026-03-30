"""
Thymic Scheduler — Adaptive scheduling for TVE operational modes.

Four modes:
1. Continuous Spot Check  — lightweight, every 15 min (configurable)
2. Comprehensive Sweep    — full library, every 6 hours (configurable)
3. Post-Change Validation — event-triggered, targets changed layers
4. Stress Validation      — on-demand only, elevated concurrency

Adaptive scheduling: per-layer stability scores track consecutive passes.
After threshold consecutive stable sweeps, probe frequency for that layer
is halved (thymic involution). Resets to full on any failure.

Biological analog: The thymus involutes (shrinks) as the immune system
matures, reducing T-cell production. But it can reactivate when the body
encounters new pathogens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aegis.layers.thymic.engine import ThymicValidationEngine, ValidationReport

logger = logging.getLogger(__name__)


@dataclass
class ScheduleStatus:
    """Current scheduler state."""

    is_running: bool = False
    spot_check_interval_minutes: float = 15.0
    sweep_interval_hours: float = 6.0
    next_spot_check: datetime | None = None
    next_sweep: datetime | None = None
    last_spot_check: datetime | None = None
    last_sweep: datetime | None = None
    total_spot_checks: int = 0
    total_sweeps: int = 0
    stability_scores: dict[str, int] = field(default_factory=dict)
    decayed_layers: list[str] = field(default_factory=list)


class ThymicScheduler:
    """Adaptive scheduler for TVE operational modes."""

    def __init__(
        self,
        engine: ThymicValidationEngine,
        spot_check_interval_minutes: float = 15.0,
        sweep_interval_hours: float = 6.0,
        adaptive_decay_threshold: int = 30,
        stress_max_concurrency: int = 50,
        probes_per_tier: int = 50,
    ) -> None:
        self._engine = engine
        self._spot_interval = spot_check_interval_minutes * 60  # seconds
        self._sweep_interval = sweep_interval_hours * 3600  # seconds
        self._decay_threshold = adaptive_decay_threshold
        self._stress_max_concurrency = stress_max_concurrency
        self._probes_per_tier = probes_per_tier

        # State
        self._spot_task: asyncio.Task | None = None
        self._sweep_task: asyncio.Task | None = None
        self._is_running = False
        self._stability_scores: dict[str, int] = {}
        self._decayed_layers: set[str] = set()
        self._total_spot_checks = 0
        self._total_sweeps = 0
        self._last_spot_check: datetime | None = None
        self._last_sweep: datetime | None = None
        self._next_spot_check: datetime | None = None
        self._next_sweep: datetime | None = None
        self._reports: list[ValidationReport] = []
        self._max_reports = 100

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def reports(self) -> list[ValidationReport]:
        return list(self._reports)

    async def start(self) -> None:
        """Begin background scheduling (spot checks + sweeps)."""
        if self._is_running:
            return
        self._is_running = True
        self._spot_task = asyncio.create_task(self._spot_check_loop())
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        logger.info("TVE scheduler started (spot=%ss, sweep=%ss)",
                     self._spot_interval, self._sweep_interval)

    async def stop(self) -> None:
        """Cancel all scheduled tasks."""
        self._is_running = False
        for task in (self._spot_task, self._sweep_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._spot_task = None
        self._sweep_task = None
        logger.info("TVE scheduler stopped")

    async def trigger_post_change(self, changed_layers: list[str]) -> ValidationReport:
        """Trigger post-change validation for specified layers."""
        report = await self._engine.run_post_change(changed_layers)
        self._store_report(report)
        self._update_stability(report)
        return report

    async def trigger_stress(self, concurrency_multiplier: int = 10) -> ValidationReport:
        """Trigger stress validation (on-demand only)."""
        report = await self._engine.run_stress_validation(
            concurrency_multiplier=concurrency_multiplier,
            max_concurrency=self._stress_max_concurrency,
        )
        self._store_report(report)
        return report

    def get_schedule_status(self) -> dict:
        """Returns current schedule state, stability scores, adaptive decay."""
        return {
            "is_running": self._is_running,
            "spot_check_interval_minutes": self._spot_interval / 60,
            "sweep_interval_hours": self._sweep_interval / 3600,
            "next_spot_check": self._next_spot_check.isoformat() if self._next_spot_check else None,
            "next_sweep": self._next_sweep.isoformat() if self._next_sweep else None,
            "last_spot_check": self._last_spot_check.isoformat() if self._last_spot_check else None,
            "last_sweep": self._last_sweep.isoformat() if self._last_sweep else None,
            "total_spot_checks": self._total_spot_checks,
            "total_sweeps": self._total_sweeps,
            "stability_scores": dict(self._stability_scores),
            "decayed_layers": sorted(self._decayed_layers),
            "adaptive_decay_threshold": self._decay_threshold,
        }

    def get_stability_score(self, layer_id: str) -> int:
        """Get stability score for a layer."""
        return self._stability_scores.get(layer_id, 0)

    def is_layer_decayed(self, layer_id: str) -> bool:
        """Check if a layer has entered adaptive decay."""
        return layer_id in self._decayed_layers

    async def _spot_check_loop(self) -> None:
        """Periodic spot check loop."""
        while self._is_running:
            self._next_spot_check = datetime.now(timezone.utc)
            try:
                report = await self._engine.run_spot_check(
                    probes_per_tier=self._probes_per_tier,
                )
                self._store_report(report)
                self._update_stability(report)
                self._total_spot_checks += 1
                self._last_spot_check = datetime.now(timezone.utc)
            except Exception as e:
                logger.error("TVE spot check failed: %s", e)
            try:
                await asyncio.sleep(self._spot_interval)
            except asyncio.CancelledError:
                break

    async def _sweep_loop(self) -> None:
        """Periodic comprehensive sweep loop."""
        while self._is_running:
            self._next_sweep = datetime.now(timezone.utc)
            try:
                report = await self._engine.run_comprehensive_sweep()
                self._store_report(report)
                self._update_stability(report)
                self._total_sweeps += 1
                self._last_sweep = datetime.now(timezone.utc)
            except Exception as e:
                logger.error("TVE sweep failed: %s", e)
            try:
                await asyncio.sleep(self._sweep_interval)
            except asyncio.CancelledError:
                break

    def _store_report(self, report: ValidationReport) -> None:
        """Store report in rolling buffer."""
        self._reports.append(report)
        if len(self._reports) > self._max_reports:
            self._reports = self._reports[-self._max_reports:]

    def _update_stability(self, report: ValidationReport) -> None:
        """Update per-layer stability scores from report verdict."""
        if not report.verdict:
            return

        verdict = report.verdict

        # Check each layer in detection verdicts
        for dv in verdict.detection_verdicts:
            if dv.passed:
                self._stability_scores[dv.layer_id] = (
                    self._stability_scores.get(dv.layer_id, 0) + 1
                )
            else:
                self._stability_scores[dv.layer_id] = 0
                self._decayed_layers.discard(dv.layer_id)

        # Check for decay threshold
        for layer_id, score in self._stability_scores.items():
            if score >= self._decay_threshold:
                self._decayed_layers.add(layer_id)
            # Layers that failed have score 0, already removed above
