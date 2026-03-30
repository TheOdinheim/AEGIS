"""
Response Emitter — Publishes TVE findings to the AEGIS event bus.

Six event types:
1. tve.validation.complete  — after every run
2. tve.detection.failure    — for each missed Tier 1-4 probe
3. tve.false_positive.detected — for each flagged Tier 5 probe
4. tve.scanner.dead         — for dead scanners
5. tve.latency.regression   — for latency regressions
6. tve.tli.recommendation   — when TLI adjustment is warranted

All events carry source: "tve" and run_id for correlation.
Graceful degradation: if event bus is unavailable, logs locally.
"""

from __future__ import annotations

import logging
from typing import Any

from aegis.layers.thymic.engine import ValidationReport
from aegis.layers.thymic.verdict_analyzer import VerdictReport

logger = logging.getLogger(__name__)

# TVE event channel
CHANNEL_TVE = "tve_validation"


class ResponseEmitter:
    """Publishes TVE findings to the AEGIS event bus."""

    def __init__(self, event_bus: Any | None = None) -> None:
        self._event_bus = event_bus

    async def emit(
        self,
        verdict_report: VerdictReport,
        validation_report: ValidationReport,
    ) -> list[dict[str, Any]]:
        """Emit all applicable events. Returns list of emitted events."""
        events: list[dict[str, Any]] = []
        run_id = validation_report.run_id

        # 1. tve.validation.complete — always emitted
        events.append(self._make_event("tve.validation.complete", run_id, {
            "run_type": validation_report.run_type,
            "total_probes": validation_report.total_probes,
            "probes_detected": validation_report.probes_detected,
            "probes_missed": validation_report.probes_missed,
            "false_positives": validation_report.false_positives,
            "overall_tpr": validation_report.overall_tpr,
            "overall_fpr": validation_report.overall_fpr,
            "overall_passed": verdict_report.overall_passed,
            "recommended_actions": verdict_report.recommended_actions,
            "duration_seconds": validation_report.duration_seconds,
        }))

        # 2. tve.detection.failure — for each missed probe
        for pr in validation_report.probe_results:
            if pr.probe_tier in (1, 2, 3, 4) and not pr.detected:
                # Find the probe to get text
                probe_text = ""
                for p_result in validation_report.probe_results:
                    if p_result.probe_id == pr.probe_id:
                        break
                # We don't have direct probe text access from ProbeResult,
                # so include probe_id and tier for correlation
                events.append(self._make_event("tve.detection.failure", run_id, {
                    "probe_id": pr.probe_id,
                    "probe_tier": pr.probe_tier,
                    "missed_layers": pr.missed_layers,
                }))

        # 3. tve.false_positive.detected — for each flagged Tier 5 probe
        for pr in validation_report.probe_results:
            if pr.probe_tier == 5 and pr.detected:
                events.append(self._make_event("tve.false_positive.detected", run_id, {
                    "probe_id": pr.probe_id,
                    "flagging_layers": pr.detection_layers,
                    "confidence_scores": pr.confidence_scores,
                }))

        # 4. tve.scanner.dead — for dead scanners
        if verdict_report.dead_scanner_verdict and not verdict_report.dead_scanner_verdict.passed:
            for scanner_id in verdict_report.dead_scanner_verdict.dead_scanners:
                events.append(self._make_event("tve.scanner.dead", run_id, {
                    "scanner_id": scanner_id,
                    "total_probes_tested": validation_report.total_probes,
                }))

        # 5. tve.latency.regression — for latency regressions
        for lv in verdict_report.latency_verdicts:
            if not lv.passed:
                events.append(self._make_event("tve.latency.regression", run_id, {
                    "layer_id": lv.layer_id,
                    "observed_p95_ms": lv.observed_p95,
                    "baseline_p95_ms": lv.baseline_p95,
                    "regression_factor": lv.regression_factor,
                }))

        # 6. tve.tli.recommendation
        tli_event = self._compute_tli_recommendation(verdict_report, run_id)
        if tli_event:
            events.append(tli_event)

        # Publish all events to event bus
        await self._publish_events(events)

        return events

    def _make_event(
        self, event_type: str, run_id: str, data: dict[str, Any],
    ) -> dict[str, Any]:
        """Create an event dict with standard fields."""
        return {
            "event_type": event_type,
            "source": "tve",
            "run_id": run_id,
            **data,
        }

    def _compute_tli_recommendation(
        self, verdict: VerdictReport, run_id: str,
    ) -> dict[str, Any] | None:
        """Compute TLI recommendation based on verdict results."""
        has_detection_failure = any(
            not v.passed for v in verdict.detection_verdicts
        )
        has_dead_scanners = (
            verdict.dead_scanner_verdict is not None
            and not verdict.dead_scanner_verdict.passed
        )

        if has_detection_failure and has_dead_scanners:
            return self._make_event("tve.tli.recommendation", run_id, {
                "recommended_direction": "up",
                "reason": "Detection failures combined with dead scanners",
            })

        # Check if all passed and FPR is well below threshold
        all_passed = verdict.overall_passed
        fpr_low = (
            verdict.false_positive_verdict is not None
            and verdict.false_positive_verdict.observed_fpr < verdict.false_positive_verdict.threshold * 0.5
        )

        if all_passed and fpr_low:
            return self._make_event("tve.tli.recommendation", run_id, {
                "recommended_direction": "down",
                "reason": "All checks passed with FPR well below threshold",
            })

        return None

    async def _publish_events(self, events: list[dict[str, Any]]) -> None:
        """Publish events to event bus, gracefully degrading if unavailable."""
        if not self._event_bus:
            if events:
                logger.info(
                    "TVE emitter: no event bus — logged %d events locally", len(events),
                )
            return

        for event in events:
            try:
                await self._event_bus.publish(CHANNEL_TVE, event)
            except Exception as e:
                logger.error("TVE event publish failed for %s: %s", event.get("event_type"), e)
