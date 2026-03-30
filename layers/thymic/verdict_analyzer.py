"""
Verdict Analyzer — Compares observed detection performance against baselines.

Four validation checks:
1. Detection Validation (Positive Selection) — per-layer TPR
2. False Positive Validation (Negative Selection) — FPR from Tier 5
3. Dead Scanner Detection — scanners with zero detections
4. Latency Regression — P95 latency vs baseline

Biological analog: Thymic selection pressure — T-cells that fail to recognize
foreign antigens (detection failure) or attack self-antigens (false positives)
are eliminated. The verdict analyzer is the selection mechanism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aegis.layers.thymic.engine import ValidationReport, LayerResult
from aegis.layers.thymic.telemetry_collector import TelemetryCollector, LayerBaseline

logger = logging.getLogger(__name__)


@dataclass
class DetectionVerdict:
    """Result of Check 1: per-layer TPR validation."""

    passed: bool
    layer_id: str
    observed_tpr: float
    baseline_tpr: float
    threshold: float
    deficit: float = 0.0  # How far below threshold


@dataclass
class FalsePositiveVerdict:
    """Result of Check 2: FPR validation from Tier 5 probes."""

    passed: bool
    observed_fpr: float
    baseline_fpr: float
    threshold: float
    flagged_probes: list[str] = field(default_factory=list)


@dataclass
class DeadScannerVerdict:
    """Result of Check 3: dead scanner detection."""

    passed: bool
    dead_scanners: list[str] = field(default_factory=list)


@dataclass
class LatencyVerdict:
    """Result of Check 4: latency regression detection."""

    passed: bool
    layer_id: str
    observed_p95: float
    baseline_p95: float
    regression_factor: float = 0.0


@dataclass
class VerdictReport:
    """Complete results from all four validation checks."""

    run_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    overall_passed: bool = True
    detection_verdicts: list[DetectionVerdict] = field(default_factory=list)
    false_positive_verdict: FalsePositiveVerdict | None = None
    dead_scanner_verdict: DeadScannerVerdict | None = None
    latency_verdicts: list[LatencyVerdict] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)


class VerdictAnalyzer:
    """Compares observed detection performance against baselines."""

    def __init__(
        self,
        default_tpr_threshold: float = 0.95,
        default_fpr_threshold: float = 0.005,
        latency_regression_factor: float = 2.0,
    ) -> None:
        self._default_tpr = default_tpr_threshold
        self._default_fpr = default_fpr_threshold
        self._regression_factor = latency_regression_factor

    def analyze(
        self,
        validation_report: ValidationReport,
        telemetry: TelemetryCollector,
    ) -> VerdictReport:
        """Run all four checks and produce a VerdictReport."""
        report = VerdictReport(run_id=validation_report.run_id)
        baselines = telemetry.compute_baselines()

        # Check 1: Detection Validation (Positive Selection)
        self._check_detection(report, validation_report, baselines)

        # Check 2: False Positive Validation (Negative Selection)
        self._check_false_positives(report, validation_report, baselines)

        # Check 3: Dead Scanner Detection
        self._check_dead_scanners(report, validation_report)

        # Check 4: Latency Regression
        self._check_latency(report, validation_report, baselines)

        # Determine overall pass
        report.overall_passed = (
            all(v.passed for v in report.detection_verdicts)
            and (report.false_positive_verdict is None or report.false_positive_verdict.passed)
            and (report.dead_scanner_verdict is None or report.dead_scanner_verdict.passed)
            and all(v.passed for v in report.latency_verdicts)
        )

        return report

    def _check_detection(
        self,
        report: VerdictReport,
        validation: ValidationReport,
        baselines: dict[str, LayerBaseline],
    ) -> None:
        """Check 1: Per-layer TPR validation."""
        for layer_id, layer_result in validation.per_layer_results.items():
            baseline = baselines.get(layer_id)
            threshold = baseline.tpr if baseline and baseline.sample_count > 0 else self._default_tpr
            # Use observed TPR from the validation report
            observed = layer_result.tpr if layer_result.probes_tested > 0 else 1.0

            passed = observed >= threshold
            deficit = max(0.0, threshold - observed)

            verdict = DetectionVerdict(
                passed=passed,
                layer_id=layer_id,
                observed_tpr=observed,
                baseline_tpr=threshold,
                threshold=threshold,
                deficit=deficit,
            )
            report.detection_verdicts.append(verdict)

            if not passed:
                report.recommended_actions.append(
                    f"{layer_id} TPR dropped to {observed:.2f} "
                    f"(baseline: {threshold:.2f}). Scanner diagnostic recommended."
                )

    def _check_false_positives(
        self,
        report: VerdictReport,
        validation: ValidationReport,
        baselines: dict[str, LayerBaseline],
    ) -> None:
        """Check 2: FPR validation from Tier 5 probes."""
        tier5 = validation.per_tier_results.get(5)
        if not tier5 or tier5.total == 0:
            # No Tier 5 probes — skip gracefully
            return

        observed_fpr = validation.overall_fpr
        # Use minimum baseline FPR across all layers, or default
        baseline_fprs = [b.fpr for b in baselines.values() if b.sample_count > 0]
        baseline_fpr = min(baseline_fprs) if baseline_fprs else 0.0
        threshold = self._default_fpr

        # Identify flagged Tier 5 probes
        flagged = [
            r.probe_id for r in validation.probe_results
            if r.probe_tier == 5 and r.detected
        ]

        passed = observed_fpr <= threshold
        verdict = FalsePositiveVerdict(
            passed=passed,
            observed_fpr=observed_fpr,
            baseline_fpr=baseline_fpr,
            threshold=threshold,
            flagged_probes=flagged,
        )
        report.false_positive_verdict = verdict

        if not passed:
            report.recommended_actions.append(
                f"FPR increased to {observed_fpr:.3f} (threshold: {threshold:.3f}). "
                f"{len(flagged)} benign probes incorrectly flagged. "
                f"Tolerance training recommended."
            )

    def _check_dead_scanners(
        self,
        report: VerdictReport,
        validation: ValidationReport,
    ) -> None:
        """Check 3: Scanners that produced zero detections."""
        if validation.total_probes == 0:
            return

        # Collect all layers that were expected to detect something
        expected_layers: set[str] = set()
        for r in validation.probe_results:
            if r.detection_layers:
                expected_layers.update(r.detection_layers)

        # A layer is "dead" if it was expected for some probes but detected none
        dead: list[str] = []
        layers_expected: dict[str, int] = {}
        layers_detected: dict[str, int] = {}

        for pr in validation.probe_results:
            if pr.missed_layers:
                for ml in pr.missed_layers:
                    layers_expected[ml] = layers_expected.get(ml, 0) + 1
                    if ml not in layers_detected:
                        layers_detected[ml] = 0

        for layer_id, expected_count in layers_expected.items():
            if layers_detected.get(layer_id, 0) == 0 and expected_count >= 3:
                dead.append(layer_id)

        passed = len(dead) == 0
        verdict = DeadScannerVerdict(passed=passed, dead_scanners=dead)
        report.dead_scanner_verdict = verdict

        for scanner in dead:
            report.recommended_actions.append(
                f"Scanner {scanner} produced zero detections across "
                f"{layers_expected.get(scanner, 0)} expected probes. "
                f"Scanner may be non-functional."
            )

    def _check_latency(
        self,
        report: VerdictReport,
        validation: ValidationReport,
        baselines: dict[str, LayerBaseline],
    ) -> None:
        """Check 4: P95 latency regression detection."""
        for layer_id, layer_result in validation.per_layer_results.items():
            baseline = baselines.get(layer_id)
            if not baseline or baseline.p95_latency_ms <= 0:
                continue

            observed = layer_result.p95_latency_ms
            baseline_p95 = baseline.p95_latency_ms
            factor = observed / baseline_p95 if baseline_p95 > 0 else 0.0

            passed = factor <= self._regression_factor
            verdict = LatencyVerdict(
                passed=passed,
                layer_id=layer_id,
                observed_p95=observed,
                baseline_p95=baseline_p95,
                regression_factor=factor,
            )
            report.latency_verdicts.append(verdict)

            if not passed:
                report.recommended_actions.append(
                    f"{layer_id} P95 latency at {observed:.0f}ms "
                    f"(baseline: {baseline_p95:.0f}ms, regression factor {factor:.2f}). "
                    f"Performance diagnostic recommended."
                )
