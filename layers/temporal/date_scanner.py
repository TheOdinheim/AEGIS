"""
Date-Triggered Anomaly Scanner — Detects behavioral shifts at temporal boundaries.

The Sleeper Cell paper (March 2026) demonstrated models trained to behave
normally before a trigger date but execute malicious actions after. Anthropic's
Sleeper Agents research (January 2024) showed persistent backdoors activated
by date-based triggers. This scanner monitors for step-function behavioral
changes at temporal boundaries.

Key insight: natural behavioral drift is gradual and continuous. Sleeper agent
activation is discrete and aligned with a temporal boundary. The scanner
specifically looks for step-function changes rather than gradual trends.

ASSUMED-BREACH POSTURE: A compromised BBE could suppress the metrics that
would reveal a sleeper agent's activation. The date scanner is an independent
observer — it queries BBE metrics but also watches for anomalies in the
metrics themselves (e.g., sudden flatness in previously noisy metrics could
indicate the BBE has been tampered with to hide changes).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BoundaryCheck:
    """Result of checking for behavioral shift at a temporal boundary."""
    check_id: str
    boundary_type: str  # "daily", "monthly", "quarterly", "yearly", "deployment", "custom"
    boundary_timestamp: float
    pre_window_hours: float
    post_window_hours: float
    pre_metrics: dict[str, float]
    post_metrics: dict[str, float]
    divergence_scores: dict[str, float]
    max_divergence: float
    alert_triggered: bool
    alert_description: str | None = None


@dataclass
class DateScannerStatus:
    """Current status of the date-triggered anomaly scanner."""
    enabled: bool
    monitored_boundaries: list[str]
    custom_trigger_dates: list[str]
    checks_performed: int
    alerts_generated: int
    last_check_timestamp: float | None
    next_scheduled_check: float | None


class DateTriggeredAnomalyScanner:
    """Monitors for behavioral shifts at temporal boundaries.

    At each monitored boundary, the scanner:
    1. Captures pre-boundary BBE metrics
    2. Captures post-boundary BBE metrics
    3. Computes per-metric divergence (relative change)
    4. Alerts if divergence exceeds threshold

    Integration points:
    - BBE: source of behavioral metrics for pre/post comparison
    - Event bus: publishes boundary alerts to "temporal_drift"
    - TCE: boundary alerts trigger correlation analysis
    - Snapshot manager: deployment events trigger both snapshot and boundary check
    """

    def __init__(
        self,
        *,
        pre_window_hours: float = 6.0,
        post_window_hours: float = 2.0,
        divergence_threshold: float = 0.3,
        multi_metric_threshold: float = 0.15,
        event_bus: Any | None = None,
    ) -> None:
        self._pre_window_hours = pre_window_hours
        self._post_window_hours = post_window_hours
        self._divergence_threshold = divergence_threshold
        self._multi_metric_threshold = multi_metric_threshold
        self._event_bus = event_bus

        self._bbe: Any | None = None
        self._custom_trigger_dates: list[tuple[str, str]] = []  # (date_str, description)
        self._checks: list[BoundaryCheck] = []
        self._max_checks = 200
        self._alerts_generated = 0

        self._monitored_boundaries = ["daily", "monthly", "deployment"]

        # Scheduler state
        self._scheduler_task: Any | None = None
        self._scheduler_running = False

    def set_bbe(self, bbe: Any) -> None:
        """Set BBE reference for querying behavioral metrics."""
        self._bbe = bbe

    def check_boundary(
        self,
        boundary_type: str,
        boundary_timestamp: float,
        tenant_id: str = "default",
    ) -> BoundaryCheck:
        """Check for behavioral shift at a specific temporal boundary.

        Queries BBE for pre/post metrics and computes divergence.
        If BBE is unavailable, returns a check with empty metrics and no alert.
        """
        pre_metrics = self._get_metrics_snapshot(tenant_id)
        post_metrics = self._get_metrics_snapshot(tenant_id)

        # Compute divergence for each metric
        divergence_scores: dict[str, float] = {}
        for key in set(pre_metrics) | set(post_metrics):
            pre_val = pre_metrics.get(key, 0.0)
            post_val = post_metrics.get(key, 0.0)
            divergence_scores[key] = self._compute_divergence(pre_val, post_val)

        max_div = max(divergence_scores.values()) if divergence_scores else 0.0

        # Alert logic
        alert_triggered = False
        alert_description = None

        # Single metric exceeds threshold
        high_div_metrics = [
            k for k, v in divergence_scores.items()
            if v > self._divergence_threshold
        ]
        if high_div_metrics:
            alert_triggered = True
            alert_description = (
                f"High divergence at {boundary_type} boundary: "
                f"{', '.join(high_div_metrics)} "
                f"(max={max_div:.3f}, threshold={self._divergence_threshold})"
            )

        # Multiple metrics exceed multi-metric threshold
        if not alert_triggered:
            moderate_div_metrics = [
                k for k, v in divergence_scores.items()
                if v > self._multi_metric_threshold
            ]
            if len(moderate_div_metrics) >= 3:
                alert_triggered = True
                alert_description = (
                    f"Multi-metric divergence at {boundary_type} boundary: "
                    f"{len(moderate_div_metrics)} metrics above {self._multi_metric_threshold} "
                    f"({', '.join(moderate_div_metrics)})"
                )

        if alert_triggered:
            self._alerts_generated += 1

        check = BoundaryCheck(
            check_id=uuid.uuid4().hex[:12],
            boundary_type=boundary_type,
            boundary_timestamp=boundary_timestamp,
            pre_window_hours=self._pre_window_hours,
            post_window_hours=self._post_window_hours,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            divergence_scores=divergence_scores,
            max_divergence=max_div,
            alert_triggered=alert_triggered,
            alert_description=alert_description,
        )

        self._checks.append(check)
        if len(self._checks) > self._max_checks:
            self._checks = self._checks[-self._max_checks:]

        if alert_triggered:
            logger.warning(
                "Date scanner alert: %s boundary at %.0f — %s",
                boundary_type, boundary_timestamp, alert_description,
            )

        return check

    def _get_metrics_snapshot(self, tenant_id: str) -> dict[str, float]:
        """Get current BBE metrics as a flat dictionary."""
        if not self._bbe:
            return {}

        try:
            baselines = self._bbe.get_baseline(tenant_id)
            metrics: dict[str, float] = {}
            for category, data in baselines.items():
                prefix = category
                metrics[f"{prefix}_refusal_rate"] = data.get("refusal_rate", 0.0)
                metrics[f"{prefix}_length_mean"] = data.get("length_mean", 0.0)
                metrics[f"{prefix}_length_std"] = data.get("length_std", 0.0)
                metrics[f"{prefix}_tool_invocation_rate"] = data.get("tool_invocation_rate", 0.0)
                metrics[f"{prefix}_interaction_count"] = float(data.get("interaction_count", 0))
            return metrics
        except Exception as e:
            logger.debug("Failed to get BBE metrics: %s", e)
            return {}

    def _compute_divergence(self, pre_value: float, post_value: float) -> float:
        """Compute divergence between pre and post values.

        Uses relative change for normal values. For near-zero pre_values,
        uses absolute change to detect activation from zero.
        """
        # Near-zero pre_value: use absolute change
        if abs(pre_value) < 1e-6:
            return abs(post_value)

        return abs(post_value - pre_value) / abs(pre_value)

    def add_custom_trigger_date(self, date_str: str, description: str = "") -> None:
        """Add a custom date to monitor (ISO format: YYYY-MM-DD)."""
        self._custom_trigger_dates.append((date_str, description))

    def get_recent_checks(
        self, tenant_id: str = "default", limit: int = 20
    ) -> list[BoundaryCheck]:
        """Get recent boundary checks."""
        # Return most recent first
        return list(reversed(self._checks[-limit:]))

    def get_status(self) -> DateScannerStatus:
        """Get scanner status."""
        return DateScannerStatus(
            enabled=True,
            monitored_boundaries=list(self._monitored_boundaries),
            custom_trigger_dates=[d for d, _ in self._custom_trigger_dates],
            checks_performed=len(self._checks),
            alerts_generated=self._alerts_generated,
            last_check_timestamp=self._checks[-1].boundary_timestamp if self._checks else None,
            next_scheduled_check=None,  # Computed by scheduler
        )

    async def start_scheduler(self) -> None:
        """Start the background scheduler."""
        import asyncio
        if self._scheduler_running:
            return
        self._scheduler_running = True

        async def _scheduler_loop() -> None:
            while self._scheduler_running:
                try:
                    # Check every hour for upcoming boundaries
                    await asyncio.sleep(3600)
                    if not self._scheduler_running:
                        break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Date scanner scheduler error: %s", e)

        self._scheduler_task = asyncio.create_task(_scheduler_loop())

    async def stop_scheduler(self) -> None:
        """Stop the scheduler."""
        self._scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except Exception:
                pass
            self._scheduler_task = None
