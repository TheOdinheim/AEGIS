"""
Continuous Re-Validation Scheduler — Extension 3.5

Periodically re-checks all approved components against updated threat
intelligence. Active components (invoked within 7 days) are checked more
frequently than inactive ones.

Also supports trigger-based re-validation when new threat signatures or
CVEs are ingested.

ASSUMED-BREACH POSTURE: Previously-approved components can become dangerous
when new vulnerabilities are disclosed or supply chain attacks are discovered
retroactively. The scheduler ensures no approval is permanent — every
component is continuously re-validated against the latest threat intelligence.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# 7 days in seconds — boundary between "active" and "inactive"
_ACTIVE_THRESHOLD_SECONDS = 7 * 24 * 3600


@dataclass
class RevalidationRun:
    run_id: str
    started_at: float
    completed_at: float | None = None
    components_checked: int = 0
    status_changes: list[dict[str, str]] = field(default_factory=list)
    new_findings: int = 0
    errors: int = 0
    trigger: str = "scheduled"  # "scheduled", "manual", "threat_intel"


@dataclass
class RevalidationSchedule:
    active_interval_hours: float
    inactive_interval_hours: float
    last_run: float | None = None
    next_run: float | None = None
    total_runs: int = 0


class RevalidationScheduler:
    """Background scheduler for continuous supply chain re-validation.

    Runs as an asyncio background task. Checks all approved components
    from the validation cache on a configurable interval.
    """

    def __init__(
        self,
        *,
        cache: Any = None,
        active_interval_hours: float = 6.0,
        inactive_interval_hours: float = 24.0,
        revalidate_fn: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._cache = cache
        self._active_interval = active_interval_hours * 3600
        self._inactive_interval = inactive_interval_hours * 3600
        self._revalidate_fn = revalidate_fn
        self._task: asyncio.Task | None = None
        self._running = False
        self._runs: list[RevalidationRun] = []
        self._schedule = RevalidationSchedule(
            active_interval_hours=active_interval_hours,
            inactive_interval_hours=inactive_interval_hours,
        )

    @property
    def schedule(self) -> RevalidationSchedule:
        return self._schedule

    @property
    def last_run(self) -> RevalidationRun | None:
        return self._runs[-1] if self._runs else None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the background re-validation loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Re-validation scheduler started (active=%sh, inactive=%sh)",
            self._schedule.active_interval_hours,
            self._schedule.inactive_interval_hours,
        )

    async def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Re-validation scheduler stopped")

    async def trigger_revalidation(self, trigger: str = "manual") -> RevalidationRun:
        """Trigger an immediate re-validation run."""
        return await self._run_revalidation(trigger=trigger)

    async def on_new_threat(self, threat_signature: str) -> RevalidationRun | None:
        """Handle new threat intelligence — retroactive check + re-validation.

        Called when a new threat signature is added to L4's vault or a
        new CVE is ingested via STIX/TAXII.
        """
        if self._cache is None:
            return None

        # Retroactive check
        matches = self._cache.retroactive_check(threat_signature)
        if matches:
            logger.warning(
                "Retroactive check found %d cached approval(s) matching threat '%s'",
                len(matches), threat_signature[:60],
            )
            # Invalidate matching approvals to force re-validation
            for entry in matches:
                self._cache.invalidate(entry.component_id)

            return await self._run_revalidation(trigger="threat_intel")

        return None

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status for the API endpoint."""
        last = self.last_run
        return {
            "running": self._running,
            "active_interval_hours": self._schedule.active_interval_hours,
            "inactive_interval_hours": self._schedule.inactive_interval_hours,
            "total_runs": self._schedule.total_runs,
            "last_run": {
                "run_id": last.run_id,
                "started_at": last.started_at,
                "completed_at": last.completed_at,
                "components_checked": last.components_checked,
                "status_changes": last.status_changes,
                "new_findings": last.new_findings,
                "errors": last.errors,
                "trigger": last.trigger,
            } if last else None,
            "next_run": self._schedule.next_run,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Background loop — runs re-validation on the shorter interval."""
        interval = min(self._active_interval, self._inactive_interval)
        # Wait one interval before first run
        self._schedule.next_run = time.time() + interval

        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                await self._run_revalidation(trigger="scheduled")
                self._schedule.next_run = time.time() + interval
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Re-validation loop error")
                await asyncio.sleep(60)  # backoff on error

    async def _run_revalidation(self, trigger: str = "scheduled") -> RevalidationRun:
        """Execute a single re-validation run."""
        run = RevalidationRun(
            run_id=uuid.uuid4().hex[:12],
            started_at=time.time(),
            trigger=trigger,
        )

        if self._cache is None:
            run.completed_at = time.time()
            self._runs.append(run)
            return run

        now = time.time()
        approved = self._cache.get_all_approved()

        for entry in approved:
            # Determine if active or inactive
            is_active = (now - entry.last_accessed) < _ACTIVE_THRESHOLD_SECONDS
            check_interval = self._active_interval if is_active else self._inactive_interval

            # Skip if not due for re-validation
            if trigger == "scheduled" and (now - entry.validated_at) < check_interval:
                continue

            run.components_checked += 1

            try:
                if self._revalidate_fn:
                    new_result = await self._revalidate_fn(
                        entry.component_id, entry.component_type,
                    )
                    if new_result is not None:
                        new_status = self._extract_status(new_result)
                        if new_status != entry.status:
                            run.status_changes.append({
                                "component_id": entry.component_id,
                                "old_status": entry.status,
                                "new_status": new_status,
                            })
                            self._cache.update_status(entry.component_id, new_status)

                        # Count new findings
                        findings_count = self._count_findings(new_result)
                        run.new_findings += findings_count

                        # Update cache with new validation time
                        self._cache.cache_result(
                            entry.component_id,
                            entry.component_type,
                            new_result,
                            entry.content_hash,
                            status=new_status,
                        )
                else:
                    # No revalidation function — just refresh TTL
                    self._cache.cache_result(
                        entry.component_id,
                        entry.component_type,
                        entry.validation_result,
                        entry.content_hash,
                        status=entry.status,
                    )

            except Exception as e:
                run.errors += 1
                logger.error(
                    "Re-validation error for %s (%s): %s",
                    entry.component_id, entry.component_type, e,
                )

        run.completed_at = time.time()
        self._runs.append(run)
        self._schedule.last_run = run.started_at
        self._schedule.total_runs += 1

        # Keep only last 100 runs
        if len(self._runs) > 100:
            self._runs = self._runs[-100:]

        logger.info(
            "Re-validation run %s: checked=%d, changes=%d, findings=%d, errors=%d (%.1fms)",
            run.run_id, run.components_checked, len(run.status_changes),
            run.new_findings, run.errors,
            (run.completed_at - run.started_at) * 1000,
        )

        return run

    @staticmethod
    def _extract_status(result: Any) -> str:
        """Extract approval status from a validation result."""
        # ProvenanceReport
        if hasattr(result, "verdict"):
            v = result.verdict
            v_str = v.value if hasattr(v, "value") else str(v)
            if v_str in ("trusted", "approved"):
                return "approved"
            elif v_str in ("rejected",):
                return "rejected"
            return "flagged"

        # DependencyAnalysisReport
        if hasattr(result, "overall_risk"):
            risk = result.overall_risk
            if risk == "clean":
                return "approved"
            elif risk in ("critical", "high"):
                return "rejected"
            return "flagged"

        # SkillAuditReport
        if hasattr(result, "blocked"):
            if result.blocked:
                return "rejected"
            return "approved"

        return "flagged"

    @staticmethod
    def _count_findings(result: Any) -> int:
        """Count findings from a validation result."""
        if hasattr(result, "findings"):
            return len(result.findings)
        if hasattr(result, "checks"):
            return sum(
                len(c.findings) if hasattr(c, "findings") else 0
                for c in result.checks
                if not getattr(c, "passed", True)
            )
        return 0
