"""
Clean State Snapshot Manager — Forensic state capture and replay.

When the TCE identifies a suspected poisoning event, the snapshot manager
provides forensic comparison: what was the system state before the event,
and what changed? By capturing hashes and metadata (not full state), snapshots
are lightweight (<10KB) and fast to capture (<10ms).

Research grounding: Sleeper Cell paper (March 2026) demonstrated delayed
activation requiring pre/post comparison. Anthropic's Sleeper Agents research
(January 2024) showed models behaving differently after temporal triggers.
Replay comparison provides the evidence chain for forensic attribution.

ASSUMED-BREACH POSTURE: A compromised component could return a false hash
during snapshot capture, making it appear unchanged. The integrity_hash
field provides tamper detection for the snapshot itself, but cannot detect
a component lying about its own state. Multiple independent observers
(BBE metrics, canary results, MPR provenance) provide defense-in-depth
against this scenario.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ComponentSnapshot:
    """Snapshot of a single component's state."""
    component_name: str
    content_hash: str
    record_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """Complete state snapshot at a point in time."""
    snapshot_id: str
    timestamp: float
    tenant_id: str
    trigger: str  # "scheduled", "pre_deployment", "pre_config_change", "pre_rag_refresh", "manual", "pre_model_update"
    components: dict[str, ComponentSnapshot]
    metadata: dict = field(default_factory=dict)
    integrity_hash: str = ""


@dataclass
class ComponentDiff:
    """Difference between a historical and current component state."""
    component_name: str
    before_hash: str
    current_hash: str
    before_count: int
    current_count: int
    change_summary: str


@dataclass
class ReplayReport:
    """Result of replaying a snapshot comparison."""
    report_id: str
    before_snapshot_id: str
    before_timestamp: float
    current_timestamp: float
    components_changed: list[ComponentDiff]
    components_unchanged: list[str]
    canary_results: list[dict] | None
    behavioral_divergence_detected: bool
    analysis_latency_ms: float


class CleanStateSnapshotManager:
    """Captures, stores, and replays system state snapshots.

    Components register themselves via register_component() with callbacks
    that return their current hash, item count, and metadata. Snapshots
    capture only hashes — never full state — for storage efficiency.

    Integration points:
    - TCE: finds pre-poisoning snapshot, triggers replay comparison
    - Canary system: replay can re-run canaries for behavioral comparison
    - Event bus: publishes snapshot events to "temporal_drift"
    - BBE: pre/post boundary snapshots support date scanner
    """

    def __init__(
        self,
        *,
        max_snapshots: int = 90,
        event_bus: Any | None = None,
    ) -> None:
        self._max_snapshots = max_snapshots
        self._event_bus = event_bus
        self._lock = threading.Lock()

        # Registered components: name -> (hash_fn, count_fn, metadata_fn)
        self._components: dict[str, tuple[
            Callable[[], str],
            Callable[[], int] | None,
            Callable[[], dict] | None,
        ]] = {}

        # Snapshot storage (ring buffer, ordered by timestamp)
        self._snapshots: list[StateSnapshot] = []

        # Scheduler state
        self._scheduler_task: Any | None = None
        self._scheduler_running = False

        # Canary system reference (optional)
        self._canary_system: Any | None = None

    def register_component(
        self,
        name: str,
        hash_fn: Callable[[], str],
        count_fn: Callable[[], int] | None = None,
        metadata_fn: Callable[[], dict] | None = None,
    ) -> None:
        """Register a component for snapshot capture."""
        self._components[name] = (hash_fn, count_fn, metadata_fn)

    def set_canary_system(self, canary_system: Any) -> None:
        """Set canary system reference for replay."""
        self._canary_system = canary_system

    def capture_snapshot(
        self,
        tenant_id: str = "default",
        trigger: str = "manual",
        metadata: dict | None = None,
    ) -> StateSnapshot:
        """Capture current state snapshot. Fast (<10ms) — only collects hashes."""
        components: dict[str, ComponentSnapshot] = {}

        for name, (hash_fn, count_fn, meta_fn) in self._components.items():
            try:
                content_hash = hash_fn()
                record_count = count_fn() if count_fn else 0
                comp_meta = meta_fn() if meta_fn else {}
                components[name] = ComponentSnapshot(
                    component_name=name,
                    content_hash=content_hash,
                    record_count=record_count,
                    metadata=comp_meta,
                )
            except Exception as e:
                logger.debug("Snapshot: failed to capture component %s: %s", name, e)

        # Compute integrity hash from all component hashes
        hash_parts = sorted(
            f"{name}:{cs.content_hash}" for name, cs in components.items()
        )
        integrity_hash = hashlib.sha256(
            "|".join(hash_parts).encode()
        ).hexdigest()

        snapshot = StateSnapshot(
            snapshot_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            tenant_id=tenant_id,
            trigger=trigger,
            components=components,
            metadata=metadata or {},
            integrity_hash=integrity_hash,
        )

        with self._lock:
            self._snapshots.append(snapshot)
            # Ring buffer eviction
            while len(self._snapshots) > self._max_snapshots:
                self._snapshots.pop(0)

        logger.info(
            "Snapshot captured: id=%s, trigger=%s, components=%d",
            snapshot.snapshot_id, trigger, len(components),
        )
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> StateSnapshot | None:
        """Get a specific snapshot by ID."""
        with self._lock:
            for s in self._snapshots:
                if s.snapshot_id == snapshot_id:
                    return s
        return None

    def get_snapshot_before(
        self, timestamp: float, tenant_id: str = "default"
    ) -> StateSnapshot | None:
        """Get the most recent snapshot before a given timestamp."""
        with self._lock:
            candidates = [
                s for s in self._snapshots
                if s.timestamp < timestamp and s.tenant_id == tenant_id
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.timestamp)

    def get_snapshots(
        self, tenant_id: str = "default", limit: int = 10
    ) -> list[StateSnapshot]:
        """Get recent snapshots for a tenant."""
        with self._lock:
            filtered = [
                s for s in self._snapshots if s.tenant_id == tenant_id
            ]
        # Most recent first
        filtered.sort(key=lambda s: s.timestamp, reverse=True)
        return filtered[:limit]

    async def replay_comparison(
        self,
        before_snapshot_id: str,
        canary_queries: list[str] | None = None,
    ) -> ReplayReport:
        """Compare a historical snapshot against current state."""
        start = time.perf_counter()

        before = self.get_snapshot(before_snapshot_id)
        if before is None:
            elapsed = (time.perf_counter() - start) * 1000.0
            return ReplayReport(
                report_id=uuid.uuid4().hex[:12],
                before_snapshot_id=before_snapshot_id,
                before_timestamp=0.0,
                current_timestamp=time.time(),
                components_changed=[],
                components_unchanged=[],
                canary_results=None,
                behavioral_divergence_detected=False,
                analysis_latency_ms=elapsed,
            )

        # Capture current state for comparison
        current = self.capture_snapshot(
            tenant_id=before.tenant_id,
            trigger="replay_comparison",
        )

        changed: list[ComponentDiff] = []
        unchanged: list[str] = []

        # Compare each component from the before snapshot
        for name, before_comp in before.components.items():
            current_comp = current.components.get(name)
            if current_comp is None:
                changed.append(ComponentDiff(
                    component_name=name,
                    before_hash=before_comp.content_hash,
                    current_hash="<missing>",
                    before_count=before_comp.record_count,
                    current_count=0,
                    change_summary=f"{name}: component no longer registered",
                ))
            elif current_comp.content_hash != before_comp.content_hash:
                count_delta = current_comp.record_count - before_comp.record_count
                sign = "+" if count_delta >= 0 else ""
                changed.append(ComponentDiff(
                    component_name=name,
                    before_hash=before_comp.content_hash,
                    current_hash=current_comp.content_hash,
                    before_count=before_comp.record_count,
                    current_count=current_comp.record_count,
                    change_summary=f"{name}: hash changed, records {sign}{count_delta}",
                ))
            else:
                unchanged.append(name)

        # Check for new components not in the before snapshot
        for name in current.components:
            if name not in before.components:
                cc = current.components[name]
                changed.append(ComponentDiff(
                    component_name=name,
                    before_hash="<not_present>",
                    current_hash=cc.content_hash,
                    before_count=0,
                    current_count=cc.record_count,
                    change_summary=f"{name}: new component added since snapshot",
                ))

        # Run canary queries if available
        canary_results = None
        if canary_queries and self._canary_system:
            canary_results = []
            for q in canary_queries:
                try:
                    result = await self._canary_system.inject_canary(canary_id=None)
                    if result:
                        canary_results.append({
                            "query": q,
                            "verdict": result.verdict,
                            "keyword_score": result.keyword_score,
                            "semantic_score": result.semantic_score,
                        })
                except Exception as e:
                    canary_results.append({"query": q, "error": str(e)})

        # Behavioral divergence: any component changed
        behavioral_divergence = len(changed) > 0

        elapsed = (time.perf_counter() - start) * 1000.0

        report = ReplayReport(
            report_id=uuid.uuid4().hex[:12],
            before_snapshot_id=before_snapshot_id,
            before_timestamp=before.timestamp,
            current_timestamp=current.timestamp,
            components_changed=changed,
            components_unchanged=unchanged,
            canary_results=canary_results,
            behavioral_divergence_detected=behavioral_divergence,
            analysis_latency_ms=elapsed,
        )

        logger.info(
            "Replay comparison: before=%s, changed=%d, unchanged=%d, divergence=%s",
            before_snapshot_id, len(changed), len(unchanged), behavioral_divergence,
        )
        return report

    def get_stats(self) -> dict:
        """Get snapshot manager statistics."""
        with self._lock:
            count = len(self._snapshots)
            triggers: dict[str, int] = {}
            for s in self._snapshots:
                triggers[s.trigger] = triggers.get(s.trigger, 0) + 1
            oldest = self._snapshots[0].timestamp if self._snapshots else None
            newest = self._snapshots[-1].timestamp if self._snapshots else None
        return {
            "total_snapshots": count,
            "max_snapshots": self._max_snapshots,
            "registered_components": list(self._components.keys()),
            "triggers": triggers,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
            "scheduler_running": self._scheduler_running,
        }

    async def start_scheduler(self, interval_hours: float = 24.0) -> None:
        """Start periodic snapshot scheduler."""
        import asyncio
        if self._scheduler_running:
            return
        self._scheduler_running = True
        interval_seconds = interval_hours * 3600.0

        async def _scheduler_loop() -> None:
            while self._scheduler_running:
                try:
                    await asyncio.sleep(interval_seconds)
                    if not self._scheduler_running:
                        break
                    self.capture_snapshot(trigger="scheduled")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Snapshot scheduler error: %s", e)

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
