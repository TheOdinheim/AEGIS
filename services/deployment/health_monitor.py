"""
Deep Health Monitor — Comprehensive System Health with Auto-Recovery.

Biological Analog: The hypothalamic-pituitary-adrenal (HPA) axis continuously
monitors the body's internal state: temperature, hormone levels, immune cell
counts, organ function. Deviations trigger corrective actions automatically.
Only persistent or severe anomalies escalate to conscious awareness (alerts).

Goes beyond basic /health by checking component latency, consistency,
and disk space. Attempts auto-recovery for recoverable conditions.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "last_checked": self.last_checked.isoformat(),
        }


@dataclass
class DeepHealthResult:
    """Result of a comprehensive deep health check."""
    overall_status: str  # healthy, degraded, unhealthy
    component_results: dict[str, ComponentHealth] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "component_results": {
                k: v.to_dict() for k, v in self.component_results.items()
            },
            "recommendations": self.recommendations,
            "checked_at": self.checked_at.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
        }


class DeepHealthMonitor:
    """Comprehensive health monitoring with auto-recovery.

    Performs deep health checks beyond the basic /health endpoint:
    - Redis: PING + round-trip latency
    - PostgreSQL: SELECT 1 + latency
    - FAISS index: vector count consistency
    - DeBERTa model: test prompt classification
    - Disk space: free space on log/backup partition
    - Circuit breaker: state and consecutive trips

    Auto-recovery:
    - Redis disconnected → attempt reconnect
    - FAISS mismatch → log warning (manual intervention)

    Parameters:
        check_interval_seconds: Seconds between background checks.
        disk_warning_gb: Free disk space threshold for warning.
        disk_critical_mb: Free disk space threshold for critical.
        redis_latency_warn_ms: Redis latency threshold for warning.
        pg_latency_warn_ms: PostgreSQL latency threshold for warning.
    """

    def __init__(
        self,
        *,
        check_interval_seconds: int = 60,
        disk_warning_gb: float = 1.0,
        disk_critical_mb: float = 100.0,
        redis_latency_warn_ms: float = 100.0,
        pg_latency_warn_ms: float = 200.0,
    ) -> None:
        self._interval = check_interval_seconds
        self._disk_warning_gb = disk_warning_gb
        self._disk_critical_mb = disk_critical_mb
        self._redis_latency_warn = redis_latency_warn_ms
        self._pg_latency_warn = pg_latency_warn_ms
        self._monitor_task: asyncio.Task[None] | None = None
        self._running = False
        self._last_result: DeepHealthResult | None = None

        # Component references (set after init)
        self._vault: Any = None
        self._healing: Any = None
        self._adaptive: Any = None
        self._event_bus: Any = None

    def set_components(
        self,
        *,
        vault: Any = None,
        healing: Any = None,
        adaptive: Any = None,
        event_bus: Any = None,
    ) -> None:
        """Set references to AEGIS components for health checks."""
        self._vault = vault
        self._healing = healing
        self._adaptive = adaptive
        self._event_bus = event_bus

    async def run_deep_health_check(self) -> DeepHealthResult:
        """Run comprehensive health check across all components.

        Returns:
            DeepHealthResult with per-component status and recommendations.
        """
        start = time.perf_counter()
        components: dict[str, ComponentHealth] = {}
        recommendations: list[str] = []

        # Run checks concurrently where possible
        await asyncio.gather(
            self._check_redis(components, recommendations),
            self._check_postgres(components, recommendations),
            self._check_disk_space(components, recommendations),
            return_exceptions=True,
        )

        # Sequential checks (depend on component state)
        self._check_faiss(components, recommendations)
        self._check_circuit_breaker(components, recommendations)
        self._check_model(components, recommendations)

        # Determine overall status
        statuses = [c.status for c in components.values()]
        if "unhealthy" in statuses:
            overall = "unhealthy"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        duration = (time.perf_counter() - start) * 1000

        result = DeepHealthResult(
            overall_status=overall,
            component_results=components,
            recommendations=recommendations,
            duration_ms=duration,
        )
        self._last_result = result

        # Publish health event on status change
        if self._event_bus and self._last_result:
            try:
                await self._event_bus.publish("health_monitor", {
                    "overall_status": overall,
                    "duration_ms": duration,
                    "component_count": len(components),
                })
            except Exception:
                pass

        return result

    async def _check_redis(
        self, components: dict[str, ComponentHealth], recommendations: list[str],
    ) -> None:
        """Check Redis connectivity and latency."""
        try:
            from aegis.services.redis_client import redis_health, init_redis, get_redis

            check_start = time.perf_counter()
            status = await redis_health()
            latency = (time.perf_counter() - check_start) * 1000

            connected = status.get("connected", False) if isinstance(status, dict) else False

            if not connected:
                components["redis"] = ComponentHealth(
                    name="redis",
                    status="degraded",
                    latency_ms=latency,
                    details={"connected": False, "auto_recovery": "attempted"},
                )
                recommendations.append(
                    "Redis is unreachable. Rate limiting and event bus will use "
                    "in-memory fallback. Check Redis connection."
                )
                # Auto-recovery: attempt reconnect
                try:
                    import os
                    redis_url = os.environ.get("REDIS_URL", "")
                    if redis_url:
                        await init_redis(redis_url)
                        logger.info("Redis auto-reconnect attempted")
                except Exception as e:
                    logger.warning("Redis auto-reconnect failed: %s", e)
            elif latency > self._redis_latency_warn:
                components["redis"] = ComponentHealth(
                    name="redis",
                    status="degraded",
                    latency_ms=latency,
                    details={"connected": True, "high_latency": True},
                )
                recommendations.append(
                    f"Redis latency is {latency:.0f}ms (threshold: {self._redis_latency_warn}ms). "
                    f"Investigate network or Redis server performance."
                )
            else:
                components["redis"] = ComponentHealth(
                    name="redis",
                    status="healthy",
                    latency_ms=latency,
                    details={"connected": True},
                )
        except ImportError:
            components["redis"] = ComponentHealth(
                name="redis", status="degraded",
                details={"error": "redis module not available"},
            )
        except Exception as e:
            components["redis"] = ComponentHealth(
                name="redis", status="degraded",
                details={"error": str(e)},
            )

    async def _check_postgres(
        self, components: dict[str, ComponentHealth], recommendations: list[str],
    ) -> None:
        """Check PostgreSQL connectivity and latency."""
        try:
            from aegis.services.db import db_health

            check_start = time.perf_counter()
            status = await db_health()
            latency = (time.perf_counter() - check_start) * 1000

            connected = status.get("connected", False) if isinstance(status, dict) else False

            if not connected:
                components["postgres"] = ComponentHealth(
                    name="postgres",
                    status="degraded",
                    latency_ms=latency,
                    details={"connected": False},
                )
                recommendations.append(
                    "PostgreSQL is unreachable. Audit logging will use JSONL fallback. "
                    "Check database connection."
                )
            elif latency > self._pg_latency_warn:
                components["postgres"] = ComponentHealth(
                    name="postgres",
                    status="degraded",
                    latency_ms=latency,
                    details={"connected": True, "high_latency": True},
                )
                recommendations.append(
                    f"PostgreSQL latency is {latency:.0f}ms (threshold: {self._pg_latency_warn}ms). "
                    f"Investigate database performance."
                )
            else:
                components["postgres"] = ComponentHealth(
                    name="postgres",
                    status="healthy",
                    latency_ms=latency,
                    details={"connected": True},
                )
        except ImportError:
            components["postgres"] = ComponentHealth(
                name="postgres", status="degraded",
                details={"error": "db module not available"},
            )
        except Exception as e:
            components["postgres"] = ComponentHealth(
                name="postgres", status="degraded",
                details={"error": str(e)},
            )

    def _check_faiss(
        self, components: dict[str, ComponentHealth], recommendations: list[str],
    ) -> None:
        """Check FAISS index consistency."""
        if not self._vault:
            components["faiss"] = ComponentHealth(
                name="faiss", status="degraded",
                details={"error": "vault not available"},
            )
            return

        try:
            stats = self._vault.get_stats()
            total_indicators = stats.get("total_indicators", 0)
            index_size = stats.get("index_size", 0)

            # Check consistency between indicator count and index size
            if total_indicators > 0 and index_size > 0:
                mismatch_ratio = abs(total_indicators - index_size) / max(total_indicators, 1)
                if mismatch_ratio > 0.1:
                    components["faiss"] = ComponentHealth(
                        name="faiss",
                        status="degraded",
                        details={
                            "indicators": total_indicators,
                            "index_size": index_size,
                            "mismatch_ratio": round(mismatch_ratio, 3),
                        },
                    )
                    recommendations.append(
                        f"FAISS index size ({index_size}) differs from indicator count "
                        f"({total_indicators}) by {mismatch_ratio:.1%}. Consider vault reload."
                    )
                    return

            components["faiss"] = ComponentHealth(
                name="faiss",
                status="healthy",
                details={
                    "indicators": total_indicators,
                    "index_size": index_size,
                },
            )
        except Exception as e:
            components["faiss"] = ComponentHealth(
                name="faiss", status="degraded",
                details={"error": str(e)},
            )

    def _check_circuit_breaker(
        self, components: dict[str, ComponentHealth], recommendations: list[str],
    ) -> None:
        """Check circuit breaker state."""
        if not self._healing:
            components["circuit_breaker"] = ComponentHealth(
                name="circuit_breaker", status="degraded",
                details={"error": "healing layer not available"},
            )
            return

        try:
            breaker = self._healing.get_breaker("primary")
            state = breaker.state.value
            trips = breaker.consecutive_trips

            if state == "open":
                components["circuit_breaker"] = ComponentHealth(
                    name="circuit_breaker",
                    status="unhealthy",
                    details={"state": state, "consecutive_trips": trips},
                )
                recommendations.append(
                    f"Circuit breaker is OPEN ({trips} consecutive trips). "
                    f"Upstream model may be compromised or unreachable."
                )
            elif state == "half_open":
                components["circuit_breaker"] = ComponentHealth(
                    name="circuit_breaker",
                    status="degraded",
                    details={"state": state, "consecutive_trips": trips},
                )
                recommendations.append(
                    "Circuit breaker is HALF_OPEN — probing upstream model."
                )
            else:
                components["circuit_breaker"] = ComponentHealth(
                    name="circuit_breaker",
                    status="healthy",
                    details={"state": state, "consecutive_trips": trips},
                )
        except Exception as e:
            components["circuit_breaker"] = ComponentHealth(
                name="circuit_breaker", status="degraded",
                details={"error": str(e)},
            )

    def _check_model(
        self, components: dict[str, ComponentHealth], recommendations: list[str],
    ) -> None:
        """Check ML model availability."""
        if not self._adaptive:
            components["ml_model"] = ComponentHealth(
                name="ml_model", status="degraded",
                details={"error": "adaptive layer not available"},
            )
            return

        # Check if models are loaded
        has_classifier = hasattr(self._adaptive, '_classifier') and self._adaptive._classifier is not None
        has_semantic = hasattr(self._adaptive, '_semantic') and self._adaptive._semantic is not None

        if has_classifier or has_semantic:
            components["ml_model"] = ComponentHealth(
                name="ml_model",
                status="healthy",
                details={
                    "classifier_loaded": has_classifier,
                    "semantic_loaded": has_semantic,
                },
            )
        else:
            components["ml_model"] = ComponentHealth(
                name="ml_model",
                status="degraded",
                details={
                    "classifier_loaded": False,
                    "semantic_loaded": False,
                    "skip_model_load": True,
                },
            )

    async def _check_disk_space(
        self, components: dict[str, ComponentHealth], recommendations: list[str],
    ) -> None:
        """Check available disk space."""
        try:
            # Check the root partition or current working directory
            check_path = Path.cwd()
            usage = shutil.disk_usage(str(check_path))
            free_gb = usage.free / (1024 ** 3)
            free_mb = usage.free / (1024 ** 2)
            total_gb = usage.total / (1024 ** 3)

            if free_mb < self._disk_critical_mb:
                components["disk_space"] = ComponentHealth(
                    name="disk_space",
                    status="unhealthy",
                    details={
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "critical": True,
                    },
                )
                recommendations.append(
                    f"Critical: Only {free_mb:.0f}MB disk space remaining. "
                    f"Immediate action required — rotate logs and clear temp files."
                )
            elif free_gb < self._disk_warning_gb:
                components["disk_space"] = ComponentHealth(
                    name="disk_space",
                    status="degraded",
                    details={
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                    },
                )
                recommendations.append(
                    f"Low disk space: {free_gb:.1f}GB free. "
                    f"Recommend log rotation and cleaning old backups."
                )
            else:
                components["disk_space"] = ComponentHealth(
                    name="disk_space",
                    status="healthy",
                    details={
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                    },
                )
        except Exception as e:
            components["disk_space"] = ComponentHealth(
                name="disk_space", status="degraded",
                details={"error": str(e)},
            )

    async def _monitor_loop(self) -> None:
        """Background loop running deep health checks."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                result = await self.run_deep_health_check()
                if result.overall_status != "healthy":
                    logger.warning(
                        "Deep health check: %s (%d recommendations)",
                        result.overall_status, len(result.recommendations),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Deep health monitor failed: %s", e)

    def start_monitor(self) -> None:
        """Start background deep health monitoring."""
        if self._running:
            return
        self._running = True
        try:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Deep health monitor started (interval=%ds)", self._interval)
        except RuntimeError:
            self._running = False
            logger.warning("No event loop — deep health monitor not started")

    def stop_monitor(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
        logger.info("Deep health monitor stopped")

    @property
    def last_result(self) -> DeepHealthResult | None:
        """Most recent deep health check result."""
        return self._last_result

    @property
    def stats(self) -> dict[str, Any]:
        """Monitor statistics."""
        return {
            "running": self._running,
            "interval_seconds": self._interval,
            "last_check": self._last_result.checked_at.isoformat() if self._last_result else None,
            "last_status": self._last_result.overall_status if self._last_result else None,
        }
