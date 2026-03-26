"""
FL Scheduler — Automated Federated Training Rounds.

Manages periodic local training, update submission, aggregation,
and global model distribution. Runs as an asyncio background task.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from aegis.services.federated.fl_server import FLServer
from aegis.services.federated.fl_client import FLClient

logger = logging.getLogger(__name__)


class FLScheduler:
    """Automated training round management.

    Parameters:
        server: FL aggregation server.
        client: Local FL client.
        interval_hours: Hours between training rounds (default 6.0).
    """

    def __init__(
        self,
        server: FLServer,
        client: FLClient,
        *,
        interval_hours: float = 6.0,
    ) -> None:
        self._server = server
        self._client = client
        self._interval_seconds = interval_hours * 3600
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._rounds_completed: int = 0
        self._last_round_time: str | None = None
        self._total_samples_trained: int = 0

    async def start(self) -> None:
        """Begin the scheduling loop."""
        if self._running:
            return
        self._running = True
        try:
            self._task = asyncio.create_task(self._loop())
            logger.info("FL scheduler started (interval=%.1fh)", self._interval_seconds / 3600)
        except RuntimeError:
            self._running = False
            logger.warning("No event loop — FL scheduler not started")

    async def stop(self) -> None:
        """Cancel the scheduling loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("FL scheduler stopped")

    async def run_round(self) -> dict[str, Any]:
        """Execute one full training round.

        1. Train locally on accumulated buffer
        2. Submit update to server
        3. If enough updates, trigger aggregation
        4. Apply new global model
        5. Clear training buffer
        """
        # 1. Train locally
        train_result = self._client.train_local()
        if train_result.get("status") != "trained":
            logger.debug("FL round skipped: %s", train_result.get("status"))
            return {"status": "skipped", "reason": train_result.get("status", "unknown")}

        samples = train_result.get("samples", 0)

        # 2. Submit update
        update = self._client.get_update()
        update["metrics"]["loss"] = train_result.get("loss", 0.0)
        submit_result = self._server.submit_update(
            node_id=update["node_id"],
            weights=update["weights"],
            num_samples=update["num_samples"],
            metrics=update["metrics"],
        )

        # 3. Aggregate if triggered (auto-aggregate on min_clients met)
        aggregated = submit_result.get("aggregated", False)
        if not aggregated and len(self._server._current_updates) >= self._server._min_clients:
            self._server.aggregate()
            aggregated = True

        # 4. Apply new global model
        if aggregated:
            global_model = self._server.get_global_model()
            self._client.apply_global_model(global_model["weights"])

        # 5. Clear buffer
        self._client._buffer.clear()

        self._rounds_completed += 1
        self._last_round_time = datetime.now(timezone.utc).isoformat()
        self._total_samples_trained += samples

        logger.info(
            "FL round %d complete: %d samples, aggregated=%s",
            self._rounds_completed, samples, aggregated,
        )

        return {
            "status": "completed",
            "round": self._rounds_completed,
            "samples": samples,
            "aggregated": aggregated,
            "loss": train_result.get("loss", 0.0),
        }

    async def _loop(self) -> None:
        """Background loop."""
        while self._running:
            try:
                await asyncio.sleep(self._interval_seconds)
                if not self._running:
                    break
                if self._client._buffer.size >= 2:
                    await self.run_round()
                else:
                    logger.debug(
                        "FL round skipped: only %d samples buffered",
                        self._client._buffer.size,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("FL scheduled round failed: %s", e)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "rounds_completed": self._rounds_completed,
            "last_round_time": self._last_round_time,
            "interval_hours": self._interval_seconds / 3600,
            "total_samples_trained": self._total_samples_trained,
        }
