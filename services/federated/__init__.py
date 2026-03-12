"""
Federated Threat Intelligence — L8 Herd Immunity.

Biological Analog: Herd immunity protects unvaccinated individuals when enough
of the population is immune. Similarly, when one AEGIS instance detects a novel
attack, the knowledge is federated to all instances — an attack seen by one
customer immunizes all customers.

Architecture:
- Each AEGIS instance trains a local prompt injection model on its own data
- Only model weight updates (gradients) are shared — never training data
- Federated Averaging (FedAvg) produces improved global model
- Differential Privacy (ε=3) on all shared data
- Anonymized threat indicators shared via STIX-compatible format

Components:
- DifferentialPrivacyEngine: Gaussian mechanism with budget tracking
- LocalTrainer: Logistic regression on TF-IDF features
- FederatedAggregator: FedAvg with weight poisoning defense
- IndicatorSharingService: DP-noised indicator exchange with dedup
- FederatedIntelligenceManager: Orchestrator with round scheduling
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from aegis.services.federated.aggregation import AggregationResult, FederatedAggregator
from aegis.services.federated.indicator_sharing import IndicatorSharingService, SharedIndicator
from aegis.services.federated.local_trainer import LocalModelUpdate, LocalTrainer
from aegis.services.federated.privacy import DifferentialPrivacyEngine, PrivacyBudgetStatus

logger = logging.getLogger(__name__)


@dataclass
class FederatedRoundResult:
    """Result of a complete federated learning round."""
    round_number: int
    local_update: LocalModelUpdate | None = None
    aggregation: AggregationResult | None = None
    indicators_shared: int = 0
    indicators_received: int = 0
    privacy_budget: PrivacyBudgetStatus | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "round_number": self.round_number,
            "timestamp": self.timestamp.isoformat(),
            "indicators_shared": self.indicators_shared,
            "indicators_received": self.indicators_received,
        }
        if self.local_update:
            result["local_update"] = self.local_update.to_dict()
        if self.aggregation:
            result["aggregation"] = self.aggregation.to_dict()
        if self.privacy_budget:
            result["privacy_budget"] = self.privacy_budget.to_dict()
        if self.error:
            result["error"] = self.error
        return result


class FederatedIntelligenceManager:
    """Orchestrator for L8 federated threat intelligence.

    Manages the complete federated learning lifecycle:
    1. Local training on instance-specific data
    2. DP noise application to weight updates
    3. Federated aggregation (FedAvg)
    4. Global model distribution back to local trainer
    5. Indicator sharing with DP-noised embeddings
    6. Background scheduling of federated rounds

    Parameters:
        instance_id: Unique identifier for this AEGIS instance.
        epsilon: DP epsilon parameter (default 3.0).
        delta: DP delta parameter (default 1e-5).
        privacy_budget: Total privacy budget (default 100.0).
        min_participants: Minimum updates for aggregation (default 1).
        round_interval_seconds: Seconds between automatic rounds (default 21600 = 6h).
    """

    def __init__(
        self,
        *,
        instance_id: str = "local",
        epsilon: float = 3.0,
        delta: float = 1e-5,
        privacy_budget: float = 100.0,
        min_participants: int = 1,
        round_interval_seconds: int = 21600,
    ) -> None:
        self._instance_id = instance_id
        self._round_interval = round_interval_seconds

        # Core components
        self._dp_engine = DifferentialPrivacyEngine(
            epsilon=epsilon,
            delta=delta,
            total_budget=privacy_budget,
        )
        self._trainer = LocalTrainer(instance_id=instance_id)
        self._aggregator = FederatedAggregator(min_participants=min_participants)
        self._sharing = IndicatorSharingService(
            dp_engine=self._dp_engine,
            instance_id=instance_id,
        )

        # Round history
        self._rounds: list[FederatedRoundResult] = []
        self._scheduler_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def dp_engine(self) -> DifferentialPrivacyEngine:
        return self._dp_engine

    @property
    def trainer(self) -> LocalTrainer:
        return self._trainer

    @property
    def aggregator(self) -> FederatedAggregator:
        return self._aggregator

    @property
    def sharing(self) -> IndicatorSharingService:
        return self._sharing

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def round_count(self) -> int:
        return len(self._rounds)

    def add_training_sample(self, text: str, is_threat: bool) -> None:
        """Add a labeled sample for federated training."""
        self._trainer.add_training_sample(text, is_threat)

    async def run_federated_round(
        self,
        external_updates: list[LocalModelUpdate] | None = None,
    ) -> FederatedRoundResult:
        """Execute a complete federated learning round.

        Steps:
        1. Train local model on buffered data
        2. Apply DP noise to local weights
        3. Aggregate with any external updates (FedAvg)
        4. Apply global weights back to local trainer

        Args:
            external_updates: Model updates from other instances.

        Returns:
            FederatedRoundResult with round details.
        """
        round_num = len(self._rounds) + 1
        result = FederatedRoundResult(round_number=round_num)

        try:
            # Step 1: Local training
            local_update: LocalModelUpdate | None = None
            if self._trainer.training_samples >= 2:
                local_update = self._trainer.train()
                result.local_update = local_update

                # Step 2: Apply DP noise to local weights before sharing
                if not self._dp_engine.is_budget_exhausted:
                    noised_weights = self._dp_engine.add_noise_to_weights(local_update.weights)
                    local_update = LocalModelUpdate(
                        instance_id=local_update.instance_id,
                        round_number=local_update.round_number,
                        weights=noised_weights,
                        num_samples=local_update.num_samples,
                        metrics=local_update.metrics,
                        timestamp=local_update.timestamp,
                    )

                # Step 3: Aggregate
                all_updates = [local_update]
                if external_updates:
                    all_updates.extend(external_updates)

                try:
                    agg_result = self._aggregator.aggregate(all_updates)
                    result.aggregation = agg_result

                    # Step 4: Apply global weights
                    self._trainer.apply_global_weights(agg_result.global_weights)
                except ValueError as e:
                    logger.warning("Aggregation skipped: %s", e)

                # Clear training buffer after successful round
                self._trainer.clear_training_buffer()

            elif external_updates:
                # No local data but have external updates — aggregate and apply
                try:
                    agg_result = self._aggregator.aggregate(external_updates)
                    result.aggregation = agg_result
                    self._trainer.apply_global_weights(agg_result.global_weights)
                except ValueError as e:
                    logger.warning("Aggregation skipped: %s", e)

            result.privacy_budget = self._dp_engine.get_budget_status()

        except Exception as e:
            result.error = str(e)
            logger.error("Federated round %d failed: %s", round_num, e)

        self._rounds.append(result)
        logger.info(
            "Federated round %d completed (local=%s, aggregation=%s)",
            round_num,
            result.local_update is not None,
            result.aggregation is not None,
        )

        return result

    def share_indicator(
        self,
        *,
        text: str,
        embedding: np.ndarray,
        mitre_tactic: str = "",
        affected_models: list[str] | None = None,
        confidence: float = 0.0,
        detection_source: str = "",
    ) -> SharedIndicator | None:
        """Share a threat indicator with DP noise.

        Returns None if privacy budget is exhausted.
        """
        if self._dp_engine.is_budget_exhausted:
            logger.warning("Cannot share indicator: privacy budget exhausted")
            return None

        return self._sharing.prepare_indicator(
            text=text,
            embedding=embedding,
            mitre_tactic=mitre_tactic,
            affected_models=affected_models,
            confidence=confidence,
            detection_source=detection_source,
        )

    def receive_indicators(self, indicators: list[SharedIndicator]) -> int:
        """Receive indicators from other instances. Returns count accepted."""
        return self._sharing.receive_batch(indicators)

    async def _scheduler_loop(self) -> None:
        """Background loop running federated rounds at configured interval."""
        while self._running:
            try:
                await asyncio.sleep(self._round_interval)
                if not self._running:
                    break
                if self._trainer.training_samples >= 2:
                    await self.run_federated_round()
                else:
                    logger.debug(
                        "Skipping scheduled round: only %d samples buffered",
                        self._trainer.training_samples,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduled federated round failed: %s", e)

    def start_scheduler(self) -> None:
        """Start the background federated round scheduler."""
        if self._running:
            return
        self._running = True
        try:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info(
                "Federated scheduler started (interval=%ds)", self._round_interval,
            )
        except RuntimeError:
            self._running = False
            logger.warning("No event loop — federated scheduler not started")

    def stop_scheduler(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            self._scheduler_task = None
        logger.info("Federated scheduler stopped")

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive federated intelligence status."""
        return {
            "instance_id": self._instance_id,
            "scheduler_running": self._running,
            "round_interval_seconds": self._round_interval,
            "total_rounds": len(self._rounds),
            "last_round": self._rounds[-1].to_dict() if self._rounds else None,
            "trainer": self._trainer.stats,
            "aggregator": self._aggregator.stats,
            "sharing": self._sharing.stats,
            "privacy": self._dp_engine.stats,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Summary statistics for /health endpoint."""
        return {
            "total_rounds": len(self._rounds),
            "scheduler_running": self._running,
            "training_samples_buffered": self._trainer.training_samples,
            "indicators_shared": self._sharing.stats["indicators_shared"],
            "indicators_received": self._sharing.stats["indicators_received"],
            "privacy_budget_remaining": round(
                max(0, self._dp_engine.stats["remaining_epsilon"]), 4
            ),
            "privacy_budget_exhausted": self._dp_engine.is_budget_exhausted,
        }
