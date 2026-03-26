"""
Federation Pipeline — Event Bus to Indicator Generation.

Subscribes to threat_detected events and automatically generates
STIX indicators for novel attacks (caught by L3 adaptive but missed
by L2 innate). Applies DP noise before sharing.

All processing is asynchronous — no latency added to the request path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FederationPipeline:
    """Orchestrates automatic indicator generation from detections.

    Parameters:
        federated: FederatedIntelligenceManager instance.
        indicator_registry: IndicatorRegistry instance.
        event_bus: Event bus for subscribing and publishing.
    """

    def __init__(
        self,
        *,
        federated: Any,
        indicator_registry: Any,
        event_bus: Any,
        vault: Any = None,
    ) -> None:
        self._federated = federated
        self._registry = indicator_registry
        self._event_bus = event_bus
        self._vault = vault
        self._running = False
        self._indicators_generated = 0
        self._events_processed = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "indicators_generated": self._indicators_generated,
            "events_processed": self._events_processed,
        }

    async def start(self) -> None:
        """Subscribe to threat events on the event bus."""
        if self._running:
            return
        self._running = True

        if self._event_bus:
            try:
                await self._event_bus.subscribe("threat_detected", self._on_threat_detected)
                logger.info("Federation pipeline started — subscribed to threat_detected")
            except Exception as e:
                logger.warning("Failed to subscribe to event bus: %s", e)
        else:
            logger.info("Federation pipeline started (no event bus)")

    async def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        logger.info("Federation pipeline stopped")

    async def _on_threat_detected(self, event: Any) -> None:
        """Handle a threat_detected event from the event bus."""
        if not self._running:
            return

        self._events_processed += 1

        try:
            payload = event.payload if hasattr(event, "payload") else event
            if not isinstance(payload, dict):
                return

            # Only generate indicators for novel attacks:
            # caught by adaptive (L3) but missed by innate (L2)
            is_novel = payload.get("is_novel", False)
            adaptive_caught = payload.get("adaptive_caught", False)

            if not (is_novel or adaptive_caught):
                return

            await self.generate_indicator_from_detection(payload)

        except Exception as e:
            logger.debug("Federation pipeline event handling error: %s", e)

    async def generate_indicator_from_detection(
        self, detection: dict[str, Any],
    ) -> str | None:
        """Generate a STIX indicator from a detection event.

        Returns the indicator ID if successful, None otherwise.
        """
        if not self._federated or not self._registry:
            return None

        try:
            from aegis.services.federated.stix_generator import generate_indicator

            # Get or generate embedding
            embedding = detection.get("embedding")
            if embedding is not None:
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float64)
            else:
                # Generate a deterministic pseudo-embedding from the detection
                text = detection.get("text", detection.get("prompt", ""))
                if text and self._vault and hasattr(self._vault, "_embed"):
                    try:
                        embedding = self._vault._embed(text)
                    except Exception:
                        embedding = None

                if embedding is None:
                    # Fallback: hash-based pseudo-embedding (384 dims like MiniLM)
                    import hashlib
                    seed_bytes = hashlib.sha256(
                        str(detection).encode("utf-8")
                    ).digest()
                    rng = np.random.RandomState(
                        int.from_bytes(seed_bytes[:4], "big")
                    )
                    embedding = rng.randn(384).astype(np.float64)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

            # Apply DP noise
            dp_engine = self._federated.dp_engine
            if dp_engine.is_budget_exhausted:
                logger.warning("DP budget exhausted — skipping indicator generation")
                return None

            noised_embedding = dp_engine.add_noise_to_embedding(embedding)

            # Generate STIX indicator
            mitre_tactic = detection.get("mitre_tactic", detection.get("threat_category", ""))
            indicator = generate_indicator(
                noised_embedding=noised_embedding,
                mitre_tactic=mitre_tactic,
                detection_layer=detection.get("detection_layer", "L3"),
                confidence=detection.get("confidence", 0.0),
                attack_category=detection.get("attack_category", detection.get("threat_category", "")),
                description=detection.get("description", ""),
            )

            # Store in registry
            ind_id = self._registry.add_indicator(indicator)
            self._indicators_generated += 1

            # Publish event
            if self._event_bus:
                try:
                    from aegis.services.event_bus import Event
                    await self._event_bus.publish(
                        "indicator_generated",
                        Event(channel="indicator_generated", payload={"indicator_id": ind_id}),
                    )
                except Exception:
                    pass

            logger.info(
                "Generated federation indicator %s (tactic=%s, confidence=%.2f)",
                ind_id, mitre_tactic, detection.get("confidence", 0),
            )
            return ind_id

        except Exception as e:
            logger.error("Failed to generate indicator: %s", e)
            return None
