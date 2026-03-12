"""
Indicator Sharing Service — Privacy-Preserving Threat Intelligence Exchange.

Biological Analog: Dendritic cells capture antigen fragments and present them
to other immune cells via MHC molecules. The antigen is processed — not the
original pathogen — ensuring the information shared is useful for recognition
without exposing the full infection context.

Shares anonymized threat indicators between AEGIS instances with differential
privacy noise on embeddings. Deduplication by content hash prevents bloat.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from aegis.services.federated.privacy import DifferentialPrivacyEngine

logger = logging.getLogger(__name__)


@dataclass
class SharedIndicator:
    """A threat indicator prepared for sharing across instances.

    Contains DP-noised embedding and metadata — never raw prompt text.
    """
    indicator_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    content_hash: str = ""
    mitre_tactic: str = ""
    affected_models: list[str] = field(default_factory=list)
    confidence: float = 0.0
    detection_source: str = ""
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    shared_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "content_hash": self.content_hash,
            "mitre_tactic": self.mitre_tactic,
            "affected_models": self.affected_models,
            "confidence": self.confidence,
            "detection_source": self.detection_source,
            "first_seen": self.first_seen.isoformat(),
            "shared_at": self.shared_at.isoformat(),
            "embedding_dim": len(self.embedding) if self.embedding is not None else 0,
        }


class IndicatorSharingService:
    """Privacy-preserving threat indicator sharing.

    Manages the creation, anonymization, and reception of threat indicators.
    All embeddings are noised with differential privacy before sharing.
    Deduplication uses content hashes to prevent duplicate indicators.

    Parameters:
        dp_engine: DifferentialPrivacyEngine for adding noise.
        instance_id: Identifier for this AEGIS instance.
    """

    def __init__(
        self,
        *,
        dp_engine: DifferentialPrivacyEngine,
        instance_id: str = "local",
    ) -> None:
        self._dp_engine = dp_engine
        self._instance_id = instance_id
        self._shared: list[SharedIndicator] = []
        self._received: list[SharedIndicator] = []
        self._seen_hashes: set[str] = set()

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute SHA-256 hash for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def prepare_indicator(
        self,
        *,
        text: str,
        embedding: np.ndarray,
        mitre_tactic: str = "",
        affected_models: list[str] | None = None,
        confidence: float = 0.0,
        detection_source: str = "",
    ) -> SharedIndicator:
        """Prepare a threat indicator for sharing.

        Adds DP noise to the embedding and computes a content hash.
        The raw text is NOT included in the shared indicator.

        Args:
            text: Raw prompt text (used only for hashing, not shared).
            embedding: Vector embedding of the threat.
            mitre_tactic: MITRE ATLAS tactic ID.
            affected_models: List of affected model families.
            confidence: Detection confidence score.
            detection_source: Which AEGIS layer detected this.

        Returns:
            SharedIndicator with DP-noised embedding.
        """
        content_hash = self.compute_content_hash(text)

        # Add DP noise to embedding before sharing
        noised_embedding = self._dp_engine.add_noise_to_embedding(embedding)

        indicator = SharedIndicator(
            embedding=noised_embedding,
            content_hash=content_hash,
            mitre_tactic=mitre_tactic,
            affected_models=affected_models or [],
            confidence=confidence,
            detection_source=detection_source,
        )

        self._shared.append(indicator)
        self._seen_hashes.add(content_hash)

        logger.info(
            "Prepared indicator %s for sharing (tactic=%s, confidence=%.2f)",
            indicator.indicator_id, mitre_tactic, confidence,
        )

        return indicator

    def receive_indicator(self, indicator: SharedIndicator) -> bool:
        """Receive a shared indicator from another instance.

        Deduplicates by content hash. Returns True if new, False if duplicate.

        Args:
            indicator: SharedIndicator from another AEGIS instance.

        Returns:
            True if indicator was new and accepted, False if duplicate.
        """
        if indicator.content_hash in self._seen_hashes:
            logger.debug(
                "Duplicate indicator %s (hash=%s), skipping",
                indicator.indicator_id, indicator.content_hash[:16],
            )
            return False

        self._received.append(indicator)
        self._seen_hashes.add(indicator.content_hash)

        logger.info(
            "Received new indicator %s (tactic=%s, confidence=%.2f)",
            indicator.indicator_id, indicator.mitre_tactic, indicator.confidence,
        )

        return True

    def receive_batch(self, indicators: list[SharedIndicator]) -> int:
        """Receive a batch of indicators. Returns count of new ones accepted."""
        accepted = 0
        for ind in indicators:
            if self.receive_indicator(ind):
                accepted += 1
        return accepted

    def get_shared_indicators(self) -> list[SharedIndicator]:
        """Get all indicators this instance has shared."""
        return list(self._shared)

    def get_received_indicators(self) -> list[SharedIndicator]:
        """Get all indicators received from other instances."""
        return list(self._received)

    def get_all_indicator_embeddings(self) -> list[np.ndarray]:
        """Get embeddings from all received indicators for vault integration."""
        return [ind.embedding for ind in self._received if ind.embedding is not None and len(ind.embedding) > 0]

    @property
    def stats(self) -> dict[str, Any]:
        """Sharing service statistics."""
        return {
            "instance_id": self._instance_id,
            "indicators_shared": len(self._shared),
            "indicators_received": len(self._received),
            "unique_hashes": len(self._seen_hashes),
        }
