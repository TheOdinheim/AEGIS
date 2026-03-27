"""
STIX/TAXII Threat Intelligence Ingestion & Export — L4 Tier 3

Parses STIX 2.1 JSON bundles containing AI-specific threat indicators,
embeds them via MiniLM, and stores them in the Threat Vault (L4) as
antibodies. Enables the future L8 federated intelligence network by
providing STIX export with differential privacy on embeddings.

STIX 2.1 Extension: AEGIS extends the standard ``indicator`` object with
an ``x_aegis_embedding`` property containing a pre-computed 384-dim
sentence-transformer vector. When present, ingestion skips re-embedding
and uses the supplied vector directly (faster, preserves original model).

Deduplication: Indicators with the same ``payload_hash`` (SHA-256 of
attack text) are not duplicated — instead the existing indicator's
``hit_count`` and ``last_seen`` are updated.

Differential Privacy Export: When exporting indicators for federated
sharing, Gaussian noise (calibrated to epsilon) is added to embeddings.
This prevents reconstruction of the original attack text from the
embedding while preserving enough semantic signal for similarity search.

ASSUMED-BREACH POSTURE: Ingested STIX bundles may be adversarial. A
poisoned feed could inject false indicators to cause FPs or create blind
spots for FNs. All ingested indicators are marked ``confirmed=False``
and carry reduced weight (50%) until human review. Source tracking via
``IndicatorSource.STIX_FEED`` enables targeted revocation if a feed is
compromised.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aegis.layers.memory.threat_vault import ThreatVault
from aegis.models.scan_result import ThreatCategory
from aegis.models.threat_indicator import (
    IndicatorSource,
    MemoryPhase,
    ThreatIndicator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MITRE ATLAS tactic ID → ThreatCategory mapping
# ---------------------------------------------------------------------------

_MITRE_TO_CATEGORY: dict[str, ThreatCategory] = {
    "AML.T0051": ThreatCategory.PROMPT_INJECTION,
    "AML.T0051.001": ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
    "AML.T0054": ThreatCategory.JAILBREAK,
    "AML.T0048": ThreatCategory.PII_EXFILTRATION,
    "AML.T0044": ThreatCategory.MODEL_EXTRACTION,
    "AML.T0020": ThreatCategory.DATA_POISONING,
    "AML.T0015": ThreatCategory.ENCODING_OBFUSCATION,
}

# Label strings that may appear in STIX indicator.labels
_LABEL_TO_CATEGORY: dict[str, ThreatCategory] = {
    "prompt-injection": ThreatCategory.PROMPT_INJECTION,
    "system-prompt-extraction": ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
    "jailbreak": ThreatCategory.JAILBREAK,
    "pii-exfiltration": ThreatCategory.PII_EXFILTRATION,
    "model-extraction": ThreatCategory.MODEL_EXTRACTION,
    "data-poisoning": ThreatCategory.DATA_POISONING,
    "encoding-obfuscation": ThreatCategory.ENCODING_OBFUSCATION,
}


def _resolve_category(
    labels: list[str] | None,
    mitre_id: str = "",
) -> ThreatCategory:
    """Resolve a ThreatCategory from STIX labels or MITRE ATLAS ID."""
    if mitre_id and mitre_id in _MITRE_TO_CATEGORY:
        return _MITRE_TO_CATEGORY[mitre_id]
    for lbl in (labels or []):
        lbl_lower = lbl.lower()
        if lbl_lower in _LABEL_TO_CATEGORY:
            return _LABEL_TO_CATEGORY[lbl_lower]
        # Check MITRE IDs in labels
        if lbl.upper() in _MITRE_TO_CATEGORY:
            return _MITRE_TO_CATEGORY[lbl.upper()]
    return ThreatCategory.PROMPT_INJECTION


class ThreatIntelManager:
    """STIX 2.1 threat intelligence ingestion and export manager.

    Parses STIX bundles, embeds indicator descriptions via MiniLM,
    stores them in the Threat Vault, and can export vault contents
    as STIX bundles with differential privacy noise on embeddings.
    """

    def __init__(
        self,
        threat_vault: ThreatVault,
        embed_fn: Callable[[str], list[float]] | None = None,
        session_factory: Any | None = None,
        dp_epsilon: float = 3.0,
    ):
        self._vault = threat_vault
        self._embed_fn = embed_fn
        self._session_factory = session_factory
        self._dp_epsilon = dp_epsilon

        # Stats
        self._total_ingested: int = 0
        self._total_deduplicated: int = 0
        self._total_exported: int = 0
        self._last_ingestion: datetime | None = None
        self._sources: dict[str, int] = {}  # source_name -> count

    # ------------------------------------------------------------------
    # STIX Bundle Validation
    # ------------------------------------------------------------------

    def _validate_bundle(self, bundle: Any) -> list[dict]:
        """Validate a STIX 2.1 bundle and extract indicator objects.

        Raises ValueError on invalid bundle structure.
        Returns list of indicator objects.
        """
        if not isinstance(bundle, dict):
            raise ValueError("STIX bundle must be a JSON object (dict)")

        if bundle.get("type") != "bundle":
            raise ValueError(
                f"Expected type='bundle', got type='{bundle.get('type', '<missing>')}'"
            )

        objects = bundle.get("objects")
        if objects is None:
            raise ValueError("STIX bundle missing 'objects' array")

        if not isinstance(objects, list):
            raise ValueError("STIX bundle 'objects' must be a list")

        # Extract only indicator objects
        indicators = [
            obj for obj in objects
            if isinstance(obj, dict) and obj.get("type") == "indicator"
        ]
        return indicators

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_stix_bundle(self, bundle: dict) -> int:
        """Ingest a STIX 2.1 JSON bundle.

        Parses indicator objects, generates embeddings, and stores them
        in the Threat Vault. Deduplicates by payload_hash.

        Args:
            bundle: A parsed STIX 2.1 bundle dict.

        Returns:
            Count of new indicators added (excludes deduplicated).

        Raises:
            ValueError: If the bundle structure is invalid.
        """
        indicators = self._validate_bundle(bundle)

        if not indicators:
            return 0

        source_name = bundle.get("id", "unknown-feed")
        new_count = 0

        for stix_ind in indicators:
            try:
                added = self._ingest_indicator(stix_ind, source_name)
                if added:
                    new_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to ingest indicator %s: %s",
                    stix_ind.get("id", "unknown"), e,
                )

        if new_count > 0:
            self._total_ingested += new_count
            self._last_ingestion = datetime.now(timezone.utc)
            self._sources[source_name] = (
                self._sources.get(source_name, 0) + new_count
            )

        logger.info(
            "Ingested %d new indicators from STIX bundle (source=%s)",
            new_count, source_name,
        )
        return new_count

    def _ingest_indicator(
        self, stix_ind: dict, source_name: str,
    ) -> bool:
        """Ingest a single STIX indicator object.

        Returns True if a new indicator was created, False if deduplicated.
        """
        # RT-P3B-006: Sanitize STIX fields to prevent stored XSS
        for field in ("name", "description"):
            if field in stix_ind and isinstance(stix_ind[field], str):
                stix_ind[field] = re.sub(r"<[^>]+>", "", stix_ind[field])

        # Extract attack text from pattern or description
        pattern_text = self._extract_pattern_text(stix_ind)
        description = stix_ind.get("description", "")
        text_for_embedding = pattern_text or description

        if not text_for_embedding:
            logger.debug("Skipping indicator with no pattern/description")
            return False

        # Compute payload hash for deduplication
        payload_hash = hashlib.sha256(
            text_for_embedding.encode("utf-8")
        ).hexdigest()

        # Deduplication check
        existing = self._find_by_hash(payload_hash)
        if existing is not None:
            existing.last_seen = datetime.now(timezone.utc)
            existing.frequency += 1
            self._total_deduplicated += 1
            self._vault.record_hit(existing.indicator_id)
            logger.debug(
                "Deduplicated indicator %s (hash=%s, hits=%d)",
                existing.indicator_id, payload_hash[:12], existing.frequency,
            )
            return False

        # Resolve threat category
        labels = stix_ind.get("labels", [])
        mitre_id = ""
        for ext_key in ("x_mitre_atlas_id", "x_aegis_mitre_id"):
            if ext_key in stix_ind:
                mitre_id = stix_ind[ext_key]
                break
        # Also check labels for MITRE IDs
        for lbl in labels:
            if lbl.startswith("AML."):
                mitre_id = lbl
                break

        category = _resolve_category(labels, mitre_id)

        # Get or generate embedding
        embedding = self._get_embedding(stix_ind, text_for_embedding)

        # Extract confidence
        confidence = stix_ind.get("confidence", 80) / 100.0
        if confidence > 1.0:
            confidence = confidence / 100.0  # Handle 0-100 scale

        # Extract affected models
        affected_models = stix_ind.get("x_aegis_affected_models", [])

        # Create ThreatIndicator
        stix_id = stix_ind.get("id", f"indicator--{uuid.uuid4()}")
        indicator = ThreatIndicator(
            indicator_id=f"stix-{uuid.uuid4().hex[:8]}",
            source=IndicatorSource.STIX_FEED,
            confirmed=False,  # Never auto-confirm external feeds
            threat_category=category,
            confidence=min(confidence, 1.0),
            severity=min(confidence * 0.8, 1.0),
            embedding=embedding,
            payload_hash=payload_hash,
            payload_summary=text_for_embedding[:200],
            mitre_atlas_id=mitre_id or category.value,
            stix_id=stix_id,
            affected_models=affected_models,
            metadata={
                "stix_source": source_name,
                "stix_created": stix_ind.get("created", ""),
                "stix_modified": stix_ind.get("modified", ""),
                "description": description[:500] if description else "",
                "mitigation": stix_ind.get("x_aegis_mitigation", ""),
            },
        )

        self._vault.add(indicator)
        return True

    def _extract_pattern_text(self, stix_ind: dict) -> str:
        """Extract attack text from a STIX indicator's pattern field.

        Supports standard STIX pattern and AEGIS-specific extensions.
        """
        # Check AEGIS extension first
        aegis_pattern = stix_ind.get("x_aegis_pattern_text", "")
        if aegis_pattern:
            return aegis_pattern

        # Try standard STIX pattern
        pattern = stix_ind.get("pattern", "")
        if not pattern:
            return ""

        # STIX patterns look like: [artifact:payload_bin = 'attack text here']
        # or custom: [x-aegis-prompt:value = 'attack text']
        # Extract the quoted value
        if "'" in pattern:
            parts = pattern.split("'")
            if len(parts) >= 2:
                return parts[1]

        return pattern

    def _get_embedding(
        self, stix_ind: dict, text: str,
    ) -> list[float]:
        """Get embedding from STIX extension or generate via embed_fn."""
        # Check for pre-computed AEGIS embedding
        pre_computed = stix_ind.get("x_aegis_embedding")
        if pre_computed and isinstance(pre_computed, list):
            if len(pre_computed) == self._vault._dim:
                return pre_computed

        # Generate embedding
        if self._embed_fn is not None:
            return self._embed_fn(text)

        # Fallback: deterministic hash-based embedding
        return self._vault._deterministic_embedding(text)

    def _find_by_hash(self, payload_hash: str) -> ThreatIndicator | None:
        """Find an existing indicator by payload hash."""
        for ind in self._vault.get_indicators():
            if ind.payload_hash == payload_hash:
                return ind
        return None

    # ------------------------------------------------------------------
    # File Ingestion
    # ------------------------------------------------------------------

    async def ingest_from_file(self, path: Path) -> int:
        """Load and ingest a STIX 2.1 JSON file.

        Args:
            path: Path to the STIX JSON file.

        Returns:
            Count of new indicators ingested.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file contains invalid STIX.
        """
        if not path.exists():
            raise FileNotFoundError(f"STIX feed file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            bundle = json.load(f)

        return await self.ingest_stix_bundle(bundle)

    async def ingest_directory(self, directory: Path) -> int:
        """Ingest all .json files from a directory.

        Returns total count of new indicators across all files.
        """
        if not directory.is_dir():
            return 0

        total = 0
        for json_file in sorted(directory.glob("*.json")):
            try:
                count = await self.ingest_from_file(json_file)
                total += count
                logger.info(
                    "Ingested %d indicators from %s", count, json_file.name,
                )
            except Exception as e:
                logger.error("Failed to ingest %s: %s", json_file.name, e)

        return total

    # ------------------------------------------------------------------
    # STIX Export with Differential Privacy
    # ------------------------------------------------------------------

    async def export_stix_bundle(
        self,
        since: datetime | None = None,
    ) -> dict:
        """Export threat indicators as a STIX 2.1 bundle.

        Applies differential privacy noise to embeddings before export
        to prevent prompt reconstruction from embeddings.

        Args:
            since: If provided, only export indicators created after
                this timestamp.

        Returns:
            A valid STIX 2.1 bundle dict.
        """
        indicators = self._vault.get_indicators()

        if since is not None:
            indicators = [
                ind for ind in indicators
                if ind.created_at >= since
            ]

        stix_objects = []
        for ind in indicators:
            stix_obj = self._indicator_to_stix(ind)
            stix_objects.append(stix_obj)

        self._total_exported += len(stix_objects)

        return {
            "type": "bundle",
            "id": f"bundle--{uuid.uuid4()}",
            "spec_version": "2.1",
            "created": datetime.now(timezone.utc).isoformat() + "Z",
            "objects": stix_objects,
        }

    def _indicator_to_stix(self, ind: ThreatIndicator) -> dict:
        """Convert a ThreatIndicator to a STIX 2.1 indicator object.

        Applies differential privacy noise to the embedding.
        """
        # Apply DP noise to embedding
        noisy_embedding = self._apply_dp_noise(ind.embedding)

        labels = [ind.threat_category.value]
        if ind.mitre_atlas_id:
            labels.append(ind.mitre_atlas_id)

        stix_id = ind.stix_id or f"indicator--{uuid.uuid4()}"

        return {
            "type": "indicator",
            "spec_version": "2.1",
            "id": stix_id,
            "created": ind.created_at.isoformat() + "Z",
            "modified": ind.updated_at.isoformat() + "Z",
            "name": f"AEGIS Threat: {ind.threat_category.value}",
            "description": ind.payload_summary,
            "pattern": f"[x-aegis-prompt:value = '{ind.payload_hash}']",
            "pattern_type": "x-aegis-hash",
            "valid_from": ind.first_seen.isoformat() + "Z",
            "labels": labels,
            "confidence": int(ind.confidence * 100),
            "x_aegis_embedding": noisy_embedding,
            "x_aegis_affected_models": ind.affected_models,
            "x_aegis_mitre_id": ind.mitre_atlas_id,
            "x_aegis_phase": ind.phase.value,
            "x_aegis_frequency": ind.frequency,
            "x_aegis_severity": ind.severity,
            "x_aegis_mitigation": ind.metadata.get("mitigation", ""),
        }

    def _apply_dp_noise(self, embedding: list[float]) -> list[float]:
        """Apply differential privacy Gaussian noise to an embedding.

        Uses the Gaussian mechanism calibrated to the configured epsilon.
        Sensitivity is 1.0 (unit-normalized embeddings). Noise scale
        sigma = sensitivity / epsilon.

        After adding noise, the vector is re-normalized to unit length
        so it can be used directly for cosine similarity search.
        """
        if not embedding:
            return embedding

        vec = np.array(embedding, dtype=np.float64)
        sensitivity = 1.0
        sigma = sensitivity / self._dp_epsilon

        noise = np.random.normal(0, sigma, size=len(vec))
        noisy = vec + noise

        # Re-normalize to unit length
        norm = np.linalg.norm(noisy)
        if norm > 0:
            noisy = noisy / norm

        return noisy.tolist()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Return feed statistics for the /v1/threat-intel/stats endpoint."""
        vault_stats = self._vault.get_stats()
        return {
            "total_ingested": self._total_ingested,
            "total_deduplicated": self._total_deduplicated,
            "total_exported": self._total_exported,
            "last_ingestion": (
                self._last_ingestion.isoformat()
                if self._last_ingestion else None
            ),
            "sources": dict(self._sources),
            "vault_size": vault_stats["total_indicators"],
            "vault_phase_counts": vault_stats["phase_counts"],
        }
