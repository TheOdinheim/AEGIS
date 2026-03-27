"""
L4 — Threat Vault / FAISS Index Management (Memory B-Cell Analog)

Manages the FAISS HNSW vector index storing embeddings of all known attack
prompts. Sub-millisecond approximate nearest neighbor search at 1M+ vectors.
Supports real-time addition as attacks are confirmed. Three-tier storage:
Vector Index (Tier 1), Signature Database (Tier 2), Threat Intel Feeds
(Tier 3). Memory lifecycle: Acute -> Persistent -> Dormant.

Persistence: FAISS index saved to disk on every N updates (batched) and
loaded on startup. Metadata sidecar stored as JSON alongside the index.
Learned antibodies survive process restarts.

Maintenance cycle: Runs on startup and every 6 hours via background task.
Evaluates all indicators and transitions them through lifecycle phases:
  - Acute (0-30 days): Full payload, high monitoring priority
  - Persistent (promoted after 3+ distinct sources or analyst confirmation):
    Optimized storage, embedding retained, fast-path detection
  - Dormant (180+ days unseen): Excluded from active scanning, retained in
    vector index for instant reactivation on re-emergence

ASSUMED-BREACH POSTURE: The vault assumes its own contents may be poisoned.
A compromised adaptive layer could insert embeddings designed to:
  - Cause false negatives on future attacks (adversarial antibodies)
  - Cause false positives on legitimate traffic (autoimmune induction)
  - Inflate the index to degrade search performance (resource exhaustion)
Mitigations: provenance tracking, confirmation-weighted scoring, lifecycle
expiration, and index size monitoring. The vault never deletes — dormant
entries are excluded from active scanning but retained for reactivation,
preventing an attacker from waiting out a detection window.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aegis.config import MemoryConfig
from aegis.models.scan_result import ThreatCategory
from aegis.models.threat_indicator import (
    IndicatorSource,
    MemoryPhase,
    ThreatIndicator,
)

logger = logging.getLogger(__name__)

# Map category strings from seed JSON to ThreatCategory enum
_CATEGORY_MAP: dict[str, ThreatCategory] = {
    "prompt_injection": ThreatCategory.PROMPT_INJECTION,
    "system_prompt_extraction": ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
    "jailbreak": ThreatCategory.JAILBREAK,
    "pii_exfiltration": ThreatCategory.PII_EXFILTRATION,
    "model_extraction": ThreatCategory.MODEL_EXTRACTION,
    "data_poisoning": ThreatCategory.DATA_POISONING,
    "encoding_obfuscation": ThreatCategory.ENCODING_OBFUSCATION,
}

# Reverse map for serialization
_CATEGORY_REVERSE: dict[ThreatCategory, str] = {v: k for k, v in _CATEGORY_MAP.items()}


class ThreatVault:
    """FAISS HNSW vector index for threat indicator storage and search.

    Stores ThreatIndicator objects with their embeddings. Supports:
    - add(): Insert a new indicator with its embedding
    - search(): Find the k nearest neighbors by cosine similarity
    - load_seed(): Load initial seed threats from JSON
    - promote(): Move acute indicators to persistent when criteria met
    - demote(): Move indicators to dormant after inactivity
    - lifecycle_update(): Run full maintenance cycle
    - save_to_disk() / load_from_disk(): FAISS index persistence

    PostgreSQL dual-write: When a session_factory is set, all add/lifecycle
    operations are also persisted to the ``threat_indicators`` table.
    FAISS remains the primary index for sub-ms search; PostgreSQL provides
    durability across restarts and cluster-wide visibility.
    """

    def __init__(self, config: MemoryConfig | None = None, session_factory: Any | None = None):
        self._config = config or MemoryConfig()
        self._dim = self._config.faiss_dimension  # 384
        self._indicators: list[ThreatIndicator] = []
        self._index = None  # Lazy init
        self._embeddings: list[np.ndarray] = []
        self._adds_since_persist = 0
        self._last_update: datetime | None = None
        self._maintenance_task: asyncio.Task | None = None
        self._lock = threading.Lock()
        self._session_factory = session_factory
        self._init_index()

    def _init_index(self) -> None:
        """Initialize the FAISS HNSW index."""
        try:
            import faiss

            self._index = faiss.IndexHNSWFlat(self._dim, self._config.faiss_hnsw_m)
            self._index.hnsw.efSearch = self._config.faiss_ef_search
        except ImportError:
            logger.warning("FAISS not available — using brute-force numpy fallback")
            self._index = None

    @property
    def session_factory(self) -> Any | None:
        return self._session_factory

    @session_factory.setter
    def session_factory(self, value: Any | None) -> None:
        self._session_factory = value

    @property
    def size(self) -> int:
        return len(self._indicators)

    @property
    def last_update(self) -> datetime | None:
        return self._last_update

    def add(self, indicator: ThreatIndicator) -> None:
        """Add a threat indicator to the vault.

        The indicator must have a non-empty embedding of the correct dimension.
        Triggers batched persistence every N updates.
        """
        if not indicator.embedding or len(indicator.embedding) != self._dim:
            raise ValueError(
                f"Embedding must be {self._dim}-dim, got {len(indicator.embedding) if indicator.embedding else 0}"
            )

        vec = np.array(indicator.embedding, dtype=np.float32)
        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with self._lock:
            self._indicators.append(indicator)
            self._embeddings.append(vec)

            if self._index is not None:
                self._index.add(vec.reshape(1, -1))

            self._last_update = datetime.now(timezone.utc)
            self._adds_since_persist += 1

            # Batched persistence
            if self._adds_since_persist >= self._config.persist_every_n_updates:
                self._persist_if_configured()

        # PostgreSQL dual-write (fire-and-forget, outside lock)
        self._pg_upsert_indicator(indicator)

    def _pg_upsert_indicator(self, indicator: ThreatIndicator) -> None:
        """Fire-and-forget PostgreSQL upsert for an indicator."""
        if self._session_factory is None:
            return
        try:
            from aegis.services.audit_logger import upsert_threat_indicator
            asyncio.create_task(upsert_threat_indicator(
                session_factory=self._session_factory,
                indicator_id=indicator.indicator_id,
                mitre_tactic=indicator.mitre_atlas_id or "",
                threat_category=indicator.threat_category.value,
                confidence=indicator.confidence,
                source=indicator.source.value,
                prompt_hash=indicator.payload_hash or "",
                embedding_ref=str(len(self._indicators) - 1),
                status=indicator.phase.value,
                hit_count=indicator.frequency,
                source_count=indicator.seen_by_tenants,
            ))
        except (RuntimeError, ImportError):
            pass  # No event loop or missing module — skip silently

    def _pg_update_status(self, indicator_id: str, new_status: str) -> None:
        """Fire-and-forget PostgreSQL status update."""
        if self._session_factory is None:
            return
        try:
            from aegis.services.audit_logger import update_indicator_status
            asyncio.create_task(update_indicator_status(
                session_factory=self._session_factory,
                indicator_id=indicator_id,
                new_status=new_status,
            ))
        except (RuntimeError, ImportError):
            pass

    def _persist_if_configured(self) -> None:
        """Save to disk if persistence paths are configured. Resets counter."""
        try:
            self.save_to_disk()
            self._adds_since_persist = 0
        except Exception as e:
            logger.error("Auto-persist failed: %s", e)

    def search(
        self, query_embedding: list[float], k: int = 5, include_dormant: bool = False,
    ) -> list[tuple[ThreatIndicator, float]]:
        """Search for the k nearest threat indicators by cosine similarity.

        Returns list of (indicator, similarity_score) tuples sorted by
        descending similarity. Dormant indicators are excluded by default.
        Unconfirmed indicators have their similarity weighted by 0.5.
        """
        if not self._indicators:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Snapshot shared state under lock to avoid races with concurrent add()
        with self._lock:
            if self._index is not None and self._index.ntotal > 0:
                # FAISS HNSW uses L2 distance on normalized vectors
                # For normalized vectors: L2^2 = 2 - 2*cos_sim
                distances, indices = self._index.search(
                    query.reshape(1, -1), min(k * 3, self._index.ntotal)
                )
                indicator_snapshot = list(self._indicators)
                embedding_snapshot = None
            else:
                distances, indices = None, None
                indicator_snapshot = list(self._indicators)
                embedding_snapshot = list(self._embeddings)

        # Process candidates from snapshots (no lock needed)
        candidates = []
        if distances is not None and indices is not None:
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(indicator_snapshot):
                    continue
                indicator = indicator_snapshot[idx]
                if not include_dormant and indicator.phase == MemoryPhase.DORMANT:
                    continue
                # Convert L2 distance to cosine similarity for normalized vectors
                cos_sim = 1.0 - dist / 2.0
                weighted_sim = cos_sim * indicator.weight
                candidates.append((indicator, weighted_sim))
        else:
            # Brute-force numpy fallback
            for i, (indicator, emb) in enumerate(
                zip(indicator_snapshot, embedding_snapshot)
            ):
                if not include_dormant and indicator.phase == MemoryPhase.DORMANT:
                    continue
                cos_sim = float(np.dot(query, emb))
                weighted_sim = cos_sim * indicator.weight
                candidates.append((indicator, weighted_sim))

        # Sort by weighted similarity descending, return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def load_seed(
        self,
        seed_file: str | Path,
        embed_fn: Any = None,
    ) -> int:
        """Load seed threats from a JSON file.

        If embed_fn is provided, it is called with the attack text to generate
        embeddings. Otherwise, a deterministic hash-based pseudo-embedding is
        used (sufficient for testing, not for production similarity search).

        Returns the number of indicators loaded.
        """
        path = Path(seed_file)
        if not path.exists():
            logger.warning("Seed file not found: %s", path)
            return 0

        with path.open("r") as f:
            data = json.load(f)

        threats = data.get("threats", [])
        count = 0

        for entry in threats:
            text = entry.get("text", "")
            category_str = entry.get("category", "prompt_injection")
            category = _CATEGORY_MAP.get(category_str, ThreatCategory.PROMPT_INJECTION)

            if embed_fn is not None:
                embedding = embed_fn(text)
            else:
                embedding = self._deterministic_embedding(text)

            indicator = ThreatIndicator(
                indicator_id=entry.get("id", f"seed-{uuid.uuid4().hex[:8]}"),
                source=IndicatorSource.SEED,
                confirmed=True,
                threat_category=category,
                confidence=entry.get("confidence", 0.90),
                severity=entry.get("severity", 0.8),
                embedding=embedding,
                payload_hash=hashlib.sha256(text.encode()).hexdigest(),
                payload_summary=text[:200],
                mitre_atlas_id=category.value,
            )
            self.add(indicator)
            count += 1

        logger.info("Loaded %d seed threats from %s", count, path)
        return count

    def _deterministic_embedding(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from text hash.

        NOT suitable for real similarity search — only for testing and
        bootstrap when the sentence-transformer model is unavailable.
        """
        h = hashlib.sha256(text.encode()).digest()
        # Use hash bytes to seed a reproducible random vector
        rng = np.random.RandomState(
            int.from_bytes(h[:4], byteorder="big")
        )
        vec = rng.randn(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    def record_hit(self, indicator_id: str) -> None:
        """Record that an indicator matched an incoming request.

        Updates last_seen and increments frequency. Reactivates dormant
        indicators (a re-emerging attack should be actively scanned).
        """
        reactivated = False
        with self._lock:
            for ind in self._indicators:
                if ind.indicator_id == indicator_id:
                    ind.last_seen = datetime.now(timezone.utc)
                    ind.frequency += 1
                    if ind.phase == MemoryPhase.DORMANT:
                        ind.phase = MemoryPhase.ACUTE
                        ind.updated_at = datetime.now(timezone.utc)
                        reactivated = True
                        logger.info(
                            "Reactivated dormant indicator %s", indicator_id
                        )
                    self._last_update = datetime.now(timezone.utc)
                    # Dual-write hit to PostgreSQL
                    self._pg_upsert_indicator(ind)
                    break
        if reactivated:
            self._pg_update_status(indicator_id, MemoryPhase.ACUTE.value)

    # ------------------------------------------------------------------
    # Three-Phase Memory Lifecycle
    # ------------------------------------------------------------------

    def promote(self, indicator_id: str) -> bool:
        """Promote an Acute indicator to Persistent.

        Promotion criteria (any one sufficient):
        - Seen by >= promote_min_sources distinct sources/tenants
        - Confirmed by analyst (confirmed=True)
        - Age >= acute_memory_days AND frequency >= 2

        Returns True if promoted, False otherwise.
        """
        with self._lock:
            for ind in self._indicators:
                if ind.indicator_id != indicator_id:
                    continue
                if ind.phase != MemoryPhase.ACUTE:
                    return False

                now = datetime.now(timezone.utc)
                age_days = (now - ind.created_at).days
                should_promote = (
                    ind.seen_by_tenants >= self._config.promote_min_sources
                    or ind.confirmed
                    or (age_days >= self._config.acute_memory_days and ind.frequency >= 2)
                )

                if should_promote:
                    ind.phase = MemoryPhase.PERSISTENT
                    ind.updated_at = now
                    # Optimize storage: truncate payload for persistent phase
                    if len(ind.payload_summary) > 200:
                        ind.payload_summary = ind.payload_summary[:200]
                    self._last_update = now
                    logger.info(
                        "Promoted indicator %s to PERSISTENT (tenants=%d, confirmed=%s, age=%dd)",
                        indicator_id, ind.seen_by_tenants, ind.confirmed, age_days,
                    )
                    self._pg_update_status(indicator_id, MemoryPhase.PERSISTENT.value)
                    return True
                return False
        return False

    def demote(self, indicator_id: str) -> bool:
        """Demote an indicator to Dormant phase.

        Dormant indicators are excluded from active scanning but retained
        in the vector index for instant reactivation.

        Returns True if demoted, False otherwise.
        """
        with self._lock:
            for ind in self._indicators:
                if ind.indicator_id != indicator_id:
                    continue
                if ind.phase == MemoryPhase.DORMANT:
                    return False

                now = datetime.now(timezone.utc)
                ind.phase = MemoryPhase.DORMANT
                ind.updated_at = now
                self._last_update = now
                logger.info("Demoted indicator %s to DORMANT", indicator_id)
                self._pg_update_status(indicator_id, MemoryPhase.DORMANT.value)
                return True
        return False

    def lifecycle_update(self) -> dict[str, int]:
        """Run lifecycle transitions on all indicators.

        Acute -> Persistent: after acute_memory_days if seen from 3+ sources
            or confirmed, OR after acute_memory_days with frequency >= 2.
            Simple age-based promotion as fallback.
        Persistent -> Dormant: after dormant_memory_days unseen.
        Dormant -> Acute: handled by record_hit() on re-emergence.

        Returns counts of transitions.
        """
        now = datetime.now(timezone.utc)
        transitions = {"to_persistent": 0, "to_dormant": 0, "reactivated": 0}
        transitioned_ids: list[tuple[str, str]] = []  # (indicator_id, new_status)

        with self._lock:
            for ind in self._indicators:
                age_days = (now - ind.created_at).days

                if ind.phase == MemoryPhase.ACUTE:
                    # Check promotion criteria
                    should_promote = (
                        ind.seen_by_tenants >= self._config.promote_min_sources
                        or ind.confirmed
                        or (age_days >= self._config.acute_memory_days and ind.frequency >= 2)
                        or age_days >= self._config.acute_memory_days
                    )
                    if should_promote:
                        ind.phase = MemoryPhase.PERSISTENT
                        ind.updated_at = now
                        if len(ind.payload_summary) > 200:
                            ind.payload_summary = ind.payload_summary[:200]
                        transitions["to_persistent"] += 1
                        transitioned_ids.append((ind.indicator_id, MemoryPhase.PERSISTENT.value))

                elif ind.phase == MemoryPhase.PERSISTENT:
                    days_unseen = (now - ind.last_seen).days
                    if days_unseen >= self._config.dormant_memory_days:
                        ind.phase = MemoryPhase.DORMANT
                        ind.updated_at = now
                        transitions["to_dormant"] += 1
                        transitioned_ids.append((ind.indicator_id, MemoryPhase.DORMANT.value))

            if any(v > 0 for v in transitions.values()):
                self._last_update = now

        # Fire PostgreSQL status updates outside lock
        for ind_id, new_status in transitioned_ids:
            self._pg_update_status(ind_id, new_status)

        logger.info(
            "Lifecycle maintenance: %d→persistent, %d→dormant",
            transitions["to_persistent"], transitions["to_dormant"],
        )
        return transitions

    # ------------------------------------------------------------------
    # FAISS Index Persistence
    # ------------------------------------------------------------------

    def save_to_disk(
        self,
        index_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ) -> None:
        """Save FAISS index, embeddings, and metadata to disk.

        Storage layout:
            - FAISS binary index (if faiss available)
            - Embeddings as numpy .npy file (separate from JSON for size)
            - JSON metadata sidecar (no embeddings — ~200 bytes/indicator)

        Args:
            index_path: Override config path for index file.
            meta_path: Override config path for metadata JSON.
        """
        idx_path = Path(index_path or self._config.faiss_index_path)
        m_path = Path(meta_path or self._config.metadata_path)
        emb_path = m_path.with_suffix(".embeddings.npy")

        # Ensure parent directories exist
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            try:
                import faiss
                faiss.write_index(self._index, str(idx_path))
                logger.info("FAISS index saved to %s (%d vectors)", idx_path, self._index.ntotal)
            except ImportError:
                pass

        # Save embeddings as separate numpy file
        if self._embeddings:
            matrix = np.stack(self._embeddings)
            np.save(str(emb_path), matrix)
            logger.info("Embeddings saved to %s (%d vectors)", emb_path, len(self._embeddings))

        # Save metadata sidecar (without embeddings for compactness)
        meta_entries = []
        for ind in self._indicators:
            entry = {
                "indicator_id": ind.indicator_id,
                "created_at": ind.created_at.isoformat(),
                "updated_at": ind.updated_at.isoformat(),
                "phase": ind.phase.value,
                "source": ind.source.value,
                "confirmed": ind.confirmed,
                "threat_category": ind.threat_category.value,
                "confidence": ind.confidence,
                "severity": ind.severity,
                "payload_hash": ind.payload_hash,
                "payload_summary": ind.payload_summary,
                "mitre_atlas_id": ind.mitre_atlas_id,
                "first_seen": ind.first_seen.isoformat(),
                "last_seen": ind.last_seen.isoformat(),
                "frequency": ind.frequency,
                "seen_by_tenants": ind.seen_by_tenants,
                "affected_models": ind.affected_models,
                "stix_id": ind.stix_id,
                "metadata": ind.metadata,
            }
            meta_entries.append(entry)

        meta_doc = {
            "version": 2,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "dimension": self._dim,
            "count": len(meta_entries),
            "indicators": meta_entries,
        }

        with m_path.open("w") as f:
            json.dump(meta_doc, f, indent=2)

        logger.info("Vault metadata saved to %s (%d indicators)", m_path, len(meta_entries))

    def load_from_disk(
        self,
        index_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ) -> int:
        """Load FAISS index and metadata from disk.

        Fast path: If a FAISS binary index exists, load it via
        faiss.read_index() for O(1) loading. Fall back to per-indicator
        reconstruction only if the binary is missing or corrupted.

        Embeddings are loaded from a separate .npy file (v2 format).
        Backward compatible: if .npy is missing but JSON contains
        embeddings (v1 format), fall back to JSON-embedded vectors.

        Returns the number of indicators loaded. If no persisted data
        exists, returns 0 (caller should fall back to seed loading).
        """
        idx_path = Path(index_path or self._config.faiss_index_path)
        m_path = Path(meta_path or self._config.metadata_path)
        emb_path = m_path.with_suffix(".embeddings.npy")

        if not m_path.exists():
            logger.info("No persisted metadata at %s — starting fresh", m_path)
            return 0

        with m_path.open("r") as f:
            meta_doc = json.load(f)

        entries = meta_doc.get("indicators", [])
        if not entries:
            return 0

        # Clear current state
        self._indicators.clear()
        self._embeddings.clear()

        # --- Fast path: try loading FAISS binary index ---
        faiss_loaded = False
        if idx_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(idx_path))
                faiss_loaded = True
                logger.info(
                    "FAISS fast-path: loaded %d vectors from %s",
                    self._index.ntotal, idx_path,
                )
            except (ImportError, Exception) as e:
                logger.warning("FAISS fast-path failed, rebuilding: %s", e)
                self._init_index()
        else:
            self._init_index()

        # --- Load embeddings from numpy file (v2) or JSON fallback (v1) ---
        embeddings_matrix = None
        if emb_path.exists():
            try:
                embeddings_matrix = np.load(str(emb_path))
                logger.info("Loaded embeddings from %s (%d vectors)", emb_path, len(embeddings_matrix))
            except Exception as e:
                logger.warning("Failed to load embeddings .npy: %s", e)

        count = 0
        for i, entry in enumerate(entries):
            # Get embedding: prefer numpy file, fall back to JSON
            embedding: list[float] | None = None
            if embeddings_matrix is not None and i < len(embeddings_matrix):
                embedding = embeddings_matrix[i].tolist()
            else:
                embedding = entry.get("embedding", [])

            if not embedding or len(embedding) != self._dim:
                continue

            # Parse enums safely
            try:
                phase = MemoryPhase(entry["phase"])
            except (ValueError, KeyError):
                phase = MemoryPhase.ACUTE
            try:
                source = IndicatorSource(entry["source"])
            except (ValueError, KeyError):
                source = IndicatorSource.SEED
            try:
                category = ThreatCategory(entry["threat_category"])
            except (ValueError, KeyError):
                category = ThreatCategory.PROMPT_INJECTION

            indicator = ThreatIndicator(
                indicator_id=entry["indicator_id"],
                created_at=datetime.fromisoformat(entry["created_at"]),
                updated_at=datetime.fromisoformat(entry["updated_at"]),
                phase=phase,
                source=source,
                confirmed=entry.get("confirmed", False),
                threat_category=category,
                confidence=entry.get("confidence", 0.5),
                severity=entry.get("severity", 0.5),
                embedding=embedding,
                payload_hash=entry.get("payload_hash", ""),
                payload_summary=entry.get("payload_summary", ""),
                mitre_atlas_id=entry.get("mitre_atlas_id", ""),
                first_seen=datetime.fromisoformat(entry["first_seen"]),
                last_seen=datetime.fromisoformat(entry["last_seen"]),
                frequency=entry.get("frequency", 1),
                seen_by_tenants=entry.get("seen_by_tenants", 1),
                affected_models=entry.get("affected_models", []),
                stix_id=entry.get("stix_id"),
                metadata=entry.get("metadata", {}),
            )

            # If FAISS was loaded from binary, skip re-adding to index
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._indicators.append(indicator)
            self._embeddings.append(vec)
            if not faiss_loaded and self._index is not None:
                self._index.add(vec.reshape(1, -1))
            count += 1

        self._last_update = datetime.now(timezone.utc)
        self._adds_since_persist = 0  # Just loaded, no pending saves
        logger.info("Loaded %d indicators from disk", count)
        return count

    async def load_from_postgres(self) -> int:
        """Load threat indicator metadata from PostgreSQL.

        Loads metadata (IDs, status, hit counts) but NOT embeddings —
        those come from the FAISS index or numpy files. This method
        syncs lifecycle status and hit counts from PostgreSQL so that
        indicators persist across restarts even without disk persistence.

        Returns the number of indicators synced.
        """
        if self._session_factory is None:
            return 0

        try:
            from aegis.services.audit_logger import load_threat_indicators
            rows = await load_threat_indicators(self._session_factory)
            if not rows:
                return 0

            synced = 0
            pg_ids = {r["indicator_id"] for r in rows}
            pg_map = {r["indicator_id"]: r for r in rows}

            # Sync existing in-memory indicators with PostgreSQL state
            with self._lock:
                for ind in self._indicators:
                    if ind.indicator_id in pg_map:
                        row = pg_map[ind.indicator_id]
                        # Sync lifecycle status from PostgreSQL
                        try:
                            pg_phase = MemoryPhase(row["status"])
                            if pg_phase != ind.phase:
                                ind.phase = pg_phase
                        except (ValueError, KeyError):
                            pass
                        # Sync hit count (take max)
                        ind.frequency = max(ind.frequency, row.get("hit_count", 0))
                        ind.seen_by_tenants = max(
                            ind.seen_by_tenants, row.get("source_count", 1)
                        )
                        synced += 1

            logger.info("Synced %d indicators from PostgreSQL (%d in DB)", synced, len(rows))
            return synced
        except Exception as e:
            logger.warning("PostgreSQL indicator load failed: %s", e)
            return 0

    def index_size_bytes(self) -> int:
        """Return the on-disk size of the FAISS index or embeddings file."""
        idx_path = Path(self._config.faiss_index_path)
        if idx_path.exists():
            return idx_path.stat().st_size
        # Check embeddings .npy (v2 format — next to metadata file)
        m_path = Path(self._config.metadata_path)
        emb_path = m_path.with_suffix(".embeddings.npy")
        if emb_path.exists():
            return emb_path.stat().st_size
        # Legacy: check numpy fallback at idx_path + ".npy"
        legacy_npy = Path(str(idx_path) + ".npy")
        if legacy_npy.exists():
            return legacy_npy.stat().st_size
        return 0

    # ------------------------------------------------------------------
    # Vault Statistics (for /v1/vault/stats endpoint)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive vault statistics for the health dashboard."""
        with self._lock:
            total = len(self._indicators)

            # Count per lifecycle phase
            phase_counts = {phase.value: 0 for phase in MemoryPhase}
            for ind in self._indicators:
                phase_counts[ind.phase.value] += 1

            # Count per threat category
            category_counts: dict[str, int] = {}
            for ind in self._indicators:
                cat = ind.threat_category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # Top 5 most frequently matched indicators
            sorted_by_freq = sorted(
                self._indicators, key=lambda x: x.frequency, reverse=True
            )
            top_matched = [
                {
                    "indicator_id": ind.indicator_id,
                    "frequency": ind.frequency,
                    "threat_category": ind.threat_category.value,
                    "phase": ind.phase.value,
                    # RT-P3B-005: Redact payload text — expose only length
                    "payload_length": len(ind.payload_summary),
                }
                for ind in sorted_by_freq[:5]
            ]

        return {
            "total_indicators": total,
            "phase_counts": phase_counts,
            "category_counts": category_counts,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "index_size_bytes": self.index_size_bytes(),
            "top_matched_indicators": top_matched,
        }

    # ------------------------------------------------------------------
    # Indicator Retrieval
    # ------------------------------------------------------------------

    def get_indicators(
        self, phase: MemoryPhase | None = None
    ) -> list[ThreatIndicator]:
        """Return indicators, optionally filtered by phase."""
        if phase is None:
            return list(self._indicators)
        return [i for i in self._indicators if i.phase == phase]

    def get_indicator(self, indicator_id: str) -> ThreatIndicator | None:
        """Return a single indicator by ID, or None if not found."""
        for ind in self._indicators:
            if ind.indicator_id == indicator_id:
                return ind
        return None

    # ------------------------------------------------------------------
    # Background Maintenance Task
    # ------------------------------------------------------------------

    async def start_maintenance_loop(self) -> None:
        """Start the background maintenance cycle.

        Runs lifecycle_update() immediately, then every maintenance_interval_hours.
        Saves to disk after each maintenance cycle.
        """
        interval = self._config.maintenance_interval_hours * 3600

        # Run immediately on startup
        self.lifecycle_update()
        self._persist_if_configured()
        logger.info("Initial maintenance cycle complete")

        while True:
            await asyncio.sleep(interval)
            try:
                transitions = self.lifecycle_update()
                self._persist_if_configured()
                logger.info("Maintenance cycle complete: %s", transitions)
            except Exception as e:
                logger.error("Maintenance cycle failed: %s", e)

    def start_maintenance(self) -> None:
        """Schedule the maintenance loop as a background asyncio task.

        Safe to call from synchronous code during startup — the task runs
        in the current event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._maintenance_task = loop.create_task(
                    self.start_maintenance_loop()
                )
            else:
                logger.info("No running event loop — maintenance will be started manually")
        except RuntimeError:
            logger.info("No event loop — maintenance will be started manually")

    def stop_maintenance(self) -> None:
        """Cancel the background maintenance task."""
        if self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            self._maintenance_task = None
