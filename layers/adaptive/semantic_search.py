"""
Analyzer 2 — Semantic Similarity Engine (B-Cell / Antibody Analog)

Embeds the incoming prompt using sentence-transformers (all-MiniLM-L6-v2,
384-dim) and performs approximate nearest neighbor search against the Threat
Vault (FAISS HNSW index). Cosine similarity >0.85 indicates the prompt is
semantically similar to a known attack, even with completely different
wording. Each stored embedding is an antibody recognizing an antigen shape.

ASSUMED-BREACH POSTURE: This analyzer assumes L2 Innate missed the attack
and the injection classifier could also miss it. The FAISS index could be
poisoned with false embeddings by a compromised L3 or L4 — the analyzer
reports similarity scores, not verdicts. The threat vault tracks provenance
and confirmation status; unconfirmed entries carry reduced weight. An empty
or corrupted FAISS index degrades this analyzer to no-op but does not
compromise other layers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from aegis.config import AdaptiveConfig
from aegis.layers.memory.threat_vault import ThreatVault
from aegis.models.adaptive_result import (
    AdaptiveAnalysisResult,
    AnalyzerType,
    DCASignal,
    SignalType,
)
from aegis.models.scan_result import ThreatCategory

logger = logging.getLogger(__name__)


class SemanticSearchAnalyzer:
    """Semantic similarity search against the Threat Vault.

    Embeds the incoming prompt and searches the FAISS HNSW index for
    similar known attacks. Reports similarity scores and matched indicators.
    """

    def __init__(
        self,
        config: AdaptiveConfig,
        threat_vault: ThreatVault,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self._config = config
        self._vault = threat_vault
        self._embed_fn = embed_fn
        self._model = None
        self._online = False

        if embed_fn is None:
            self._load_model()

    def _load_model(self) -> None:
        """Attempt to load sentence-transformers model. Never raises."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._config.embedding_model)
            self._online = True
            logger.info("Semantic search model loaded: %s", self._config.embedding_model)
        except Exception as e:
            logger.warning(
                "Semantic search model offline: %s. Using vault deterministic embeddings.", e
            )
            self._online = False

    @property
    def is_online(self) -> bool:
        return self._online or self._embed_fn is not None

    def embed(self, text: str) -> list[float]:
        """Embed text using the configured embedding function or model."""
        if self._embed_fn is not None:
            return self._embed_fn(text)
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        # Fallback to vault's deterministic embedding
        return self._vault._deterministic_embedding(text)

    async def analyze(self, text: str) -> AdaptiveAnalysisResult:
        """Search the threat vault for semantically similar known attacks."""
        start = time.perf_counter()

        try:
            embedding = self.embed(text)
            results = self._vault.search(
                embedding,
                k=5,
                include_dormant=False,
            )

            elapsed = (time.perf_counter() - start) * 1000

            if not results:
                return AdaptiveAnalysisResult(
                    analyzer_id=AnalyzerType.SEMANTIC_SIMILARITY,
                    is_threat=False,
                    confidence=0.0,
                    details={"top_similarity": 0.0, "vault_size": self._vault.size},
                    dca_signals=[
                        DCASignal(
                            signal_type=SignalType.SAFE,
                            source="semantic_similarity",
                            value=0.8,
                            description="No similar threats in vault",
                        )
                    ],
                    latency_ms=elapsed,
                )

            top_indicator, top_sim = results[0]
            threshold = self._config.similarity_threshold  # 0.85

            is_threat = top_sim >= threshold
            # Scale confidence: 0.85 sim -> 0.70 confidence, 1.0 sim -> 1.0 confidence
            confidence = min(max((top_sim - 0.5) / 0.5, 0.0), 1.0) if top_sim > 0.5 else 0.0

            signals = []
            if top_sim >= threshold:
                signals.append(DCASignal(
                    signal_type=SignalType.PAMP,
                    source="semantic_similarity",
                    value=min(float(top_sim), 1.0),
                    description=f"Vault match: {top_indicator.indicator_id} sim={top_sim:.3f}",
                ))
                # Record the hit for lifecycle tracking
                self._vault.record_hit(top_indicator.indicator_id)
            elif top_sim >= 0.6:
                signals.append(DCASignal(
                    signal_type=SignalType.DANGER,
                    source="semantic_similarity",
                    value=min(float(top_sim), 1.0),
                    description=f"Moderate vault similarity: {top_sim:.3f}",
                ))
            else:
                signals.append(DCASignal(
                    signal_type=SignalType.SAFE,
                    source="semantic_similarity",
                    value=min(max(float(1.0 - top_sim), 0.0), 1.0),
                    description=f"Low vault similarity: {top_sim:.3f}",
                ))

            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.SEMANTIC_SIMILARITY,
                is_threat=is_threat,
                confidence=confidence,
                threat_category=top_indicator.threat_category if is_threat else ThreatCategory.UNKNOWN,
                details={
                    "top_similarity": top_sim,
                    "top_indicator_id": top_indicator.indicator_id,
                    "top_category": top_indicator.threat_category.value,
                    "matches_above_threshold": sum(1 for _, s in results if s >= threshold),
                    "vault_size": self._vault.size,
                },
                dca_signals=signals,
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Semantic search crashed: %s", e)
            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.SEMANTIC_SIMILARITY,
                is_threat=False,
                confidence=0.0,
                details={"error": str(e)},
                dca_signals=[
                    DCASignal(
                        signal_type=SignalType.DANGER,
                        source="semantic_similarity",
                        value=0.3,
                        description=f"Semantic search crash: {e}",
                    )
                ],
                latency_ms=elapsed,
            )
