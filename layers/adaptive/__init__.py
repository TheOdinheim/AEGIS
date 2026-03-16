"""
L3 — Adaptive Analysis (T-Cells / B-Cells / Clonal Selection)

Asynchronous deep ML-based analysis. 10-50ms. Five analyzer modules run
concurrently with the model call.

ASSUMED-BREACH POSTURE: This layer assumes both L1 Barrier and L2 Innate
have been fully compromised. Attacks that evade regex, blocklists, schema
validation, and token guards are EXPECTED here — that is this layer's
entire purpose. The DeBERTa classifier, semantic similarity engine, and
behavioral analyzer each operate independently. A compromised innate layer
could report false negatives — adaptive analysis re-examines the raw prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from aegis.config import AdaptiveConfig, MemoryConfig
from aegis.layers.adaptive.behavioral import BehavioralAnalyzer
from aegis.layers.adaptive.injection_classifier import InjectionClassifier
from aegis.layers.adaptive.margin_booster import ConfidenceMarginBooster
from aegis.layers.adaptive.multi_turn import MultiTurnAnalyzer
from aegis.layers.adaptive.semantic_search import SemanticSearchAnalyzer
from aegis.layers.memory.threat_vault import ThreatVault
from aegis.models.adaptive_result import (
    AdaptiveAnalysisReport,
    AdaptiveAnalysisResult,
    AnalyzerType,
    DCASignal,
    SignalType,
)
from aegis.models.request_context import RequestContext
from aegis.models.scan_result import InnateScanReport, ThreatCategory
from aegis.models.threat_indicator import IndicatorSource, ThreatIndicator

logger = logging.getLogger(__name__)


def _compute_mcav(signals: list[DCASignal]) -> float:
    """Compute the Mature Context Antigen Value from DCA signals.

    DCA fusion formula:
        MCAV = (sum(PAMP * weight) + sum(DANGER * weight)) /
               (sum(PAMP * weight) + sum(DANGER * weight) + sum(SAFE * weight))

    PAMPs are weighted 3x, danger 2x, safe 1x. This ensures confirmed
    threat signals dominate the score while safe signals can only dilute,
    never cancel, strong threat evidence.

    Returns 0.0-1.0 where >0.7 is anomalous and >0.9 is definitive threat.
    """
    pamp_sum = 0.0
    danger_sum = 0.0
    safe_sum = 0.0

    for signal in signals:
        if signal.signal_type == SignalType.PAMP:
            pamp_sum += signal.value * 3.0
        elif signal.signal_type == SignalType.DANGER:
            danger_sum += signal.value * 2.0
        elif signal.signal_type == SignalType.SAFE:
            safe_sum += signal.value * 1.0

    threat_numerator = pamp_sum + danger_sum
    total = threat_numerator + safe_sum

    if total == 0:
        return 0.0

    return min(threat_numerator / total, 1.0)


class AdaptiveAnalysisLayer:
    """L3 Adaptive Analysis — orchestrates all analyzers with DCA fusion.

    Runs injection classifier, semantic similarity, and behavioral analyzer
    concurrently. Fuses their DCA signals into an MCAV score. When adaptive
    catches an attack that innate missed, triggers the antibody generation
    learning loop (stores the new attack in the Threat Vault).
    """

    def __init__(
        self,
        config: AdaptiveConfig,
        threat_vault: ThreatVault,
        embed_fn: Callable[[str], list[float]] | None = None,
        skip_model_load: bool = False,
        on_antibody: Callable[[str, str], None] | None = None,
    ):
        self._config = config
        self._vault = threat_vault
        self._on_antibody = on_antibody

        if skip_model_load:
            self._classifier = InjectionClassifier.__new__(InjectionClassifier)
            self._classifier._model_name = config.injection_model
            self._classifier._pipeline = None
            self._classifier._online = False
        else:
            self._classifier = InjectionClassifier(config.injection_model)

        self._semantic = SemanticSearchAnalyzer(config, threat_vault, embed_fn=embed_fn)
        self._behavioral = BehavioralAnalyzer(config)
        self._multi_turn = MultiTurnAnalyzer(config, embed_fn=embed_fn)
        self._margin_booster = ConfidenceMarginBooster()

    @property
    def classifier(self) -> InjectionClassifier:
        return self._classifier

    @property
    def semantic_search(self) -> SemanticSearchAnalyzer:
        return self._semantic

    @property
    def behavioral(self) -> BehavioralAnalyzer:
        return self._behavioral

    @property
    def multi_turn(self) -> MultiTurnAnalyzer:
        return self._multi_turn

    @property
    def threat_vault(self) -> ThreatVault:
        return self._vault

    async def analyze(
        self,
        context: RequestContext,
        innate_report: InnateScanReport | None = None,
    ) -> AdaptiveAnalysisReport:
        """Run all adaptive analyzers and produce a fused report.

        Args:
            context: The request context from L1.
            innate_report: Optional L2 innate scan report. Used to determine
                if this is a novel attack (adaptive caught, innate missed).
        """
        start = time.perf_counter()
        prompt = context.last_user_message or context.prompt_text

        # Run analyzers concurrently (4 analyzers)
        classifier_result, semantic_result, behavioral_result, multi_turn_result = (
            await asyncio.gather(
                self._classifier.analyze(prompt),
                self._semantic.analyze(prompt),
                self._behavioral.analyze(context),
                self._multi_turn.analyze(context, innate_report=innate_report),
            )
        )

        # Margin boosting for fragile DeBERTa detections (Finding 2 hardening)
        if (
            classifier_result.is_threat
            and 0.85 <= classifier_result.confidence <= 0.95
        ):
            embed_fn = self._semantic.embed if self._semantic else None
            boost_result = await self._margin_booster.boost(
                prompt,
                classifier_result.confidence,
                threat_vault=self._vault,
                embed_fn=embed_fn,
            )
            if boost_result.boost_amount > 0:
                logger.info(
                    "Margin boost: %.4f → %.4f (strategies: %s)",
                    boost_result.original_confidence,
                    boost_result.boosted_confidence,
                    ", ".join(boost_result.strategies_triggered),
                )
                # Update classifier result with boosted confidence
                classifier_result = AdaptiveAnalysisResult(
                    analyzer_id=classifier_result.analyzer_id,
                    is_threat=classifier_result.is_threat,
                    confidence=boost_result.boosted_confidence,
                    threat_category=classifier_result.threat_category,
                    details={
                        **classifier_result.details,
                        "margin_boost": boost_result.boost_amount,
                        "margin_strategies": boost_result.strategies_triggered,
                    },
                    dca_signals=classifier_result.dca_signals,
                )

        results = [classifier_result, semantic_result, behavioral_result, multi_turn_result]

        # Collect all DCA signals
        all_signals = []
        for r in results:
            all_signals.extend(r.dca_signals)

        # Compute MCAV
        mcav = _compute_mcav(all_signals)

        # Determine blocking — two paths:
        # 1. MCAV fusion score exceeds threat threshold (multi-signal consensus)
        # 2. Any single analyzer reports high-confidence threat (≥0.90).
        #    A single strong signal (e.g., DeBERTa at 0.9999) must not be
        #    diluted below blocking threshold by SAFE signals from analyzers
        #    that simply lack data (first turn, empty vault, etc.).
        #    Threshold is 0.90 (not 0.85) to avoid false positives from
        #    borderline DeBERTa scores on business prompts containing
        #    injection-adjacent vocabulary like "override the previous estimate".
        any_high_confidence = any(
            r.is_threat and r.confidence >= 0.90 for r in results
        )
        should_block = mcav >= self._config.mcav_threat_threshold or any_high_confidence

        # Determine if this is a novel attack (adaptive caught, innate missed)
        innate_missed = True
        if innate_report is not None:
            innate_missed = not innate_report.is_threat

        adaptive_caught = any(r.is_threat for r in results)
        is_novel = adaptive_caught and innate_missed

        elapsed = (time.perf_counter() - start) * 1000

        # Antibody generation learning loop — only when adaptive actually
        # blocks.  Sub-threshold detections (DeBERTa is_threat=True but
        # below the 0.90 single-analyzer override) must NOT generate
        # antibodies, because borderline/false-positive prompts would
        # pollute the vault and cascade into L2 false positives via
        # clonal selection.
        if is_novel and should_block:
            self._generate_antibody(prompt, results)

        return AdaptiveAnalysisReport(
            request_id=context.request_id,
            analyzer_results=results,
            mcav_score=mcav,
            should_block=should_block,
            is_novel_attack=is_novel,
            total_latency_ms=elapsed,
        )

    def _generate_antibody(
        self, text: str, results: list[AdaptiveAnalysisResult]
    ) -> None:
        """Antibody generation: store novel attack in the Threat Vault.

        When adaptive catches an attack that innate missed, embed the attack
        text and store it as a new ThreatIndicator. This makes the vault
        progressively stronger — future similar attacks will be caught by
        semantic similarity search in addition to ML classification.
        """
        try:
            embedding = self._semantic.embed(text)
            # Determine the most relevant threat category from results
            category = ThreatCategory.PROMPT_INJECTION
            max_conf = 0.0
            for r in results:
                if r.is_threat and r.confidence > max_conf:
                    category = r.threat_category
                    max_conf = r.confidence

            indicator = ThreatIndicator(
                indicator_id=f"antibody-{uuid.uuid4().hex[:8]}",
                source=IndicatorSource.ADAPTIVE_DETECTION,
                confirmed=False,  # Unconfirmed until human review
                threat_category=category,
                confidence=max_conf,
                severity=max_conf * 0.8,
                embedding=embedding,
                payload_hash=hashlib.sha256(text.encode()).hexdigest(),
                payload_summary=text[:200],
                mitre_atlas_id=category.value,
            )
            self._vault.add(indicator)
            logger.info(
                "Antibody generated: %s (category=%s, confidence=%.2f)",
                indicator.indicator_id, category.value, max_conf,
            )

            # Trigger clonal selection signature generation
            if self._on_antibody:
                try:
                    self._on_antibody(text, indicator.indicator_id)
                except Exception as cb_err:
                    logger.error("Antibody callback failed: %s", cb_err)

        except Exception as e:
            logger.error("Antibody generation failed: %s", e)
