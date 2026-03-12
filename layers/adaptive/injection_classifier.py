"""
Analyzer 1 — Prompt Injection Classifier (Helper T-Cell Analog)

Fine-tuned DeBERTa-v3-base (ProtectAI/deberta-v3-base-prompt-injection-v2).
ONNX-optimized for ~20ms inference. Classifies inputs as benign or injection
at the semantic level — catches paraphrased attacks, novel phrasing, and
obfuscated injections that regex cannot detect. Periodically retrained on
newly collected attack data (Clonal Selection).

ASSUMED-BREACH POSTURE: This classifier assumes L1 and L2 have been fully
bypassed — attacks reaching this analyzer are expected to be sophisticated
enough to evade pattern matching. The model itself could be compromised:
Anthropic's Sleeper Agents research showed backdoored models that pass
standard safety training. The classifier's verdict is one input to the DCA
aggregator, never the sole decision. A poisoned classifier reporting
persistent false negatives would be detected by behavioral drift monitoring
(Analyzer 3) and the DCA's multi-signal fusion.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aegis.models.adaptive_result import (
    AdaptiveAnalysisResult,
    AnalyzerType,
    DCASignal,
    SignalType,
)
from aegis.models.scan_result import ThreatCategory

logger = logging.getLogger(__name__)


class InjectionClassifier:
    """DeBERTa-v3 prompt injection classifier with graceful degradation.

    Attempts to load the ProtectAI/deberta-v3-base-prompt-injection-v2
    model at initialization. If the model cannot be loaded (network issues,
    disk space, missing dependencies), degrades to offline mode where it
    reports classifier_offline=True in its results. Never crashes.
    """

    def __init__(self, model_name: str = "ProtectAI/deberta-v3-base-prompt-injection-v2"):
        self._model_name = model_name
        self._pipeline = None
        self._online = False
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the classification pipeline. Never raises."""
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                truncation=True,
                max_length=512,
            )
            self._online = True
            logger.info("Injection classifier loaded: %s", self._model_name)
        except Exception as e:
            logger.warning(
                "Injection classifier offline — model load failed: %s. "
                "Degrading to classifier-offline mode. This is safe: the DCA "
                "aggregator handles missing signals.", e
            )
            self._online = False

    @property
    def is_online(self) -> bool:
        return self._online

    async def analyze(self, text: str) -> AdaptiveAnalysisResult:
        """Classify text as injection or benign.

        If the model is offline, returns a degraded result with
        classifier_offline=True. The DCA aggregator treats this as
        a missing signal (neither PAMP nor safe).
        """
        start = time.perf_counter()

        if not self._online or self._pipeline is None:
            elapsed = (time.perf_counter() - start) * 1000
            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.INJECTION_CLASSIFIER,
                is_threat=False,
                confidence=0.0,
                details={"classifier_offline": True},
                dca_signals=[
                    DCASignal(
                        signal_type=SignalType.DANGER,
                        source="injection_classifier",
                        value=0.3,
                        description="Classifier offline — degraded detection",
                    )
                ],
                latency_ms=elapsed,
            )

        try:
            result = self._pipeline(text[:512])[0]
            elapsed = (time.perf_counter() - start) * 1000

            label = result["label"].upper()
            score = result["score"]

            is_injection = label == "INJECTION"
            confidence = score if is_injection else 1.0 - score

            signals = []
            if is_injection and confidence >= 0.7:
                signals.append(DCASignal(
                    signal_type=SignalType.PAMP,
                    source="injection_classifier",
                    value=confidence,
                    description=f"DeBERTa injection detection: {confidence:.2f}",
                ))
            elif is_injection and confidence >= 0.4:
                signals.append(DCASignal(
                    signal_type=SignalType.DANGER,
                    source="injection_classifier",
                    value=confidence,
                    description=f"DeBERTa moderate injection signal: {confidence:.2f}",
                ))
            else:
                signals.append(DCASignal(
                    signal_type=SignalType.SAFE,
                    source="injection_classifier",
                    value=1.0 - confidence,
                    description=f"DeBERTa benign: {1.0 - confidence:.2f}",
                ))

            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.INJECTION_CLASSIFIER,
                is_threat=is_injection and confidence >= 0.5,
                confidence=confidence,
                threat_category=ThreatCategory.PROMPT_INJECTION if is_injection else ThreatCategory.UNKNOWN,
                details={
                    "label": label,
                    "raw_score": score,
                    "classifier_offline": False,
                },
                dca_signals=signals,
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Injection classifier crashed: %s", e)
            # Fail-closed: crash → treat as danger signal
            return AdaptiveAnalysisResult(
                analyzer_id=AnalyzerType.INJECTION_CLASSIFIER,
                is_threat=True,
                confidence=0.5,
                threat_category=ThreatCategory.UNKNOWN,
                details={"classifier_crash": str(e), "classifier_offline": False},
                dca_signals=[
                    DCASignal(
                        signal_type=SignalType.DANGER,
                        source="injection_classifier",
                        value=0.5,
                        description=f"Classifier crash: {e}",
                    )
                ],
                latency_ms=elapsed,
            )
