"""
Reasoning Length Anomaly Detector — Extension 1.3

Flags interactions where the reasoning trace exceeds baseline by a
configurable multiplier. Chain-of-thought hijacking requires significantly
longer reasoning sequences than normal interactions to achieve safety
signal dilution.

Maintains per-(tenant, session) rolling averages via exponential moving
average. LRU eviction prevents unbounded memory growth.

ASSUMED-BREACH POSTURE: An attacker must pad reasoning to dilute safety
signals. This detector catches the padding before more expensive analysis
runs. The baseline is maintained from clean (non-anomalous) observations
only, preventing an attacker from inflating the baseline via sustained
long-trace submissions.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LengthAnomalyReport:
    trace_length_tokens: int = 0
    baseline_tokens: float = 0.0
    multiplier: float = 3.0
    ratio: float = 0.0
    is_anomalous: bool = False
    confidence: float = 0.0
    tenant_id: str = ""
    session_id: str = ""


class ReasoningLengthAnomalyDetector:
    """Detects anomalously long reasoning traces via EMA baselines.

    Thread-safe. Maintains per-(tenant, session) baselines with LRU eviction.
    Anomalous traces do NOT update the baseline to prevent inflation attacks.
    """

    def __init__(
        self,
        multiplier: float = 3.0,
        default_baseline: int = 500,
        max_baselines: int = 10000,
        ema_alpha: float = 0.1,
    ):
        self._multiplier = multiplier
        self._default_baseline = float(default_baseline)
        self._max_baselines = max_baselines
        self._ema_alpha = ema_alpha

        self._lock = threading.Lock()
        self._baselines: OrderedDict[tuple[str, str], float] = OrderedDict()

    def analyze(
        self,
        trace_length_tokens: int,
        tenant_id: str = "",
        session_id: str = "",
    ) -> LengthAnomalyReport:
        """Check if a reasoning trace length is anomalous.

        Args:
            trace_length_tokens: Number of tokens in the reasoning trace.
            tenant_id: Tenant identifier for per-tenant baselines.
            session_id: Session identifier for per-session baselines.

        Returns:
            LengthAnomalyReport with anomaly detection results.
        """
        key = (tenant_id, session_id)
        baseline = self._get_baseline(key)
        ratio = trace_length_tokens / baseline if baseline > 0 else 0.0
        is_anomalous = ratio >= self._multiplier

        # Confidence scales with excess beyond multiplier
        if is_anomalous:
            confidence = min(1.0, (ratio - self._multiplier) / self._multiplier)
        else:
            confidence = 0.0

        # Only update baseline with non-anomalous observations
        if not is_anomalous:
            self._update_baseline(key, float(trace_length_tokens))

        return LengthAnomalyReport(
            trace_length_tokens=trace_length_tokens,
            baseline_tokens=baseline,
            multiplier=self._multiplier,
            ratio=ratio,
            is_anomalous=is_anomalous,
            confidence=confidence,
            tenant_id=tenant_id,
            session_id=session_id,
        )

    def _get_baseline(self, key: tuple[str, str]) -> float:
        """Get the current baseline for a (tenant, session) pair."""
        with self._lock:
            if key in self._baselines:
                self._baselines.move_to_end(key)
                return self._baselines[key]
        return self._default_baseline

    def _update_baseline(self, key: tuple[str, str], value: float) -> None:
        """Update the EMA baseline for a (tenant, session) pair."""
        with self._lock:
            if key in self._baselines:
                old = self._baselines[key]
                self._baselines[key] = old * (1 - self._ema_alpha) + value * self._ema_alpha
                self._baselines.move_to_end(key)
            else:
                if len(self._baselines) >= self._max_baselines:
                    self._baselines.popitem(last=False)
                self._baselines[key] = value

    def get_baseline(self, tenant_id: str = "", session_id: str = "") -> float:
        """Get current baseline for external inspection."""
        return self._get_baseline((tenant_id, session_id))
