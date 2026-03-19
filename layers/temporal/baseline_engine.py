"""
Behavioral Baseline Engine (BBE) — Extension 2 Infrastructure

Maintains statistical models of the protected system's normal behavior
and generates drift alerts. Tracks six metrics chosen for their
discriminative power against temporal attacks.

ASSUMED-BREACH POSTURE: Baselines are the reference against which temporal
attacks are measured. The BBE explicitly tracks calibration tier (synthetic,
canary, production) and adjusts alert confidence accordingly. Synthetic
baselines are low-confidence; production baselines are full-confidence.
A compromised BBE that reports false all-clear would blind temporal
detection — the engine's state is append-only and requires admin action
to reset.
"""

from __future__ import annotations

import enum
import logging
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import sentence-transformers for semantic drift
_EMBED_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _EMBED_AVAILABLE = True
except ImportError:
    pass


class BaselineTier(enum.Enum):
    """Calibration tier — determines alert confidence level."""

    SYNTHETIC = "synthetic"
    CANARY_AUGMENTED = "canary"
    PRODUCTION = "production"


@dataclass
class DriftAlert:
    """A single drift detection alert."""

    alert_id: str
    timestamp: float
    tenant_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_sigmas: float
    confidence: str  # "low", "medium", "high"
    tier: BaselineTier
    description: str


@dataclass
class _MetricWindow:
    """Rolling window for a single metric."""

    values: deque
    max_size: int

    def add(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.max_size:
            self.values.popleft()

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self.values) / len(self.values)
        return variance ** 0.5

    @property
    def count(self) -> int:
        return len(self.values)


@dataclass
class BaselineState:
    """Per-(tenant_id, category) baseline statistics."""

    tenant_id: str
    category: str

    # Metric 1 — Refusal rate
    refusal_count: int = 0
    total_count: int = 0

    # Metric 2 — Semantic drift
    semantic_centroid: Any = None  # numpy array or None
    semantic_distances: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Metric 3 — Output length
    length_mean: float = 0.0
    length_std: float = 0.0
    length_ema_alpha: float = 0.1
    _length_ema: float = 0.0
    _length_ema_var: float = 0.0

    # Metric 4 — Topic distribution
    topic_distribution: dict[str, float] = field(default_factory=dict)
    _topic_counts: dict[str, int] = field(default_factory=dict)

    # Metric 5 — Tool invocation
    tool_invocation_count: int = 0
    tool_type_distribution: dict[str, float] = field(default_factory=dict)
    _tool_type_counts: dict[str, int] = field(default_factory=dict)

    # Metric 6 — Response consistency (placeholder — needs clustering)
    response_consistency_score: float = 1.0

    # Metadata
    last_updated: float = 0.0
    interaction_count: int = 0
    tier: BaselineTier = BaselineTier.SYNTHETIC

    # Window trackers
    _short_refusal: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=100), 100)
    )
    _medium_refusal: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=1000), 1000)
    )
    _short_length: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=100), 100)
    )
    _medium_length: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=1000), 1000)
    )
    _short_semantic: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=100), 100)
    )
    _medium_semantic: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=1000), 1000)
    )
    _short_tool_rate: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=100), 100)
    )
    _medium_tool_rate: _MetricWindow = field(
        default_factory=lambda: _MetricWindow(deque(maxlen=1000), 1000)
    )

    # Tier tracking
    _real_interaction_count: int = 0
    _synthetic_count: int = 0
    _canary_count: int = 0

    @property
    def refusal_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.refusal_count / self.total_count

    @property
    def tool_invocation_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.tool_invocation_count / self.total_count


class BehavioralBaselineEngine:
    """Maintains behavioral baselines and generates drift alerts.

    Tracks six metrics per (tenant, category) pair:
    1. Refusal rate
    2. Semantic drift score
    3. Output length distribution
    4. Topic domain distribution
    5. Tool invocation pattern
    6. Response consistency
    """

    _REFUSAL_INDICATORS = (
        "I cannot",
        "I'm unable",
        "I don't have the ability",
        "As an AI",
        "I'm not able to",
        "I can't help with",
        "I must decline",
        "I won't be able to",
    )

    def __init__(
        self,
        *,
        warning_threshold_sigma: float = 2.0,
        critical_threshold_sigma: float = 3.0,
        production_transition_count: int = 500,
        max_baselines: int = 10000,
        short_window: int = 100,
        medium_window: int = 1000,
        embed_fn: Any | None = None,
    ) -> None:
        self._warning_sigma = warning_threshold_sigma
        self._critical_sigma = critical_threshold_sigma
        self._production_threshold = production_transition_count
        self._max_baselines = max_baselines
        self._short_window = short_window
        self._medium_window = medium_window
        self._embed_fn = embed_fn

        # LRU-ordered baselines: key = (tenant_id, category)
        self._baselines: OrderedDict[tuple[str, str], BaselineState] = OrderedDict()

        # Alert history per tenant
        self._alerts: dict[str, list[DriftAlert]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_interaction(
        self,
        tenant_id: str,
        category: str,
        query: str,
        response: str,
        was_refused: bool,
        tools_invoked: list[str] | None = None,
        is_synthetic: bool = False,
        is_canary: bool = False,
    ) -> list[DriftAlert]:
        """Record an interaction and return any drift alerts generated."""
        baseline = self._get_or_create(tenant_id, category)
        now = time.time()

        # --- Update tier ---
        if is_synthetic:
            baseline._synthetic_count += 1
        elif is_canary:
            baseline._canary_count += 1
        else:
            baseline._real_interaction_count += 1

        self._update_tier(baseline, tenant_id)

        # --- Update metrics ---
        baseline.total_count += 1
        baseline.interaction_count += 1
        baseline.last_updated = now

        # Metric 1: Refusal rate
        is_refusal = was_refused or self._detect_refusal(response)
        if is_refusal:
            baseline.refusal_count += 1
        refusal_val = 1.0 if is_refusal else 0.0
        baseline._short_refusal.add(refusal_val)
        baseline._medium_refusal.add(refusal_val)

        # Metric 3: Output length
        response_len = len(response.split())  # word count as proxy
        baseline._short_length.add(float(response_len))
        baseline._medium_length.add(float(response_len))
        # EMA update
        if baseline.interaction_count == 1:
            baseline._length_ema = float(response_len)
            baseline._length_ema_var = 0.0
        else:
            alpha = baseline.length_ema_alpha
            diff = response_len - baseline._length_ema
            baseline._length_ema += alpha * diff
            baseline._length_ema_var = (1 - alpha) * (baseline._length_ema_var + alpha * diff ** 2)
        baseline.length_mean = baseline._length_ema
        baseline.length_std = max(baseline._length_ema_var ** 0.5, 0.1)

        # Metric 2: Semantic drift
        semantic_dist = 0.0
        if self._embed_fn and response:
            try:
                emb = self._embed_fn(response)
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                if emb.ndim > 1:
                    emb = emb[0]
                if baseline.semantic_centroid is None:
                    baseline.semantic_centroid = emb.copy()
                else:
                    # Cosine distance
                    dot = float(np.dot(emb, baseline.semantic_centroid))
                    norm_a = float(np.linalg.norm(emb))
                    norm_b = float(np.linalg.norm(baseline.semantic_centroid))
                    if norm_a > 0 and norm_b > 0:
                        cos_sim = dot / (norm_a * norm_b)
                        semantic_dist = 1.0 - cos_sim
                    # Update centroid with running average
                    n = baseline.interaction_count
                    baseline.semantic_centroid = (
                        baseline.semantic_centroid * ((n - 1) / n) + emb * (1.0 / n)
                    )
                baseline.semantic_distances.append(semantic_dist)
            except Exception as e:
                logger.debug("Semantic embedding failed: %s", e)
                # Fallback: token-level Jaccard distance
                semantic_dist = self._jaccard_distance(query, response)
        else:
            # No embeddings: use Jaccard distance as fallback
            semantic_dist = self._jaccard_distance(query, response)

        baseline._short_semantic.add(semantic_dist)
        baseline._medium_semantic.add(semantic_dist)

        # Metric 4: Topic distribution
        baseline._topic_counts[category] = baseline._topic_counts.get(category, 0) + 1
        total_topics = sum(baseline._topic_counts.values())
        baseline.topic_distribution = {
            k: v / total_topics for k, v in baseline._topic_counts.items()
        }

        # Metric 5: Tool invocation
        tools = tools_invoked or []
        if tools:
            baseline.tool_invocation_count += 1
            for tool in tools:
                baseline._tool_type_counts[tool] = baseline._tool_type_counts.get(tool, 0) + 1
            total_tools = sum(baseline._tool_type_counts.values())
            baseline.tool_type_distribution = {
                k: v / total_tools for k, v in baseline._tool_type_counts.items()
            }
        tool_val = 1.0 if tools else 0.0
        baseline._short_tool_rate.add(tool_val)
        baseline._medium_tool_rate.add(tool_val)

        # --- Check for drift ---
        alerts = self._check_drift(baseline, tenant_id, now)
        if alerts:
            self._alerts.setdefault(tenant_id, []).extend(alerts)
            # Trim alert history
            if len(self._alerts[tenant_id]) > 10000:
                self._alerts[tenant_id] = self._alerts[tenant_id][-5000:]

        return alerts

    def get_baseline(
        self, tenant_id: str, category: str | None = None
    ) -> dict[str, Any]:
        """Get current baseline statistics."""
        result: dict[str, Any] = {}

        for (tid, cat), bs in self._baselines.items():
            if tid != tenant_id:
                continue
            if category and cat != category:
                continue
            result[cat] = {
                "refusal_rate": bs.refusal_rate,
                "total_count": bs.total_count,
                "length_mean": bs.length_mean,
                "length_std": bs.length_std,
                "tool_invocation_rate": bs.tool_invocation_rate,
                "topic_distribution": dict(bs.topic_distribution),
                "interaction_count": bs.interaction_count,
                "tier": bs.tier.value,
                "last_updated": bs.last_updated,
            }

        return result

    def get_tier(self, tenant_id: str) -> BaselineTier:
        """Get the current calibration tier for a tenant."""
        # Use the highest tier from any category
        best = BaselineTier.SYNTHETIC
        for (tid, _), bs in self._baselines.items():
            if tid == tenant_id:
                if bs.tier.value == "production":
                    return BaselineTier.PRODUCTION
                if bs.tier.value == "canary" and best == BaselineTier.SYNTHETIC:
                    best = BaselineTier.CANARY_AUGMENTED
        return best

    def force_tier(self, tenant_id: str, tier: BaselineTier) -> None:
        """Admin override: force a specific calibration tier."""
        for (tid, _), bs in self._baselines.items():
            if tid == tenant_id:
                bs.tier = tier

    def get_drift_history(
        self, tenant_id: str, hours: float = 24.0
    ) -> list[DriftAlert]:
        """Get recent drift alerts for a tenant."""
        cutoff = time.time() - hours * 3600
        alerts = self._alerts.get(tenant_id, [])
        return [a for a in alerts if a.timestamp >= cutoff]

    def reset_baseline(
        self, tenant_id: str, category: str | None = None
    ) -> None:
        """Reset baseline to empty state."""
        keys_to_remove = [
            k for k in self._baselines
            if k[0] == tenant_id and (category is None or k[1] == category)
        ]
        for k in keys_to_remove:
            del self._baselines[k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, tenant_id: str, category: str) -> BaselineState:
        """Get or create a baseline, with LRU eviction."""
        key = (tenant_id, category)
        if key in self._baselines:
            self._baselines.move_to_end(key)
            return self._baselines[key]

        # Evict oldest if at capacity
        while len(self._baselines) >= self._max_baselines:
            self._baselines.popitem(last=False)

        bs = BaselineState(
            tenant_id=tenant_id,
            category=category,
            _short_refusal=_MetricWindow(deque(maxlen=self._short_window), self._short_window),
            _medium_refusal=_MetricWindow(deque(maxlen=self._medium_window), self._medium_window),
            _short_length=_MetricWindow(deque(maxlen=self._short_window), self._short_window),
            _medium_length=_MetricWindow(deque(maxlen=self._medium_window), self._medium_window),
            _short_semantic=_MetricWindow(deque(maxlen=self._short_window), self._short_window),
            _medium_semantic=_MetricWindow(deque(maxlen=self._medium_window), self._medium_window),
            _short_tool_rate=_MetricWindow(deque(maxlen=self._short_window), self._short_window),
            _medium_tool_rate=_MetricWindow(deque(maxlen=self._medium_window), self._medium_window),
        )
        self._baselines[key] = bs
        return bs

    def _update_tier(self, baseline: BaselineState, tenant_id: str) -> None:
        """Auto-transition calibration tier based on interaction counts."""
        if baseline.tier == BaselineTier.SYNTHETIC:
            if baseline._real_interaction_count >= 1:
                baseline.tier = BaselineTier.CANARY_AUGMENTED
                logger.info(
                    "Tenant %s category %s: tier → CANARY_AUGMENTED",
                    tenant_id, baseline.category,
                )
        elif baseline.tier == BaselineTier.CANARY_AUGMENTED:
            if baseline._real_interaction_count >= self._production_threshold:
                baseline.tier = BaselineTier.PRODUCTION
                logger.info(
                    "Tenant %s category %s: tier → PRODUCTION (real=%d)",
                    tenant_id, baseline.category, baseline._real_interaction_count,
                )

    def _detect_refusal(self, response: str) -> bool:
        """Check if a response contains refusal indicators."""
        lower = response.lower()
        return any(ind.lower() in lower for ind in self._REFUSAL_INDICATORS)

    @staticmethod
    def _jaccard_distance(text_a: str, text_b: str) -> float:
        """Token-level Jaccard distance as fallback for semantic drift."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a and not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union == 0:
            return 0.0
        return 1.0 - intersection / union

    def _check_drift(
        self, baseline: BaselineState, tenant_id: str, now: float
    ) -> list[DriftAlert]:
        """Check all metrics for drift, generate alerts."""
        # Need minimum observations before checking
        if baseline.interaction_count < 10:
            return []

        alerts: list[DriftAlert] = []

        # Check each metric using short-term window vs medium-term baseline
        metrics_to_check = [
            ("refusal_rate", baseline._short_refusal, baseline._medium_refusal),
            ("output_length", baseline._short_length, baseline._medium_length),
            ("semantic_drift", baseline._short_semantic, baseline._medium_semantic),
            ("tool_invocation_rate", baseline._short_tool_rate, baseline._medium_tool_rate),
        ]

        for metric_name, short_win, medium_win in metrics_to_check:
            if short_win.count < 5 or medium_win.count < 10:
                continue

            short_mean = short_win.mean
            medium_mean = medium_win.mean
            medium_std = medium_win.std

            if medium_std < 1e-9:
                # No variance — skip (or flag if sudden change)
                if abs(short_mean - medium_mean) > 0.01:
                    deviation = 999.0  # Infinite sigma
                else:
                    continue
            else:
                deviation = abs(short_mean - medium_mean) / medium_std

            if deviation >= self._warning_sigma:
                severity = "critical" if deviation >= self._critical_sigma else "warning"
                confidence = self._tier_confidence(baseline.tier, severity)

                alerts.append(DriftAlert(
                    alert_id=uuid.uuid4().hex[:12],
                    timestamp=now,
                    tenant_id=tenant_id,
                    metric_name=metric_name,
                    current_value=short_mean,
                    baseline_value=medium_mean,
                    deviation_sigmas=deviation,
                    confidence=confidence,
                    tier=baseline.tier,
                    description=(
                        f"{metric_name} shifted {deviation:.1f}σ: "
                        f"{medium_mean:.4f} → {short_mean:.4f} ({severity})"
                    ),
                ))

        # Compound alert: 2+ metrics exceed warning → elevate all to critical
        if len(alerts) >= 2:
            for alert in alerts:
                if alert.confidence != "high":
                    alert.confidence = self._tier_confidence(baseline.tier, "critical")
                alert.description = f"[COMPOUND] {alert.description}"

        return alerts

    @staticmethod
    def _tier_confidence(tier: BaselineTier, severity: str) -> str:
        """Determine alert confidence based on tier and severity."""
        if tier == BaselineTier.SYNTHETIC:
            return "low"
        elif tier == BaselineTier.CANARY_AUGMENTED:
            return "medium" if severity == "critical" else "low"
        else:  # PRODUCTION
            return "high" if severity == "critical" else "medium"
