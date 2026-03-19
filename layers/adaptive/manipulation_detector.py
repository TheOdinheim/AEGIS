"""
Multi-Turn Manipulation Detector (MTMD) — Extension 5.1

Detects autonomous jailbreak agents that use multi-turn manipulation
strategies: progressive boundary testing, tactic switching, persona
adoption, escalation gradients, and response adaptation.

Research context: Reasoning models can autonomously jailbreak other AI
models at 97% success rate (Nature Communications, 2026). These agents
exhibit specific behavioral signatures that single-turn detectors miss.

ASSUMED-BREACH POSTURE: This detector assumes L2 innate and L3 adaptive
single-turn classifiers have been bypassed. Autonomous jailbreak agents
deliberately stay below per-turn thresholds while systematically probing
across turns. A compromised session store could suppress history — the
detector operates on its own maintained turn records, independent of the
session store used by multi_turn.py.
"""

from __future__ import annotations

import enum
import logging
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ManipulationSignalType(str, enum.Enum):
    BOUNDARY_TESTING = "boundary_testing"
    TACTIC_SWITCHING = "tactic_switching"
    PERSONA_ADOPTION = "persona_adoption"
    ESCALATION_GRADIENT = "escalation_gradient"
    RESPONSE_ADAPTATION = "response_adaptation"


@dataclass
class ManipulationSignal:
    signal_type: ManipulationSignalType
    confidence: float
    turn_index: int
    description: str


@dataclass
class ManipulationReport:
    source_id: str
    session_id: str
    signals: list[ManipulationSignal]
    manipulation_score: float
    should_alert: bool
    should_block: bool
    turn_count: int
    analysis_window: int


@dataclass
class TurnRecord:
    turn_index: int
    timestamp: float
    content: str
    embedding: np.ndarray | None = None
    injection_score: float = 0.0
    was_blocked: bool = False
    response_was_refusal: bool = False
    detection_categories: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal weights for score fusion
# ---------------------------------------------------------------------------

_SIGNAL_WEIGHTS: dict[ManipulationSignalType, float] = {
    ManipulationSignalType.BOUNDARY_TESTING: 1.0,
    ManipulationSignalType.TACTIC_SWITCHING: 1.2,
    ManipulationSignalType.PERSONA_ADOPTION: 0.8,
    ManipulationSignalType.ESCALATION_GRADIENT: 1.0,
    ManipulationSignalType.RESPONSE_ADAPTATION: 1.5,
}


class MultiTurnManipulationDetector:
    """Analyzes sequences of interactions from a single source for patterns
    characteristic of autonomous jailbreaking.

    Maintains a sliding window of the last N turns per source (default 20).
    For each new turn, runs five detection strategies and fuses their signals
    into a single manipulation_score.
    """

    def __init__(
        self,
        window_size: int = 20,
        max_history: int = 100,
        block_threshold: float = 0.7,
        alert_threshold: float = 0.4,
        boundary_slope_threshold: float = 0.03,
        embed_fn: Any | None = None,
    ):
        self._window_size = window_size
        self._max_history = max_history
        self._block_threshold = block_threshold
        self._alert_threshold = alert_threshold
        self._boundary_slope_threshold = boundary_slope_threshold
        self._embed_fn = embed_fn
        self._lock = threading.Lock()

        # source_id -> list[TurnRecord], with LRU eviction
        self._sources: OrderedDict[str, list[TurnRecord]] = OrderedDict()

    @property
    def source_count(self) -> int:
        with self._lock:
            return len(self._sources)

    def record_turn(
        self,
        source_id: str,
        content: str,
        injection_score: float = 0.0,
        was_blocked: bool = False,
        response_was_refusal: bool = False,
        detection_categories: list[str] | None = None,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Record a new turn for a source. Called after L2/L3 scoring."""
        now = time.time()
        with self._lock:
            if source_id not in self._sources:
                # LRU eviction
                if len(self._sources) >= self._max_history:
                    self._sources.popitem(last=False)
                self._sources[source_id] = []
            else:
                # Move to end (most recent)
                self._sources.move_to_end(source_id)

            turns = self._sources[source_id]
            turn_index = len(turns)

            # Compute embedding if function available and not provided
            emb = embedding
            if emb is None and self._embed_fn is not None:
                try:
                    emb = np.array(self._embed_fn(content), dtype=np.float32)
                except Exception:
                    emb = None

            turns.append(TurnRecord(
                turn_index=turn_index,
                timestamp=now,
                content=content[:500],
                embedding=emb,
                injection_score=injection_score,
                was_blocked=was_blocked,
                response_was_refusal=response_was_refusal,
                detection_categories=detection_categories or [],
            ))

            # Keep only max_history turns per source
            if len(turns) > self._max_history:
                self._sources[source_id] = turns[-self._max_history:]

    def analyze(
        self,
        source_id: str,
        session_id: str = "",
    ) -> ManipulationReport:
        """Run all five detection strategies on the source's turn history.

        Returns a ManipulationReport with fused score and signals.
        Must complete in <5ms (operates on in-memory data, no I/O).
        """
        with self._lock:
            turns = list(self._sources.get(source_id, []))

        if not turns:
            return ManipulationReport(
                source_id=source_id,
                session_id=session_id,
                signals=[],
                manipulation_score=0.0,
                should_alert=False,
                should_block=False,
                turn_count=0,
                analysis_window=self._window_size,
            )

        # Use the analysis window
        window = turns[-self._window_size:]
        signals: list[ManipulationSignal] = []

        # Strategy 1: Boundary Testing
        sig = self._detect_boundary_testing(window)
        if sig:
            signals.append(sig)

        # Strategy 2: Tactic Switching
        sig = self._detect_tactic_switching(window)
        if sig:
            signals.append(sig)

        # Strategy 3: Persona Adoption (requires embeddings)
        sig = self._detect_persona_adoption(window)
        if sig:
            signals.append(sig)

        # Strategy 4: Escalation Gradient
        sig = self._detect_escalation_gradient(window)
        if sig:
            signals.append(sig)

        # Strategy 5: Response Adaptation
        sigs = self._detect_response_adaptation(window)
        signals.extend(sigs)

        # Score fusion
        score = self._fuse_scores(signals)

        return ManipulationReport(
            source_id=source_id,
            session_id=session_id,
            signals=signals,
            manipulation_score=score,
            should_alert=score > self._alert_threshold,
            should_block=score > self._block_threshold,
            turn_count=len(turns),
            analysis_window=self._window_size,
        )

    # ------------------------------------------------------------------
    # Strategy 1: Boundary Testing Detection
    # ------------------------------------------------------------------

    def _detect_boundary_testing(
        self, window: list[TurnRecord],
    ) -> ManipulationSignal | None:
        """Linear regression on injection scores — positive slope = probing."""
        scores = [t.injection_score for t in window]
        if len(scores) < 5:
            return None

        # Use the last 10 (or fewer) scores for regression
        recent = scores[-10:]
        n = len(recent)
        x = np.arange(n, dtype=np.float64)
        y = np.array(recent, dtype=np.float64)

        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        var_x = ((x - x_mean) ** 2).sum()
        if var_x == 0:
            return None
        slope = ((x - x_mean) * (y - y_mean)).sum() / var_x

        if slope <= self._boundary_slope_threshold:
            return None

        confidence = min(1.0, slope / 0.1)
        return ManipulationSignal(
            signal_type=ManipulationSignalType.BOUNDARY_TESTING,
            confidence=confidence,
            turn_index=window[-1].turn_index,
            description=f"Monotonically increasing injection scores: slope={slope:.4f}",
        )

    # ------------------------------------------------------------------
    # Strategy 2: Tactic Switching Detection
    # ------------------------------------------------------------------

    def _detect_tactic_switching(
        self, window: list[TurnRecord],
    ) -> ManipulationSignal | None:
        """Count distinct attack category prefixes within window."""
        categories: set[str] = set()
        for t in window:
            for cat in t.detection_categories:
                # Extract prefix (e.g., PI-001 -> PI, RC-002 -> RC)
                prefix = cat.split("-")[0] if "-" in cat else cat
                if prefix:
                    categories.add(prefix)

        n_cats = len(categories)
        if n_cats < 3:
            return None

        if n_cats >= 5:
            confidence = 0.9
        elif n_cats == 4:
            confidence = 0.75
        else:
            confidence = 0.6

        return ManipulationSignal(
            signal_type=ManipulationSignalType.TACTIC_SWITCHING,
            confidence=confidence,
            turn_index=window[-1].turn_index,
            description=f"Cycled through {n_cats} attack categories: {sorted(categories)}",
        )

    # ------------------------------------------------------------------
    # Strategy 3: Persona Adoption Detection
    # ------------------------------------------------------------------

    def _detect_persona_adoption(
        self, window: list[TurnRecord],
    ) -> ManipulationSignal | None:
        """Detect abrupt persona shifts via embedding similarity drops."""
        # Requires embeddings
        embedded = [t for t in window if t.embedding is not None]
        if len(embedded) < 3:
            return None

        low_sim_count = 0
        elevated_injection = 0

        for i in range(1, len(embedded)):
            prev_emb = embedded[i - 1].embedding
            curr_emb = embedded[i].embedding
            assert prev_emb is not None and curr_emb is not None

            # Cosine similarity
            norm_prev = np.linalg.norm(prev_emb)
            norm_curr = np.linalg.norm(curr_emb)
            if norm_prev == 0 or norm_curr == 0:
                continue
            sim = float(np.dot(prev_emb, curr_emb) / (norm_prev * norm_curr))

            if sim < 0.3:
                low_sim_count += 1
            if embedded[i].injection_score > 0.3:
                elevated_injection += 1

        if low_sim_count >= 2 and elevated_injection >= 2:
            confidence = min(0.5 + low_sim_count * 0.15, 0.95)
            return ManipulationSignal(
                signal_type=ManipulationSignalType.PERSONA_ADOPTION,
                confidence=confidence,
                turn_index=window[-1].turn_index,
                description=(
                    f"Abrupt persona shifts: {low_sim_count} low-similarity transitions "
                    f"with {elevated_injection} elevated injection scores"
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Strategy 4: Escalation Gradient Detection
    # ------------------------------------------------------------------

    def _detect_escalation_gradient(
        self, window: list[TurnRecord],
    ) -> ManipulationSignal | None:
        """Detect slow-and-low escalation via rolling average climb."""
        scores = [t.injection_score for t in window]
        if len(scores) < 5:
            return None

        # 5-turn rolling average
        rolling = []
        for i in range(len(scores) - 4):
            avg = sum(scores[i:i + 5]) / 5.0
            rolling.append(avg)

        if len(rolling) < 2:
            return None

        first_avg = rolling[0]
        last_avg = rolling[-1]

        if first_avg < 0.2 and last_avg > 0.5:
            climb = last_avg - first_avg
            confidence = min(climb / 0.5, 0.95)
            return ManipulationSignal(
                signal_type=ManipulationSignalType.ESCALATION_GRADIENT,
                confidence=confidence,
                turn_index=window[-1].turn_index,
                description=(
                    f"Rolling average climbed from {first_avg:.2f} to {last_avg:.2f}"
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Strategy 5: Response Adaptation Detection
    # ------------------------------------------------------------------

    def _detect_response_adaptation(
        self, window: list[TurnRecord],
    ) -> list[ManipulationSignal]:
        """Detect technique switching after blocks."""
        signals: list[ManipulationSignal] = []
        adaptation_count = 0

        for i in range(len(window) - 1):
            if not window[i].was_blocked:
                continue

            # Check if next turn uses a different technique category
            blocked_cats = set(
                c.split("-")[0] for c in window[i].detection_categories if "-" in c
            )
            next_cats = set(
                c.split("-")[0] for c in window[i + 1].detection_categories if "-" in c
            )

            if not blocked_cats or not next_cats:
                continue

            if blocked_cats != next_cats:
                adaptation_count += 1
                confidence = min(0.7 + (adaptation_count - 1) * 0.1, 0.95)
                signals.append(ManipulationSignal(
                    signal_type=ManipulationSignalType.RESPONSE_ADAPTATION,
                    confidence=confidence,
                    turn_index=window[i + 1].turn_index,
                    description=(
                        f"Switched from {sorted(blocked_cats)} to {sorted(next_cats)} "
                        f"after block (adaptation #{adaptation_count})"
                    ),
                ))

        return signals

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    def _fuse_scores(self, signals: list[ManipulationSignal]) -> float:
        """Weighted average of all signal confidences."""
        if not signals:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for sig in signals:
            w = _SIGNAL_WEIGHTS.get(sig.signal_type, 1.0)
            weighted_sum += sig.confidence * w
            weight_total += w

        if weight_total == 0:
            return 0.0

        return min(weighted_sum / weight_total, 1.0)

    def get_turn_history(self, source_id: str) -> list[TurnRecord]:
        """Get turn history for a source (for testing)."""
        with self._lock:
            return list(self._sources.get(source_id, []))

    def clear_source(self, source_id: str) -> None:
        """Remove all turn records for a source."""
        with self._lock:
            self._sources.pop(source_id, None)
