"""Data models for the model distillation defense module.

Detects systematic attempts to extract model knowledge through
query diversity analysis, boundary mapping, reasoning coercion,
information harvesting, and complexity escalation patterns.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class DistillationStrategy(enum.Enum):
    """Strategies used by adversaries to distill model knowledge."""

    QUERY_DIVERSITY = "query_diversity"
    BOUNDARY_MAPPING = "boundary_mapping"
    REASONING_COERCION = "reasoning_coercion"
    INFO_HARVESTING = "info_harvesting"
    COMPLEXITY_ESCALATION = "complexity_escalation"


@dataclass
class InteractionRecord:
    """A single query-response interaction within a monitoring window.

    Records metadata about each interaction for pattern analysis
    without storing full prompt or response content.
    """

    timestamp: float = field(default_factory=time.time)
    topic_hash: str = ""
    query_text_summary: str = ""
    response_length: int = 0
    was_blocked: bool = False
    block_reason: str = ""
    complexity_score: float = 0.0
    is_reasoning_query: bool = False


@dataclass
class DistillationSignal:
    """Detection signal for a specific distillation strategy.

    Each signal carries strategy-specific evidence and a confidence
    score indicating the likelihood of active distillation.
    """

    strategy: DistillationStrategy
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)
    triggered: bool = False


@dataclass
class DistillationReport:
    """Aggregated distillation detection report for a monitoring window.

    Combines signals from all detection strategies into a unified
    threat assessment with recommended actions.
    """

    signals: list[DistillationSignal] = field(default_factory=list)
    combined_threat_score: float = 0.0
    should_alert: bool = False
    should_block: bool = False
    recommended_action: str = "none"
    api_key: str = ""
    window_start: float = 0.0
    window_end: float = 0.0
    total_queries: int = 0
    blocked_queries: int = 0


@dataclass
class ReasoningScanResult:
    """Result of scanning a response for reasoning trace leakage.

    Detects chain-of-thought traces, governance disclosures, and
    decision-making process revelations that could aid distillation.
    """

    has_reasoning_trace: bool = False
    has_governance_disclosure: bool = False
    has_decision_disclosure: bool = False
    trace_count: int = 0
    redacted_text: str | None = None
    detections: list[dict] = field(default_factory=list)
