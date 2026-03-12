"""Adversarial ML attack engine for AEGIS red team.

Provides black-box query-based attacks, evolutionary mutation breeding,
timing side-channel analysis, and decision boundary mapping.  All algorithms
accept a ``QueryFn`` callable that abstracts transport — httpx for live,
TestClient for in-process, mock for CI tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Transport abstraction
# ---------------------------------------------------------------------------
# (status_code, response_text, latency_ms)
QueryFn = Callable[[str], tuple[int, str, float]]

# ---------------------------------------------------------------------------
# Black-box attack result
# ---------------------------------------------------------------------------


@dataclass
class AdversarialResult:
    """Result of a single black-box adversarial attack."""

    algorithm: str
    original_prompt: str
    adversarial_prompt: str
    success: bool  # True if evasion achieved (blocked -> allowed)
    edits_made: list[str]
    iterations: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Evolutionary engine result
# ---------------------------------------------------------------------------


@dataclass
class EvolutionResult:
    """Result of an evolutionary mutation campaign."""

    best_individual: str
    best_fitness: float
    generations_run: int
    total_evaluations: int
    evasion_found: bool
    fitness_history: list[float]  # best fitness per generation
    diversity_history: list[float]  # avg Jaccard distance per generation
    mutation_operator_stats: dict[str, int]  # operator -> times applied


# ---------------------------------------------------------------------------
# Timing oracle result
# ---------------------------------------------------------------------------


@dataclass
class TimingAnalysis:
    """Result of timing side-channel analysis."""

    benign_mean: float
    benign_std: float
    blocked_mean: float
    blocked_std: float
    borderline_mean: float
    borderline_std: float
    t_statistic: float
    p_value: float
    timing_leak_detected: bool


# ---------------------------------------------------------------------------
# Boundary mapper result
# ---------------------------------------------------------------------------


@dataclass
class BoundaryMap:
    """Decision boundary mapping results."""

    l2_regex_boundaries: list[dict]
    l3_deberta_boundaries: list[dict]
    l5_pii_boundaries: list[dict]
    total_probes: int


# ---------------------------------------------------------------------------
# Real-world attack corpus entry
# ---------------------------------------------------------------------------


@dataclass
class RealWorldAttack:
    """A real-world attack from the adversarial corpus."""

    attack_id: str
    category: str
    prompt: str
    source: str
    expected_layer: str
    severity: int  # 1-5

    def to_dict(self) -> dict:
        return {
            "attack_id": self.attack_id,
            "category": self.category,
            "prompt": self.prompt,
            "source": self.source,
            "expected_layer": self.expected_layer,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RealWorldAttack:
        return cls(
            attack_id=data["attack_id"],
            category=data["category"],
            prompt=data["prompt"],
            source=data["source"],
            expected_layer=data["expected_layer"],
            severity=data["severity"],
        )


# ---------------------------------------------------------------------------
# Full assessment result
# ---------------------------------------------------------------------------


@dataclass
class AdversarialAssessment:
    """Complete adversarial ML assessment report."""

    corpus_baseline: dict  # {total, blocked, evaded, evasion_rate}
    textfooler_results: list[AdversarialResult]
    charswap_results: list[AdversarialResult]
    paraphrase_results: list[AdversarialResult]
    evolution_result: EvolutionResult | None
    timing_analysis: TimingAnalysis | None
    boundary_map: BoundaryMap | None
    overall_evasion_rate: float
    recommendations: list[str] = field(default_factory=list)
