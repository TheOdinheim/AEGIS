"""
Temporal Threat Detection — Extension 2 Infrastructure

Provides the Synthetic Traffic Generator, Behavioral Baseline Engine,
Canary Injection System, Memory Provenance Registry, and Temporal
Correlation Engine — the foundation for all temporal detection components.

ASSUMED-BREACH POSTURE: Temporal attacks exploit the gap between injection
and activation. Point-in-time monitors cannot detect them. Detection
requires asking "has something changed?" which requires knowing what
"normal" looks like. These modules bootstrap and maintain that definition.
"""

from aegis.layers.temporal.traffic_generator import (
    SyntheticTrafficGenerator,
    SyntheticQuery,
    DomainProfile,
)
from aegis.layers.temporal.baseline_engine import (
    BehavioralBaselineEngine,
    BaselineTier,
    BaselineState,
    DriftAlert,
)
from aegis.layers.temporal.canary_system import (
    CanaryInjectionSystem,
    CanaryQuery,
    CanaryResult,
    CanaryAlert,
    CanarySystemStatus,
)
from aegis.layers.temporal.provenance_registry import (
    MemoryProvenanceRegistry,
    ProvenanceCategory,
    ProvenanceRecord,
    ProvenanceTimeline,
)
from aegis.layers.temporal.correlation_engine import (
    TemporalCorrelationEngine,
    CorrelationCandidate,
    CorrelationReport,
)

__all__ = [
    "SyntheticTrafficGenerator",
    "SyntheticQuery",
    "DomainProfile",
    "BehavioralBaselineEngine",
    "BaselineTier",
    "BaselineState",
    "DriftAlert",
    "CanaryInjectionSystem",
    "CanaryQuery",
    "CanaryResult",
    "CanaryAlert",
    "CanarySystemStatus",
    "MemoryProvenanceRegistry",
    "ProvenanceCategory",
    "ProvenanceRecord",
    "ProvenanceTimeline",
    "TemporalCorrelationEngine",
    "CorrelationCandidate",
    "CorrelationReport",
]
