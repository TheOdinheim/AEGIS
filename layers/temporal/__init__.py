"""
Temporal Threat Detection — Extension 2 Infrastructure

Provides the Synthetic Traffic Generator, Behavioral Baseline Engine,
and Canary Injection System — the foundation for all temporal detection
components (correlation engine, date-triggered scanner, clean-state replay).

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
]
