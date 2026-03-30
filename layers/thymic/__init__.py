"""
L9 — Thymic Validation Engine (Thymic Education)

Biological analog: Thymic selection — the thymus trains T-cells by exposing
them to self-antigens. Cells that react too strongly (autoimmune) or too weakly
(incompetent) are eliminated. Only competent, tolerant T-cells mature.

AEGIS analog: The TVE continuously generates adversarial probes, routes them
through the defense pipeline (L1-L7), and measures whether defenses detect
attacks (positive selection) without flagging legitimate traffic (negative
selection / tolerance). This validates defense competence without adding
latency to the production request path.

ASSUMED-BREACH POSTURE: The TVE assumes all upstream layers may be compromised.
It independently validates that each layer is performing within specification.
A compromised layer cannot disable TVE validation — the TVE runs as an
independent background process with its own probe corpus.
"""

from aegis.layers.thymic.engine import (
    ThymicValidationEngine,
    ValidationReport,
    HealthSummary,
    LayerResult,
    TierResult,
)
from aegis.layers.thymic.probe_generator import ProbeGenerator, Probe
from aegis.layers.thymic.mutation_engine import MutationEngine
from aegis.layers.thymic.attack_profile_library import AttackProfileLibrary
from aegis.layers.thymic.layer_probe_router import LayerProbeRouter, ProbeResult

__all__ = [
    "ThymicValidationEngine",
    "ValidationReport",
    "HealthSummary",
    "LayerResult",
    "TierResult",
    "ProbeGenerator",
    "Probe",
    "MutationEngine",
    "AttackProfileLibrary",
    "LayerProbeRouter",
    "ProbeResult",
]
