"""
AEGIS Data Contracts — Pydantic models for every inter-layer boundary.

ASSUMED-BREACH POSTURE: These models are the ONLY trusted interface between
layers. Each model validates its own fields on construction. No layer may
pass raw dicts or unvalidated data across boundaries. If a model fails
validation, the request is rejected — a malformed internal message is
treated as evidence of compromise, not a bug.
"""

from aegis.models.request_context import RequestContext
from aegis.models.scan_result import ScanResult, InnateScanReport
from aegis.models.adaptive_result import AdaptiveAnalysisResult, AdaptiveAnalysisReport
from aegis.models.policy_decision import PolicyDecision
from aegis.models.threat_indicator import ThreatIndicator

__all__ = [
    "RequestContext",
    "ScanResult",
    "InnateScanReport",
    "AdaptiveAnalysisResult",
    "AdaptiveAnalysisReport",
    "PolicyDecision",
    "ThreatIndicator",
]
