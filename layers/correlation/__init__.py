"""
Campaign Correlation Engine — Extension 6 (XBOW Phase A1)

Detects coordinated multi-agent campaigns from distributed fragments.
Autonomous adversarial platforms decompose attacks into thousands of
narrowly-scoped parallel solver agents. No individual solver exceeds
anomaly thresholds. Detection requires correlating all solver activities
into a unified campaign view.

This is the immune system's epidemiological surveillance: detecting an
outbreak from early case reports before it becomes a pandemic.

ASSUMED-BREACH POSTURE: Individual detection layers (L2–L5) may miss
coordinated campaigns because each fragment appears benign in isolation.
The correlation engine operates on aggregate population statistics across
agents, not individual event analysis. A compromised correlation engine
that suppresses campaign alerts could blind AEGIS to coordinated attacks —
all alerts are persisted to the audit log independently.
"""

from aegis.layers.correlation.events import (
    ActionType,
    AgentActionEvent,
    AlertLevel,
    CampaignAlert,
    FingerprintType,
)
from aegis.layers.correlation.fingerprint_detector import (
    FingerprintDetector,
    FingerprintMatch,
)
from aegis.layers.correlation.campaign_graph import CampaignGraph
from aegis.layers.correlation.engine import CampaignCorrelationEngine
from aegis.layers.correlation.intent_features import (
    IntentFeatureExtractor,
    IntentFeatureVector,
)
from aegis.layers.correlation.intent_classifier import IntentClassifier
from aegis.layers.correlation.intent_alert import (
    IntentCategory,
    IntentClassification,
    IntentAlert,
)

__all__ = [
    "ActionType",
    "AgentActionEvent",
    "AlertLevel",
    "CampaignAlert",
    "FingerprintType",
    "FingerprintDetector",
    "FingerprintMatch",
    "CampaignGraph",
    "CampaignCorrelationEngine",
    "IntentFeatureExtractor",
    "IntentFeatureVector",
    "IntentClassifier",
    "IntentCategory",
    "IntentClassification",
    "IntentAlert",
]
