"""
Intent detection alert and classification models.

IntentClassification is the result of classifying a feature vector.
IntentAlert wraps classification with campaign context for publishing.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from aegis.layers.correlation.intent_features import IntentFeatureVector


class IntentCategory(str, enum.Enum):
    """Operational intent categories for agent action sequences."""
    SYSTEMATIC_ENUMERATION = "systematic_enumeration"
    BOUNDARY_PROBING = "boundary_probing"
    VULNERABILITY_CONFIRMATION = "vulnerability_confirmation"
    DATA_EXFILTRATION_STAGING = "data_exfiltration_staging"
    PRIVILEGE_ESCALATION_PROBING = "privilege_escalation_probing"
    BENIGN_ACTIVITY = "benign_activity"


class IntentClassification(BaseModel):
    """Result of intent classification on a feature vector."""
    category: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    stylistic_discontinuity_amplified: bool = False
    feature_vector: IntentFeatureVector | None = None
    timestamp: float = Field(default_factory=time.time)


class IntentAlert(BaseModel):
    """Alert emitted when malicious intent is classified."""
    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    campaign_id: str | None = None
    intent: IntentClassification
    contributing_events_count: int = 0
    window_summary: dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = "investigate"
