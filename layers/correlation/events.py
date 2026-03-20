"""
Event schema for campaign correlation.

AgentActionEvent captures a single agent action. CampaignAlert is emitted
when coordinated multi-agent activity is detected. Both are Pydantic models
for validation and serialization.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, enum.Enum):
    """Types of agent actions tracked for campaign correlation."""
    API_CALL = "api_call"
    DATA_ACCESS = "data_access"
    TOOL_INVOCATION = "tool_invocation"
    SKILL_EXECUTION = "skill_execution"
    OUTPUT_GENERATION = "output_generation"
    PROBE = "probe"
    AUTH_ATTEMPT = "auth_attempt"


class FingerprintType(str, enum.Enum):
    """Campaign fingerprint signature types."""
    TEMPORAL_CLUSTER = "temporal_cluster"
    SYSTEMATIC_ENUMERATION = "systematic_enumeration"
    PARAMETER_FUZZING = "parameter_fuzzing"
    RECON_TO_EXPLOIT = "recon_to_exploit"
    INFORMATION_FLOW = "information_flow"


class AlertLevel(str, enum.Enum):
    """Alert severity mapped to TLI levels."""
    LOW = "low"          # TLI GREEN/BLUE
    MEDIUM = "medium"    # TLI YELLOW
    HIGH = "high"        # TLI ORANGE
    CRITICAL = "critical"  # TLI RED


class AgentActionEvent(BaseModel):
    """A single agent action event for campaign correlation."""
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_id: str
    timestamp: float = Field(default_factory=time.time)
    action_type: ActionType
    target_resource: str
    input_hash: str = ""
    output_hash: str = ""
    session_id: str = ""
    tenant_id: str = "default"
    source_model: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Parameter value for fuzzing detection
    parameter_value: str = ""


class CampaignAlert(BaseModel):
    """Alert emitted when a coordinated campaign is detected."""
    campaign_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    detection_timestamp: float = Field(default_factory=time.time)
    confidence: float = Field(ge=0.0, le=1.0)
    alert_level: AlertLevel
    contributing_agent_ids: list[str]
    fingerprint_type: FingerprintType
    campaign_graph_snapshot: dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = "investigate"
    evidence: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = "default"
