"""
PolicyDecision — L6 Policy Engine output contract.

The Policy Engine (Regulatory T-Cells) merges innate and adaptive threat
scores with tenant policy, global rules, and the current Threat Level
Indicator to render a final allow/block/escalate decision.

ASSUMED-BREACH POSTURE: The PolicyDecision is the final arbiter, but it is
not infallible. A compromised policy engine could allow all requests through
or block all legitimate traffic (autoimmune). Downstream components (L5
Output Validation) still independently validate model responses even when
the policy engine allowed the request. The decision includes full reasoning
for audit trail compliance (EU AI Act Art. 12/19, SOC 2 CC7.2).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from aegis.config import ThreatLevel


class PolicyAction(str, Enum):
    """Final action rendered by the Policy Engine."""
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"        # Route to human review queue
    ALLOW_DEGRADED = "allow_degraded"  # Allow with tighter output scrutiny
    QUARANTINE = "quarantine"    # Route to honeypot for intel collection


class PolicyTier(str, Enum):
    """Which policy tier triggered the decision."""
    GLOBAL = "global"
    TENANT = "tenant"
    ADAPTIVE = "adaptive"


class PolicyDecision(BaseModel):
    """Final decision from the L6 Policy Engine.

    This is the single object that determines whether a request reaches
    the upstream model. It includes full reasoning for compliance audit.

    Fields:
        request_id: Links to the RequestContext.
        timestamp: When the decision was rendered.
        action: The final action (allow, block, escalate, etc.).
        triggered_by: Which policy tier produced this decision.
        threat_level: Current TLI at time of decision.
        innate_max_confidence: Highest confidence from L2 innate.
        adaptive_mcav: MCAV score from L3 adaptive.
        fused_score: Combined threat score after policy fusion logic.
        reasons: Human-readable list of reasons for the decision.
        applied_policies: List of policy rule IDs that contributed.
        output_scrutiny_level: Elevated scrutiny level for L5 output validation.
        block_message: Message returned to client if blocked.
    """
    request_id: str = Field(description="Request ID from RequestContext")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the decision was rendered",
    )
    action: PolicyAction = Field(description="Final action: allow, block, escalate")
    triggered_by: PolicyTier = Field(
        default=PolicyTier.ADAPTIVE,
        description="Which policy tier triggered this decision",
    )
    threat_level: ThreatLevel = Field(
        default=ThreatLevel.GREEN,
        description="Current Threat Level Indicator at time of decision",
    )

    # --- Scores from upstream layers ---
    innate_max_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Highest confidence from L2 innate scanners",
    )
    adaptive_mcav: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="MCAV score from L3 adaptive analysis",
    )
    fused_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Combined threat score after policy fusion",
    )

    # --- Reasoning (for audit / compliance) ---
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons for the decision",
    )
    applied_policies: list[str] = Field(
        default_factory=list,
        description="Policy rule IDs that contributed to the decision",
    )

    # --- Downstream instructions ---
    output_scrutiny_level: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplier for L5 output validation sensitivity (1.0 = normal)",
    )
    block_message: str = Field(
        default="Request blocked by AEGIS security policy.",
        description="Message returned to client when action is BLOCK",
    )

    @property
    def is_allowed(self) -> bool:
        """Whether the request should proceed to the model."""
        return self.action in (PolicyAction.ALLOW, PolicyAction.ALLOW_DEGRADED)
