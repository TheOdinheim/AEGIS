"""
Multi-Agent Security Layer — MHC Identity Verification.

Biological Analog: MHC molecules present antigens on cell surfaces. T-cells
verify identity before interacting. Cells lacking proper MHC-I are destroyed.

Orchestrates agent identity verification, tool authorization, and inter-agent
message validation. Every agent interaction is verified for identity, checked
for authorization, and scanned for injection before being allowed.

ASSUMED-BREACH POSTURE: Any agent in a multi-agent system may be compromised.
A compromised agent may attempt to escalate capabilities, inject instructions
into other agents, or exfiltrate data through tool calls. The security layer
operates independently of agent self-declarations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.agent_security.authorization import (
    AgentAuthorizationEngine,
    AuthorizationResult,
)
from aegis.layers.agent_security.identity import (
    AgentIdentity,
    AgentIdentityManager,
)
from aegis.layers.agent_security.message_validator import (
    AgentMessageValidator,
    MessageValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentSecurityResult:
    """Result from the full agent security check."""

    allowed: bool = False
    agent_identity: AgentIdentity | None = None
    reason: str = ""
    violations: list[str] = field(default_factory=list)


class AgentSecurityLayer:
    """Orchestrator for multi-agent security.

    Wraps identity verification, tool authorization, and message validation
    into a single security check pipeline.

    Usage:
        layer = AgentSecurityLayer()
        result = await layer.process_agent_request(agent_id, "file_read", {"path": "/data/file.txt"})
        if not result.allowed:
            print(f"Denied: {result.reason}")
    """

    def __init__(
        self,
        *,
        signing_key: str | None = None,
        innate_layer: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.identity_manager = AgentIdentityManager(signing_key=signing_key)
        self.authorization_engine = AgentAuthorizationEngine()
        self.message_validator = AgentMessageValidator(
            identity_manager=self.identity_manager,
            innate_layer=innate_layer,
        )
        self._event_bus = event_bus

    async def process_agent_request(
        self,
        agent_id: str,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> AgentSecurityResult:
        """Full agent security pipeline: verify → authorize → allow/deny.

        Args:
            agent_id: The agent's unique identifier.
            action: The tool/action being requested.
            params: Parameters for the action.

        Returns:
            AgentSecurityResult with allow/deny and details.
        """
        params = params or {}
        result = AgentSecurityResult()
        violations: list[str] = []

        # Step 1: Verify identity
        agent = await self.identity_manager.verify_agent(agent_id)
        if agent is None:
            result.reason = "Agent identity verification failed"
            violations.append(f"Unknown/expired/inactive agent: {agent_id}")
            result.violations = violations
            await self._publish_violation(agent_id, action, violations)
            return result

        result.agent_identity = agent

        # Step 2: Authorize tool/action
        auth_result = await self.authorization_engine.authorize_tool(
            agent, action, params,
        )
        if not auth_result.authorized:
            result.reason = auth_result.reason
            violations.append(f"Authorization denied: {auth_result.reason}")
            result.violations = violations
            await self._publish_violation(agent_id, action, violations)
            return result

        # All checks passed
        result.allowed = True
        result.reason = "authorized"
        return result

    async def validate_agent_message(
        self,
        sender_id: str,
        receiver_id: str,
        message: dict[str, Any],
    ) -> MessageValidationResult:
        """Validate an inter-agent message.

        Verifies sender identity and delegates to message validator.
        """
        sender = await self.identity_manager.verify_agent(sender_id)
        if sender is None:
            return MessageValidationResult(
                blocked_reason="sender_identity_verification_failed",
            )

        return await self.message_validator.validate_message(
            sender, receiver_id, message,
        )

    async def _publish_violation(
        self,
        agent_id: str,
        action: str,
        violations: list[str],
    ) -> None:
        """Publish agent security violation to event bus."""
        if not self._event_bus:
            return

        try:
            await self._event_bus.publish("agent_security", {
                "agent_id": agent_id,
                "action": action,
                "violations": violations,
            })
        except Exception as e:
            logger.error("Failed to publish agent security event: %s", e)

    def stats(self) -> dict[str, Any]:
        """Return agent security statistics."""
        return {
            "total_agents": self.identity_manager.registry_size,
            "active_agents": self.identity_manager.active_count,
            "average_trust": self.identity_manager.average_trust,
            "quarantined_count": self.message_validator.quarantined_count,
        }
