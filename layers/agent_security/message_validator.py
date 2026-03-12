"""
Agent Message Validator — Antigen Screening at Cell Junctions.

Biological Analog: Before immune cells communicate, they verify each other's
identity via MHC molecules and co-stimulatory signals. Messages (antigens)
are screened for dangerous content before being processed.

Validates inter-agent messages by checking sender identity, scanning message
content for injection attacks (via L2 innate scanner), verifying requested
actions are within sender capabilities, and enforcing per-agent quarantine.

ASSUMED-BREACH POSTURE: Every inter-agent message is assumed to contain a
potential injection attack. A compromised agent may attempt to inject
instructions into other agents via message content. The message validator
applies the same innate detection that protects against external prompt
injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.agent_security.identity import (
    AgentIdentity,
    AgentIdentityManager,
    TRUST_INJECTION_DETECTED,
)

logger = logging.getLogger(__name__)


@dataclass
class MessageValidationResult:
    """Result of inter-agent message validation."""

    valid: bool = False
    blocked_reason: str | None = None
    injection_detected: bool = False
    sanitized_message: dict[str, Any] | None = None
    sender_trust: float = 0.0


class AgentMessageValidator:
    """Validates inter-agent messages for safety.

    Uses L2 innate detection (when available) to scan message content
    for prompt injection. Maintains per-agent quarantine sets.
    """

    def __init__(
        self,
        identity_manager: AgentIdentityManager,
        innate_layer: Any | None = None,
    ) -> None:
        self._identity_manager = identity_manager
        self._innate = innate_layer
        self._quarantine_sets: dict[str, set[str]] = {}  # receiver_id → {quarantined sender_ids}

    async def validate_message(
        self,
        sender: AgentIdentity,
        receiver_id: str,
        message: dict[str, Any],
    ) -> MessageValidationResult:
        """Validate an inter-agent message.

        Checks in order:
        1. Sender identity is active and not expired
        2. Receiver has not quarantined the sender
        3. Message content scanned for injection (via L2 innate if available)
        4. Requested action within sender's capabilities

        Args:
            sender: The sending agent's identity.
            receiver_id: ID of the receiving agent.
            message: The message dict with at least a "content" field.

        Returns:
            MessageValidationResult with validation status.
        """
        result = MessageValidationResult(sender_trust=sender.trust_level)

        # 1. Sender identity check
        if not sender.is_active:
            result.blocked_reason = "sender_inactive"
            return result

        if sender.is_expired:
            result.blocked_reason = "sender_expired"
            return result

        # 2. Quarantine check
        if self.is_quarantined(receiver_id, sender.agent_id):
            result.blocked_reason = "sender_quarantined"
            return result

        # 3. Injection scanning
        content = message.get("content", "")
        if content and self._innate:
            injection_detected = await self._scan_for_injection(content)
            if injection_detected:
                result.injection_detected = True
                result.blocked_reason = "injection_detected_in_message"

                # Reduce sender trust
                await self._identity_manager.update_trust(
                    sender.agent_id,
                    TRUST_INJECTION_DETECTED,
                    "Injection detected in inter-agent message",
                )
                result.sender_trust = sender.trust_level
                return result
        elif content and not self._innate:
            logger.warning(
                "Innate layer not available — skipping injection scan for agent message"
            )

        # 4. Action capability check
        action = message.get("action")
        if action and action not in sender.capabilities:
            result.blocked_reason = (
                f"sender lacks capability for action '{action}'. "
                f"Has: {sender.capabilities}"
            )
            return result

        # All checks passed
        result.valid = True
        result.sanitized_message = message
        return result

    async def _scan_for_injection(self, content: str) -> bool:
        """Scan message content using the innate detection layer."""
        if not self._innate:
            return False

        try:
            from aegis.models.request_context import RequestContext, ChatMessage

            # Create a minimal context for scanning
            context = RequestContext(
                messages=[ChatMessage(role="user", content=content)],
            )
            report = await self._innate.scan(context)
            return report.should_block
        except Exception as e:
            logger.error("Injection scan failed: %s", e)
            return False

    def quarantine_sender(self, receiver_id: str, sender_id: str) -> None:
        """Quarantine a sender for a specific receiver."""
        if receiver_id not in self._quarantine_sets:
            self._quarantine_sets[receiver_id] = set()
        self._quarantine_sets[receiver_id].add(sender_id)
        logger.info("Agent %s quarantined by %s", sender_id, receiver_id)

    def unquarantine_sender(self, receiver_id: str, sender_id: str) -> None:
        """Remove a sender from quarantine."""
        if receiver_id in self._quarantine_sets:
            self._quarantine_sets[receiver_id].discard(sender_id)

    def is_quarantined(self, receiver_id: str, sender_id: str) -> bool:
        """Check if sender is quarantined by receiver."""
        return sender_id in self._quarantine_sets.get(receiver_id, set())

    @property
    def quarantined_count(self) -> int:
        """Total number of quarantine entries across all receivers."""
        return sum(len(s) for s in self._quarantine_sets.values())
