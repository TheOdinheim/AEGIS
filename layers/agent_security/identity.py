"""
Agent Identity Manager — MHC-I Molecules.

Biological Analog: MHC-I molecules present self-antigens on every nucleated
cell's surface, allowing NK cells and CD8+ T-cells to verify identity.
Cells lacking proper MHC-I are destroyed.

Every agent receives a cryptographic identity encoding capabilities, scope,
trust level, and lineage. Agents without valid identity are rejected.

ASSUMED-BREACH POSTURE: Agent identities may be stolen, forged, or replayed.
JWT tokens are signed with HMAC-SHA256 but an attacker with the signing key
can forge any identity. Trust levels decay over time and are reduced on
violations. The identity system is one layer of defense — authorization and
message validation provide independent checks.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Trust level adjustments
TRUST_CLEAN_INTERACTION = 0.02
TRUST_POLICY_VIOLATION = -0.1
TRUST_CONFIRMED_ATTACK = -0.5
TRUST_INJECTION_DETECTED = -0.2
TRUST_DECAY_PER_DAY = 0.1
TRUST_DECAY_THRESHOLD_HOURS = 24

# Default trust for new agents
DEFAULT_TRUST = 0.5

# JWT token expiry
DEFAULT_TTL_HOURS = 24


@dataclass
class AgentIdentity:
    """Cryptographic identity for an AI agent."""

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_agent_id: str | None = None
    capabilities: list[str] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)
    trust_level: float = DEFAULT_TRUST
    lineage: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=DEFAULT_TTL_HOURS)
    )
    is_active: bool = True
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "capabilities": self.capabilities,
            "scope": self.scope,
            "trust_level": round(self.trust_level, 4),
            "lineage": self.lineage,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_active": self.is_active,
            "last_active": self.last_active.isoformat(),
        }

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class AgentIdentityManager:
    """Manages agent identities with JWT-signed tokens.

    In-memory registry for bootstrap. Production upgrade: PostgreSQL agents table.
    """

    def __init__(self, *, signing_key: str | None = None) -> None:
        if signing_key:
            self._signing_key = signing_key
        else:
            env_key = os.environ.get("AEGIS_AGENT_SIGNING_KEY")
            if env_key:
                self._signing_key = env_key
            else:
                self._signing_key = secrets.token_hex(32)
                logger.warning(
                    "AEGIS_AGENT_SIGNING_KEY not set — generated ephemeral key. "
                    "Agent tokens will not persist across restarts."
                )

        self._registry: dict[str, AgentIdentity] = {}
        self._trust_log: list[dict[str, Any]] = []
        self._registry_lock = asyncio.Lock()

    @property
    def registry_size(self) -> int:
        return len(self._registry)

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._registry.values() if a.is_active and not a.is_expired)

    @property
    def average_trust(self) -> float:
        active = [a for a in self._registry.values() if a.is_active and not a.is_expired]
        if not active:
            return 0.0
        return round(sum(a.trust_level for a in active) / len(active), 4)

    async def register_agent(
        self,
        parent_id: str | None = None,
        capabilities: list[str] | None = None,
        scope: dict[str, Any] | None = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> AgentIdentity:
        """Register a new agent and return its signed identity."""
        now = datetime.now(timezone.utc)

        # Build lineage from parent
        lineage: list[str] = []
        if parent_id and parent_id in self._registry:
            parent = self._registry[parent_id]
            lineage = parent.lineage + [parent_id]

        agent = AgentIdentity(
            parent_agent_id=parent_id,
            capabilities=capabilities or [],
            scope=scope or {},
            trust_level=DEFAULT_TRUST,
            lineage=lineage,
            created_at=now,
            expires_at=now + timedelta(hours=ttl_hours),
            last_active=now,
        )

        # Sign JWT token
        agent.token = self._sign_token(agent)

        self._registry[agent.agent_id] = agent
        logger.info(
            "Registered agent %s (parent=%s, capabilities=%s, ttl=%dh)",
            agent.agent_id, parent_id, capabilities, ttl_hours,
        )
        return agent

    async def verify_agent(self, agent_id: str) -> AgentIdentity | None:
        """Verify an agent identity. Returns None if invalid."""
        agent = self._registry.get(agent_id)
        if agent is None:
            return None

        if not agent.is_active:
            return None

        if agent.is_expired:
            return None

        # Apply trust decay for inactive agents
        self._apply_trust_decay(agent)

        # Update last_active
        agent.last_active = datetime.now(timezone.utc)

        return agent

    async def verify_token(self, token: str) -> AgentIdentity | None:
        """Verify a JWT token and return the agent identity."""
        payload = self._verify_token_signature(token)
        if payload is None:
            return None

        agent_id = payload.get("agent_id")
        if not agent_id:
            return None

        return await self.verify_agent(agent_id)

    async def update_trust(
        self, agent_id: str, delta: float, reason: str,
    ) -> float:
        """Adjust an agent's trust level. Returns new trust level."""
        agent = self._registry.get(agent_id)
        if agent is None:
            return 0.0

        old_trust = agent.trust_level
        agent.trust_level = max(0.0, min(1.0, agent.trust_level + delta))

        self._trust_log.append({
            "agent_id": agent_id,
            "old_trust": round(old_trust, 4),
            "new_trust": round(agent.trust_level, 4),
            "delta": round(delta, 4),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            "Trust update for %s: %.3f → %.3f (delta=%.3f, reason=%s)",
            agent_id, old_trust, agent.trust_level, delta, reason,
        )

        return agent.trust_level

    async def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent. Returns True if found and deactivated."""
        agent = self._registry.get(agent_id)
        if agent is None:
            return False

        agent.is_active = False
        logger.info("Deactivated agent %s", agent_id)
        return True

    def _apply_trust_decay(self, agent: AgentIdentity) -> None:
        """Apply trust decay for agents inactive > 24h."""
        now = datetime.now(timezone.utc)
        hours_inactive = (now - agent.last_active).total_seconds() / 3600

        if hours_inactive < TRUST_DECAY_THRESHOLD_HOURS:
            return

        days_over = (hours_inactive - TRUST_DECAY_THRESHOLD_HOURS) / 24
        decay = days_over * TRUST_DECAY_PER_DAY

        if decay > 0:
            old_trust = agent.trust_level
            agent.trust_level = max(0.0, agent.trust_level - decay)
            if agent.trust_level != old_trust:
                logger.debug(
                    "Trust decay for %s: %.3f → %.3f (%.1f days inactive)",
                    agent.agent_id, old_trust, agent.trust_level, days_over,
                )

    def _sign_token(self, agent: AgentIdentity) -> str:
        """Create a signed JWT for the agent."""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "agent_id": agent.agent_id,
            "capabilities": agent.capabilities,
            "scope": agent.scope,
            "trust_level": agent.trust_level,
            "lineage": agent.lineage,
            "exp": int(agent.expires_at.timestamp()),
            "iat": int(agent.created_at.timestamp()),
        }

        header_b64 = _b64_encode(json.dumps(header, separators=(",", ":")))
        payload_b64 = _b64_encode(json.dumps(payload, separators=(",", ":")))
        message = f"{header_b64}.{payload_b64}"
        sig = hmac.new(
            self._signing_key.encode(), message.encode(), hashlib.sha256,
        ).digest()
        sig_b64 = urlsafe_b64encode(sig).rstrip(b"=").decode()

        return f"{message}.{sig_b64}"

    def _verify_token_signature(self, token: str) -> dict[str, Any] | None:
        """Verify JWT signature and return payload, or None if invalid."""
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        message = f"{header_b64}.{payload_b64}"

        # Recompute signature
        expected_sig = hmac.new(
            self._signing_key.encode(), message.encode(), hashlib.sha256,
        ).digest()
        expected_b64 = urlsafe_b64encode(expected_sig).rstrip(b"=").decode()

        if not hmac.compare_digest(sig_b64, expected_b64):
            return None

        try:
            payload_json = _b64_decode(payload_b64)
            payload = json.loads(payload_json)
        except (json.JSONDecodeError, ValueError):
            return None

        # Check expiry
        exp = payload.get("exp", 0)
        if time.time() > exp:
            return None

        return payload

    def get_trust_log(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        """Get trust adjustment log, optionally filtered by agent."""
        if agent_id:
            return [e for e in self._trust_log if e["agent_id"] == agent_id]
        return list(self._trust_log)


def _b64_encode(data: str) -> str:
    return urlsafe_b64encode(data.encode()).rstrip(b"=").decode()


def _b64_decode(data: str) -> str:
    # Add padding
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return urlsafe_b64decode(data).decode()
