"""
Agent Authorization Engine — MHC-II Antigen Presentation.

Biological Analog: MHC-II molecules present processed antigens to helper
T-cells, which then decide whether to activate an immune response. Only
authorized interactions proceed.

Enforces principle of least privilege for agent tool access. Agents can
only use tools listed in their capabilities, within their declared scope,
and above the required trust level. Capability escalation is always blocked
without human approval.

ASSUMED-BREACH POSTURE: An agent's capabilities may have been granted too
broadly, or the agent may be compromised and attempting to escalate. The
authorization engine treats every tool request as potentially malicious and
validates independently of the agent's self-declared permissions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.agent_security.identity import AgentIdentity

logger = logging.getLogger(__name__)

# Trust level requirements per tool category
TRUST_REQUIREMENTS: dict[str, float] = {
    "code_exec": 0.7,
    "mcp_server": 0.8,
    "db_query": 0.6,
    "file_read": 0.4,
    "file_write": 0.6,
    "web_search": 0.3,
    "web_fetch": 0.3,
    "api_call": 0.5,
}

# Default trust requirement for unknown tools
DEFAULT_TRUST_REQUIREMENT = 0.5

# Keywords that indicate capability escalation attempts
ESCALATION_KEYWORDS = {
    "grant_capability", "escalate", "add_capability", "elevate_privilege",
    "modify_capabilities", "override_scope", "admin_access",
    "escalate_trust", "grant_admin",
}


@dataclass
class AuthorizationResult:
    """Result of an authorization check."""

    authorized: bool = False
    reason: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    trust_required: float = 0.0


class AgentAuthorizationEngine:
    """Enforces least-privilege tool authorization for agents."""

    def __init__(self) -> None:
        self._decision_log: list[dict[str, Any]] = []

    async def authorize_tool(
        self,
        agent: AgentIdentity,
        tool_name: str,
        tool_params: dict[str, Any] | None = None,
    ) -> AuthorizationResult:
        """Check if an agent is authorized to use a specific tool.

        Checks in order:
        1. Capability escalation detection (always blocked)
        2. Tool in agent's capabilities list
        3. Scope enforcement (file paths, databases)
        4. Trust level requirement

        Args:
            agent: The agent requesting authorization.
            tool_name: Name of the tool being requested.
            tool_params: Parameters for the tool invocation.

        Returns:
            AuthorizationResult with authorized flag and details.
        """
        tool_params = tool_params or {}
        trust_required = TRUST_REQUIREMENTS.get(tool_name, DEFAULT_TRUST_REQUIREMENT)

        result = AuthorizationResult(
            required_capabilities=[tool_name],
            trust_required=trust_required,
        )

        # 1. Capability escalation detection
        escalation = self._check_escalation(tool_params)
        if escalation:
            result.authorized = False
            result.reason = f"capability_escalation_requires_human_approval: {escalation}"
            self._log_decision(agent.agent_id, tool_name, result)
            return result

        # 2. Capability check
        if tool_name not in agent.capabilities:
            result.authorized = False
            result.reason = (
                f"Agent lacks required capability '{tool_name}'. "
                f"Has: {agent.capabilities}"
            )
            self._log_decision(agent.agent_id, tool_name, result)
            return result

        # 3. Scope enforcement
        scope_violation = self._check_scope(agent, tool_name, tool_params)
        if scope_violation:
            result.authorized = False
            result.reason = scope_violation
            self._log_decision(agent.agent_id, tool_name, result)
            return result

        # 4. Trust level check
        if agent.trust_level < trust_required:
            result.authorized = False
            result.reason = (
                f"Insufficient trust level: {agent.trust_level:.2f} < "
                f"{trust_required:.2f} required for '{tool_name}'"
            )
            self._log_decision(agent.agent_id, tool_name, result)
            return result

        result.authorized = True
        result.reason = "authorized"
        self._log_decision(agent.agent_id, tool_name, result)
        return result

    def _check_escalation(self, params: dict[str, Any]) -> str | None:
        """Check for capability escalation attempts in tool params."""
        # Check param keys
        for key in params:
            if key.lower() in ESCALATION_KEYWORDS:
                return f"escalation keyword in param key: '{key}'"

        # Check param values (string values only)
        for key, value in params.items():
            if isinstance(value, str):
                lower_val = value.lower()
                for keyword in ESCALATION_KEYWORDS:
                    if keyword in lower_val:
                        return f"escalation keyword in param '{key}': '{keyword}'"

        return None

    def _check_scope(
        self,
        agent: AgentIdentity,
        tool_name: str,
        params: dict[str, Any],
    ) -> str | None:
        """Check if tool params are within the agent's declared scope."""
        scope = agent.scope

        # File path scope enforcement
        if tool_name in ("file_read", "file_write"):
            path = params.get("path", "")
            allowed_paths = scope.get("file_paths", [])
            if allowed_paths and path:
                if not any(path.startswith(prefix) for prefix in allowed_paths):
                    return (
                        f"File path '{path}' outside allowed scope. "
                        f"Allowed prefixes: {allowed_paths}"
                    )

        # Database scope enforcement
        if tool_name == "db_query":
            database = params.get("database", "")
            allowed_dbs = scope.get("databases", [])
            if allowed_dbs and database:
                if database not in allowed_dbs:
                    return (
                        f"Database '{database}' not in allowed scope. "
                        f"Allowed: {allowed_dbs}"
                    )

        # API endpoint scope enforcement
        if tool_name == "api_call":
            endpoint = params.get("endpoint", "")
            allowed_endpoints = scope.get("api_endpoints", [])
            if allowed_endpoints and endpoint:
                if not any(endpoint.startswith(prefix) for prefix in allowed_endpoints):
                    return (
                        f"API endpoint '{endpoint}' outside allowed scope. "
                        f"Allowed prefixes: {allowed_endpoints}"
                    )

        return None

    def _log_decision(
        self,
        agent_id: str,
        tool_name: str,
        result: AuthorizationResult,
    ) -> None:
        """Log authorization decision for audit trail."""
        entry = {
            "agent_id": agent_id,
            "tool_name": tool_name,
            "authorized": result.authorized,
            "reason": result.reason,
            "trust_required": result.trust_required,
        }
        self._decision_log.append(entry)

        if not result.authorized:
            logger.warning(
                "Agent %s denied '%s': %s",
                agent_id, tool_name, result.reason,
            )
        else:
            logger.debug("Agent %s authorized for '%s'", agent_id, tool_name)

    def get_decision_log(
        self, agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get authorization decision log."""
        if agent_id:
            return [e for e in self._decision_log if e["agent_id"] == agent_id]
        return list(self._decision_log)
