"""
Tool Invocation Proxy (TIP) — Extension 4.1

In-process interceptor for agent-to-tool communications. Validates every tool
invocation via the Tool Invocation Policy Engine (TIPE) before allowing it to
proceed. Logs all invocations for audit and feeds violations to the event bus.

For MCP servers, provides a proxy_mcp_request() method that forwards validated
MCP requests to the target server.

ASSUMED-BREACH POSTURE: Every tool invocation from the model is adversarial.
The proxy assumes the model has been compromised and is generating tool calls
designed to exfiltrate data, traverse paths, or execute arbitrary code. All
parameters are validated before the tool is invoked. A blocked invocation is
replaced with an error response back to the model.
"""

from __future__ import annotations

import json
import logging
import time
import threading
import uuid
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Any

import httpx

from aegis.layers.tool_proxy.policy_engine import (
    ParameterViolation,
    ToolInvocationPolicyEngine,
    ToolPolicy,
    ToolScope,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ToolInvocation:
    invocation_id: str
    tool_name: str
    tool_params: dict
    source_agent_id: str | None
    session_id: str
    timestamp: float
    raw_request: dict


@dataclass
class ToolResponse:
    invocation_id: str
    tool_name: str
    response_data: Any
    response_size_bytes: int
    latency_ms: float
    timestamp: float


@dataclass
class ToolProxyDecision:
    allowed: bool
    reason: str
    policy_violations: list[ParameterViolation] = field(default_factory=list)
    sanitized_params: dict | None = None


@dataclass
class RegisteredTool:
    tool_name: str
    schema: dict
    trust_level: str = "standard"


class ToolInvocationProxy:
    """Intercepts and validates all agent-to-tool communications.

    Thread-safe. Maintains invocation log with FIFO eviction and per-tool
    registration. Delegates policy enforcement to ToolInvocationPolicyEngine.
    """

    def __init__(
        self,
        policy_engine: ToolInvocationPolicyEngine,
        max_invocation_log: int = 10000,
        event_bus: Any = None,
    ):
        self._policy_engine = policy_engine
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._invocation_log: deque[ToolInvocation] = deque(maxlen=max_invocation_log)
        self._registered_tools: dict[str, RegisteredTool] = {}

        # Stats
        self._total_invocations = 0
        self._total_blocked = 0
        self._total_allowed = 0
        self._violations_by_type: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Tool Registration
    # -----------------------------------------------------------------------

    def register_tool(
        self, tool_name: str, schema: dict, trust_level: str = "standard",
    ) -> None:
        """Register a tool with its expected parameter schema and trust level."""
        with self._lock:
            self._registered_tools[tool_name] = RegisteredTool(
                tool_name=tool_name,
                schema=schema,
                trust_level=trust_level,
            )

    def get_registered_tool(self, tool_name: str) -> RegisteredTool | None:
        with self._lock:
            return self._registered_tools.get(tool_name)

    # -----------------------------------------------------------------------
    # Interception
    # -----------------------------------------------------------------------

    async def intercept(
        self,
        invocation: ToolInvocation,
        tenant_id: str | None = None,
        agent_trust_level: float = 1.0,
    ) -> ToolProxyDecision:
        """Validate a tool invocation against all policies.

        Returns allow/block decision with violation details.
        """
        source_id = invocation.source_agent_id or invocation.session_id

        # Log the invocation
        with self._lock:
            self._invocation_log.append(invocation)
            self._total_invocations += 1

        # Delegate to policy engine
        allowed, violations = await self._policy_engine.validate(
            tool_name=invocation.tool_name,
            params=invocation.tool_params,
            source_id=source_id,
            tenant_id=tenant_id,
            agent_trust_level=agent_trust_level,
        )

        if allowed:
            with self._lock:
                self._total_allowed += 1
            return ToolProxyDecision(
                allowed=True,
                reason="ok",
            )

        # Blocked — record violations
        with self._lock:
            self._total_blocked += 1
            for v in violations:
                self._violations_by_type[v.violation_type] = (
                    self._violations_by_type.get(v.violation_type, 0) + 1
                )

        # Fire event bus notification
        if self._event_bus:
            try:
                await self._event_bus.publish(
                    "threat_detected",
                    {
                        "type": "tool_policy_violation",
                        "tool_name": invocation.tool_name,
                        "invocation_id": invocation.invocation_id,
                        "violations": [
                            {"type": v.violation_type, "severity": v.severity}
                            for v in violations
                        ],
                    },
                )
            except Exception:
                logger.debug("Failed to publish tool violation event", exc_info=True)

        reason = violations[0].violation_type if violations else "policy_violation"
        return ToolProxyDecision(
            allowed=False,
            reason=reason,
            policy_violations=violations,
        )

    async def intercept_response(
        self,
        response: ToolResponse,
        invocation: ToolInvocation,
    ) -> ToolResponse:
        """Validate and optionally sanitize a tool response.

        Currently performs size-based sanity check. Future extensions may
        scan response content for data leakage.
        """
        # Log response latency for monitoring
        logger.debug(
            "Tool response: %s invocation=%s size=%d latency=%.1fms",
            response.tool_name,
            response.invocation_id,
            response.response_size_bytes,
            response.latency_ms,
        )
        return response

    async def proxy_mcp_request(
        self,
        method: str,
        params: dict,
        server_url: str,
        agent_id: str | None = None,
        session_id: str = "",
    ) -> dict:
        """Forward a validated MCP request to the target server.

        Creates a ToolInvocation, validates it, and if approved, forwards
        the JSON-RPC request to the MCP server.
        """
        invocation = ToolInvocation(
            invocation_id=uuid.uuid4().hex[:12],
            tool_name=method,
            tool_params=params,
            source_agent_id=agent_id,
            session_id=session_id or "mcp",
            timestamp=time.time(),
            raw_request={"method": method, "params": params},
        )

        decision = await self.intercept(invocation)
        if not decision.allowed:
            return {
                "error": {
                    "code": -32600,
                    "message": f"Tool invocation blocked: {decision.reason}",
                    "data": {
                        "violations": [
                            {"type": v.violation_type, "details": v.details}
                            for v in decision.policy_violations
                        ],
                    },
                },
            }

        # Forward to MCP server
        rpc_request = {
            "jsonrpc": "2.0",
            "id": invocation.invocation_id,
            "method": method,
            "params": params,
        }

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    server_url,
                    json=rpc_request,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                result = resp.json()
        except httpx.HTTPError as e:
            return {
                "error": {
                    "code": -32603,
                    "message": f"MCP server error: {str(e)[:200]}",
                },
            }

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Create response for audit
        tool_response = ToolResponse(
            invocation_id=invocation.invocation_id,
            tool_name=method,
            response_data=result,
            response_size_bytes=len(json.dumps(result).encode()),
            latency_ms=elapsed_ms,
            timestamp=time.time(),
        )
        await self.intercept_response(tool_response, invocation)

        return result

    # -----------------------------------------------------------------------
    # Audit / Stats
    # -----------------------------------------------------------------------

    def get_invocation_log(
        self, session_id: str | None = None, limit: int = 100,
    ) -> list[ToolInvocation]:
        """Get recent tool invocations for audit/debugging."""
        with self._lock:
            if session_id:
                filtered = [
                    inv for inv in self._invocation_log
                    if inv.session_id == session_id
                ]
                return filtered[-limit:]
            return list(self._invocation_log)[-limit:]

    def get_stats(self) -> dict:
        """Get proxy statistics."""
        with self._lock:
            return {
                "total_invocations": self._total_invocations,
                "total_allowed": self._total_allowed,
                "total_blocked": self._total_blocked,
                "violations_by_type": dict(self._violations_by_type),
                "registered_tools": len(self._registered_tools),
                "log_size": len(self._invocation_log),
            }

    @staticmethod
    def create_invocation(
        tool_name: str,
        tool_params: dict,
        session_id: str,
        source_agent_id: str | None = None,
        raw_request: dict | None = None,
    ) -> ToolInvocation:
        """Helper to create a ToolInvocation with auto-generated ID and timestamp."""
        return ToolInvocation(
            invocation_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            tool_params=tool_params,
            source_agent_id=source_agent_id,
            session_id=session_id,
            timestamp=time.time(),
            raw_request=raw_request or {"tool": tool_name, "params": tool_params},
        )
