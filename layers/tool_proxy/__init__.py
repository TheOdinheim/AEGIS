"""
Tool Invocation Proxy (TIP) — Extension 4.1

Intercepts all agent-to-tool communications and applies parameter validation,
scope enforcement, and rate limiting before any tool is actually invoked.

Operates as an in-process interceptor: when the model returns a function call /
tool use response, AEGIS intercepts it before the tool is invoked. The TIP
validates the invocation via the Tool Invocation Policy Engine (TIPE), and if
approved, allows it to proceed. If blocked, the tool call is replaced with an
error response back to the model.

ASSUMED-BREACH POSTURE: An attacker who controls the model's output can craft
arbitrary tool calls with malicious parameters (path traversal, SSRF, code
injection). The TIP treats every tool invocation as adversarial regardless of
which model produced it. A compromised tool server cannot influence the TIP's
policy decisions — validation happens before the request leaves AEGIS.
"""

from aegis.layers.tool_proxy.proxy import (
    ToolInvocation,
    ToolInvocationProxy,
    ToolProxyDecision,
    ToolResponse,
)
from aegis.layers.tool_proxy.policy_engine import (
    ParameterViolation,
    ToolInvocationPolicyEngine,
    ToolPolicy,
    ToolScope,
)

__all__ = [
    "ToolInvocation",
    "ToolInvocationProxy",
    "ToolProxyDecision",
    "ToolResponse",
    "ParameterViolation",
    "ToolInvocationPolicyEngine",
    "ToolPolicy",
    "ToolScope",
]
