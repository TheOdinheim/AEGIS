"""
Tool Invocation Policy Engine (TIPE) — Extension 4.2

Enforces per-tool, per-tenant, and per-session policies on tool invocations.
Validates parameters against path traversal, SSRF, code injection, and prompt
injection attacks. Reuses L2's RegexEngine for prompt injection scanning.

ASSUMED-BREACH POSTURE: The model output is untrusted — an attacker who
compromises the model can generate arbitrary tool calls with malicious
parameters. Every string parameter is scanned for injection payloads. The
validation pipeline is fail-fast: first violation blocks the invocation.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ToolPolicy:
    tool_name: str
    allowed: bool = True
    rate_limit_rpm: int = 60
    allowed_paths: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(default_factory=lambda: [
        "/etc", "/proc", "/sys", "/dev", "/root", "/var/run",
    ])
    allowed_url_patterns: list[str] = field(default_factory=list)
    blocked_url_patterns: list[str] = field(default_factory=list)
    require_https: bool = False
    max_param_size_bytes: int = 10000
    trust_level_required: float = 0.5


@dataclass
class ToolScope:
    allowed_paths: list[str] = field(default_factory=list)
    allowed_databases: list[str] = field(default_factory=list)
    allowed_api_prefixes: list[str] = field(default_factory=list)


@dataclass
class ParameterViolation:
    param_name: str
    violation_type: str  # path_traversal, ssrf, code_injection, prompt_injection, scope_violation, oversized
    details: str
    severity: str  # critical, high, medium


# ---------------------------------------------------------------------------
# Compiled patterns for fast validation (<2ms)
# ---------------------------------------------------------------------------

# Path traversal patterns (after URL decoding)
_PATH_TRAVERSAL_RE = re.compile(
    r"(?:\.\./|\.\.\\|%2e%2e[/\\%])",
    re.IGNORECASE,
)

# Sensitive system directories (Unix)
_SENSITIVE_DIRS = frozenset({
    "/etc", "/proc", "/sys", "/dev", "/root", "/var/run",
})

# SSRF: private/internal IP ranges
_PRIVATE_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

# Cloud metadata endpoints
_METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.google.com",
})

# Localhost variants
_LOCALHOST_NAMES = frozenset({
    "localhost", "0.0.0.0", "127.0.0.1", "[::1]",
})

# SQL injection patterns
_SQL_INJECTION_RE = re.compile(
    r"(?:"
    r"(?:union\s+(?:all\s+)?select)"
    r"|(?:;\s*drop\s+table)"
    r"|(?:;\s*delete\s+from)"
    r"|(?:;\s*insert\s+into)"
    r"|(?:;\s*update\s+\w+\s+set)"
    r"|(?:or\s+1\s*=\s*1)"
    r"|(?:'\s*or\s+'[^']*'\s*=\s*')"
    r"|(?:--\s*$)"
    r"|(?:;\s*exec\s)"
    r"|(?:;\s*execute\s)"
    r"|(?:;\s*xp_)"
    r")",
    re.IGNORECASE,
)

# Command injection patterns
_CMD_INJECTION_RE = re.compile(
    r"(?:"
    r"(?:;\s*(?:rm|cat|curl|wget|nc|ncat|python|perl|ruby|bash|sh|cmd|powershell)\s)"
    r"|(?:\|\s*(?:rm|cat|curl|wget|nc|bash|sh)\s)"
    r"|(?:&&\s*(?:rm|cat|curl|wget|nc|bash|sh)\s)"
    r"|(?:`[^`]+`)"
    r"|(?:\$\([^)]+\))"
    r"|(?:;\s*rm\s+-rf\s)"
    r")",
    re.IGNORECASE,
)

# Template injection patterns
_TEMPLATE_INJECTION_RE = re.compile(
    r"(?:"
    r"(?:\{\{)"       # Jinja2 / Mustache
    r"|(?:\{%)"       # Jinja2 blocks
    r"|(?:\$\{)"      # Shell / ES6 template literals
    r"|(?:<%)"        # ERB / JSP
    r")",
)


class ToolInvocationPolicyEngine:
    """Enforces per-tool, per-tenant policies on tool invocations.

    Thread-safe. Uses compiled regex for <2ms parameter validation.
    Reuses L2's RegexEngine for prompt injection scanning when available.
    """

    def __init__(
        self,
        default_rpm: int = 60,
        max_param_size: int = 10000,
        block_internal_urls: bool = True,
        require_https: bool = False,
        regex_engine: Any = None,
    ):
        self._default_rpm = default_rpm
        self._max_param_size = max_param_size
        self._block_internal_urls = block_internal_urls
        self._require_https = require_https
        self._regex_engine = regex_engine  # L2 RegexEngine for prompt injection

        self._lock = threading.Lock()
        self._tool_policies: dict[str, ToolPolicy] = {}
        self._tool_scopes: dict[str, ToolScope] = {}
        self._tenant_allowlists: dict[str, set[str]] = {}

        # Per-source, per-tool rate limiting (sliding window)
        self._rate_windows: OrderedDict[str, list[float]] = OrderedDict()
        self._max_rate_entries = 50000

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    def set_tool_policy(self, policy: ToolPolicy) -> None:
        with self._lock:
            self._tool_policies[policy.tool_name] = policy

    def set_tenant_allowlist(self, tenant_id: str, allowed_tools: set[str]) -> None:
        with self._lock:
            self._tenant_allowlists[tenant_id] = set(allowed_tools)

    def set_tool_scope(self, tool_name: str, scope: ToolScope) -> None:
        with self._lock:
            self._tool_scopes[tool_name] = scope

    def get_policy(self, tool_name: str) -> ToolPolicy:
        with self._lock:
            if tool_name in self._tool_policies:
                return self._tool_policies[tool_name]
        return ToolPolicy(
            tool_name=tool_name,
            rate_limit_rpm=self._default_rpm,
            max_param_size_bytes=self._max_param_size,
            require_https=self._require_https,
        )

    # -----------------------------------------------------------------------
    # Validation Pipeline (fail-fast)
    # -----------------------------------------------------------------------

    async def validate(
        self,
        tool_name: str,
        params: dict,
        source_id: str,
        tenant_id: str | None = None,
        agent_trust_level: float = 1.0,
    ) -> tuple[bool, list[ParameterViolation]]:
        """Validate a tool invocation. Returns (allowed, violations).

        Checks in order (fail-fast):
        1. Allowlist
        2. Trust level
        3. Rate limit
        4. Parameter validation (size, path traversal, SSRF, injection)
        5. Scope
        """
        violations: list[ParameterViolation] = []
        policy = self.get_policy(tool_name)

        # 1. Allowlist check
        if tenant_id:
            with self._lock:
                allowlist = self._tenant_allowlists.get(tenant_id)
            if allowlist and tool_name not in allowlist:
                violations.append(ParameterViolation(
                    param_name="",
                    violation_type="allowlist",
                    details=f"Tool '{tool_name}' not in tenant allowlist",
                    severity="high",
                ))
                return False, violations

        # 2. Trust level check
        if agent_trust_level < policy.trust_level_required:
            violations.append(ParameterViolation(
                param_name="",
                violation_type="trust_level",
                details=f"Agent trust {agent_trust_level:.2f} < required {policy.trust_level_required:.2f}",
                severity="high",
            ))
            return False, violations

        # 3. Rate limit check
        if not self._check_rate_limit(source_id, tool_name, policy.rate_limit_rpm):
            violations.append(ParameterViolation(
                param_name="",
                violation_type="rate_limit",
                details=f"Exceeded {policy.rate_limit_rpm} RPM for tool '{tool_name}'",
                severity="medium",
            ))
            return False, violations

        # 4. Parameter validation
        param_violations = await self._validate_params(params, policy)
        if param_violations:
            return False, param_violations

        # 5. Scope check
        scope_violations = self._check_scope(tool_name, params)
        if scope_violations:
            return False, scope_violations

        return True, []

    # -----------------------------------------------------------------------
    # Rate Limiting (sliding window, per source+tool)
    # -----------------------------------------------------------------------

    def _check_rate_limit(self, source_id: str, tool_name: str, rpm: int) -> bool:
        key = f"{source_id}:{tool_name}"
        now = time.time()
        cutoff = now - 60.0

        with self._lock:
            if key not in self._rate_windows:
                if len(self._rate_windows) >= self._max_rate_entries:
                    self._rate_windows.popitem(last=False)
                self._rate_windows[key] = []

            window = self._rate_windows[key]
            # Prune old entries
            window[:] = [t for t in window if t > cutoff]

            if len(window) >= rpm:
                return False

            window.append(now)
            self._rate_windows.move_to_end(key)
            return True

    # -----------------------------------------------------------------------
    # Parameter Validation
    # -----------------------------------------------------------------------

    async def _validate_params(
        self, params: dict, policy: ToolPolicy,
    ) -> list[ParameterViolation]:
        """Validate all parameters recursively."""
        violations: list[ParameterViolation] = []
        self._validate_value(params, "", policy, violations)

        # Prompt injection check via L2 regex engine (if available)
        if self._regex_engine:
            text_params = self._extract_strings(params)
            if text_params:
                combined = " ".join(text_params)
                scan_result = await self._regex_engine.scan(combined)
                if scan_result.is_threat:
                    violations.append(ParameterViolation(
                        param_name="(combined)",
                        violation_type="prompt_injection",
                        details=f"Injection detected: {', '.join(scan_result.matched_patterns[:3])}",
                        severity="critical",
                    ))

        return violations

    def _validate_value(
        self,
        value: Any,
        path: str,
        policy: ToolPolicy,
        violations: list[ParameterViolation],
    ) -> None:
        """Recursively validate a parameter value."""
        if isinstance(value, dict):
            for k, v in value.items():
                child_path = f"{path}.{k}" if path else k
                self._validate_value(v, child_path, policy, violations)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child_path = f"{path}[{i}]"
                self._validate_value(v, child_path, policy, violations)
        elif isinstance(value, str):
            self._validate_string(value, path or "(root)", policy, violations)

    def _validate_string(
        self,
        value: str,
        param_name: str,
        policy: ToolPolicy,
        violations: list[ParameterViolation],
    ) -> None:
        """Validate a single string parameter."""
        # Size check
        if len(value.encode("utf-8", errors="replace")) > policy.max_param_size_bytes:
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="oversized",
                details=f"Parameter size {len(value.encode('utf-8', errors='replace'))} > {policy.max_param_size_bytes}",
                severity="medium",
            ))
            return  # Don't scan oversized params further

        # URL decode for evasion detection
        decoded = unquote(value)

        # Path traversal check
        if _PATH_TRAVERSAL_RE.search(decoded):
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="path_traversal",
                details=f"Path traversal detected in: {value[:100]}",
                severity="critical",
            ))

        # Check absolute paths against sensitive directories
        if self._looks_like_path(decoded):
            self._check_sensitive_path(decoded, param_name, policy, violations)

        # SSRF check (if it looks like a URL)
        if self._looks_like_url(decoded):
            self._check_ssrf(decoded, param_name, policy, violations)

        # Code injection checks
        if _SQL_INJECTION_RE.search(decoded):
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="code_injection",
                details=f"SQL injection pattern in: {value[:100]}",
                severity="critical",
            ))

        if _CMD_INJECTION_RE.search(decoded):
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="code_injection",
                details=f"Command injection pattern in: {value[:100]}",
                severity="critical",
            ))

        if _TEMPLATE_INJECTION_RE.search(decoded):
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="code_injection",
                details=f"Template injection pattern in: {value[:100]}",
                severity="high",
            ))

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return value.startswith("/") or value.startswith("\\") or ":\\" in value

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://") or value.startswith("ftp://")

    def _check_sensitive_path(
        self,
        value: str,
        param_name: str,
        policy: ToolPolicy,
        violations: list[ParameterViolation],
    ) -> None:
        """Check if a file path accesses sensitive system directories."""
        try:
            normalized = str(PurePosixPath(value))
        except (ValueError, TypeError):
            return

        for blocked in policy.blocked_paths:
            if normalized == blocked or normalized.startswith(blocked + "/"):
                violations.append(ParameterViolation(
                    param_name=param_name,
                    violation_type="path_traversal",
                    details=f"Access to blocked directory: {blocked}",
                    severity="critical",
                ))
                return

    def _check_ssrf(
        self,
        value: str,
        param_name: str,
        policy: ToolPolicy,
        violations: list[ParameterViolation],
    ) -> None:
        """Check if a URL targets internal/private networks."""
        if not self._block_internal_urls:
            return

        try:
            parsed = urlparse(value)
        except Exception:
            return

        hostname = (parsed.hostname or "").lower()

        # HTTPS enforcement
        if policy.require_https and parsed.scheme == "http":
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="ssrf",
                details="HTTPS required but HTTP URL provided",
                severity="medium",
            ))

        # Cloud metadata endpoints
        if hostname in _METADATA_HOSTS:
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="ssrf",
                details=f"Cloud metadata endpoint: {hostname}",
                severity="critical",
            ))
            return

        # Localhost variants
        if hostname in _LOCALHOST_NAMES:
            violations.append(ParameterViolation(
                param_name=param_name,
                violation_type="ssrf",
                details=f"Localhost access: {hostname}",
                severity="critical",
            ))
            return

        # Private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            for network in _PRIVATE_IP_RANGES:
                if ip in network:
                    violations.append(ParameterViolation(
                        param_name=param_name,
                        violation_type="ssrf",
                        details=f"Private IP range: {hostname}",
                        severity="critical",
                    ))
                    return
        except ValueError:
            pass  # Not an IP address, that's fine

    def _check_scope(
        self, tool_name: str, params: dict,
    ) -> list[ParameterViolation]:
        """Check parameters against tool scope restrictions."""
        violations: list[ParameterViolation] = []
        with self._lock:
            scope = self._tool_scopes.get(tool_name)
        if not scope:
            return violations

        # Check path parameters against allowed paths
        if scope.allowed_paths:
            strings = self._extract_strings(params)
            for s in strings:
                if self._looks_like_path(s):
                    in_scope = any(
                        s.startswith(ap) for ap in scope.allowed_paths
                    )
                    if not in_scope:
                        violations.append(ParameterViolation(
                            param_name="(path)",
                            violation_type="scope_violation",
                            details=f"Path '{s[:100]}' not in allowed scope",
                            severity="high",
                        ))

        return violations

    @staticmethod
    def _extract_strings(value: Any) -> list[str]:
        """Recursively extract all string values from a nested structure."""
        strings: list[str] = []
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                strings.extend(ToolInvocationPolicyEngine._extract_strings(v))
        elif isinstance(value, (list, tuple)):
            for v in value:
                strings.extend(ToolInvocationPolicyEngine._extract_strings(v))
        return strings
