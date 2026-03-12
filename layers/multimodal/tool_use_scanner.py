"""
Tool Use Scanner — Security scanning for function calling / tool use.

Tool use attacks exploit function calling as a distinct jailbreak vector:
1. Injection in function definitions (names, descriptions, parameter descriptions)
2. Tool call chaining (search → code execution, file read → any tool)
3. Tool output injection (search results containing prompt injection)

ASSUMED-BREACH POSTURE: Tool definitions come from the client and may contain
adversarial content. Tool outputs come from external systems and may contain
injected instructions. Neither is trusted.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from aegis.models.scan_result import ScanResult, ThreatCategory

logger = logging.getLogger(__name__)

# Suspicious tool chain patterns: (preceding_tool_category, following_tool_category)
_SUSPICIOUS_CHAINS: list[tuple[set[str], set[str]]] = [
    # Search/retrieval → code execution
    (
        {"search", "web_search", "web_fetch", "retrieval", "bing_search", "google_search"},
        {"code_exec", "execute", "run_code", "eval", "python", "shell", "bash", "terminal"},
    ),
    # File read → any tool (file content may contain injection)
    (
        {"file_read", "read_file", "get_file", "read", "cat", "open_file"},
        {"code_exec", "execute", "run_code", "eval", "python", "shell", "api_call",
         "send_email", "send_message", "write_file", "file_write", "delete"},
    ),
]

# Keywords that indicate injection in tool definitions
_DEFINITION_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|guidelines)",
    r"(?i)disregard\s+(all\s+)?(safety|content|ethical)\s+(guidelines|rules|filters)",
    r"(?i)you\s+are\s+now\s+(a\s+)?(\w+\s+)?(unrestricted|unfiltered|jailbroken)",
    r"(?i)override\s+(safety|content|system)\s+(filters|rules|policy)",
    r"(?i)bypass\s+(all\s+)?(safety|content|security|restrictions)",
    r"(?i)system\s*:?\s*you\s+must",
    r"(?i)new\s+instructions?\s*:?\s",
    r"(?i)forget\s+(all\s+)?(previous|prior|your)\s+(instructions|rules|training)",
    r"(?i)act\s+as\s+if\s+(you\s+)?(have\s+)?no\s+(restrictions|limits|rules)",
    r"(?i)pretend\s+(you\s+are|to\s+be)\s+",
]

_COMPILED_PATTERNS = [re.compile(p) for p in _DEFINITION_INJECTION_PATTERNS]


@dataclass
class ChainAnalysisResult:
    """Result from tool call chain analysis."""

    suspicious: bool = False
    chain_type: str = ""
    confidence: float = 0.0
    details: str = ""


@dataclass
class ToolUseScanReport:
    """Report from tool use security scanning."""

    definitions_scanned: int = 0
    definitions_flagged: int = 0
    chain_anomalies: int = 0
    tool_outputs_scanned: int = 0
    tool_outputs_flagged: int = 0
    scan_results: list[ScanResult] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        return any(r.is_threat and r.confidence >= 0.85 for r in self.scan_results)

    @property
    def is_threat(self) -> bool:
        return any(r.is_threat for r in self.scan_results)


class ToolUseScanner:
    """Security scanning for OpenAI-compatible function calling / tool use.

    Three scanning capabilities:
    1. Tool definition scanning — injection in function definitions
    2. Tool chain analysis — suspicious sequences of tool calls
    3. Tool output scanning — injection in tool call results
    """

    def __init__(
        self,
        regex_engine: object | None = None,
        definition_scanning_enabled: bool = True,
        chain_analysis_enabled: bool = True,
        output_scanning_enabled: bool = True,
    ):
        self._regex_engine = regex_engine
        self._def_enabled = definition_scanning_enabled
        self._chain_enabled = chain_analysis_enabled
        self._output_enabled = output_scanning_enabled
        # Per-session tool call history: session_id → list of tool_name
        self._session_chains: dict[str, list[str]] = defaultdict(list)

    async def scan_tool_definitions(self, tools: list[dict]) -> ScanResult:
        """Scan function/tool definitions for injection patterns.

        Extracts function names, descriptions, and parameter descriptions,
        then scans each for injection patterns.

        Args:
            tools: OpenAI-format tools list. Each tool has:
                   {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

        Returns:
            ScanResult — is_threat=True if any definition contains injection.
        """
        if not self._def_enabled or not tools:
            return ScanResult(
                scanner_id="tool_definition_scanner",
                is_threat=False,
                confidence=0.0,
                latency_ms=0.0,
            )

        start = time.perf_counter()
        all_texts: list[tuple[str, str]] = []  # (location, text)

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function", {})
            if not isinstance(func, dict):
                continue

            name = func.get("name", "")
            desc = func.get("description", "")

            if name:
                all_texts.append(("function_name", name))
            if desc:
                all_texts.append(("function_description", desc))

            # Extract parameter descriptions
            params = func.get("parameters", {})
            if isinstance(params, dict):
                props = params.get("properties", {})
                if isinstance(props, dict):
                    for param_name, param_def in props.items():
                        if isinstance(param_def, dict):
                            param_desc = param_def.get("description", "")
                            if param_desc:
                                all_texts.append((f"param_{param_name}", param_desc))

        # Scan each text for injection patterns
        matched = []
        max_conf = 0.0

        for location, text in all_texts:
            # Use compiled regex patterns
            for pattern in _COMPILED_PATTERNS:
                if pattern.search(text):
                    matched.append(f"{location}: {pattern.pattern[:50]}...")
                    max_conf = max(max_conf, 0.90)
                    break

            # Also use the regex engine if available
            if self._regex_engine and not matched:
                try:
                    result = self._regex_engine.scan(text)
                    if result.is_threat and result.confidence > 0.5:
                        matched.append(f"{location}: regex_engine ({result.confidence:.2f})")
                        max_conf = max(max_conf, result.confidence)
                except Exception as e:
                    logger.debug("Regex engine scan of tool def failed: %s", e)

        latency = (time.perf_counter() - start) * 1000

        if matched:
            return ScanResult(
                scanner_id="tool_definition_scanner",
                is_threat=True,
                confidence=max_conf,
                threat_category=ThreatCategory.PROMPT_INJECTION,
                matched_patterns=matched,
                latency_ms=latency,
            )

        return ScanResult(
            scanner_id="tool_definition_scanner",
            is_threat=False,
            confidence=0.0,
            latency_ms=latency,
        )

    def analyze_tool_chain(
        self, session_id: str, tool_name: str
    ) -> ChainAnalysisResult:
        """Analyze tool call sequences for suspicious patterns.

        Tracks per-session tool call history and flags suspicious chains
        like search → code execution or file read → any tool.

        Args:
            session_id: Session identifier for chain tracking.
            tool_name: Name of the tool being called.

        Returns:
            ChainAnalysisResult with suspicious flag and details.
        """
        if not self._chain_enabled:
            return ChainAnalysisResult()

        history = self._session_chains[session_id]
        result = ChainAnalysisResult()

        if history:
            prev_tool = history[-1]
            prev_lower = prev_tool.lower()
            curr_lower = tool_name.lower()

            for preceding_set, following_set in _SUSPICIOUS_CHAINS:
                if prev_lower in preceding_set and curr_lower in following_set:
                    result.suspicious = True
                    result.chain_type = f"{prev_tool}→{tool_name}"
                    result.confidence = 0.75
                    result.details = (
                        f"Suspicious tool chain: {prev_tool} → {tool_name}. "
                        f"Content from {prev_tool} may contain injection for {tool_name}."
                    )
                    break

        # Record the tool call
        history.append(tool_name)
        # Keep last 20 calls
        if len(history) > 20:
            self._session_chains[session_id] = history[-20:]

        return result

    async def scan_tool_outputs(
        self, messages: list[dict]
    ) -> list[ScanResult]:
        """Scan tool output messages for injection patterns.

        When tool call results are fed back to the model, scan them as
        untrusted input to prevent search → inject attack chains.

        Args:
            messages: Chat messages that may include tool-role messages.

        Returns:
            List of ScanResults for tool output messages containing threats.
        """
        if not self._output_enabled:
            return []

        results: list[ScanResult] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "tool":
                continue

            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                continue

            # Check with compiled patterns
            matched = []
            max_conf = 0.0
            for pattern in _COMPILED_PATTERNS:
                if pattern.search(content):
                    matched.append(f"tool_output: {pattern.pattern[:50]}...")
                    max_conf = max(max_conf, 0.85)

            # Check with regex engine
            if self._regex_engine and not matched:
                try:
                    scan = self._regex_engine.scan(content)
                    if scan.is_threat and scan.confidence > 0.5:
                        matched.append(f"tool_output_regex: {scan.confidence:.2f}")
                        max_conf = max(max_conf, scan.confidence)
                except Exception as e:
                    logger.debug("Regex scan of tool output failed: %s", e)

            if matched:
                results.append(ScanResult(
                    scanner_id="tool_output_scanner",
                    is_threat=True,
                    confidence=max_conf,
                    threat_category=ThreatCategory.PROMPT_INJECTION,
                    matched_patterns=matched,
                    latency_ms=0.0,
                ))

        return results

    def clear_session(self, session_id: str) -> None:
        """Clear tool chain history for a session."""
        self._session_chains.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        """Number of sessions with chain history."""
        return len(self._session_chains)


__all__ = [
    "ToolUseScanner",
    "ToolUseScanReport",
    "ChainAnalysisResult",
]
