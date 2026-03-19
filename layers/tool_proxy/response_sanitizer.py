"""
Tool Response Sanitizer (TRS) — Extension 4.4

Processes all tool outputs before they reach the agent. Scans for indirect
prompt injection, enforces size limits, detects schema anomalies, and wraps
responses in explicit untrusted content markers.

Tool responses — file contents, web page data, API responses, database results
— are the primary vector for indirect prompt injection in agentic systems.

ASSUMED-BREACH POSTURE: Every tool response is treated as hostile content from
an untrusted source. A compromised tool server, poisoned database, or malicious
web page can embed injection payloads in tool output. The TRS applies L2
scanning to catch these before the agent processes the output.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Schema anomaly heuristics
_HTML_RE = re.compile(r"<(?:html|script|style|body|head|div|iframe)\b", re.IGNORECASE)
_CODE_MARKERS_RE = re.compile(
    r"(?:function\s+\w+\s*\(|import\s+\w+|class\s+\w+|def\s+\w+\s*\(|"
    r"var\s+\w+\s*=|const\s+\w+\s*=|let\s+\w+\s*=)",
)


@dataclass
class ToolResponseScan:
    tool_name: str
    original_size_bytes: int
    sanitized_size_bytes: int
    injection_detected: bool
    injection_score: float
    schema_anomaly: bool
    schema_anomaly_details: str | None
    truncated: bool
    blocked: bool
    scan_latency_ms: float


class ToolResponseSanitizer:
    """Sanitizes tool responses before they reach the agent.

    Thread-safe. Reuses L2's regex engine for injection scanning.
    """

    def __init__(
        self,
        regex_engine: Any = None,
        default_max_output_size: int = 102400,  # 100KB
        block_threshold: float = 0.85,
    ):
        self._regex_engine = regex_engine
        self._default_max_output_size = default_max_output_size
        self._block_threshold = block_threshold

    async def sanitize(
        self,
        tool_name: str,
        response_data: Any,
        max_output_size: int | None = None,
        expected_content_type: str | None = None,
    ) -> tuple[Any, ToolResponseScan]:
        """Sanitize a tool response.

        Returns (sanitized_data, scan_report).
        """
        start = time.perf_counter()
        max_size = max_output_size or self._default_max_output_size

        # Convert response to text for scanning
        text = self._to_text(response_data)
        original_size = len(text.encode("utf-8", errors="replace"))

        injection_detected = False
        injection_score = 0.0
        schema_anomaly = False
        schema_anomaly_details: str | None = None
        truncated = False
        blocked = False

        sanitized = response_data

        # 1. Size enforcement
        if original_size > max_size:
            text = text[: max_size]
            truncated = True
            sanitized = text + f"\n[TOOL_OUTPUT_TRUNCATED: exceeded {max_size} bytes]"

        # 2. Injection scanning via L2 regex engine
        if self._regex_engine and text.strip():
            try:
                scan_result = await self._regex_engine.scan(text)
                if scan_result.is_threat:
                    injection_detected = True
                    injection_score = scan_result.confidence
            except Exception:
                logger.debug("TRS regex scan failed", exc_info=True)

        # 3. Schema anomaly detection
        if expected_content_type and text.strip():
            anomaly = self._check_schema_anomaly(text, expected_content_type)
            if anomaly:
                schema_anomaly = True
                schema_anomaly_details = anomaly

        # 4. Block or wrap
        if injection_detected and injection_score >= self._block_threshold:
            blocked = True
            sanitized = (
                f"[TOOL_OUTPUT_BLOCKED: security policy violation "
                f"detected in tool response from '{tool_name}']"
            )
        elif truncated and isinstance(sanitized, str):
            pass  # already truncated above
        else:
            # Wrap in content delimiters
            sanitized = self._delimit(tool_name, sanitized)

        sanitized_text = self._to_text(sanitized)
        sanitized_size = len(sanitized_text.encode("utf-8", errors="replace"))
        elapsed = (time.perf_counter() - start) * 1000

        return sanitized, ToolResponseScan(
            tool_name=tool_name,
            original_size_bytes=original_size,
            sanitized_size_bytes=sanitized_size,
            injection_detected=injection_detected,
            injection_score=injection_score,
            schema_anomaly=schema_anomaly,
            schema_anomaly_details=schema_anomaly_details,
            truncated=truncated,
            blocked=blocked,
            scan_latency_ms=elapsed,
        )

    @staticmethod
    def _delimit(tool_name: str, data: Any) -> Any:
        """Wrap tool output in explicit untrusted content markers."""
        if isinstance(data, str):
            return (
                f"[TOOL_OUTPUT_BEGIN: {tool_name}]\n"
                f"{data}\n"
                f"[TOOL_OUTPUT_END: {tool_name}]"
            )
        if isinstance(data, dict):
            # For dict responses, wrap the string representation
            return {
                "_aegis_tool_output": tool_name,
                "_aegis_delimited": True,
                "data": data,
            }
        return data

    @staticmethod
    def _to_text(data: Any) -> str:
        """Convert any response data to scannable text."""
        if isinstance(data, str):
            return data
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8", errors="replace")
            except Exception:
                return ""
        if isinstance(data, dict):
            import json
            try:
                return json.dumps(data, default=str)
            except Exception:
                return str(data)
        return str(data) if data is not None else ""

    @staticmethod
    def _check_schema_anomaly(text: str, expected_type: str) -> str | None:
        """Check if response content matches expected type."""
        expected_lower = expected_type.lower()

        if expected_lower in ("json", "application/json"):
            if _HTML_RE.search(text):
                return "Expected JSON but found HTML content"
        elif expected_lower in ("csv", "text/csv", "data"):
            if _HTML_RE.search(text):
                return "Expected data but found HTML content"
            if _CODE_MARKERS_RE.search(text):
                return "Expected data but found code content"
        elif expected_lower in ("text", "text/plain"):
            if _HTML_RE.search(text):
                return "Expected plain text but found HTML/script content"

        return None
