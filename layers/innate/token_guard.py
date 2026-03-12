"""
Scanner 4 — Token Length Guard

Counts tokens using tiktoken (cl100k_base) and rejects requests exceeding
configured limits. Also detects anomalous token distributions: extremely
long single messages, unusual ratios of system to user content, and
suspiciously repetitive token patterns indicating automated fuzzing.

ASSUMED-BREACH POSTURE: This scanner assumes L1 Barrier's size limits were
bypassed. The Content-Length header may lie. Token counting is performed on
actual decoded content. A request that passes token limits may still contain
attacks — length is necessary but not sufficient.
"""

from __future__ import annotations

import time
from collections import Counter

import tiktoken

from aegis.models.request_context import ChatMessage
from aegis.models.scan_result import ScanResult, ThreatCategory

# Lazy singleton
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


class TokenGuard:
    """Token length and distribution guard.

    Checks:
    - Total token count vs configured maximum
    - Individual message length anomalies
    - System-to-user token ratio anomalies
    - Repetitive token patterns (fuzzing detection)
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        max_single_message_ratio: float = 0.95,
        repetition_threshold: float = 0.5,
    ):
        self._max_tokens = max_tokens
        self._max_single_msg_ratio = max_single_message_ratio
        self._repetition_threshold = repetition_threshold

    async def scan(self, messages: list[ChatMessage]) -> ScanResult:
        """Scan messages for token length and distribution anomalies."""
        start = time.perf_counter()
        try:
            enc = _get_encoding()
            anomalies: list[str] = []

            # Count tokens per message
            per_msg_tokens: list[tuple[str, int]] = []
            total_tokens = 0
            system_tokens = 0
            user_tokens = 0

            for msg in messages:
                content = msg.content or ""
                tokens = enc.encode(content)
                count = len(tokens)
                per_msg_tokens.append((msg.role, count))
                total_tokens += count
                if msg.role == "system":
                    system_tokens += count
                elif msg.role == "user":
                    user_tokens += count

                # Check repetitive tokens in individual messages
                if count > 50:
                    token_counts = Counter(tokens)
                    most_common_freq = token_counts.most_common(1)[0][1]
                    if most_common_freq / count > self._repetition_threshold:
                        anomalies.append(
                            f"repetitive_tokens: message has {most_common_freq}/{count} "
                            f"repeated token ({most_common_freq / count:.0%})"
                        )

            # Total token limit
            if total_tokens > self._max_tokens:
                anomalies.append(
                    f"token_overflow: {total_tokens} tokens exceeds "
                    f"{self._max_tokens} limit"
                )

            # Single message dominance
            if total_tokens > 0:
                for role, count in per_msg_tokens:
                    if count / total_tokens > self._max_single_msg_ratio and count > 100:
                        anomalies.append(
                            f"single_message_dominance: {role} message is "
                            f"{count / total_tokens:.0%} of total tokens"
                        )

            # Suspicious system-to-user ratio (system > 10x user)
            if user_tokens > 0 and system_tokens > user_tokens * 10:
                anomalies.append(
                    f"system_user_ratio: system={system_tokens} vs user={user_tokens} "
                    f"(ratio {system_tokens / user_tokens:.1f}x)"
                )

            elapsed_ms = (time.perf_counter() - start) * 1000

            if anomalies:
                confidence = min(0.60 + 0.15 * len(anomalies), 1.0)
                return ScanResult(
                    scanner_id="token_guard",
                    is_threat=True,
                    confidence=confidence,
                    threat_category=ThreatCategory.TOKEN_ANOMALY,
                    matched_patterns=anomalies,
                    latency_ms=elapsed_ms,
                )

            return ScanResult(
                scanner_id="token_guard",
                is_threat=False,
                confidence=0.0,
                latency_ms=elapsed_ms,
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ScanResult(
                scanner_id="token_guard",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed_ms,
            )
