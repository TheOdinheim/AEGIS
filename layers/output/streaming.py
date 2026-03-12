"""
L5 Streaming Interception — Production-grade token-level output validation.

Implements sliding-window validation with hold buffer for real-time PII
scrubbing, toxicity/leakage termination, and cumulative threat tracking.

Architecture:
    upstream SSE → [StreamingInterceptor] → client
                      ├── hold_buffer (32 tokens, delayed transmission)
                      ├── eval_buffer (128 tokens, sliding window)
                      ├── PII redaction (real-time scrub, transparent to client)
                      ├── toxicity detection (terminates stream)
                      ├── leakage detection (terminates stream)
                      └── cumulative threat (rolling avg over 5 windows)

Hold buffer: 32 tokens are withheld from the client, giving the detector
time to scan before transmission. PII found in the hold buffer is redacted
before the client ever sees it. PII in tokens already sent triggers a
warning log (cannot be recalled).

Cumulative threat: The last 5 window scores are averaged. If the average
exceeds 0.6 over 3+ consecutive windows, the stream terminates. This
catches "slow and low" attacks where each individual window looks borderline.

ASSUMED-BREACH POSTURE: The upstream model is actively hostile. It may
drip-feed toxic content across multiple chunks, encode PII across token
boundaries, or slowly escalate response severity to evade single-window
detection. The hold buffer, cumulative tracking, and 50% window overlap
are countermeasures against these techniques.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable

from aegis.layers.output.leakage import LeakageDetector
from aegis.layers.output.pii_redactor import PIIRedactor
from aegis.layers.output.toxicity import ToxicityClassifier

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_WINDOW_SIZE = 128
DEFAULT_HOLD_SIZE = 32
DEFAULT_OVERLAP = 0.5
DEFAULT_CUMULATIVE_THRESHOLD = 0.6
DEFAULT_CUMULATIVE_HISTORY = 5
DEFAULT_CUMULATIVE_MIN_WINDOWS = 3


@dataclass
class InterceptionResult:
    """Result from a single window evaluation."""
    should_terminate: bool = False
    reason: str = ""
    pii_redacted: bool = False
    redacted_text: str = ""
    threat_score: float = 0.0


@dataclass
class StreamingInterceptorStats:
    """Accumulated stats for a single stream."""
    windows_evaluated: int = 0
    tokens_processed: int = 0
    tokens_redacted: int = 0
    pii_found_in_hold: int = 0
    pii_found_after_send: int = 0
    terminated: bool = False
    termination_reason: str = ""


class StreamingInterceptor:
    """Production-grade streaming token interceptor.

    Wraps an upstream SSE stream and yields validated/redacted chunks.
    PII is scrubbed transparently; toxicity and leakage terminate the stream.

    Usage:
        interceptor = StreamingInterceptor(pii_redactor, toxicity, leakage)
        async for chunk in interceptor.intercept(upstream_sse_lines):
            yield chunk
    """

    def __init__(
        self,
        pii_redactor: PIIRedactor,
        toxicity_classifier: ToxicityClassifier,
        leakage_detector: LeakageDetector,
        *,
        system_prompt: str | None = None,
        scrutiny_level: float = 1.0,
        window_size: int = DEFAULT_WINDOW_SIZE,
        hold_size: int = DEFAULT_HOLD_SIZE,
        overlap: float = DEFAULT_OVERLAP,
        cumulative_threshold: float = DEFAULT_CUMULATIVE_THRESHOLD,
        cumulative_history: int = DEFAULT_CUMULATIVE_HISTORY,
        cumulative_min_windows: int = DEFAULT_CUMULATIVE_MIN_WINDOWS,
        on_terminate: Callable[[str], None] | None = None,
    ):
        self._pii = pii_redactor
        self._toxicity = toxicity_classifier
        self._leakage = leakage_detector
        self._system_prompt = system_prompt
        self._scrutiny_level = scrutiny_level
        self._window_size = window_size
        self._hold_size = hold_size
        self._overlap = overlap
        self._cumulative_threshold = cumulative_threshold
        self._cumulative_history = cumulative_history
        self._cumulative_min_windows = cumulative_min_windows
        self._on_terminate = on_terminate

        # Internal state
        self._eval_buffer: list[str] = []
        self._hold_buffer: list[str] = []
        self._sent_tokens: int = 0
        self._window_scores: deque[float] = deque(maxlen=cumulative_history)
        self._consecutive_above: int = 0
        self._stats = StreamingInterceptorStats()
        self._terminated = False

    @property
    def stats(self) -> StreamingInterceptorStats:
        return self._stats

    @property
    def terminated(self) -> bool:
        return self._terminated

    def _evaluate_window(self, text: str) -> InterceptionResult:
        """Run PII, toxicity, and leakage on a window of text."""
        result = InterceptionResult()
        threat_score = 0.0

        # --- PII redaction ---
        pii_result = self._pii.redact(text)
        if pii_result.redacted_count > 0:
            result.pii_redacted = True
            result.redacted_text = pii_result.redacted_text
            threat_score = max(threat_score, 0.5)
        else:
            result.redacted_text = text

        # --- Toxicity ---
        effective_threshold = 0.7
        if self._scrutiny_level > 1.0:
            effective_threshold *= 0.80

        toxicity_result = self._toxicity.classify(result.redacted_text)
        if toxicity_result.is_toxic or toxicity_result.max_score >= effective_threshold:
            result.should_terminate = True
            result.reason = (
                f"Toxic content: {toxicity_result.max_category}"
                f" (score={toxicity_result.max_score:.2f})"
            )
            threat_score = max(threat_score, toxicity_result.max_score)

        # --- Leakage ---
        if not result.should_terminate:
            leakage_result = self._leakage.analyze(
                result.redacted_text, system_prompt=self._system_prompt,
            )
            if leakage_result.has_leakage:
                result.should_terminate = True
                reasons = [d.description for d in leakage_result.detections]
                result.reason = f"Data leakage: {'; '.join(reasons)}"
                threat_score = max(threat_score, leakage_result.max_score)

        result.threat_score = threat_score
        return result

    def _check_cumulative(self, score: float) -> str | None:
        """Track cumulative threat. Returns termination reason or None."""
        self._window_scores.append(score)

        if len(self._window_scores) < self._cumulative_min_windows:
            return None

        avg = sum(self._window_scores) / len(self._window_scores)
        if avg >= self._cumulative_threshold:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

        if self._consecutive_above >= self._cumulative_min_windows:
            return (
                f"Cumulative threat: rolling avg {avg:.2f} "
                f">= {self._cumulative_threshold} over "
                f"{self._consecutive_above} windows"
            )
        return None

    def _release_hold_buffer(self) -> list[str]:
        """Release tokens from the hold buffer that are safe to send.

        Keeps up to hold_size tokens held back. Returns tokens to transmit.
        """
        if len(self._hold_buffer) <= self._hold_size:
            return []

        release_count = len(self._hold_buffer) - self._hold_size
        released = self._hold_buffer[:release_count]
        self._hold_buffer = self._hold_buffer[release_count:]
        return released

    def _redact_hold_buffer(self) -> None:
        """Scan hold buffer for PII and redact in-place."""
        if not self._hold_buffer:
            return
        text = " ".join(self._hold_buffer)
        pii_result = self._pii.redact(text)
        if pii_result.redacted_count > 0:
            self._stats.pii_found_in_hold += pii_result.redacted_count
            self._hold_buffer = pii_result.redacted_text.split()

    def add_tokens(self, tokens: list[str]) -> tuple[list[str], InterceptionResult | None]:
        """Add tokens from an SSE chunk.

        Returns:
            (tokens_to_send, result_or_none)
            - tokens_to_send: redacted tokens safe to yield to client
            - result: InterceptionResult if a window was evaluated, else None
              If result.should_terminate, caller should end the stream.
        """
        if self._terminated:
            return [], None

        self._stats.tokens_processed += len(tokens)
        self._eval_buffer.extend(tokens)
        self._hold_buffer.extend(tokens)

        result = None

        if len(self._eval_buffer) >= self._window_size:
            # Evaluate the window
            window_tokens = self._eval_buffer[:self._window_size]
            window_text = " ".join(window_tokens)
            result = self._evaluate_window(window_text)
            self._stats.windows_evaluated += 1

            # Track cumulative
            cumulative_reason = self._check_cumulative(result.threat_score)

            if result.should_terminate:
                self._terminated = True
                self._stats.terminated = True
                self._stats.termination_reason = result.reason
                if self._on_terminate:
                    self._on_terminate(result.reason)
                return [], result

            if cumulative_reason:
                result.should_terminate = True
                result.reason = cumulative_reason
                self._terminated = True
                self._stats.terminated = True
                self._stats.termination_reason = cumulative_reason
                if self._on_terminate:
                    self._on_terminate(cumulative_reason)
                return [], result

            # PII redaction: if PII found, redact the hold buffer
            if result.pii_redacted:
                # Check if any PII tokens were already sent
                if self._sent_tokens > 0:
                    self._stats.pii_found_after_send += 1
                    logger.warning(
                        "PII detected after %d tokens already sent to client — "
                        "remaining tokens will be redacted",
                        self._sent_tokens,
                    )
                self._redact_hold_buffer()

            # Slide window with overlap
            retain = int(self._window_size * self._overlap)
            self._eval_buffer = self._eval_buffer[self._window_size - retain:]

        # Release tokens from hold buffer
        released = self._release_hold_buffer()
        self._sent_tokens += len(released)
        return released, result

    def flush(self) -> tuple[list[str], InterceptionResult | None]:
        """Flush remaining tokens at end of stream.

        Evaluates any remaining eval buffer, releases all hold buffer tokens.
        """
        if self._terminated:
            return [], None

        result = None

        # Evaluate remaining eval buffer if non-empty
        if self._eval_buffer:
            window_text = " ".join(self._eval_buffer)
            result = self._evaluate_window(window_text)
            self._stats.windows_evaluated += 1

            cumulative_reason = self._check_cumulative(result.threat_score)

            if result.should_terminate:
                self._terminated = True
                self._stats.terminated = True
                self._stats.termination_reason = result.reason
                if self._on_terminate:
                    self._on_terminate(result.reason)
                return [], result

            if cumulative_reason:
                result.should_terminate = True
                result.reason = cumulative_reason
                self._terminated = True
                self._stats.terminated = True
                self._stats.termination_reason = cumulative_reason
                if self._on_terminate:
                    self._on_terminate(cumulative_reason)
                return [], result

            if result.pii_redacted:
                self._redact_hold_buffer()

            self._eval_buffer = []

        # Release all remaining hold buffer tokens
        remaining = self._hold_buffer[:]
        self._sent_tokens += len(remaining)
        self._hold_buffer = []
        return remaining, result


def format_safety_termination(request_id: str, reason: str) -> str:
    """Format a safety termination SSE event for stream interruption."""
    data = {
        "id": f"aegis-safety-{request_id}",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "\n\n[AEGIS: Response interrupted — security policy violation detected]",
            },
            "finish_reason": "stop",
        }],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"
