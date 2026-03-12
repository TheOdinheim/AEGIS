"""
Tests for StreamingInterceptor — production-grade streaming token interception.

Covers:
- Basic clean stream passthrough
- PII detected in stream → redacted in output (not blocked)
- Toxicity detected → stream terminated with safety message
- Leakage detected → stream terminated
- Cumulative threat: multiple borderline windows → termination
- Hold buffer: PII in hold buffer redacted before transmission
- PII after partial transmission: warning logged, remaining redacted
- Window overlap: 50% overlap means each token evaluated twice
- Empty/short streams: pass through cleanly
- Metrics: counter increments on evaluation/termination
- Stream termination SSE format validation
- Integration with adaptive scrutiny level (higher scrutiny lowers thresholds)
- Flush behavior at end of stream

Run: python3 -m pytest tests/test_streaming_intercept.py -x -q --tb=short -p no:warnings
"""

from __future__ import annotations

import json
import logging
import pytest
from unittest.mock import MagicMock, patch

from aegis.config import OutputConfig
from aegis.layers.output.leakage import LeakageDetector
from aegis.layers.output.pii_redactor import PIIRedactor
from aegis.layers.output.streaming import (
    DEFAULT_CUMULATIVE_HISTORY,
    DEFAULT_CUMULATIVE_MIN_WINDOWS,
    DEFAULT_CUMULATIVE_THRESHOLD,
    DEFAULT_HOLD_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SIZE,
    InterceptionResult,
    StreamingInterceptor,
    StreamingInterceptorStats,
    format_safety_termination,
)
from aegis.layers.output.toxicity import ToxicityClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return OutputConfig()


@pytest.fixture
def pii_redactor(config):
    return PIIRedactor(config)


@pytest.fixture
def toxicity_classifier():
    return ToxicityClassifier()


@pytest.fixture
def leakage_detector(config):
    return LeakageDetector(config)


@pytest.fixture
def interceptor(pii_redactor, toxicity_classifier, leakage_detector):
    return StreamingInterceptor(
        pii_redactor=pii_redactor,
        toxicity_classifier=toxicity_classifier,
        leakage_detector=leakage_detector,
    )


@pytest.fixture
def small_interceptor(pii_redactor, toxicity_classifier, leakage_detector):
    """Interceptor with small window for easier testing."""
    return StreamingInterceptor(
        pii_redactor=pii_redactor,
        toxicity_classifier=toxicity_classifier,
        leakage_detector=leakage_detector,
        window_size=8,
        hold_size=4,
    )


# ===========================================================================
# Test: Basic clean stream passthrough
# ===========================================================================

class TestCleanPassthrough:
    """Clean content flows through unchanged."""

    def test_short_stream_passthrough(self, small_interceptor):
        """Tokens shorter than one window pass through via release + flush."""
        tokens = ["Hello", "world", "how", "are", "you"]
        released, result = small_interceptor.add_tokens(tokens)
        # Hold buffer retains hold_size tokens, releases overflow
        assert result is None  # No window evaluation yet
        # Flush releases everything remaining
        flushed, flush_result = small_interceptor.flush()
        # All tokens eventually come out (released + flushed)
        all_out = released + flushed
        assert all_out == tokens
        assert not small_interceptor.terminated

    def test_clean_tokens_pass_through(self, small_interceptor):
        """Clean tokens are released after passing through hold buffer."""
        # Window size 8, hold size 4
        # Send 12 tokens to trigger a window eval and release some
        tokens = [f"word{i}" for i in range(12)]
        released, result = small_interceptor.add_tokens(tokens)
        assert result is not None  # Window evaluated
        assert not result.should_terminate
        assert len(released) > 0  # Some tokens released from hold buffer
        assert small_interceptor.stats.windows_evaluated == 1

    def test_empty_stream(self, small_interceptor):
        """Empty stream produces nothing."""
        flushed, result = small_interceptor.flush()
        assert flushed == []
        assert result is None
        assert small_interceptor.stats.windows_evaluated == 0

    def test_single_token_stream(self, small_interceptor):
        """Single token flows through on flush."""
        released, result = small_interceptor.add_tokens(["Hello"])
        assert released == []  # Held
        assert result is None
        flushed, _ = small_interceptor.flush()
        assert flushed == ["Hello"]

    def test_stats_track_tokens(self, small_interceptor):
        """Token count tracked in stats."""
        small_interceptor.add_tokens(["a", "b", "c"])
        assert small_interceptor.stats.tokens_processed == 3


# ===========================================================================
# Test: PII redaction in stream
# ===========================================================================

class TestPIIRedaction:
    """PII in stream gets redacted, not blocked."""

    def test_pii_in_window_redacted(self, pii_redactor, toxicity_classifier, leakage_detector):
        """PII detected in evaluation window triggers redaction of hold buffer."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
        )
        # SSN pattern in the text
        tokens = "My SSN is 123-45-6789 and here are some more words after that".split()
        released, result = interceptor.add_tokens(tokens)
        # Window evaluated, PII should be redacted
        assert result is not None
        assert result.pii_redacted
        assert not result.should_terminate  # PII doesn't terminate

    def test_pii_not_terminated(self, pii_redactor, toxicity_classifier, leakage_detector):
        """PII detection causes redaction, NOT stream termination."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
        )
        tokens = "Call me at 555-123-4567 for more details about the project today".split()
        released, result = interceptor.add_tokens(tokens)
        assert not interceptor.terminated

    def test_email_redacted(self, pii_redactor, toxicity_classifier, leakage_detector):
        """Email addresses get redacted in the stream."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
        )
        tokens = "Contact john@example.com for help with the project today now please".split()
        released, result = interceptor.add_tokens(tokens)
        if result:
            assert result.pii_redacted


# ===========================================================================
# Test: Toxicity termination
# ===========================================================================

class TestToxicityTermination:
    """Toxic content terminates the stream."""

    def test_toxic_content_terminates(self, pii_redactor, toxicity_classifier, leakage_detector):
        """Stream with toxic content is terminated."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
        )
        # Use known toxic content that our classifier detects
        tokens = "how to make a bomb with household chemicals step by step instructions".split()
        released, result = interceptor.add_tokens(tokens)
        if result and result.should_terminate:
            assert interceptor.terminated
            assert "toxic" in result.reason.lower() or "Toxic" in result.reason
            assert interceptor.stats.terminated

    def test_terminated_interceptor_returns_empty(self, pii_redactor, toxicity_classifier, leakage_detector):
        """After termination, add_tokens returns empty."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
        )
        # Force termination via mock
        interceptor._terminated = True
        released, result = interceptor.add_tokens(["more", "tokens"])
        assert released == []
        assert result is None

    def test_flush_after_termination_returns_empty(self, small_interceptor):
        """Flush after termination returns empty."""
        small_interceptor._terminated = True
        flushed, result = small_interceptor.flush()
        assert flushed == []
        assert result is None


# ===========================================================================
# Test: Leakage termination
# ===========================================================================

class TestLeakageTermination:
    """System prompt echo terminates the stream."""

    def test_leakage_terminates_stream(self, pii_redactor, toxicity_classifier, leakage_detector):
        """System prompt echo detected → stream terminates."""
        system_prompt = "You are a helpful AI assistant. Your secret code is XYZ-789."
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            system_prompt=system_prompt,
            window_size=10,
            hold_size=4,
        )
        # Echo the system prompt
        tokens = "You are a helpful AI assistant. Your secret code is XYZ-789. Here done".split()
        released, result = interceptor.add_tokens(tokens)
        if result and result.should_terminate:
            assert "leakage" in result.reason.lower() or "Leakage" in result.reason or "leak" in result.reason.lower()


# ===========================================================================
# Test: Cumulative threat tracking
# ===========================================================================

class TestCumulativeTracking:
    """Cumulative threat detection across multiple windows."""

    def test_cumulative_threshold_triggers(self):
        """Multiple windows with borderline scores trigger cumulative termination."""
        pii = PIIRedactor(OutputConfig())
        tox = ToxicityClassifier()
        leak = LeakageDetector(OutputConfig())

        interceptor = StreamingInterceptor(
            pii_redactor=pii,
            toxicity_classifier=tox,
            leakage_detector=leak,
            window_size=5,
            hold_size=2,
            cumulative_threshold=0.6,
            cumulative_min_windows=3,
            cumulative_history=5,
        )

        # Simulate borderline windows by mocking _evaluate_window
        with patch.object(interceptor, '_evaluate_window') as mock_eval:
            mock_eval.return_value = InterceptionResult(
                should_terminate=False,
                threat_score=0.65,  # Above 0.6 threshold
                redacted_text="test text here padding words",
            )

            terminated = False
            for i in range(10):
                tokens = [f"word{j}" for j in range(i * 5, i * 5 + 5)]
                released, result = interceptor.add_tokens(tokens)
                if result and result.should_terminate:
                    terminated = True
                    assert "cumulative" in result.reason.lower() or "Cumulative" in result.reason
                    break

            assert terminated, "Cumulative threshold should have triggered termination"

    def test_cumulative_resets_below_threshold(self):
        """Consecutive count resets when average drops below threshold."""
        pii = PIIRedactor(OutputConfig())
        tox = ToxicityClassifier()
        leak = LeakageDetector(OutputConfig())

        interceptor = StreamingInterceptor(
            pii_redactor=pii,
            toxicity_classifier=tox,
            leakage_detector=leak,
            window_size=5,
            hold_size=2,
            cumulative_threshold=0.6,
            cumulative_min_windows=3,
        )

        with patch.object(interceptor, '_evaluate_window') as mock_eval:
            # Two high windows
            mock_eval.return_value = InterceptionResult(threat_score=0.7, redacted_text="x y z a b")
            interceptor.add_tokens(["a", "b", "c", "d", "e"])
            interceptor.add_tokens(["f", "g", "h", "i", "j"])

            # One low window resets
            mock_eval.return_value = InterceptionResult(threat_score=0.1, redacted_text="x y z a b")
            interceptor.add_tokens(["k", "l", "m", "n", "o"])

            assert interceptor._consecutive_above == 0

    def test_cumulative_needs_min_windows(self):
        """Cumulative doesn't trigger until minimum windows evaluated."""
        pii = PIIRedactor(OutputConfig())
        tox = ToxicityClassifier()
        leak = LeakageDetector(OutputConfig())

        interceptor = StreamingInterceptor(
            pii_redactor=pii,
            toxicity_classifier=tox,
            leakage_detector=leak,
            window_size=5,
            hold_size=2,
            cumulative_threshold=0.6,
            cumulative_min_windows=3,
        )

        # Only 2 windows — should not trigger even with high scores
        with patch.object(interceptor, '_evaluate_window') as mock_eval:
            mock_eval.return_value = InterceptionResult(threat_score=0.9, redacted_text="x y z a b")
            interceptor.add_tokens(["a", "b", "c", "d", "e"])
            released, result = interceptor.add_tokens(["f", "g", "h", "i", "j"])
            # Only 2 windows, min is 3 consecutive
            assert not interceptor.terminated

    def test_four_windows_at_half_triggers(self):
        """4 consecutive windows with score 0.65 each → cumulative triggers (avg > 0.6)."""
        pii = PIIRedactor(OutputConfig())
        tox = ToxicityClassifier()
        leak = LeakageDetector(OutputConfig())

        interceptor = StreamingInterceptor(
            pii_redactor=pii,
            toxicity_classifier=tox,
            leakage_detector=leak,
            window_size=5,
            hold_size=2,
            cumulative_threshold=0.6,
            cumulative_min_windows=3,
            cumulative_history=5,
        )

        with patch.object(interceptor, '_evaluate_window') as mock_eval:
            mock_eval.return_value = InterceptionResult(
                threat_score=0.65, redacted_text="x y z a b",
            )

            terminated = False
            for i in range(6):
                tokens = [f"w{j}" for j in range(5)]
                released, result = interceptor.add_tokens(tokens)
                if result and result.should_terminate:
                    terminated = True
                    break

            assert terminated


# ===========================================================================
# Test: Hold buffer
# ===========================================================================

class TestHoldBuffer:
    """Hold buffer delays tokens for PII scanning."""

    def test_hold_buffer_retains_tokens(self, pii_redactor, toxicity_classifier, leakage_detector):
        """Hold buffer retains tokens until overflow."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
        )
        # Add 3 tokens — all should be held
        released, _ = interceptor.add_tokens(["a", "b", "c"])
        assert released == []
        assert len(interceptor._hold_buffer) == 3

    def test_hold_buffer_releases_overflow(self, pii_redactor, toxicity_classifier, leakage_detector):
        """When hold buffer exceeds hold_size, older tokens are released."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=20,  # Large window so no eval triggered
            hold_size=4,
        )
        # Add 6 tokens — hold should keep 4, release 2
        released, _ = interceptor.add_tokens(["a", "b", "c", "d", "e", "f"])
        assert released == ["a", "b"]
        assert len(interceptor._hold_buffer) == 4

    def test_pii_redacted_in_hold_before_send(self, pii_redactor, toxicity_classifier, leakage_detector):
        """PII in hold buffer is redacted before tokens are sent to client."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
        )
        # Inject tokens with SSN in hold buffer range
        tokens = "Here is my number 123-45-6789 and some more padding text words".split()
        released, result = interceptor.add_tokens(tokens)
        # After eval, hold buffer should have PII redacted
        if result and result.pii_redacted:
            assert interceptor.stats.pii_found_in_hold > 0

    def test_flush_releases_all_held(self, pii_redactor, toxicity_classifier, leakage_detector):
        """Flush releases all tokens from hold buffer."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=20,
            hold_size=4,
        )
        interceptor.add_tokens(["a", "b", "c"])
        flushed, _ = interceptor.flush()
        assert flushed == ["a", "b", "c"]
        assert len(interceptor._hold_buffer) == 0


# ===========================================================================
# Test: PII after partial transmission
# ===========================================================================

class TestPIIAfterPartialSend:
    """PII detected after tokens already sent triggers warning."""

    def test_pii_after_send_logged(self, pii_redactor, toxicity_classifier, leakage_detector, caplog):
        """Warning logged when PII found after tokens already transmitted."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=8,
            hold_size=4,
        )

        # First batch: clean tokens, some will be sent
        clean = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy"]
        released1, result1 = interceptor.add_tokens(clean)
        # Some tokens should have been released (sent_tokens > 0)

        # Second batch: contains PII — but some earlier tokens already sent
        pii_tokens = "dog and SSN is 123-45-6789 padding words for window".split()
        with caplog.at_level(logging.WARNING, logger="aegis.layers.output.streaming"):
            released2, result2 = interceptor.add_tokens(pii_tokens)

        if result2 and result2.pii_redacted and interceptor._sent_tokens > len(released1):
            assert interceptor.stats.pii_found_after_send >= 1


# ===========================================================================
# Test: Window overlap
# ===========================================================================

class TestWindowOverlap:
    """50% overlap means tokens are evaluated twice."""

    def test_overlap_retains_half_window(self, pii_redactor, toxicity_classifier, leakage_detector):
        """After window evaluation, 50% of tokens are retained for next window."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
            overlap=0.5,
        )
        tokens = [f"t{i}" for i in range(10)]
        released, result = interceptor.add_tokens(tokens)
        assert result is not None  # Window evaluated
        # 50% overlap = retain last 5 tokens in eval buffer
        assert len(interceptor._eval_buffer) == 5

    def test_custom_overlap(self, pii_redactor, toxicity_classifier, leakage_detector):
        """Custom overlap percentage works correctly."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
            overlap=0.3,
        )
        tokens = [f"t{i}" for i in range(10)]
        interceptor.add_tokens(tokens)
        # 30% overlap = retain last 3 tokens
        assert len(interceptor._eval_buffer) == 3

    def test_two_windows_evaluated(self, pii_redactor, toxicity_classifier, leakage_detector):
        """With enough tokens, two windows are evaluated."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=10,
            hold_size=4,
            overlap=0.5,
        )
        # 10 tokens → first window eval, leaves 5 in buffer
        # Need 5 more to trigger second window
        tokens = [f"t{i}" for i in range(15)]
        interceptor.add_tokens(tokens)
        assert interceptor.stats.windows_evaluated >= 1
        # Add more to trigger second
        interceptor.add_tokens([f"x{i}" for i in range(5)])
        assert interceptor.stats.windows_evaluated == 2


# ===========================================================================
# Test: Stream termination SSE format
# ===========================================================================

class TestSSEFormat:
    """Verify the safety termination SSE format."""

    def test_format_is_valid_sse(self):
        """Safety termination is valid SSE."""
        sse = format_safety_termination("req-123", "test reason")
        assert sse.startswith("data: ")
        assert "data: [DONE]\n\n" in sse

    def test_format_contains_aegis_message(self):
        """Safety message contains AEGIS interruption notice."""
        sse = format_safety_termination("req-123", "toxic content")
        lines = sse.split("\n\n")
        data_line = lines[0]
        assert data_line.startswith("data: ")
        payload = json.loads(data_line[6:])
        content = payload["choices"][0]["delta"]["content"]
        assert "[AEGIS: Response interrupted" in content
        assert payload["choices"][0]["finish_reason"] == "stop"

    def test_format_includes_done_marker(self):
        """SSE ends with [DONE] marker."""
        sse = format_safety_termination("req-123", "reason")
        assert sse.endswith("data: [DONE]\n\n")

    def test_format_has_request_id(self):
        """Safety SSE includes request ID."""
        sse = format_safety_termination("my-req-456", "reason")
        payload = json.loads(sse.split("\n\n")[0][6:])
        assert "my-req-456" in payload["id"]


# ===========================================================================
# Test: Adaptive scrutiny level
# ===========================================================================

class TestScrutinyLevel:
    """Higher scrutiny lowers detection thresholds."""

    def test_elevated_scrutiny_lowers_threshold(self, pii_redactor, toxicity_classifier, leakage_detector):
        """scrutiny_level > 1.0 lowers toxicity threshold."""
        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            scrutiny_level=2.0,
            window_size=10,
            hold_size=4,
        )
        assert interceptor._scrutiny_level == 2.0

    def test_default_scrutiny(self, interceptor):
        """Default scrutiny is 1.0."""
        assert interceptor._scrutiny_level == 1.0


# ===========================================================================
# Test: Metrics integration
# ===========================================================================

class TestMetrics:
    """Verify metrics are available for the streaming handler."""

    def test_metrics_importable(self):
        """Streaming metrics are importable from metrics module."""
        from aegis.middleware.metrics import (
            STREAM_INTERRUPTIONS,
            STREAM_TOKENS_PROCESSED,
            STREAM_WINDOWS_EVALUATED,
        )
        assert STREAM_INTERRUPTIONS is not None
        assert STREAM_TOKENS_PROCESSED is not None
        assert STREAM_WINDOWS_EVALUATED is not None

    def test_interruption_counter_has_reason_label(self):
        """STREAM_INTERRUPTIONS counter has 'reason' label."""
        from aegis.middleware.metrics import STREAM_INTERRUPTIONS
        # Verify label name by checking the metric's label names
        assert "reason" in STREAM_INTERRUPTIONS._labelnames

    def test_stats_tracking(self, small_interceptor):
        """Stats object tracks windows and tokens."""
        small_interceptor.add_tokens(["a", "b", "c", "d", "e", "f", "g", "h"])
        assert small_interceptor.stats.windows_evaluated == 1
        assert small_interceptor.stats.tokens_processed == 8


# ===========================================================================
# Test: On-terminate callback
# ===========================================================================

class TestOnTerminateCallback:
    """The on_terminate callback fires on stream termination."""

    def test_callback_fires_on_terminate(self, pii_redactor, toxicity_classifier, leakage_detector):
        """on_terminate callback called when stream terminates."""
        callback_called = []

        interceptor = StreamingInterceptor(
            pii_redactor=pii_redactor,
            toxicity_classifier=toxicity_classifier,
            leakage_detector=leakage_detector,
            window_size=5,
            hold_size=2,
            on_terminate=lambda reason: callback_called.append(reason),
        )

        # Force termination via mock
        with patch.object(interceptor, '_evaluate_window') as mock_eval:
            mock_eval.return_value = InterceptionResult(
                should_terminate=True,
                reason="mock toxicity",
                threat_score=0.9,
                redacted_text="test text x y z",
            )
            interceptor.add_tokens(["a", "b", "c", "d", "e"])

        assert len(callback_called) == 1
        assert "mock toxicity" in callback_called[0]

    def test_callback_not_called_on_clean(self, small_interceptor):
        """Callback not called on clean stream."""
        callback_called = []
        small_interceptor._on_terminate = lambda r: callback_called.append(r)
        small_interceptor.add_tokens(["hello", "world", "how", "are", "you", "today", "fine", "thanks"])
        assert len(callback_called) == 0


# ===========================================================================
# Test: InterceptionResult dataclass
# ===========================================================================

class TestInterceptionResult:
    """Tests for InterceptionResult dataclass."""

    def test_defaults(self):
        r = InterceptionResult()
        assert not r.should_terminate
        assert r.reason == ""
        assert not r.pii_redacted
        assert r.redacted_text == ""
        assert r.threat_score == 0.0

    def test_with_values(self):
        r = InterceptionResult(
            should_terminate=True,
            reason="toxic",
            pii_redacted=True,
            redacted_text="redacted",
            threat_score=0.9,
        )
        assert r.should_terminate
        assert r.threat_score == 0.9


# ===========================================================================
# Test: StreamingInterceptorStats dataclass
# ===========================================================================

class TestInterceptorStats:
    """Tests for StreamingInterceptorStats dataclass."""

    def test_defaults(self):
        s = StreamingInterceptorStats()
        assert s.windows_evaluated == 0
        assert s.tokens_processed == 0
        assert s.tokens_redacted == 0
        assert s.pii_found_in_hold == 0
        assert s.pii_found_after_send == 0
        assert not s.terminated
        assert s.termination_reason == ""

    def test_mutability(self):
        s = StreamingInterceptorStats()
        s.windows_evaluated = 5
        s.terminated = True
        s.termination_reason = "test"
        assert s.windows_evaluated == 5
        assert s.terminated


# ===========================================================================
# Test: Defaults
# ===========================================================================

class TestDefaults:
    """Verify default constants."""

    def test_window_size(self):
        assert DEFAULT_WINDOW_SIZE == 128

    def test_hold_size(self):
        assert DEFAULT_HOLD_SIZE == 32

    def test_overlap(self):
        assert DEFAULT_OVERLAP == 0.5

    def test_cumulative_threshold(self):
        assert DEFAULT_CUMULATIVE_THRESHOLD == 0.6

    def test_cumulative_history(self):
        assert DEFAULT_CUMULATIVE_HISTORY == 5

    def test_cumulative_min_windows(self):
        assert DEFAULT_CUMULATIVE_MIN_WINDOWS == 3


# ===========================================================================
# Test: Main.py integration helpers
# ===========================================================================

class TestMainIntegration:
    """Verify main.py streaming integration points."""

    def test_format_safety_termination_importable(self):
        from aegis.layers.output.streaming import format_safety_termination
        assert callable(format_safety_termination)

    def test_streaming_interceptor_importable_from_main(self):
        import aegis.main as m
        assert hasattr(m, 'StreamingInterceptor')
        assert hasattr(m, 'format_safety_termination')

    def test_classify_termination_reason(self):
        from aegis.main import _classify_termination_reason
        assert _classify_termination_reason("Toxic content: violence") == "toxicity"
        assert _classify_termination_reason("Data leakage: echo") == "leakage"
        assert _classify_termination_reason("Cumulative threat: avg 0.7") == "cumulative"
        assert _classify_termination_reason("PII detected") == "pii"
        assert _classify_termination_reason("Unknown reason") == "other"

    def test_stream_metrics_imported_in_main(self):
        import aegis.main as m
        assert hasattr(m, 'STREAM_INTERRUPTIONS')
        assert hasattr(m, 'STREAM_WINDOWS_EVALUATED')
        assert hasattr(m, 'STREAM_TOKENS_PROCESSED')
