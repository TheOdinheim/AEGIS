"""
Multimodal Audio Security Tests — Phase 3.

Tests validate:
1. Audio transcriber — backend detection, format recognition, graceful fallback
2. Spectral analyzer — ultrasonic/infrasonic detection, entropy, discontinuities
3. Audio sanitizer — bandpass filtering, WAV re-encoding, format normalization
4. Audio scanner integration — L2 fast path, format validation, size limits
5. Audio analyzer — WaveGuard comparison, divergence scoring
6. MultimodalPreprocessor — audio extraction from messages
7. Config integration (audio fields in MultimodalConfig)
8. Metrics registration

All tests use wave module (stdlib) to generate synthetic WAV audio —
no test fixtures needed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import struct
import wave

import numpy as np
import pytest

from aegis.config import AegisConfig, MultimodalConfig
from aegis.layers.multimodal import (
    MultimodalPreprocessor,
    MultimodalScanReport,
    AudioTranscriber,
    TranscriptionResult,
    SpectralAnalyzer,
    SpectralAnalysisResult,
    AudioSanitizer,
    AudioSanitizationReport,
    AudioScanner,
    AudioScanReport,
    AudioAnalyzer,
    AudioAnalysisReport,
)
from aegis.layers.multimodal.audio_transcriber import detect_audio_format, _parse_wav_duration
from aegis.layers.multimodal.audio_analyzer import _compute_divergence, _word_edit_distance
from aegis.models.scan_result import ScanResult, ThreatCategory


def _run(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test helpers — generate synthetic WAV audio
# ---------------------------------------------------------------------------

def make_wav(
    duration_seconds: float = 1.0,
    frequency_hz: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    """Generate a WAV file with a sine wave tone."""
    n_samples = int(sample_rate * duration_seconds)
    samples = np.zeros(n_samples, dtype=np.float64)
    t = np.arange(n_samples) / sample_rate
    samples = amplitude * np.sin(2 * np.pi * frequency_hz * t)

    # Convert to 16-bit PCM
    int_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def make_wav_with_ultrasonic(
    duration_seconds: float = 1.0,
    base_freq: float = 440.0,
    ultrasonic_freq: float = 22000.0,
    sample_rate: int = 48000,
    ultrasonic_amplitude: float = 0.3,
) -> bytes:
    """Generate WAV with both audible and ultrasonic content."""
    n_samples = int(sample_rate * duration_seconds)
    t = np.arange(n_samples) / sample_rate
    # Audible tone + ultrasonic component
    samples = 0.5 * np.sin(2 * np.pi * base_freq * t) + \
              ultrasonic_amplitude * np.sin(2 * np.pi * ultrasonic_freq * t)

    int_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def make_wav_with_infrasonic(
    duration_seconds: float = 1.0,
    infrasonic_freq: float = 10.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate WAV with infrasonic content."""
    n_samples = int(sample_rate * duration_seconds)
    t = np.arange(n_samples) / sample_rate
    # Strong infrasonic component + weak audible
    samples = 0.8 * np.sin(2 * np.pi * infrasonic_freq * t) + \
              0.1 * np.sin(2 * np.pi * 440.0 * t)

    int_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def make_silent_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file."""
    n_samples = int(sample_rate * duration_seconds)
    int_samples = np.zeros(n_samples, dtype=np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def make_stereo_wav(duration_seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a stereo WAV file."""
    n_samples = int(sample_rate * duration_seconds)
    t = np.arange(n_samples) / sample_rate
    left = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    right = (0.5 * np.sin(2 * np.pi * 880.0 * t) * 32767).astype(np.int16)
    # Interleave channels
    interleaved = np.empty(n_samples * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Audio Transcriber Tests
# ---------------------------------------------------------------------------

class TestAudioTranscriber:
    """Test AudioTranscriber backend detection and graceful fallback."""

    def test_transcriber_handles_missing_whisper(self):
        """Transcriber doesn't crash if whisper not installed."""
        t = AudioTranscriber()
        # Should not raise
        result = t.transcribe(make_wav(0.5))
        assert isinstance(result, TranscriptionResult)
        # Without whisper/speech_recognition, returns unavailable or empty
        assert result.method in ("whisper", "speech_recognition", "unavailable")

    def test_transcriber_detects_wav_format(self):
        """Transcriber identifies WAV by magic bytes."""
        wav = make_wav(0.5)
        fmt = detect_audio_format(wav)
        assert fmt == "wav"

    def test_transcriber_returns_valid_result(self):
        """TranscriptionResult has all expected fields."""
        t = AudioTranscriber()
        result = t.transcribe(make_wav(0.5))
        assert isinstance(result.text, str)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.language, str)
        assert isinstance(result.duration_seconds, float)
        assert isinstance(result.method, str)

    def test_empty_audio_returns_empty_transcription(self):
        """Empty bytes returns empty transcription."""
        t = AudioTranscriber()
        result = t.transcribe(b"")
        assert not result.has_text
        assert result.method == "unavailable"

    def test_invalid_audio_handled_gracefully(self):
        """Garbage bytes don't crash transcriber."""
        t = AudioTranscriber()
        result = t.transcribe(b"this is not audio data at all")
        assert isinstance(result, TranscriptionResult)
        # Should not crash, returns empty or unavailable


class TestFormatDetection:
    """Test audio format detection by magic bytes."""

    def test_wav_detection(self):
        assert detect_audio_format(make_wav(0.1)) == "wav"

    def test_mp3_id3_detection(self):
        # ID3 tag header
        fake_mp3 = b"ID3" + b"\x00" * 100
        assert detect_audio_format(fake_mp3) == "mp3"

    def test_flac_detection(self):
        fake_flac = b"fLaC" + b"\x00" * 100
        assert detect_audio_format(fake_flac) == "flac"

    def test_ogg_detection(self):
        fake_ogg = b"OggS" + b"\x00" * 100
        assert detect_audio_format(fake_ogg) == "ogg"

    def test_webm_detection(self):
        fake_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        assert detect_audio_format(fake_webm) == "webm"

    def test_unknown_format(self):
        assert detect_audio_format(b"\x00\x01\x02\x03" * 10) == ""

    def test_short_data(self):
        assert detect_audio_format(b"abc") == ""


class TestWavDuration:
    """Test WAV duration parsing."""

    def test_parses_duration(self):
        wav = make_wav(2.0, sample_rate=16000)
        dur = _parse_wav_duration(wav)
        assert abs(dur - 2.0) < 0.1

    def test_short_wav(self):
        wav = make_wav(0.1, sample_rate=16000)
        dur = _parse_wav_duration(wav)
        assert abs(dur - 0.1) < 0.05


# ---------------------------------------------------------------------------
# 2. Spectral Analysis Tests
# ---------------------------------------------------------------------------

class TestSpectralAnalyzer:
    """Test spectral analysis for adversarial audio detection."""

    def test_normal_audio_passes(self):
        """440Hz tone at 16kHz sample rate — no ultrasonic/infrasonic."""
        analyzer = SpectralAnalyzer()
        wav = make_wav(1.0, frequency_hz=440.0, sample_rate=16000)
        result = analyzer.analyze(wav)
        assert isinstance(result, SpectralAnalysisResult)
        assert not result.has_ultrasonic
        assert result.sample_rate == 16000
        assert result.duration_seconds > 0.9

    def test_ultrasonic_detected(self):
        """22kHz component flagged as ultrasonic."""
        analyzer = SpectralAnalyzer()
        wav = make_wav_with_ultrasonic(
            duration_seconds=1.0,
            base_freq=440.0,
            ultrasonic_freq=22000.0,
            sample_rate=48000,
            ultrasonic_amplitude=0.5,
        )
        result = analyzer.analyze(wav)
        assert result.has_ultrasonic

    def test_infrasonic_detected(self):
        """10Hz component flagged as infrasonic."""
        analyzer = SpectralAnalyzer()
        wav = make_wav_with_infrasonic(
            duration_seconds=2.0,
            infrasonic_freq=10.0,
            sample_rate=16000,
        )
        result = analyzer.analyze(wav)
        assert result.has_infrasonic

    def test_spectral_entropy_computed(self):
        """Spectral entropy is a positive number for tonal audio."""
        analyzer = SpectralAnalyzer()
        wav = make_wav(1.0, frequency_hz=440.0, sample_rate=16000)
        result = analyzer.analyze(wav)
        assert result.spectral_entropy > 0.0

    def test_short_audio_handled(self):
        """Very short audio (<1s) doesn't crash."""
        analyzer = SpectralAnalyzer()
        wav = make_wav(0.05, sample_rate=16000)
        result = analyzer.analyze(wav)
        assert isinstance(result, SpectralAnalysisResult)
        assert result.duration_seconds > 0

    def test_invalid_audio_handled(self):
        """Non-WAV bytes handled gracefully."""
        analyzer = SpectralAnalyzer()
        result = analyzer.analyze(b"not audio")
        assert isinstance(result, SpectralAnalysisResult)
        assert "error" in result.details

    def test_silent_audio(self):
        """Silent audio produces low anomaly score."""
        analyzer = SpectralAnalyzer()
        wav = make_silent_wav(1.0)
        result = analyzer.analyze(wav)
        assert isinstance(result, SpectralAnalysisResult)
        assert result.anomaly_score < 0.5

    def test_stereo_handled(self):
        """Stereo WAV is mixed to mono and analyzed."""
        analyzer = SpectralAnalyzer()
        wav = make_stereo_wav(1.0)
        result = analyzer.analyze(wav)
        assert isinstance(result, SpectralAnalysisResult)
        assert result.sample_rate > 0


# ---------------------------------------------------------------------------
# 3. Audio Sanitizer Tests
# ---------------------------------------------------------------------------

class TestAudioSanitizer:
    """Test audio re-encoding/sanitization."""

    def test_sanitized_output_is_valid_wav(self):
        """Sanitized output can be opened as WAV."""
        sanitizer = AudioSanitizer()
        wav = make_wav(1.0, sample_rate=44100)
        sanitized, report = sanitizer.sanitize(wav)
        assert report.success
        assert len(sanitized) > 0
        # Verify it's a valid WAV
        buf = io.BytesIO(sanitized)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000  # Resampled

    def test_ultrasonic_removed(self):
        """Ultrasonic content removed after sanitization."""
        sanitizer = AudioSanitizer(output_sample_rate=16000, lowpass_cutoff_hz=8000)
        wav = make_wav_with_ultrasonic(1.0, sample_rate=48000, ultrasonic_freq=22000)
        sanitized, report = sanitizer.sanitize(wav)
        assert report.success
        assert report.ultrasonic_removed

        # Verify no ultrasonic in sanitized output
        analyzer = SpectralAnalyzer()
        result = analyzer.analyze(sanitized)
        # At 16kHz sample rate, Nyquist is 8kHz — no ultrasonic possible
        assert not result.has_ultrasonic

    def test_duration_preserved(self):
        """Duration approximately preserved after sanitization."""
        sanitizer = AudioSanitizer()
        duration = 2.0
        wav = make_wav(duration, sample_rate=44100)
        sanitized, report = sanitizer.sanitize(wav)
        assert report.success
        assert abs(report.duration_seconds - duration) < 0.1

    def test_format_normalized(self):
        """Output is always WAV regardless of input format."""
        sanitizer = AudioSanitizer()
        wav = make_wav(0.5, sample_rate=44100)
        sanitized, report = sanitizer.sanitize(wav)
        assert report.success
        assert report.format_changed
        fmt = detect_audio_format(sanitized)
        assert fmt == "wav"

    def test_non_wav_returns_error(self):
        """Non-WAV input returns error (no external deps)."""
        sanitizer = AudioSanitizer()
        _, report = sanitizer.sanitize(b"ID3" + b"\x00" * 100)
        assert not report.success
        assert "unsupported format" in report.error


# ---------------------------------------------------------------------------
# 4. Audio Scanner Integration Tests
# ---------------------------------------------------------------------------

class TestAudioScanner:
    """Test L2 fast-path audio scanner."""

    def test_scanner_returns_valid_scan_result(self):
        """Scanner returns ScanResult for valid WAV."""
        scanner = AudioScanner()
        wav = make_wav(1.0)
        result = _run(scanner.scan(wav))
        assert isinstance(result, ScanResult)
        assert isinstance(result.confidence, float)

    def test_scanner_handles_missing_transcription(self):
        """Scanner doesn't crash without transcription backends."""
        scanner = AudioScanner()
        wav = make_wav(0.5)
        result = _run(scanner.scan(wav))
        assert isinstance(result, ScanResult)
        # Should complete without error

    def test_format_validation_rejects_non_audio(self):
        """Non-audio bytes rejected with high confidence."""
        scanner = AudioScanner()
        result = _run(scanner.scan(b"this is not audio"))
        assert result.is_threat
        assert result.confidence >= 0.9
        assert "unknown_audio_format" in result.matched_patterns

    def test_oversized_audio_rejected(self):
        """Audio exceeding size limit rejected."""
        scanner = AudioScanner(max_audio_size_bytes=100)
        wav = make_wav(1.0)  # >100 bytes
        result = _run(scanner.scan(wav))
        assert result.is_threat
        assert "oversized_audio" in result.matched_patterns[0]

    def test_sync_report(self):
        """Synchronous scan_sync returns AudioScanReport."""
        scanner = AudioScanner()
        wav = make_wav(0.5)
        report = scanner.scan_sync(wav)
        assert isinstance(report, AudioScanReport)
        assert report.detected_format == "wav"
        assert len(report.scan_results) > 0


# ---------------------------------------------------------------------------
# 5. WaveGuard Comparison Tests
# ---------------------------------------------------------------------------

class TestWaveGuardComparison:
    """Test WaveGuard-style sanitize-and-compare analysis."""

    def test_clean_audio_low_divergence(self):
        """Clean audio: original and sanitized transcriptions should match."""
        # Both will have empty transcriptions (no whisper), so divergence = 0
        analyzer = AudioAnalyzer()
        wav = make_wav(1.0, frequency_hz=440.0, sample_rate=16000)
        report = analyzer.analyze_full(wav)
        # With no transcription backend, both transcriptions are empty → divergence 0
        assert report.divergence_score == 0.0

    def test_divergence_function_identical(self):
        """Identical texts produce zero divergence."""
        assert _compute_divergence("hello world", "hello world") == 0.0

    def test_divergence_function_different(self):
        """Completely different texts produce high divergence."""
        div = _compute_divergence("hello world", "foo bar baz qux")
        assert div > 0.5

    def test_divergence_function_empty(self):
        """Both empty = zero divergence."""
        assert _compute_divergence("", "") == 0.0

    def test_divergence_one_empty(self):
        """One empty, one not = high divergence."""
        div = _compute_divergence("hello world", "")
        assert div == 0.8

    def test_word_edit_distance(self):
        """Word-level edit distance correct."""
        assert _word_edit_distance("the cat sat", "the dog sat") == 1
        assert _word_edit_distance("hello", "hello") == 0
        assert _word_edit_distance("a b c", "x y z") == 3


# ---------------------------------------------------------------------------
# 6. Audio Analyzer Full Tests
# ---------------------------------------------------------------------------

class TestAudioAnalyzer:
    """Test L3 slow-path audio analyzer."""

    def test_analyzer_returns_scan_result(self):
        """Async analyze returns ScanResult."""
        analyzer = AudioAnalyzer()
        wav = make_wav(1.0)
        result = _run(analyzer.analyze(wav))
        assert isinstance(result, ScanResult)

    def test_analyzer_full_report(self):
        """Full analysis returns detailed report."""
        analyzer = AudioAnalyzer()
        wav = make_wav(1.0, sample_rate=16000)
        report = analyzer.analyze_full(wav)
        assert isinstance(report, AudioAnalysisReport)
        assert report.original_transcription is not None
        assert len(report.scan_results) > 0

    def test_analyzer_unknown_format(self):
        """Unknown format flagged."""
        analyzer = AudioAnalyzer()
        result = _run(analyzer.analyze(b"garbage data here"))
        assert result.is_threat
        assert result.confidence >= 0.85


# ---------------------------------------------------------------------------
# 7. MultimodalPreprocessor Audio Integration
# ---------------------------------------------------------------------------

class TestPreprocessorAudio:
    """Test audio extraction and routing in MultimodalPreprocessor."""

    def _make_audio_message(self, wav_bytes: bytes) -> list[dict]:
        """Create an OpenAI-format message with input_audio."""
        b64 = base64.b64encode(wav_bytes).decode()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": "wav"},
                    },
                ],
            }
        ]

    def _make_audio_data_uri_message(self, wav_bytes: bytes) -> list[dict]:
        """Create a message with audio as data URI."""
        b64 = base64.b64encode(wav_bytes).decode()
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:audio/wav;base64,{b64}"},
                    },
                ],
            }
        ]

    def test_preprocessor_detects_audio(self):
        """Preprocessor finds audio content in messages."""
        pp = MultimodalPreprocessor(enabled=False, document_scanning_enabled=False)
        wav = make_wav(0.5)
        messages = self._make_audio_message(wav)
        report = pp.preprocess_messages(messages)
        assert report.audio_found == 1

    def test_preprocessor_extracts_data_uri_audio(self):
        """Preprocessor finds audio in data URI format."""
        pp = MultimodalPreprocessor(enabled=False, document_scanning_enabled=False)
        wav = make_wav(0.5)
        messages = self._make_audio_data_uri_message(wav)
        report = pp.preprocess_messages(messages)
        assert report.audio_found == 1

    def test_preprocessor_scan_audio(self):
        """L2 audio scanning via preprocessor."""
        pp = MultimodalPreprocessor(enabled=False, document_scanning_enabled=False)
        wav = make_wav(0.5)
        messages = self._make_audio_message(wav)
        results = _run(pp.scan_audio(messages))
        assert len(results) == 1
        assert isinstance(results[0], ScanResult)

    def test_preprocessor_analyze_audio(self):
        """L3 audio analysis via preprocessor."""
        pp = MultimodalPreprocessor(enabled=False, document_scanning_enabled=False)
        wav = make_wav(0.5)
        messages = self._make_audio_message(wav)
        results = _run(pp.analyze_audio(messages))
        assert len(results) == 1
        assert isinstance(results[0], ScanResult)

    def test_preprocessor_audio_disabled(self):
        """No audio scanning when disabled."""
        pp = MultimodalPreprocessor(
            enabled=False,
            document_scanning_enabled=False,
            audio_scanning_enabled=False,
        )
        wav = make_wav(0.5)
        messages = self._make_audio_message(wav)
        report = pp.preprocess_messages(messages)
        assert report.audio_found == 0
        results = _run(pp.scan_audio(messages))
        assert results == []

    def test_preprocessor_no_audio_in_text_messages(self):
        """Text-only messages produce no audio findings."""
        pp = MultimodalPreprocessor(enabled=False, document_scanning_enabled=False)
        messages = [{"role": "user", "content": "Hello, world!"}]
        report = pp.preprocess_messages(messages)
        assert report.audio_found == 0

    def test_preprocessor_file_attachment_audio(self):
        """Audio via file attachment format extracted correctly."""
        pp = MultimodalPreprocessor(enabled=False, document_scanning_enabled=False)
        wav = make_wav(0.5)
        b64 = base64.b64encode(wav).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "data": b64,
                            "mime_type": "audio/wav",
                            "name": "recording.wav",
                        },
                    },
                ],
            }
        ]
        report = pp.preprocess_messages(messages)
        assert report.audio_found == 1


# ---------------------------------------------------------------------------
# 8. Config Integration Tests
# ---------------------------------------------------------------------------

class TestAudioConfig:
    """Test audio config fields in MultimodalConfig."""

    def test_default_audio_enabled(self):
        cfg = MultimodalConfig()
        assert cfg.audio_scanning_enabled is True

    def test_default_audio_max_size(self):
        cfg = MultimodalConfig()
        assert cfg.audio_max_size_mb == 100.0

    def test_default_audio_max_duration(self):
        cfg = MultimodalConfig()
        assert cfg.audio_max_duration_seconds == 1800.0

    def test_aegis_config_has_audio_fields(self):
        cfg = AegisConfig()
        assert hasattr(cfg.multimodal, "audio_scanning_enabled")
        assert hasattr(cfg.multimodal, "audio_max_size_mb")
        assert hasattr(cfg.multimodal, "audio_max_duration_seconds")


# ---------------------------------------------------------------------------
# 9. Metrics Registration Tests
# ---------------------------------------------------------------------------

class TestAudioMetrics:
    """Test that audio metrics are registered."""

    def test_audio_scanned_counter_exists(self):
        from aegis.middleware.metrics import MULTIMODAL_AUDIO_SCANNED
        assert MULTIMODAL_AUDIO_SCANNED is not None

    def test_audio_threats_counter_exists(self):
        from aegis.middleware.metrics import MULTIMODAL_AUDIO_THREATS
        assert MULTIMODAL_AUDIO_THREATS is not None

    def test_audio_transcription_latency_exists(self):
        from aegis.middleware.metrics import MULTIMODAL_AUDIO_TRANSCRIPTION_LATENCY
        assert MULTIMODAL_AUDIO_TRANSCRIPTION_LATENCY is not None
