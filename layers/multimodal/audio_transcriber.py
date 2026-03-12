"""
Audio Transcriber — Audio-to-text transcription engine.

Transcribes audio content to text for feeding through existing L2/L3 text
detection pipeline. Supports multiple backends with graceful fallback:
1. Primary: OpenAI Whisper (openai-whisper)
2. Secondary: speech_recognition library
3. Fallback: Empty transcription with method="unavailable"

ASSUMED-BREACH POSTURE: Audio is an untrusted input channel. Adversarial
audio perturbations achieve 80-100% attack success rates against commercial
ASR systems. WhisperInject demonstrates benign-sounding audio that covertly
induces harmful text generation with 86%+ success. The transcription is
never trusted — it feeds into L2/L3 scanning.
"""

from __future__ import annotations

import io
import logging
import struct
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Magic bytes for audio format detection
_AUDIO_SIGNATURES: dict[str, bytes] = {
    "WAV": b"RIFF",
    "FLAC": b"fLaC",
    "OGG": b"OggS",
}
_MP3_SYNC = b"\xff\xfb"
_MP3_SYNC2 = b"\xff\xf3"
_MP3_SYNC3 = b"\xff\xf2"
_MP3_ID3 = b"ID3"
_WEBM_MAGIC = b"\x1a\x45\xdf\xa3"  # EBML header (used by WebM)


def detect_audio_format(data: bytes) -> str:
    """Detect audio format from magic bytes.

    Returns format string: 'wav', 'mp3', 'flac', 'ogg', 'webm', or ''.
    """
    if len(data) < 12:
        return ""

    # WAV: RIFF....WAVE
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"

    # FLAC
    if data[:4] == b"fLaC":
        return "flac"

    # OGG (also used for Opus)
    if data[:4] == b"OggS":
        return "ogg"

    # MP3: ID3 tag or sync bytes
    if data[:3] == _MP3_ID3:
        return "mp3"
    if data[:2] in (_MP3_SYNC, _MP3_SYNC2, _MP3_SYNC3):
        return "mp3"

    # WebM/Matroska (EBML header)
    if data[:4] == _WEBM_MAGIC:
        return "webm"

    return ""


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""
    text: str
    confidence: float
    language: str
    duration_seconds: float
    method: str  # "whisper", "speech_recognition", "unavailable"

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


def _empty_result(method: str = "unavailable", duration: float = 0.0) -> TranscriptionResult:
    """Return an empty transcription result."""
    return TranscriptionResult(
        text="",
        confidence=0.0,
        language="",
        duration_seconds=duration,
        method=method,
    )


def _parse_wav_duration(data: bytes) -> float:
    """Parse WAV header to get duration in seconds."""
    try:
        if len(data) < 44:
            return 0.0
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return 0.0
        # Find fmt chunk
        pos = 12
        sample_rate = 0
        byte_rate = 0
        while pos < len(data) - 8:
            chunk_id = data[pos:pos + 4]
            chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
            if chunk_id == b"fmt ":
                if chunk_size >= 16:
                    sample_rate = struct.unpack_from("<I", data, pos + 12)[0]
                    byte_rate = struct.unpack_from("<I", data, pos + 16)[0]
            elif chunk_id == b"data":
                if byte_rate > 0:
                    return chunk_size / byte_rate
                elif sample_rate > 0:
                    # Fallback: assume 16-bit mono
                    return chunk_size / (sample_rate * 2)
                return 0.0
            pos += 8 + chunk_size
            if chunk_size % 2:  # WAV chunks are word-aligned
                pos += 1
    except Exception:
        pass
    return 0.0


class AudioTranscriber:
    """Audio-to-text transcription engine with multiple backend support.

    Attempts backends in order:
    1. OpenAI Whisper (openai-whisper package)
    2. speech_recognition library
    3. Returns empty result with method="unavailable"

    CRITICAL: Never crashes on missing dependencies — graceful degradation.
    """

    def __init__(self):
        self._whisper_model = None
        self._whisper_available = False
        self._sr_available = False
        self._init_backends()

    def _init_backends(self) -> None:
        """Probe available transcription backends."""
        # Try whisper
        try:
            import whisper  # noqa: F401
            self._whisper_available = True
        except ImportError:
            self._whisper_available = False

        # Try speech_recognition
        try:
            import speech_recognition  # noqa: F401
            self._sr_available = True
        except ImportError:
            self._sr_available = False

    def _load_whisper(self):
        """Lazy-load whisper model."""
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            import whisper
            self._whisper_model = whisper.load_model("base")
            return self._whisper_model
        except Exception as e:
            logger.warning("Failed to load whisper model: %s", e)
            self._whisper_available = False
            return None

    def transcribe(self, audio_bytes: bytes, format: str = "wav") -> TranscriptionResult:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes.
            format: Audio format hint. Auto-detected from magic bytes if possible.

        Returns:
            TranscriptionResult with transcribed text (may be empty).
        """
        if not audio_bytes:
            return _empty_result()

        # Auto-detect format
        detected = detect_audio_format(audio_bytes)
        if detected:
            format = detected

        # Get duration from WAV header if possible
        duration = 0.0
        if format == "wav":
            duration = _parse_wav_duration(audio_bytes)

        # Try whisper first
        if self._whisper_available:
            result = self._transcribe_whisper(audio_bytes, format, duration)
            if result is not None:
                return result

        # Try speech_recognition
        if self._sr_available and format == "wav":
            result = self._transcribe_sr(audio_bytes, duration)
            if result is not None:
                return result

        return _empty_result(duration=duration)

    def _transcribe_whisper(
        self, audio_bytes: bytes, format: str, duration: float
    ) -> TranscriptionResult | None:
        """Transcribe using OpenAI Whisper."""
        try:
            import whisper
            import tempfile
            import os

            model = self._load_whisper()
            if model is None:
                return None

            # Whisper needs a file path
            suffix = f".{format}" if format else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            try:
                result = model.transcribe(tmp_path)
                text = result.get("text", "").strip()
                language = result.get("language", "")
                return TranscriptionResult(
                    text=text,
                    confidence=0.8,  # Whisper doesn't provide per-segment confidence easily
                    language=language,
                    duration_seconds=duration,
                    method="whisper",
                )
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            logger.warning("Whisper transcription failed: %s", e)
            return None

    def _transcribe_sr(
        self, audio_bytes: bytes, duration: float
    ) -> TranscriptionResult | None:
        """Transcribe using speech_recognition (WAV only)."""
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            audio_file = io.BytesIO(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)

            # Try Google Web Speech API (requires internet)
            text = recognizer.recognize_google(audio_data)
            return TranscriptionResult(
                text=text,
                confidence=0.7,
                language="en",
                duration_seconds=duration,
                method="speech_recognition",
            )
        except Exception as e:
            logger.debug("speech_recognition failed: %s", e)
            return None

    @property
    def available_backends(self) -> list[str]:
        """List available transcription backends."""
        backends = []
        if self._whisper_available:
            backends.append("whisper")
        if self._sr_available:
            backends.append("speech_recognition")
        return backends


__all__ = [
    "AudioTranscriber",
    "TranscriptionResult",
    "detect_audio_format",
]
