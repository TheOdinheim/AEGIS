"""
Audio Sanitizer — Re-encode audio to destroy adversarial perturbations.

WaveGuard-style defense: decode audio to raw PCM samples, apply bandpass
filtering (80Hz–16kHz), re-encode as normalized WAV. This destroys:
- Ultrasonic perturbations (removed by 16kHz low-pass)
- Infrasonic hidden data (removed by 80Hz high-pass)
- Encoding-specific adversarial artifacts (lost during re-encoding)

Uses only numpy + wave (stdlib) — no heavy audio processing dependencies.

ASSUMED-BREACH POSTURE: Sanitization is not a detection mechanism — it's a
destruction mechanism. The sanitized audio is used for comparison with the
original (WaveGuard technique). If transcriptions diverge significantly,
the original likely contains adversarial content.
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSanitizationReport:
    """Report from audio sanitization."""
    original_format: str
    original_sample_rate: int
    output_sample_rate: int
    duration_seconds: float
    ultrasonic_removed: bool
    format_changed: bool
    success: bool = True
    error: str = ""


def _simple_lowpass(samples: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Apply a simple low-pass filter using FFT.

    Zeroes FFT bins above cutoff frequency, then inverse FFT.
    """
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    fft[freqs > cutoff_hz] = 0
    return np.fft.irfft(fft, n=len(samples))


def _simple_highpass(samples: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Apply a simple high-pass filter using FFT.

    Zeroes FFT bins below cutoff frequency, then inverse FFT.
    """
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    fft[freqs < cutoff_hz] = 0
    return np.fft.irfft(fft, n=len(samples))


def _parse_wav(audio_bytes: bytes) -> tuple[np.ndarray, int, int] | None:
    """Parse WAV to (samples, sample_rate, n_channels). Returns None on failure."""
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None
            raw = wf.readframes(n_frames)

        if sampwidth == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
            samples /= 128.0
        elif sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            samples /= 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
            samples /= 2147483648.0
        else:
            return None

        # Mix to mono
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples, sample_rate, n_channels
    except Exception as e:
        logger.debug("Failed to parse WAV for sanitization: %s", e)
        return None


def _encode_wav(
    samples: np.ndarray, sample_rate: int, sampwidth: int = 2
) -> bytes:
    """Encode numpy float64 samples to WAV bytes (mono, 16-bit PCM)."""
    # Clip to [-1, 1] and convert to int16
    clipped = np.clip(samples, -1.0, 1.0)
    if sampwidth == 2:
        int_samples = (clipped * 32767).astype(np.int16)
    else:
        int_samples = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())

    return buf.getvalue()


class AudioSanitizer:
    """Re-encode audio to destroy adversarial perturbations.

    Pipeline:
    1. Decode audio to raw PCM samples
    2. Apply low-pass filter at cutoff_hz (removes ultrasonic)
    3. Apply high-pass filter at highpass_hz (removes infrasonic)
    4. Re-encode as WAV at output_sample_rate, 16-bit mono PCM

    Non-WAV input formats are not supported in the bootstrap (would
    require ffmpeg/pydub). Returns error report for non-WAV.
    """

    def __init__(
        self,
        output_sample_rate: int = 16000,
        lowpass_cutoff_hz: float = 16000.0,
        highpass_cutoff_hz: float = 80.0,
    ):
        self._output_sr = output_sample_rate
        self._lowpass_hz = lowpass_cutoff_hz
        self._highpass_hz = highpass_cutoff_hz

    def sanitize(self, audio_bytes: bytes) -> tuple[bytes, AudioSanitizationReport]:
        """Sanitize audio by re-encoding through bandpass filter.

        Args:
            audio_bytes: Raw audio file bytes (WAV format).

        Returns:
            Tuple of (sanitized_wav_bytes, report).
            On failure, returns (b"", report_with_error).
        """
        from aegis.layers.multimodal.audio_transcriber import detect_audio_format

        detected_format = detect_audio_format(audio_bytes)
        if not detected_format:
            detected_format = "unknown"

        # Currently only WAV is supported without external deps
        if detected_format != "wav":
            report = AudioSanitizationReport(
                original_format=detected_format,
                original_sample_rate=0,
                output_sample_rate=self._output_sr,
                duration_seconds=0.0,
                ultrasonic_removed=False,
                format_changed=False,
                success=False,
                error=f"unsupported format: {detected_format}",
            )
            return b"", report

        parsed = _parse_wav(audio_bytes)
        if parsed is None:
            report = AudioSanitizationReport(
                original_format=detected_format,
                original_sample_rate=0,
                output_sample_rate=self._output_sr,
                duration_seconds=0.0,
                ultrasonic_removed=False,
                format_changed=False,
                success=False,
                error="failed to parse WAV",
            )
            return b"", report

        samples, orig_sr, n_channels = parsed
        duration = len(samples) / orig_sr

        # Track whether we removed ultrasonic
        has_ultrasonic = orig_sr > self._lowpass_hz * 2

        # Apply low-pass filter (remove ultrasonic)
        if has_ultrasonic and self._lowpass_hz < orig_sr / 2:
            samples = _simple_lowpass(samples, self._lowpass_hz, orig_sr)

        # Apply high-pass filter (remove infrasonic)
        if self._highpass_hz > 0:
            samples = _simple_highpass(samples, self._highpass_hz, orig_sr)

        # Resample to output sample rate if needed
        if orig_sr != self._output_sr:
            # Simple resampling via linear interpolation
            n_out = int(len(samples) * self._output_sr / orig_sr)
            if n_out > 0:
                x_orig = np.linspace(0, 1, len(samples))
                x_new = np.linspace(0, 1, n_out)
                samples = np.interp(x_new, x_orig, samples)

        # Encode to WAV
        sanitized_bytes = _encode_wav(samples, self._output_sr)

        report = AudioSanitizationReport(
            original_format=detected_format,
            original_sample_rate=orig_sr,
            output_sample_rate=self._output_sr,
            duration_seconds=duration,
            ultrasonic_removed=has_ultrasonic,
            format_changed=True,
        )

        return sanitized_bytes, report


__all__ = [
    "AudioSanitizer",
    "AudioSanitizationReport",
]
