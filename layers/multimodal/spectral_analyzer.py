"""
Spectral Analyzer — FFT-based audio spectral analysis for adversarial detection.

Adversarial audio perturbations often exhibit spectral characteristics that
differ from natural speech:
- Ultrasonic content (>20kHz): inaudible to humans but processable by models
- Infrasonic content (<20Hz): can carry hidden modulated data
- Abnormal spectral entropy: adversarial perturbations create non-natural
  frequency distributions
- Spectral discontinuities: adversarial bursts embedded in normal audio

Uses only numpy + wave module (stdlib) — no heavy audio processing deps.

ASSUMED-BREACH POSTURE: Audio is untrusted. Spectral analysis is one layer
of defense. A sophisticated adversary can craft perturbations that pass
spectral checks — the transcribe-and-scan path provides independent defense.
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
class SpectralAnalysisResult:
    """Result from spectral analysis of audio."""
    has_ultrasonic: bool = False
    has_infrasonic: bool = False
    spectral_entropy: float = 0.0
    anomaly_score: float = 0.0
    duration_seconds: float = 0.0
    sample_rate: int = 0
    suspicious: bool = False
    details: str = ""

    @property
    def is_clean(self) -> bool:
        return not self.suspicious


def _error_result(msg: str) -> SpectralAnalysisResult:
    """Return a result for unparseable audio."""
    return SpectralAnalysisResult(details=f"error: {msg}")


def _parse_wav_samples(audio_bytes: bytes) -> tuple[np.ndarray, int] | None:
    """Parse WAV audio bytes into numpy array and sample rate.

    Returns (samples_array, sample_rate) or None on failure.
    """
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

        # Convert raw bytes to numpy array based on sample width
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

        # Mix to mono if stereo
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples, sample_rate
    except Exception as e:
        logger.debug("Failed to parse WAV: %s", e)
        return None


def _compute_spectral_entropy(power_spectrum: np.ndarray) -> float:
    """Compute Shannon entropy of normalized power spectrum.

    Natural speech has moderate entropy (diverse but structured frequencies).
    Adversarial perturbations may have unusually high or low entropy.

    Returns entropy in range [0.0, ~log2(N)].
    """
    total = power_spectrum.sum()
    if total < 1e-10:
        return 0.0
    p = power_spectrum / total
    # Avoid log(0)
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log2(p)))


class SpectralAnalyzer:
    """FFT-based spectral analysis for adversarial audio detection.

    Analyzes WAV audio for:
    1. Ultrasonic content (>20kHz) — inaudible but model-processable
    2. Infrasonic content (<20Hz) — may carry hidden data
    3. Spectral entropy — adversarial perturbations alter frequency distribution
    4. Spectral discontinuities — adversarial bursts in normal audio
    """

    def __init__(
        self,
        ultrasonic_threshold_hz: float = 20000.0,
        infrasonic_threshold_hz: float = 20.0,
        ultrasonic_power_threshold: float = 0.01,
        infrasonic_power_threshold: float = 0.05,
        entropy_anomaly_low: float = 1.0,
        entropy_anomaly_high: float = 12.0,
    ):
        self._ultrasonic_hz = ultrasonic_threshold_hz
        self._infrasonic_hz = infrasonic_threshold_hz
        self._ultrasonic_power = ultrasonic_power_threshold
        self._infrasonic_power = infrasonic_power_threshold
        self._entropy_low = entropy_anomaly_low
        self._entropy_high = entropy_anomaly_high

    def analyze(self, audio_bytes: bytes) -> SpectralAnalysisResult:
        """Perform spectral analysis on audio bytes (WAV format).

        Non-WAV audio returns a clean result with details explaining
        the format limitation.
        """
        if not audio_bytes or len(audio_bytes) < 44:
            return _error_result("audio too short")

        parsed = _parse_wav_samples(audio_bytes)
        if parsed is None:
            return _error_result("failed to parse WAV")

        samples, sample_rate = parsed

        if len(samples) == 0:
            return _error_result("empty audio")

        duration = len(samples) / sample_rate
        result = SpectralAnalysisResult(
            duration_seconds=duration,
            sample_rate=sample_rate,
        )

        # Compute FFT
        try:
            fft_result = np.fft.rfft(samples)
            power_spectrum = np.abs(fft_result) ** 2
            freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
        except Exception as e:
            return _error_result(f"FFT failed: {e}")

        total_power = power_spectrum.sum()
        if total_power < 1e-10:
            result.details = "silent audio"
            return result

        # 1. Ultrasonic check (>20kHz)
        if sample_rate > self._ultrasonic_hz * 2:
            ultrasonic_mask = freqs > self._ultrasonic_hz
            ultrasonic_power = power_spectrum[ultrasonic_mask].sum() / total_power
            if ultrasonic_power > self._ultrasonic_power:
                result.has_ultrasonic = True

        # 2. Infrasonic check (<20Hz)
        infrasonic_mask = (freqs > 0) & (freqs < self._infrasonic_hz)
        if infrasonic_mask.any():
            infrasonic_power = power_spectrum[infrasonic_mask].sum() / total_power
            if infrasonic_power > self._infrasonic_power:
                result.has_infrasonic = True

        # 3. Spectral entropy
        result.spectral_entropy = _compute_spectral_entropy(power_spectrum)

        # 4. Compute anomaly score
        anomaly_score = 0.0
        reasons = []

        # Ultrasonic is highly suspicious (humans can't hear it)
        if result.has_ultrasonic:
            anomaly_score += 0.4
            reasons.append("ultrasonic_content")

        # Infrasonic is mildly suspicious
        if result.has_infrasonic:
            anomaly_score += 0.2
            reasons.append("infrasonic_content")

        # Abnormal spectral entropy
        entropy = result.spectral_entropy
        if entropy < self._entropy_low:
            anomaly_score += 0.3
            reasons.append(f"low_entropy={entropy:.2f}")
        elif entropy > self._entropy_high:
            anomaly_score += 0.2
            reasons.append(f"high_entropy={entropy:.2f}")

        # 5. Check for spectral discontinuities (windowed analysis)
        disc_score = self._check_discontinuities(samples, sample_rate)
        if disc_score > 0.3:
            anomaly_score += disc_score * 0.3
            reasons.append(f"spectral_discontinuity={disc_score:.2f}")

        result.anomaly_score = min(anomaly_score, 1.0)
        result.suspicious = result.anomaly_score >= 0.3
        result.details = "; ".join(reasons) if reasons else "clean"

        return result

    def _check_discontinuities(
        self, samples: np.ndarray, sample_rate: int
    ) -> float:
        """Check for sudden spectral changes (adversarial bursts).

        Splits audio into windows and compares adjacent window spectra.
        Large jumps suggest embedded adversarial segments.
        """
        # Use 50ms windows
        window_size = max(int(sample_rate * 0.05), 256)
        n_windows = len(samples) // window_size

        if n_windows < 3:
            return 0.0

        # Compute per-window spectral centroid
        centroids = []
        for i in range(n_windows):
            chunk = samples[i * window_size : (i + 1) * window_size]
            fft_chunk = np.fft.rfft(chunk)
            power = np.abs(fft_chunk) ** 2
            freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)

            total = power.sum()
            if total < 1e-10:
                centroids.append(0.0)
            else:
                centroid = float(np.sum(freqs * power) / total)
                centroids.append(centroid)

        if len(centroids) < 3:
            return 0.0

        centroids = np.array(centroids)
        # Compute differences between adjacent windows
        diffs = np.abs(np.diff(centroids))
        median_diff = float(np.median(diffs))

        if median_diff < 1e-5:
            return 0.0

        # A sudden jump >5x the median is suspicious
        max_jump = float(diffs.max())
        ratio = max_jump / (median_diff + 1e-5)

        if ratio > 10:
            return min(ratio / 20.0, 1.0)
        elif ratio > 5:
            return min((ratio - 5) / 10.0, 0.5)

        return 0.0


__all__ = [
    "SpectralAnalyzer",
    "SpectralAnalysisResult",
]
