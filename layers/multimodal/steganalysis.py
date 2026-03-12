"""
Steganalysis — Detect steganographic content hidden in images.

Implements two detection methods:
1. Chi-square analysis on LSB plane distribution
2. RS (Regular/Singular) analysis for LSB embedding detection

Both methods detect statistical anomalies in pixel values that indicate
data has been hidden in the least significant bits.

Uses Pillow + numpy only — no external steg libraries required.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SteganalysisResult:
    """Result of steganographic analysis."""
    chi_square_score: float = 0.0
    chi_square_suspicious: bool = False
    rs_score: float = 0.0
    rs_suspicious: bool = False
    overall_suspicious: bool = False
    details: str = ""
    latency_ms: float = 0.0


class Steganalyzer:
    """Detect steganographic content in images using statistical analysis.

    Chi-square threshold: images with uniform LSB distributions (chi-square
    p-value > 0.95) are suspicious — natural images have non-uniform LSBs.

    RS threshold: Regular/Singular group ratio significantly different from
    expected indicates LSB replacement steganography.
    """

    def __init__(
        self,
        chi_square_threshold: float = 0.95,
        rs_threshold: float = 0.1,
    ):
        self._chi_threshold = chi_square_threshold
        self._rs_threshold = rs_threshold

    def analyze(self, img: Image.Image) -> SteganalysisResult:
        """Run both chi-square and RS analysis on an image."""
        start = time.perf_counter()

        # Convert to numpy array
        try:
            arr = np.array(img)
        except Exception as e:
            logger.warning("Failed to convert image to array: %s", e)
            return SteganalysisResult(
                details=f"Analysis failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        if arr.ndim < 2:
            return SteganalysisResult(
                details="Image too small for analysis",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Use first channel if color, or grayscale
        if arr.ndim == 3:
            channel = arr[:, :, 0].flatten()
        else:
            channel = arr.flatten()

        chi_score = self._chi_square_lsb(channel)
        chi_suspicious = chi_score > self._chi_threshold

        rs_score = self._rs_analysis(channel)
        rs_suspicious = abs(rs_score) > self._rs_threshold

        overall = chi_suspicious or rs_suspicious

        details_parts = []
        if chi_suspicious:
            details_parts.append(
                f"Chi-square LSB score {chi_score:.3f} exceeds threshold {self._chi_threshold}"
            )
        if rs_suspicious:
            details_parts.append(
                f"RS analysis score {rs_score:.3f} exceeds threshold {self._rs_threshold}"
            )

        return SteganalysisResult(
            chi_square_score=chi_score,
            chi_square_suspicious=chi_suspicious,
            rs_score=rs_score,
            rs_suspicious=rs_suspicious,
            overall_suspicious=overall,
            details="; ".join(details_parts) if details_parts else "No steganographic indicators",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def _chi_square_lsb(self, pixels: np.ndarray) -> float:
        """Chi-square test on LSB distribution.

        Natural images have non-uniform LSB distributions. LSB steganography
        makes the distribution more uniform. A high chi-square p-value
        (close to 1.0) indicates suspiciously uniform LSBs.

        Returns a score from 0.0 to 1.0 (approximate p-value).
        """
        if len(pixels) < 100:
            return 0.0

        # Count pairs of values (2k, 2k+1) — LSB steganography equalizes these
        max_val = int(pixels.max())
        if max_val < 2:
            return 0.0

        hist = np.bincount(pixels, minlength=max_val + 1).astype(np.float64)

        chi2 = 0.0
        num_pairs = 0

        for k in range(0, len(hist) - 1, 2):
            expected = (hist[k] + hist[k + 1]) / 2.0
            if expected > 0:
                chi2 += ((hist[k] - expected) ** 2) / expected
                chi2 += ((hist[k + 1] - expected) ** 2) / expected
                num_pairs += 1

        if num_pairs == 0:
            return 0.0

        # Approximate p-value using standard normal approximation
        # For large degrees of freedom, (chi2 - df) / sqrt(2*df) ~ N(0,1)
        df = float(num_pairs)
        if df < 1:
            return 0.0

        z = (chi2 - df) / max(np.sqrt(2.0 * df), 1e-10)

        # Convert z-score to approximate p-value
        # Low chi2 (z < 0) = uniform distribution = suspicious
        # We want p-value of chi2 being THIS LOW (one-sided)
        # Using normal CDF approximation
        p_value = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))

        # Invert: we want high score when LSBs are suspiciously uniform
        # Low chi2 → low p_value from standard test → suspicious
        return 1.0 - p_value

    def _rs_analysis(self, pixels: np.ndarray) -> float:
        """RS (Regular/Singular) analysis for LSB replacement detection.

        Divides image into groups, applies flipping mask, counts regular (R)
        and singular (S) groups. For clean images, R ≈ R_neg and S ≈ S_neg.
        For stego images, R > R_neg and S < S_neg.

        Returns: estimated embedding rate (0.0 = clean, positive = stego).
        """
        n = len(pixels)
        group_size = 4
        if n < group_size * 10:
            return 0.0

        # Trim to multiple of group_size
        n_groups = n // group_size
        trimmed = pixels[:n_groups * group_size].reshape(n_groups, group_size).astype(np.float64)

        def discrimination(group: np.ndarray) -> float:
            """Sum of absolute differences between adjacent elements."""
            return float(np.sum(np.abs(np.diff(group))))

        def flip_lsb(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
            """Apply LSB flipping according to mask. +1: flip, 0: keep, -1: flip and negate."""
            result = values.copy()
            for i, m in enumerate(mask):
                if m == 1:
                    result[i] = values[i] ^ 1  # flip LSB
                elif m == -1:
                    result[i] = (values[i] ^ 1) if values[i] % 2 == 0 else (values[i] ^ 1)
            return result

        # Mask pattern [0, 1, 1, 0] — standard RS mask
        mask = np.array([0, 1, 1, 0])
        neg_mask = np.array([0, -1, -1, 0])

        r_m, s_m, r_neg, s_neg = 0, 0, 0, 0
        sample_size = min(n_groups, 5000)  # Limit for performance

        for i in range(sample_size):
            group = trimmed[i]
            d_orig = discrimination(group)

            flipped = flip_lsb(group.astype(np.int64), mask)
            d_flipped = discrimination(flipped)

            neg_flipped = flip_lsb(group.astype(np.int64), neg_mask)
            d_neg = discrimination(neg_flipped)

            if d_flipped > d_orig:
                r_m += 1
            elif d_flipped < d_orig:
                s_m += 1

            if d_neg > d_orig:
                r_neg += 1
            elif d_neg < d_orig:
                s_neg += 1

        if sample_size == 0:
            return 0.0

        # Embedding rate estimation
        # For clean images: r_m ≈ r_neg, s_m ≈ s_neg
        # For stego: r_m > r_neg, s_m < s_neg
        r_diff = (r_m - r_neg) / sample_size
        s_diff = (s_neg - s_m) / sample_size

        return (r_diff + s_diff) / 2.0


def _erf(x: float) -> float:
    """Approximate error function (Abramowitz and Stegun approximation)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)
    return sign * y
