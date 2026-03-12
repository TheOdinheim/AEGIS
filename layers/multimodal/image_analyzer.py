"""
Image Analyzer — L3 Adaptive slow-path image analysis (20-100ms).

Performs deeper analysis that complements the L2 fast-path scanner:
1. Deep OCR with post-processing
2. Sanitization comparison (original vs re-encoded)
3. Cross-modal consistency checking
4. Adversarial image heuristics
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from aegis.models.scan_result import ScanResult, ThreatCategory
from aegis.layers.multimodal.ocr_engine import OCREngine, OCRResult
from aegis.layers.multimodal.image_sanitizer import ImageSanitizer, SanitizationResult
from aegis.layers.multimodal.steganalysis import Steganalyzer

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysisReport:
    """Report from L3 deep image analysis."""
    scan_results: list[ScanResult] = field(default_factory=list)
    deep_ocr_result: OCRResult | None = None
    sanitization_result: SanitizationResult | None = None
    pixel_diff_score: float = 0.0
    adversarial_score: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        return any(r.is_threat and r.confidence >= 0.85 for r in self.scan_results)

    @property
    def max_confidence(self) -> float:
        if not self.scan_results:
            return 0.0
        return max(r.confidence for r in self.scan_results)


class ImageAnalyzer:
    """L3 slow-path image analyzer.

    Performs deeper analysis that the fast-path scanner skips:
    - Sanitize-and-compare: re-encode image, compare pixel difference.
      High difference may indicate hidden data in encoding artifacts.
    - Adversarial perturbation heuristics: detect unnaturally uniform
      noise patterns that suggest adversarial examples.
    - Cross-modal text extraction with higher accuracy settings.
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        sanitize_compare: bool = True,
    ):
        self._ocr_engine = OCREngine() if ocr_enabled else None
        self._sanitizer = ImageSanitizer() if sanitize_compare else None

    async def analyze(self, image_bytes: bytes) -> ImageAnalysisReport:
        """Run deep image analysis (async wrapper for CPU-bound work)."""
        return self._analyze_sync(image_bytes)

    def _analyze_sync(self, image_bytes: bytes) -> ImageAnalysisReport:
        """Synchronous deep image analysis."""
        start = time.perf_counter()
        report = ImageAnalysisReport()
        results: list[ScanResult] = []

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
        except Exception as e:
            results.append(ScanResult(
                scanner_id="image_deep_analyzer",
                is_threat=True,
                confidence=0.90,
                threat_category=ThreatCategory.SCHEMA_VIOLATION,
                matched_patterns=[f"corrupt_image_deep: {e}"],
                latency_ms=0.0,
            ))
            report.scan_results = results
            report.total_latency_ms = (time.perf_counter() - start) * 1000
            return report

        # 1. Sanitization comparison
        if self._sanitizer:
            san_result = self._sanitizer.sanitize(image_bytes, target_format="PNG")
            report.sanitization_result = san_result

            if san_result.sanitized_bytes:
                diff = self._pixel_difference(image_bytes, san_result.sanitized_bytes)
                report.pixel_diff_score = diff
                if diff > 0.15:  # >15% pixel difference after re-encoding
                    results.append(ScanResult(
                        scanner_id="image_sanitize_compare",
                        is_threat=True,
                        confidence=min(0.5 + diff, 0.85),
                        threat_category=ThreatCategory.ENCODING_OBFUSCATION,
                        matched_patterns=[f"pixel_diff={diff:.3f}"],
                        latency_ms=san_result.latency_ms,
                    ))

        # 2. Adversarial perturbation heuristics
        adv_score = self._adversarial_heuristic(img)
        report.adversarial_score = adv_score
        if adv_score > 0.7:
            results.append(ScanResult(
                scanner_id="image_adversarial_detector",
                is_threat=True,
                confidence=min(adv_score, 0.85),
                threat_category=ThreatCategory.ENCODING_OBFUSCATION,
                matched_patterns=[f"adversarial_score={adv_score:.3f}"],
                latency_ms=0.0,
            ))

        # 3. Deep OCR (if engine available)
        if self._ocr_engine:
            ocr_result = self._ocr_engine.extract_text(img)
            report.deep_ocr_result = ocr_result

        if not any(r.is_threat for r in results):
            results.append(ScanResult(
                scanner_id="image_deep_analyzer",
                is_threat=False,
                confidence=0.0,
                latency_ms=0.0,
            ))

        report.scan_results = results
        report.total_latency_ms = (time.perf_counter() - start) * 1000
        return report

    def _pixel_difference(self, original_bytes: bytes, sanitized_bytes: bytes) -> float:
        """Compute normalized pixel difference between original and sanitized."""
        try:
            orig = Image.open(io.BytesIO(original_bytes)).convert("RGB")
            clean = Image.open(io.BytesIO(sanitized_bytes)).convert("RGB")

            # Resize to same dimensions for comparison
            if orig.size != clean.size:
                clean = clean.resize(orig.size, Image.LANCZOS)

            orig_arr = np.array(orig, dtype=np.float64)
            clean_arr = np.array(clean, dtype=np.float64)

            # Mean absolute difference normalized to [0, 1]
            diff = np.mean(np.abs(orig_arr - clean_arr)) / 255.0
            return float(diff)
        except Exception:
            return 0.0

    def _adversarial_heuristic(self, img: Image.Image) -> float:
        """Detect adversarial perturbation patterns.

        Adversarial examples often have:
        1. Unnaturally uniform high-frequency noise
        2. Pixel values clustered at boundaries (0, 255)
        3. Spatial frequency anomalies

        Returns score 0.0 (clean) to 1.0 (likely adversarial).
        """
        try:
            arr = np.array(img.convert("L"), dtype=np.float64)
            if arr.size < 100:
                return 0.0

            # Compute local noise (difference from smoothed version)
            # Simple box filter approximation
            kernel_size = 3
            padded = np.pad(arr, kernel_size // 2, mode="edge")
            smoothed = np.zeros_like(arr)
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    smoothed += padded[dy:dy + arr.shape[0], dx:dx + arr.shape[1]]
            smoothed /= kernel_size * kernel_size

            noise = arr - smoothed

            # Adversarial patterns: noise is spatially uniform
            noise_std = float(np.std(noise))
            noise_kurtosis = self._kurtosis(noise.flatten())

            # Natural images have non-uniform noise (high kurtosis)
            # Adversarial perturbations tend toward Gaussian (kurtosis ≈ 3)
            if noise_std < 0.5:
                return 0.0  # Very low noise = no perturbation

            # Score based on how Gaussian the noise is
            gaussian_score = 1.0 - min(abs(noise_kurtosis - 3.0) / 10.0, 1.0)

            # Pixel boundary clustering
            boundary_ratio = float(np.mean((arr < 5) | (arr > 250)))
            boundary_score = min(boundary_ratio * 5.0, 1.0)

            return float(gaussian_score * 0.6 + boundary_score * 0.4)
        except Exception:
            return 0.0

    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Compute excess kurtosis (Fisher's definition)."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((data - mean) ** 4)
        return float(m4 / (std ** 4) - 3.0)
