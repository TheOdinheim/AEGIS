"""
Image Scanner — L2 Innate fast-path image scanning (<10ms target).

Performs format validation, metadata extraction, OCR text extraction
for L2/L3 text pipeline, steganalysis, and perceptual hashing.

Returns ScanResult objects per the existing L2 contract.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from aegis.models.scan_result import ScanResult, ThreatCategory
from aegis.layers.multimodal.ocr_engine import OCREngine, OCRResult
from aegis.layers.multimodal.metadata_stripper import MetadataStripper, MetadataReport
from aegis.layers.multimodal.steganalysis import Steganalyzer, SteganalysisResult

logger = logging.getLogger(__name__)

# Magic bytes for image format validation
_FORMAT_SIGNATURES: dict[str, bytes] = {
    "PNG": b"\x89PNG",
    "JPEG": b"\xff\xd8\xff",
    "GIF": b"GIF8",
    "BMP": b"BM",
}
# WebP: RIFF....WEBP (bytes 0-3 = RIFF, bytes 8-11 = WEBP)
_WEBP_RIFF = b"RIFF"
_WEBP_TAG = b"WEBP"


@dataclass
class ImageScanReport:
    """Aggregated report from image scanning."""
    scan_results: list[ScanResult] = field(default_factory=list)
    ocr_result: OCRResult | None = None
    metadata_report: MetadataReport | None = None
    steg_result: SteganalysisResult | None = None
    perceptual_hash: str = ""
    detected_format: str = ""
    image_dimensions: tuple[int, int] = (0, 0)
    total_latency_ms: float = 0.0

    @property
    def extracted_text(self) -> str:
        """All text extracted from the image (OCR + metadata)."""
        parts = []
        if self.ocr_result and self.ocr_result.text:
            parts.append(self.ocr_result.text)
        if self.metadata_report and self.metadata_report.text_found_in_metadata:
            parts.append(self.metadata_report.text_found_in_metadata)
        return "\n".join(parts)

    @property
    def should_block(self) -> bool:
        return any(r.is_threat and r.confidence >= 0.85 for r in self.scan_results)

    @property
    def max_confidence(self) -> float:
        if not self.scan_results:
            return 0.0
        return max(r.confidence for r in self.scan_results)


class ImageScanner:
    """L2 fast-path image scanner.

    Validates format, extracts metadata text, runs OCR for text extraction,
    performs steganalysis, and computes perceptual hash.

    All extracted text is returned for feeding through the existing
    L2/L3 text detection pipeline.
    """

    def __init__(
        self,
        max_image_size_mb: float = 20.0,
        ocr_enabled: bool = True,
        steg_enabled: bool = True,
    ):
        self._max_size_bytes = int(max_image_size_mb * 1024 * 1024)
        self._ocr_engine = OCREngine() if ocr_enabled else None
        self._metadata_stripper = MetadataStripper()
        self._steganalyzer = Steganalyzer() if steg_enabled else None

    def scan_image_bytes(self, image_bytes: bytes) -> ImageScanReport:
        """Scan raw image bytes through all image security checks."""
        start = time.perf_counter()
        report = ImageScanReport()
        results: list[ScanResult] = []

        # 1. Size validation
        if len(image_bytes) > self._max_size_bytes:
            results.append(ScanResult(
                scanner_id="image_size_guard",
                is_threat=True,
                confidence=0.9,
                threat_category=ThreatCategory.TOKEN_ANOMALY,
                matched_patterns=["oversized_image"],
                latency_ms=0.0,
            ))
            report.scan_results = results
            report.total_latency_ms = (time.perf_counter() - start) * 1000
            return report

        # 2. Format validation
        fmt = self._detect_format(image_bytes)
        report.detected_format = fmt
        if not fmt:
            results.append(ScanResult(
                scanner_id="image_format_validator",
                is_threat=True,
                confidence=0.95,
                threat_category=ThreatCategory.SCHEMA_VIOLATION,
                matched_patterns=["unknown_image_format"],
                latency_ms=0.0,
            ))
            report.scan_results = results
            report.total_latency_ms = (time.perf_counter() - start) * 1000
            return report

        # 3. Open image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()  # Force full decode to catch truncated/corrupt images
            report.image_dimensions = img.size
        except Exception as e:
            results.append(ScanResult(
                scanner_id="image_format_validator",
                is_threat=True,
                confidence=0.90,
                threat_category=ThreatCategory.SCHEMA_VIOLATION,
                matched_patterns=[f"corrupt_image: {e}"],
                latency_ms=0.0,
            ))
            report.scan_results = results
            report.total_latency_ms = (time.perf_counter() - start) * 1000
            return report

        # 4. Metadata extraction (text from EXIF/XMP for scanning)
        meta_report = self._metadata_stripper.extract_metadata_text(img)
        report.metadata_report = meta_report
        if meta_report.text_found_in_metadata:
            results.append(ScanResult(
                scanner_id="image_metadata_scanner",
                is_threat=False,  # Text extracted, not yet scanned
                confidence=0.0,
                matched_patterns=[f"metadata_text_found: {len(meta_report.fields_stripped)} fields"],
                latency_ms=meta_report.latency_ms,
            ))

        # 5. OCR text extraction
        if self._ocr_engine:
            ocr_result = self._ocr_engine.extract_text(img)
            report.ocr_result = ocr_result
            if ocr_result.text:
                results.append(ScanResult(
                    scanner_id="image_ocr_scanner",
                    is_threat=False,  # Text extracted, will be scanned by L2/L3
                    confidence=0.0,
                    matched_patterns=[f"ocr_text_extracted: {ocr_result.word_count} words via {ocr_result.engine_used}"],
                    latency_ms=ocr_result.latency_ms,
                ))

        # 6. Steganalysis
        if self._steganalyzer:
            steg_result = self._steganalyzer.analyze(img)
            report.steg_result = steg_result
            if steg_result.overall_suspicious:
                results.append(ScanResult(
                    scanner_id="image_steganalysis",
                    is_threat=True,
                    confidence=0.70,  # Steg detection has false positives
                    threat_category=ThreatCategory.ENCODING_OBFUSCATION,
                    matched_patterns=[steg_result.details],
                    latency_ms=steg_result.latency_ms,
                ))

        # 7. Perceptual hash
        try:
            phash = self._compute_phash(img)
            report.perceptual_hash = phash
        except Exception:
            pass

        # Add clean result if no threats found
        if not any(r.is_threat for r in results):
            results.append(ScanResult(
                scanner_id="image_scanner",
                is_threat=False,
                confidence=0.0,
                latency_ms=0.0,
            ))

        report.scan_results = results
        report.total_latency_ms = (time.perf_counter() - start) * 1000
        return report

    def _detect_format(self, data: bytes) -> str:
        """Detect image format from magic bytes."""
        if len(data) < 12:
            return ""
        for fmt, sig in _FORMAT_SIGNATURES.items():
            if data[:len(sig)] == sig:
                return fmt
        # WebP check: RIFF....WEBP
        if data[:4] == _WEBP_RIFF and data[8:12] == _WEBP_TAG:
            return "WEBP"
        return ""

    def _compute_phash(self, img: Image.Image) -> str:
        """Compute perceptual hash (pHash) of image.

        1. Resize to 32x32
        2. Convert to grayscale
        3. Apply DCT
        4. Take top-left 8x8 of DCT
        5. Threshold at median → 64-bit hash
        """
        # Resize to 32x32 grayscale
        resized = img.convert("L").resize((32, 32), Image.LANCZOS)
        arr = np.array(resized, dtype=np.float64)

        # Simple DCT approximation via matrix multiply
        # Full DCT would use scipy, so we use a simplified approach
        # that still produces a useful perceptual hash
        dct = self._dct2d(arr)

        # Take top-left 8x8 (low frequency components)
        low_freq = dct[:8, :8]

        # Threshold at median (excluding DC component)
        flat = low_freq.flatten()
        median_val = np.median(flat[1:])  # Skip DC
        bits = (flat > median_val).astype(np.uint8)

        # Convert to hex string
        hash_int = 0
        for bit in bits:
            hash_int = (hash_int << 1) | int(bit)
        return f"{hash_int:016x}"

    @staticmethod
    def _dct2d(arr: np.ndarray) -> np.ndarray:
        """Simplified 2D DCT using row/column 1D DCT."""
        n = arr.shape[0]
        # Create DCT matrix
        dct_matrix = np.zeros((n, n))
        for k in range(n):
            for i in range(n):
                dct_matrix[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        # Apply row then column
        result = dct_matrix @ arr @ dct_matrix.T
        return result
