"""
Multimodal Security Layer — Image scanning and text extraction for AEGIS.

Core principle: extract text from images via OCR and metadata analysis,
then feed extracted text through the EXISTING L2/L3 text detection pipeline.

The MultimodalPreprocessor sits between L1 Barrier and L2 Innate. When a
request contains images (OpenAI multimodal content format), it:
1. Decodes and validates each image
2. Extracts text via OCR and metadata scanning
3. Scans for steganographic content
4. Returns extracted text for L2/L3 scanning + image-specific scan results

ASSUMED-BREACH POSTURE: Images are an untrusted input channel. Attackers
can embed prompt injections as text in images, hide instructions in metadata,
or use steganography to evade text-only scanners. This layer ensures visual
content receives the same scrutiny as text content.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field

from aegis.models.scan_result import ScanResult, ThreatCategory

from aegis.layers.multimodal.image_scanner import ImageScanner, ImageScanReport
from aegis.layers.multimodal.image_analyzer import ImageAnalyzer, ImageAnalysisReport
from aegis.layers.multimodal.ocr_engine import OCREngine, OCRResult
from aegis.layers.multimodal.image_sanitizer import ImageSanitizer, SanitizationResult
from aegis.layers.multimodal.steganalysis import Steganalyzer, SteganalysisResult
from aegis.layers.multimodal.metadata_stripper import MetadataStripper, MetadataReport

logger = logging.getLogger(__name__)


@dataclass
class MultimodalScanReport:
    """Aggregated report from multimodal preprocessing."""
    images_found: int = 0
    images_scanned: int = 0
    image_reports: list[ImageScanReport] = field(default_factory=list)
    extracted_text: str = ""
    scan_results: list[ScanResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        return any(r.is_threat and r.confidence >= 0.85 for r in self.scan_results)

    @property
    def max_confidence(self) -> float:
        if not self.scan_results:
            return 0.0
        return max(r.confidence for r in self.scan_results)

    @property
    def has_images(self) -> bool:
        return self.images_found > 0


class MultimodalPreprocessor:
    """Orchestrates image scanning and text extraction.

    Processes OpenAI multimodal content format:
    ```json
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ```

    Extracts images from messages, scans them through the image security
    pipeline, and returns extracted text for L2/L3 text scanning.

    When disabled or no images present: zero overhead.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_image_size_mb: float = 20.0,
        ocr_enabled: bool = True,
        steg_enabled: bool = True,
        max_images_per_request: int = 10,
    ):
        self._enabled = enabled
        self._max_images = max_images_per_request

        if enabled:
            self._scanner = ImageScanner(
                max_image_size_mb=max_image_size_mb,
                ocr_enabled=ocr_enabled,
                steg_enabled=steg_enabled,
            )
            self._analyzer = ImageAnalyzer(ocr_enabled=ocr_enabled)
        else:
            self._scanner = None
            self._analyzer = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def preprocess_messages(self, messages: list[dict]) -> MultimodalScanReport:
        """Scan all images in message content arrays.

        Args:
            messages: List of OpenAI chat message dicts. Content may be
                      string (text-only) or list of content parts
                      (multimodal with image_url entries).

        Returns:
            MultimodalScanReport with extracted text and scan results.
        """
        if not self._enabled:
            return MultimodalScanReport()

        start = time.perf_counter()
        report = MultimodalScanReport()

        # Extract image data from messages
        images = self._extract_images(messages)
        report.images_found = len(images)

        if not images:
            report.total_latency_ms = (time.perf_counter() - start) * 1000
            return report

        # Limit images per request
        if len(images) > self._max_images:
            report.scan_results.append(ScanResult(
                scanner_id="multimodal_preprocessor",
                is_threat=True,
                confidence=0.80,
                threat_category=ThreatCategory.TOKEN_ANOMALY,
                matched_patterns=[f"too_many_images: {len(images)} > {self._max_images}"],
                latency_ms=0.0,
            ))
            images = images[:self._max_images]

        # Scan each image
        text_parts: list[str] = []
        for img_data in images:
            if self._scanner:
                img_report = self._scanner.scan_image_bytes(img_data)
                report.image_reports.append(img_report)
                report.scan_results.extend(img_report.scan_results)
                report.images_scanned += 1

                # Collect extracted text
                if img_report.extracted_text:
                    text_parts.append(img_report.extracted_text)

        report.extracted_text = "\n".join(text_parts)
        report.total_latency_ms = (time.perf_counter() - start) * 1000
        return report

    async def analyze_images(self, messages: list[dict]) -> ImageAnalysisReport | None:
        """L3 slow-path deep image analysis (async)."""
        if not self._enabled or not self._analyzer:
            return None

        images = self._extract_images(messages)
        if not images:
            return None

        # Analyze first image (most likely attack vector)
        return await self._analyzer.analyze(images[0])

    def _extract_images(self, messages: list[dict]) -> list[bytes]:
        """Extract base64-encoded image data from OpenAI multimodal messages.

        Handles both:
        - {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}
        - {"content": [{"type": "image_url", "image_url": {"url": "https://..."}}]}
          (URL images are skipped — we only process inline base64)
        """
        images: list[bytes] = []

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "image_url":
                    continue

                image_url = part.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    continue

                if not isinstance(url, str):
                    continue

                # Only process base64 data URIs
                if url.startswith("data:image/"):
                    try:
                        # data:image/png;base64,iVBORw0KG...
                        _, b64_data = url.split(",", 1)
                        image_bytes = base64.b64decode(b64_data)
                        images.append(image_bytes)
                    except Exception as e:
                        logger.warning("Failed to decode base64 image: %s", e)
                        continue

        return images


__all__ = [
    "MultimodalPreprocessor",
    "MultimodalScanReport",
    "ImageScanner",
    "ImageScanReport",
    "ImageAnalyzer",
    "ImageAnalysisReport",
    "OCREngine",
    "OCRResult",
    "ImageSanitizer",
    "SanitizationResult",
    "Steganalyzer",
    "SteganalysisResult",
    "MetadataStripper",
    "MetadataReport",
]
