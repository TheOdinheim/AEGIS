"""
Multimodal Security Layer — Image and document scanning for AEGIS.

Core principle: extract text from images and documents via OCR, metadata
analysis, and format-aware text extraction, then feed extracted text through
the EXISTING L2/L3 text detection pipeline.

The MultimodalPreprocessor sits between L1 Barrier and L2 Innate. When a
request contains images or documents (OpenAI multimodal content format), it:
1. Decodes and validates each image / document
2. Extracts text via OCR, metadata scanning, and format-aware parsing
3. Scans for steganographic content and hidden text injection
4. Returns extracted text for L2/L3 scanning + media-specific scan results

ASSUMED-BREACH POSTURE: Images and documents are untrusted input channels.
Attackers can embed prompt injections as text in images, hide instructions
in metadata / comments / invisible CSS, use polyglot files to evade parsers,
or embed macros/scripts for code execution. This layer ensures all visual
and document content receives the same scrutiny as text content.
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
from aegis.layers.multimodal.document_scanner import DocumentScanner
from aegis.layers.multimodal.document_analyzer import DocumentAnalyzer
from aegis.layers.multimodal.text_extractor import DocumentTextExtractor, DocumentExtractionResult
from aegis.layers.multimodal.hidden_content_detector import (
    HiddenContentDetector,
    HiddenContentReport,
)
from aegis.layers.multimodal.format_validator import FormatValidator, FormatValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIME type prefixes for document content detection
# ---------------------------------------------------------------------------
_DOCUMENT_MIME_PREFIXES = (
    "application/pdf",
    "text/html",
    "text/markdown",
    "application/json",
    "text/plain",
    "application/vnd.openxmlformats",
    "application/msword",
    "application/vnd.ms-",
    "text/xml",
    "application/xml",
    "text/csv",
    "application/yaml",
    "text/yaml",
)


def _is_document_mime(mime: str) -> bool:
    """Return True if the MIME type indicates a document (not an image)."""
    lower = mime.lower()
    return any(lower.startswith(prefix) for prefix in _DOCUMENT_MIME_PREFIXES)


def _is_image_mime(mime: str) -> bool:
    """Return True if the MIME type indicates an image."""
    return mime.lower().startswith("image/")


@dataclass
class MultimodalScanReport:
    """Aggregated report from multimodal preprocessing."""
    images_found: int = 0
    images_scanned: int = 0
    documents_found: int = 0
    documents_scanned: int = 0
    image_reports: list[ImageScanReport] = field(default_factory=list)
    document_scan_results: list[ScanResult] = field(default_factory=list)
    extracted_text: str = ""
    document_extracted_text: str = ""
    scan_results: list[ScanResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        all_results = self.scan_results + self.document_scan_results
        return any(r.is_threat and r.confidence >= 0.85 for r in all_results)

    @property
    def max_confidence(self) -> float:
        all_results = self.scan_results + self.document_scan_results
        if not all_results:
            return 0.0
        return max(r.confidence for r in all_results)

    @property
    def has_images(self) -> bool:
        return self.images_found > 0

    @property
    def has_documents(self) -> bool:
        return self.documents_found > 0


class MultimodalPreprocessor:
    """Orchestrates image and document scanning and text extraction.

    Processes OpenAI multimodal content format:
    ```json
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,..."}}
    ```

    Extracts images and documents from messages, scans them through the
    appropriate security pipeline, and returns extracted text for L2/L3
    text scanning.

    When disabled or no media present: zero overhead.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_image_size_mb: float = 20.0,
        ocr_enabled: bool = True,
        steg_enabled: bool = True,
        max_images_per_request: int = 10,
        document_scanning_enabled: bool = True,
        document_max_size_mb: float = 50.0,
        document_block_macros: bool = True,
        document_block_scripts: bool = True,
        regex_engine: object | None = None,
        injection_classifier: object | None = None,
        semantic_engine: object | None = None,
    ):
        self._enabled = enabled
        self._max_images = max_images_per_request
        self._document_scanning_enabled = document_scanning_enabled

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

        if document_scanning_enabled:
            self._doc_scanner = DocumentScanner(
                regex_engine=regex_engine,
                max_document_size_mb=document_max_size_mb,
                block_macros=document_block_macros,
                block_scripts=document_block_scripts,
            )
            self._doc_analyzer = DocumentAnalyzer(
                injection_classifier=injection_classifier,
                semantic_engine=semantic_engine,
            )
        else:
            self._doc_scanner = None
            self._doc_analyzer = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def document_scanning_enabled(self) -> bool:
        return self._document_scanning_enabled

    def preprocess_messages(self, messages: list[dict]) -> MultimodalScanReport:
        """Scan all images in message content arrays.

        Args:
            messages: List of OpenAI chat message dicts. Content may be
                      string (text-only) or list of content parts
                      (multimodal with image_url entries).

        Returns:
            MultimodalScanReport with extracted text and scan results.
        """
        if not self._enabled and not self._document_scanning_enabled:
            return MultimodalScanReport()

        start = time.perf_counter()
        report = MultimodalScanReport()

        # Extract image data from messages
        if self._enabled:
            images = self._extract_images(messages)
            report.images_found = len(images)

            # Limit images per request
            if images and len(images) > self._max_images:
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

                    if img_report.extracted_text:
                        text_parts.append(img_report.extracted_text)

            report.extracted_text = "\n".join(text_parts)

        # Extract and scan documents
        if self._document_scanning_enabled:
            documents = self._extract_documents(messages)
            report.documents_found = len(documents)

        report.total_latency_ms = (time.perf_counter() - start) * 1000
        return report

    async def scan_documents(
        self,
        messages: list[dict],
    ) -> list[ScanResult]:
        """L2 fast-path document scanning (async).

        Extracts base64-encoded documents from messages, runs each through
        the DocumentScanner pipeline, and returns scan results.
        """
        if not self._document_scanning_enabled or not self._doc_scanner:
            return []

        documents = self._extract_documents(messages)
        if not documents:
            return []

        results: list[ScanResult] = []
        for doc_bytes, filename, content_type in documents:
            result = await self._doc_scanner.scan(
                file_bytes=doc_bytes,
                filename=filename,
                content_type=content_type,
            )
            results.append(result)
        return results

    async def analyze_documents(
        self,
        messages: list[dict],
    ) -> list[ScanResult]:
        """L3 slow-path deep document analysis (async).

        Runs each document through DocumentAnalyzer (ML classification,
        semantic search, structural analysis).
        """
        if not self._document_scanning_enabled or not self._doc_analyzer:
            return []

        documents = self._extract_documents(messages)
        if not documents:
            return []

        results: list[ScanResult] = []
        for doc_bytes, filename, content_type in documents:
            result = await self._doc_analyzer.analyze(
                file_bytes=doc_bytes,
                filename=filename,
                content_type=content_type,
            )
            results.append(result)
        return results

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
        """Extract base64-encoded image data from OpenAI multimodal messages."""
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

                # Only process base64 data URIs for images
                if url.startswith("data:image/"):
                    try:
                        _, b64_data = url.split(",", 1)
                        image_bytes = base64.b64decode(b64_data)
                        images.append(image_bytes)
                    except Exception as e:
                        logger.warning("Failed to decode base64 image: %s", e)
                        continue

        return images

    def _extract_documents(self, messages: list[dict]) -> list[tuple[bytes, str, str]]:
        """Extract base64-encoded document data from multimodal messages.

        Returns a list of ``(file_bytes, filename, content_type)`` tuples.

        Handles:
        - ``{"type": "image_url", "image_url": {"url": "data:application/pdf;base64,..."}}``
        - ``{"type": "file", "file": {"data": "base64...", "mime_type": "...", "name": "..."}}``
        - Any data URI with a document MIME type
        """
        documents: list[tuple[bytes, str, str]] = []

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue

                # Handle data URIs in image_url (reused for documents by some clients)
                if part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    if not isinstance(image_url, dict):
                        continue
                    url = image_url.get("url", "")
                    if not isinstance(url, str) or not url.startswith("data:"):
                        continue

                    # Parse "data:application/pdf;base64,..."
                    try:
                        header, b64_data = url.split(",", 1)
                    except ValueError:
                        continue

                    # Extract MIME type from "data:application/pdf;base64"
                    mime_part = header.replace("data:", "").split(";")[0].strip()

                    # Skip images — those are handled by _extract_images
                    if _is_image_mime(mime_part):
                        continue

                    if _is_document_mime(mime_part):
                        try:
                            doc_bytes = base64.b64decode(b64_data)
                            documents.append((doc_bytes, "", mime_part))
                        except Exception as e:
                            logger.warning("Failed to decode base64 document: %s", e)

                # Handle explicit file attachments
                elif part.get("type") == "file":
                    file_data = part.get("file", {})
                    if not isinstance(file_data, dict):
                        continue
                    b64 = file_data.get("data", "")
                    mime = file_data.get("mime_type", "")
                    name = file_data.get("name", "")
                    if not b64:
                        continue
                    try:
                        doc_bytes = base64.b64decode(b64)
                        documents.append((doc_bytes, name, mime))
                    except Exception as e:
                        logger.warning("Failed to decode file attachment '%s': %s", name, e)

        return documents


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
    "DocumentScanner",
    "DocumentAnalyzer",
    "DocumentTextExtractor",
    "DocumentExtractionResult",
    "HiddenContentDetector",
    "HiddenContentReport",
    "FormatValidator",
    "FormatValidationResult",
]
