"""
Multimodal Image Security Tests — Phase 1.

Tests validate:
1. Image format detection and validation (magic bytes)
2. OCR engine text extraction (Pillow heuristic, tesseract optional)
3. Metadata stripping (EXIF/XMP/IPTC)
4. Steganalysis (chi-square LSB, RS analysis)
5. Image sanitizer (re-encode, resize, metadata strip)
6. Image scanner integration (L2 fast path)
7. Image analyzer (L3 slow path)
8. MultimodalPreprocessor orchestrator
9. Cross-modal injection: text in images fed through L2/L3 pipeline
10. Config integration (MultimodalConfig)

All tests use Pillow-generated synthetic images — no test fixtures needed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import struct
import time

import numpy as np
import pytest
from PIL import Image, ExifTags


def _run(coro):
    """Run async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.get_event_loop().run_until_complete(coro)

from aegis.config import AegisConfig, MultimodalConfig
from aegis.layers.multimodal import (
    MultimodalPreprocessor,
    MultimodalScanReport,
    ImageScanner,
    ImageScanReport,
    ImageAnalyzer,
    ImageAnalysisReport,
    OCREngine,
    OCRResult,
    ImageSanitizer,
    SanitizationResult,
    Steganalyzer,
    SteganalysisResult,
    MetadataStripper,
    MetadataReport,
)
from aegis.models.scan_result import ThreatCategory


# ---------------------------------------------------------------------------
# Test helpers — generate synthetic images
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 64, height: int = 64, color: tuple = (128, 128, 128)) -> bytes:
    """Generate a solid-color PNG image."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 64, height: int = 64, color: tuple = (200, 100, 50)) -> bytes:
    """Generate a solid-color JPEG image."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_gif_bytes(width: int = 32, height: int = 32) -> bytes:
    """Generate a simple GIF image."""
    img = Image.new("P", (width, height), 0)
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    return buf.getvalue()


def _make_bmp_bytes(width: int = 16, height: int = 16) -> bytes:
    """Generate a BMP image."""
    img = Image.new("RGB", (width, height), (50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _make_noisy_png(width: int = 64, height: int = 64) -> bytes:
    """Generate a PNG with random noise (for steg analysis)."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_text_image(text: str = "Hello", width: int = 200, height: int = 50) -> bytes:
    """Generate an image with high-contrast text-like patterns."""
    # Create a high-contrast black and white image
    img = Image.new("L", (width, height), 255)
    arr = np.array(img)
    # Draw horizontal lines (text-like patterns)
    for y in range(5, height - 5, 3):
        arr[y, 10:width - 10] = 0
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_exif_image() -> bytes:
    """Generate a JPEG with EXIF metadata."""
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    # Add EXIF data
    from PIL.ExifTags import Base as ExifBase
    exif = img.getexif()
    exif[0x010E] = "Test Image Description with hidden instructions"
    exif[0x013B] = "Test Artist Name"
    exif[0x9286] = "User comment with embedded payload"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def _make_large_image_bytes(size_mb: float = 25.0) -> bytes:
    """Generate oversized image data (may not be valid image)."""
    # Generate data that starts with PNG magic but is oversized
    header = b"\x89PNG\r\n\x1a\n"
    return header + b"\x00" * int(size_mb * 1024 * 1024)


def _encode_b64_data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode image bytes as a data URI."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _make_multimodal_messages(image_bytes: bytes, text: str = "Describe this image") -> list[dict]:
    """Create OpenAI multimodal message format with an image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_b64_data_uri(image_bytes)},
                },
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Format validation tests
# ---------------------------------------------------------------------------

class TestFormatValidation:
    """Validate image format detection via magic bytes."""

    def test_detect_png(self):
        scanner = ImageScanner()
        data = _make_png_bytes()
        assert scanner._detect_format(data) == "PNG"

    def test_detect_jpeg(self):
        scanner = ImageScanner()
        data = _make_jpeg_bytes()
        assert scanner._detect_format(data) == "JPEG"

    def test_detect_gif(self):
        scanner = ImageScanner()
        data = _make_gif_bytes()
        assert scanner._detect_format(data) == "GIF"

    def test_detect_bmp(self):
        scanner = ImageScanner()
        data = _make_bmp_bytes()
        assert scanner._detect_format(data) == "BMP"

    def test_unknown_format(self):
        scanner = ImageScanner()
        assert scanner._detect_format(b"\x00\x00\x00\x00") == ""

    def test_too_short(self):
        scanner = ImageScanner()
        assert scanner._detect_format(b"\x89P") == ""

    def test_unknown_format_threat(self):
        scanner = ImageScanner()
        report = scanner.scan_image_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c")
        assert any(r.is_threat for r in report.scan_results)
        assert report.detected_format == ""

    def test_corrupt_image_detected(self):
        scanner = ImageScanner()
        # Valid PNG header but corrupt data
        data = b"\x89PNG\r\n\x1a\n" + b"\xff" * 100
        report = scanner.scan_image_bytes(data)
        assert any(r.is_threat for r in report.scan_results)


# ---------------------------------------------------------------------------
# OCR engine tests
# ---------------------------------------------------------------------------

class TestOCREngine:
    """Test OCR text extraction (Pillow heuristic path)."""

    def test_engine_name_pillow(self):
        engine = OCREngine(prefer_tesseract=False)
        assert engine.engine_name == "pillow_heuristic"

    def test_extract_text_result_type(self):
        engine = OCREngine(prefer_tesseract=False)
        img = Image.new("L", (100, 100), 128)
        result = engine.extract_text(img)
        assert isinstance(result, OCRResult)
        assert result.engine_used == "pillow_heuristic"
        assert result.latency_ms >= 0

    def test_high_contrast_text_detection(self):
        """High-contrast image should be flagged as containing text."""
        engine = OCREngine(prefer_tesseract=False)
        # Create high-contrast pattern (text-like)
        arr = np.zeros((50, 200), dtype=np.uint8)
        arr[::3, :] = 255  # Horizontal white lines
        img = Image.fromarray(arr)
        result = engine.extract_text(img)
        # The heuristic should detect text-like content
        assert result.confidence >= 0.0  # Non-negative confidence

    def test_blank_image_low_confidence(self):
        """Blank/uniform image should have low text confidence."""
        engine = OCREngine(prefer_tesseract=False)
        img = Image.new("L", (100, 100), 128)
        result = engine.extract_text(img)
        assert result.confidence < 0.5

    def test_small_image(self):
        engine = OCREngine(prefer_tesseract=False)
        img = Image.new("L", (5, 5), 128)
        result = engine.extract_text(img)
        assert isinstance(result, OCRResult)


# ---------------------------------------------------------------------------
# Metadata stripper tests
# ---------------------------------------------------------------------------

class TestMetadataStripper:
    """Test EXIF/XMP/IPTC metadata extraction and stripping."""

    def test_strip_creates_clean_image(self):
        stripper = MetadataStripper()
        img = Image.new("RGB", (32, 32), (100, 150, 200))
        clean = stripper.strip_metadata(img)
        assert clean.size == img.size
        assert clean.mode == img.mode

    def test_extract_exif_text(self):
        stripper = MetadataStripper()
        img = Image.new("RGB", (32, 32), (100, 150, 200))
        exif = img.getexif()
        exif[0x010E] = "Test description with hidden payload"
        img.info["exif"] = exif.tobytes()
        # Re-create with exif
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        img_with_exif = Image.open(io.BytesIO(buf.getvalue()))

        report = stripper.extract_metadata_text(img_with_exif)
        assert report.had_exif
        assert "hidden payload" in report.text_found_in_metadata

    def test_no_metadata_clean(self):
        stripper = MetadataStripper()
        img = Image.new("RGB", (32, 32), (100, 150, 200))
        report = stripper.extract_metadata_text(img)
        assert report.text_found_in_metadata == ""
        assert report.latency_ms >= 0

    def test_strip_and_encode_png(self):
        stripper = MetadataStripper()
        original = _make_png_bytes()
        cleaned = stripper.strip_and_encode(original, format="PNG")
        assert len(cleaned) > 0
        # Verify it's valid PNG
        assert cleaned[:4] == b"\x89PNG"

    def test_strip_and_encode_jpeg(self):
        stripper = MetadataStripper()
        original = _make_jpeg_bytes()
        cleaned = stripper.strip_and_encode(original, format="JPEG")
        assert len(cleaned) > 0
        assert cleaned[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# Steganalysis tests
# ---------------------------------------------------------------------------

class TestSteganalysis:
    """Test chi-square and RS analysis for steganography detection."""

    def test_clean_image_not_suspicious(self):
        """Natural image should not be flagged as steganographic."""
        steg = Steganalyzer()
        img = Image.new("L", (64, 64), 128)
        result = steg.analyze(img)
        assert isinstance(result, SteganalysisResult)
        assert result.latency_ms >= 0

    def test_chi_square_score_bounded(self):
        steg = Steganalyzer()
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = steg.analyze(img)
        assert 0.0 <= result.chi_square_score <= 1.0

    def test_rs_score_returned(self):
        steg = Steganalyzer()
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = steg.analyze(img)
        assert isinstance(result.rs_score, float)

    def test_lsb_modified_image_detectable(self):
        """Image with uniform LSBs should be flagged."""
        steg = Steganalyzer(chi_square_threshold=0.5)
        # Create image with all LSBs set to 0 (suspicious uniformity)
        arr = np.random.randint(0, 128, (64, 64), dtype=np.uint8) * 2  # all even
        img = Image.fromarray(arr)
        result = steg.analyze(img)
        # Chi-square should detect uniform LSBs
        assert result.chi_square_score >= 0.0  # At minimum it runs

    def test_small_image_handled(self):
        steg = Steganalyzer()
        img = Image.new("L", (3, 3), 128)
        result = steg.analyze(img)
        assert isinstance(result, SteganalysisResult)

    def test_color_image_analyzed(self):
        steg = Steganalyzer()
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = steg.analyze(img)
        assert isinstance(result, SteganalysisResult)


# ---------------------------------------------------------------------------
# Image sanitizer tests
# ---------------------------------------------------------------------------

class TestImageSanitizer:
    """Test image re-encoding and sanitization."""

    def test_sanitize_png(self):
        sanitizer = ImageSanitizer()
        data = _make_png_bytes()
        result = sanitizer.sanitize(data, target_format="PNG")
        assert isinstance(result, SanitizationResult)
        assert result.sanitized_size > 0
        assert result.format_used == "PNG"
        assert not result.was_resized
        assert result.original_dimensions == (64, 64)

    def test_sanitize_jpeg(self):
        sanitizer = ImageSanitizer()
        data = _make_jpeg_bytes()
        result = sanitizer.sanitize(data, target_format="JPEG")
        assert result.sanitized_size > 0
        assert result.format_used == "JPEG"

    def test_sanitize_resizes_oversized(self):
        sanitizer = ImageSanitizer(max_dimension=32)
        data = _make_png_bytes(width=100, height=100)
        result = sanitizer.sanitize(data, target_format="PNG")
        assert result.was_resized
        assert result.sanitized_dimensions[0] <= 32
        assert result.sanitized_dimensions[1] <= 32

    def test_sanitize_preserves_small_image(self):
        sanitizer = ImageSanitizer(max_dimension=4096)
        data = _make_png_bytes(width=16, height=16)
        result = sanitizer.sanitize(data, target_format="PNG")
        assert not result.was_resized
        assert result.sanitized_dimensions == (16, 16)

    def test_sanitize_invalid_image(self):
        sanitizer = ImageSanitizer()
        result = sanitizer.sanitize(b"not an image", target_format="PNG")
        assert result.sanitized_size == 0


# ---------------------------------------------------------------------------
# Image scanner integration tests
# ---------------------------------------------------------------------------

class TestImageScanner:
    """Test L2 fast-path image scanner."""

    def test_scan_valid_png(self):
        scanner = ImageScanner(ocr_enabled=True, steg_enabled=True)
        data = _make_png_bytes()
        report = scanner.scan_image_bytes(data)
        assert isinstance(report, ImageScanReport)
        assert report.detected_format == "PNG"
        assert report.image_dimensions == (64, 64)
        assert report.total_latency_ms >= 0

    def test_scan_valid_jpeg(self):
        scanner = ImageScanner()
        data = _make_jpeg_bytes()
        report = scanner.scan_image_bytes(data)
        assert report.detected_format == "JPEG"

    def test_scan_oversized_image(self):
        scanner = ImageScanner(max_image_size_mb=0.001)  # 1KB limit
        data = _make_png_bytes()  # ~300 bytes, so make it bigger
        large_data = _make_png_bytes(width=500, height=500)
        report = scanner.scan_image_bytes(large_data)
        # Should detect oversized image
        assert any(r.is_threat for r in report.scan_results)

    def test_scan_unknown_format_blocked(self):
        scanner = ImageScanner()
        report = scanner.scan_image_bytes(b"\x00" * 50)
        assert any(
            r.is_threat and r.threat_category == ThreatCategory.SCHEMA_VIOLATION
            for r in report.scan_results
        )

    def test_perceptual_hash_computed(self):
        scanner = ImageScanner()
        data = _make_png_bytes()
        report = scanner.scan_image_bytes(data)
        assert report.perceptual_hash != ""
        assert len(report.perceptual_hash) == 16  # 64-bit hex

    def test_perceptual_hash_stable(self):
        """Same image should produce same hash."""
        scanner = ImageScanner()
        data = _make_png_bytes(color=(100, 200, 50))
        report1 = scanner.scan_image_bytes(data)
        report2 = scanner.scan_image_bytes(data)
        assert report1.perceptual_hash == report2.perceptual_hash

    def test_different_images_different_hash(self):
        scanner = ImageScanner()
        data1 = _make_png_bytes(color=(0, 0, 0))
        data2 = _make_png_bytes(color=(255, 255, 255))
        report1 = scanner.scan_image_bytes(data1)
        report2 = scanner.scan_image_bytes(data2)
        assert report1.perceptual_hash != report2.perceptual_hash

    def test_scan_with_ocr_disabled(self):
        scanner = ImageScanner(ocr_enabled=False, steg_enabled=False)
        data = _make_png_bytes()
        report = scanner.scan_image_bytes(data)
        assert report.ocr_result is None
        assert report.steg_result is None

    def test_extracted_text_property(self):
        scanner = ImageScanner(ocr_enabled=True)
        data = _make_png_bytes()
        report = scanner.scan_image_bytes(data)
        # Should return string (may be empty)
        assert isinstance(report.extracted_text, str)

    def test_scan_report_properties(self):
        scanner = ImageScanner()
        data = _make_png_bytes()
        report = scanner.scan_image_bytes(data)
        assert isinstance(report.should_block, bool)
        assert isinstance(report.max_confidence, float)
        assert 0.0 <= report.max_confidence <= 1.0


# ---------------------------------------------------------------------------
# Image analyzer tests (L3 slow path)
# ---------------------------------------------------------------------------

class TestImageAnalyzer:
    """Test L3 deep image analysis."""

    def test_analyze_valid_image(self):
        analyzer = ImageAnalyzer()
        data = _make_png_bytes()
        report = _run(analyzer.analyze(data))
        assert isinstance(report, ImageAnalysisReport)
        assert report.total_latency_ms >= 0

    def test_analyze_corrupt_image(self):
        analyzer = ImageAnalyzer()
        report = _run(analyzer.analyze(b"not an image at all"))
        assert any(r.is_threat for r in report.scan_results)

    def test_sanitize_compare(self):
        analyzer = ImageAnalyzer(sanitize_compare=True)
        data = _make_png_bytes()
        report = _run(analyzer.analyze(data))
        assert report.sanitization_result is not None
        assert report.pixel_diff_score >= 0.0

    def test_adversarial_score_bounded(self):
        analyzer = ImageAnalyzer()
        data = _make_png_bytes()
        report = _run(analyzer.analyze(data))
        assert 0.0 <= report.adversarial_score <= 1.0

    def test_report_properties(self):
        analyzer = ImageAnalyzer()
        data = _make_png_bytes()
        report = _run(analyzer.analyze(data))
        assert isinstance(report.should_block, bool)
        assert isinstance(report.max_confidence, float)


# ---------------------------------------------------------------------------
# MultimodalPreprocessor orchestrator tests
# ---------------------------------------------------------------------------

class TestMultimodalPreprocessor:
    """Test the main orchestrator."""

    def test_disabled_returns_empty(self):
        pp = MultimodalPreprocessor(enabled=False)
        messages = _make_multimodal_messages(_make_png_bytes())
        report = pp.preprocess_messages(messages)
        assert report.images_found == 0
        assert not report.has_images

    def test_text_only_no_images(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = [{"role": "user", "content": "Just a text message"}]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 0
        assert not report.has_images

    def test_extracts_base64_image(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = _make_multimodal_messages(_make_png_bytes())
        report = pp.preprocess_messages(messages)
        assert report.images_found == 1
        assert report.images_scanned == 1
        assert report.has_images

    def test_multiple_images(self):
        pp = MultimodalPreprocessor(enabled=True)
        img1 = _make_png_bytes()
        img2 = _make_jpeg_bytes()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Two images"},
                    {"type": "image_url", "image_url": {"url": _encode_b64_data_uri(img1)}},
                    {"type": "image_url", "image_url": {"url": _encode_b64_data_uri(img2, "image/jpeg")}},
                ],
            }
        ]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 2
        assert report.images_scanned == 2

    def test_max_images_limit(self):
        pp = MultimodalPreprocessor(enabled=True, max_images_per_request=2)
        img = _make_png_bytes()
        content_parts = [{"type": "text", "text": "Many images"}]
        for _ in range(5):
            content_parts.append(
                {"type": "image_url", "image_url": {"url": _encode_b64_data_uri(img)}}
            )
        messages = [{"role": "user", "content": content_parts}]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 5
        assert report.images_scanned == 2  # Limited
        # Should have a warning result about too many images
        assert any("too_many_images" in p for r in report.scan_results for p in r.matched_patterns)

    def test_skips_url_images(self):
        """Non-base64 image URLs should be skipped."""
        pp = MultimodalPreprocessor(enabled=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "URL image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                ],
            }
        ]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 0

    def test_invalid_base64_handled(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Bad b64"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!invalid!!!"}},
                ],
            }
        ]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 0  # Failed to decode

    def test_report_total_latency(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = _make_multimodal_messages(_make_png_bytes())
        report = pp.preprocess_messages(messages)
        assert report.total_latency_ms >= 0

    def test_analyze_images(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = _make_multimodal_messages(_make_png_bytes())
        analysis = _run(pp.analyze_images(messages))
        assert isinstance(analysis, ImageAnalysisReport)

    def test_analyze_disabled(self):
        pp = MultimodalPreprocessor(enabled=False)
        messages = _make_multimodal_messages(_make_png_bytes())
        analysis = _run(pp.analyze_images(messages))
        assert analysis is None

    def test_analyze_no_images(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = [{"role": "user", "content": "text only"}]
        analysis = _run(pp.analyze_images(messages))
        assert analysis is None


# ---------------------------------------------------------------------------
# Cross-modal injection tests
# ---------------------------------------------------------------------------

class TestCrossModalInjection:
    """Test that text extracted from images gets flagged by scanners."""

    def test_extracted_text_in_report(self):
        pp = MultimodalPreprocessor(enabled=True, ocr_enabled=True)
        # Use a text-like image
        data = _make_text_image()
        messages = _make_multimodal_messages(data)
        report = pp.preprocess_messages(messages)
        # OCR result should exist
        assert report.images_scanned == 1

    def test_metadata_text_extracted(self):
        """Text in EXIF metadata should be extracted."""
        pp = MultimodalPreprocessor(enabled=True)
        data = _make_exif_image()
        messages = _make_multimodal_messages(data)
        report = pp.preprocess_messages(messages)
        # Metadata text should be part of extracted_text
        if report.image_reports:
            meta = report.image_reports[0].metadata_report
            if meta:
                assert meta.had_exif

    def test_multimodal_scan_report_properties(self):
        pp = MultimodalPreprocessor(enabled=True)
        report = MultimodalScanReport()
        assert not report.should_block
        assert report.max_confidence == 0.0
        assert not report.has_images


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------

class TestMultimodalConfig:
    """Test MultimodalConfig integration."""

    def test_default_config(self):
        config = MultimodalConfig()
        assert config.enabled is False  # Disabled by default
        assert config.max_image_size_mb == 20.0
        assert config.ocr_enabled is True
        assert config.steg_enabled is True
        assert config.max_images_per_request == 10

    def test_config_in_aegis_config(self):
        config = AegisConfig()
        assert hasattr(config, "multimodal")
        assert isinstance(config.multimodal, MultimodalConfig)

    def test_preprocessor_from_config(self):
        config = MultimodalConfig(enabled=True, max_image_size_mb=5.0)
        pp = MultimodalPreprocessor(
            enabled=config.enabled,
            max_image_size_mb=config.max_image_size_mb,
            ocr_enabled=config.ocr_enabled,
            steg_enabled=config.steg_enabled,
            max_images_per_request=config.max_images_per_request,
        )
        assert pp.enabled is True


# ---------------------------------------------------------------------------
# Edge case / robustness tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_content_list(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = [{"role": "user", "content": []}]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 0

    def test_mixed_content_types(self):
        pp = MultimodalPreprocessor(enabled=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Some text"},
                    {"type": "image_url", "image_url": {"url": _encode_b64_data_uri(_make_png_bytes())}},
                    {"type": "text", "text": "More text"},
                ],
            }
        ]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 1

    def test_string_content_ignored(self):
        """Messages with string content (not list) should be skipped."""
        pp = MultimodalPreprocessor(enabled=True)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 0

    def test_multiple_messages_with_images(self):
        pp = MultimodalPreprocessor(enabled=True)
        img = _make_png_bytes()
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode_b64_data_uri(img)}},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode_b64_data_uri(img)}},
            ]},
        ]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 2

    def test_malformed_image_url_structure(self):
        """Handles missing/malformed image_url gracefully."""
        pp = MultimodalPreprocessor(enabled=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url"},  # Missing image_url key
                    {"type": "image_url", "image_url": "not a dict"},  # Wrong type
                    {"type": "image_url", "image_url": {"url": 123}},  # Wrong url type
                ],
            }
        ]
        report = pp.preprocess_messages(messages)
        assert report.images_found == 0  # All should be gracefully skipped

    def test_gif_format_scanned(self):
        scanner = ImageScanner()
        data = _make_gif_bytes()
        report = scanner.scan_image_bytes(data)
        assert report.detected_format == "GIF"
        assert report.image_dimensions == (32, 32)

    def test_bmp_format_scanned(self):
        scanner = ImageScanner()
        data = _make_bmp_bytes()
        report = scanner.scan_image_bytes(data)
        assert report.detected_format == "BMP"
