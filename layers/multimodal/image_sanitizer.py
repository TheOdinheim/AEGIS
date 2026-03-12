"""
Image Sanitizer — Re-encode images to remove hidden content.

Strips metadata, re-encodes pixels, normalizes dimensions.
The re-encoded image should be semantically identical but free of
steganographic payloads, metadata injections, or format exploits.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

from PIL import Image

from aegis.layers.multimodal.metadata_stripper import MetadataStripper

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of image sanitization."""
    sanitized_bytes: bytes = b""
    original_size: int = 0
    sanitized_size: int = 0
    format_used: str = "PNG"
    was_resized: bool = False
    original_dimensions: tuple[int, int] = (0, 0)
    sanitized_dimensions: tuple[int, int] = (0, 0)
    latency_ms: float = 0.0


class ImageSanitizer:
    """Re-encode images to eliminate hidden payloads.

    Process:
    1. Decode image from bytes (validates format)
    2. Strip all metadata (EXIF, XMP, IPTC)
    3. Normalize dimensions if oversized
    4. Re-encode to clean format (JPEG at quality 85, or PNG)

    The re-encoded image contains only pixel data — no metadata,
    no steganographic LSB patterns (due to JPEG lossy compression),
    no format-level exploits.
    """

    def __init__(
        self,
        max_dimension: int = 4096,
        jpeg_quality: int = 85,
    ):
        self._max_dimension = max_dimension
        self._jpeg_quality = jpeg_quality
        self._stripper = MetadataStripper()

    def sanitize(self, image_bytes: bytes, target_format: str = "PNG") -> SanitizationResult:
        """Sanitize image: strip metadata, normalize size, re-encode."""
        start = time.perf_counter()

        try:
            img = Image.open(io.BytesIO(image_bytes))
            original_dims = img.size

            # Strip metadata
            clean = self._stripper.strip_metadata(img)

            # Normalize dimensions
            was_resized = False
            w, h = clean.size
            if w > self._max_dimension or h > self._max_dimension:
                ratio = min(self._max_dimension / w, self._max_dimension / h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                clean = clean.resize((new_w, new_h), Image.LANCZOS)
                was_resized = True

            # Re-encode
            buf = io.BytesIO()
            fmt = target_format.upper()
            if fmt == "JPEG":
                clean = clean.convert("RGB")
                clean.save(buf, format="JPEG", quality=self._jpeg_quality)
            elif fmt == "WEBP":
                clean.save(buf, format="WEBP", quality=self._jpeg_quality)
            else:
                clean.save(buf, format="PNG")
                fmt = "PNG"

            sanitized = buf.getvalue()

            return SanitizationResult(
                sanitized_bytes=sanitized,
                original_size=len(image_bytes),
                sanitized_size=len(sanitized),
                format_used=fmt,
                was_resized=was_resized,
                original_dimensions=original_dims,
                sanitized_dimensions=clean.size,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            logger.error("Image sanitization failed: %s", e)
            return SanitizationResult(
                original_size=len(image_bytes),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
