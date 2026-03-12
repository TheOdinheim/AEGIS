"""
OCR Engine — Extract text from images for injection scanning.

Three-tier OCR with graceful degradation:
1. Tesseract (pytesseract) — best accuracy, optional dependency
2. Pillow pixel analysis — basic text detection heuristic, always available
3. Fallback "unavailable" — returns empty string, logs warning

Core principle: extracted text feeds through the EXISTING L2/L3 text
detection pipeline. OCR is the bridge between images and text security.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import pytesseract
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OCRResult:
    """Result of OCR text extraction."""
    text: str = ""
    engine_used: str = "unavailable"
    confidence: float = 0.0
    language_hints: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    word_count: int = 0


class OCREngine:
    """Multi-tier OCR engine with graceful degradation.

    Tesseract → Pillow pixel analysis → empty fallback.
    All extracted text should be fed through L2/L3 text scanners.
    """

    def __init__(self, prefer_tesseract: bool = True):
        self._prefer_tesseract = prefer_tesseract and _TESSERACT_AVAILABLE
        if prefer_tesseract and not _TESSERACT_AVAILABLE:
            logger.info("pytesseract not available — using Pillow pixel analysis fallback")

    @property
    def engine_name(self) -> str:
        if self._prefer_tesseract:
            return "tesseract"
        return "pillow_heuristic"

    def extract_text(self, img: Image.Image) -> OCRResult:
        """Extract text from image using best available engine."""
        start = time.perf_counter()

        if self._prefer_tesseract:
            result = self._tesseract_extract(img)
        else:
            result = self._pillow_heuristic(img)

        result.latency_ms = (time.perf_counter() - start) * 1000
        result.word_count = len(result.text.split()) if result.text.strip() else 0
        return result

    def _tesseract_extract(self, img: Image.Image) -> OCRResult:
        """Extract text using Tesseract OCR."""
        try:
            text = pytesseract.image_to_string(img)
            # Get confidence from detailed data
            try:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) > 0]
                avg_conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
            except Exception:
                avg_conf = 0.5  # Default confidence if detailed data fails

            return OCRResult(
                text=text.strip(),
                engine_used="tesseract",
                confidence=min(avg_conf, 1.0),
            )
        except Exception as e:
            logger.warning("Tesseract extraction failed: %s — falling back to Pillow", e)
            return self._pillow_heuristic(img)

    def _pillow_heuristic(self, img: Image.Image) -> OCRResult:
        """Basic text detection heuristic using Pillow pixel analysis.

        Detects high-contrast regions that likely contain text by analyzing
        edge density and contrast ratios. Cannot actually read text but can
        estimate whether text is present and approximately how much.

        This is a detection heuristic, not real OCR — it identifies that
        text-like content exists but cannot extract the actual characters.
        """
        try:
            # Convert to grayscale
            gray = img.convert("L")
            arr = np.array(gray)

            if arr.size < 100:
                return OCRResult(engine_used="pillow_heuristic", confidence=0.0)

            # Edge detection via gradient magnitude
            # High edge density in horizontal direction suggests text
            gx = np.abs(np.diff(arr.astype(np.float64), axis=1))
            gy = np.abs(np.diff(arr.astype(np.float64), axis=0))

            # Compute edge density
            edge_threshold = 30.0
            h_edge_density = float(np.mean(gx > edge_threshold))
            v_edge_density = float(np.mean(gy > edge_threshold))

            # Text typically has high horizontal edge density
            # and moderate vertical edge density
            text_score = h_edge_density * 0.6 + v_edge_density * 0.4

            # Contrast analysis — text images tend to be bimodal
            std_dev = float(np.std(arr))
            contrast_score = min(std_dev / 80.0, 1.0)

            # Combined heuristic
            has_text_likely = text_score > 0.05 and contrast_score > 0.3

            # We can't extract actual text, but we can flag the image
            text_indicator = ""
            if has_text_likely:
                text_indicator = "[IMAGE_CONTAINS_TEXT_CONTENT]"

            return OCRResult(
                text=text_indicator,
                engine_used="pillow_heuristic",
                confidence=min(text_score * contrast_score * 2.0, 1.0),
            )
        except Exception as e:
            logger.warning("Pillow heuristic failed: %s", e)
            return OCRResult(engine_used="pillow_heuristic", confidence=0.0)
