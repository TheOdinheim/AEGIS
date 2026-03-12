"""
Format Validator — File format validation, polyglot detection, macro detection.

Validates document format via magic bytes, detects polyglot files (valid in
multiple formats simultaneously — common attack vector), Office macros,
embedded executables, and format mismatches between declared and actual type.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Magic byte signatures
# ---------------------------------------------------------------------------

_MAGIC_PDF = b"%PDF"
_MAGIC_ZIP = b"PK\x03\x04"
_MAGIC_JPEG = b"\xff\xd8\xff"
_MAGIC_PNG = b"\x89PNG"
_MAGIC_GIF87 = b"GIF87a"
_MAGIC_GIF89 = b"GIF89a"
_MAGIC_OLE2 = b"\xd0\xcf\x11\xe0"
_MAGIC_XML = b"<?xml"
_MAGIC_MZ = b"MZ"
_MAGIC_ELF = b"\x7fELF"

# Declared content-type / extension → canonical format name
_DECLARED_TYPE_MAP: dict[str, str] = {
    # MIME types
    "application/pdf": "pdf",
    "text/html": "html",
    "application/html": "html",
    "text/xml": "xml",
    "application/xml": "xml",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "zip",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "zip",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "zip",
    "application/msword": "ole2",
    "application/vnd.ms-excel": "ole2",
    "application/vnd.ms-powerpoint": "ole2",
    "text/plain": "plain",
    "application/json": "json",
    "text/json": "json",
    # File extensions (lower-cased, dot-stripped)
    "pdf": "pdf",
    "html": "html",
    "htm": "html",
    "xml": "xml",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
    "gif": "gif",
    "zip": "zip",
    "docx": "zip",
    "xlsx": "zip",
    "pptx": "zip",
    "doc": "ole2",
    "xls": "ole2",
    "ppt": "ole2",
    "txt": "plain",
    "json": "json",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FormatValidationResult:
    """Result of format validation for a single file/document."""

    detected_format: str  # e.g. "pdf", "html", "zip", "json", "xml", "jpeg", "png", "gif", "plain", "unknown"
    declared_format: str  # Normalised from content-type or extension; empty string if not provided
    format_mismatch: bool  # detected_format != declared_format when declared_format is non-empty
    is_polyglot: bool      # File matches two or more format signatures simultaneously
    has_macros: bool       # Office VBA macros or PDF JavaScript detected
    has_embedded_executables: bool  # MZ/ELF/shell-script magic found at non-zero offset
    file_size: int
    threat_score: float    # 0.0–1.0 composite threat score
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class FormatValidator:
    """Validates file format via magic bytes and detects polyglots, macros,
    and embedded executables."""

    def validate(self, file_bytes: bytes, declared_type: str = "") -> FormatValidationResult:
        """Validate *file_bytes* against its *declared_type*.

        Args:
            file_bytes: Raw bytes of the file/document.
            declared_type: Content-Type header value or file extension (optional).
                           E.g. "application/pdf", "pdf", ".pdf".

        Returns:
            FormatValidationResult with all findings populated.
        """
        warnings: list[str] = []

        # ------------------------------------------------------------------ #
        # Normalise declared type
        # ------------------------------------------------------------------ #
        declared_format = self._normalise_declared_type(declared_type)

        # ------------------------------------------------------------------ #
        # Detect actual format via magic bytes
        # ------------------------------------------------------------------ #
        detected_format = self._detect_format(file_bytes)

        # ------------------------------------------------------------------ #
        # Polyglot detection
        # ------------------------------------------------------------------ #
        is_polyglot, polyglot_warnings = self._detect_polyglot(file_bytes)
        warnings.extend(polyglot_warnings)

        # ------------------------------------------------------------------ #
        # Macro detection
        # ------------------------------------------------------------------ #
        has_macros, macro_warnings = self._detect_macros(file_bytes, detected_format)
        warnings.extend(macro_warnings)

        # ------------------------------------------------------------------ #
        # Embedded executable detection
        # ------------------------------------------------------------------ #
        has_embedded_executables, exe_warnings = self._detect_embedded_executables(file_bytes)
        warnings.extend(exe_warnings)

        # ------------------------------------------------------------------ #
        # Format mismatch
        # ------------------------------------------------------------------ #
        format_mismatch = False
        if declared_format and detected_format != "unknown" and declared_format != detected_format:
            format_mismatch = True
            warnings.append(
                f"Format mismatch: declared '{declared_format}' but detected '{detected_format}'"
            )

        # ------------------------------------------------------------------ #
        # Threat score
        # ------------------------------------------------------------------ #
        threat_score = self._calculate_threat_score(
            is_polyglot=is_polyglot,
            has_macros=has_macros,
            has_embedded_executables=has_embedded_executables,
            format_mismatch=format_mismatch,
        )

        return FormatValidationResult(
            detected_format=detected_format,
            declared_format=declared_format,
            format_mismatch=format_mismatch,
            is_polyglot=is_polyglot,
            has_macros=has_macros,
            has_embedded_executables=has_embedded_executables,
            file_size=len(file_bytes),
            threat_score=threat_score,
            warnings=warnings,
        )

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _normalise_declared_type(declared_type: str) -> str:
        """Convert a raw declared type string to a canonical format name."""
        if not declared_type:
            return ""
        key = declared_type.strip().lower().lstrip(".")
        # Strip parameters (e.g. "text/html; charset=utf-8" → "text/html")
        key = key.split(";")[0].strip()
        return _DECLARED_TYPE_MAP.get(key, "")

    @staticmethod
    def _detect_format(data: bytes) -> str:
        """Return the primary detected format based on magic bytes."""
        if not data:
            return "unknown"

        header = data[:16]
        first_1k = data[:1024].lower()

        if header[:4] == _MAGIC_OLE2:
            return "ole2"
        if header[:4] == _MAGIC_PDF:
            return "pdf"
        if header[:4] == _MAGIC_ZIP:
            return "zip"
        if header[:3] == _MAGIC_JPEG:
            return "jpeg"
        if header[:4] == _MAGIC_PNG[:4]:
            return "png"
        if header[:6] in (_MAGIC_GIF87, _MAGIC_GIF89):
            return "gif"
        if data[:5] == _MAGIC_XML:
            return "xml"
        # HTML detection — case-insensitive scan of first 1 KB
        if (
            b"<!doctype" in first_1k
            or b"<html" in first_1k
        ):
            return "html"
        # JSON: starts with whitespace then '{' or '['
        stripped = data[:256].lstrip()
        if stripped and stripped[0:1] in (b"{", b"["):
            return "json"

        # Printable ASCII / UTF-8 text heuristic
        try:
            data[:512].decode("utf-8")
            return "plain"
        except (UnicodeDecodeError, ValueError):
            pass

        return "unknown"

    def _detect_polyglot(self, data: bytes) -> tuple[bool, list[str]]:
        """Check whether *data* is simultaneously valid in multiple formats."""
        warnings: list[str] = []
        matched_formats: list[str] = []

        header = data[:16]
        body_lower = data.lower()

        # Collect every matching signature
        if header[:4] == _MAGIC_PDF:
            matched_formats.append("pdf")
        if header[:4] == _MAGIC_ZIP:
            matched_formats.append("zip")
        if header[:4] == _MAGIC_OLE2:
            matched_formats.append("ole2")
        if header[:3] == _MAGIC_JPEG:
            matched_formats.append("jpeg")
        if header[:4] == _MAGIC_PNG[:4]:
            matched_formats.append("png")
        if header[:6] in (_MAGIC_GIF87, _MAGIC_GIF89):
            matched_formats.append("gif")
        if data[:5] == _MAGIC_XML:
            matched_formats.append("xml")

        # HTML markers anywhere in the file (can coexist with binary magic)
        if b"<!doctype" in body_lower or b"<html" in body_lower:
            matched_formats.append("html")

        # PDF magic at a non-zero offset (e.g. embedded in ZIP/image)
        if "pdf" not in matched_formats and _MAGIC_PDF in data[4:]:
            matched_formats.append("pdf")

        # ZIP magic at a non-zero offset (embedded ZIP)
        if "zip" not in matched_formats and _MAGIC_ZIP in data[4:]:
            matched_formats.append("zip")

        is_polyglot = len(matched_formats) >= 2
        if is_polyglot:
            warnings.append(
                f"Polyglot file detected — matches formats: {', '.join(matched_formats)}"
            )

        return is_polyglot, warnings

    def _detect_macros(self, data: bytes, detected_format: str) -> tuple[bool, list[str]]:
        """Detect VBA macros or JavaScript in documents."""
        warnings: list[str] = []

        # OLE2 compound document — legacy Office with potential macros
        if detected_format == "ole2":
            warnings.append("OLE2 compound document may contain VBA macros")
            return True, warnings

        # ZIP-based Office (OOXML) — check for vbaProject.bin entry
        if detected_format == "zip" or data[:4] == _MAGIC_ZIP:
            has_macros = self._zip_contains_vba(data)
            if has_macros:
                warnings.append("Office document contains vbaProject.bin — VBA macros present")
                return True, warnings

        # PDF — check for JavaScript actions
        if detected_format == "pdf":
            if b"/JS" in data or b"/JavaScript" in data:
                warnings.append("PDF contains JavaScript (/JS or /JavaScript action)")
                return True, warnings

        return False, warnings

    @staticmethod
    def _zip_contains_vba(data: bytes) -> bool:
        """Return True if the ZIP archive contains a vbaProject.bin entry."""
        # Fast byte-level check first (avoids ZIP parse overhead)
        if b"vbaProject.bin" in data:
            return True
        # Authoritative check via zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                names = zf.namelist()
                return any("vbaProject.bin" in n for n in names)
        except Exception:
            # Corrupted or password-protected ZIP — already checked raw bytes above
            return False

    @staticmethod
    def _detect_embedded_executables(data: bytes) -> tuple[bool, list[str]]:
        """Detect MZ (PE), ELF, or shell-script magic embedded after offset 4."""
        warnings: list[str] = []

        search_region = data[4:]  # Skip initial magic bytes

        if _MAGIC_MZ in search_region:
            warnings.append("Embedded PE executable (MZ header) found after initial magic bytes")
            return True, warnings

        if _MAGIC_ELF in search_region:
            warnings.append("Embedded ELF executable found after initial magic bytes")
            return True, warnings

        for shebang in (b"#!/bin/", b"#!/usr/bin/"):
            if shebang in search_region:
                warnings.append(
                    f"Embedded shell script ({shebang.decode()!r}...) found after initial magic bytes"
                )
                return True, warnings

        return False, warnings

    @staticmethod
    def _calculate_threat_score(
        *,
        is_polyglot: bool,
        has_macros: bool,
        has_embedded_executables: bool,
        format_mismatch: bool,
    ) -> float:
        """Compute a composite 0.0–1.0 threat score."""
        score = 0.0
        if is_polyglot:
            score += 0.50
        if has_macros:
            score += 0.40
        if has_embedded_executables:
            score += 0.60
        if format_mismatch:
            score += 0.20
        return min(score, 1.0)
