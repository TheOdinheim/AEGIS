"""
Document Scanner — L2 innate fast-path document scanning (<15ms target).

Validates format, extracts text (visible + hidden), detects hidden content
injection, and feeds extracted text through the existing regex engine.

Returns ScanResult per the L2 contract. Document scanning supplements
text scanning — it does NOT replace it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aegis.layers.multimodal.format_validator import FormatValidator
from aegis.layers.multimodal.hidden_content_detector import HiddenContentDetector
from aegis.layers.multimodal.text_extractor import DocumentTextExtractor
from aegis.models.scan_result import ScanResult, ThreatCategory

logger = logging.getLogger(__name__)

# Confidence caps for each detection signal
_CONF_POLYGLOT = 0.90
_CONF_MACROS = 0.85
_CONF_EMBEDDED_EXE = 0.95
_CONF_HIDDEN_INJECTION = 0.90
_CONF_METADATA_INJECTION = 0.75
_CONF_SCRIPTS = 0.85

# When scanning hidden/metadata/comments text, multiply regex confidence by
# this factor because hidden injection is inherently more suspicious than
# visible injection.
_HIDDEN_CONF_MULTIPLIER = 1.2


class DocumentScanner:
    """L2 innate fast-path scanner for uploaded documents.

    Orchestrates three sub-components in sequence:

    1. :class:`~aegis.layers.multimodal.format_validator.FormatValidator`
       — magic-byte format validation, polyglot and macro detection.
    2. :class:`~aegis.layers.multimodal.text_extractor.DocumentTextExtractor`
       — multi-format text extraction (visible + hidden surfaces).
    3. :class:`~aegis.layers.multimodal.hidden_content_detector.HiddenContentDetector`
       — injection pattern scanning on hidden surfaces.

    Optionally wraps a
    :class:`~aegis.layers.innate.regex_engine.RegexEngine` to feed all
    extracted text surfaces through the full MITRE-ATLAS-tagged pattern
    library.

    All failures are handled gracefully — document scanning is supplementary
    to (not a replacement for) the primary text-based innate scan pipeline.
    A parsing exception returns a clean ScanResult so the request can
    continue through the normal pipeline.
    """

    def __init__(
        self,
        regex_engine: Any | None = None,
        max_document_size_mb: float = 50.0,
        block_macros: bool = True,
        block_scripts: bool = True,
    ) -> None:
        """Initialise the document scanner.

        Args:
            regex_engine: Optional ``RegexEngine`` instance. When provided,
                all extracted text surfaces are passed through the full
                compiled-pattern library after local checks complete.
            max_document_size_mb: Maximum accepted document size in megabytes.
                Documents exceeding this limit are rejected immediately
                (confidence 1.0, ``SCHEMA_VIOLATION``).
            block_macros: If ``True``, documents with VBA macros or PDF
                JavaScript are blocked (confidence 0.85). If ``False``, the
                finding is recorded as a warning in ``matched_patterns`` but
                the request is not blocked.
            block_scripts: If ``True``, documents with embedded script bodies
                (e.g. ``<script>`` tags in HTML) are blocked
                (confidence 0.85). If ``False``, scripts are noted but not
                blocked.
        """
        self._regex_engine = regex_engine
        self._max_bytes = int(max_document_size_mb * 1024 * 1024)
        self._block_macros = block_macros
        self._block_scripts = block_scripts

        self._format_validator = FormatValidator()
        self._text_extractor = DocumentTextExtractor()
        self._hidden_detector = HiddenContentDetector()

        logger.debug(
            "DocumentScanner initialised: max_size=%.1fMB block_macros=%s "
            "block_scripts=%s regex_engine=%s",
            max_document_size_mb,
            block_macros,
            block_scripts,
            "yes" if regex_engine is not None else "no",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan(
        self,
        file_bytes: bytes,
        filename: str = "",
        content_type: str = "",
        context: Any | None = None,
    ) -> ScanResult:
        """Scan a document for threats.

        Args:
            file_bytes: Raw bytes of the uploaded document.
            filename: Original filename (used for format detection and logging).
            content_type: MIME type declared by the client (used alongside
                magic-byte detection).
            context: Optional ``RequestContext`` — currently unused, reserved
                for future per-tenant threshold overrides.

        Returns:
            :class:`~aegis.models.scan_result.ScanResult` with
            ``scanner_id="document_scanner"``. If any check raises an
            unhandled exception, returns a *clean* (``is_threat=False``)
            result so the request can continue through the text pipeline.
        """
        t_start = time.perf_counter()

        try:
            result = await self._scan_internal(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
            )
        except Exception as exc:  # pylint: disable=broad-except
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.warning(
                "DocumentScanner: unhandled exception for '%s' — returning clean result. "
                "Error: %s",
                filename or "<unnamed>",
                exc,
            )
            return ScanResult(
                scanner_id="document_scanner",
                is_threat=False,
                confidence=0.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=[],
                sanitized_input=None,
                latency_ms=round(elapsed_ms, 2),
            )

        return result

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    async def _scan_internal(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> ScanResult:
        t_start = time.perf_counter()

        threats: list[tuple[float, ThreatCategory, str]] = []
        # Each entry: (confidence, category, description)

        # ------------------------------------------------------------------ #
        # Step 1 — Size check
        # ------------------------------------------------------------------ #
        if len(file_bytes) > self._max_bytes:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            size_mb = len(file_bytes) / (1024 * 1024)
            logger.warning(
                "DocumentScanner: document '%s' rejected — size %.1fMB exceeds %.1fMB limit",
                filename or "<unnamed>",
                size_mb,
                self._max_bytes / (1024 * 1024),
            )
            return ScanResult(
                scanner_id="document_scanner",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.SCHEMA_VIOLATION,
                matched_patterns=[
                    f"document_size_exceeded:{size_mb:.1f}MB>{self._max_bytes/(1024*1024):.1f}MB"
                ],
                sanitized_input=None,
                latency_ms=round(elapsed_ms, 2),
            )

        # ------------------------------------------------------------------ #
        # Step 2 — Format validation
        # ------------------------------------------------------------------ #
        declared_type = content_type or (
            filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        )
        fmt_result = self._format_validator.validate(file_bytes, declared_type)

        if fmt_result.is_polyglot:
            threats.append((
                _CONF_POLYGLOT,
                ThreatCategory.SCHEMA_VIOLATION,
                f"polyglot_file:formats={fmt_result.warnings[0] if fmt_result.warnings else 'multiple'}",
            ))
            logger.warning(
                "DocumentScanner: polyglot file detected for '%s'", filename or "<unnamed>"
            )

        if fmt_result.has_macros and self._block_macros:
            threats.append((
                _CONF_MACROS,
                ThreatCategory.SCHEMA_VIOLATION,
                "document_macros:vba_or_pdf_javascript_detected",
            ))
            logger.warning(
                "DocumentScanner: macro-enabled document rejected for '%s'",
                filename or "<unnamed>",
            )
        elif fmt_result.has_macros:
            # Non-blocking warning
            threats.append((
                0.40,
                ThreatCategory.SCHEMA_VIOLATION,
                "document_macros:detected_not_blocked",
            ))

        if fmt_result.has_embedded_executables:
            threats.append((
                _CONF_EMBEDDED_EXE,
                ThreatCategory.SCHEMA_VIOLATION,
                "embedded_executable:pe_elf_or_shell_detected",
            ))
            logger.warning(
                "DocumentScanner: embedded executable detected in '%s'",
                filename or "<unnamed>",
            )

        if fmt_result.format_mismatch:
            # Warning — not a block by itself, but recorded
            threats.append((
                0.35,
                ThreatCategory.SCHEMA_VIOLATION,
                f"format_mismatch:declared={fmt_result.declared_format}"
                f",detected={fmt_result.detected_format}",
            ))

        # ------------------------------------------------------------------ #
        # Step 3 — Text extraction
        # ------------------------------------------------------------------ #
        extraction = self._text_extractor.extract(file_bytes, filename, content_type)

        if extraction.warnings:
            logger.debug(
                "DocumentScanner: extraction warnings for '%s': %s",
                filename or "<unnamed>",
                extraction.warnings,
            )

        # ------------------------------------------------------------------ #
        # Step 4 — Hidden content detection
        # ------------------------------------------------------------------ #
        hidden_report = self._hidden_detector.detect(extraction)

        if hidden_report.hidden_text_injection:
            threats.append((
                _CONF_HIDDEN_INJECTION,
                ThreatCategory.PROMPT_INJECTION,
                "hidden_text_injection:injection_patterns_in_invisible_content",
            ))

        if hidden_report.metadata_injection:
            threats.append((
                _CONF_METADATA_INJECTION,
                ThreatCategory.PROMPT_INJECTION,
                "metadata_injection:injection_patterns_in_document_metadata",
            ))

        if hidden_report.comment_injection:
            # Comment injection is less severe than hidden text injection
            threats.append((
                0.70,
                ThreatCategory.PROMPT_INJECTION,
                "comment_injection:injection_patterns_in_document_comments",
            ))

        if hidden_report.bidi_found:
            threats.append((
                0.60,
                ThreatCategory.ENCODING_OBFUSCATION,
                "bidi_chars:bidirectional_override_characters_detected",
            ))

        if hidden_report.zero_width_found:
            threats.append((
                0.55,
                ThreatCategory.ENCODING_OBFUSCATION,
                "zero_width_chars:invisible_characters_detected",
            ))

        if hidden_report.scripts_found and self._block_scripts:
            threats.append((
                _CONF_SCRIPTS,
                ThreatCategory.PROMPT_INJECTION,
                f"scripts_found:count={len(extraction.scripts)}",
            ))
            logger.warning(
                "DocumentScanner: scripts blocked in '%s' (%d script(s))",
                filename or "<unnamed>",
                len(extraction.scripts),
            )
        elif hidden_report.scripts_found:
            threats.append((
                0.40,
                ThreatCategory.PROMPT_INJECTION,
                f"scripts_found:count={len(extraction.scripts)},not_blocked",
            ))

        # ------------------------------------------------------------------ #
        # Step 5 — Regex engine scan of all extracted text surfaces
        # ------------------------------------------------------------------ #
        if self._regex_engine is not None:
            regex_threats = await self._run_regex_on_surfaces(extraction)
            threats.extend(regex_threats)

        # ------------------------------------------------------------------ #
        # Step 6 — Combine results
        # ------------------------------------------------------------------ #
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if not threats:
            logger.debug(
                "DocumentScanner: clean result for '%s' in %.2fms",
                filename or "<unnamed>",
                elapsed_ms,
            )
            return ScanResult(
                scanner_id="document_scanner",
                is_threat=False,
                confidence=0.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=[],
                sanitized_input=None,
                latency_ms=round(elapsed_ms, 2),
            )

        # Pick the highest-confidence threat as the primary category
        threats_sorted = sorted(threats, key=lambda t: t[0], reverse=True)
        top_conf, top_cat, _ = threats_sorted[0]

        all_patterns = [desc for _, _, desc in threats]
        max_conf = min(top_conf, 1.0)

        logger.info(
            "DocumentScanner: threat detected in '%s' — confidence=%.2f category=%s "
            "in %.2fms (%d signal(s))",
            filename or "<unnamed>",
            max_conf,
            top_cat.value,
            elapsed_ms,
            len(threats),
        )

        return ScanResult(
            scanner_id="document_scanner",
            is_threat=True,
            confidence=max_conf,
            threat_category=top_cat,
            matched_patterns=all_patterns,
            sanitized_input=None,
            latency_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Regex surface scanning
    # ------------------------------------------------------------------

    async def _run_regex_on_surfaces(
        self,
        extraction: Any,
    ) -> list[tuple[float, ThreatCategory, str]]:
        """Scan all text surfaces through the regex engine.

        Surfaces:
        - ``visible_text`` — scanned at face value.
        - ``hidden_text``, ``metadata_text``, ``comments_text`` — confidence
          multiplied by ``_HIDDEN_CONF_MULTIPLIER`` (1.2) and capped at 1.0
          because injection in non-visible surfaces is inherently more
          suspicious.

        Returns a list of ``(confidence, category, description)`` tuples for
        every surface that produced a positive scan result.
        """
        results: list[tuple[float, ThreatCategory, str]] = []

        surfaces: list[tuple[str, str, bool]] = [
            # (name, text, apply_hidden_boost)
            ("visible_text", extraction.visible_text, False),
            ("hidden_text", extraction.hidden_text, True),
            ("metadata_text", extraction.metadata_text, True),
            ("comments_text", extraction.comments_text, True),
        ]

        for surface_name, text, is_hidden in surfaces:
            if not text or not text.strip():
                continue
            try:
                scan_result: ScanResult = await self._regex_engine.scan(text)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "DocumentScanner: regex engine error on surface '%s': %s",
                    surface_name,
                    exc,
                )
                continue

            if not scan_result.is_threat:
                continue

            conf = scan_result.confidence
            if is_hidden:
                conf = min(conf * _HIDDEN_CONF_MULTIPLIER, 1.0)

            for pattern in scan_result.matched_patterns:
                results.append((
                    conf,
                    scan_result.threat_category,
                    f"regex:{surface_name}:{pattern}",
                ))

            # If no individual patterns were reported, still record the hit
            if not scan_result.matched_patterns:
                results.append((
                    conf,
                    scan_result.threat_category,
                    f"regex:{surface_name}:pattern_match",
                ))

        return results
