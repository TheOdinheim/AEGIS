"""
Document Analyzer — L3 adaptive slow-path document analysis (50-200ms target).

Deep analysis of document content using ML classifiers and semantic search.
Hidden text is analyzed separately and with elevated suspicion — legitimate
documents rarely contain hidden prompt injection instructions.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from aegis.layers.multimodal.hidden_content_detector import HiddenContentDetector
from aegis.layers.multimodal.text_extractor import DocumentTextExtractor
from aegis.models.scan_result import ScanResult, ThreatCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suspicious URL detection patterns
# ---------------------------------------------------------------------------

# IP address used as a domain (e.g. http://192.168.1.1/payload)
_IP_DOMAIN_RE = re.compile(r"https?://\d+\.\d+\.\d+\.\d+")

# Suspicious top-level domains associated with free/abuse hosting
_SUSPICIOUS_TLDS = frozenset([
    ".tk", ".ml", ".cf", ".ga", ".gq", ".xyz", ".top", ".work", ".click", ".loan",
])

# Dangerous URI schemes
_DATA_URI_RE = re.compile(r"^data:", re.IGNORECASE)
_JS_URI_RE = re.compile(r"^javascript:", re.IGNORECASE)

# Hidden-to-visible ratio threshold: hidden text > 2× visible is suspicious
_HIDDEN_RATIO_THRESHOLD = 2.0

# Minimum visible text length before ratio check is meaningful
_MIN_VISIBLE_FOR_RATIO = 20

# Minimum metadata length relative to visible content considered suspicious
_METADATA_RELATIVE_THRESHOLD = 3.0

# Minimum visible content length before metadata ratio check fires
_MIN_VISIBLE_FOR_META = 50

# Confidence boost multiplier applied to classifier results on hidden text
_HIDDEN_CONFIDENCE_BOOST = 1.3

# Many scripts in a single document is structurally suspicious
_MANY_SCRIPTS_THRESHOLD = 5


class DocumentAnalyzer:
    """L3 adaptive slow-path deep document analysis.

    Orchestrates four analysis stages in sequence:

    1. **Text extraction** — DocumentTextExtractor splits document bytes into
       visible, hidden, metadata, comments, scripts, and URL surfaces.
    2. **Hidden content detection** — HiddenContentDetector scans non-visible
       surfaces for injection patterns and Unicode obfuscation.
    3. **ML classification** — optional InjectionClassifier runs on visible
       and hidden text separately.  Hidden-surface detections receive a 1.3×
       confidence boost because legitimate documents essentially never contain
       instruction overrides in invisible content.
    4. **Semantic search** — optional SemanticSearchAnalyzer queries the
       Threat Vault (FAISS HNSW) for embeddings similar to both visible and
       hidden text.

    All failures are handled gracefully — a parsing or inference exception
    logs a warning and returns a clean ScanResult so the pipeline continues.
    """

    def __init__(
        self,
        injection_classifier: Any | None = None,
        semantic_engine: Any | None = None,
    ) -> None:
        """Initialise the document analyzer.

        Args:
            injection_classifier: Optional InjectionClassifier instance
                (``aegis.layers.adaptive.injection_classifier``).  Must expose
                ``async def analyze(text: str) -> AdaptiveAnalysisResult``.
                When ``None``, ML classification is skipped.
            semantic_engine: Optional SemanticSearchAnalyzer instance
                (``aegis.layers.adaptive.semantic_search``).  Must expose
                ``async def analyze(text: str) -> AdaptiveAnalysisResult``.
                When ``None``, vault similarity search is skipped.
        """
        self._classifier = injection_classifier
        self._semantic = semantic_engine
        self._extractor = DocumentTextExtractor()
        self._hidden_detector = HiddenContentDetector()

        logger.debug(
            "DocumentAnalyzer initialised: classifier=%s semantic=%s",
            "yes" if injection_classifier is not None else "no",
            "yes" if semantic_engine is not None else "no",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        file_bytes: bytes,
        filename: str = "",
        content_type: str = "",
        context: Any | None = None,
    ) -> ScanResult:
        """Run deep document analysis and return a combined ScanResult.

        Args:
            file_bytes: Raw document bytes.
            filename: Original filename (used for format detection and logging).
            content_type: MIME type declared by the client.
            context: Optional ``RequestContext`` — reserved for future
                per-tenant threshold overrides.

        Returns:
            :class:`~aegis.models.scan_result.ScanResult` with
            ``scanner_id="document_analyzer"``.  On any unhandled exception,
            returns a *clean* (``is_threat=False``) result so the request
            can continue through the pipeline.
        """
        t_start = time.perf_counter()

        try:
            result = await self._analyze_internal(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
            )
        except Exception as exc:  # pylint: disable=broad-except
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.warning(
                "DocumentAnalyzer: unhandled exception for '%s' — returning clean result. "
                "Error: %s",
                filename or "<unnamed>",
                exc,
            )
            return ScanResult(
                scanner_id="document_analyzer",
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

    async def _analyze_internal(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> ScanResult:
        t_start = time.perf_counter()

        # Each entry: (confidence, ThreatCategory, description)
        findings: list[tuple[float, ThreatCategory, str]] = []

        # ------------------------------------------------------------------ #
        # Step 1 — Text extraction
        # ------------------------------------------------------------------ #
        extraction = self._extractor.extract(file_bytes, filename, content_type)

        if extraction.warnings:
            logger.debug(
                "DocumentAnalyzer: extraction warnings for '%s': %s",
                filename or "<unnamed>",
                extraction.warnings,
            )

        visible_text = extraction.visible_text or ""
        hidden_text = extraction.hidden_text or ""

        # Combine non-visible surfaces into a single hidden corpus
        combined_hidden = "\n".join(
            part for part in (
                hidden_text,
                extraction.metadata_text or "",
                extraction.comments_text or "",
            )
            if part.strip()
        )

        # ------------------------------------------------------------------ #
        # Step 2 — Hidden content detection
        # ------------------------------------------------------------------ #
        hidden_report = self._hidden_detector.detect(extraction)

        if hidden_report.threat_score > 0.0:
            if hidden_report.hidden_text_injection:
                findings.append((
                    min(hidden_report.threat_score, 0.90),
                    ThreatCategory.PROMPT_INJECTION,
                    "hidden_content:injection_in_invisible_text",
                ))
            if hidden_report.metadata_injection:
                findings.append((
                    min(hidden_report.threat_score * 0.80, 0.80),
                    ThreatCategory.PROMPT_INJECTION,
                    "hidden_content:injection_in_metadata",
                ))
            if hidden_report.comment_injection:
                findings.append((
                    min(hidden_report.threat_score * 0.70, 0.75),
                    ThreatCategory.PROMPT_INJECTION,
                    "hidden_content:injection_in_comments",
                ))
            if hidden_report.bidi_found:
                findings.append((
                    0.60,
                    ThreatCategory.ENCODING_OBFUSCATION,
                    "hidden_content:bidi_override_chars_detected",
                ))
            if hidden_report.zero_width_found:
                findings.append((
                    0.55,
                    ThreatCategory.ENCODING_OBFUSCATION,
                    "hidden_content:zero_width_chars_detected",
                ))

        # ------------------------------------------------------------------ #
        # Step 3 — ML injection classification (visible + hidden separately)
        # ------------------------------------------------------------------ #
        if self._classifier is not None:
            # Classify visible text
            if visible_text.strip() and len(visible_text) > 10:
                try:
                    visible_result = await self._classifier.analyze(visible_text[:4096])
                    if visible_result.is_threat and visible_result.confidence > 0.0:
                        findings.append((
                            min(visible_result.confidence, 1.0),
                            ThreatCategory.PROMPT_INJECTION,
                            f"classifier:visible_text:confidence={visible_result.confidence:.3f}",
                        ))
                        logger.debug(
                            "DocumentAnalyzer: classifier flagged visible text "
                            "confidence=%.3f for '%s'",
                            visible_result.confidence,
                            filename or "<unnamed>",
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "DocumentAnalyzer: classifier failed on visible text for '%s': %s",
                        filename or "<unnamed>",
                        exc,
                    )

            # Classify hidden text — boost confidence because hidden injection
            # is far more suspicious than visible injection attempts
            if combined_hidden.strip():
                try:
                    hidden_result = await self._classifier.analyze(combined_hidden[:4096])
                    if hidden_result.is_threat and hidden_result.confidence > 0.0:
                        boosted = min(hidden_result.confidence * _HIDDEN_CONFIDENCE_BOOST, 1.0)
                        findings.append((
                            boosted,
                            ThreatCategory.PROMPT_INJECTION,
                            f"classifier:hidden_text:confidence={hidden_result.confidence:.3f}"
                            f":boosted={boosted:.3f}",
                        ))
                        logger.warning(
                            "DocumentAnalyzer: classifier flagged hidden text "
                            "raw=%.3f boosted=%.3f for '%s'",
                            hidden_result.confidence,
                            boosted,
                            filename or "<unnamed>",
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "DocumentAnalyzer: classifier failed on hidden text for '%s': %s",
                        filename or "<unnamed>",
                        exc,
                    )

        # ------------------------------------------------------------------ #
        # Step 4 — Semantic vault search (visible + hidden separately)
        # ------------------------------------------------------------------ #
        if self._semantic is not None:
            if visible_text.strip():
                try:
                    sem_visible = await self._semantic.analyze(visible_text[:2048])
                    if sem_visible.is_threat and sem_visible.confidence > 0.0:
                        findings.append((
                            min(sem_visible.confidence, 1.0),
                            ThreatCategory.PROMPT_INJECTION,
                            f"semantic:visible_text:similarity={sem_visible.confidence:.3f}",
                        ))
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "DocumentAnalyzer: semantic search failed on visible text for '%s': %s",
                        filename or "<unnamed>",
                        exc,
                    )

            if combined_hidden.strip():
                try:
                    sem_hidden = await self._semantic.analyze(combined_hidden[:2048])
                    if sem_hidden.is_threat and sem_hidden.confidence > 0.0:
                        boosted = min(sem_hidden.confidence * _HIDDEN_CONFIDENCE_BOOST, 1.0)
                        findings.append((
                            boosted,
                            ThreatCategory.PROMPT_INJECTION,
                            f"semantic:hidden_text:similarity={sem_hidden.confidence:.3f}"
                            f":boosted={boosted:.3f}",
                        ))
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "DocumentAnalyzer: semantic search failed on hidden text for '%s': %s",
                        filename or "<unnamed>",
                        exc,
                    )

        # ------------------------------------------------------------------ #
        # Step 5 — URL analysis
        # ------------------------------------------------------------------ #
        suspicious_urls = self._check_urls(extraction.urls)
        for url, reason in suspicious_urls:
            findings.append((
                0.65,
                ThreatCategory.PROMPT_INJECTION,
                f"url:suspicious:{reason}:{url[:120]}",
            ))

        # ------------------------------------------------------------------ #
        # Step 6 — Structural complexity analysis
        # ------------------------------------------------------------------ #
        struct_findings = self._check_structure(visible_text, combined_hidden, extraction)
        findings.extend(struct_findings)

        # ------------------------------------------------------------------ #
        # Step 7 — Combine and return
        # ------------------------------------------------------------------ #
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if not findings:
            logger.debug(
                "DocumentAnalyzer: clean result for '%s' in %.2fms",
                filename or "<unnamed>",
                elapsed_ms,
            )
            return ScanResult(
                scanner_id="document_analyzer",
                is_threat=False,
                confidence=0.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=[],
                sanitized_input=None,
                latency_ms=round(elapsed_ms, 2),
            )

        findings_sorted = sorted(findings, key=lambda f: f[0], reverse=True)
        top_conf, top_cat, _ = findings_sorted[0]
        all_patterns = [desc for _, _, desc in findings_sorted]
        max_conf = min(top_conf, 1.0)

        logger.info(
            "DocumentAnalyzer: threat detected in '%s' — confidence=%.3f "
            "category=%s in %.2fms (%d finding(s))",
            filename or "<unnamed>",
            max_conf,
            top_cat.value,
            elapsed_ms,
            len(findings),
        )

        return ScanResult(
            scanner_id="document_analyzer",
            is_threat=True,
            confidence=max_conf,
            threat_category=top_cat,
            matched_patterns=all_patterns,
            sanitized_input=None,
            latency_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # URL analysis helpers
    # ------------------------------------------------------------------

    def _check_urls(self, urls: list[str]) -> list[tuple[str, str]]:
        """Return list of (url, reason) for suspicious URLs.

        Checks for:
        - IP address domains (potential C2 or SSRF vectors)
        - Known abuse-hosting TLDs
        - data: URIs (potential inline payload delivery)
        - javascript: URIs (XSS/code injection)
        """
        suspicious: list[tuple[str, str]] = []
        for url in urls:
            url_stripped = url.strip()
            if not url_stripped:
                continue

            if _DATA_URI_RE.match(url_stripped):
                suspicious.append((url_stripped, "data_uri"))
                continue

            if _JS_URI_RE.match(url_stripped):
                suspicious.append((url_stripped, "javascript_uri"))
                continue

            if _IP_DOMAIN_RE.match(url_stripped):
                suspicious.append((url_stripped, "ip_address_domain"))
                continue

            url_lower = url_stripped.lower()
            for tld in _SUSPICIOUS_TLDS:
                # Match TLD at end of host (before path/query/fragment)
                host_part = url_lower.split("/")[2] if "//" in url_lower else url_lower
                host_only = host_part.split(":")[0]
                if host_only.endswith(tld):
                    suspicious.append((url_stripped, f"suspicious_tld:{tld}"))
                    break

        return suspicious

    # ------------------------------------------------------------------
    # Structural analysis helpers
    # ------------------------------------------------------------------

    def _check_structure(
        self,
        visible_text: str,
        combined_hidden: str,
        extraction: Any,
    ) -> list[tuple[float, ThreatCategory, str]]:
        """Detect structurally suspicious document properties.

        Checks:
        - Very high hidden-to-visible text ratio (hidden > 2× visible)
        - Excessive number of embedded scripts
        - Very large metadata relative to visible content

        Returns a list of ``(confidence, ThreatCategory, description)`` tuples.
        """
        results: list[tuple[float, ThreatCategory, str]] = []

        visible_len = len(visible_text.strip())
        hidden_len = len(combined_hidden.strip())
        metadata_len = len((extraction.metadata_text or "").strip())

        # High hidden-to-visible ratio
        if visible_len >= _MIN_VISIBLE_FOR_RATIO and hidden_len > 0:
            ratio = hidden_len / visible_len
            if ratio > _HIDDEN_RATIO_THRESHOLD:
                results.append((
                    min(0.45 + (ratio - _HIDDEN_RATIO_THRESHOLD) * 0.05, 0.75),
                    ThreatCategory.ENCODING_OBFUSCATION,
                    f"structure:high_hidden_ratio:{ratio:.1f}x",
                ))
                logger.debug(
                    "DocumentAnalyzer: suspicious hidden/visible ratio %.1fx", ratio
                )

        # Many embedded scripts
        script_count = len(extraction.scripts) if extraction.scripts else 0
        if script_count >= _MANY_SCRIPTS_THRESHOLD:
            results.append((
                min(0.40 + (script_count - _MANY_SCRIPTS_THRESHOLD) * 0.03, 0.70),
                ThreatCategory.PROMPT_INJECTION,
                f"structure:many_scripts:count={script_count}",
            ))

        # Disproportionately large metadata relative to visible content
        if visible_len >= _MIN_VISIBLE_FOR_META and metadata_len > 0:
            meta_ratio = metadata_len / visible_len
            if meta_ratio > _METADATA_RELATIVE_THRESHOLD:
                results.append((
                    min(0.35 + (meta_ratio - _METADATA_RELATIVE_THRESHOLD) * 0.04, 0.65),
                    ThreatCategory.PROMPT_INJECTION,
                    f"structure:large_metadata_ratio:{meta_ratio:.1f}x",
                ))

        return results
