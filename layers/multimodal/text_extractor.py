"""
Document Text Extraction Engine — Multi-format text extraction for AEGIS.

Extracts visible AND hidden text from documents for feeding through the
existing L2/L3 text detection pipeline. Hidden content (invisible text,
metadata, comments, scripts) is separated from visible text so the
hidden content detector can apply elevated threat scoring.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# Optional dependency flags
_PDFPLUMBER_AVAILABLE = False
_PYPDF2_AVAILABLE = False
_YAML_AVAILABLE = False

try:
    import pdfplumber  # type: ignore
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pass

try:
    import PyPDF2  # type: ignore
    _PYPDF2_AVAILABLE = True
except ImportError:
    pass

try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except ImportError:
    pass

# Zero-width and invisible characters (mirrors regex_engine.py set)
_ZERO_WIDTH_CHARS = (
    "\u200b\u200c\u200d\u200e\u200f"
    "\u00ad\ufeff\u2060\u2061\u2062\u2063\u2064"
    "\u180e\u00a0\u202a\u202b\u202c\u202d\u202e"
    "\u17b4\u17b5\u115f\u1160\u3164\uFFA0\u2800\u034f"
)
_ZERO_WIDTH_RE = re.compile(f"[{re.escape(_ZERO_WIDTH_CHARS)}]")

# CSS patterns that hide text
_CSS_HIDDEN_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0",
    re.IGNORECASE,
)

# PDF stream and JavaScript patterns
_PDF_JS_RE = re.compile(rb"/J(?:avaScript|S)\s*\(", re.IGNORECASE)
_PDF_STREAM_RE = re.compile(rb"stream\r?\n(.*?)endstream", re.DOTALL)

# JSON/YAML keys whose values go to metadata_text rather than visible_text
_METADATA_KEYS = frozenset([
    "author", "description", "title", "comments", "comment",
    "instructions", "system", "prompt", "note", "notes", "message",
])


@dataclass
class DocumentExtractionResult:
    """Result of multi-format document text extraction."""
    visible_text: str = ""
    hidden_text: str = ""
    metadata_text: str = ""
    comments_text: str = ""
    scripts: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    format_detected: str = "unknown"
    extraction_method: str = "none"
    warnings: list[str] = field(default_factory=list)


class _HTMLContentCollector(HTMLParser):
    """Separates HTML content into visible, hidden, comments, scripts, and URLs."""

    _INVISIBLE_TAGS = frozenset(["script", "style", "head"])

    def __init__(self) -> None:
        super().__init__()
        self.visible_parts: list[str] = []
        self.hidden_parts: list[str] = []
        self.metadata_parts: list[str] = []
        self.comments: list[str] = []
        self.scripts: list[str] = []
        self.urls: list[str] = []
        self._tag_stack: list[str] = []
        self._hidden_depth: int = 0
        self._script_depth: int = 0
        self._current_script: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        self._tag_stack.append(tag_lower)

        if tag_lower == "script":
            self._script_depth += 1
            self._current_script = []

        if tag_lower == "meta":
            content = dict(attrs).get("content", "")
            if content:
                self.metadata_parts.append(content)

        for attr_name, attr_value in attrs:
            if attr_name in ("href", "src") and attr_value:
                self.urls.append(attr_value)

        css_hidden = any(
            n == "style" and v and _CSS_HIDDEN_RE.search(v) for n, v in attrs
        )
        if css_hidden or tag_lower in self._INVISIBLE_TAGS:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._tag_stack and self._tag_stack[-1] == tag_lower:
            self._tag_stack.pop()

        if tag_lower == "script":
            self._script_depth = max(0, self._script_depth - 1)
            script_text = "".join(self._current_script).strip()
            if script_text:
                self.scripts.append(script_text)
                self.hidden_parts.append(script_text)
            self._current_script = []

        if tag_lower in self._INVISIBLE_TAGS or self._hidden_depth > 0:
            self._hidden_depth = max(0, self._hidden_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._script_depth > 0:
            self._current_script.append(data)
            return
        stripped = data.strip()
        if not stripped:
            return
        in_invisible = self._tag_stack and self._tag_stack[-1] in self._INVISIBLE_TAGS
        if self._hidden_depth > 0 or in_invisible:
            self.hidden_parts.append(stripped)
        else:
            self.visible_parts.append(stripped)

    def handle_comment(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.comments.append(stripped)

    def handle_entityref(self, name: str) -> None:
        pass

    def handle_charref(self, name: str) -> None:
        pass


class DocumentTextExtractor:
    """Multi-format document text extractor for AEGIS threat detection.

    Extracts text from PDF, HTML, Markdown, JSON, YAML, and plain text.
    Hidden text (invisible CSS, metadata, comments, scripts) is separated
    from visible text to enable elevated threat scoring downstream.

    All optional dependencies (pdfplumber, PyPDF2, yaml) degrade gracefully
    to stdlib or regex-based fallbacks when unavailable.
    """

    def extract(
        self,
        file_bytes: bytes,
        filename: str = "",
        content_type: str = "",
    ) -> DocumentExtractionResult:
        """Extract text from document bytes.

        Detection order: magic bytes → content_type → file extension.
        Returns a DocumentExtractionResult with all text surfaces populated.
        """
        fmt = self._detect_format(file_bytes, filename, content_type)
        result = DocumentExtractionResult(format_detected=fmt)

        try:
            if fmt == "pdf":
                self._extract_pdf(file_bytes, result)
            elif fmt == "html":
                self._extract_html(file_bytes, result)
            elif fmt == "markdown":
                self._extract_markdown(file_bytes, result)
            elif fmt == "json":
                self._extract_json(file_bytes, result)
            elif fmt == "yaml":
                self._extract_yaml(file_bytes, result)
            elif fmt in ("image/jpeg", "image/png", "image/gif"):
                result.warnings.append(f"Binary image format '{fmt}' — no text extraction")
                result.extraction_method = "skipped_binary"
            elif fmt == "zip":
                result.warnings.append("ZIP/Office Open XML — extraction not yet implemented")
                result.extraction_method = "skipped_zip"
            else:
                self._extract_plain(file_bytes, result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Text extraction failed for format '%s': %s", fmt, exc)
            result.warnings.append(f"Extraction error: {exc}")
            try:
                result.visible_text = file_bytes.decode("utf-8", errors="replace")
                result.extraction_method = "error_fallback_plain"
            except Exception:  # pylint: disable=broad-except
                result.extraction_method = "error_no_text"

        return result

    def _detect_format(self, data: bytes, filename: str, content_type: str) -> str:
        """Identify format by magic bytes first, then content_type, then extension."""
        if data[:4] == b"%PDF":
            return "pdf"
        if data[:4] == b"PK\x03\x04":
            return "zip"
        if data[:9].lower() in (b"<!doctype", b"<!doctyp") or data[:5].lower() == b"<html":
            return "html"
        if data[:5] == b"<?xml":
            return "html"
        if data[:2] == b"\xff\xd8":
            return "image/jpeg"
        if data[:4] == b"\x89PNG":
            return "image/png"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"

        ct = content_type.lower()
        if "pdf" in ct:
            return "pdf"
        if "html" in ct or "xml" in ct:
            return "html"
        if "json" in ct:
            return "json"
        if "yaml" in ct:
            return "yaml"
        if "markdown" in ct:
            return "markdown"

        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        return {
            "pdf": "pdf",
            "html": "html", "htm": "html",
            "md": "markdown", "markdown": "markdown",
            "json": "json",
            "yaml": "yaml", "yml": "yaml",
            "txt": "text", "text": "text",
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "docx": "zip", "xlsx": "zip", "pptx": "zip",
        }.get(ext, "text")

    def _extract_pdf(self, data: bytes, result: DocumentExtractionResult) -> None:
        if _PDFPLUMBER_AVAILABLE:
            self._extract_pdf_pdfplumber(data, result)
        elif _PYPDF2_AVAILABLE:
            self._extract_pdf_pypdf2(data, result)
        else:
            self._extract_pdf_regex(data, result)
        self._detect_pdf_javascript(data, result)

    def _extract_pdf_pdfplumber(self, data: bytes, result: DocumentExtractionResult) -> None:
        pages_text: list[str] = []
        hidden_parts: list[str] = []
        meta_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            meta = pdf.metadata or {}
            for key in ("Title", "Author", "Subject", "Keywords", "Creator", "Producer"):
                val = meta.get(key, "")
                if val:
                    meta_parts.append(f"{key}: {val}")
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    pages_text.append(text)
                for annot in (page.annots or []):
                    contents = annot.get("data", {}).get("Contents", "")
                    if contents:
                        hidden_parts.append(str(contents))
        result.visible_text = "\n".join(pages_text)
        result.hidden_text = "\n".join(hidden_parts)
        result.metadata_text = "\n".join(meta_parts)
        result.extraction_method = "pdfplumber"

    def _extract_pdf_pypdf2(self, data: bytes, result: DocumentExtractionResult) -> None:
        pages_text: list[str] = []
        meta_parts: list[str] = []
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        meta = reader.metadata or {}
        for key in ("/Title", "/Author", "/Subject", "/Keywords", "/Creator", "/Producer"):
            val = meta.get(key, "")
            if val:
                meta_parts.append(f"{key.lstrip('/')}: {val}")
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                pages_text.append(text)
        result.visible_text = "\n".join(pages_text)
        result.metadata_text = "\n".join(meta_parts)
        result.extraction_method = "pypdf2"

    def _extract_pdf_regex(self, data: bytes, result: DocumentExtractionResult) -> None:
        """Regex fallback: pull printable ASCII runs from raw PDF streams."""
        texts: list[str] = []
        for match in _PDF_STREAM_RE.finditer(data):
            for run in re.findall(rb"[\x20-\x7e]{4,}", match.group(1)):
                try:
                    texts.append(run.decode("ascii"))
                except UnicodeDecodeError:
                    pass
        result.visible_text = "\n".join(texts)
        result.extraction_method = "pdf_regex_fallback"
        result.warnings.append(
            "pdfplumber and PyPDF2 unavailable — using regex stream extraction (reduced fidelity)"
        )

    def _detect_pdf_javascript(self, data: bytes, result: DocumentExtractionResult) -> None:
        if not _PDF_JS_RE.search(data):
            return
        result.hidden_text += "\n[AEGIS: Embedded JavaScript detected in PDF]"
        result.warnings.append("Embedded JavaScript found in PDF")
        for match in re.finditer(rb"/JS\s*\(([^)]{0,512})\)", data, re.IGNORECASE):
            try:
                result.scripts.append(match.group(1).decode("latin-1"))
            except UnicodeDecodeError:
                pass

    def _extract_html(self, data: bytes, result: DocumentExtractionResult) -> None:
        text = data.decode("utf-8", errors="replace")
        collector = _HTMLContentCollector()
        try:
            collector.feed(text)
        except Exception as exc:  # pylint: disable=broad-except
            result.warnings.append(f"HTML parser error (partial extraction): {exc}")
        result.visible_text = " ".join(collector.visible_parts)
        result.hidden_text = "\n".join(collector.hidden_parts)
        result.metadata_text = "\n".join(collector.metadata_parts)
        result.comments_text = "\n".join(collector.comments)
        result.scripts = list(collector.scripts)
        result.urls = list(dict.fromkeys(collector.urls))
        result.extraction_method = "html_parser_stdlib"

    def _extract_markdown(self, data: bytes, result: DocumentExtractionResult) -> None:
        text = data.decode("utf-8", errors="replace")

        md_comment_re = re.compile(r"^\[//\]:\s*#\s*\((.+?)\)", re.MULTILINE)
        result.comments_text = "\n".join(md_comment_re.findall(text))

        # HTML blocks embedded in markdown
        for block in re.findall(r"(<[a-zA-Z][^>]*>.*?</[a-zA-Z]+>)", text, re.DOTALL):
            sub = DocumentExtractionResult()
            self._extract_html(block.encode("utf-8"), sub)
            if sub.hidden_text:
                result.hidden_text += "\n" + sub.hidden_text
            if sub.comments_text:
                result.comments_text += "\n" + sub.comments_text
            result.urls.extend(sub.urls)

        # Image alt text and URLs
        for alt, rest in re.findall(r"!\[([^\]]*)\]\(([^)]*)\)", text):
            if alt:
                result.metadata_text += f" {alt}"
            url_m = re.match(r"(\S+)", rest.strip())
            if url_m:
                result.urls.append(url_m.group(1))

        # Link titles and URLs
        for _lt, url, title in re.findall(r'\[([^\]]+)\]\(([^)"]+)(?:"([^"]*)")?\)', text):
            result.urls.append(url.strip())
            if title:
                result.metadata_text += f" {title}"

        # Strip syntax for visible text
        clean = md_comment_re.sub("", text)
        clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", clean)
        clean = re.sub(r"`{1,3}[^`]*`{1,3}", "", clean)
        clean = re.sub(r"^```.*?^```", "", clean, flags=re.MULTILINE | re.DOTALL)
        clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", clean)
        clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
        clean = re.sub(r"^[-*+]\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^\d+\.\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^\s*>\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        result.visible_text = clean.strip()
        result.extraction_method = "markdown_regex"

    def _extract_json(self, data: bytes, result: DocumentExtractionResult) -> None:
        try:
            obj = json.loads(data.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            result.warnings.append(f"JSON parse error: {exc}")
            result.visible_text = data.decode("utf-8", errors="replace")
            result.extraction_method = "json_parse_failed"
            return
        visible: list[str] = []
        metadata: list[str] = []
        self._collect_json_strings(obj, key=None, visible=visible, metadata=metadata)
        result.visible_text = "\n".join(visible)
        result.metadata_text = "\n".join(metadata)
        result.extraction_method = "json_stdlib"

    def _collect_json_strings(
        self, node: object, key: str | None, visible: list[str], metadata: list[str]
    ) -> None:
        if isinstance(node, str):
            (metadata if key and key.lower() in _METADATA_KEYS else visible).append(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                self._collect_json_strings(v, key=k, visible=visible, metadata=metadata)
        elif isinstance(node, list):
            for item in node:
                self._collect_json_strings(item, key=key, visible=visible, metadata=metadata)

    def _extract_yaml(self, data: bytes, result: DocumentExtractionResult) -> None:
        if not _YAML_AVAILABLE:
            result.warnings.append("PyYAML unavailable — treating YAML as plain text")
            self._extract_plain(data, result)
            result.extraction_method = "yaml_fallback_plain"
            return
        try:
            obj = yaml.safe_load(data.decode("utf-8", errors="replace"))
        except Exception as exc:  # pylint: disable=broad-except
            result.warnings.append(f"YAML parse error: {exc}")
            self._extract_plain(data, result)
            result.extraction_method = "yaml_parse_failed"
            return
        visible: list[str] = []
        metadata: list[str] = []
        self._collect_json_strings(obj, key=None, visible=visible, metadata=metadata)
        result.visible_text = "\n".join(visible)
        result.metadata_text = "\n".join(metadata)
        result.extraction_method = "yaml_safe_load"

    def _extract_plain(self, data: bytes, result: DocumentExtractionResult) -> None:
        text = data.decode("utf-8", errors="replace")
        if _ZERO_WIDTH_RE.search(text):
            result.warnings.append("Zero-width / invisible characters detected in plain text")
            result.hidden_text = "".join(c for c in text if c in _ZERO_WIDTH_CHARS)
        result.visible_text = text
        result.extraction_method = "plain_text"
