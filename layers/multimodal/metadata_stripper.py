"""
Metadata Stripper — Remove EXIF, XMP, IPTC metadata from images.

Strips all metadata that could contain hidden instructions, geolocation,
device fingerprints, or steganographic payloads embedded in metadata fields.

Uses Pillow only — no external dependencies required.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MetadataReport:
    """Report of metadata found and stripped."""
    had_exif: bool = False
    had_xmp: bool = False
    had_iptc: bool = False
    fields_stripped: list[str] = field(default_factory=list)
    text_found_in_metadata: str = ""
    latency_ms: float = 0.0


class MetadataStripper:
    """Strip all metadata from images to remove hidden content.

    Metadata fields (EXIF UserComment, XMP descriptions, IPTC captions)
    can contain hidden instructions or data. This stripper extracts any
    text found in metadata for scanning, then removes all metadata.
    """

    # EXIF tags that commonly contain user-editable text
    _TEXT_EXIF_TAGS = {
        0x010E: "ImageDescription",
        0x010F: "Make",
        0x0110: "Model",
        0x0131: "Software",
        0x013B: "Artist",
        0x8298: "Copyright",
        0x9286: "UserComment",
        0x9C9B: "XPTitle",
        0x9C9C: "XPComment",
        0x9C9D: "XPAuthor",
        0x9C9E: "XPKeywords",
        0x9C9F: "XPSubject",
    }

    def extract_metadata_text(self, img: Image.Image) -> MetadataReport:
        """Extract text from all metadata fields and report what was found."""
        start = time.perf_counter()
        report = MetadataReport()
        text_parts: list[str] = []

        # EXIF
        exif_data = img.getexif()
        if exif_data:
            report.had_exif = True
            for tag_id, tag_name in self._TEXT_EXIF_TAGS.items():
                value = exif_data.get(tag_id)
                if value and isinstance(value, (str, bytes)):
                    text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
                    if text.strip():
                        text_parts.append(text.strip())
                        report.fields_stripped.append(f"EXIF:{tag_name}")

        # XMP — stored in img.info as "xmp" key (XML blob)
        xmp_data = img.info.get("xmp", b"")
        if xmp_data:
            report.had_xmp = True
            if isinstance(xmp_data, bytes):
                xmp_text = xmp_data.decode("utf-8", errors="ignore")
            else:
                xmp_text = str(xmp_data)
            # Extract text between XML tags (simple extraction)
            import re
            tag_contents = re.findall(r">([^<]+)<", xmp_text)
            for content in tag_contents:
                content = content.strip()
                if content and len(content) > 3 and not content.startswith("http"):
                    text_parts.append(content)
            if tag_contents:
                report.fields_stripped.append("XMP:content")

        # IPTC — stored in img.info as "photoshop" key
        iptc_data = img.info.get("photoshop")
        if iptc_data:
            report.had_iptc = True
            report.fields_stripped.append("IPTC:photoshop")
            # Try to extract printable text from IPTC binary data
            if isinstance(iptc_data, bytes):
                try:
                    printable = iptc_data.decode("utf-8", errors="ignore")
                    # Filter to meaningful strings
                    import re
                    strings = re.findall(r"[\x20-\x7E]{4,}", printable)
                    text_parts.extend(strings)
                except Exception:
                    pass

        report.text_found_in_metadata = "\n".join(text_parts)
        report.latency_ms = (time.perf_counter() - start) * 1000
        return report

    def strip_metadata(self, img: Image.Image) -> Image.Image:
        """Return a new image with all metadata removed.

        Creates a new Image from pixel data only — no EXIF, XMP, IPTC,
        ICC profiles, or other metadata survives.
        """
        # Create clean image from pixel data only
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        return clean

    def strip_and_encode(self, image_bytes: bytes, format: str = "PNG") -> bytes:
        """Strip metadata and re-encode to bytes."""
        img = Image.open(io.BytesIO(image_bytes))
        clean = self.strip_metadata(img)
        buf = io.BytesIO()
        save_format = format.upper()
        if save_format == "JPEG":
            clean = clean.convert("RGB")
            clean.save(buf, format="JPEG", quality=85)
        else:
            clean.save(buf, format=save_format)
        return buf.getvalue()
