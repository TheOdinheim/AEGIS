"""
Stage 2: Serialization Safety — detect unsafe model formats.

Biological Analog: Screening for known pathogenic structures at mucosal
barriers before allowing entry.

Scans model files for unsafe serialization formats (pickle, joblib with
arbitrary code execution). Rejects pickle-based files unless from a trusted
allowlisted source. Promotes SafeTensors and ONNX as safe alternatives.

Enhanced with format risk scoring, pickle-within-ZIP detection, ONNX
structural validation, and SafeTensors preference recommendations.

ASSUMED-BREACH POSTURE: Model files are untrusted artifacts. An attacker can
embed arbitrary Python code inside pickle-serialized model weights. This stage
does not attempt to sandbox deserialization — it blocks unsafe formats outright.
"""

from __future__ import annotations

import logging
import struct
import time
import zipfile
from pathlib import Path
from typing import Any

from aegis.layers.supply_chain.models import StageResult

logger = logging.getLogger(__name__)

# Formats considered safe — no arbitrary code execution on load
SAFE_EXTENSIONS = {".safetensors", ".onnx", ".onnx_data", ".json", ".txt", ".md", ".yaml", ".yml", ".toml", ".cfg"}

# Formats that allow arbitrary code execution
UNSAFE_EXTENSIONS = {".pkl", ".pickle", ".pt", ".pth", ".joblib", ".npy", ".npz"}

# Magic bytes for pickle protocol
_PICKLE_MAGIC = b"\x80"  # pickle protocol 2+ opcode

# Magic bytes for PyTorch (zip-based, contains pickle internally)
_ZIP_MAGIC = b"PK\x03\x04"

# ONNX protobuf magic: field 1 (ir_version), wire type 0 (varint) = 0x08
_ONNX_MAGIC = b"\x08"

# SafeTensors start with a JSON header length (little-endian u64)
# followed by a '{' character
_SAFETENSORS_HEADER_CHAR = ord("{")

# ---------------------------------------------------------------------------
# Format risk scores — lower is safer
# ---------------------------------------------------------------------------

FORMAT_RISK_SCORES: dict[str, float] = {
    ".safetensors": 0.0,    # Safe: no code execution, validated header
    ".onnx": 0.1,           # Safe: protobuf, no arbitrary code, but complex parser
    ".onnx_data": 0.1,      # Safe: raw tensor data for ONNX external data
    ".json": 0.0,           # Safe: text format
    ".txt": 0.0,            # Safe: text format
    ".md": 0.0,             # Safe: text format
    ".yaml": 0.15,          # Mostly safe: YAML can have code via unsafe_load
    ".yml": 0.15,           # Same as .yaml
    ".toml": 0.0,           # Safe: no code execution
    ".cfg": 0.05,           # Safe: config format
    ".pkl": 0.8,            # Dangerous: arbitrary code execution
    ".pickle": 0.8,         # Dangerous: arbitrary code execution
    ".pt": 0.8,             # Dangerous: ZIP with pickle inside
    ".pth": 0.8,            # Dangerous: ZIP with pickle inside
    ".joblib": 0.8,         # Dangerous: pickle-based
    ".npy": 0.5,            # Medium: allow_pickle possible
    ".npz": 0.5,            # Medium: allow_pickle possible
}

# Default risk score for unknown extensions
UNKNOWN_FORMAT_RISK = 0.9


def get_format_risk(extension: str) -> float:
    """Get the risk score for a file extension.

    Returns:
        Risk score from 0.0 (safe) to 1.0 (dangerous).
    """
    return FORMAT_RISK_SCORES.get(extension.lower(), UNKNOWN_FORMAT_RISK)


def scan_serialization(
    model_path: Path,
    *,
    allowlisted_sources: set[str] | None = None,
    source: str | None = None,
) -> StageResult:
    """Run Stage 2: Serialization Safety scan.

    Args:
        model_path: Directory containing model files.
        allowlisted_sources: Set of trusted sources for which pickle is tolerated.
        source: Declared source/publisher of this model.

    Returns:
        StageResult with pass/fail/warn and per-file analysis including risk scores.
    """
    start = time.perf_counter()
    allowlisted_sources = allowlisted_sources or set()
    is_allowlisted = source in allowlisted_sources if source else False

    details: dict[str, Any] = {
        "files_scanned": 0,
        "safe_files": [],
        "unsafe_files": [],
        "warnings": [],
        "recommendations": [],
        "risk_scores": {},
        "aggregate_risk": 0.0,
        "source": source,
        "source_allowlisted": is_allowlisted,
    }

    if not model_path.exists():
        return StageResult(
            stage="serialization_safety",
            status="fail",
            details={"error": f"Model path does not exist: {model_path}"},
            duration_ms=_elapsed(start),
        )

    files = [f for f in model_path.rglob("*") if f.is_file()]
    if not files:
        return StageResult(
            stage="serialization_safety",
            status="fail",
            details={"error": "No files found in model directory"},
            duration_ms=_elapsed(start),
        )

    risk_scores: list[float] = []
    has_safetensors = False
    has_pickle_variant = False

    for file_path in files:
        details["files_scanned"] += 1
        ext = file_path.suffix.lower()
        rel = str(file_path.relative_to(model_path))
        file_risk = get_format_risk(ext)
        details["risk_scores"][rel] = file_risk
        risk_scores.append(file_risk)

        if ext == ".safetensors":
            has_safetensors = True

        if ext in SAFE_EXTENSIONS:
            # Validate ONNX files structurally
            if ext == ".onnx":
                onnx_warning = _validate_onnx(file_path)
                if onnx_warning:
                    details["warnings"].append({
                        "file": rel, "extension": ext, "note": onnx_warning,
                        "severity": "info",
                    })
            # Validate SafeTensors header
            elif ext == ".safetensors":
                st_warning = _validate_safetensors(file_path)
                if st_warning:
                    details["warnings"].append({
                        "file": rel, "extension": ext, "note": st_warning,
                        "severity": "info",
                    })

            details["safe_files"].append(rel)
            continue

        if ext in UNSAFE_EXTENSIONS:
            has_pickle_variant = True
            # Check file content for actual unsafe serialization
            threat_type = _detect_unsafe_content(file_path)
            if threat_type:
                entry = {"file": rel, "extension": ext, "threat": threat_type, "risk_score": file_risk}
                if is_allowlisted:
                    details["warnings"].append({**entry, "note": "Allowed via source allowlist"})
                else:
                    details["unsafe_files"].append(entry)
            else:
                details["safe_files"].append(rel)
            continue

        # Unknown extension — warn with high risk score
        details["warnings"].append({
            "file": rel, "extension": ext,
            "note": "Unknown format", "risk_score": file_risk,
        })

    # Compute aggregate risk (max of all file risks)
    details["aggregate_risk"] = max(risk_scores) if risk_scores else 0.0

    # Generate recommendations
    if has_pickle_variant and not has_safetensors:
        details["recommendations"].append(
            "Convert pickle/PyTorch files to SafeTensors format for safer loading. "
            "See: https://huggingface.co/docs/safetensors/"
        )
    if has_pickle_variant and has_safetensors:
        details["recommendations"].append(
            "SafeTensors files detected alongside pickle files. "
            "Remove pickle files and use SafeTensors exclusively."
        )

    # Determine status — informational warnings (severity=info) on safe files
    # don't affect status; only unknown formats and unsafe files do.
    status_warnings = [
        w for w in details["warnings"] if w.get("severity") != "info"
    ]
    if details["unsafe_files"]:
        status = "fail"
    elif status_warnings:
        status = "warn"
    else:
        status = "pass"

    return StageResult(
        stage="serialization_safety",
        status=status,
        details=details,
        duration_ms=_elapsed(start),
    )


def _detect_unsafe_content(file_path: Path) -> str | None:
    """Inspect file header bytes for known unsafe serialization formats.

    Returns a threat description string if unsafe content is detected, None otherwise.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
    except (OSError, PermissionError):
        return "unreadable_file"

    if not header:
        return None

    # Pickle protocol 2+ starts with \x80\x02-\x05
    if header[0:1] == _PICKLE_MAGIC and len(header) > 1 and header[1] in range(2, 6):
        return "pickle_arbitrary_code_execution"

    # ZIP-based (PyTorch .pt files are ZIP archives containing pickle)
    if header[:4] == _ZIP_MAGIC:
        # Further check: does the ZIP contain pickle files?
        pickle_inside = _detect_pickle_in_zip(file_path)
        if pickle_inside:
            return "pytorch_zip_pickle"
        return "zip_archive_no_pickle"

    # NumPy .npy magic
    if header[:6] == b"\x93NUMPY":
        return "numpy_allow_pickle_possible"

    # Fallback: check extension heuristic for known unsafe extensions
    ext = file_path.suffix.lower()
    if ext in {".pkl", ".pickle", ".joblib"}:
        return "pickle_format_by_extension"

    return None


def _detect_pickle_in_zip(file_path: Path) -> bool:
    """Check if a ZIP archive contains pickle-serialized files.

    PyTorch .pt/.pth files are ZIP archives that typically contain
    pickle-serialized tensors (data.pkl, archive/data.pkl, etc.).
    """
    try:
        if not zipfile.is_zipfile(file_path):
            return False
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in zf.namelist():
                lower_name = name.lower()
                # Check for pickle files inside the archive
                if lower_name.endswith((".pkl", ".pickle")):
                    return True
                # Check for data.pkl (PyTorch convention)
                if "data.pkl" in lower_name:
                    return True
                # Read first bytes of each file to check for pickle magic
                try:
                    with zf.open(name) as member:
                        member_header = member.read(4)
                        if (
                            len(member_header) >= 2
                            and member_header[0:1] == _PICKLE_MAGIC
                            and member_header[1] in range(2, 6)
                        ):
                            return True
                except (KeyError, RuntimeError, zipfile.BadZipFile):
                    continue
    except (zipfile.BadZipFile, OSError):
        pass
    return False


def _validate_onnx(file_path: Path) -> str | None:
    """Validate ONNX file structural integrity.

    Checks that the file starts with a valid protobuf field tag for
    ir_version (the first field in ONNX ModelProto). This is a lightweight
    structural check — full validation requires the onnx library.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
        if not header:
            return "Empty ONNX file"
        # ONNX ModelProto starts with field 1 (ir_version, varint)
        # Wire type 0, field number 1 → tag byte 0x08
        if header[0:1] != _ONNX_MAGIC:
            return f"Invalid ONNX header: expected 0x08, got 0x{header[0]:02x}"
        return None
    except (OSError, PermissionError) as e:
        return f"Cannot read ONNX file: {e}"


def _validate_safetensors(file_path: Path) -> str | None:
    """Validate SafeTensors file header.

    SafeTensors files start with an 8-byte little-endian u64 indicating the
    JSON header length, followed by a JSON object (starting with '{').
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        if len(header) < 9:
            return "SafeTensors file too small"
        # First 8 bytes are header length (u64 LE)
        header_len = struct.unpack("<Q", header[:8])[0]
        if header_len == 0 or header_len > 100_000_000:  # 100MB header is suspicious
            return f"Suspicious SafeTensors header length: {header_len}"
        # Byte after length should be '{' (start of JSON header)
        if header[8] != _SAFETENSORS_HEADER_CHAR:
            return f"Invalid SafeTensors header: expected '{{', got {chr(header[8])!r}"
        return None
    except (OSError, PermissionError, struct.error) as e:
        return f"Cannot validate SafeTensors: {e}"


def _elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
