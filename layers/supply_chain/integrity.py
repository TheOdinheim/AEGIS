"""
Stage 1: Cryptographic Integrity — SHA-256 hash verification.

Biological Analog: Secretory IgA screening at mucosal portals.

Verifies that every model file matches its expected SHA-256 hash from a
manifest. Optionally validates Sigstore signatures against a transparency log.

ASSUMED-BREACH POSTURE: The manifest itself could be tampered with. In
production, manifests should be signed and stored separately from model files.
This stage treats any hash mismatch as a hard failure — no partial trust.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from aegis.layers.supply_chain.models import StageResult

logger = logging.getLogger(__name__)

# 8 KB read chunks for hashing large files
_HASH_CHUNK_SIZE = 8192


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_integrity(
    model_path: Path,
    manifest: dict[str, str],
    *,
    sigstore_verify: bool = False,
) -> StageResult:
    """Run Stage 1: Cryptographic Integrity verification.

    Args:
        model_path: Directory containing model files.
        manifest: Mapping of relative filename → expected SHA-256 hex digest.
        sigstore_verify: If True, attempt Sigstore signature validation
                         (placeholder — requires sigstore-python in production).

    Returns:
        StageResult with pass/fail and per-file details.
    """
    start = time.perf_counter()
    details: dict[str, Any] = {"files_checked": 0, "files_passed": 0, "mismatches": [], "missing": []}

    if not manifest:
        return StageResult(
            stage="cryptographic_integrity",
            status="fail",
            details={"error": "Empty manifest — no hashes to verify"},
            duration_ms=_elapsed(start),
        )

    for filename, expected_hash in manifest.items():
        file_path = model_path / filename
        details["files_checked"] += 1

        if not file_path.exists():
            details["missing"].append(filename)
            continue

        actual_hash = compute_sha256(file_path)
        if actual_hash == expected_hash:
            details["files_passed"] += 1
        else:
            details["mismatches"].append({
                "file": filename,
                "expected": expected_hash,
                "actual": actual_hash,
            })

    # Determine overall status
    if details["missing"] or details["mismatches"]:
        status = "fail"
    else:
        status = "pass"

    # Sigstore placeholder
    if sigstore_verify:
        details["sigstore"] = "not_implemented"
        logger.info("Sigstore verification requested but not yet implemented")

    return StageResult(
        stage="cryptographic_integrity",
        status=status,
        details=details,
        duration_ms=_elapsed(start),
    )


def _elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
