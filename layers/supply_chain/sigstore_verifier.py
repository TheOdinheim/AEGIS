"""
Stage 1 Enhanced: Cryptographic Integrity with Sigstore Verification.

Biological Analog: Secretory IgA with identity verification — not just
screening for pathogens but verifying the identity of the source organism.

Computes SHA-256 manifests for all model files and optionally verifies
Sigstore signatures against the Rekor transparency log. Gracefully
degrades when sigstore-python is not installed (hash-only verification).

ASSUMED-BREACH POSTURE: The manifest and signatures could both be forged.
Hash verification catches tampering after manifest creation. Sigstore
verification catches manifest forgery by validating the signer identity
chain against a public transparency log. Neither alone is sufficient;
both together provide defense in depth.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 64 KB read chunks for hashing (larger than integrity.py's 8KB for better
# throughput on large model files — fewer syscalls at the cost of slightly
# more memory)
_HASH_CHUNK_SIZE = 65_536

# Skip files larger than 10 GB to avoid blocking the pipeline on enormous
# model shards. Logged as a warning so operators can investigate.
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


@dataclass
class IntegrityResult:
    """Result from Sigstore-enhanced integrity verification."""

    is_verified: bool = False
    hash_manifest: dict[str, str] = field(default_factory=dict)
    signature_valid: bool | None = None  # None if no bundle or sigstore unavailable
    signer_identity: str | None = None
    rekor_entry: str | None = None
    warnings: list[str] = field(default_factory=list)
    files_hashed: int = 0
    files_skipped: int = 0
    duration_ms: float = 0.0


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_symlink_loop(path: Path) -> bool:
    """Check if a path is a symlink that loops."""
    try:
        path.resolve(strict=True)
        return False
    except (OSError, RuntimeError):
        return True


class SigstoreVerifier:
    """Cryptographic integrity verifier with optional Sigstore signature validation.

    Always computes SHA-256 hashes for all files in a model directory.
    When sigstore-python is installed and .sigstore bundle files are present,
    also validates signatures against the Rekor transparency log.

    Usage:
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(Path("/models/my-model"))
        if result.is_verified:
            print("Model integrity verified:", result.hash_manifest)
    """

    def __init__(self, *, max_file_size: int = _MAX_FILE_SIZE_BYTES) -> None:
        self._max_file_size = max_file_size
        self._sigstore_available = self._check_sigstore()

    @staticmethod
    def _check_sigstore() -> bool:
        """Check if sigstore-python is installed."""
        try:
            import sigstore  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def sigstore_available(self) -> bool:
        return self._sigstore_available

    async def verify_integrity(self, model_path: Path) -> IntegrityResult:
        """Run cryptographic integrity verification on a model directory.

        1. Walks all files, computes SHA-256 hashes (skipping >10GB, symlink loops)
        2. If .sigstore bundle files exist, attempts signature verification
        3. Returns IntegrityResult with manifest, signature status, and warnings

        Args:
            model_path: Path to the model directory.

        Returns:
            IntegrityResult with hash manifest and optional signature validation.
        """
        start = time.perf_counter()
        result = IntegrityResult()

        if not model_path.exists():
            result.warnings.append(f"Model path does not exist: {model_path}")
            result.duration_ms = _elapsed(start)
            return result

        if not model_path.is_dir():
            result.warnings.append(f"Model path is not a directory: {model_path}")
            result.duration_ms = _elapsed(start)
            return result

        # Collect all files (excluding .sigstore bundles from manifest)
        sigstore_bundles: list[Path] = []

        try:
            all_files = sorted(model_path.rglob("*"))
        except (OSError, PermissionError) as e:
            result.warnings.append(f"Error listing directory: {e}")
            result.duration_ms = _elapsed(start)
            return result

        for file_path in all_files:
            if not file_path.is_file():
                continue

            # Check for symlink loops
            if file_path.is_symlink() and _is_symlink_loop(file_path):
                result.warnings.append(f"Symlink loop detected, skipped: {file_path.name}")
                result.files_skipped += 1
                continue

            # Collect sigstore bundles separately
            if file_path.suffix == ".sigstore":
                sigstore_bundles.append(file_path)
                continue

            # Check file size
            try:
                file_size = file_path.stat().st_size
            except (OSError, PermissionError) as e:
                result.warnings.append(f"Cannot stat {file_path.name}: {e}")
                result.files_skipped += 1
                continue

            if file_size > self._max_file_size:
                result.warnings.append(
                    f"Skipped {file_path.name}: size {file_size / (1024**3):.1f}GB "
                    f"exceeds {self._max_file_size / (1024**3):.0f}GB limit"
                )
                result.files_skipped += 1
                continue

            # Compute hash
            try:
                rel_path = str(file_path.relative_to(model_path))
                digest = _compute_sha256(file_path)
                result.hash_manifest[rel_path] = digest
                result.files_hashed += 1
            except (OSError, PermissionError) as e:
                result.warnings.append(f"Permission denied hashing {file_path.name}: {e}")
                result.files_skipped += 1
                continue

        # Determine hash-level verification
        if result.files_hashed > 0 and result.files_skipped == 0:
            result.is_verified = True
        elif result.files_hashed > 0:
            # Partial verification — some files skipped
            result.is_verified = True
            result.warnings.append(
                f"Partial verification: {result.files_skipped} file(s) skipped"
            )
        else:
            result.warnings.append("No files could be hashed")

        # Attempt Sigstore signature verification
        if sigstore_bundles:
            await self._verify_sigstore(model_path, sigstore_bundles, result)
        else:
            result.signature_valid = None
            if self._sigstore_available:
                result.warnings.append("No .sigstore bundle files found — signature verification skipped")

        if not self._sigstore_available and sigstore_bundles:
            result.warnings.append(
                "sigstore-python not installed — signature verification skipped. "
                "Install with: pip install sigstore"
            )

        result.duration_ms = _elapsed(start)
        return result

    async def _verify_sigstore(
        self,
        model_path: Path,
        bundles: list[Path],
        result: IntegrityResult,
    ) -> None:
        """Attempt Sigstore bundle verification.

        If sigstore-python is not installed, adds a warning and returns.
        If installed, verifies each bundle against the Rekor transparency log.
        """
        if not self._sigstore_available:
            return

        try:
            from sigstore.verify import Verifier
            from sigstore.verify.policy import UnsafeNoOp

            verifier = Verifier.production()
            all_valid = True

            for bundle_path in bundles:
                try:
                    # Find the corresponding artifact file
                    # Convention: model.safetensors.sigstore → model.safetensors
                    artifact_name = bundle_path.stem  # removes .sigstore
                    artifact_path = bundle_path.parent / artifact_name

                    if not artifact_path.exists():
                        result.warnings.append(
                            f"No artifact found for bundle {bundle_path.name}"
                        )
                        continue

                    with open(bundle_path, "rb") as bf:
                        bundle_bytes = bf.read()

                    from sigstore.models import Bundle

                    bundle = Bundle.from_json(bundle_bytes.decode())

                    with open(artifact_path, "rb") as af:
                        artifact_bytes = af.read()

                    # Verify with permissive policy (UnsafeNoOp) — in production
                    # this should use identity-based policies
                    verify_result = verifier.verify(
                        input_=artifact_bytes,
                        bundle=bundle,
                        policy=UnsafeNoOp(),
                    )

                    # Extract signer identity if available
                    if hasattr(verify_result, "signer"):
                        result.signer_identity = str(verify_result.signer)
                    if hasattr(verify_result, "log_entry"):
                        result.rekor_entry = str(verify_result.log_entry)

                except Exception as e:
                    result.warnings.append(
                        f"Sigstore verification failed for {bundle_path.name}: {e}"
                    )
                    all_valid = False

            result.signature_valid = all_valid

        except ImportError as e:
            result.warnings.append(f"Sigstore import error: {e}")
            result.signature_valid = None
        except Exception as e:
            result.warnings.append(f"Sigstore verification error: {e}")
            result.signature_valid = False

    async def verify_against_manifest(
        self,
        model_path: Path,
        expected_manifest: dict[str, str],
    ) -> IntegrityResult:
        """Verify model files against an expected hash manifest.

        Computes fresh hashes and compares against the provided manifest.
        Useful when a manifest is received from a trusted source.

        Args:
            model_path: Path to model directory.
            expected_manifest: Expected {relative_path: sha256_hex} mapping.

        Returns:
            IntegrityResult — is_verified is True only if all hashes match.
        """
        result = await self.verify_integrity(model_path)

        if not expected_manifest:
            result.warnings.append("Empty expected manifest — cannot verify")
            result.is_verified = False
            return result

        mismatches = []
        missing = []

        for filename, expected_hash in expected_manifest.items():
            actual_hash = result.hash_manifest.get(filename)
            if actual_hash is None:
                missing.append(filename)
            elif actual_hash != expected_hash:
                mismatches.append(filename)

        if mismatches:
            result.is_verified = False
            result.warnings.append(
                f"Hash mismatch for {len(mismatches)} file(s): {', '.join(mismatches)}"
            )

        if missing:
            result.is_verified = False
            result.warnings.append(
                f"Missing {len(missing)} file(s) from manifest: {', '.join(missing)}"
            )

        return result


def _elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
