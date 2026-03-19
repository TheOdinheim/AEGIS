"""
Model Provenance Validator (MPV) — Extension 3.1

Five-check validation pipeline for model provenance before any model
artifact is loaded or served:

    1. Format Safety   — reuse serialization.py risk scoring
    2. Source Registry  — check publisher against trusted registries/orgs
    3. Hash Verification — reuse integrity.py SHA-256 verification
    4. Metadata Injection — scan model card / config for injection via L2 regex
    5. Namespace Integrity — typosquatting detection (Levenshtein distance ≤ 2)

ASSUMED-BREACH POSTURE: Model artifacts are untrusted by default. A
compromised model hub or poisoned registry can serve tampered weights,
inject instructions into model cards, or register typosquatted names.
MPV blocks loading before any deserialization occurs.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegis.layers.innate.regex_engine import normalize_text
from aegis.layers.supply_chain.integrity import compute_sha256
from aegis.layers.supply_chain.models import StageResult
from aegis.layers.supply_chain.serialization import get_format_risk, SAFE_EXTENSIONS, UNSAFE_EXTENSIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default trusted registries and organizations
# ---------------------------------------------------------------------------

DEFAULT_TRUSTED_REGISTRIES: list[str] = [
    "huggingface.co",
    "pytorch.org",
    "tensorflow.org",
    "onnx.ai",
]

DEFAULT_TRUSTED_ORGS: list[str] = [
    "meta-llama",
    "google",
    "microsoft",
    "openai",
    "mistralai",
    "anthropic",
    "ProtectAI",
    "sentence-transformers",
]


class ProvenanceVerdict(str, enum.Enum):
    TRUSTED = "trusted"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


@dataclass
class ProvenanceCheckResult:
    check_name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceReport:
    model_id: str
    verdict: ProvenanceVerdict
    checks: list[ProvenanceCheckResult] = field(default_factory=list)
    risk_score: float = 0.0
    validation_latency_ms: float = 0.0
    blocked: bool = False


class ModelProvenanceValidator:
    """Five-check model provenance validation pipeline.

    Reuses existing supply chain utilities (serialization risk scoring,
    SHA-256 hashing) and L2 regex engine for metadata injection scanning.
    """

    def __init__(
        self,
        *,
        regex_engine: Any | None = None,
        trusted_registries: list[str] | None = None,
        trusted_orgs: list[str] | None = None,
        block_untrusted: bool = False,
    ) -> None:
        self._regex_engine = regex_engine
        self._trusted_registries = set(trusted_registries or DEFAULT_TRUSTED_REGISTRIES)
        self._trusted_orgs = set(trusted_orgs or DEFAULT_TRUSTED_ORGS)
        self._block_untrusted = block_untrusted

    async def validate(
        self,
        model_id: str,
        *,
        model_path: Path | None = None,
        manifest: dict[str, str] | None = None,
        source_registry: str | None = None,
        source_org: str | None = None,
        model_card_text: str | None = None,
        config_text: str | None = None,
    ) -> ProvenanceReport:
        """Run all 5 provenance checks and return a report.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-3-8B").
            model_path: Path to model directory for format/hash checks.
            manifest: filename → SHA-256 hash mapping.
            source_registry: Registry hostname (e.g., "huggingface.co").
            source_org: Organization name (e.g., "meta-llama").
            model_card_text: README / model card content for injection scan.
            config_text: config.json or similar metadata text.
        """
        start = time.perf_counter()
        checks: list[ProvenanceCheckResult] = []
        risk_scores: list[float] = []

        # Check 1: Format Safety
        c1 = self._check_format_safety(model_path)
        checks.append(c1)
        risk_scores.append(c1.details.get("risk", 0.0) if not c1.passed else 0.0)

        # Check 2: Source Registry
        c2 = self._check_source_registry(source_registry, source_org)
        checks.append(c2)
        risk_scores.append(0.0 if c2.passed else 0.5)

        # Check 3: Hash Verification
        c3 = self._check_hash_verification(model_path, manifest)
        checks.append(c3)
        risk_scores.append(0.0 if c3.passed else 0.9)

        # Check 4: Metadata Injection
        c4 = await self._check_metadata_injection(model_card_text, config_text)
        checks.append(c4)
        risk_scores.append(c4.details.get("risk", 0.0))

        # Check 5: Namespace Integrity
        c5 = self._check_namespace_integrity(model_id, source_org)
        checks.append(c5)
        risk_scores.append(0.0 if c5.passed else 0.6)

        # Compute overall risk and verdict
        risk_score = max(risk_scores) if risk_scores else 0.0
        failed_checks = [c for c in checks if not c.passed]

        if risk_score >= 0.8 or any(
            c.check_name in ("hash_verification", "format_safety") and not c.passed
            for c in checks
        ):
            verdict = ProvenanceVerdict.REJECTED
        elif failed_checks:
            verdict = ProvenanceVerdict.UNVERIFIED
        else:
            verdict = ProvenanceVerdict.TRUSTED

        blocked = verdict == ProvenanceVerdict.REJECTED or (
            self._block_untrusted and verdict == ProvenanceVerdict.UNVERIFIED
        )

        elapsed = (time.perf_counter() - start) * 1000

        report = ProvenanceReport(
            model_id=model_id,
            verdict=verdict,
            checks=checks,
            risk_score=round(risk_score, 4),
            validation_latency_ms=round(elapsed, 2),
            blocked=blocked,
        )

        logger.info(
            "Provenance validation for %s: %s (risk=%.3f, blocked=%s, %.1fms)",
            model_id, verdict.value, risk_score, blocked, elapsed,
        )

        return report

    # ------------------------------------------------------------------
    # Check 1: Format Safety
    # ------------------------------------------------------------------

    def _check_format_safety(self, model_path: Path | None) -> ProvenanceCheckResult:
        if model_path is None or not model_path.exists():
            return ProvenanceCheckResult(
                check_name="format_safety",
                passed=True,
                details={"note": "skipped — no model path provided"},
            )

        files = [f for f in model_path.rglob("*") if f.is_file()]
        if not files:
            return ProvenanceCheckResult(
                check_name="format_safety",
                passed=True,
                details={"note": "no files found"},
            )

        unsafe_files: list[str] = []
        max_risk = 0.0

        for f in files:
            ext = f.suffix.lower()
            risk = get_format_risk(ext)
            max_risk = max(max_risk, risk)
            if ext in UNSAFE_EXTENSIONS:
                unsafe_files.append(str(f.relative_to(model_path)))

        passed = len(unsafe_files) == 0
        return ProvenanceCheckResult(
            check_name="format_safety",
            passed=passed,
            details={
                "files_scanned": len(files),
                "unsafe_files": unsafe_files,
                "max_format_risk": max_risk,
                "risk": max_risk if not passed else 0.0,
            },
        )

    # ------------------------------------------------------------------
    # Check 2: Source Registry
    # ------------------------------------------------------------------

    def _check_source_registry(
        self, source_registry: str | None, source_org: str | None,
    ) -> ProvenanceCheckResult:
        if source_registry is None and source_org is None:
            return ProvenanceCheckResult(
                check_name="source_registry",
                passed=False,
                details={"note": "no source information provided"},
            )

        registry_trusted = (
            source_registry is not None
            and source_registry.lower() in {r.lower() for r in self._trusted_registries}
        )
        org_trusted = (
            source_org is not None
            and source_org.lower() in {o.lower() for o in self._trusted_orgs}
        )

        passed = registry_trusted or org_trusted
        return ProvenanceCheckResult(
            check_name="source_registry",
            passed=passed,
            details={
                "source_registry": source_registry,
                "source_org": source_org,
                "registry_trusted": registry_trusted,
                "org_trusted": org_trusted,
            },
        )

    # ------------------------------------------------------------------
    # Check 3: Hash Verification
    # ------------------------------------------------------------------

    def _check_hash_verification(
        self, model_path: Path | None, manifest: dict[str, str] | None,
    ) -> ProvenanceCheckResult:
        if model_path is None or manifest is None:
            return ProvenanceCheckResult(
                check_name="hash_verification",
                passed=True,
                details={"note": "skipped — no path or manifest provided"},
            )

        mismatches: list[dict[str, str]] = []
        missing: list[str] = []
        verified = 0

        for filename, expected_hash in manifest.items():
            file_path = model_path / filename
            if not file_path.exists():
                missing.append(filename)
                continue
            actual = compute_sha256(file_path)
            if actual == expected_hash:
                verified += 1
            else:
                mismatches.append({
                    "file": filename,
                    "expected": expected_hash[:16] + "...",
                    "actual": actual[:16] + "...",
                })

        passed = len(mismatches) == 0 and len(missing) == 0
        return ProvenanceCheckResult(
            check_name="hash_verification",
            passed=passed,
            details={
                "files_checked": len(manifest),
                "verified": verified,
                "mismatches": mismatches,
                "missing": missing,
            },
        )

    # ------------------------------------------------------------------
    # Check 4: Metadata Injection
    # ------------------------------------------------------------------

    async def _check_metadata_injection(
        self, model_card_text: str | None, config_text: str | None,
    ) -> ProvenanceCheckResult:
        texts_to_scan: list[tuple[str, str]] = []
        if model_card_text:
            texts_to_scan.append(("model_card", model_card_text))
        if config_text:
            texts_to_scan.append(("config", config_text))

        if not texts_to_scan:
            return ProvenanceCheckResult(
                check_name="metadata_injection",
                passed=True,
                details={"note": "no metadata text provided"},
            )

        injection_findings: list[dict[str, Any]] = []

        for source_name, text in texts_to_scan:
            normalized = normalize_text(text)

            # Use L2 regex engine if available
            if self._regex_engine is not None:
                try:
                    scan_result = await self._regex_engine.scan(normalized)
                    if scan_result.is_threat:
                        injection_findings.append({
                            "source": source_name,
                            "patterns": list(scan_result.matched_patterns),
                            "confidence": scan_result.confidence,
                        })
                except Exception:
                    logger.debug("MPV regex scan failed for %s", source_name, exc_info=True)

        passed = len(injection_findings) == 0
        risk = 0.0
        if injection_findings:
            risk = max(f.get("confidence", 0.7) for f in injection_findings)

        return ProvenanceCheckResult(
            check_name="metadata_injection",
            passed=passed,
            details={
                "texts_scanned": len(texts_to_scan),
                "injection_findings": injection_findings,
                "risk": risk,
            },
        )

    # ------------------------------------------------------------------
    # Check 5: Namespace Integrity (typosquatting detection)
    # ------------------------------------------------------------------

    def _check_namespace_integrity(
        self, model_id: str, source_org: str | None,
    ) -> ProvenanceCheckResult:
        # Extract org from model_id if not provided
        org = source_org
        if org is None and "/" in model_id:
            org = model_id.split("/")[0]

        if org is None:
            return ProvenanceCheckResult(
                check_name="namespace_integrity",
                passed=True,
                details={"note": "no org to check"},
            )

        org_lower = org.lower()

        # Exact match = trusted
        if org_lower in {o.lower() for o in self._trusted_orgs}:
            return ProvenanceCheckResult(
                check_name="namespace_integrity",
                passed=True,
                details={"org": org, "exact_match": True},
            )

        # Check Levenshtein distance against trusted orgs
        close_matches: list[dict[str, Any]] = []
        for trusted_org in self._trusted_orgs:
            dist = _levenshtein_distance(org_lower, trusted_org.lower())
            if 1 <= dist <= 2:
                close_matches.append({
                    "claimed_org": org,
                    "similar_to": trusted_org,
                    "distance": dist,
                })

        if close_matches:
            return ProvenanceCheckResult(
                check_name="namespace_integrity",
                passed=False,
                details={
                    "org": org,
                    "potential_typosquatting": close_matches,
                },
            )

        return ProvenanceCheckResult(
            check_name="namespace_integrity",
            passed=True,
            details={"org": org, "exact_match": False, "no_close_matches": True},
        )


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]
