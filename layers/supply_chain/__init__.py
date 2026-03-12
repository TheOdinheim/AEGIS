"""
Model Supply Chain Verification Pipeline — Mucosal Immunity.

Biological Analog: Secretory IgA and specialized immune cells screen everything
entering via high-traffic mucosal portals (gut, lungs, eyes).

Orchestrates a four-stage verification pipeline:
    1. Cryptographic Integrity  — SHA-256 hash verification + Sigstore signatures
    2. Serialization Safety     — Detect unsafe formats (pickle, joblib) + risk scoring
    3. Dependency Audit         — CVE matching + SBOM generation
    4. Behavioral Probing       — Jailbreak / sleeper agent / consistency testing

A model must pass all four stages before being approved for production use.
Any stage failure results in overall pipeline failure.

Enhanced with:
- SigstoreVerifier integration for cryptographic model verification
- Aggregate risk_score (weighted: integrity=0.3, serialization=0.3,
  dependency=0.2, behavioral=0.2)
- Critical finding escalation (risk_score >= 0.9 → immediate alert)
- Enhanced API response with risk scores

ASSUMED-BREACH POSTURE: Every model artifact is assumed hostile until proven
otherwise. The pipeline runs all four stages regardless of intermediate results
to provide a complete audit trail — early-exit would hide useful diagnostic data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from aegis.layers.supply_chain.behavioral_probe import probe_behavior
from aegis.layers.supply_chain.dependency_audit import (
    audit_dependencies,
    load_cve_database,
)
from aegis.layers.supply_chain.integrity import verify_integrity
from aegis.layers.supply_chain.models import ModelVerificationReport, StageResult
from aegis.layers.supply_chain.serialization import scan_serialization
from aegis.layers.supply_chain.sigstore_verifier import SigstoreVerifier

logger = logging.getLogger(__name__)

# Weighted risk score components
_WEIGHT_INTEGRITY = 0.3
_WEIGHT_SERIALIZATION = 0.3
_WEIGHT_DEPENDENCY = 0.2
_WEIGHT_BEHAVIORAL = 0.2

# Critical risk threshold — triggers immediate escalation
_CRITICAL_RISK_THRESHOLD = 0.9


def _stage_risk_score(stage: StageResult) -> float:
    """Compute a 0.0-1.0 risk score from a StageResult.

    Maps status to base score, then adjusts from stage-specific details.
    """
    base_scores = {"pass": 0.0, "warn": 0.4, "fail": 0.8}
    score = base_scores.get(stage.status, 0.5)

    # Stage-specific adjustments
    if stage.stage == "serialization_safety":
        aggregate_risk = stage.details.get("aggregate_risk", 0.0)
        if aggregate_risk > score:
            score = aggregate_risk

    elif stage.stage == "dependency_audit":
        severity_counts = stage.details.get("severity_counts", {})
        if severity_counts.get("CRITICAL", 0) > 0:
            score = max(score, 0.95)
        elif severity_counts.get("HIGH", 0) > 0:
            score = max(score, 0.85)

    elif stage.stage == "behavioral_probing":
        jailbreak_rate = stage.details.get("jailbreak_pass_rate", 0.0)
        sleeper_results = stage.details.get("sleeper_results", [])
        jailbreak_results = stage.details.get("jailbreak_results", [])

        # Only count jailbreak risk if probes were actually executed
        jailbreak_executed = sum(
            1 for r in jailbreak_results if r.get("status") != "skipped"
        )
        if jailbreak_executed > 0 and jailbreak_rate < 1.0:
            score = max(score, 1.0 - jailbreak_rate)

        # Any sleeper indicator is critical
        if any(r.get("alert") for r in sleeper_results):
            score = max(score, 0.95)

    elif stage.stage == "cryptographic_integrity":
        details = stage.details
        if isinstance(details, dict):
            if details.get("mismatches"):
                score = max(score, 0.9)
            if details.get("missing"):
                score = max(score, 0.85)

    return min(score, 1.0)


class SupplyChainVerifier:
    """Orchestrator for the four-stage model supply chain verification pipeline.

    Usage:
        verifier = SupplyChainVerifier(cve_file=Path("data/known_cves.json"))
        report = verifier.verify(
            model_id="my-model-v1",
            model_path=Path("/models/my-model"),
            manifest={"weights.safetensors": "abc123..."},
            dependencies={"torch": "2.1.0", "transformers": "4.35.0"},
        )
        if report.overall_status == "fail":
            print("Model rejected:", report.summary)
    """

    def __init__(
        self,
        *,
        cve_file: Path | None = None,
        allowlisted_sources: set[str] | None = None,
    ) -> None:
        self._cve_database: list[dict[str, Any]] = []
        self._allowlisted_sources = allowlisted_sources or set()
        self._reports: dict[str, ModelVerificationReport] = {}
        self._sigstore = SigstoreVerifier()

        if cve_file:
            self._cve_database = load_cve_database(cve_file)
            logger.info("Loaded %d CVEs from %s", len(self._cve_database), cve_file)

    @property
    def cve_count(self) -> int:
        return len(self._cve_database)

    @property
    def sigstore_available(self) -> bool:
        return self._sigstore.sigstore_available

    def get_report(self, model_id: str) -> ModelVerificationReport | None:
        """Retrieve a previously generated verification report."""
        return self._reports.get(model_id)

    def verify(
        self,
        model_id: str,
        *,
        model_path: Path | None = None,
        manifest: dict[str, str] | None = None,
        dependencies: dict[str, str] | None = None,
        source: str | None = None,
        model_fn: Callable[[str], str] | None = None,
        model_url: str | None = None,
        behavioral_responses: dict[str, str] | None = None,
        sigstore_verify: bool = False,
    ) -> ModelVerificationReport:
        """Run the full four-stage verification pipeline.

        All four stages run regardless of intermediate failures to produce a
        complete audit trail.

        Args:
            model_id: Unique identifier for the model.
            model_path: Path to model directory (for stages 1 & 2).
            manifest: SHA-256 hash manifest for stage 1.
            dependencies: Package dependencies for stage 3.
            source: Declared model source/publisher.
            model_fn: Live model callable for stage 4.
            model_url: URL for remote model probing in stage 4.
            behavioral_responses: Pre-computed probe responses for stage 4.
            sigstore_verify: Enable Sigstore verification in stage 1.

        Returns:
            ModelVerificationReport with overall and per-stage results.
        """
        start = time.perf_counter()
        stages: list[StageResult] = []

        # Stage 1: Cryptographic Integrity
        if model_path and manifest:
            stage1 = verify_integrity(
                model_path, manifest, sigstore_verify=sigstore_verify,
            )
        else:
            stage1 = StageResult(
                stage="cryptographic_integrity",
                status="warn",
                details={"note": "Skipped — no model path or manifest provided"},
            )
        stages.append(stage1)

        # Stage 2: Serialization Safety
        if model_path:
            stage2 = scan_serialization(
                model_path,
                allowlisted_sources=self._allowlisted_sources,
                source=source,
            )
        else:
            stage2 = StageResult(
                stage="serialization_safety",
                status="warn",
                details={"note": "Skipped — no model path provided"},
            )
        stages.append(stage2)

        # Stage 3: Dependency Audit
        if dependencies:
            stage3 = audit_dependencies(dependencies, self._cve_database)
        else:
            stage3 = StageResult(
                stage="dependency_audit",
                status="warn",
                details={"note": "Skipped — no dependencies provided"},
            )
        stages.append(stage3)

        # Stage 4: Behavioral Probing
        stage4 = probe_behavior(
            model_fn=model_fn,
            model_url=model_url,
            responses=behavioral_responses,
        )
        stages.append(stage4)

        # Determine overall status
        statuses = [s.status for s in stages]
        if "fail" in statuses:
            overall = "fail"
        elif "warn" in statuses:
            overall = "warn"
        else:
            overall = "pass"

        total_ms = round((time.perf_counter() - start) * 1000, 2)

        # Compute aggregate risk score
        stage_scores = {s.stage: _stage_risk_score(s) for s in stages}
        risk_score = (
            stage_scores.get("cryptographic_integrity", 0.0) * _WEIGHT_INTEGRITY
            + stage_scores.get("serialization_safety", 0.0) * _WEIGHT_SERIALIZATION
            + stage_scores.get("dependency_audit", 0.0) * _WEIGHT_DEPENDENCY
            + stage_scores.get("behavioral_probing", 0.0) * _WEIGHT_BEHAVIORAL
        )
        risk_score = round(min(risk_score, 1.0), 4)

        # Critical finding escalation
        critical_findings: list[str] = []
        if risk_score >= _CRITICAL_RISK_THRESHOLD:
            critical_findings.append(
                f"Aggregate risk score {risk_score} >= {_CRITICAL_RISK_THRESHOLD}"
            )
        for stage in stages:
            s_score = stage_scores[stage.stage]
            if s_score >= _CRITICAL_RISK_THRESHOLD:
                critical_findings.append(
                    f"{stage.stage}: risk {s_score:.2f} (status={stage.status})"
                )

        # Build summary
        failed = [s.stage for s in stages if s.status == "fail"]
        warned = [s.stage for s in stages if s.status == "warn"]
        if critical_findings:
            summary = (
                f"CRITICAL: {len(critical_findings)} critical finding(s) — "
                f"risk score {risk_score}. "
                + "; ".join(critical_findings)
            )
            if overall != "fail":
                overall = "fail"
        elif failed:
            summary = f"REJECTED: {len(failed)} stage(s) failed: {', '.join(failed)}"
        elif warned:
            summary = f"WARNING: {len(warned)} stage(s) raised warnings: {', '.join(warned)}"
        else:
            summary = "APPROVED: All 4 stages passed verification"

        report = ModelVerificationReport(
            model_id=model_id,
            overall_status=overall,
            stages=stages,
            total_duration_ms=total_ms,
            summary=summary,
            risk_score=risk_score,
            stage_risk_scores=stage_scores,
            critical_findings=critical_findings,
        )

        self._reports[model_id] = report
        logger.info(
            "Supply chain verification for %s: %s (risk=%.3f, %.1fms)",
            model_id, overall, risk_score, total_ms,
        )

        if critical_findings:
            logger.warning(
                "CRITICAL supply chain findings for %s: %s",
                model_id, "; ".join(critical_findings),
            )

        return report
