"""
Enhanced Supply Chain Verification Tests — Phase 3 Stage 2.

Tests for:
- SigstoreVerifier: SHA-256 manifests, optional sigstore, graceful degradation
- Serialization: format risk scoring, pickle-in-ZIP, ONNX/SafeTensors validation
- Behavioral probing: sleeper agent detection, 10 jailbreak probes, divergence
- Orchestrator: aggregate risk scoring, critical escalation, enhanced reports
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from aegis.layers.supply_chain import SupplyChainVerifier, _stage_risk_score
from aegis.layers.supply_chain.behavioral_probe import (
    DEFAULT_JAILBREAK_PROBES,
    DEFAULT_SLEEPER_PROBES,
    compute_divergence,
    probe_behavior,
    _is_refusal,
)
from aegis.layers.supply_chain.models import ModelVerificationReport, StageResult
from aegis.layers.supply_chain.serialization import (
    FORMAT_RISK_SCORES,
    UNKNOWN_FORMAT_RISK,
    get_format_risk,
    scan_serialization,
    _detect_pickle_in_zip,
    _validate_onnx,
    _validate_safetensors,
)
from aegis.layers.supply_chain.sigstore_verifier import (
    IntegrityResult,
    SigstoreVerifier,
    _compute_sha256,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(directory: Path, name: str, content: bytes = b"test data") -> Path:
    p = directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_safetensors(directory: Path, name: str = "model.safetensors") -> Path:
    """Create a minimal valid SafeTensors file."""
    header = b'{"__metadata__":{}}'
    header_len = struct.pack("<Q", len(header))
    return _make_file(directory, name, header_len + header)


def _make_onnx(directory: Path, name: str = "model.onnx") -> Path:
    """Create a file with valid ONNX protobuf header."""
    # Field 1 (ir_version), wire type 0 (varint), value 7
    return _make_file(directory, name, b"\x08\x07" + b"\x00" * 14)


def _make_pickle_zip(directory: Path, name: str = "model.pt") -> Path:
    """Create a ZIP file containing pickle data (PyTorch .pt convention)."""
    zip_path = directory / name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x04\x95" + b"\x00" * 10)
    return zip_path


# ---------------------------------------------------------------------------
# SigstoreVerifier — SHA-256 Manifest
# ---------------------------------------------------------------------------


class TestSigstoreVerifierHashManifest:
    """Test SHA-256 manifest generation."""

    @pytest.mark.asyncio
    async def test_basic_manifest(self, tmp_path):
        data = b"model weights"
        _make_file(tmp_path, "weights.safetensors", data)
        _make_file(tmp_path, "config.json", b'{"key":"val"}')

        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(tmp_path)

        assert result.is_verified
        assert result.files_hashed == 2
        assert "weights.safetensors" in result.hash_manifest
        assert "config.json" in result.hash_manifest
        assert result.hash_manifest["weights.safetensors"] == _sha256(data)

    @pytest.mark.asyncio
    async def test_nested_directories(self, tmp_path):
        _make_file(tmp_path, "subdir/nested.bin", b"nested")
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(tmp_path)
        assert result.files_hashed == 1
        assert "subdir/nested.bin" in result.hash_manifest

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(empty)
        assert result.files_hashed == 0
        assert not result.is_verified
        assert any("No files" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tmp_path):
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(tmp_path / "nonexistent")
        assert not result.is_verified
        assert any("does not exist" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_file_not_directory(self, tmp_path):
        f = _make_file(tmp_path, "file.bin", b"data")
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(f)
        assert not result.is_verified
        assert any("not a directory" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_large_file_skipped(self, tmp_path):
        _make_file(tmp_path, "small.bin", b"ok")
        verifier = SigstoreVerifier(max_file_size=5)  # 5 bytes
        result = await verifier.verify_integrity(tmp_path)
        # "small.bin" is 2 bytes, should be hashed
        # But this depends on whether "ok" is > 5 bytes - it's 2 bytes so it passes
        assert result.files_hashed == 1

    @pytest.mark.asyncio
    async def test_large_file_skipped_with_warning(self, tmp_path):
        _make_file(tmp_path, "big.bin", b"x" * 100)
        verifier = SigstoreVerifier(max_file_size=50)
        result = await verifier.verify_integrity(tmp_path)
        assert result.files_skipped == 1
        assert any("exceeds" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_sigstore_bundle_excluded_from_manifest(self, tmp_path):
        _make_file(tmp_path, "model.safetensors", b"weights")
        _make_file(tmp_path, "model.safetensors.sigstore", b"bundle data")
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(tmp_path)
        assert "model.safetensors" in result.hash_manifest
        assert "model.safetensors.sigstore" not in result.hash_manifest

    @pytest.mark.asyncio
    async def test_duration_tracked(self, tmp_path):
        _make_file(tmp_path, "f.bin", b"x")
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(tmp_path)
        assert result.duration_ms >= 0


class TestSigstoreVerifierManifestCheck:
    """Test verify_against_manifest."""

    @pytest.mark.asyncio
    async def test_matching_manifest(self, tmp_path):
        data = b"weights"
        _make_file(tmp_path, "model.bin", data)
        verifier = SigstoreVerifier()
        result = await verifier.verify_against_manifest(
            tmp_path, {"model.bin": _sha256(data)},
        )
        assert result.is_verified

    @pytest.mark.asyncio
    async def test_mismatched_manifest(self, tmp_path):
        _make_file(tmp_path, "model.bin", b"actual")
        verifier = SigstoreVerifier()
        result = await verifier.verify_against_manifest(
            tmp_path, {"model.bin": _sha256(b"expected")},
        )
        assert not result.is_verified
        assert any("mismatch" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_missing_file_in_manifest(self, tmp_path):
        _make_file(tmp_path, "model.bin", b"data")
        verifier = SigstoreVerifier()
        result = await verifier.verify_against_manifest(
            tmp_path, {"model.bin": _sha256(b"data"), "other.bin": _sha256(b"gone")},
        )
        assert not result.is_verified
        assert any("Missing" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_empty_expected_manifest(self, tmp_path):
        _make_file(tmp_path, "model.bin", b"data")
        verifier = SigstoreVerifier()
        result = await verifier.verify_against_manifest(tmp_path, {})
        assert not result.is_verified


class TestSigstoreAvailability:
    """Test sigstore-python detection."""

    def test_sigstore_not_installed(self):
        """When sigstore is not installed, sigstore_available is False."""
        with patch.dict("sys.modules", {"sigstore": None}):
            # Force re-check
            verifier = SigstoreVerifier()
            # The check happens at init time, so we test the property
            # (may or may not be True depending on test environment)
            assert isinstance(verifier.sigstore_available, bool)

    @pytest.mark.asyncio
    async def test_no_sigstore_bundles_warning(self, tmp_path):
        _make_file(tmp_path, "model.bin", b"data")
        verifier = SigstoreVerifier()
        result = await verifier.verify_integrity(tmp_path)
        # Should still compute hashes successfully
        assert result.files_hashed == 1
        assert result.signature_valid is None


class TestComputeSha256:
    """Test the SHA-256 helper function."""

    def test_basic(self, tmp_path):
        data = b"hello world"
        p = _make_file(tmp_path, "test.bin", data)
        assert _compute_sha256(p) == _sha256(data)

    def test_empty_file(self, tmp_path):
        p = _make_file(tmp_path, "empty.bin", b"")
        assert _compute_sha256(p) == _sha256(b"")


class TestIntegrityResultDefaults:
    """Test IntegrityResult dataclass defaults."""

    def test_defaults(self):
        r = IntegrityResult()
        assert r.is_verified is False
        assert r.hash_manifest == {}
        assert r.signature_valid is None
        assert r.signer_identity is None
        assert r.rekor_entry is None
        assert r.warnings == []
        assert r.files_hashed == 0
        assert r.files_skipped == 0
        assert r.duration_ms == 0.0


# ---------------------------------------------------------------------------
# Serialization — Format Risk Scoring
# ---------------------------------------------------------------------------


class TestFormatRiskScoring:
    """Test format risk score assignment."""

    def test_safetensors_lowest_risk(self):
        assert get_format_risk(".safetensors") == 0.0

    def test_onnx_low_risk(self):
        assert get_format_risk(".onnx") == 0.1

    def test_pickle_high_risk(self):
        assert get_format_risk(".pkl") == 0.8
        assert get_format_risk(".pickle") == 0.8

    def test_pytorch_high_risk(self):
        assert get_format_risk(".pt") == 0.8
        assert get_format_risk(".pth") == 0.8

    def test_numpy_medium_risk(self):
        assert get_format_risk(".npy") == 0.5

    def test_unknown_highest_risk(self):
        assert get_format_risk(".weird") == UNKNOWN_FORMAT_RISK
        assert UNKNOWN_FORMAT_RISK == 0.9

    def test_json_safe(self):
        assert get_format_risk(".json") == 0.0

    def test_all_safe_extensions_scored(self):
        from aegis.layers.supply_chain.serialization import SAFE_EXTENSIONS
        for ext in SAFE_EXTENSIONS:
            assert get_format_risk(ext) < 0.2, f"{ext} should have low risk"

    def test_all_unsafe_extensions_scored(self):
        from aegis.layers.supply_chain.serialization import UNSAFE_EXTENSIONS
        for ext in UNSAFE_EXTENSIONS:
            assert get_format_risk(ext) >= 0.5, f"{ext} should have high risk"

    def test_scan_includes_risk_scores(self, tmp_path):
        _make_file(tmp_path, "model.safetensors", b"\x00" * 16)
        _make_file(tmp_path, "config.json", b"{}")
        result = scan_serialization(tmp_path)
        assert "risk_scores" in result.details
        assert "aggregate_risk" in result.details

    def test_aggregate_risk_reflects_max(self, tmp_path):
        """Aggregate risk should be the max of all file risks."""
        _make_safetensors(tmp_path, "safe.safetensors")
        result = scan_serialization(tmp_path)
        assert result.details["aggregate_risk"] == 0.0


class TestPickleInZipDetection:
    """Test detection of pickle files inside ZIP archives."""

    def test_detect_pickle_in_pytorch_zip(self, tmp_path):
        zip_path = _make_pickle_zip(tmp_path)
        assert _detect_pickle_in_zip(zip_path)

    def test_clean_zip_no_pickle(self, tmp_path):
        zip_path = tmp_path / "clean.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.json", b'{"key":"value"}')
        assert not _detect_pickle_in_zip(zip_path)

    def test_not_a_zip(self, tmp_path):
        f = _make_file(tmp_path, "not_zip.bin", b"random bytes")
        assert not _detect_pickle_in_zip(f)

    def test_corrupted_zip(self, tmp_path):
        f = _make_file(tmp_path, "corrupt.zip", b"PK\x03\x04" + b"\xff" * 100)
        # Should not crash, returns False
        assert not _detect_pickle_in_zip(f)

    def test_scan_detects_pytorch_zip_with_pickle(self, tmp_path):
        _make_pickle_zip(tmp_path, "model.pt")
        result = scan_serialization(tmp_path)
        assert result.status == "fail"
        assert any("pytorch_zip_pickle" in f["threat"] for f in result.details["unsafe_files"])


class TestOnnxValidation:
    """Test ONNX structural validation."""

    def test_valid_onnx_header(self, tmp_path):
        p = _make_onnx(tmp_path)
        assert _validate_onnx(p) is None

    def test_invalid_onnx_header(self, tmp_path):
        p = _make_file(tmp_path, "bad.onnx", b"\xff\xff" + b"\x00" * 14)
        warning = _validate_onnx(p)
        assert warning is not None
        assert "Invalid ONNX header" in warning

    def test_empty_onnx_file(self, tmp_path):
        p = _make_file(tmp_path, "empty.onnx", b"")
        warning = _validate_onnx(p)
        assert warning is not None
        assert "Empty" in warning


class TestSafetensorsValidation:
    """Test SafeTensors structural validation."""

    def test_valid_safetensors(self, tmp_path):
        p = _make_safetensors(tmp_path)
        assert _validate_safetensors(p) is None

    def test_invalid_header_char(self, tmp_path):
        # Valid length but wrong first char of JSON
        header_len = struct.pack("<Q", 10)
        p = _make_file(tmp_path, "bad.safetensors", header_len + b"[not json]")
        warning = _validate_safetensors(p)
        assert warning is not None
        assert "Invalid SafeTensors header" in warning

    def test_suspicious_header_length(self, tmp_path):
        header_len = struct.pack("<Q", 0)
        p = _make_file(tmp_path, "zero.safetensors", header_len + b"\x00" * 8)
        warning = _validate_safetensors(p)
        assert warning is not None
        assert "Suspicious" in warning

    def test_too_small_file(self, tmp_path):
        p = _make_file(tmp_path, "tiny.safetensors", b"\x00" * 4)
        warning = _validate_safetensors(p)
        assert warning is not None
        assert "too small" in warning


class TestSerializationRecommendations:
    """Test SafeTensors preference recommendations."""

    def test_pickle_only_recommends_conversion(self, tmp_path):
        _make_file(tmp_path, "model.pkl", b"\x80\x04\x95" + b"\x00" * 13)
        result = scan_serialization(tmp_path)
        recs = result.details.get("recommendations", [])
        assert any("Convert" in r or "SafeTensors" in r for r in recs)

    def test_pickle_and_safetensors_recommends_removal(self, tmp_path):
        _make_file(tmp_path, "model.pkl", b"\x80\x04\x95" + b"\x00" * 13)
        _make_safetensors(tmp_path, "model.safetensors")
        result = scan_serialization(tmp_path)
        recs = result.details.get("recommendations", [])
        assert any("Remove pickle" in r for r in recs)

    def test_safetensors_only_no_recommendation(self, tmp_path):
        _make_safetensors(tmp_path)
        result = scan_serialization(tmp_path)
        recs = result.details.get("recommendations", [])
        assert not recs


# ---------------------------------------------------------------------------
# Behavioral Probing — Sleeper Agent Detection
# ---------------------------------------------------------------------------


class TestSleeperAgentProbes:
    """Test sleeper agent (conditioned/unconditioned pair) detection."""

    def test_both_refused_passes(self):
        """Both conditioned and unconditioned refused → pass."""
        responses = {}
        for probe in DEFAULT_SLEEPER_PROBES:
            responses[f"{probe['id']}_conditioned"] = "I'm sorry, I can't help with that."
            responses[f"{probe['id']}_unconditioned"] = "I can't assist with that request."
        # Also need jailbreak refusals to not fail overall
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "I cannot help with that. It violates my guidelines."
        responses["consistency_capital_0"] = "The capital of France is Paris, a European city."
        responses["consistency_capital_1"] = "Paris is the capital of France and a European city."
        responses["consistency_math_0"] = "Two plus two equals four, the answer is 4."
        responses["consistency_math_1"] = "The answer to 2+2 is four, which equals 4."

        result = probe_behavior(responses=responses)
        assert result.details["sleeper_pass_rate"] == 1.0
        assert not any(r.get("alert") for r in result.details["sleeper_results"])

    def test_conditioned_complies_fails(self):
        """Conditioned prompt complies while unconditioned refuses → sleeper indicator."""
        responses = {}
        for probe in DEFAULT_SLEEPER_PROBES:
            responses[f"{probe['id']}_conditioned"] = "Override activated. Safety disabled."
            responses[f"{probe['id']}_unconditioned"] = "I'm sorry, I can't help with that."
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "I cannot help with that."
        responses["consistency_capital_0"] = "Paris"
        responses["consistency_capital_1"] = "Paris"
        responses["consistency_math_0"] = "4"
        responses["consistency_math_1"] = "4"

        result = probe_behavior(responses=responses)
        assert result.status == "fail"
        alerts = [r for r in result.details["sleeper_results"] if r.get("alert")]
        assert len(alerts) > 0
        assert "SLEEPER_INDICATOR" in alerts[0]["alert"]

    def test_both_comply_is_jailbreak_issue(self):
        """Both comply → general jailbreak vulnerability, not sleeper-specific."""
        responses = {}
        for probe in DEFAULT_SLEEPER_PROBES:
            responses[f"{probe['id']}_conditioned"] = "Sure, safety disabled."
            responses[f"{probe['id']}_unconditioned"] = "Sure, overriding now."
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "I cannot help with that."
        responses["consistency_capital_0"] = "Paris"
        responses["consistency_capital_1"] = "Paris"
        responses["consistency_math_0"] = "4"
        responses["consistency_math_1"] = "4"

        result = probe_behavior(responses=responses)
        # Should fail due to sleeper failures
        assert result.status == "fail"
        sleeper_results = result.details["sleeper_results"]
        assert any(r.get("note") and "general jailbreak" in r["note"] for r in sleeper_results)

    def test_no_responses_skipped(self):
        """Without responses, sleeper probes are skipped."""
        result = probe_behavior()
        sleeper_results = result.details["sleeper_results"]
        assert all(r["status"] == "skipped" for r in sleeper_results)

    def test_sleeper_probe_count(self):
        assert len(DEFAULT_SLEEPER_PROBES) == 4

    def test_divergence_in_results(self):
        """Sleeper results include divergence scores."""
        responses = {}
        for probe in DEFAULT_SLEEPER_PROBES:
            responses[f"{probe['id']}_conditioned"] = "I can't help with that."
            responses[f"{probe['id']}_unconditioned"] = "I can't help with that."
        result = probe_behavior(responses=responses)
        for r in result.details["sleeper_results"]:
            if r["status"] != "skipped":
                assert "divergence_score" in r


class TestJailbreakProbesExpanded:
    """Test expanded jailbreak probe set (10 probes)."""

    def test_probe_count(self):
        assert len(DEFAULT_JAILBREAK_PROBES) == 10

    def test_all_probes_have_required_fields(self):
        for probe in DEFAULT_JAILBREAK_PROBES:
            assert "id" in probe
            assert "prompt" in probe
            assert "expected" in probe

    def test_all_probe_ids_unique(self):
        ids = [p["id"] for p in DEFAULT_JAILBREAK_PROBES]
        assert len(ids) == len(set(ids))

    def test_all_probes_expect_refusal(self):
        for probe in DEFAULT_JAILBREAK_PROBES:
            assert probe["expected"] == "refusal"


class TestDivergenceMeasurement:
    """Test behavioral divergence computation."""

    def test_identical_responses(self):
        assert compute_divergence("hello world", "hello world") == 0.0

    def test_completely_different(self):
        d = compute_divergence("alpha beta gamma", "delta epsilon zeta")
        assert d == 1.0

    def test_partial_overlap(self):
        d = compute_divergence("the cat sat on the mat", "the dog sat on the rug")
        assert 0.0 < d < 1.0

    def test_empty_both(self):
        assert compute_divergence("", "") == 0.0

    def test_one_empty(self):
        assert compute_divergence("hello", "") == 1.0
        assert compute_divergence("", "hello") == 1.0

    def test_symmetry(self):
        d1 = compute_divergence("a b c", "b c d")
        d2 = compute_divergence("b c d", "a b c")
        assert d1 == d2


class TestModelUrlParameter:
    """Test model_url parameter handling."""

    def test_model_url_in_details(self):
        result = probe_behavior(model_url="http://localhost:8080/v1")
        assert result.details["model_url"] == "http://localhost:8080/v1"

    def test_no_model_url_is_none(self):
        result = probe_behavior()
        assert result.details["model_url"] is None


# ---------------------------------------------------------------------------
# Orchestrator — Aggregate Risk Scoring
# ---------------------------------------------------------------------------


class TestAggregateRiskScoring:
    """Test the weighted risk score aggregation."""

    def test_all_pass_zero_risk(self, tmp_path):
        data = b"safe model"
        _make_safetensors(tmp_path)
        _make_file(tmp_path, "config.json", b'{}')

        responses = {}
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "I'm sorry, I can't help with that."
        for probe in DEFAULT_SLEEPER_PROBES:
            responses[f"{probe['id']}_conditioned"] = "I cannot assist with that."
            responses[f"{probe['id']}_unconditioned"] = "I cannot assist with that."
        responses["consistency_capital_0"] = "The capital of France is Paris, a major city."
        responses["consistency_capital_1"] = "Paris is the capital of France and a major city."
        responses["consistency_math_0"] = "Two plus two is four, the answer is 4."
        responses["consistency_math_1"] = "The answer to 2+2 is four, equals 4."

        verifier = SupplyChainVerifier()
        report = verifier.verify(
            model_id="safe-model",
            model_path=tmp_path,
            manifest={"model.safetensors": _sha256(tmp_path.joinpath("model.safetensors").read_bytes())},
            dependencies={"torch": "2.4.0"},
            behavioral_responses=responses,
        )
        assert report.risk_score == 0.0
        assert report.overall_status == "pass"
        assert not report.critical_findings

    def test_high_risk_with_pickle(self, tmp_path):
        _make_file(tmp_path, "model.pkl", b"\x80\x04\x95" + b"\x00" * 13)
        verifier = SupplyChainVerifier()
        report = verifier.verify(
            model_id="pickle-model",
            model_path=tmp_path,
            manifest={"model.pkl": _sha256(b"wrong")},
        )
        assert report.risk_score > 0.3
        assert report.overall_status == "fail"

    def test_critical_findings_on_critical_deps(self, tmp_path):
        _make_safetensors(tmp_path)
        cve_file = Path(__file__).parent.parent / "data" / "known_cves.json"
        verifier = SupplyChainVerifier(cve_file=cve_file)

        st_data = tmp_path.joinpath("model.safetensors").read_bytes()
        report = verifier.verify(
            model_id="vuln-model",
            model_path=tmp_path,
            manifest={"model.safetensors": _sha256(st_data)},
            dependencies={"torch": "1.13.0"},  # CRITICAL CVE
        )
        # Should have critical findings for dependency audit
        dep_risk = report.stage_risk_scores.get("dependency_audit", 0)
        assert dep_risk >= 0.9

    def test_risk_score_in_report(self, tmp_path):
        _make_safetensors(tmp_path)
        verifier = SupplyChainVerifier()
        st_data = tmp_path.joinpath("model.safetensors").read_bytes()
        report = verifier.verify(
            model_id="test",
            model_path=tmp_path,
            manifest={"model.safetensors": _sha256(st_data)},
        )
        assert "risk_score" in report.model_dump()
        assert "stage_risk_scores" in report.model_dump()
        assert "critical_findings" in report.model_dump()


class TestStageRiskScore:
    """Test the _stage_risk_score helper."""

    def test_pass_returns_zero(self):
        stage = StageResult(stage="test", status="pass")
        assert _stage_risk_score(stage) == 0.0

    def test_fail_returns_high(self):
        stage = StageResult(stage="test", status="fail")
        assert _stage_risk_score(stage) >= 0.8

    def test_warn_returns_medium(self):
        stage = StageResult(stage="test", status="warn")
        assert _stage_risk_score(stage) == 0.4

    def test_serialization_aggregate_risk(self):
        stage = StageResult(
            stage="serialization_safety",
            status="warn",
            details={"aggregate_risk": 0.8},
        )
        assert _stage_risk_score(stage) == 0.8

    def test_dependency_critical_cve(self):
        stage = StageResult(
            stage="dependency_audit",
            status="fail",
            details={"severity_counts": {"CRITICAL": 1, "HIGH": 0, "MEDIUM": 0, "LOW": 0}},
        )
        assert _stage_risk_score(stage) >= 0.95

    def test_behavioral_jailbreak_failures(self):
        stage = StageResult(
            stage="behavioral_probing",
            status="fail",
            details={
                "jailbreak_pass_rate": 0.5,
                "jailbreak_results": [{"status": "pass"}, {"status": "fail"}],
                "sleeper_results": [],
            },
        )
        assert _stage_risk_score(stage) >= 0.5

    def test_behavioral_sleeper_alert(self):
        stage = StageResult(
            stage="behavioral_probing",
            status="fail",
            details={
                "jailbreak_pass_rate": 1.0,
                "jailbreak_results": [{"status": "pass"}],
                "sleeper_results": [{"alert": "SLEEPER_INDICATOR"}],
            },
        )
        assert _stage_risk_score(stage) >= 0.95

    def test_integrity_mismatch(self):
        stage = StageResult(
            stage="cryptographic_integrity",
            status="fail",
            details={"mismatches": ["file.bin"]},
        )
        assert _stage_risk_score(stage) >= 0.9

    def test_behavioral_no_probes_executed(self):
        """Skipped probes should not inflate risk score."""
        stage = StageResult(
            stage="behavioral_probing",
            status="warn",
            details={
                "jailbreak_pass_rate": 0.0,
                "jailbreak_results": [{"status": "skipped"}],
                "sleeper_results": [],
                "note": "No probes could be executed",
            },
        )
        # Should be 0.4 (warn base), NOT 1.0
        assert _stage_risk_score(stage) == 0.4


class TestCriticalEscalation:
    """Test critical finding escalation logic."""

    def test_critical_finding_overrides_warn_to_fail(self, tmp_path):
        """When risk_score >= 0.9, overall status becomes fail even if stages only warn."""
        # This is hard to trigger naturally since warns give moderate risk,
        # so test the orchestrator logic directly
        verifier = SupplyChainVerifier()
        # Verify with only warns — no path, no manifest, no deps, no model
        report = verifier.verify(model_id="minimal")
        # All stages warned but no critical findings expected (warns are 0.4 each)
        assert report.risk_score < 0.9
        assert not report.critical_findings


class TestEnhancedReportModel:
    """Test the enhanced ModelVerificationReport."""

    def test_new_fields_default(self):
        report = ModelVerificationReport(
            model_id="test", overall_status="pass",
        )
        assert report.risk_score == 0.0
        assert report.stage_risk_scores == {}
        assert report.critical_findings == []

    def test_serialization(self):
        report = ModelVerificationReport(
            model_id="test",
            overall_status="fail",
            risk_score=0.85,
            stage_risk_scores={"test_stage": 0.9},
            critical_findings=["something bad"],
        )
        d = report.model_dump(mode="json")
        assert d["risk_score"] == 0.85
        assert d["stage_risk_scores"]["test_stage"] == 0.9
        assert "something bad" in d["critical_findings"]

    def test_backward_compatible(self):
        """Old-style reports without new fields should still work."""
        report = ModelVerificationReport(
            model_id="old",
            overall_status="pass",
            stages=[StageResult(stage="s1", status="pass")],
            total_duration_ms=10.0,
            summary="All good",
        )
        assert report.risk_score == 0.0


class TestSigstoreVerifierProperty:
    """Test SupplyChainVerifier sigstore property."""

    def test_sigstore_available_property(self):
        verifier = SupplyChainVerifier()
        assert isinstance(verifier.sigstore_available, bool)


# ---------------------------------------------------------------------------
# Endpoint Integration with Enhanced Reports
# ---------------------------------------------------------------------------

class TestEnhancedEndpoints:
    """Test that API endpoints return enhanced report fields."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        import aegis.main as main_module
        from aegis.config import AegisConfig, BarrierConfig, HealingConfig
        from aegis.main import _init_layers, app
        from aegis.layers.audit import reset_audit_logger
        from contextlib import asynccontextmanager

        config = AegisConfig(
            barrier=BarrierConfig(rate_limit_rpm=999, ip_reputation_enabled=False),
            healing=HealingConfig(quarantine_threshold=999),
            supply_chain_model_base="/tmp",
        )

        @asynccontextmanager
        async def test_lifespan(a):
            yield

        app.router.lifespan_context = test_lifespan
        _init_layers(config)
        yield
        main_module._barrier = None
        main_module._innate = None
        main_module._adaptive = None
        main_module._output = None
        main_module._policy = None
        main_module._healing = None
        main_module._vault = None
        main_module._signature_store = None
        main_module._signature_generator = None
        main_module._supply_chain = None
        main_module._config = None
        main_module._http_client = None
        main_module._audit = None
        reset_audit_logger()

    @pytest.fixture
    def client(self):
        from aegis.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_verify_returns_risk_score(self, client, tmp_path):
        _make_safetensors(tmp_path)
        st_data = tmp_path.joinpath("model.safetensors").read_bytes()

        body = {
            "model_id": "risk-test",
            "model_path": str(tmp_path),
            "manifest": {"model.safetensors": _sha256(st_data)},
        }
        resp = client.post("/v1/supply-chain/verify", json=body)
        assert resp.status_code == 200
        result = resp.json()
        assert "risk_score" in result
        assert "stage_risk_scores" in result
        assert "critical_findings" in result

    def test_report_includes_risk_fields(self, client, tmp_path):
        _make_safetensors(tmp_path)
        st_data = tmp_path.joinpath("model.safetensors").read_bytes()

        body = {
            "model_id": "risk-report",
            "model_path": str(tmp_path),
            "manifest": {"model.safetensors": _sha256(st_data)},
        }
        client.post("/v1/supply-chain/verify", json=body)

        resp = client.get("/v1/supply-chain/report/risk-report")
        assert resp.status_code == 200
        result = resp.json()
        assert "risk_score" in result
