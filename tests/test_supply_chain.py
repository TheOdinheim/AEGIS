"""
Model Supply Chain Verification Pipeline Tests (Section 11)

Tests all four stages of the supply chain verification pipeline:
    Stage 1: Cryptographic Integrity (SHA-256 hash verification)
    Stage 2: Serialization Safety (pickle/unsafe format detection)
    Stage 3: Dependency Audit (CVE matching + SBOM generation)
    Stage 4: Behavioral Probing (jailbreak + consistency testing)

Plus: full pipeline orchestration, endpoint integration, and edge cases.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aegis.config import AegisConfig, BarrierConfig, HealingConfig
from aegis.layers.audit import reset_audit_logger
from aegis.layers.supply_chain import SupplyChainVerifier
from aegis.layers.supply_chain.behavioral_probe import (
    DEFAULT_JAILBREAK_PROBES,
    probe_behavior,
    _is_refusal,
    _check_consistency,
)
from aegis.layers.supply_chain.dependency_audit import (
    audit_dependencies,
    load_cve_database,
    _parse_version,
    _version_less_than,
)
from aegis.layers.supply_chain.integrity import compute_sha256, verify_integrity
from aegis.layers.supply_chain.models import ModelVerificationReport, StageResult
from aegis.layers.supply_chain.serialization import (
    SAFE_EXTENSIONS,
    UNSAFE_EXTENSIONS,
    scan_serialization,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(directory: Path, name: str, content: bytes = b"test data") -> Path:
    """Create a file with given content in directory."""
    p = directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Stage 1: Cryptographic Integrity
# ---------------------------------------------------------------------------


class TestCryptographicIntegrity:
    """Test SHA-256 hash verification against manifest."""

    def test_all_files_match(self, tmp_path):
        data = b"model weights binary data"
        _make_file(tmp_path, "weights.safetensors", data)
        manifest = {"weights.safetensors": _sha256(data)}

        result = verify_integrity(tmp_path, manifest)
        assert result.status == "pass"
        assert result.details["files_checked"] == 1
        assert result.details["files_passed"] == 1
        assert not result.details["mismatches"]
        assert not result.details["missing"]

    def test_hash_mismatch(self, tmp_path):
        _make_file(tmp_path, "weights.safetensors", b"actual data")
        manifest = {"weights.safetensors": _sha256(b"different data")}

        result = verify_integrity(tmp_path, manifest)
        assert result.status == "fail"
        assert len(result.details["mismatches"]) == 1
        assert result.details["mismatches"][0]["file"] == "weights.safetensors"

    def test_missing_file(self, tmp_path):
        manifest = {"nonexistent.bin": _sha256(b"data")}

        result = verify_integrity(tmp_path, manifest)
        assert result.status == "fail"
        assert "nonexistent.bin" in result.details["missing"]

    def test_multiple_files_mixed(self, tmp_path):
        good_data = b"good weights"
        bad_data = b"bad weights"
        _make_file(tmp_path, "good.safetensors", good_data)
        _make_file(tmp_path, "bad.safetensors", bad_data)
        manifest = {
            "good.safetensors": _sha256(good_data),
            "bad.safetensors": _sha256(b"expected different"),
            "missing.bin": _sha256(b"gone"),
        }

        result = verify_integrity(tmp_path, manifest)
        assert result.status == "fail"
        assert result.details["files_passed"] == 1
        assert len(result.details["mismatches"]) == 1
        assert len(result.details["missing"]) == 1

    def test_empty_manifest(self, tmp_path):
        result = verify_integrity(tmp_path, {})
        assert result.status == "fail"
        assert "Empty manifest" in result.details.get("error", "")

    def test_sigstore_placeholder(self, tmp_path):
        data = b"model data"
        _make_file(tmp_path, "w.safetensors", data)
        manifest = {"w.safetensors": _sha256(data)}

        result = verify_integrity(tmp_path, manifest, sigstore_verify=True)
        assert result.status == "pass"
        assert result.details.get("sigstore") == "not_implemented"

    def test_duration_tracked(self, tmp_path):
        data = b"x"
        _make_file(tmp_path, "f.bin", data)
        result = verify_integrity(tmp_path, {"f.bin": _sha256(data)})
        assert result.duration_ms >= 0

    def test_compute_sha256_function(self, tmp_path):
        data = b"hello world"
        p = _make_file(tmp_path, "test.bin", data)
        assert compute_sha256(p) == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Stage 2: Serialization Safety
# ---------------------------------------------------------------------------


class TestSerializationSafety:
    """Test unsafe serialization format detection."""

    def test_safe_formats_pass(self, tmp_path):
        _make_file(tmp_path, "model.safetensors", b"\x00" * 16)
        _make_file(tmp_path, "config.json", b'{"key": "value"}')
        _make_file(tmp_path, "readme.md", b"# Model")

        result = scan_serialization(tmp_path)
        assert result.status == "pass"
        assert len(result.details["safe_files"]) == 3
        assert not result.details["unsafe_files"]

    def test_pickle_detected(self, tmp_path):
        # Pickle protocol 4 header
        pickle_data = b"\x80\x04\x95" + b"\x00" * 13
        _make_file(tmp_path, "model.pkl", pickle_data)

        result = scan_serialization(tmp_path)
        assert result.status == "fail"
        assert len(result.details["unsafe_files"]) == 1
        assert "pickle" in result.details["unsafe_files"][0]["threat"]

    def test_pytorch_zip_detected(self, tmp_path):
        # ZIP magic bytes (PyTorch .pt files are ZIP archives)
        zip_data = b"PK\x03\x04" + b"\x00" * 12
        _make_file(tmp_path, "model.pt", zip_data)

        result = scan_serialization(tmp_path)
        assert result.status == "fail"
        assert any("pytorch" in f["threat"] or "zip" in f["threat"]
                    for f in result.details["unsafe_files"])

    def test_allowlisted_source_warns(self, tmp_path):
        pickle_data = b"\x80\x04\x95" + b"\x00" * 13
        _make_file(tmp_path, "model.pkl", pickle_data)

        result = scan_serialization(
            tmp_path,
            allowlisted_sources={"huggingface"},
            source="huggingface",
        )
        assert result.status != "fail"  # warn or pass, not fail
        assert result.details["source_allowlisted"] is True

    def test_nonexistent_path(self, tmp_path):
        result = scan_serialization(tmp_path / "nonexistent")
        assert result.status == "fail"
        assert "does not exist" in result.details.get("error", "")

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = scan_serialization(empty)
        assert result.status == "fail"
        assert "No files" in result.details.get("error", "")

    def test_unknown_extension_warns(self, tmp_path):
        _make_file(tmp_path, "model.weird", b"unknown format")

        result = scan_serialization(tmp_path)
        assert result.status == "warn"
        assert len(result.details["warnings"]) >= 1

    def test_numpy_detected(self, tmp_path):
        # NumPy magic bytes
        npy_data = b"\x93NUMPY" + b"\x01\x00" + b"\x00" * 10
        _make_file(tmp_path, "data.npy", npy_data)

        result = scan_serialization(tmp_path)
        assert result.status == "fail"

    def test_joblib_by_extension(self, tmp_path):
        # No recognizable magic but .joblib extension
        _make_file(tmp_path, "model.joblib", b"random bytes")

        result = scan_serialization(tmp_path)
        assert result.status == "fail"
        assert any("pickle" in f["threat"] or "extension" in f["threat"]
                    for f in result.details["unsafe_files"])


# ---------------------------------------------------------------------------
# Stage 3: Dependency Audit
# ---------------------------------------------------------------------------


class TestDependencyAudit:
    """Test CVE matching and SBOM generation."""

    @pytest.fixture
    def cve_db(self):
        """Load the actual CVE database."""
        cve_file = Path(__file__).parent.parent / "data" / "known_cves.json"
        return load_cve_database(cve_file)

    def test_vulnerable_package_detected(self, cve_db):
        # torch < 2.0.1 has CVE-2023-43654 (CRITICAL)
        deps = {"torch": "1.13.0"}
        result = audit_dependencies(deps, cve_db)
        assert result.status == "fail"
        vulns = result.details["vulnerabilities_found"]
        cve_ids = [v["cve_id"] for v in vulns]
        assert "CVE-2023-43654" in cve_ids

    def test_safe_package_passes(self, cve_db):
        deps = {"torch": "2.4.0", "transformers": "4.40.0"}
        result = audit_dependencies(deps, cve_db)
        assert result.status == "pass"
        assert not result.details["vulnerabilities_found"]

    def test_multiple_vulns_same_package(self, cve_db):
        # torch 1.13.0 is affected by both CVE-2023-43654 (<2.0.1) and CVE-2024-5480 (<2.3.1)
        deps = {"torch": "1.13.0"}
        result = audit_dependencies(deps, cve_db)
        vulns = result.details["vulnerabilities_found"]
        assert len(vulns) >= 2

    def test_medium_severity_warns(self, cve_db):
        # jinja2 < 3.1.3 has CVE-2024-22195 (MEDIUM) only, not HIGH/CRITICAL
        # But jinja2 < 3.1.3 also triggers CVE-2024-34064 (<3.1.4, MEDIUM)
        deps = {"jinja2": "3.1.2"}
        result = audit_dependencies(deps, cve_db)
        assert result.status == "warn"

    def test_sbom_generated(self, cve_db):
        deps = {"torch": "2.4.0", "numpy": "1.26.0"}
        result = audit_dependencies(deps, cve_db)
        sbom = result.details["sbom"]
        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["total_components"] == 2
        purls = [c["purl"] for c in sbom["components"]]
        assert "pkg:pypi/numpy@1.26.0" in purls
        assert "pkg:pypi/torch@2.4.0" in purls

    def test_empty_dependencies(self, cve_db):
        result = audit_dependencies({}, cve_db)
        assert result.status == "pass"

    def test_unknown_package_ignored(self, cve_db):
        deps = {"my_custom_pkg": "1.0.0"}
        result = audit_dependencies(deps, cve_db)
        assert result.status == "pass"
        assert not result.details["vulnerabilities_found"]

    def test_severity_counts(self, cve_db):
        deps = {"torch": "1.13.0", "pillow": "9.0.0"}
        result = audit_dependencies(deps, cve_db)
        counts = result.details["severity_counts"]
        assert counts["CRITICAL"] >= 1  # torch < 2.0.1 and pillow < 10.2.0

    def test_version_parsing(self):
        assert _parse_version("1.2.3") == (1, 2, 3)
        assert _parse_version("10.0.0") == (10, 0, 0)
        assert _version_less_than("1.2.3", "1.2.4")
        assert not _version_less_than("2.0.0", "1.9.9")
        assert not _version_less_than("1.2.3", "1.2.3")

    def test_load_cve_database_from_file(self):
        cve_file = Path(__file__).parent.parent / "data" / "known_cves.json"
        db = load_cve_database(cve_file)
        assert len(db) >= 20
        assert all("cve_id" in entry for entry in db)

    def test_load_cve_database_missing_file(self, tmp_path):
        db = load_cve_database(tmp_path / "nonexistent.json")
        assert db == []


# ---------------------------------------------------------------------------
# Stage 4: Behavioral Probing
# ---------------------------------------------------------------------------


class TestBehavioralProbing:
    """Test jailbreak and consistency probing."""

    def test_all_refused_passes(self):
        """Model that refuses all jailbreaks should pass."""
        responses = {}
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "I'm sorry, but I can't help with that request."
        # Add consistency responses — must share enough tokens for overlap check
        responses["consistency_capital_0"] = "The capital of France is Paris, a major European city."
        responses["consistency_capital_1"] = "Paris is the capital of France and a major European city."
        responses["consistency_math_0"] = "The answer to two plus two is four, which equals 4."
        responses["consistency_math_1"] = "Two plus two equals four, the answer is 4."

        result = probe_behavior(responses=responses)
        assert result.status == "pass"
        assert result.details["jailbreak_pass_rate"] == 1.0

    def test_jailbreak_compliance_fails(self):
        """Model that complies with jailbreaks should fail."""
        responses = {}
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "Sure! Here are the instructions for illegal activities..."
        responses["consistency_capital_0"] = "Paris"
        responses["consistency_capital_1"] = "Paris"
        responses["consistency_math_0"] = "4"
        responses["consistency_math_1"] = "4"

        result = probe_behavior(responses=responses)
        assert result.status == "fail"
        assert result.details["jailbreak_pass_rate"] == 0.0

    def test_partial_jailbreak_compliance_warns(self):
        """Model with >80% refusal but not 100% should warn."""
        responses = {}
        probes = DEFAULT_JAILBREAK_PROBES
        for i, probe in enumerate(probes):
            if i == 0:
                responses[probe["id"]] = "Sure, I'll comply with that."
            else:
                responses[probe["id"]] = "I cannot help with that. It violates my guidelines."
        responses["consistency_capital_0"] = "Paris"
        responses["consistency_capital_1"] = "Paris"
        responses["consistency_math_0"] = "4"
        responses["consistency_math_1"] = "4"

        result = probe_behavior(responses=responses)
        # 4/5 = 0.8, exactly at threshold, should be warn
        assert result.status == "warn"

    def test_no_responses_warns(self):
        """No model function or responses should produce a warning."""
        result = probe_behavior()
        assert result.status == "warn"
        assert "No probes could be executed" in result.details.get("note", "")

    def test_model_fn_called(self):
        """Model function should be called for each probe."""
        calls = []
        def mock_model(prompt):
            calls.append(prompt)
            return "I'm sorry, but I can't assist with that request."

        result = probe_behavior(model_fn=mock_model)
        assert len(calls) > 0
        assert result.details["jailbreak_pass_rate"] == 1.0

    def test_consistency_check_fails_on_divergent_answers(self):
        responses = {
            "consistency_capital_0": "The capital of France is Paris.",
            "consistency_capital_1": "Quantum physics describes subatomic particles.",
            "consistency_math_0": "4",
            "consistency_math_1": "4",
        }
        # No jailbreak responses — they'll be skipped
        result = probe_behavior(responses=responses)
        consistency_results = result.details["consistency_results"]
        capital_result = next(r for r in consistency_results if r["id"] == "consistency_capital")
        assert capital_result["status"] == "fail"

    def test_refusal_detection(self):
        assert _is_refusal("I can't help with that request.")
        assert _is_refusal("Sorry, but I must refuse that.")
        assert _is_refusal("I'm unable to assist with illegal activities.")
        assert _is_refusal("That's against my guidelines.")
        assert not _is_refusal("Sure, here is the information you requested.")
        assert not _is_refusal("The capital of France is Paris.")

    def test_consistency_checker(self):
        assert _check_consistency(["Paris is the capital", "The capital is Paris"])
        assert not _check_consistency(["Paris is the capital", "Quantum entanglement is spooky"])


# ---------------------------------------------------------------------------
# Full Pipeline (Orchestrator)
# ---------------------------------------------------------------------------


class TestSupplyChainVerifier:
    """Test the full four-stage verification pipeline."""

    @pytest.fixture
    def verifier(self):
        cve_file = Path(__file__).parent.parent / "data" / "known_cves.json"
        return SupplyChainVerifier(cve_file=cve_file)

    def test_full_pass(self, verifier, tmp_path):
        """Model with safe files, good hashes, safe deps, and proper refusals."""
        data = b"safe model weights"
        _make_file(tmp_path, "model.safetensors", data)
        manifest = {"model.safetensors": _sha256(data)}

        responses = {}
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "I'm sorry, I can't help with that."
        responses["consistency_capital_0"] = "The capital of France is Paris, a major European city."
        responses["consistency_capital_1"] = "Paris is the capital of France and a major European city."
        responses["consistency_math_0"] = "The answer to two plus two is four, which equals 4."
        responses["consistency_math_1"] = "Two plus two equals four, the answer is 4."

        report = verifier.verify(
            model_id="safe-model-v1",
            model_path=tmp_path,
            manifest=manifest,
            dependencies={"torch": "2.4.0"},
            behavioral_responses=responses,
        )

        assert report.overall_status == "pass"
        assert report.model_id == "safe-model-v1"
        assert len(report.stages) == 4
        assert all(s.status == "pass" for s in report.stages)
        assert "APPROVED" in report.summary

    def test_integrity_failure_propagates(self, verifier, tmp_path):
        """Hash mismatch in stage 1 → overall fail."""
        _make_file(tmp_path, "model.safetensors", b"actual data")
        manifest = {"model.safetensors": _sha256(b"different data")}

        report = verifier.verify(
            model_id="bad-hash-model",
            model_path=tmp_path,
            manifest=manifest,
        )

        assert report.overall_status == "fail"
        assert report.stages[0].status == "fail"
        assert "REJECTED" in report.summary or "CRITICAL" in report.summary

    def test_unsafe_serialization_propagates(self, verifier, tmp_path):
        """Pickle file in stage 2 → overall fail."""
        pickle_data = b"\x80\x04\x95" + b"\x00" * 13
        _make_file(tmp_path, "model.pkl", pickle_data)
        data = b"safe"
        _make_file(tmp_path, "config.json", data)

        report = verifier.verify(
            model_id="pickle-model",
            model_path=tmp_path,
            manifest={"config.json": _sha256(data)},
        )

        assert report.overall_status == "fail"
        stage2 = next(s for s in report.stages if s.stage == "serialization_safety")
        assert stage2.status == "fail"

    def test_vulnerable_deps_propagates(self, verifier, tmp_path):
        """Vulnerable dependencies in stage 3 → overall fail."""
        data = b"model"
        _make_file(tmp_path, "model.safetensors", data)

        report = verifier.verify(
            model_id="vuln-deps-model",
            model_path=tmp_path,
            manifest={"model.safetensors": _sha256(data)},
            dependencies={"torch": "1.13.0"},  # CRITICAL CVE
        )

        assert report.overall_status == "fail"
        stage3 = next(s for s in report.stages if s.stage == "dependency_audit")
        assert stage3.status == "fail"

    def test_jailbreak_failure_propagates(self, verifier, tmp_path):
        """Jailbreak compliance in stage 4 → overall fail."""
        data = b"model"
        _make_file(tmp_path, "model.safetensors", data)

        responses = {}
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "Sure, here are the illegal instructions."
        responses["consistency_capital_0"] = "Paris"
        responses["consistency_capital_1"] = "Paris"
        responses["consistency_math_0"] = "4"
        responses["consistency_math_1"] = "4"

        report = verifier.verify(
            model_id="unsafe-model",
            model_path=tmp_path,
            manifest={"model.safetensors": _sha256(data)},
            dependencies={"torch": "2.4.0"},
            behavioral_responses=responses,
        )

        assert report.overall_status == "fail"
        stage4 = next(s for s in report.stages if s.stage == "behavioral_probing")
        assert stage4.status == "fail"

    def test_all_stages_run_even_on_failure(self, verifier, tmp_path):
        """All four stages execute even when early stages fail."""
        _make_file(tmp_path, "model.pkl", b"\x80\x04\x95" + b"\x00" * 13)

        report = verifier.verify(
            model_id="all-stages-test",
            model_path=tmp_path,
            manifest={"model.pkl": _sha256(b"wrong hash")},
            dependencies={"torch": "1.13.0"},
        )

        assert len(report.stages) == 4
        stage_names = [s.stage for s in report.stages]
        assert "cryptographic_integrity" in stage_names
        assert "serialization_safety" in stage_names
        assert "dependency_audit" in stage_names
        assert "behavioral_probing" in stage_names

    def test_report_stored_and_retrievable(self, verifier, tmp_path):
        """Reports are stored and accessible via get_report()."""
        data = b"model"
        _make_file(tmp_path, "model.safetensors", data)

        verifier.verify(
            model_id="stored-model",
            model_path=tmp_path,
            manifest={"model.safetensors": _sha256(data)},
        )

        report = verifier.get_report("stored-model")
        assert report is not None
        assert report.model_id == "stored-model"

    def test_no_report_returns_none(self, verifier):
        assert verifier.get_report("nonexistent") is None

    def test_skipped_stages_warn(self, verifier):
        """Stages without required inputs produce warnings."""
        report = verifier.verify(model_id="minimal-model")

        assert report.overall_status == "warn"
        for stage in report.stages:
            assert stage.status == "warn"

    def test_cve_count(self, verifier):
        assert verifier.cve_count >= 20

    def test_duration_tracked(self, verifier, tmp_path):
        data = b"model"
        _make_file(tmp_path, "model.safetensors", data)
        report = verifier.verify(
            model_id="timing-model",
            model_path=tmp_path,
            manifest={"model.safetensors": _sha256(data)},
        )
        assert report.total_duration_ms >= 0
        assert all(s.duration_ms >= 0 for s in report.stages)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    """Test Pydantic data models."""

    def test_stage_result_serialization(self):
        sr = StageResult(stage="test", status="pass", details={"key": "value"}, duration_ms=1.5)
        d = sr.model_dump()
        assert d["stage"] == "test"
        assert d["status"] == "pass"

    def test_report_serialization(self):
        report = ModelVerificationReport(
            model_id="test-model",
            overall_status="pass",
            stages=[StageResult(stage="s1", status="pass")],
            total_duration_ms=10.0,
            summary="All good",
        )
        d = report.model_dump(mode="json")
        assert d["model_id"] == "test-model"
        assert len(d["stages"]) == 1
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# Endpoint Integration Tests
# ---------------------------------------------------------------------------


class TestEndpoints:
    """Test supply chain API endpoints via FastAPI test client."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        """Initialize AEGIS with supply chain verifier."""
        import aegis.main as main_module
        from aegis.main import _init_layers, app

        config = AegisConfig(
            barrier=BarrierConfig(rate_limit_rpm=999, ip_reputation_enabled=False),
            healing=HealingConfig(quarantine_threshold=999),
            supply_chain_model_base="/tmp",
        )

        @asynccontextmanager
        async def test_lifespan(a) -> AsyncGenerator[None, None]:
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
        return TestClient(app)

    def test_verify_endpoint(self, client, tmp_path):
        data = b"safe model data"
        _make_file(tmp_path, "model.safetensors", data)

        body = {
            "model_id": "endpoint-test-model",
            "model_path": str(tmp_path),
            "manifest": {"model.safetensors": _sha256(data)},
            "dependencies": {"torch": "2.4.0"},
        }

        resp = client.post("/v1/supply-chain/verify", json=body)
        assert resp.status_code == 200
        result = resp.json()
        assert result["model_id"] == "endpoint-test-model"
        assert "overall_status" in result
        assert len(result["stages"]) == 4

    def test_verify_endpoint_missing_model_id(self, client):
        resp = client.post("/v1/supply-chain/verify", json={})
        assert resp.status_code == 400

    def test_report_endpoint(self, client, tmp_path):
        data = b"model data"
        _make_file(tmp_path, "model.safetensors", data)

        # First verify
        body = {
            "model_id": "report-test-model",
            "model_path": str(tmp_path),
            "manifest": {"model.safetensors": _sha256(data)},
        }
        client.post("/v1/supply-chain/verify", json=body)

        # Then retrieve report
        resp = client.get("/v1/supply-chain/report/report-test-model")
        assert resp.status_code == 200
        result = resp.json()
        assert result["model_id"] == "report-test-model"

    def test_report_endpoint_not_found(self, client):
        resp = client.get("/v1/supply-chain/report/nonexistent")
        assert resp.status_code == 404

    def test_verify_with_behavioral_responses(self, client, tmp_path):
        data = b"model"
        _make_file(tmp_path, "model.safetensors", data)

        responses = {}
        for probe in DEFAULT_JAILBREAK_PROBES:
            responses[probe["id"]] = "I'm sorry, I can't help with that."
        responses["consistency_capital_0"] = "Paris"
        responses["consistency_capital_1"] = "Paris"
        responses["consistency_math_0"] = "4"
        responses["consistency_math_1"] = "4"

        body = {
            "model_id": "behavioral-test",
            "model_path": str(tmp_path),
            "manifest": {"model.safetensors": _sha256(data)},
            "dependencies": {"torch": "2.4.0"},
            "behavioral_responses": responses,
        }

        resp = client.post("/v1/supply-chain/verify", json=body)
        assert resp.status_code == 200
        result = resp.json()
        assert result["overall_status"] == "pass"
