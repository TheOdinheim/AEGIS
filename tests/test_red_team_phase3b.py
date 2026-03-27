"""Red Team Phase 3B regression tests — black-box assessment.

Each test demonstrates the exploit was possible and verifies the fix works.
Tests are named after their finding IDs from docs/red_team_phase3b_report.md.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _setup_app():
    """Set up the app with mock components for admin endpoint tests."""
    import aegis.main as m
    from aegis.config import get_config
    from aegis.services.deployment import (
        ConfigValidationResult,
        DeepHealthMonitor,
        VaultBackupManager,
    )

    config = m._config or get_config()
    if not config.api_key:
        config.api_key = "test-rt-p3b-key"
    m._config = config

    # Ensure components are initialized for admin endpoints
    if not m._vault:
        m._vault = MagicMock()
        m._vault.get_indicators.return_value = []
        m._vault.get_stats.return_value = {
            "total_indicators": 0,
            "top_indicators": [],
        }
        m._vault.get_indicator.return_value = None
    if not m._vault_backup:
        m._vault_backup = VaultBackupManager()
    if not m._config_validation:
        m._config_validation = ConfigValidationResult(
            valid=True, warnings=["some config detail about weak key"]
        )

    return TestClient(m.app), config.api_key


# ---------------------------------------------------------------------------
# RT-P3B-001: OpenAPI docs disabled
# ---------------------------------------------------------------------------


class TestRTP3B001DocsDisabled:
    """Verify /docs, /redoc, /openapi.json are not accessible."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client, _ = _setup_app()

    def test_docs_returns_404(self):
        resp = self.client.get("/docs")
        assert resp.status_code == 404

    def test_redoc_returns_404(self):
        resp = self.client.get("/redoc")
        assert resp.status_code == 404

    def test_openapi_json_returns_404(self):
        resp = self.client.get("/openapi.json")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# RT-P3B-002: Case-insensitive Bearer token
# ---------------------------------------------------------------------------


class TestRTP3B002BearerCaseInsensitive:
    """Verify Bearer prefix is matched case-insensitively per RFC 6750."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client, self.api_key = _setup_app()

    def test_lowercase_bearer_main_auth(self):
        resp = self.client.get(
            "/v1/vault/stats",
            headers={"Authorization": f"bearer {self.api_key}"},
        )
        assert resp.status_code == 200

    def test_uppercase_bearer_main_auth(self):
        resp = self.client.get(
            "/v1/vault/stats",
            headers={"Authorization": f"BEARER {self.api_key}"},
        )
        assert resp.status_code == 200

    def test_mixed_case_bearer_main_auth(self):
        resp = self.client.get(
            "/v1/vault/stats",
            headers={"Authorization": f"BeArEr {self.api_key}"},
        )
        assert resp.status_code == 200

    def test_lowercase_bearer_barrier_extract(self):
        from aegis.layers.barrier import BarrierLayer

        key = BarrierLayer._extract_api_key(
            {"authorization": "bearer my-key-123"}
        )
        assert key == "my-key-123"

    def test_mixed_case_bearer_barrier_extract(self):
        from aegis.layers.barrier import BarrierLayer

        key = BarrierLayer._extract_api_key(
            {"authorization": "BEARER my-key-123"}
        )
        assert key == "my-key-123"


# ---------------------------------------------------------------------------
# RT-P3B-003: Path traversal in /v1/admin/restore
# ---------------------------------------------------------------------------


class TestRTP3B003RestorePathTraversal:
    """Verify restore endpoint rejects paths outside backup directory."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client, self.api_key = _setup_app()
        self.auth = {"Authorization": f"Bearer {self.api_key}"}

    def test_traversal_etc_passwd(self):
        resp = self.client.post(
            "/v1/admin/restore",
            headers=self.auth,
            json={"backup_path": "/etc/passwd"},
        )
        assert resp.status_code == 400
        assert "backup directory" in resp.json()["error"]

    def test_traversal_dot_dot(self):
        resp = self.client.post(
            "/v1/admin/restore",
            headers=self.auth,
            json={"backup_path": "/tmp/aegis-backups/../../etc/shadow"},
        )
        assert resp.status_code == 400

    def test_traversal_absolute_outside(self):
        resp = self.client.post(
            "/v1/admin/restore",
            headers=self.auth,
            json={"backup_path": "/var/log/syslog"},
        )
        assert resp.status_code == 400

    def test_valid_backup_path_allowed(self):
        backup_dir = "/tmp/aegis-backups"
        os.makedirs(backup_dir, exist_ok=True)
        resp = self.client.post(
            "/v1/admin/restore",
            headers=self.auth,
            json={"backup_path": f"{backup_dir}/nonexistent.json"},
        )
        # Should NOT be 400 "backup directory" error — it's a valid path
        if resp.status_code == 400:
            assert "backup directory" not in resp.json().get("error", "")


# ---------------------------------------------------------------------------
# RT-P3B-004: Config validation warnings redacted
# ---------------------------------------------------------------------------


class TestRTP3B004ConfigWarningsRedacted:
    """Verify config validation doesn't expose warning details."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client, self.api_key = _setup_app()
        self.auth = {"Authorization": f"Bearer {self.api_key}"}

    def test_warnings_are_opaque_labels(self):
        resp = self.client.get("/v1/admin/config-validation", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        for w in data.get("warnings", []):
            assert re.match(r"^warning_\d+$", w), f"Warning not redacted: {w}"

    def test_no_sensitive_text_in_warnings(self):
        resp = self.client.get("/v1/admin/config-validation", headers=self.auth)
        data = resp.json()
        warnings_text = str(data.get("warnings", []))
        for keyword in ["key", "password", "default", "weak", "credential", "secret"]:
            assert keyword not in warnings_text.lower(), (
                f"Sensitive keyword '{keyword}' found in warnings"
            )


# ---------------------------------------------------------------------------
# RT-P3B-005: Vault stats payload redaction
# ---------------------------------------------------------------------------


class TestRTP3B005VaultStatsNoPayload:
    """Verify vault stats don't expose attack payload text."""

    def test_payload_length_replaces_summary(self):
        """Verify ThreatVault.get_stats returns payload_length not payload_summary."""
        from aegis.layers.memory.threat_vault import ThreatVault
        from aegis.models.threat_indicator import ThreatIndicator, ThreatCategory, IndicatorSource

        import numpy as np
        vault = ThreatVault()
        embedding = np.random.randn(384).astype(np.float32).tolist()
        indicator = ThreatIndicator(
            indicator_id="test-p3b-payload-length",
            embedding=embedding,
            payload_summary="test attack payload for phase3b",
            payload_hash="test-hash-p3b",
            threat_category=ThreatCategory.PROMPT_INJECTION,
            confidence=0.95,
            severity=0.9,
            source=IndicatorSource.ADAPTIVE_DETECTION,
            mitre_atlas_id="AML.T0051",
        )
        vault.add(indicator)
        stats = vault.get_stats()
        for ind in stats.get("top_indicators", []):
            assert "payload_summary" not in ind, "payload_summary still exposed"
            assert "payload_length" in ind, "payload_length missing"
            assert isinstance(ind["payload_length"], int)
            assert ind["payload_length"] > 0


# ---------------------------------------------------------------------------
# RT-P3B-006: STIX indicator HTML sanitization
# ---------------------------------------------------------------------------


class TestRTP3B006StixHtmlSanitization:
    """Verify HTML tags are stripped from STIX indicator fields."""

    def _sanitize(self, stix_ind):
        """Call _ingest_indicator to trigger sanitization."""
        from aegis.services.threat_intel import ThreatIntelManager
        from aegis.layers.memory.threat_vault import ThreatVault

        vault = ThreatVault()
        mgr = ThreatIntelManager(threat_vault=vault)
        # Call private method directly — we're testing sanitization, not full pipeline
        mgr._ingest_indicator(stix_ind, source_name="test")

    def test_html_stripped_from_name(self):
        stix_ind = {
            "type": "indicator",
            "id": "indicator--test-xss-p3b-001",
            "name": '<script>alert("xss")</script>Malicious Indicator',
            "description": "Test attack payload for sanitization",
            "pattern": "[ipv4-addr:value = '1.2.3.4']",
            "pattern_type": "stix",
            "valid_from": "2026-01-01T00:00:00Z",
            "created": "2026-01-01T00:00:00Z",
            "modified": "2026-01-01T00:00:00Z",
        }
        self._sanitize(stix_ind)
        assert "<script>" not in stix_ind["name"]
        assert "alert" in stix_ind["name"]

    def test_html_stripped_from_description(self):
        stix_ind = {
            "type": "indicator",
            "id": "indicator--test-xss-p3b-002",
            "name": "Normal Name",
            "description": '<img src=x onerror="alert(1)">Attack vector payload',
            "pattern": "[domain-name:value = 'evil.com']",
            "pattern_type": "stix",
            "valid_from": "2026-01-01T00:00:00Z",
            "created": "2026-01-01T00:00:00Z",
            "modified": "2026-01-01T00:00:00Z",
        }
        self._sanitize(stix_ind)
        assert "<img" not in stix_ind["description"]
        assert "Attack vector" in stix_ind["description"]

    def test_nested_html_stripped(self):
        stix_ind = {
            "type": "indicator",
            "id": "indicator--test-xss-p3b-003",
            "name": '<div><b>bold</b><script>evil()</script></div>clean text',
            "description": "Test payload for nested HTML",
            "pattern": "[url:value = 'http://evil.com']",
            "pattern_type": "stix",
            "valid_from": "2026-01-01T00:00:00Z",
            "created": "2026-01-01T00:00:00Z",
            "modified": "2026-01-01T00:00:00Z",
        }
        self._sanitize(stix_ind)
        assert "<" not in stix_ind["name"]
        assert "clean text" in stix_ind["name"]


# ---------------------------------------------------------------------------
# RT-P3B-007: Backup ignores custom output_path
# ---------------------------------------------------------------------------


class TestRTP3B007BackupPathForced:
    """Verify backup always goes to designated directory, path not exposed."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client, self.api_key = _setup_app()
        self.auth = {"Authorization": f"Bearer {self.api_key}"}

    def test_backup_ignores_custom_path(self):
        resp = self.client.post(
            "/v1/admin/backup",
            headers=self.auth,
            json={"output_path": "/tmp/evil-location/steal.json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        if "output_path" in data:
            assert "evil-location" not in data["output_path"]
            assert "/" not in data["output_path"]

    def test_backup_response_redacts_path(self):
        resp = self.client.post(
            "/v1/admin/backup",
            headers=self.auth,
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        if "output_path" in data:
            assert not data["output_path"].startswith("/")
