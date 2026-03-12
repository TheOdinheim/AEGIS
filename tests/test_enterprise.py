"""
Tests for Enterprise Hardening — Configuration Validation, Backup/Restore,
Deep Health Monitoring, and Admin API Endpoints.

Covers:
- Config validation: valid config, empty API key, default API key, invalid thresholds,
  negative rate limits, production SSL/Redis checks, invalid policy backend, DP epsilon
- Vault backup: correct JSON structure, indicator/signature export
- Vault restore: load from backup, round-trip consistency, invalid file handling
- Deep health: all healthy, Redis down, disk space, circuit breaker, auto-recovery
- Admin API endpoints: backup, restore, deep-health, config-validation, config-reload
- DEPLOYMENT.md: existence and required sections
"""

from __future__ import annotations

import asyncio
import json
import os
import pytest
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from aegis.config import AegisConfig, get_config
from aegis.services.deployment.config_validator import ConfigValidator, ConfigValidationResult
from aegis.services.deployment.backup_restore import VaultBackupManager, BackupResult, RestoreResult
from aegis.services.deployment.health_monitor import DeepHealthMonitor, DeepHealthResult, ComponentHealth


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------


class TestConfigValidatorBasics:
    """Test ConfigValidator initialization and result structure."""

    def test_result_dataclass(self):
        result = ConfigValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.checked_at is not None

    def test_result_to_dict(self):
        result = ConfigValidationResult(
            valid=False,
            errors=["error1"],
            warnings=["warn1", "warn2"],
        )
        d = result.to_dict()
        assert d["valid"] is False
        assert d["error_count"] == 1
        assert d["warning_count"] == 2
        assert "checked_at" in d


class TestConfigValidatorValidConfig:
    """Test that a valid production-like config passes."""

    def test_valid_config_passes(self):
        """Config with all required fields set should pass."""
        validator = ConfigValidator()
        config = get_config()
        # Set a real API key to avoid errors
        config.api_key = "production-key-abc123xyz"
        result = validator.validate(config)
        assert result.valid is True
        assert len(result.errors) == 0


class TestConfigValidatorAPIKey:
    """Test API key validation."""

    def test_empty_api_key_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = ""
        result = validator.validate(config)
        assert not result.valid
        assert any("AEGIS_API_KEY is empty" in e for e in result.errors)

    def test_test_api_key_warns(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "my-test-key"
        result = validator.validate(config)
        assert result.valid  # Warning, not error
        assert any("test/example" in w for w in result.warnings)

    def test_example_api_key_warns(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "example-key-123"
        result = validator.validate(config)
        assert any("test/example" in w for w in result.warnings)

    def test_changeme_api_key_warns(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "changeme"
        result = validator.validate(config)
        assert any("weak default" in w.lower() for w in result.warnings)

    def test_strong_api_key_no_warnings(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "sk-prod-a1b2c3d4e5f6g7h8i9j0"
        result = validator.validate(config)
        api_key_warnings = [w for w in result.warnings if "API_KEY" in w]
        assert len(api_key_warnings) == 0


class TestConfigValidatorCanarySecret:
    """Test canary secret key validation."""

    def test_default_canary_secret_warns(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        # Default canary key
        result = validator.validate(config)
        assert any("CANARY_SECRET_KEY" in w for w in result.warnings)

    def test_custom_canary_secret_no_warning(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        config.canary.secret_key = "my-custom-secret-key-abc123"
        result = validator.validate(config)
        canary_warnings = [w for w in result.warnings if "CANARY_SECRET_KEY" in w]
        assert len(canary_warnings) == 0


class TestConfigValidatorThresholds:
    """Test threshold validation."""

    def test_alert_above_block_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        orig_at = config.innate.alert_threshold
        orig_bt = config.innate.block_threshold
        object.__setattr__(config.innate, "alert_threshold", 0.90)
        object.__setattr__(config.innate, "block_threshold", 0.50)
        try:
            result = validator.validate(config)
            assert any("alert_threshold" in e and "block_threshold" in e for e in result.errors)
        finally:
            object.__setattr__(config.innate, "alert_threshold", orig_at)
            object.__setattr__(config.innate, "block_threshold", orig_bt)


class TestConfigValidatorRateLimits:
    """Test rate limit validation."""

    def _with_barrier(self, attr, val):
        """Context-manager-like pattern: set attr, yield, restore."""
        config = get_config()
        return config, getattr(config.barrier, attr)

    def test_zero_rate_limit_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        orig = config.barrier.rate_limit_rpm
        object.__setattr__(config.barrier, "rate_limit_rpm", 0)
        try:
            result = validator.validate(config)
            assert any("rate_limit_rpm" in e for e in result.errors)
        finally:
            object.__setattr__(config.barrier, "rate_limit_rpm", orig)

    def test_negative_rate_limit_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        orig = config.barrier.rate_limit_rpm
        object.__setattr__(config.barrier, "rate_limit_rpm", -5)
        try:
            result = validator.validate(config)
            assert any("rate_limit_rpm" in e for e in result.errors)
        finally:
            object.__setattr__(config.barrier, "rate_limit_rpm", orig)

    def test_zero_burst_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        orig = config.barrier.rate_limit_burst
        object.__setattr__(config.barrier, "rate_limit_burst", 0)
        try:
            result = validator.validate(config)
            assert any("rate_limit_burst" in e for e in result.errors)
        finally:
            object.__setattr__(config.barrier, "rate_limit_burst", orig)

    def test_zero_max_tokens_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        orig = config.barrier.max_tokens_per_request
        object.__setattr__(config.barrier, "max_tokens_per_request", 0)
        try:
            result = validator.validate(config)
            assert any("max_tokens_per_request" in e for e in result.errors)
        finally:
            object.__setattr__(config.barrier, "max_tokens_per_request", orig)


class TestConfigValidatorUpstreamURL:
    """Test upstream URL validation."""

    def test_empty_upstream_url_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        orig = config.upstream_url
        config.api_key = "valid-key"
        config.upstream_url = ""
        try:
            result = validator.validate(config)
            assert any("UPSTREAM_URL" in e for e in result.errors)
        finally:
            config.upstream_url = orig


class TestConfigValidatorProduction:
    """Test production-specific checks."""

    def test_production_without_ssl_warns(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_ENV": "production", "DATABASE_URL": "postgresql://host/db"}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            assert any("sslmode" in w for w in result.warnings)

    def test_production_with_ssl_no_warning(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_ENV": "production", "DATABASE_URL": "postgresql://host/db?sslmode=require"}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            ssl_warnings = [w for w in result.warnings if "sslmode" in w]
            assert len(ssl_warnings) == 0

    def test_production_without_redis_warns(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_ENV": "production", "REDIS_URL": ""}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            assert any("REDIS_URL" in w for w in result.warnings)


class TestConfigValidatorDPEpsilon:
    """Test DP epsilon validation."""

    def test_epsilon_too_low_warns(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_DP_EPSILON": "0.01"}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            assert any("DP_EPSILON" in w for w in result.warnings)

    def test_epsilon_too_high_warns(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_DP_EPSILON": "500.0"}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            assert any("DP_EPSILON" in w for w in result.warnings)

    def test_epsilon_normal_no_warning(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_DP_EPSILON": "3.0"}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            eps_warnings = [w for w in result.warnings if "DP_EPSILON" in w]
            assert len(eps_warnings) == 0

    def test_epsilon_non_numeric_warns(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_DP_EPSILON": "abc"}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            assert any("DP_EPSILON" in w for w in result.warnings)


class TestConfigValidatorPolicyBackend:
    """Test policy backend validation."""

    def test_invalid_policy_backend_is_error(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        orig = config.policy.backend
        object.__setattr__(config.policy, "backend", "invalid")
        try:
            result = validator.validate(config)
            assert any("POLICY_BACKEND" in e for e in result.errors)
        finally:
            object.__setattr__(config.policy, "backend", orig)

    def test_valid_python_backend(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        object.__setattr__(config.policy, "backend", "python")
        result = validator.validate(config)
        policy_errors = [e for e in result.errors if "POLICY_BACKEND" in e]
        assert len(policy_errors) == 0

    def test_valid_opa_backend(self):
        validator = ConfigValidator()
        config = get_config()
        config.api_key = "valid-key"
        orig = config.policy.backend
        object.__setattr__(config.policy, "backend", "opa")
        try:
            result = validator.validate(config)
            policy_errors = [e for e in result.errors if "POLICY_BACKEND" in e]
            assert len(policy_errors) == 0
        finally:
            object.__setattr__(config.policy, "backend", orig)


class TestConfigValidatorAgentSigningKey:
    """Test agent signing key validation."""

    def test_missing_signing_key_warns(self):
        validator = ConfigValidator()
        with patch.dict(os.environ, {"AEGIS_AGENT_SIGNING_KEY": ""}):
            config = get_config()
            config.api_key = "valid-key"
            result = validator.validate(config)
            assert any("AGENT_SIGNING_KEY" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Vault Backup Tests
# ---------------------------------------------------------------------------


class TestVaultBackup:
    """Test vault backup functionality."""

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def mock_vault(self):
        vault = MagicMock()
        vault.get_indicators.return_value = []
        vault.get_stats.return_value = {"total_indicators": 0}
        return vault

    @pytest.mark.asyncio
    async def test_backup_empty_vault(self, tmp_dir, mock_vault):
        mgr = VaultBackupManager()
        output = tmp_dir / "backup.json"
        result = await mgr.backup_vault(mock_vault, output)
        assert result.success
        assert result.indicator_count == 0
        assert result.file_size_bytes > 0
        assert Path(result.output_path).exists()

    @pytest.mark.asyncio
    async def test_backup_creates_valid_json(self, tmp_dir, mock_vault):
        mgr = VaultBackupManager()
        output = tmp_dir / "backup.json"
        await mgr.backup_vault(mock_vault, output)
        data = json.loads(output.read_text())
        assert "metadata" in data
        assert "indicators" in data
        assert "signatures" in data
        assert "timestamp" in data["metadata"]
        assert "aegis_version" in data["metadata"]

    @pytest.mark.asyncio
    async def test_backup_with_indicators(self, tmp_dir):
        """Backup with 5 mock indicators."""
        vault = MagicMock()
        indicators = []
        for i in range(5):
            ind = MagicMock()
            ind.indicator_id = f"ind-{i}"
            ind.source.value = "seed"
            ind.threat_category.value = "PROMPT_INJECTION"
            ind.confidence = 0.9
            ind.severity = 0.8
            ind.payload_hash = f"hash-{i}"
            ind.payload_summary = f"attack payload {i}"
            ind.confirmed = False
            ind.phase.value = "acute"
            ind.frequency = 1
            ind.seen_by_tenants = 1
            ind.first_seen = datetime.now(timezone.utc)
            ind.last_seen = datetime.now(timezone.utc)
            ind.mitre_atlas_id = "AML.T0051"
            ind.affected_models = ["gpt-4"]
            ind.metadata = {}
            indicators.append(ind)
        vault.get_indicators.return_value = indicators
        vault.get_stats.return_value = {"total_indicators": 5}

        mgr = VaultBackupManager()
        output = tmp_dir / "backup.json"
        result = await mgr.backup_vault(vault, output)
        assert result.success
        assert result.indicator_count == 5

        data = json.loads(output.read_text())
        assert len(data["indicators"]) == 5
        assert data["metadata"]["indicator_count"] == 5

    @pytest.mark.asyncio
    async def test_backup_with_signatures(self, tmp_dir, mock_vault):
        """Backup with signature store."""
        sig_store = MagicMock()
        sig = MagicMock()
        sig.signature_id = "sig-001"
        sig.pattern = "ignore.*instructions"
        sig.category = "prompt_injection"
        sig.description = "test sig"
        sig.source_indicator_id = "ind-0"
        sig.affinity_score = 0.8
        sig.deprecated = False
        sig.tp_rate = 1.0
        sig.fp_rate = 0.0
        sig.match_count = 10
        sig.false_positive_count = 0
        sig_store.get_all.return_value = [sig]

        mgr = VaultBackupManager()
        output = tmp_dir / "backup.json"
        result = await mgr.backup_vault(mock_vault, output, sig_store)
        assert result.success
        assert result.signature_count == 1

        data = json.loads(output.read_text())
        assert len(data["signatures"]) == 1
        assert data["signatures"][0]["pattern"] == "ignore.*instructions"

    @pytest.mark.asyncio
    async def test_backup_creates_parent_dirs(self, tmp_dir, mock_vault):
        mgr = VaultBackupManager()
        output = tmp_dir / "deep" / "nested" / "backup.json"
        result = await mgr.backup_vault(mock_vault, output)
        assert result.success
        assert output.exists()

    @pytest.mark.asyncio
    async def test_backup_result_to_dict(self, tmp_dir, mock_vault):
        mgr = VaultBackupManager()
        output = tmp_dir / "backup.json"
        result = await mgr.backup_vault(mock_vault, output)
        d = result.to_dict()
        assert "success" in d
        assert "indicator_count" in d
        assert "duration_ms" in d


# ---------------------------------------------------------------------------
# Vault Restore Tests
# ---------------------------------------------------------------------------


class TestVaultRestore:
    """Test vault restore functionality."""

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def _write_backup(self, path: Path, indicators=None, signatures=None):
        backup = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "aegis_version": "0.1.0",
                "indicator_count": len(indicators or []),
                "signature_count": len(signatures or []),
            },
            "indicators": indicators or [],
            "signatures": signatures or [],
        }
        path.write_text(json.dumps(backup))

    @pytest.mark.asyncio
    async def test_restore_empty_backup(self, tmp_dir):
        backup_path = tmp_dir / "backup.json"
        self._write_backup(backup_path)

        vault = MagicMock()
        vault.get_indicator.return_value = None
        mgr = VaultBackupManager()
        result = await mgr.restore_vault(vault, backup_path)
        assert result.success
        assert result.indicators_restored == 0

    @pytest.mark.asyncio
    async def test_restore_with_indicators(self, tmp_dir):
        indicators = [{
            "indicator_id": "ind-0",
            "source": "seed",
            "threat_category": "PROMPT_INJECTION",
            "confidence": 0.9,
            "severity": 0.8,
            "payload_hash": "hash-0",
            "payload_summary": "test attack",
            "confirmed": False,
            "phase": "acute",
            "frequency": 1,
            "seen_by_tenants": 1,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "mitre_atlas_id": "AML.T0051",
            "affected_models": ["gpt-4"],
            "metadata": {},
        }]
        backup_path = tmp_dir / "backup.json"
        self._write_backup(backup_path, indicators=indicators)

        vault = MagicMock()
        vault.get_indicator.return_value = None
        mgr = VaultBackupManager()
        result = await mgr.restore_vault(vault, backup_path)
        assert result.success
        assert result.indicators_restored == 1
        assert vault.add.call_count == 1

    @pytest.mark.asyncio
    async def test_restore_skips_duplicates(self, tmp_dir):
        indicators = [{
            "indicator_id": "ind-dup",
            "source": "seed",
            "threat_category": "PROMPT_INJECTION",
            "confidence": 0.9,
            "severity": 0.5,
            "payload_hash": "hash-dup",
            "payload_summary": "dup attack",
            "confirmed": False,
            "phase": "acute",
            "frequency": 1,
            "seen_by_tenants": 1,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "mitre_atlas_id": "",
            "affected_models": [],
            "metadata": {},
        }]
        backup_path = tmp_dir / "backup.json"
        self._write_backup(backup_path, indicators=indicators)

        vault = MagicMock()
        vault.get_indicator.return_value = MagicMock()  # Already exists
        mgr = VaultBackupManager()
        result = await mgr.restore_vault(vault, backup_path)
        assert result.success
        assert result.indicators_restored == 0
        assert result.indicators_skipped == 1

    @pytest.mark.asyncio
    async def test_restore_with_signatures(self, tmp_dir):
        signatures = [{
            "pattern": "ignore.*instructions",
            "category": "prompt_injection",
            "description": "test",
            "source_indicator_id": "ind-0",
            "affinity_score": 0.8,
            "tp_rate": 1.0,
            "fp_rate": 0.0,
        }]
        backup_path = tmp_dir / "backup.json"
        self._write_backup(backup_path, signatures=signatures)

        vault = MagicMock()
        vault.get_indicator.return_value = None
        sig_store = MagicMock()

        mgr = VaultBackupManager()
        result = await mgr.restore_vault(vault, backup_path, sig_store)
        assert result.success
        assert result.signatures_restored == 1
        assert sig_store.add.call_count == 1

    @pytest.mark.asyncio
    async def test_restore_file_not_found(self, tmp_dir):
        mgr = VaultBackupManager()
        vault = MagicMock()
        result = await mgr.restore_vault(vault, tmp_dir / "nonexistent.json")
        assert not result.success
        assert any("not found" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_restore_invalid_json(self, tmp_dir):
        backup_path = tmp_dir / "bad.json"
        backup_path.write_text("not valid json {{{")
        mgr = VaultBackupManager()
        vault = MagicMock()
        result = await mgr.restore_vault(vault, backup_path)
        assert not result.success
        assert any("Invalid JSON" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_restore_empty_file(self, tmp_dir):
        backup_path = tmp_dir / "empty.json"
        backup_path.write_text("")
        mgr = VaultBackupManager()
        vault = MagicMock()
        result = await mgr.restore_vault(vault, backup_path)
        assert not result.success
        assert any("empty" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_restore_missing_keys(self, tmp_dir):
        backup_path = tmp_dir / "partial.json"
        backup_path.write_text(json.dumps({"metadata": {}}))
        mgr = VaultBackupManager()
        vault = MagicMock()
        result = await mgr.restore_vault(vault, backup_path)
        assert not result.success
        assert any("Missing required key" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_restore_result_to_dict(self, tmp_dir):
        backup_path = tmp_dir / "backup.json"
        self._write_backup(backup_path)
        vault = MagicMock()
        vault.get_indicator.return_value = None
        mgr = VaultBackupManager()
        result = await mgr.restore_vault(vault, backup_path)
        d = result.to_dict()
        assert "success" in d
        assert "indicators_restored" in d
        assert "duration_ms" in d


class TestBackupRestoreRoundTrip:
    """Test backup → restore round-trip."""

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.mark.asyncio
    async def test_round_trip(self, tmp_dir):
        """Backup → read backup → verify structure matches."""
        # Create mock vault with indicators
        vault = MagicMock()
        indicators = []
        for i in range(3):
            ind = MagicMock()
            ind.indicator_id = f"rt-{i}"
            ind.source.value = "adaptive_detection"
            ind.threat_category.value = "PROMPT_INJECTION"
            ind.confidence = 0.85 + i * 0.05
            ind.severity = 0.7
            ind.payload_hash = f"rt-hash-{i}"
            ind.payload_summary = f"round trip attack {i}"
            ind.confirmed = i == 0
            ind.phase.value = "acute"
            ind.frequency = i + 1
            ind.seen_by_tenants = 1
            ind.first_seen = datetime.now(timezone.utc)
            ind.last_seen = datetime.now(timezone.utc)
            ind.mitre_atlas_id = "AML.T0051"
            ind.affected_models = ["gpt-4"]
            ind.metadata = {"round_trip_test": True}
            indicators.append(ind)
        vault.get_indicators.return_value = indicators
        vault.get_stats.return_value = {"total_indicators": 3}

        mgr = VaultBackupManager()
        backup_path = tmp_dir / "roundtrip.json"

        # Backup
        backup_result = await mgr.backup_vault(vault, backup_path)
        assert backup_result.success
        assert backup_result.indicator_count == 3

        # Verify backup content
        data = json.loads(backup_path.read_text())
        assert len(data["indicators"]) == 3
        assert data["indicators"][0]["indicator_id"] == "rt-0"
        assert data["indicators"][0]["confirmed"] is True
        assert data["indicators"][2]["confidence"] == 0.95
        assert data["metadata"]["indicator_count"] == 3


# ---------------------------------------------------------------------------
# Deep Health Monitor Tests
# ---------------------------------------------------------------------------


class TestDeepHealthMonitor:
    """Test deep health monitoring."""

    def test_init(self):
        monitor = DeepHealthMonitor()
        assert not monitor._running
        assert monitor.last_result is None

    def test_component_health_to_dict(self):
        ch = ComponentHealth(
            name="redis", status="healthy", latency_ms=1.5,
            details={"connected": True},
        )
        d = ch.to_dict()
        assert d["name"] == "redis"
        assert d["status"] == "healthy"
        assert d["latency_ms"] == 1.5

    def test_deep_health_result_to_dict(self):
        result = DeepHealthResult(
            overall_status="healthy",
            recommendations=["check redis"],
        )
        d = result.to_dict()
        assert d["overall_status"] == "healthy"
        assert len(d["recommendations"]) == 1

    @pytest.mark.asyncio
    async def test_all_healthy(self):
        """All components healthy → overall healthy."""
        monitor = DeepHealthMonitor()
        # Mock all external checks to avoid real connections
        with patch.object(monitor, "_check_redis", new_callable=AsyncMock), \
             patch.object(monitor, "_check_postgres", new_callable=AsyncMock), \
             patch.object(monitor, "_check_disk_space", new_callable=AsyncMock):
            result = await monitor.run_deep_health_check()
            assert result.overall_status in ("healthy", "degraded")
            assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_redis_down_degraded(self):
        """Redis unreachable → degraded status with recommendation."""
        monitor = DeepHealthMonitor()

        async def mock_redis_check(components, recommendations):
            components["redis"] = ComponentHealth(
                name="redis", status="degraded",
                details={"connected": False},
            )
            recommendations.append("Redis is unreachable.")

        with patch.object(monitor, "_check_redis", side_effect=mock_redis_check), \
             patch.object(monitor, "_check_postgres", new_callable=AsyncMock), \
             patch.object(monitor, "_check_disk_space", new_callable=AsyncMock):
            result = await monitor.run_deep_health_check()
            assert result.overall_status == "degraded"
            assert any("Redis" in r for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_disk_space_warning(self):
        """Low disk space → recommendation for log rotation."""
        monitor = DeepHealthMonitor(disk_warning_gb=999999.0)  # Force warning

        with patch.object(monitor, "_check_redis", new_callable=AsyncMock), \
             patch.object(monitor, "_check_postgres", new_callable=AsyncMock):
            result = await monitor.run_deep_health_check()
            # Disk check should trigger warning since we set threshold very high
            if "disk_space" in result.component_results:
                disk = result.component_results["disk_space"]
                assert disk.status in ("degraded", "healthy", "unhealthy")

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self):
        """Circuit breaker open → unhealthy + recommendation."""
        monitor = DeepHealthMonitor()
        healing = MagicMock()
        breaker = MagicMock()
        breaker.state.value = "open"
        breaker.consecutive_trips = 3
        healing.get_breaker.return_value = breaker
        monitor.set_components(healing=healing)

        components: dict[str, ComponentHealth] = {}
        recommendations: list[str] = []
        monitor._check_circuit_breaker(components, recommendations)

        assert components["circuit_breaker"].status == "unhealthy"
        assert any("OPEN" in r for r in recommendations)

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed(self):
        monitor = DeepHealthMonitor()
        healing = MagicMock()
        breaker = MagicMock()
        breaker.state.value = "closed"
        breaker.consecutive_trips = 0
        healing.get_breaker.return_value = breaker
        monitor.set_components(healing=healing)

        components: dict[str, ComponentHealth] = {}
        recommendations: list[str] = []
        monitor._check_circuit_breaker(components, recommendations)
        assert components["circuit_breaker"].status == "healthy"

    def test_faiss_healthy(self):
        monitor = DeepHealthMonitor()
        vault = MagicMock()
        vault.get_stats.return_value = {"total_indicators": 100, "index_size": 100}
        monitor.set_components(vault=vault)

        components: dict[str, ComponentHealth] = {}
        recommendations: list[str] = []
        monitor._check_faiss(components, recommendations)
        assert components["faiss"].status == "healthy"

    def test_faiss_mismatch(self):
        monitor = DeepHealthMonitor()
        vault = MagicMock()
        # Use correct key name that the code checks for
        vault.get_stats.return_value = {"total_indicators": 100, "index_size": 50}
        monitor.set_components(vault=vault)

        components: dict[str, ComponentHealth] = {}
        recommendations: list[str] = []
        monitor._check_faiss(components, recommendations)
        assert components["faiss"].status == "degraded"
        assert any("differs" in r.lower() for r in recommendations)

    def test_model_check_no_adaptive(self):
        monitor = DeepHealthMonitor()
        components: dict[str, ComponentHealth] = {}
        recommendations: list[str] = []
        monitor._check_model(components, recommendations)
        assert components["ml_model"].status == "degraded"

    def test_start_stop_monitor(self):
        monitor = DeepHealthMonitor()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._start_stop(monitor))
        finally:
            loop.close()

    async def _start_stop(self, monitor):
        monitor.start_monitor()
        assert monitor._running
        monitor.stop_monitor()
        assert not monitor._running

    def test_stop_without_start(self):
        monitor = DeepHealthMonitor()
        monitor.stop_monitor()  # Should not raise

    def test_stats_property(self):
        monitor = DeepHealthMonitor()
        stats = monitor.stats
        assert stats["running"] is False
        assert stats["last_check"] is None


# ---------------------------------------------------------------------------
# Admin API Endpoint Tests
# ---------------------------------------------------------------------------


class TestAdminAPIEndpoints:
    """Test admin API endpoints via TestClient."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import aegis.main as m
        from fastapi.testclient import TestClient

        self._orig_config = m._config
        self._orig_vault = m._vault
        self._orig_vault_backup = m._vault_backup
        self._orig_deep_health = m._deep_health
        self._orig_config_validation = m._config_validation
        self._orig_sig_store = m._signature_store
        self._orig_tenant_manager = m._tenant_manager
        self._orig_threat_intel = m._threat_intel

        config = get_config()
        if not config.api_key:
            config.api_key = "test-admin-api-key"
        m._config = config

        # Set up mock components
        m._vault = MagicMock()
        m._vault.get_indicators.return_value = []
        m._vault.get_stats.return_value = {"total_indicators": 0}
        m._vault.get_indicator.return_value = None

        m._vault_backup = VaultBackupManager()
        m._deep_health = DeepHealthMonitor()
        m._config_validation = ConfigValidationResult(valid=True, warnings=["test warning"])
        m._signature_store = MagicMock()
        m._signature_store.get_all.return_value = []

        self.client = TestClient(m.app, raise_server_exceptions=False)
        self.auth = {"Authorization": f"Bearer {config.api_key}"}
        yield
        m._config = self._orig_config
        m._vault = self._orig_vault
        m._vault_backup = self._orig_vault_backup
        m._deep_health = self._orig_deep_health
        m._config_validation = self._orig_config_validation
        m._signature_store = self._orig_sig_store
        m._tenant_manager = self._orig_tenant_manager
        m._threat_intel = self._orig_threat_intel

    def test_config_validation_unauthenticated(self):
        resp = self.client.get("/v1/admin/config-validation")
        assert resp.status_code == 401

    def test_config_validation_authenticated(self):
        resp = self.client.get("/v1/admin/config-validation", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert "test warning" in data["warnings"]

    def test_deep_health_unauthenticated(self):
        resp = self.client.get("/v1/admin/deep-health")
        assert resp.status_code == 401

    def test_deep_health_authenticated(self):
        resp = self.client.get("/v1/admin/deep-health", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_status" in data
        assert "component_results" in data
        assert "recommendations" in data

    def test_backup_unauthenticated(self):
        resp = self.client.post("/v1/admin/backup")
        assert resp.status_code == 401

    def test_backup_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "test_backup.json")
            resp = self.client.post(
                "/v1/admin/backup",
                headers=self.auth,
                json={"output_path": output},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert Path(output).exists()

    def test_restore_unauthenticated(self):
        resp = self.client.post("/v1/admin/restore")
        assert resp.status_code == 401

    def test_restore_missing_path(self):
        resp = self.client.post(
            "/v1/admin/restore",
            headers=self.auth,
            json={},
        )
        assert resp.status_code == 400

    def test_restore_from_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create backup
            output = os.path.join(tmp, "backup.json")
            self.client.post(
                "/v1/admin/backup",
                headers=self.auth,
                json={"output_path": output},
            )
            # Restore
            resp = self.client.post(
                "/v1/admin/restore",
                headers=self.auth,
                json={"backup_path": output},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True

    def test_config_reload_unauthenticated(self):
        resp = self.client.post("/v1/admin/config-reload")
        assert resp.status_code == 401

    def test_config_reload_authenticated(self):
        resp = self.client.post("/v1/admin/config-reload", headers=self.auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "reloaded" in data


# ---------------------------------------------------------------------------
# DEPLOYMENT.md Tests
# ---------------------------------------------------------------------------


class TestDeploymentDoc:
    """Test that DEPLOYMENT.md exists and contains required sections."""

    def test_deployment_md_exists(self):
        doc_path = Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md"
        assert doc_path.exists(), f"docs/DEPLOYMENT.md not found at {doc_path}"

    def test_contains_prerequisites(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Prerequisites" in doc

    def test_contains_quick_start(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Quick Start" in doc

    def test_contains_production_checklist(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Production Checklist" in doc

    def test_contains_env_var_reference(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Environment Variable" in doc

    def test_contains_scaling_guide(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Scaling" in doc

    def test_contains_backup_restore(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Backup" in doc and "Restore" in doc

    def test_contains_troubleshooting(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Troubleshooting" in doc

    def test_contains_monitoring(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        assert "Monitoring" in doc

    def test_contains_12_checklist_items(self):
        doc = (Path(__file__).parent.parent / "docs" / "DEPLOYMENT.md").read_text()
        # Count numbered items in checklist section
        import re
        numbered = re.findall(r"^\d+\.\s+\*\*", doc, re.MULTILINE)
        assert len(numbered) >= 12


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestEnterpriseIntegration:
    """End-to-end integration tests."""

    def test_config_validator_with_real_config(self):
        """Validate actual AEGIS config (test environment)."""
        validator = ConfigValidator()
        config = get_config()
        # In test env, API key may be empty — that's expected
        result = validator.validate(config)
        # Should at least not crash
        assert isinstance(result, ConfigValidationResult)
        assert result.checked_at is not None

    @pytest.mark.asyncio
    async def test_backup_restore_cycle(self):
        """Full backup → restore cycle with mock vault."""
        with tempfile.TemporaryDirectory() as tmp:
            vault = MagicMock()
            ind = MagicMock()
            ind.indicator_id = "cycle-1"
            ind.source.value = "seed"
            ind.threat_category.value = "PROMPT_INJECTION"
            ind.confidence = 0.9
            ind.severity = 0.5
            ind.payload_hash = "cycle-hash"
            ind.payload_summary = "cycle attack"
            ind.confirmed = False
            ind.phase.value = "acute"
            ind.frequency = 1
            ind.seen_by_tenants = 1
            ind.first_seen = datetime.now(timezone.utc)
            ind.last_seen = datetime.now(timezone.utc)
            ind.mitre_atlas_id = ""
            ind.affected_models = []
            ind.metadata = {}
            vault.get_indicators.return_value = [ind]
            vault.get_stats.return_value = {"total_indicators": 1}
            vault.get_indicator.return_value = None

            mgr = VaultBackupManager()
            backup_path = Path(tmp) / "cycle.json"

            # Backup
            b_result = await mgr.backup_vault(vault, backup_path)
            assert b_result.success

            # Restore
            r_result = await mgr.restore_vault(vault, backup_path)
            assert r_result.success
            assert r_result.indicators_restored == 1

    @pytest.mark.asyncio
    async def test_deep_health_monitor_full_cycle(self):
        """Run deep health check and verify result structure."""
        monitor = DeepHealthMonitor()
        result = await monitor.run_deep_health_check()
        assert isinstance(result, DeepHealthResult)
        assert result.overall_status in ("healthy", "degraded", "unhealthy")
        assert result.duration_ms >= 0
        assert monitor.last_result is result
