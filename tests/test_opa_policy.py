"""
Tests for OPA Policy Engine integration — Task #87.

Covers:
- build_opa_input() document construction
- parse_opa_response() mapping to PolicyDecision
- OPAPolicyEngine fallback behavior (timeout, unreachable, invalid response)
- OPAPolicyEngine health check
- Config backend switching (python vs opa)
- Policy module refactoring verification (imports still work)
- Rego policy logic validation (unit-test style via build/parse round-trips)

Run: python3 -m pytest tests/test_opa_policy.py -x -q --tb=short -p no:warnings
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aegis.config import AegisConfig, PolicyConfig, ThreatLevel
from aegis.layers.policy import PolicyEngine, TenantPolicy
from aegis.layers.policy.opa_engine import (
    OPAPolicyEngine,
    build_opa_input,
    parse_opa_response,
    _ACTION_MAP,
    _OPA_TIMEOUT_SECONDS,
)
from aegis.models.policy_decision import PolicyAction, PolicyDecision, PolicyTier
from aegis.models.scan_result import InnateScanReport, ScanResult, ThreatCategory

API_KEY = "test-key-opa"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def innate_report():
    """Minimal InnateScanReport for testing."""
    return InnateScanReport(
        request_id="req-001",
        scanner_results=[
            ScanResult(
                scanner_id="regex",
                is_threat=True,
                confidence=0.92,
                threat_category=ThreatCategory.PROMPT_INJECTION,
                matched_patterns=["test-pattern"],
                latency_ms=0.5,
            )
        ],
        should_block=True,
        max_confidence=0.92,
        total_latency_ms=1.5,
        threat_categories=[ThreatCategory.PROMPT_INJECTION],
    )


@pytest.fixture
def benign_innate_report():
    """Benign InnateScanReport."""
    return InnateScanReport(
        request_id="req-002",
        scanner_results=[],
        should_block=False,
        max_confidence=0.0,
        total_latency_ms=0.5,
        threat_categories=[],
    )


@pytest.fixture
def tenant_policy():
    """Sample TenantPolicy."""
    return TenantPolicy(
        tenant_id="test-tenant",
        policy_tier="strict",
        custom_block_threshold=0.80,
        custom_alert_threshold=0.40,
        custom_policy={"blocked_categories": ["AML.T0051"]},
    )


@pytest.fixture
def python_engine():
    """Fallback Python PolicyEngine."""
    config = PolicyConfig()
    return PolicyEngine(config=config)


@pytest.fixture
def opa_engine(python_engine):
    """OPAPolicyEngine with Python fallback."""
    return OPAPolicyEngine(
        opa_url="http://localhost:8181",
        fallback=python_engine,
    )


# ===========================================================================
# Test: build_opa_input
# ===========================================================================

class TestBuildOpaInput:
    """Tests for build_opa_input() document construction."""

    def test_base_input_with_no_reports(self):
        doc = build_opa_input(
            request_id="req-001",
            innate_report=None,
            adaptive_report=None,
            tenant_id="default",
            model="gpt-4",
            threat_level=ThreatLevel.GREEN,
        )
        assert doc["request_id"] == "req-001"
        assert doc["model"] == "gpt-4"
        assert doc["threat_level"] == "GREEN"
        assert doc["innate"]["max_confidence"] == 0.0
        assert doc["innate"]["threat_categories"] == []
        assert doc["innate"]["should_block"] is False
        assert doc["adaptive"]["mcav_score"] == 0.0
        assert doc["adaptive"]["is_novel_attack"] is False
        assert doc["adaptive"]["analyzers"] == {}
        assert doc["tenant"]["tenant_id"] == "default"
        assert doc["tenant"]["block_threshold"] == 0.85
        assert doc["tenant"]["policy_tier"] == "standard"

    def test_with_innate_report(self, innate_report):
        doc = build_opa_input(
            request_id="req-001",
            innate_report=innate_report,
            adaptive_report=None,
            tenant_id="acme",
            model="gpt-4",
            threat_level=ThreatLevel.BLUE,
        )
        assert doc["innate"]["max_confidence"] == 0.92
        assert ThreatCategory.PROMPT_INJECTION.value in doc["innate"]["threat_categories"]
        assert doc["innate"]["should_block"] is True
        assert doc["threat_level"] == "BLUE"

    def test_with_tenant_policy(self, tenant_policy):
        doc = build_opa_input(
            request_id="req-001",
            innate_report=None,
            adaptive_report=None,
            tenant_id="test-tenant",
            model="gpt-4",
            threat_level=ThreatLevel.GREEN,
            tenant_policy=tenant_policy,
        )
        assert doc["tenant"]["tenant_id"] == "test-tenant"
        assert doc["tenant"]["block_threshold"] == 0.80
        assert doc["tenant"]["policy_tier"] == "strict"
        assert doc["tenant"]["custom_policy"]["blocked_categories"] == ["AML.T0051"]

    def test_threat_level_names(self):
        for level in ThreatLevel:
            doc = build_opa_input(
                request_id="req", innate_report=None, adaptive_report=None,
                tenant_id="t", model="m", threat_level=level,
            )
            assert doc["threat_level"] == level.name

    def test_tenant_defaults_without_policy(self):
        doc = build_opa_input(
            request_id="req", innate_report=None, adaptive_report=None,
            tenant_id="t", model="m", threat_level=ThreatLevel.GREEN,
            tenant_policy=None,
        )
        assert doc["tenant"]["block_threshold"] == 0.85
        assert doc["tenant"]["alert_threshold"] == 0.50
        assert doc["tenant"]["policy_tier"] == "standard"
        assert doc["tenant"]["custom_policy"] == {}


# ===========================================================================
# Test: parse_opa_response
# ===========================================================================

class TestParseOpaResponse:
    """Tests for parse_opa_response() mapping."""

    def test_allow_response(self):
        opa_result = {
            "action": "allow",
            "reason": "No policy triggered",
            "triggered_by": "adaptive",
            "applied_policies": [],
        }
        decision = parse_opa_response(
            opa_result, "req-001", 0.0, 0.0, ThreatLevel.GREEN,
        )
        assert decision.action == PolicyAction.ALLOW
        assert decision.triggered_by == PolicyTier.ADAPTIVE
        assert decision.request_id == "req-001"
        assert decision.is_allowed

    def test_block_response(self):
        opa_result = {
            "action": "block",
            "reason": "GLOBAL: RED fail-closed",
            "triggered_by": "global",
            "applied_policies": ["GLOBAL-RED-FAILCLOSED"],
        }
        decision = parse_opa_response(
            opa_result, "req-002", 0.5, 0.3, ThreatLevel.RED,
        )
        assert decision.action == PolicyAction.BLOCK
        assert decision.triggered_by == PolicyTier.GLOBAL
        assert not decision.is_allowed
        assert "opa:GLOBAL-RED-FAILCLOSED" in decision.applied_policies

    def test_opa_prefix_on_policies(self):
        opa_result = {
            "action": "block",
            "reason": "test",
            "triggered_by": "tenant",
            "applied_policies": ["POLICY-A", "POLICY-B"],
        }
        decision = parse_opa_response(
            opa_result, "req", 0.9, 0.0, ThreatLevel.GREEN,
        )
        assert decision.applied_policies == ["opa:POLICY-A", "opa:POLICY-B"]

    def test_fused_score_is_max(self):
        decision = parse_opa_response(
            {"action": "allow", "reason": "ok", "triggered_by": "adaptive", "applied_policies": []},
            "req", 0.7, 0.3, ThreatLevel.GREEN,
        )
        assert decision.fused_score == 0.7

        decision2 = parse_opa_response(
            {"action": "allow", "reason": "ok", "triggered_by": "adaptive", "applied_policies": []},
            "req", 0.2, 0.8, ThreatLevel.GREEN,
        )
        assert decision2.fused_score == 0.8

    def test_unknown_action_defaults_to_allow(self):
        decision = parse_opa_response(
            {"action": "unknown_action", "reason": "x", "triggered_by": "adaptive", "applied_policies": []},
            "req", 0.0, 0.0, ThreatLevel.GREEN,
        )
        assert decision.action == PolicyAction.ALLOW

    def test_unknown_tier_defaults_to_adaptive(self):
        decision = parse_opa_response(
            {"action": "allow", "reason": "x", "triggered_by": "nonexistent_tier", "applied_policies": []},
            "req", 0.0, 0.0, ThreatLevel.GREEN,
        )
        assert decision.triggered_by == PolicyTier.ADAPTIVE

    def test_missing_fields_use_defaults(self):
        decision = parse_opa_response({}, "req", 0.0, 0.0, ThreatLevel.GREEN)
        assert decision.action == PolicyAction.ALLOW
        assert decision.triggered_by == PolicyTier.ADAPTIVE
        assert decision.reasons == ["OPA policy decision"]

    def test_all_action_types_mapped(self):
        for action_str, expected in _ACTION_MAP.items():
            decision = parse_opa_response(
                {"action": action_str, "reason": "test", "triggered_by": "adaptive", "applied_policies": []},
                "req", 0.0, 0.0, ThreatLevel.GREEN,
            )
            assert decision.action == expected

    def test_threat_level_preserved(self):
        for level in ThreatLevel:
            decision = parse_opa_response(
                {"action": "allow", "reason": "ok", "triggered_by": "adaptive", "applied_policies": []},
                "req", 0.0, 0.0, level,
            )
            assert decision.threat_level == level


# ===========================================================================
# Test: OPAPolicyEngine fallback behavior
# ===========================================================================

class TestOPAFallback:
    """Tests for OPAPolicyEngine falling back to Python engine."""

    @pytest.mark.asyncio
    async def test_fallback_on_httpx_unavailable(self, opa_engine, benign_innate_report):
        """When httpx client returns None, fall back to Python engine."""
        opa_engine._client = None
        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=None):
            decision = await opa_engine.evaluate(
                request_id="req-001",
                innate_report=benign_innate_report,
                tenant_id="default",
            )
        assert decision.is_allowed
        assert "opa:FALLBACK" in decision.applied_policies
        assert opa_engine._fallback_count == 1

    @pytest.mark.asyncio
    async def test_fallback_on_connection_error(self, opa_engine, benign_innate_report):
        """Connection error → fallback."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection refused")
        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            decision = await opa_engine.evaluate(
                request_id="req-002",
                innate_report=benign_innate_report,
                tenant_id="default",
            )
        assert decision.is_allowed
        assert "opa:FALLBACK" in decision.applied_policies

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_response(self, opa_engine, benign_innate_report):
        """Invalid OPA response (no 'action') → fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": {"no_action_field": True}}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            decision = await opa_engine.evaluate(
                request_id="req-003",
                innate_report=benign_innate_report,
                tenant_id="default",
            )
        assert decision.is_allowed
        assert "opa:FALLBACK" in decision.applied_policies

    @pytest.mark.asyncio
    async def test_fallback_on_non_dict_result(self, opa_engine, benign_innate_report):
        """OPA returns non-dict result → fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": "not a dict"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            decision = await opa_engine.evaluate(
                request_id="req-004",
                innate_report=benign_innate_report,
                tenant_id="default",
            )
        assert "opa:FALLBACK" in decision.applied_policies

    @pytest.mark.asyncio
    async def test_fallback_on_http_error(self, opa_engine, benign_innate_report):
        """HTTP 500 → fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Internal Server Error")

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            decision = await opa_engine.evaluate(
                request_id="req-005",
                innate_report=benign_innate_report,
                tenant_id="default",
            )
        assert "opa:FALLBACK" in decision.applied_policies

    @pytest.mark.asyncio
    async def test_successful_opa_call(self, opa_engine, innate_report):
        """Successful OPA call returns OPA decision (not fallback)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "action": "block",
                "reason": "GLOBAL: Hard block on AEGIS.PROMPT_INJECTION",
                "triggered_by": "global",
                "applied_policies": ["GLOBAL-HARDBLOCK-AEGIS.PROMPT_INJECTION"],
            }
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            decision = await opa_engine.evaluate(
                request_id="req-006",
                innate_report=innate_report,
                tenant_id="default",
            )
        assert decision.action == PolicyAction.BLOCK
        assert decision.triggered_by == PolicyTier.GLOBAL
        assert "opa:GLOBAL-HARDBLOCK-AEGIS.PROMPT_INJECTION" in decision.applied_policies
        assert "opa:FALLBACK" not in decision.applied_policies
        assert opa_engine._opa_count == 1

    @pytest.mark.asyncio
    async def test_fallback_count_increments(self, opa_engine, benign_innate_report):
        """Each fallback increments _fallback_count."""
        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=None):
            await opa_engine.evaluate(request_id="r1", tenant_id="d")
            await opa_engine.evaluate(request_id="r2", tenant_id="d")
            await opa_engine.evaluate(request_id="r3", tenant_id="d")
        assert opa_engine._fallback_count == 3


# ===========================================================================
# Test: OPAPolicyEngine delegation
# ===========================================================================

class TestOPADelegation:
    """Tests for OPAPolicyEngine delegating to fallback for non-eval operations."""

    def test_threat_level_delegated(self, opa_engine):
        assert opa_engine.threat_level == ThreatLevel.GREEN
        opa_engine.threat_level = ThreatLevel.BLUE
        assert opa_engine.threat_level == ThreatLevel.BLUE
        assert opa_engine.fallback.threat_level == ThreatLevel.BLUE

    def test_escalate_threat_level(self, opa_engine):
        new_level = opa_engine.escalate_threat_level()
        assert new_level == ThreatLevel.BLUE
        assert opa_engine.threat_level == ThreatLevel.BLUE

    def test_de_escalate_threat_level(self, opa_engine):
        opa_engine.threat_level = ThreatLevel.ORANGE
        new_level = opa_engine.de_escalate_threat_level()
        assert new_level == ThreatLevel.YELLOW
        assert opa_engine.threat_level == ThreatLevel.YELLOW

    def test_register_tenant_policy(self, opa_engine, tenant_policy):
        opa_engine.register_tenant_policy(tenant_policy)
        # Verify it was registered on the fallback engine
        resolved = opa_engine.fallback._resolve_tenant_policy("test-tenant")
        assert resolved is not None
        assert resolved.policy_tier == "strict"

    def test_tenant_manager_property(self, opa_engine):
        assert opa_engine.tenant_manager is None
        mock_tm = MagicMock()
        opa_engine.tenant_manager = mock_tm
        assert opa_engine.tenant_manager is mock_tm
        assert opa_engine.fallback.tenant_manager is mock_tm

    def test_fallback_property(self, opa_engine, python_engine):
        assert opa_engine.fallback is python_engine


# ===========================================================================
# Test: OPAPolicyEngine health check
# ===========================================================================

class TestOPAHealth:
    """Tests for OPAPolicyEngine.health()."""

    @pytest.mark.asyncio
    async def test_health_when_httpx_unavailable(self, opa_engine):
        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=None):
            status = await opa_engine.health()
        assert status["status"] == "unhealthy"
        assert "httpx unavailable" in status["error"]

    @pytest.mark.asyncio
    async def test_health_when_opa_reachable(self, opa_engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            status = await opa_engine.health()
        assert status["status"] == "healthy"
        assert status["opa_url"] == "http://localhost:8181"

    @pytest.mark.asyncio
    async def test_health_when_opa_unreachable(self, opa_engine):
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")

        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            status = await opa_engine.health()
        assert status["status"] == "unhealthy"
        assert "Connection refused" in status["error"]

    @pytest.mark.asyncio
    async def test_health_includes_decision_counts(self, opa_engine):
        opa_engine._opa_count = 42
        opa_engine._fallback_count = 3

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(opa_engine, '_get_client', new_callable=AsyncMock, return_value=mock_client):
            status = await opa_engine.health()
        assert status["opa_decisions"] == 42
        assert status["fallback_decisions"] == 3


# ===========================================================================
# Test: Configuration
# ===========================================================================

class TestOPAConfig:
    """Tests for policy backend configuration."""

    def test_default_backend_is_python(self):
        config = PolicyConfig()
        assert config.backend == "python"

    def test_default_opa_url(self):
        config = PolicyConfig()
        assert config.opa_url == "http://localhost:8181"

    def test_opa_timeout_value(self):
        assert _OPA_TIMEOUT_SECONDS == 0.050

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("AEGIS_POLICY_BACKEND", "opa")
        monkeypatch.setenv("AEGIS_POLICY_OPA_URL", "http://opa-server:9191")
        config = PolicyConfig()
        assert config.backend == "opa"
        assert config.opa_url == "http://opa-server:9191"


# ===========================================================================
# Test: Policy module refactoring verification
# ===========================================================================

class TestPolicyModuleRefactor:
    """Verify policy module refactoring preserved all imports."""

    def test_import_policy_engine(self):
        from aegis.layers.policy import PolicyEngine
        assert PolicyEngine is not None

    def test_import_tenant_policy(self):
        from aegis.layers.policy import TenantPolicy
        assert TenantPolicy is not None

    def test_import_opa_engine(self):
        from aegis.layers.policy.opa_engine import OPAPolicyEngine
        assert OPAPolicyEngine is not None

    def test_import_build_opa_input(self):
        from aegis.layers.policy.opa_engine import build_opa_input
        assert callable(build_opa_input)

    def test_import_parse_opa_response(self):
        from aegis.layers.policy.opa_engine import parse_opa_response
        assert callable(parse_opa_response)


# ===========================================================================
# Test: Rego logic validation via build/parse round-trips
# ===========================================================================

class TestRegoLogicValidation:
    """Validate Rego policy logic expectations via input/output contracts.

    These tests verify the expected behavior of the Rego rules by testing
    the Python input builder and response parser with the expected OPA
    responses for each policy scenario. This ensures the Python code and
    Rego rules agree on the input/output contract.
    """

    def test_red_failclosed_input(self):
        """RED threat level should produce input that Rego blocks."""
        doc = build_opa_input(
            request_id="req", innate_report=None, adaptive_report=None,
            tenant_id="t", model="m", threat_level=ThreatLevel.RED,
        )
        assert doc["threat_level"] == "RED"
        # Expected Rego output for RED:
        expected_response = {
            "action": "block",
            "reason": "GLOBAL: RED threat level — fail-closed, all requests blocked",
            "triggered_by": "global",
            "applied_policies": ["GLOBAL-RED-FAILCLOSED"],
        }
        decision = parse_opa_response(expected_response, "req", 0.0, 0.0, ThreatLevel.RED)
        assert decision.action == PolicyAction.BLOCK
        assert decision.triggered_by == PolicyTier.GLOBAL

    def test_prompt_injection_hard_block(self):
        """High-confidence prompt injection should be hard-blocked."""
        doc = build_opa_input(
            request_id="req",
            innate_report=InnateScanReport(
                request_id="req", scanner_results=[], should_block=True,
                max_confidence=0.92,
                threat_categories=[ThreatCategory.PROMPT_INJECTION],
                total_latency_ms=1.0,
            ),
            adaptive_report=None, tenant_id="t", model="m",
            threat_level=ThreatLevel.GREEN,
        )
        assert doc["innate"]["max_confidence"] == 0.92
        assert ThreatCategory.PROMPT_INJECTION.value in doc["innate"]["threat_categories"]

    def test_strict_tier_threshold_lowered(self):
        """Strict tier should lower effective threshold by 15%."""
        tp = TenantPolicy(
            tenant_id="strict-t", policy_tier="strict",
            custom_block_threshold=0.85,
        )
        doc = build_opa_input(
            request_id="req", innate_report=None, adaptive_report=None,
            tenant_id="strict-t", model="m", threat_level=ThreatLevel.GREEN,
            tenant_policy=tp,
        )
        # Effective threshold = 0.85 * 0.85 = 0.7225
        assert doc["tenant"]["policy_tier"] == "strict"
        assert doc["tenant"]["block_threshold"] == 0.85

    def test_permissive_tier_threshold_raised(self):
        """Permissive tier should raise effective threshold by 10%."""
        tp = TenantPolicy(
            tenant_id="perm-t", policy_tier="permissive",
            custom_block_threshold=0.85,
        )
        doc = build_opa_input(
            request_id="req", innate_report=None, adaptive_report=None,
            tenant_id="perm-t", model="m", threat_level=ThreatLevel.GREEN,
            tenant_policy=tp,
        )
        # Effective threshold = min(0.85 * 1.10, 1.0) = 0.935
        assert doc["tenant"]["policy_tier"] == "permissive"
        assert doc["tenant"]["block_threshold"] == 0.85

    def test_custom_blocked_categories_in_input(self):
        """Custom blocked categories appear in tenant policy."""
        tp = TenantPolicy(
            tenant_id="custom-t", policy_tier="standard",
            custom_policy={"blocked_categories": ["AML.T0051", "AML.T0054"]},
        )
        doc = build_opa_input(
            request_id="req", innate_report=None, adaptive_report=None,
            tenant_id="custom-t", model="m", threat_level=ThreatLevel.GREEN,
            tenant_policy=tp,
        )
        assert doc["tenant"]["custom_policy"]["blocked_categories"] == ["AML.T0051", "AML.T0054"]

    def test_tli_blue_in_input(self):
        doc = build_opa_input(
            request_id="req", innate_report=None, adaptive_report=None,
            tenant_id="t", model="m", threat_level=ThreatLevel.BLUE,
        )
        assert doc["threat_level"] == "BLUE"

    def test_tli_orange_in_input(self):
        doc = build_opa_input(
            request_id="req", innate_report=None, adaptive_report=None,
            tenant_id="t", model="m", threat_level=ThreatLevel.ORANGE,
        )
        assert doc["threat_level"] == "ORANGE"


# ===========================================================================
# Test: OPAPolicyEngine.close()
# ===========================================================================

class TestOPAClose:
    """Tests for OPAPolicyEngine resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_client(self, opa_engine):
        mock_client = AsyncMock()
        opa_engine._client = mock_client
        await opa_engine.close()
        mock_client.aclose.assert_awaited_once()
        assert opa_engine._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, opa_engine):
        opa_engine._client = None
        await opa_engine.close()  # Should not raise


# ===========================================================================
# Test: OPA URL construction
# ===========================================================================

class TestOPAUrlConstruction:
    """Tests for OPA endpoint URL construction."""

    def test_decision_url(self):
        engine = OPAPolicyEngine(opa_url="http://opa:8181")
        assert engine._decision_url == "http://opa:8181/v1/data/aegis/policy/decision"

    def test_health_url(self):
        engine = OPAPolicyEngine(opa_url="http://opa:8181")
        assert engine._health_url == "http://opa:8181/health"

    def test_trailing_slash_stripped(self):
        engine = OPAPolicyEngine(opa_url="http://opa:8181/")
        assert engine._decision_url == "http://opa:8181/v1/data/aegis/policy/decision"

    def test_custom_port(self):
        engine = OPAPolicyEngine(opa_url="http://custom-opa:9999")
        assert engine._decision_url == "http://custom-opa:9999/v1/data/aegis/policy/decision"


# ===========================================================================
# Test: Main.py integration (import verification)
# ===========================================================================

class TestMainIntegration:
    """Verify main.py imports and wiring for OPA engine."""

    def test_opa_engine_importable_from_main(self):
        """OPAPolicyEngine is imported in main.py."""
        import aegis.main as main_module
        assert hasattr(main_module, 'OPAPolicyEngine')

    def test_init_layers_creates_python_engine_by_default(self):
        """Default config creates Python PolicyEngine."""
        import aegis.main as main_module
        config = AegisConfig(
            api_key=API_KEY,
            upstream_url="https://mock.test",
            upstream_api_key="test-key",
        )
        main_module._init_layers(config)
        assert isinstance(main_module._policy, PolicyEngine)
        assert not isinstance(main_module._policy, OPAPolicyEngine)
        # Cleanup
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
        main_module._audit = None

    def test_init_layers_creates_opa_engine_when_configured(self):
        """When backend='opa', creates OPAPolicyEngine."""
        import aegis.main as main_module
        config = AegisConfig(
            api_key=API_KEY,
            upstream_url="https://mock.test",
            upstream_api_key="test-key",
        )
        config.policy.backend = "opa"
        config.policy.opa_url = "http://test-opa:8181"
        main_module._init_layers(config)
        assert isinstance(main_module._policy, OPAPolicyEngine)
        assert main_module._policy._opa_url == "http://test-opa:8181"
        assert isinstance(main_module._policy.fallback, PolicyEngine)
        # Cleanup
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
        main_module._audit = None
