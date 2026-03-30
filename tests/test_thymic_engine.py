"""
Tests for the ThymicValidationEngine — L9 orchestrator.

Covers: run_spot_check, run_comprehensive_sweep, ValidationReport fields,
TPR/FPR calculation, per-layer/per-tier results, health summary,
edge cases, duration tracking.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aegis.config import AegisConfig, BarrierConfig, HealingConfig
from aegis.layers.audit import reset_audit_logger
from aegis.layers.thymic.attack_profile_library import AttackProfileLibrary, Probe
from aegis.layers.thymic.engine import (
    ThymicValidationEngine,
    ValidationReport,
    HealthSummary,
    LayerResult,
    TierResult,
)
from aegis.layers.thymic.layer_probe_router import LayerProbeRouter, ProbeResult
from aegis.layers.thymic.mutation_engine import MutationEngine
from aegis.main import _init_layers, app
import aegis.main as main_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_KEY = "aegis-test-thymic-engine"


@pytest.fixture(autouse=True)
def setup_layers():
    """Initialize layers for each test."""
    config = AegisConfig(
        api_key=API_KEY,
        upstream_url="https://mock-upstream.test",
        upstream_api_key="test-key",
        barrier=BarrierConfig(rate_limit_rpm=6000, rate_limit_burst=1000),
        healing=HealingConfig(
            circuit_breaker_threshold=0.50,
            cooldown_seconds=1,
            probe_count=2,
        ),
    )
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
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "thymic"


@pytest.fixture
def engine(data_dir: Path) -> ThymicValidationEngine:
    router = LayerProbeRouter(app=app, api_key=API_KEY, concurrency=5)
    library = AttackProfileLibrary(data_dir=data_dir)
    return ThymicValidationEngine(
        library=library,
        router=router,
        data_dir=data_dir,
    )


# ---------------------------------------------------------------------------
# Spot check tests
# ---------------------------------------------------------------------------

class TestSpotCheck:
    def test_completes_without_error(self, engine: ThymicValidationEngine):
        """run_spot_check completes successfully."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=3)
        )
        assert isinstance(report, ValidationReport)

    def test_returns_report_with_all_fields(self, engine: ThymicValidationEngine):
        """run_spot_check returns ValidationReport with all required fields."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=3)
        )
        assert report.run_id is not None
        assert report.run_type == "spot_check"
        assert report.timestamp is not None
        assert report.duration_seconds >= 0
        assert report.total_probes > 0

    def test_detects_some_attacks(self, engine: ThymicValidationEngine):
        """Spot check detects at least some known attacks."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=5)
        )
        assert report.probes_detected > 0

    def test_tpr_calculated(self, engine: ThymicValidationEngine):
        """ValidationReport calculates TPR."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=5)
        )
        assert 0.0 <= report.overall_tpr <= 1.0

    def test_fpr_calculated(self, engine: ThymicValidationEngine):
        """ValidationReport calculates FPR."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=5)
        )
        assert 0.0 <= report.overall_fpr <= 1.0


# ---------------------------------------------------------------------------
# Comprehensive sweep tests
# ---------------------------------------------------------------------------

class TestComprehensiveSweep:
    def test_returns_more_results(self, engine: ThymicValidationEngine):
        """Comprehensive sweep returns more probes than spot check."""
        spot = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=3)
        )
        comp = asyncio.get_event_loop().run_until_complete(
            engine.run_comprehensive_sweep()
        )
        assert comp.total_probes > spot.total_probes

    def test_comprehensive_run_type(self, engine: ThymicValidationEngine):
        """Comprehensive sweep has correct run_type."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_comprehensive_sweep()
        )
        assert report.run_type == "comprehensive"


# ---------------------------------------------------------------------------
# Report details
# ---------------------------------------------------------------------------

class TestReportDetails:
    def test_per_tier_results_populated(self, engine: ThymicValidationEngine):
        """Per-tier results are populated for tiers with probes."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=3)
        )
        assert len(report.per_tier_results) > 0
        for tier, tr in report.per_tier_results.items():
            assert isinstance(tr, TierResult)
            assert tr.tier == tier
            assert tr.total > 0

    def test_per_layer_results_populated(self, engine: ThymicValidationEngine):
        """Per-layer results are populated for active layers."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=5)
        )
        # At least some layers should appear in results
        # (L2 at minimum, since many Tier 1 attacks are caught by innate)
        if report.probes_detected > 0:
            assert len(report.per_layer_results) > 0

    def test_duration_tracked(self, engine: ThymicValidationEngine):
        """Duration is tracked accurately."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=3)
        )
        assert report.duration_seconds > 0

    def test_probe_results_included(self, engine: ThymicValidationEngine):
        """Individual probe results are included in report."""
        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=3)
        )
        assert len(report.probe_results) == report.total_probes

    def test_tpr_correct_calculation(self):
        """TPR is correctly calculated as detected/total_attacks."""
        # Use a mock router that returns controlled results
        library = AttackProfileLibrary()
        # Add 2 attack probes and 1 benign
        library.add_probe(Probe(id="a1", text="attack1", tier=1, expected_result="block"))
        library.add_probe(Probe(id="a2", text="attack2", tier=1, expected_result="block"))
        library.add_probe(Probe(id="b1", text="benign1", tier=5, expected_result="pass"))

        async def mock_route(probe: Probe, **kwargs) -> ProbeResult:
            if probe.expected_result == "block":
                return ProbeResult(
                    probe_id=probe.id, probe_tier=probe.tier,
                    detected=True, detection_layers=["L2"],
                    response_status=403,
                )
            return ProbeResult(
                probe_id=probe.id, probe_tier=probe.tier,
                detected=False, response_status=200,
            )

        router = LayerProbeRouter(app=None, api_key="test")
        router.route_probe = mock_route  # type: ignore
        engine = ThymicValidationEngine(library=library, router=router)

        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=10)
        )
        assert report.overall_tpr == 1.0
        assert report.overall_fpr == 0.0

    def test_fpr_correct_calculation(self):
        """FPR is correctly calculated as false_positives/total_benign."""
        library = AttackProfileLibrary()
        library.add_probe(Probe(id="b1", text="benign1", tier=5, expected_result="pass"))
        library.add_probe(Probe(id="b2", text="benign2", tier=5, expected_result="pass"))

        async def mock_route(probe: Probe, **kwargs) -> ProbeResult:
            # First benign gets false-positive flagged
            if probe.id.endswith("b1") or (hasattr(probe, 'parent_id') and probe.parent_id and probe.parent_id.endswith("b1")):
                return ProbeResult(
                    probe_id=probe.id, probe_tier=5,
                    detected=True, detection_layers=["L2"],
                    false_positive=True, response_status=403,
                )
            return ProbeResult(
                probe_id=probe.id, probe_tier=5,
                detected=False, response_status=200,
            )

        router = LayerProbeRouter(app=None, api_key="test")
        router.route_probe = mock_route  # type: ignore
        engine = ThymicValidationEngine(library=library, router=router)

        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=10)
        )
        assert report.false_positives >= 1
        assert report.overall_fpr > 0.0


# ---------------------------------------------------------------------------
# Health summary
# ---------------------------------------------------------------------------

class TestHealthSummary:
    def test_returns_data_after_run(self, engine: ThymicValidationEngine):
        """get_health_summary returns data after a run."""
        asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=3)
        )
        summary = engine.get_health_summary()
        assert isinstance(summary, HealthSummary)
        assert summary.last_run_id is not None
        assert summary.last_run_type == "spot_check"
        assert summary.total_probes_last_run > 0

    def test_returns_empty_before_run(self, engine: ThymicValidationEngine):
        """get_health_summary returns empty/default before any run."""
        summary = engine.get_health_summary()
        assert summary.last_run_id is None
        assert summary.overall_tpr == 0.0
        assert summary.overall_fpr == 0.0
        assert summary.total_probes_last_run == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_library(self):
        """Engine handles empty library gracefully."""
        library = AttackProfileLibrary()
        router = LayerProbeRouter(app=None, api_key="test")
        engine = ThymicValidationEngine(library=library, router=router)

        report = asyncio.get_event_loop().run_until_complete(
            engine.run_spot_check(probes_per_tier=5)
        )
        assert report.total_probes == 0

    def test_probe_generation_failure(self, engine: ThymicValidationEngine):
        """Engine handles probe generation failure gracefully."""
        # Patch generator to raise
        with patch.object(engine._generator, "generate_spot_check", side_effect=Exception("test error")):
            report = asyncio.get_event_loop().run_until_complete(
                engine.run_spot_check()
            )
            assert report.total_probes == 0
            assert report.run_type == "spot_check"

    def test_comprehensive_failure(self, engine: ThymicValidationEngine):
        """Engine handles comprehensive generation failure gracefully."""
        with patch.object(engine._generator, "generate_comprehensive", side_effect=Exception("test")):
            report = asyncio.get_event_loop().run_until_complete(
                engine.run_comprehensive_sweep()
            )
            assert report.total_probes == 0

    def test_library_access(self, engine: ThymicValidationEngine):
        """Engine exposes library property."""
        assert engine.library is not None
        assert isinstance(engine.library, AttackProfileLibrary)

    def test_generator_access(self, engine: ThymicValidationEngine):
        """Engine exposes generator property."""
        assert engine.generator is not None

    def test_router_access(self, engine: ThymicValidationEngine):
        """Engine exposes router property."""
        assert engine.router is not None
