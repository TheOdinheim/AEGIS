"""
Tests for Dependency Chain Analyzer (DCA), Validation Cache, and
Re-Validation Scheduler.

Extension 3.3: 22 DCA tests
Extension 3.4: 12 Cache tests
Extension 3.5: 10 Scheduler tests
Integration:    5 tests
Total:         49 tests
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from aegis.layers.supply_chain.dependency_analyzer import (
    DependencyChainAnalyzer,
    DependencyAnalysisReport,
    DependencyInfo,
)
from aegis.layers.supply_chain.validation_cache import (
    CachedValidation,
    SupplyChainValidationCache,
    compute_content_hash,
)
from aegis.layers.supply_chain.revalidation_scheduler import (
    RevalidationRun,
    RevalidationScheduler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dca():
    data_dir = Path(__file__).parent.parent / "data"
    return DependencyChainAnalyzer(
        malicious_file=data_dir / "malicious_packages.json",
        popular_file=data_dir / "popular_packages.json",
    )


@pytest.fixture
def dca_empty():
    return DependencyChainAnalyzer()


@pytest.fixture
def cache():
    return SupplyChainValidationCache(max_entries=100, default_ttl=3600)


@pytest.fixture
def scheduler(cache):
    return RevalidationScheduler(cache=cache, active_interval_hours=0.001)


# ===========================================================================
# DCA Tests — Parsing
# ===========================================================================

class TestDCAParsing:
    def test_parse_requirements_standard(self, dca):
        content = "requests==2.31.0\nnumpy>=1.24.0\nflask~=3.0"
        deps = dca.parse_requirements_txt(content)
        assert len(deps) == 3
        assert deps[0].name == "requests"
        assert deps[0].version == "2.31.0"
        assert deps[0].ecosystem == "pypi"

    def test_parse_requirements_with_extras(self, dca):
        content = "uvicorn[standard]>=0.34.0"
        deps = dca.parse_requirements_txt(content)
        assert len(deps) == 1
        assert deps[0].name == "uvicorn"
        assert deps[0].extras == ["standard"]
        assert deps[0].version == "0.34.0"

    def test_parse_requirements_comments_blanks(self, dca):
        content = "# this is a comment\nrequests==2.0\n\n# another\nnumpy>=1.0\n"
        deps = dca.parse_requirements_txt(content)
        assert len(deps) == 2

    def test_parse_requirements_ignore_includes(self, dca):
        content = "-r other.txt\nrequests==2.0"
        deps = dca.parse_requirements_txt(content)
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_parse_requirements_inline_comment(self, dca):
        content = "requests==2.0 # important"
        deps = dca.parse_requirements_txt(content)
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_parse_package_json(self, dca):
        content = json.dumps({
            "dependencies": {"express": "^4.18.0", "lodash": "~4.17.21"},
            "devDependencies": {"jest": "^29.0.0"},
        })
        deps = dca.parse_package_json(content)
        assert len(deps) == 3
        assert deps[0].name == "express"
        assert deps[0].ecosystem == "npm"
        assert deps[0].version == "4.18.0"

    def test_parse_package_json_scoped(self, dca):
        content = json.dumps({
            "dependencies": {"@scope/package": "^1.0.0"},
        })
        deps = dca.parse_package_json(content)
        assert len(deps) == 1
        assert deps[0].name == "@scope/package"

    def test_parse_flat_list(self, dca):
        dep_list = [
            ("requests", "2.31.0", "pypi"),
            ("express", "4.18.0", "npm"),
        ]
        deps = dca.parse_dependency_list(dep_list)
        assert len(deps) == 2
        assert deps[0].ecosystem == "pypi"
        assert deps[1].ecosystem == "npm"

    def test_parse_invalid_package_json(self, dca):
        deps = dca.parse_package_json("not valid json")
        assert deps == []


# ===========================================================================
# DCA Tests — Known Malicious Detection
# ===========================================================================

class TestDCAMalicious:
    def test_known_malicious_npm(self, dca):
        deps = [DependencyInfo(name="event-stream", version="3.3.6", ecosystem="npm")]
        report = dca.analyze(deps)
        assert report.critical_count == 1
        assert report.findings[0].finding_type == "known_malicious"
        assert report.overall_risk == "critical"

    def test_known_malicious_pypi(self, dca):
        deps = [DependencyInfo(name="ctx", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert report.critical_count == 1

    def test_clean_package(self, dca):
        deps = [DependencyInfo(name="requests", version="2.31.0", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert report.critical_count == 0
        assert report.overall_risk == "clean"

    def test_malicious_case_insensitive(self, dca):
        deps = [DependencyInfo(name="Event-Stream", ecosystem="npm")]
        report = dca.analyze(deps)
        assert report.critical_count == 1


# ===========================================================================
# DCA Tests — Typosquatting Detection
# ===========================================================================

class TestDCATyposquatting:
    def test_typosquatting_detected(self, dca):
        deps = [DependencyInfo(name="requets", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert any(f.finding_type == "typosquatting" for f in report.findings)
        typo = next(f for f in report.findings if f.finding_type == "typosquatting")
        assert typo.similar_to == "requests"
        assert typo.severity == "high"

    def test_exact_match_not_flagged(self, dca):
        deps = [DependencyInfo(name="requests", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert not any(f.finding_type == "typosquatting" for f in report.findings)

    def test_distant_name_not_flagged(self, dca):
        deps = [DependencyInfo(name="totally-unique-pkg-xyz", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert not any(f.finding_type == "typosquatting" for f in report.findings)

    def test_npm_typosquatting(self, dca):
        deps = [DependencyInfo(name="exprss", ecosystem="npm")]
        report = dca.analyze(deps)
        assert any(f.finding_type == "typosquatting" for f in report.findings)


# ===========================================================================
# DCA Tests — Slopsquatting Detection
# ===========================================================================

class TestDCASlopsquatting:
    def test_slopsquatting_pattern_detected(self, dca):
        deps = [DependencyInfo(name="flask-auth-helper", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert any(f.finding_type == "slopsquatting" for f in report.findings)

    def test_popular_package_not_flagged(self, dca):
        deps = [DependencyInfo(name="flask", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert not any(f.finding_type == "slopsquatting" for f in report.findings)

    def test_non_pattern_not_flagged(self, dca):
        deps = [DependencyInfo(name="mypackage", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert not any(f.finding_type == "slopsquatting" for f in report.findings)


# ===========================================================================
# DCA Tests — Risk Profile
# ===========================================================================

class TestDCARiskProfile:
    def test_new_package_flagged(self, dca):
        deps = [DependencyInfo(name="somelib", ecosystem="pypi")]
        metadata = {"somelib": {"publish_date_days_ago": 5}}
        report = dca.analyze(deps, metadata=metadata)
        assert any(f.finding_type == "high_risk_profile" for f in report.findings)

    def test_low_downloads_flagged(self, dca):
        deps = [DependencyInfo(name="somelib", ecosystem="pypi")]
        metadata = {"somelib": {"download_count": 10}}
        report = dca.analyze(deps, metadata=metadata)
        assert any(f.finding_type == "high_risk_profile" for f in report.findings)

    def test_multiple_risk_factors_higher_severity(self, dca):
        deps = [DependencyInfo(name="somelib", ecosystem="pypi")]
        metadata = {"somelib": {
            "publish_date_days_ago": 2,
            "download_count": 5,
            "has_repository": False,
        }}
        report = dca.analyze(deps, metadata=metadata)
        risk_finding = next(f for f in report.findings if f.finding_type == "high_risk_profile")
        assert risk_finding.severity == "high"

    def test_no_metadata_no_flag(self, dca):
        deps = [DependencyInfo(name="somelib", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert not any(f.finding_type == "high_risk_profile" for f in report.findings)


# ===========================================================================
# DCA Tests — Report Structure
# ===========================================================================

class TestDCAReport:
    def test_empty_deps_clean(self, dca):
        report = dca.analyze([])
        assert report.overall_risk == "clean"
        assert report.total_dependencies == 0

    def test_report_structure(self, dca):
        deps = [DependencyInfo(name="requests", ecosystem="pypi")]
        report = dca.analyze(deps)
        assert isinstance(report, DependencyAnalysisReport)
        assert report.analysis_latency_ms >= 0
        assert "requests" in report.packages_checked

    def test_large_dep_list_performance(self, dca):
        """500 clean dependencies should complete quickly."""
        deps = [DependencyInfo(name=f"pkg-{i}", ecosystem="pypi") for i in range(500)]
        start = time.time()
        report = dca.analyze(deps)
        elapsed = time.time() - start
        assert elapsed < 5.0
        assert report.total_dependencies == 500

    def test_mixed_ecosystems(self, dca):
        deps = [
            DependencyInfo(name="requests", ecosystem="pypi"),
            DependencyInfo(name="express", ecosystem="npm"),
        ]
        report = dca.analyze(deps)
        assert report.total_dependencies == 2


# ===========================================================================
# Validation Cache Tests
# ===========================================================================

class TestValidationCache:
    def test_cache_store_retrieve(self, cache):
        cache.cache_result("model-1", "model", {"verdict": "trusted"}, "abc123")
        entry = cache.get_cached("model-1")
        assert entry is not None
        assert entry.component_id == "model-1"
        assert entry.validation_result == {"verdict": "trusted"}

    def test_cache_ttl_expiry(self, cache):
        cache.cache_result("model-2", "model", {}, "abc", ttl_seconds=1)
        entry = cache.get_cached("model-2")
        assert entry is not None
        # Manually expire
        entry.valid_until = time.time() - 1
        assert cache.get_cached("model-2") is None

    def test_cache_content_hash_change(self, cache):
        cache.cache_result("skill-1", "skill", {}, "hash1")
        assert cache.get_cached("skill-1", content_hash="hash1") is not None
        assert cache.get_cached("skill-1", content_hash="hash2") is None

    def test_cache_invalidate(self, cache):
        cache.cache_result("dep-1", "dependency", {}, "abc")
        assert cache.invalidate("dep-1") is True
        assert cache.get_cached("dep-1") is None
        assert cache.invalidate("nonexistent") is False

    def test_cache_invalidate_by_type(self, cache):
        cache.cache_result("m1", "model", {}, "a")
        cache.cache_result("m2", "model", {}, "b")
        cache.cache_result("s1", "skill", {}, "c")
        count = cache.invalidate_by_type("model")
        assert count == 2
        assert cache.get_cached("s1") is not None

    def test_get_all_approved(self, cache):
        cache.cache_result("m1", "model", {}, "a", status="approved")
        cache.cache_result("m2", "model", {}, "b", status="rejected")
        cache.cache_result("s1", "skill", {}, "c", status="approved")
        approved = cache.get_all_approved()
        assert len(approved) == 2
        approved_models = cache.get_all_approved(component_type="model")
        assert len(approved_models) == 1

    def test_retroactive_check(self, cache):
        cache.cache_result("model-evil", "model", {"info": "safe"}, "a", status="approved")
        cache.cache_result("model-good", "model", {"info": "safe"}, "b", status="approved")
        matches = cache.retroactive_check("evil")
        assert len(matches) == 1
        assert matches[0].component_id == "model-evil"

    def test_lru_eviction(self):
        small_cache = SupplyChainValidationCache(max_entries=3)
        small_cache.cache_result("a", "model", {}, "1")
        small_cache.cache_result("b", "model", {}, "2")
        small_cache.cache_result("c", "model", {}, "3")
        small_cache.cache_result("d", "model", {}, "4")
        assert small_cache.size == 3
        assert small_cache.get_cached("a") is None  # evicted
        assert small_cache.get_cached("d") is not None

    def test_fast_path_cached_result(self, cache):
        cache.cache_result("dep-fast", "dependency", {"risk": "clean"}, "hash1", status="approved")
        entry = cache.get_cached("dep-fast", content_hash="hash1")
        assert entry is not None
        assert entry.status == "approved"

    def test_rejected_returns_cached(self, cache):
        cache.cache_result("bad-model", "model", {"verdict": "rejected"}, "h", status="rejected")
        entry = cache.get_cached("bad-model")
        assert entry is not None
        assert entry.status == "rejected"

    def test_update_status(self, cache):
        cache.cache_result("m1", "model", {}, "a", status="approved")
        assert cache.update_status("m1", "rejected") is True
        entry = cache.get_cached("m1")
        assert entry.status == "rejected"

    def test_compute_content_hash(self):
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("hello")
        h3 = compute_content_hash("world")
        assert h1 == h2
        assert h1 != h3


# ===========================================================================
# Re-Validation Scheduler Tests
# ===========================================================================

class TestRevalidationScheduler:
    @pytest.mark.asyncio
    async def test_manual_trigger(self, scheduler):
        run = await scheduler.trigger_revalidation(trigger="manual")
        assert isinstance(run, RevalidationRun)
        assert run.trigger == "manual"
        assert run.completed_at is not None

    @pytest.mark.asyncio
    async def test_revalidation_no_change(self, cache, scheduler):
        cache.cache_result("m1", "model", {"verdict": "trusted"}, "h", status="approved")
        # Backdate so it's due for revalidation
        entry = cache.get_cached("m1")
        entry.validated_at = 0
        run = await scheduler.trigger_revalidation()
        assert run.components_checked >= 0

    @pytest.mark.asyncio
    async def test_revalidation_with_callback(self, cache):
        cache.cache_result("m1", "model", {"verdict": "trusted"}, "h", status="approved")
        entry = cache.get_cached("m1")
        entry.validated_at = 0  # force revalidation

        async def mock_revalidate(cid, ctype):
            return type("R", (), {"verdict": type("V", (), {"value": "rejected"})(), "findings": []})()

        sched = RevalidationScheduler(
            cache=cache, revalidate_fn=mock_revalidate,
            active_interval_hours=0.001,
        )
        run = await sched.trigger_revalidation()
        assert len(run.status_changes) == 1
        assert run.status_changes[0]["old_status"] == "approved"
        assert run.status_changes[0]["new_status"] == "rejected"

    @pytest.mark.asyncio
    async def test_scheduler_status(self, scheduler):
        status = scheduler.get_status()
        assert status["running"] is False
        assert status["total_runs"] == 0

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, scheduler):
        await scheduler.start()
        assert scheduler.is_running is True
        await scheduler.stop()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_scheduler_handles_errors(self, cache):
        cache.cache_result("m1", "model", {}, "h", status="approved")
        entry = cache.get_cached("m1")
        entry.validated_at = 0

        async def failing_fn(cid, ctype):
            raise RuntimeError("Validation service down")

        sched = RevalidationScheduler(
            cache=cache, revalidate_fn=failing_fn,
            active_interval_hours=0.001,
        )
        run = await sched.trigger_revalidation()
        assert run.errors >= 1

    @pytest.mark.asyncio
    async def test_retroactive_on_new_threat(self, cache):
        cache.cache_result("evil-model", "model", {"info": "clean"}, "h", status="approved")
        sched = RevalidationScheduler(cache=cache)
        run = await sched.on_new_threat("evil-model")
        # Should have invalidated and run revalidation
        assert run is not None

    @pytest.mark.asyncio
    async def test_retroactive_no_match(self, cache):
        cache.cache_result("good-model", "model", {}, "h", status="approved")
        sched = RevalidationScheduler(cache=cache)
        run = await sched.on_new_threat("totally-unrelated-threat")
        assert run is None

    @pytest.mark.asyncio
    async def test_run_summary_counts(self, cache):
        for i in range(5):
            cache.cache_result(f"m{i}", "model", {}, f"h{i}", status="approved")
            entry = cache.get_cached(f"m{i}")
            entry.validated_at = 0

        sched = RevalidationScheduler(cache=cache, active_interval_hours=0.001)
        run = await sched.trigger_revalidation()
        assert run.components_checked == 5

    @pytest.mark.asyncio
    async def test_last_run_stored(self, scheduler):
        assert scheduler.last_run is None
        await scheduler.trigger_revalidation()
        assert scheduler.last_run is not None
        assert scheduler.schedule.total_runs == 1


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_mpv_cache(self):
        """MPV validates → cached → re-validates."""
        from aegis.layers.supply_chain.provenance_validator import ModelProvenanceValidator
        mpv = ModelProvenanceValidator()
        cache = SupplyChainValidationCache()

        report = await mpv.validate(
            "google/bert", source_registry="huggingface.co", source_org="google",
        )
        cache.cache_result("google/bert", "model", report, "h1", status="approved")

        cached = cache.get_cached("google/bert", content_hash="h1")
        assert cached is not None
        assert cached.status == "approved"

    @pytest.mark.asyncio
    async def test_spa_cache_fast_path(self):
        """SPA audits → cached → second call hits cache."""
        from aegis.layers.supply_chain.skill_auditor import SkillPluginAuditor
        spa = SkillPluginAuditor()
        cache = SupplyChainValidationCache()

        report = await spa.audit("safe-tool", description="A calculator")
        cache.cache_result("safe-tool", "skill", report, "h1", status="approved")

        # Second call: cache hit
        cached = cache.get_cached("safe-tool", content_hash="h1")
        assert cached is not None

    def test_dca_findings_trigger_alert(self):
        """Malicious dependency produces a critical finding."""
        data_dir = Path(__file__).parent.parent / "data"
        dca = DependencyChainAnalyzer(
            malicious_file=data_dir / "malicious_packages.json",
            popular_file=data_dir / "popular_packages.json",
        )
        deps = [DependencyInfo(name="event-stream", ecosystem="npm")]
        report = dca.analyze(deps)
        assert report.overall_risk == "critical"
        assert report.critical_count == 1

    @pytest.mark.asyncio
    async def test_revalidation_endpoint_structure(self):
        """Scheduler trigger returns proper run structure."""
        cache = SupplyChainValidationCache()
        sched = RevalidationScheduler(cache=cache)
        run = await sched.trigger_revalidation()
        status = sched.get_status()
        assert "running" in status
        assert "total_runs" in status
        assert status["total_runs"] == 1

    @pytest.mark.asyncio
    async def test_dca_with_cache_integration(self):
        """DCA analysis result cached and retrievable."""
        data_dir = Path(__file__).parent.parent / "data"
        dca = DependencyChainAnalyzer(
            malicious_file=data_dir / "malicious_packages.json",
            popular_file=data_dir / "popular_packages.json",
        )
        cache = SupplyChainValidationCache()

        deps = [DependencyInfo(name="requests", ecosystem="pypi")]
        report = dca.analyze(deps)
        cache.cache_result("deps:abc", "dependency", report, "hash1", status="approved")

        cached = cache.get_cached("deps:abc")
        assert cached is not None
        assert cached.validation_result.overall_risk == "clean"
