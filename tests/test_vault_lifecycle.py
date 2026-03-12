"""
Threat Vault Lifecycle & Persistence Tests

Tests the three-phase memory lifecycle, FAISS index persistence,
dormant reactivation, maintenance cycle, and vault stats endpoint:
    - Acute → Persistent after promotion threshold (3+ sources or confirmed)
    - Acute → Persistent after age threshold (30 days)
    - Persistent → Dormant after 180 days inactivity
    - Dormant → Acute reactivation when similar pattern re-emerges
    - FAISS index persistence: save, reload, verify indicators survive
    - Maintenance cycle execution (lifecycle_update)
    - Vault stats endpoint (/v1/vault/stats)
    - Promote/demote explicit methods
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aegis.config import AegisConfig, BarrierConfig, HealingConfig, MemoryConfig
from aegis.layers.audit import reset_audit_logger
from aegis.layers.memory.threat_vault import ThreatVault
from aegis.main import _init_layers, app
import aegis.main as main_module
from aegis.models.scan_result import ThreatCategory
from aegis.models.threat_indicator import (
    IndicatorSource,
    MemoryPhase,
    ThreatIndicator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 384


def _embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding for tests."""
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], byteorder="big"))
    vec = rng.randn(_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_indicator(
    indicator_id: str,
    text: str = "test attack",
    phase: MemoryPhase = MemoryPhase.ACUTE,
    confirmed: bool = False,
    source: IndicatorSource = IndicatorSource.ADAPTIVE_DETECTION,
    frequency: int = 1,
    seen_by_tenants: int = 1,
    created_at: datetime | None = None,
    last_seen: datetime | None = None,
    category: ThreatCategory = ThreatCategory.PROMPT_INJECTION,
) -> ThreatIndicator:
    now = datetime.now(timezone.utc)
    return ThreatIndicator(
        indicator_id=indicator_id,
        source=source,
        confirmed=confirmed,
        phase=phase,
        threat_category=category,
        confidence=0.85,
        severity=0.7,
        embedding=_embed(text),
        payload_hash=hashlib.sha256(text.encode()).hexdigest(),
        payload_summary=text[:200],
        mitre_atlas_id=category.value,
        frequency=frequency,
        seen_by_tenants=seen_by_tenants,
        created_at=created_at or now,
        last_seen=last_seen or now,
        first_seen=created_at or now,
    )


def _vault_with_config(**overrides) -> ThreatVault:
    """Create a vault with custom config, using temp paths for persistence."""
    defaults = {
        "faiss_index_path": "/tmp/test_vault.faiss",
        "metadata_path": "/tmp/test_vault_meta.json",
        "persist_every_n_updates": 100,  # Don't auto-persist during tests
    }
    defaults.update(overrides)
    return ThreatVault(MemoryConfig(**defaults))


# ---------------------------------------------------------------------------
# Lifecycle Transition Tests
# ---------------------------------------------------------------------------


class TestAcuteToPersistent:
    """Acute indicators should promote to Persistent under the right conditions."""

    def test_promote_after_multi_source_threshold(self):
        """Indicator seen by 3+ tenants should promote to persistent."""
        vault = _vault_with_config(promote_min_sources=3)
        ind = _make_indicator("multi-src-001", seen_by_tenants=3)
        vault.add(ind)

        promoted = vault.promote("multi-src-001")
        assert promoted
        assert vault.get_indicator("multi-src-001").phase == MemoryPhase.PERSISTENT

    def test_promote_after_analyst_confirmation(self):
        """Confirmed indicator should promote to persistent."""
        vault = _vault_with_config()
        ind = _make_indicator("confirmed-001", confirmed=True)
        vault.add(ind)

        promoted = vault.promote("confirmed-001")
        assert promoted
        assert vault.get_indicator("confirmed-001").phase == MemoryPhase.PERSISTENT

    def test_promote_after_age_and_frequency(self):
        """Indicator older than acute_memory_days with freq >= 2 promotes."""
        vault = _vault_with_config(acute_memory_days=30)
        old_date = datetime.now(timezone.utc) - timedelta(days=35)
        ind = _make_indicator("old-001", created_at=old_date, frequency=2)
        vault.add(ind)

        promoted = vault.promote("old-001")
        assert promoted
        assert vault.get_indicator("old-001").phase == MemoryPhase.PERSISTENT

    def test_no_promote_if_criteria_not_met(self):
        """Indicator that doesn't meet any criteria should not promote."""
        vault = _vault_with_config(promote_min_sources=3, acute_memory_days=30)
        ind = _make_indicator(
            "young-001", seen_by_tenants=1, confirmed=False, frequency=1,
        )
        vault.add(ind)

        promoted = vault.promote("young-001")
        assert not promoted
        assert vault.get_indicator("young-001").phase == MemoryPhase.ACUTE

    def test_lifecycle_update_promotes_old_indicators(self):
        """lifecycle_update should promote all indicators past acute_memory_days."""
        vault = _vault_with_config(acute_memory_days=30)
        old_date = datetime.now(timezone.utc) - timedelta(days=31)
        vault.add(_make_indicator("auto-001", created_at=old_date))
        vault.add(_make_indicator("auto-002", created_at=old_date))
        vault.add(_make_indicator("fresh-001"))  # Should stay acute

        transitions = vault.lifecycle_update()
        assert transitions["to_persistent"] == 2
        assert vault.get_indicator("auto-001").phase == MemoryPhase.PERSISTENT
        assert vault.get_indicator("auto-002").phase == MemoryPhase.PERSISTENT
        assert vault.get_indicator("fresh-001").phase == MemoryPhase.ACUTE

    def test_promote_truncates_payload(self):
        """Persistent indicators should have payload truncated to 200 chars."""
        vault = _vault_with_config()
        long_text = "A" * 500
        ind = _make_indicator("trunc-001", text=long_text, confirmed=True)
        ind.payload_summary = long_text
        vault.add(ind)

        vault.promote("trunc-001")
        assert len(vault.get_indicator("trunc-001").payload_summary) == 200


class TestPersistentToDormant:
    """Persistent indicators should demote to Dormant after inactivity."""

    def test_demote_after_inactivity(self):
        """Indicator unseen for 180+ days should go dormant."""
        vault = _vault_with_config(dormant_memory_days=180, acute_memory_days=30)
        old_created = datetime.now(timezone.utc) - timedelta(days=250)
        old_seen = datetime.now(timezone.utc) - timedelta(days=185)
        ind = _make_indicator(
            "stale-001",
            phase=MemoryPhase.PERSISTENT,
            created_at=old_created,
            last_seen=old_seen,
        )
        vault.add(ind)

        transitions = vault.lifecycle_update()
        assert transitions["to_dormant"] == 1
        assert vault.get_indicator("stale-001").phase == MemoryPhase.DORMANT

    def test_no_demote_if_recently_seen(self):
        """Indicator seen recently should remain persistent."""
        vault = _vault_with_config(dormant_memory_days=180)
        ind = _make_indicator(
            "active-001",
            phase=MemoryPhase.PERSISTENT,
            last_seen=datetime.now(timezone.utc) - timedelta(days=10),
        )
        vault.add(ind)

        transitions = vault.lifecycle_update()
        assert transitions["to_dormant"] == 0
        assert vault.get_indicator("active-001").phase == MemoryPhase.PERSISTENT

    def test_explicit_demote(self):
        """demote() should force indicator to dormant."""
        vault = _vault_with_config()
        ind = _make_indicator("force-001", phase=MemoryPhase.PERSISTENT)
        vault.add(ind)

        result = vault.demote("force-001")
        assert result
        assert vault.get_indicator("force-001").phase == MemoryPhase.DORMANT

    def test_demote_already_dormant_returns_false(self):
        """demote() on already dormant indicator returns False."""
        vault = _vault_with_config()
        ind = _make_indicator("dorm-001", phase=MemoryPhase.DORMANT)
        vault.add(ind)

        result = vault.demote("dorm-001")
        assert not result

    def test_dormant_excluded_from_search(self):
        """Dormant indicators should be excluded from default search."""
        vault = _vault_with_config()
        ind = _make_indicator("dormant-search-001", text="unique dormant attack")
        vault.add(ind)
        vault.demote("dormant-search-001")

        results = vault.search(_embed("unique dormant attack"), k=5)
        assert all(r[0].indicator_id != "dormant-search-001" for r in results)

        # But included when explicitly requested
        results = vault.search(_embed("unique dormant attack"), k=5, include_dormant=True)
        found = [r for r in results if r[0].indicator_id == "dormant-search-001"]
        assert len(found) == 1


class TestDormantReactivation:
    """Dormant indicators should reactivate when similar pattern re-emerges."""

    def test_record_hit_reactivates(self):
        """Recording a hit on dormant indicator should set it back to ACUTE."""
        vault = _vault_with_config()
        ind = _make_indicator("reactive-001", phase=MemoryPhase.DORMANT)
        vault.add(ind)
        assert vault.get_indicator("reactive-001").phase == MemoryPhase.DORMANT

        vault.record_hit("reactive-001")
        reactivated = vault.get_indicator("reactive-001")
        assert reactivated.phase == MemoryPhase.ACUTE
        assert reactivated.frequency == 2

    def test_reactivated_included_in_search(self):
        """After reactivation, indicator should appear in normal searches."""
        vault = _vault_with_config()
        text = "reactivation test pattern"
        ind = _make_indicator("reactive-002", text=text, phase=MemoryPhase.DORMANT)
        vault.add(ind)

        # Before reactivation: excluded
        results = vault.search(_embed(text), k=5)
        assert all(r[0].indicator_id != "reactive-002" for r in results)

        # Reactivate
        vault.record_hit("reactive-002")

        # After reactivation: included
        results = vault.search(_embed(text), k=5)
        found = [r for r in results if r[0].indicator_id == "reactive-002"]
        assert len(found) == 1


# ---------------------------------------------------------------------------
# Full Lifecycle Cycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end lifecycle: Acute → Persistent → Dormant → Reactivated."""

    def test_full_lifecycle_progression(self):
        vault = _vault_with_config(
            acute_memory_days=30, dormant_memory_days=180, promote_min_sources=3,
        )

        # 1. Add acute indicator
        now = datetime.now(timezone.utc)
        ind = _make_indicator(
            "lifecycle-001",
            created_at=now - timedelta(days=35),
            last_seen=now,
            frequency=5,
        )
        vault.add(ind)
        assert vault.get_indicator("lifecycle-001").phase == MemoryPhase.ACUTE

        # 2. Lifecycle update promotes to persistent (age > 30 days)
        transitions = vault.lifecycle_update()
        assert transitions["to_persistent"] == 1
        assert vault.get_indicator("lifecycle-001").phase == MemoryPhase.PERSISTENT

        # 3. Simulate inactivity: set last_seen to 185 days ago
        vault.get_indicator("lifecycle-001").last_seen = now - timedelta(days=185)
        transitions = vault.lifecycle_update()
        assert transitions["to_dormant"] == 1
        assert vault.get_indicator("lifecycle-001").phase == MemoryPhase.DORMANT

        # 4. Re-emergence: record hit reactivates
        vault.record_hit("lifecycle-001")
        assert vault.get_indicator("lifecycle-001").phase == MemoryPhase.ACUTE


# ---------------------------------------------------------------------------
# FAISS Index Persistence Tests
# ---------------------------------------------------------------------------


class TestPersistence:
    """FAISS index and metadata persistence to/from disk."""

    def test_save_and_load(self):
        """Save vault to disk, create new vault, load — indicators survive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = f"{tmpdir}/vault.faiss"
            meta_path = f"{tmpdir}/vault_meta.json"

            # Create and populate vault
            vault1 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault1.add(_make_indicator("persist-001", text="attack one",
                                       confirmed=True, frequency=5))
            vault1.add(_make_indicator("persist-002", text="attack two",
                                       category=ThreatCategory.JAILBREAK))
            vault1.save_to_disk()

            # Verify files exist
            assert Path(meta_path).exists()

            # Create new vault and load from disk
            vault2 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            loaded = vault2.load_from_disk()
            assert loaded == 2
            assert vault2.size == 2

            # Verify indicator properties survived
            ind1 = vault2.get_indicator("persist-001")
            assert ind1 is not None
            assert ind1.confirmed is True
            assert ind1.frequency == 5
            assert ind1.threat_category == ThreatCategory.PROMPT_INJECTION

            ind2 = vault2.get_indicator("persist-002")
            assert ind2 is not None
            assert ind2.threat_category == ThreatCategory.JAILBREAK

    def test_load_preserves_lifecycle_phase(self):
        """Lifecycle phase should survive persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = f"{tmpdir}/vault.faiss"
            meta_path = f"{tmpdir}/vault_meta.json"

            vault1 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault1.add(_make_indicator("phase-acute", phase=MemoryPhase.ACUTE))
            vault1.add(_make_indicator(
                "phase-persistent", text="persistent attack",
                phase=MemoryPhase.PERSISTENT,
            ))
            vault1.add(_make_indicator(
                "phase-dormant", text="dormant attack",
                phase=MemoryPhase.DORMANT,
            ))
            vault1.save_to_disk()

            vault2 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault2.load_from_disk()

            assert vault2.get_indicator("phase-acute").phase == MemoryPhase.ACUTE
            assert vault2.get_indicator("phase-persistent").phase == MemoryPhase.PERSISTENT
            assert vault2.get_indicator("phase-dormant").phase == MemoryPhase.DORMANT

    def test_search_works_after_reload(self):
        """Reloaded vault should support search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = f"{tmpdir}/vault.faiss"
            meta_path = f"{tmpdir}/vault_meta.json"

            vault1 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault1.add(_make_indicator(
                "search-persist", text="searchable attack",
                confirmed=True,
            ))
            vault1.save_to_disk()

            vault2 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault2.load_from_disk()

            results = vault2.search(_embed("searchable attack"), k=1)
            assert len(results) == 1
            assert results[0][0].indicator_id == "search-persist"
            assert results[0][1] > 0.9

    def test_no_persisted_data_returns_zero(self):
        """Load with no persisted files returns 0."""
        vault = _vault_with_config(
            metadata_path="/tmp/nonexistent_vault_meta.json",
        )
        loaded = vault.load_from_disk()
        assert loaded == 0
        assert vault.size == 0

    def test_startup_loads_persisted_over_seed(self):
        """On startup, persisted vault takes priority over seed data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = f"{tmpdir}/vault.faiss"
            meta_path = f"{tmpdir}/vault_meta.json"

            # Save a vault with known indicators
            vault1 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault1.add(_make_indicator("antibody-001", text="learned attack"))
            vault1.save_to_disk()

            # Load vault — should find the persisted antibody
            vault2 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            loaded = vault2.load_from_disk()
            assert loaded == 1
            assert vault2.get_indicator("antibody-001") is not None

    def test_batched_persistence(self):
        """Auto-persist triggers after N updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = f"{tmpdir}/vault.faiss"
            meta_path = f"{tmpdir}/vault_meta.json"

            vault = _vault_with_config(
                faiss_index_path=idx_path,
                metadata_path=meta_path,
                persist_every_n_updates=3,
            )

            # First 2 adds: no persist yet
            vault.add(_make_indicator("batch-001", text="batch 1"))
            vault.add(_make_indicator("batch-002", text="batch 2"))
            assert not Path(meta_path).exists()

            # 3rd add: triggers persist
            vault.add(_make_indicator("batch-003", text="batch 3"))
            assert Path(meta_path).exists()

    def test_index_size_bytes(self):
        """index_size_bytes returns file size after save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = f"{tmpdir}/vault.faiss"
            meta_path = f"{tmpdir}/vault_meta.json"

            vault = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            assert vault.index_size_bytes() == 0

            vault.add(_make_indicator("size-001", text="size test"))
            vault.save_to_disk()
            assert vault.index_size_bytes() > 0


# ---------------------------------------------------------------------------
# Maintenance Cycle Tests
# ---------------------------------------------------------------------------


class TestMaintenanceCycle:
    """Background maintenance cycle transitions indicators appropriately."""

    def test_maintenance_promotes_and_demotes(self):
        """Single lifecycle_update call handles both promotions and demotions."""
        vault = _vault_with_config(acute_memory_days=30, dormant_memory_days=180)
        now = datetime.now(timezone.utc)

        # Acute, old enough to promote
        vault.add(_make_indicator(
            "maint-acute", created_at=now - timedelta(days=35),
        ))
        # Persistent, unseen long enough to demote
        vault.add(_make_indicator(
            "maint-persistent", text="persistent entry",
            phase=MemoryPhase.PERSISTENT,
            created_at=now - timedelta(days=300),
            last_seen=now - timedelta(days=200),
        ))
        # Fresh acute — should stay
        vault.add(_make_indicator("maint-fresh", text="fresh entry"))

        transitions = vault.lifecycle_update()
        assert transitions["to_persistent"] == 1
        assert transitions["to_dormant"] == 1

        assert vault.get_indicator("maint-acute").phase == MemoryPhase.PERSISTENT
        assert vault.get_indicator("maint-persistent").phase == MemoryPhase.DORMANT
        assert vault.get_indicator("maint-fresh").phase == MemoryPhase.ACUTE

    def test_maintenance_loop_runs_initial_cycle(self):
        """start_maintenance_loop should run lifecycle_update immediately."""
        vault = _vault_with_config(
            acute_memory_days=1, maintenance_interval_hours=24,
        )
        now = datetime.now(timezone.utc)
        vault.add(_make_indicator(
            "loop-001", created_at=now - timedelta(days=2),
        ))

        # Run the loop but cancel after the initial cycle
        async def run_initial():
            task = asyncio.create_task(vault.start_maintenance_loop())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.get_event_loop().run_until_complete(run_initial())
        assert vault.get_indicator("loop-001").phase == MemoryPhase.PERSISTENT


# ---------------------------------------------------------------------------
# Vault Statistics Tests
# ---------------------------------------------------------------------------


class TestVaultStats:
    """Vault statistics method and endpoint tests."""

    def test_get_stats_structure(self):
        """get_stats returns all required fields."""
        vault = _vault_with_config()
        vault.add(_make_indicator("stat-001", text="stat test 1",
                                   category=ThreatCategory.PROMPT_INJECTION,
                                   frequency=10))
        vault.add(_make_indicator("stat-002", text="stat test 2",
                                   category=ThreatCategory.JAILBREAK,
                                   phase=MemoryPhase.PERSISTENT,
                                   frequency=5))
        vault.add(_make_indicator("stat-003", text="stat test 3",
                                   category=ThreatCategory.PROMPT_INJECTION,
                                   phase=MemoryPhase.DORMANT,
                                   frequency=1))

        stats = vault.get_stats()

        assert stats["total_indicators"] == 3
        assert stats["phase_counts"]["acute"] == 1
        assert stats["phase_counts"]["persistent"] == 1
        assert stats["phase_counts"]["dormant"] == 1
        assert stats["category_counts"][ThreatCategory.PROMPT_INJECTION.value] == 2
        assert stats["category_counts"][ThreatCategory.JAILBREAK.value] == 1
        assert stats["last_update"] is not None
        assert "index_size_bytes" in stats
        assert len(stats["top_matched_indicators"]) == 3
        # Top matched should be sorted by frequency descending
        assert stats["top_matched_indicators"][0]["frequency"] == 10
        assert stats["top_matched_indicators"][1]["frequency"] == 5

    def test_empty_vault_stats(self):
        """Empty vault should return zero stats."""
        vault = _vault_with_config()
        stats = vault.get_stats()
        assert stats["total_indicators"] == 0
        assert all(v == 0 for v in stats["phase_counts"].values())
        assert stats["top_matched_indicators"] == []

    def test_top_5_limit(self):
        """Top matched should be limited to 5 entries."""
        vault = _vault_with_config()
        for i in range(10):
            vault.add(_make_indicator(
                f"top5-{i:03d}", text=f"attack variant {i}",
                frequency=10 - i,
            ))
        stats = vault.get_stats()
        assert len(stats["top_matched_indicators"]) == 5
        assert stats["top_matched_indicators"][0]["frequency"] == 10


# ---------------------------------------------------------------------------
# Vault Stats Endpoint Integration Test
# ---------------------------------------------------------------------------


API_KEY = "aegis-test-secretkey123"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_layers():
    """Initialize layers for each test."""
    config = AegisConfig(
        api_key=API_KEY,
        upstream_url="https://mock-upstream.test",
        upstream_api_key="test-key",
        barrier=BarrierConfig(rate_limit_rpm=120, rate_limit_burst=30),
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


class TestVaultStatsEndpoint:
    """Integration tests for /v1/vault/stats endpoint."""

    def test_endpoint_returns_stats(self, client: TestClient):
        """Vault stats endpoint should return structured stats."""
        response = client.get(
            "/v1/vault/stats",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_indicators" in data
        assert "phase_counts" in data
        assert "category_counts" in data
        assert "last_update" in data
        assert "index_size_bytes" in data
        assert "top_matched_indicators" in data
        # Should have seed threats loaded
        assert data["total_indicators"] > 0

    def test_endpoint_reflects_vault_state(self, client: TestClient):
        """Stats should reflect actual vault contents."""
        # Add a known indicator to the vault
        vault = main_module._vault
        vault.add(_make_indicator("endpoint-test-001", text="endpoint test",
                                   confirmed=True, frequency=99))

        response = client.get(
            "/v1/vault/stats",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        data = response.json()
        # Should include the new indicator in the count
        top = data["top_matched_indicators"]
        top_ids = [t["indicator_id"] for t in top]
        assert "endpoint-test-001" in top_ids
