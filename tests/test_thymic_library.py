"""
Tests for the Attack Profile Library — five-tier probe corpus management.

Covers: loading, get_probes, add_probe, probe_count, tier validation,
expected_result correctness, required fields, edge cases.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aegis.layers.thymic.attack_profile_library import (
    AttackProfileLibrary,
    Probe,
    VALID_TIERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir() -> Path:
    """Return the thymic data directory."""
    return Path(__file__).parent.parent / "data" / "thymic"


@pytest.fixture
def library(data_dir: Path) -> AttackProfileLibrary:
    """Return a library loaded from seed data."""
    return AttackProfileLibrary(data_dir=data_dir)


@pytest.fixture
def empty_library() -> AttackProfileLibrary:
    """Return an empty library (no data dir)."""
    return AttackProfileLibrary()


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------

class TestLibraryLoading:
    def test_loads_from_json_files(self, library: AttackProfileLibrary):
        """Library loads probes from all JSON files."""
        counts = library.probe_count()
        assert counts[1] > 0, "Tier 1 should have probes"
        assert counts[5] > 0, "Tier 5 should have probes"

    def test_tier1_has_attack_probes(self, library: AttackProfileLibrary):
        """Tier 1 contains known attack signatures."""
        probes = library.get_probes(1)
        assert len(probes) >= 100, f"Expected 100+ Tier 1 probes, got {len(probes)}"

    def test_tier3_starts_empty(self, library: AttackProfileLibrary):
        """Tier 3 (emerging) starts empty."""
        probes = library.get_probes(3)
        assert len(probes) == 0

    def test_tier4_has_campaigns(self, library: AttackProfileLibrary):
        """Tier 4 has multi-turn campaign sequences."""
        probes = library.get_probes(4)
        assert len(probes) >= 3

    def test_tier4_campaigns_have_conversations(self, library: AttackProfileLibrary):
        """Tier 4 probes have conversation_sequence fields."""
        probes = library.get_probes(4)
        for p in probes:
            assert p.conversation_sequence is not None
            assert len(p.conversation_sequence) >= 2

    def test_tier5_has_benign_probes(self, library: AttackProfileLibrary):
        """Tier 5 has verified benign prompts."""
        probes = library.get_probes(5)
        assert len(probes) >= 50

    def test_empty_library_has_zero_probes(self, empty_library: AttackProfileLibrary):
        """Empty library has zero probes in all tiers."""
        for tier in VALID_TIERS:
            assert len(empty_library.get_probes(tier)) == 0


# ---------------------------------------------------------------------------
# get_probes tests
# ---------------------------------------------------------------------------

class TestGetProbes:
    def test_returns_correct_tier(self, library: AttackProfileLibrary):
        """get_probes returns probes only from the requested tier."""
        for tier in VALID_TIERS:
            probes = library.get_probes(tier)
            for p in probes:
                assert p.tier == tier

    def test_with_count_returns_sample(self, library: AttackProfileLibrary):
        """get_probes with count returns a random sample of that size."""
        count = 5
        probes = library.get_probes(1, count=count)
        assert len(probes) == count

    def test_count_larger_than_available(self, library: AttackProfileLibrary):
        """get_probes with count larger than available returns all."""
        all_probes = library.get_probes(1)
        probes = library.get_probes(1, count=len(all_probes) + 100)
        assert len(probes) == len(all_probes)

    def test_invalid_tier_raises(self, library: AttackProfileLibrary):
        """get_probes with invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tier"):
            library.get_probes(0)
        with pytest.raises(ValueError, match="Invalid tier"):
            library.get_probes(6)

    def test_get_all_probes(self, library: AttackProfileLibrary):
        """get_all_probes returns probes from all tiers."""
        all_probes = library.get_all_probes()
        counts = library.probe_count()
        expected = sum(counts.values())
        assert len(all_probes) == expected


# ---------------------------------------------------------------------------
# add_probe tests
# ---------------------------------------------------------------------------

class TestAddProbe:
    def test_appends_correctly(self, empty_library: AttackProfileLibrary):
        """add_probe appends a probe to the correct tier."""
        probe = Probe(id="test-1", text="test", tier=1, category="test")
        empty_library.add_probe(probe)
        assert len(empty_library.get_probes(1)) == 1
        assert empty_library.get_probes(1)[0].id == "test-1"

    def test_add_multiple(self, empty_library: AttackProfileLibrary):
        """Multiple add_probe calls accumulate."""
        for i in range(5):
            empty_library.add_probe(Probe(id=f"test-{i}", text=f"test-{i}", tier=3))
        assert len(empty_library.get_probes(3)) == 5

    def test_add_invalid_tier_raises(self, empty_library: AttackProfileLibrary):
        """add_probe with invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tier"):
            empty_library.add_probe(Probe(id="bad", text="bad", tier=99))


# ---------------------------------------------------------------------------
# probe_count tests
# ---------------------------------------------------------------------------

class TestProbeCount:
    def test_accurate_per_tier(self, library: AttackProfileLibrary):
        """probe_count returns accurate counts per tier."""
        counts = library.probe_count()
        for tier in VALID_TIERS:
            assert counts[tier] == len(library.get_probes(tier))

    def test_includes_all_tiers(self, library: AttackProfileLibrary):
        """probe_count includes all valid tiers."""
        counts = library.probe_count()
        for tier in VALID_TIERS:
            assert tier in counts


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------

class TestDataQuality:
    def test_tier5_all_pass(self, library: AttackProfileLibrary):
        """Tier 5 probes all have expected_result='pass'."""
        for p in library.get_probes(5):
            assert p.expected_result == "pass", f"Tier 5 probe {p.id} should be 'pass'"

    def test_tier1_all_block(self, library: AttackProfileLibrary):
        """Tier 1 probes all have expected_result='block'."""
        for p in library.get_probes(1):
            assert p.expected_result == "block", f"Tier 1 probe {p.id} should be 'block'"

    def test_tier4_all_block(self, library: AttackProfileLibrary):
        """Tier 4 probes all have expected_result='block'."""
        for p in library.get_probes(4):
            assert p.expected_result == "block", f"Tier 4 probe {p.id} should be 'block'"

    def test_probes_have_required_fields(self, library: AttackProfileLibrary):
        """All probes have required fields: id, text, tier, category."""
        for p in library.get_all_probes():
            assert p.id, f"Probe missing id"
            assert p.text, f"Probe {p.id} missing text"
            assert p.tier in VALID_TIERS, f"Probe {p.id} invalid tier: {p.tier}"

    def test_tier1_probes_have_mitre_tactic(self, library: AttackProfileLibrary):
        """Tier 1 probes have mitre_tactic set."""
        for p in library.get_probes(1):
            assert p.mitre_tactic, f"Tier 1 probe {p.id} missing mitre_tactic"

    def test_tier1_probes_have_expected_layer(self, library: AttackProfileLibrary):
        """Tier 1 probes have expected_detection_layer set."""
        for p in library.get_probes(1):
            assert p.expected_detection_layer, f"Tier 1 probe {p.id} missing expected_detection_layer"


# ---------------------------------------------------------------------------
# Probe dataclass validation
# ---------------------------------------------------------------------------

class TestProbeValidation:
    def test_invalid_tier_in_constructor(self):
        """Probe constructor rejects invalid tier."""
        with pytest.raises(ValueError, match="Invalid tier"):
            Probe(id="x", text="x", tier=0)

    def test_valid_tiers_accepted(self):
        """All valid tier numbers are accepted."""
        for tier in VALID_TIERS:
            p = Probe(id=f"t{tier}", text="test", tier=tier)
            assert p.tier == tier


# ---------------------------------------------------------------------------
# Save/load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_reload(self, library: AttackProfileLibrary):
        """Library can save to disk and reload with same counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            library.save(tmp_path)

            reloaded = AttackProfileLibrary(data_dir=tmp_path)
            orig_counts = library.probe_count()
            new_counts = reloaded.probe_count()
            assert new_counts[1] == orig_counts[1]
            assert new_counts[5] == orig_counts[5]
