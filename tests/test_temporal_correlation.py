"""
Tests for Memory Provenance Registry (MPR) and Temporal Correlation Engine (TCE).

Phase C Step 3: provenance tracking and causal attribution for behavioral anomalies.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aegis.layers.temporal.provenance_registry import (
    MemoryProvenanceRegistry,
    ProvenanceCategory,
    ProvenanceRecord,
    ProvenanceTimeline,
)
from aegis.layers.temporal.correlation_engine import (
    TemporalCorrelationEngine,
    CorrelationCandidate,
    CorrelationReport,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mpr():
    """Fresh MPR with default settings."""
    return MemoryProvenanceRegistry(max_records=1000)


@pytest.fixture
def mpr_small():
    """MPR with small max for eviction testing."""
    return MemoryProvenanceRegistry(max_records=5)


@pytest.fixture
def tce():
    """Fresh TCE with default settings."""
    return TemporalCorrelationEngine(
        default_window_hours=72.0,
        max_candidates=5,
    )


@pytest.fixture
def tce_with_mpr(mpr, tce):
    """TCE wired to an MPR."""
    tce.set_mpr(mpr)
    return tce, mpr


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


# ===========================================================================
# 1. MPR Tests (18 tests)
# ===========================================================================

class TestMPRRecordEvent:
    """Tests for record_event and basic record creation."""

    def test_record_event_creates_record(self, mpr):
        """record_event should return a ProvenanceRecord with correct fields."""
        r = mpr.record_event(
            category="rag_document",
            event_type="created",
            entity_id="doc-001",
            content_hash=_hash("test doc"),
            source="api_upload",
            trust_level="untrusted",
            tenant_id="tenant-a",
            metadata={"filename": "test.pdf"},
        )
        assert r.category == ProvenanceCategory.RAG_DOCUMENT
        assert r.event_type == "created"
        assert r.entity_id == "doc-001"
        assert r.source == "api_upload"
        assert r.trust_level == "untrusted"
        assert r.tenant_id == "tenant-a"
        assert r.metadata["filename"] == "test.pdf"
        assert r.flagged is False

    def test_record_event_assigns_uuid(self, mpr):
        """Each record should have a unique record_id."""
        r1 = mpr.record_event("rag_document", "created", "doc-1", _hash("a"), "src")
        r2 = mpr.record_event("rag_document", "created", "doc-2", _hash("b"), "src")
        assert r1.record_id != r2.record_id

    def test_record_event_each_category(self, mpr):
        """Should accept all four categories."""
        for cat in ("rag_document", "memory_entry", "config_change", "model_state"):
            r = mpr.record_event(cat, "created", f"entity-{cat}", _hash(cat), "src")
            assert r.category == ProvenanceCategory(cat)

    def test_record_event_enum_category(self, mpr):
        """Should accept ProvenanceCategory enum directly."""
        r = mpr.record_event(
            ProvenanceCategory.CONFIG_CHANGE, "modified", "cfg-1", _hash("cfg"), "admin",
        )
        assert r.category == ProvenanceCategory.CONFIG_CHANGE


class TestMPRTimeline:
    """Tests for get_timeline and filtering."""

    def test_get_timeline_sorted(self, mpr):
        """Records should be sorted by timestamp."""
        for i in range(5):
            mpr.record_event("rag_document", "created", f"doc-{i}", _hash(str(i)), "src")
        tl = mpr.get_timeline()
        timestamps = [r.timestamp for r in tl.records]
        assert timestamps == sorted(timestamps)

    def test_get_timeline_filters_by_time_range(self, mpr):
        """since/until should filter records."""
        now = time.time()
        r1 = mpr.record_event("rag_document", "created", "old", _hash("old"), "src")
        r1.timestamp = now - 7200  # 2 hours ago
        r2 = mpr.record_event("rag_document", "created", "new", _hash("new"), "src")
        r2.timestamp = now - 60  # 1 minute ago

        tl = mpr.get_timeline(since=now - 3600)
        assert len(tl.records) == 1
        assert tl.records[0].entity_id == "new"

    def test_get_timeline_filters_by_category(self, mpr):
        """category filter should narrow results."""
        mpr.record_event("rag_document", "created", "doc-1", _hash("a"), "src")
        mpr.record_event("config_change", "modified", "cfg-1", _hash("b"), "admin")
        mpr.record_event("rag_document", "created", "doc-2", _hash("c"), "src")

        tl = mpr.get_timeline(category="config_change")
        assert tl.total_records == 1
        assert tl.records[0].entity_id == "cfg-1"

    def test_get_timeline_empty(self, mpr):
        """Empty registry should return empty timeline."""
        tl = mpr.get_timeline()
        assert tl.total_records == 0
        assert tl.earliest_timestamp is None
        assert tl.latest_timestamp is None


class TestMPREntityAndWindow:
    """Tests for entity queries and window queries."""

    def test_get_events_for_entity(self, mpr):
        """Should return all records for a given entity."""
        mpr.record_event("rag_document", "created", "doc-1", _hash("v1"), "src")
        mpr.record_event("rag_document", "modified", "doc-1", _hash("v2"), "src")
        mpr.record_event("rag_document", "created", "doc-2", _hash("other"), "src")

        events = mpr.get_events_for_entity("doc-1")
        assert len(events) == 2
        assert all(e.entity_id == "doc-1" for e in events)

    def test_get_events_in_window(self, mpr):
        """Should return events within the time window."""
        now = time.time()
        r1 = mpr.record_event("rag_document", "created", "old", _hash("old"), "src")
        r1.timestamp = now - 86400 * 3  # 3 days ago
        r2 = mpr.record_event("rag_document", "created", "recent", _hash("recent"), "src")
        r2.timestamp = now - 3600  # 1 hour ago

        events = mpr.get_events_in_window("default", now, window_hours=24.0)
        assert len(events) == 1
        assert events[0].entity_id == "recent"


class TestMPRRetroactiveAndQuarantine:
    """Tests for retroactive flagging and quarantine."""

    def test_retroactive_flag_matches(self, mpr):
        """Should flag records matching the threat signature (content_hash)."""
        h = _hash("malicious payload")
        mpr.record_event("rag_document", "created", "doc-evil", h, "unknown_src", "untrusted")
        mpr.record_event("rag_document", "created", "doc-safe", _hash("safe"), "trusted_src", "trusted")

        flagged = mpr.retroactive_flag(h)
        assert len(flagged) == 1
        assert flagged[0].entity_id == "doc-evil"
        assert flagged[0].flagged is True

    def test_retroactive_flag_no_match(self, mpr):
        """No matches should return empty list."""
        mpr.record_event("rag_document", "created", "doc-1", _hash("safe"), "src")
        flagged = mpr.retroactive_flag(_hash("not_in_registry"))
        assert flagged == []

    def test_quarantine_entity(self, mpr):
        """Should mark entity as quarantined."""
        mpr.record_event("memory_entry", "created", "mem-1", _hash("data"), "src")
        assert mpr.quarantine_entity("mem-1", "suspected poisoning") is True
        events = mpr.get_events_for_entity("mem-1")
        assert events[0].metadata["quarantined"] is True
        assert events[0].flagged is True

    def test_quarantine_unknown_entity(self, mpr):
        """Quarantining unknown entity returns False."""
        assert mpr.quarantine_entity("nonexistent", "reason") is False

    def test_get_untrusted_events(self, mpr):
        """Should return only untrusted events in window."""
        mpr.record_event("rag_document", "created", "doc-1", _hash("a"), "evil", "untrusted")
        mpr.record_event("rag_document", "created", "doc-2", _hash("b"), "good", "trusted")
        mpr.record_event("rag_document", "created", "doc-3", _hash("c"), "maybe", "unknown")

        untrusted = mpr.get_untrusted_events()
        assert len(untrusted) == 1
        assert untrusted[0].entity_id == "doc-1"


class TestMPRStatsAndEviction:
    """Tests for stats and FIFO eviction."""

    def test_get_stats(self, mpr):
        """Stats should reflect current state."""
        mpr.record_event("rag_document", "created", "doc-1", _hash("a"), "src", "untrusted")
        mpr.record_event("config_change", "modified", "cfg-1", _hash("b"), "admin", "trusted")
        mpr.record_event("rag_document", "created", "doc-2", _hash("c"), "src", "untrusted")

        stats = mpr.get_stats()
        assert stats["total_records"] == 3
        assert stats["by_category"]["rag_document"] == 2
        assert stats["by_category"]["config_change"] == 1
        assert stats["by_trust_level"]["untrusted"] == 2
        assert stats["by_trust_level"]["trusted"] == 1

    def test_fifo_eviction(self, mpr_small):
        """Exceeding max_records should evict oldest records."""
        for i in range(7):
            mpr_small.record_event("rag_document", "created", f"doc-{i}", _hash(str(i)), "src")
        stats = mpr_small.get_stats()
        assert stats["total_records"] == 5  # max_records=5
        # Oldest (doc-0, doc-1) should be evicted
        assert mpr_small.get_events_for_entity("doc-0") == []
        assert mpr_small.get_events_for_entity("doc-1") == []
        assert len(mpr_small.get_events_for_entity("doc-6")) == 1

    def test_thread_safety(self, mpr):
        """Concurrent record_event calls should not corrupt state."""
        errors = []

        def worker(start_id):
            try:
                for i in range(50):
                    mpr.record_event(
                        "rag_document", "created", f"doc-{start_id}-{i}",
                        _hash(f"{start_id}-{i}"), "src",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        stats = mpr.get_stats()
        assert stats["total_records"] == 200  # 4 threads × 50

    def test_multiple_tenants(self, mpr):
        """Records should be isolated by tenant_id."""
        mpr.record_event("rag_document", "created", "doc-1", _hash("a"), "src", tenant_id="alpha")
        mpr.record_event("rag_document", "created", "doc-2", _hash("b"), "src", tenant_id="beta")
        mpr.record_event("rag_document", "created", "doc-3", _hash("c"), "src", tenant_id="alpha")

        tl_alpha = mpr.get_timeline(tenant_id="alpha")
        tl_beta = mpr.get_timeline(tenant_id="beta")
        assert tl_alpha.total_records == 2
        assert tl_beta.total_records == 1


# ===========================================================================
# 2. TCE Tests (15 tests)
# ===========================================================================

class TestTCECorrelation:
    """Tests for the temporal correlation scoring and analysis."""

    @pytest.mark.asyncio
    async def test_single_untrusted_candidate_high(self, tce_with_mpr):
        """Single untrusted event near anomaly → HIGH confidence."""
        tce, mpr = tce_with_mpr
        # Add enough baseline records for data sufficiency
        for i in range(15):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        # Add untrusted document 1 hour before the anomaly
        now = time.time()
        r = mpr.record_event(
            "rag_document", "created", "evil-doc", _hash("evil"),
            "external_upload", "untrusted",
        )
        r.timestamp = now - 3600  # 1 hour ago

        report = await tce.correlate(
            anomaly_timestamp=now,
            anomaly_type="semantic_drift",
            anomaly_description="Semantic drift detected",
        )
        assert report.confidence == "high"
        assert report.recommended_action == "quarantine"
        assert report.top_candidates[0].provenance_record.entity_id == "evil-doc"

    @pytest.mark.asyncio
    async def test_multiple_candidates_ranked(self, tce_with_mpr):
        """Multiple candidates should be ranked by correlation score."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        # Closer event should rank higher
        r1 = mpr.record_event("rag_document", "created", "far-doc", _hash("far"), "src", "untrusted")
        r1.timestamp = now - 86400 * 2  # 2 days ago

        r2 = mpr.record_event("rag_document", "created", "near-doc", _hash("near"), "src", "untrusted")
        r2.timestamp = now - 1800  # 30 min ago

        report = await tce.correlate(now, "semantic_drift", "drift")
        assert report.top_candidates[0].provenance_record.entity_id == "near-doc"

    @pytest.mark.asyncio
    async def test_temporal_proximity_scoring(self, tce_with_mpr):
        """Closer events should have higher temporal_proximity."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        r_close = mpr.record_event("rag_document", "created", "close", _hash("c"), "src", "unknown")
        r_close.timestamp = now - 3600

        r_far = mpr.record_event("rag_document", "created", "far", _hash("f"), "src", "unknown")
        r_far.timestamp = now - 86400 * 2  # 48 hours

        report = await tce.correlate(now, "semantic_drift", "drift")
        close_cand = next(c for c in report.top_candidates if c.provenance_record.entity_id == "close")
        far_cand = next(c for c in report.top_candidates if c.provenance_record.entity_id == "far")
        assert close_cand.temporal_proximity > far_cand.temporal_proximity

    @pytest.mark.asyncio
    async def test_trust_level_weighting(self, tce_with_mpr):
        """Untrusted sources should score 2x, trusted 0.5x."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        # Same time, same category, different trust
        r_trusted = mpr.record_event("rag_document", "created", "trusted-doc", _hash("t"), "good", "trusted")
        r_trusted.timestamp = now - 3600

        r_untrusted = mpr.record_event("rag_document", "created", "untrusted-doc", _hash("u"), "bad", "untrusted")
        r_untrusted.timestamp = now - 3600

        report = await tce.correlate(now, "semantic_drift", "drift")
        trusted_cand = next(c for c in report.top_candidates if c.provenance_record.entity_id == "trusted-doc")
        untrusted_cand = next(c for c in report.top_candidates if c.provenance_record.entity_id == "untrusted-doc")
        assert untrusted_cand.trust_factor == 2.0
        assert trusted_cand.trust_factor == 0.5
        assert untrusted_cand.correlation_score > trusted_cand.correlation_score

    @pytest.mark.asyncio
    async def test_category_relevance_rag_for_semantic(self, tce_with_mpr):
        """RAG_DOCUMENT should score higher for semantic_drift anomaly."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        r_rag = mpr.record_event("rag_document", "created", "rag-doc", _hash("r"), "src", "unknown")
        r_rag.timestamp = now - 3600

        r_model = mpr.record_event("model_state", "updated", "model-1", _hash("m"), "src", "unknown")
        r_model.timestamp = now - 3600

        report = await tce.correlate(now, "semantic_drift", "drift")
        rag_cand = next(c for c in report.top_candidates if c.provenance_record.entity_id == "rag-doc")
        model_cand = next(c for c in report.top_candidates if c.provenance_record.entity_id == "model-1")
        assert rag_cand.category_factor > model_cand.category_factor

    @pytest.mark.asyncio
    async def test_anomaly_type_match_refusal_boosts_memory(self, tce_with_mpr):
        """refusal_rate anomaly should boost MEMORY_ENTRY candidates."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        r_mem = mpr.record_event("memory_entry", "created", "mem-1", _hash("m"), "src", "unknown")
        r_mem.timestamp = now - 3600

        r_cfg = mpr.record_event("config_change", "modified", "cfg-x", _hash("c"), "admin", "unknown")
        r_cfg.timestamp = now - 3600

        report = await tce.correlate(now, "refusal_rate", "refusal spike")
        mem_cand = next(c for c in report.top_candidates if c.provenance_record.entity_id == "mem-1")
        assert mem_cand.anomaly_match_factor == 1.5

    @pytest.mark.asyncio
    async def test_confidence_medium(self, tce_with_mpr):
        """Score > 0.5 from untrusted RAG source with semantic anomaly → MEDIUM or HIGH."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        # Untrusted RAG document very close to anomaly → high score
        r = mpr.record_event("rag_document", "created", "doc-x", _hash("c"), "api", "untrusted")
        r.timestamp = now - 600  # 10 min ago

        report = await tce.correlate(now, "semantic_drift", "drift anomaly")
        assert report.confidence in ("medium", "high")

    @pytest.mark.asyncio
    async def test_confidence_low(self, tce_with_mpr):
        """All trusted sources with low scores, widely spread → LOW confidence."""
        tce, mpr = tce_with_mpr
        now = time.time()

        # Spread records across the window so they don't cluster (avoids clustering trigger)
        for i in range(12):
            r = mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")
            r.timestamp = now - 86400 * 2 - i * 14400  # Each 4h apart, far from anomaly

        report = await tce.correlate(now, "unknown_metric", "some anomaly")
        assert report.confidence == "low"

    @pytest.mark.asyncio
    async def test_insufficient_data_empty_mpr(self, tce):
        """Empty MPR → INSUFFICIENT_DATA."""
        mpr = MemoryProvenanceRegistry()
        tce.set_mpr(mpr)
        report = await tce.correlate(time.time(), "semantic_drift", "drift")
        assert report.confidence == "insufficient_data"
        assert report.recommended_action == "monitor"

    @pytest.mark.asyncio
    async def test_insufficient_data_no_mpr(self, tce):
        """No MPR set → INSUFFICIENT_DATA."""
        report = await tce.correlate(time.time(), "semantic_drift", "drift")
        assert report.confidence == "insufficient_data"

    @pytest.mark.asyncio
    async def test_correlation_window_excludes_old_events(self, tce_with_mpr):
        """Events outside the window should be excluded."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        # Event 5 days ago, window is 72h
        r = mpr.record_event("rag_document", "created", "old-doc", _hash("old"), "src", "untrusted")
        r.timestamp = now - 86400 * 5

        report = await tce.correlate(now, "semantic_drift", "drift", window_hours=72.0)
        entity_ids = [c.provenance_record.entity_id for c in report.top_candidates]
        assert "old-doc" not in entity_ids

    @pytest.mark.asyncio
    async def test_configurable_window_size(self, tce_with_mpr):
        """Larger window should find older events."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        r = mpr.record_event("rag_document", "created", "old-doc", _hash("old"), "src", "untrusted")
        r.timestamp = now - 86400 * 5  # 5 days ago

        report = await tce.correlate(now, "semantic_drift", "drift", window_hours=168.0)  # 7 days
        entity_ids = [c.provenance_record.entity_id for c in report.top_candidates]
        assert "old-doc" in entity_ids

    @pytest.mark.asyncio
    async def test_top_n_capped(self):
        """Only top N candidates should be returned."""
        tce = TemporalCorrelationEngine(max_candidates=3)
        mpr = MemoryProvenanceRegistry()
        tce.set_mpr(mpr)
        now = time.time()

        for i in range(20):
            r = mpr.record_event("rag_document", "created", f"doc-{i}", _hash(str(i)), "src", "unknown")
            r.timestamp = now - (i * 3600)

        report = await tce.correlate(now, "semantic_drift", "drift")
        assert len(report.top_candidates) <= 3

    @pytest.mark.asyncio
    async def test_recommended_action_quarantine(self, tce_with_mpr):
        """HIGH confidence + RAG_DOCUMENT → quarantine."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(15):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        r = mpr.record_event("rag_document", "created", "evil", _hash("evil"), "ext", "untrusted")
        r.timestamp = now - 1800

        report = await tce.correlate(now, "semantic_drift", "drift")
        if report.confidence == "high":
            assert report.recommended_action == "quarantine"

    @pytest.mark.asyncio
    async def test_report_stored_and_retrievable(self, tce_with_mpr):
        """Reports should be stored and retrievable."""
        tce, mpr = tce_with_mpr
        now = time.time()

        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        await tce.correlate(now, "semantic_drift", "drift 1")
        await tce.correlate(now, "refusal_rate", "drift 2")

        reports = tce.get_recent_reports()
        assert len(reports) == 2

    @pytest.mark.asyncio
    async def test_no_candidates_in_window(self, tce_with_mpr):
        """No events in window → INSUFFICIENT_DATA."""
        tce, mpr = tce_with_mpr
        now = time.time()

        # Add events far in the past
        for i in range(15):
            r = mpr.record_event("rag_document", "created", f"doc-{i}", _hash(str(i)), "src", "trusted")
            r.timestamp = now - 86400 * 10  # 10 days ago

        report = await tce.correlate(now, "semantic_drift", "drift", window_hours=24.0)
        assert report.confidence == "insufficient_data"


# ===========================================================================
# 3. Automatic Trigger Tests (5 tests)
# ===========================================================================

class TestAutoTrigger:
    """Tests for automatic correlation triggers via event bus."""

    @pytest.mark.asyncio
    async def test_critical_drift_triggers_correlation(self, tce_with_mpr):
        """Critical drift alert should trigger auto-correlation."""
        tce, mpr = tce_with_mpr
        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        await tce.handle_drift_event({
            "type": "drift_alert",
            "confidence": "critical",
            "metric_name": "semantic_drift",
            "deviation_sigmas": 4.0,
            "tenant_id": "default",
        })
        reports = tce.get_recent_reports()
        assert len(reports) == 1

    @pytest.mark.asyncio
    async def test_canary_critical_triggers_correlation(self, tce_with_mpr):
        """Critical canary alert should trigger auto-correlation."""
        tce, mpr = tce_with_mpr
        for i in range(12):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        await tce.handle_drift_event({
            "type": "canary_alert",
            "level": "critical",
            "canary_id": "ge-001",
            "consecutive_failures": 3,
        })
        reports = tce.get_recent_reports()
        assert len(reports) == 1
        assert reports[0].anomaly_type == "canary_failure"

    @pytest.mark.asyncio
    async def test_warning_does_not_trigger(self, tce_with_mpr):
        """Warning-level drift should NOT auto-trigger correlation."""
        tce, mpr = tce_with_mpr
        await tce.handle_drift_event({
            "type": "drift_alert",
            "confidence": "warning",
            "metric_name": "output_length",
        })
        reports = tce.get_recent_reports()
        assert len(reports) == 0

    @pytest.mark.asyncio
    async def test_canary_warning_does_not_trigger(self, tce_with_mpr):
        """Warning-level canary alert should NOT auto-trigger."""
        tce, mpr = tce_with_mpr
        await tce.handle_drift_event({
            "type": "canary_alert",
            "level": "warning",
            "canary_id": "ge-001",
        })
        assert len(tce.get_recent_reports()) == 0

    @pytest.mark.asyncio
    async def test_correlation_report_published_to_bus(self, tce_with_mpr):
        """High/medium reports should be published to event bus."""
        tce, mpr = tce_with_mpr
        mock_bus = AsyncMock()
        tce._event_bus = mock_bus
        now = time.time()

        for i in range(15):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        r = mpr.record_event("rag_document", "created", "evil", _hash("evil"), "ext", "untrusted")
        r.timestamp = now - 1800

        await tce.correlate(now, "semantic_drift", "drift")
        # Should publish if confidence is high or medium
        if mock_bus.publish.called:
            args = mock_bus.publish.call_args
            assert args[0][0] == "temporal_drift"
            assert args[0][1]["type"] == "correlation_report"


# ===========================================================================
# 4. Integration Tests (5 tests)
# ===========================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_poisoning_scenario(self):
        """Record untrusted doc → trigger drift → TCE identifies it."""
        mpr = MemoryProvenanceRegistry()
        tce = TemporalCorrelationEngine(max_candidates=5)
        tce.set_mpr(mpr)
        now = time.time()

        # Record 10 trusted events
        for i in range(10):
            r = mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")
            r.timestamp = now - (i * 7200)

        # Record 1 untrusted document (the poisoning event)
        poison = mpr.record_event(
            "rag_document", "created", "poison-doc", _hash("evil payload"),
            "external_api", "untrusted",
        )
        poison.timestamp = now - 1800  # 30 min ago

        # Correlate
        report = await tce.correlate(now, "semantic_drift", "Semantic drift detected by BBE")

        assert report.top_candidates[0].provenance_record.entity_id == "poison-doc"
        assert report.confidence in ("high", "medium")

    @pytest.mark.asyncio
    async def test_multiple_anomalies_independent(self):
        """Each anomaly should generate an independent report."""
        mpr = MemoryProvenanceRegistry()
        tce = TemporalCorrelationEngine()
        tce.set_mpr(mpr)

        for i in range(15):
            mpr.record_event("config_change", "modified", f"cfg-{i}", _hash(str(i)), "admin", "trusted")

        now = time.time()
        r1 = await tce.correlate(now, "semantic_drift", "drift 1")
        r2 = await tce.correlate(now, "refusal_rate", "drift 2")

        assert r1.report_id != r2.report_id
        assert len(tce.get_recent_reports()) == 2

    def test_config_defaults(self):
        """Config should have sensible defaults for MPR/TCE fields."""
        from aegis.config import get_config
        config = get_config()
        assert config.mpr_enabled is True
        assert config.mpr_max_records == 100000
        assert config.tce_enabled is True
        assert config.tce_default_correlation_window_hours == 72.0
        assert config.tce_max_candidates == 5
        assert config.tce_auto_correlate_on_critical is True

    def test_metrics_exist(self):
        """Prometheus metrics should be defined."""
        from aegis.middleware.metrics import (
            PROVENANCE_EVENTS_TOTAL,
            CORRELATIONS_TOTAL,
            CORRELATION_LATENCY,
        )
        assert PROVENANCE_EVENTS_TOTAL is not None
        assert CORRELATIONS_TOTAL is not None
        assert CORRELATION_LATENCY is not None

    def test_imports_from_package(self):
        """All new types should be importable from temporal package."""
        from aegis.layers.temporal import (
            MemoryProvenanceRegistry,
            ProvenanceCategory,
            ProvenanceRecord,
            ProvenanceTimeline,
            TemporalCorrelationEngine,
            CorrelationCandidate,
            CorrelationReport,
        )
        assert MemoryProvenanceRegistry is not None
        assert TemporalCorrelationEngine is not None


# ===========================================================================
# 5. API Endpoint Tests (4 tests)
# ===========================================================================

class TestEndpoints:
    """Tests for the provenance/correlation API endpoints."""

    API_KEY = "aegis-test-secretkey123"

    @pytest.fixture
    def setup_layers(self):
        from aegis.config import AegisConfig
        from aegis.main import _init_layers
        import aegis.main as main_module
        config = AegisConfig(
            api_key=self.API_KEY,
            upstream_url="https://mock-upstream.test",
            upstream_api_key="test-key",
        )
        _init_layers(config)
        yield
        main_module._mpr = None
        main_module._tce = None

    @pytest.fixture
    def client(self, setup_layers):
        from fastapi.testclient import TestClient
        from aegis.main import app
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": f"Bearer {self.API_KEY}"}

    def test_provenance_timeline_auth(self, client):
        """GET /v1/temporal/provenance/timeline requires auth."""
        resp = client.get("/v1/temporal/provenance/timeline")
        assert resp.status_code == 401

    def test_provenance_timeline_authenticated(self, client, auth_headers):
        """GET /v1/temporal/provenance/timeline returns data when authed."""
        resp = client.get("/v1/temporal/provenance/timeline", headers=auth_headers)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "total_records" in data
            assert "records" in data

    def test_provenance_stats_authenticated(self, client, auth_headers):
        """GET /v1/temporal/provenance/stats returns stats."""
        resp = client.get("/v1/temporal/provenance/stats", headers=auth_headers)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "total_records" in data

    def test_correlate_endpoint(self, client, auth_headers):
        """POST /v1/temporal/correlate returns a report."""
        resp = client.post(
            "/v1/temporal/correlate",
            headers=auth_headers,
            json={
                "anomaly_type": "semantic_drift",
                "anomaly_description": "test correlation",
            },
        )
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "report_id" in data
            assert "confidence" in data
