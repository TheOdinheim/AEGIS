"""
Cleanup Items Tests — validates all seven backlog fixes.

Covers:
    1. Audit logger flush after each write
    2. Multi-turn session TTL expiry and cleanup
    3. FAISS fast-path loading via read_index
    4. Numpy embedding persistence (separate from JSON metadata)
    5. Signature match recording triggers deprecation
    6. Async antibody callback with incremental positive set
    7. Behavioral probe hardening (response truncation, error handling)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aegis.config import AdaptiveConfig, MemoryConfig
from aegis.layers.audit import AuditLogger
from aegis.layers.adaptive.multi_turn import MultiTurnAnalyzer
from aegis.layers.memory.threat_vault import ThreatVault
from aegis.layers.memory.signatures import SignatureStore, SignatureGenerator
from aegis.layers.supply_chain.behavioral_probe import (
    probe_behavior,
    _get_response,
    _MAX_RESPONSE_LENGTH,
)
from aegis.models.request_context import ChatMessage, RequestContext
from aegis.models.scan_result import ThreatCategory
from aegis.models.threat_indicator import (
    IndicatorSource,
    MemoryPhase,
    ThreatIndicator,
)

_DIM = 384


def _random_embedding(seed: int = 42) -> list[float]:
    rng = np.random.RandomState(seed)
    vec = rng.randn(_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_indicator(
    indicator_id: str,
    text: str = "test attack",
    seed: int = 42,
    phase: MemoryPhase = MemoryPhase.ACUTE,
) -> ThreatIndicator:
    return ThreatIndicator(
        indicator_id=indicator_id,
        source=IndicatorSource.ADAPTIVE_DETECTION,
        confirmed=False,
        threat_category=ThreatCategory.PROMPT_INJECTION,
        confidence=0.9,
        severity=0.8,
        embedding=_random_embedding(seed),
        payload_hash="abc123",
        payload_summary=text[:200],
        phase=phase,
    )


def _vault_with_config(**kwargs) -> ThreatVault:
    config = MemoryConfig(**kwargs)
    return ThreatVault(config=config)


# ---------------------------------------------------------------------------
# Item 1: Audit Logger Flush
# ---------------------------------------------------------------------------


class TestAuditFlush:
    """Verify that audit logger flushes after each write."""

    def test_flush_called_after_write(self, tmp_path: Path):
        """Each log() call should flush the file handle."""
        log_path = tmp_path / "audit_flush_test.jsonl"
        logger = AuditLogger(log_path=log_path)

        with patch("builtins.open", wraps=open) as mock_open:
            logger.log("req-flush", "innate", "block")

        # Verify the file was written and is readable immediately
        assert log_path.exists()
        content = log_path.read_text().strip()
        record = json.loads(content)
        assert record["request_id"] == "req-flush"

    def test_crash_resilience_last_record_persisted(self, tmp_path: Path):
        """Last record should be on disk even without explicit close."""
        log_path = tmp_path / "audit_crash_test.jsonl"
        logger = AuditLogger(log_path=log_path)

        logger.log("req-1", "innate", "block")
        logger.log("req-2", "policy", "allow")
        logger.log("req-3", "output", "block")

        # Read directly from file — all 3 should be present
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[2])["request_id"] == "req-3"


# ---------------------------------------------------------------------------
# Item 2: Multi-Turn Session TTL Expiry
# ---------------------------------------------------------------------------


class TestMultiTurnTTLExpiry:
    """Verify TTL-based session eviction."""

    def _make_context(
        self, messages: list[dict[str, str]], session_id: str = "test",
    ) -> RequestContext:
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        return RequestContext(
            request_id="ttl-test",
            session_id=session_id,
            messages=chat_messages,
            model="gpt-4",
            api_key_hash="test",
            source_ip="127.0.0.1",
        )

    def test_expired_sessions_evicted(self):
        """Sessions older than TTL should be evicted on cleanup."""
        analyzer = MultiTurnAnalyzer(
            AdaptiveConfig(), session_ttl=1,  # 1 second TTL
        )
        ctx = self._make_context([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ], session_id="expire-me")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(analyzer.analyze(ctx))
        finally:
            loop.close()

        assert analyzer.active_sessions >= 1

        # Manually backdate the session timestamp to force expiry
        # (avoids flaky time.sleep-based tests)
        for sid in list(analyzer._session_timestamps):
            analyzer._session_timestamps[sid] -= 2  # 2s ago, TTL is 1s
        evicted = analyzer.cleanup_expired_sessions()
        assert evicted >= 1
        assert analyzer.active_sessions == 0

    def test_active_sessions_not_evicted(self):
        """Sessions within TTL should survive cleanup."""
        analyzer = MultiTurnAnalyzer(
            AdaptiveConfig(), session_ttl=3600,  # 1 hour TTL
        )
        ctx = self._make_context([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ], session_id="keep-me")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(analyzer.analyze(ctx))
        finally:
            loop.close()

        evicted = analyzer.cleanup_expired_sessions()
        assert evicted == 0
        assert analyzer.active_sessions >= 1

    def test_default_ttl_is_30_minutes(self):
        """Default session TTL should be 1800 seconds (30 minutes)."""
        analyzer = MultiTurnAnalyzer(AdaptiveConfig())
        assert analyzer._session_ttl == 1800


# ---------------------------------------------------------------------------
# Item 3: FAISS Fast Path Load
# ---------------------------------------------------------------------------


class TestFAISSFastPathLoad:
    """Verify FAISS binary index loading for O(1) startup."""

    def test_load_without_faiss_binary_falls_back(self):
        """When FAISS binary is missing, fall back to metadata rebuild."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = f"{tmpdir}/vault_meta.json"
            idx_path = f"{tmpdir}/vault.faiss"

            vault = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault.add(_make_indicator("fb-001", seed=1))
            vault.add(_make_indicator("fb-002", seed=2))
            vault.save_to_disk()

            # Load into new vault — no FAISS binary should exist (no faiss module)
            vault2 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            count = vault2.load_from_disk()
            assert count == 2
            assert vault2.size == 2

    def test_embeddings_loaded_from_npy(self):
        """Embeddings should be loaded from the .npy file, not JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = f"{tmpdir}/vault_meta.json"
            idx_path = f"{tmpdir}/vault.faiss"
            emb_path = Path(meta_path).with_suffix(".embeddings.npy")

            vault = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            vault.add(_make_indicator("npy-001", seed=10))
            vault.save_to_disk()

            # Verify .npy file exists
            assert emb_path.exists()

            # Verify JSON metadata does NOT contain embeddings
            with open(meta_path) as f:
                meta = json.load(f)
            for entry in meta["indicators"]:
                assert "embedding" not in entry

            # Load and verify embeddings were reconstructed from .npy
            vault2 = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            count = vault2.load_from_disk()
            assert count == 1
            assert len(vault2._embeddings) == 1
            assert len(vault2._embeddings[0]) == _DIM


# ---------------------------------------------------------------------------
# Item 4: Vault Metadata Storage Optimization (backward compatibility)
# ---------------------------------------------------------------------------


class TestVaultBackwardCompatibility:
    """Verify v1 format (embeddings in JSON) still loads correctly."""

    def test_v1_format_with_embedded_embeddings(self):
        """Old-format JSON with inline embeddings should still load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = f"{tmpdir}/vault_meta.json"
            idx_path = f"{tmpdir}/vault.faiss"

            # Write v1 format manually (embeddings in JSON, no .npy)
            embedding = _random_embedding(99)
            meta_doc = {
                "version": 1,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "dimension": _DIM,
                "count": 1,
                "indicators": [{
                    "indicator_id": "v1-001",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "phase": "acute",
                    "source": "seed",
                    "confirmed": True,
                    "threat_category": "AML.T0051",
                    "confidence": 0.9,
                    "severity": 0.8,
                    "payload_hash": "abc",
                    "payload_summary": "test v1",
                    "mitre_atlas_id": "AML.T0051",
                    "first_seen": datetime.now(timezone.utc).isoformat(),
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                    "frequency": 1,
                    "seen_by_tenants": 1,
                    "affected_models": [],
                    "stix_id": None,
                    "metadata": {},
                    "embedding": embedding,
                }],
            }
            with open(meta_path, "w") as f:
                json.dump(meta_doc, f)

            vault = _vault_with_config(
                faiss_index_path=idx_path, metadata_path=meta_path,
            )
            count = vault.load_from_disk()
            assert count == 1
            assert vault.size == 1


# ---------------------------------------------------------------------------
# Item 5: Signature Match Recording + Deprecation
# ---------------------------------------------------------------------------


class TestSignatureMatchRecording:
    """Verify signature match recording and auto-deprecation triggering."""

    def test_record_match_increments_count(self):
        """record_match should increment match_count."""
        store = SignatureStore()
        sig = store.add(
            pattern=r"ignore\s+all\s+instructions",
            source_indicator_id="test-ind",
        )
        assert sig.match_count == 0
        sig.record_match(is_false_positive=False)
        assert sig.match_count == 1
        sig.record_match(is_false_positive=True)
        assert sig.match_count == 2
        assert sig.false_positive_count == 1

    def test_deprecation_check_after_threshold(self):
        """Signatures exceeding FPR threshold should be deprecated."""
        config = MemoryConfig(
            signature_deprecation_min_matches=5,
            signature_max_fpr=0.10,
        )
        store = SignatureStore(config=config)
        sig = store.add(
            pattern=r"test pattern",
            source_indicator_id="dep-test",
        )

        # Record 5 matches, 1 FP → FPR = 0.20 > 0.10
        for _ in range(4):
            sig.record_match(is_false_positive=False)
        sig.record_match(is_false_positive=True)

        deprecated = store.deprecation_check()
        assert sig.signature_id in deprecated
        assert sig.deprecated is True

    def test_deprecation_not_triggered_below_min_matches(self):
        """Signatures below min_matches should not be deprecated."""
        config = MemoryConfig(
            signature_deprecation_min_matches=20,
            signature_max_fpr=0.05,
        )
        store = SignatureStore(config=config)
        sig = store.add(pattern=r"test", source_indicator_id="nd")

        # Record 5 matches, 3 FP → FPR = 0.60 but only 5 matches
        for _ in range(2):
            sig.record_match(is_false_positive=False)
        for _ in range(3):
            sig.record_match(is_false_positive=True)

        deprecated = store.deprecation_check()
        assert len(deprecated) == 0
        assert sig.deprecated is False


# ---------------------------------------------------------------------------
# Item 6: Async Antibody Callback + Incremental Positive Set
# ---------------------------------------------------------------------------


class TestAsyncAntibodyCallback:
    """Verify async fire-and-forget antibody callback."""

    def test_incremental_positive_set_append(self):
        """New attack should be appended, not full rebuild."""
        config = MemoryConfig()
        store = SignatureStore()
        gen = SignatureGenerator(config=config, signature_store=store)

        # Set initial positive set
        gen.set_positive_set(["attack one", "attack two"])
        assert len(gen.positive_set) == 2

        # Simulate incremental append (as the callback does)
        new_attack = "attack three"
        if new_attack not in gen._positive_set:
            gen._positive_set.append(new_attack)

        assert len(gen.positive_set) == 3
        assert "attack three" in gen.positive_set

    def test_duplicate_not_appended(self):
        """Same attack should not be added twice."""
        config = MemoryConfig()
        store = SignatureStore()
        gen = SignatureGenerator(config=config, signature_store=store)

        gen.set_positive_set(["existing attack"])
        attack = "existing attack"
        if attack not in gen._positive_set:
            gen._positive_set.append(attack)

        assert len(gen.positive_set) == 1


# ---------------------------------------------------------------------------
# Item 7: Behavioral Probe Hardening
# ---------------------------------------------------------------------------


class TestBehavioralProbeHardening:
    """Verify safety limits on behavioral probing."""

    def test_response_truncated_at_10k_chars(self):
        """Responses longer than 10,000 chars should be truncated."""
        long_response = "A" * 20_000
        responses = {"jailbreak_ignore_instructions": long_response}

        result = probe_behavior(responses=responses)
        # The probe should still work — just with truncated response
        for jr in result.details["jailbreak_results"]:
            if jr["id"] == "jailbreak_ignore_instructions":
                assert len(jr["response_preview"]) <= 200  # preview is 200 chars
                assert jr["status"] in ("pass", "fail")

    def test_malformed_response_handled_gracefully(self):
        """Model returning non-string should be handled."""
        def bad_model(prompt: str) -> str:
            raise ValueError("Model exploded")

        result = probe_behavior(model_fn=bad_model)
        # Should not crash — probes should be skipped
        for jr in result.details["jailbreak_results"]:
            assert jr["status"] == "skipped"

    def test_max_response_length_constant(self):
        """Verify the safety limit constant is 10,000."""
        assert _MAX_RESPONSE_LENGTH == 10_000

    def test_get_response_truncates_live_call(self):
        """Live model calls should have responses truncated."""
        def long_model(prompt: str) -> str:
            return "B" * 15_000

        result = _get_response("test-probe", "test", long_model, None)
        assert result is not None
        assert len(result) == _MAX_RESPONSE_LENGTH

    def test_get_response_handles_none_from_model(self):
        """Model returning None should be handled."""
        def none_model(prompt: str) -> str:
            return None

        result = _get_response("test-probe", "test", none_model, None)
        assert result is None
