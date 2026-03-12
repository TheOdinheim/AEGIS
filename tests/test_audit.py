"""
Audit Log Sink Tests

Validates tamper-evident SHA-256 chain hashing, append-only JSONL logging,
compliance mapping, and the /v1/audit/recent endpoint.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aegis.layers.audit import AuditLogger, AuditRecord, _GENESIS_HASH


class TestAuditRecord:
    def test_compute_hash_deterministic(self):
        """Same record content should produce same hash."""
        r1 = AuditRecord(
            timestamp="2025-01-01T00:00:00Z",
            request_id="req-1",
            tenant_id="t1",
            layer="innate",
            action="block",
        )
        r2 = AuditRecord(
            timestamp="2025-01-01T00:00:00Z",
            request_id="req-1",
            tenant_id="t1",
            layer="innate",
            action="block",
        )
        h1 = r1.compute_hash(_GENESIS_HASH)
        h2 = r2.compute_hash(_GENESIS_HASH)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self):
        r1 = AuditRecord(
            timestamp="2025-01-01T00:00:00Z",
            request_id="req-1",
            tenant_id="t1",
            layer="innate",
            action="block",
        )
        r2 = AuditRecord(
            timestamp="2025-01-01T00:00:00Z",
            request_id="req-2",
            tenant_id="t1",
            layer="innate",
            action="allow",
        )
        assert r1.compute_hash(_GENESIS_HASH) != r2.compute_hash(_GENESIS_HASH)

    def test_chain_hash_depends_on_previous(self):
        r = AuditRecord(
            timestamp="2025-01-01T00:00:00Z",
            request_id="req-1",
            tenant_id="t1",
            layer="innate",
            action="block",
        )
        h1 = r.compute_hash(_GENESIS_HASH)
        h2 = r.compute_hash("abc" * 21 + "a")  # different previous
        assert h1 != h2


class TestAuditLogger:
    @pytest.fixture
    def log_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_audit.jsonl"

    @pytest.fixture
    def logger(self, log_path: Path) -> AuditLogger:
        return AuditLogger(log_path=log_path)

    def test_log_creates_file(self, logger: AuditLogger, log_path: Path):
        logger.log("req-1", "innate", "block")
        assert log_path.exists()

    def test_log_records_are_jsonl(self, logger: AuditLogger, log_path: Path):
        logger.log("req-1", "innate", "block")
        logger.log("req-2", "policy", "allow")

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "record_hash" in record
            assert "previous_hash" in record

    def test_chain_hash_integrity(self, logger: AuditLogger, log_path: Path):
        """Chain hash should link records sequentially."""
        logger.log("req-1", "innate", "block")
        logger.log("req-2", "policy", "allow")
        logger.log("req-3", "output", "block")

        lines = log_path.read_text().strip().split("\n")
        records = [json.loads(line) for line in lines]

        # First record's previous_hash should be genesis
        assert records[0]["previous_hash"] == _GENESIS_HASH

        # Each subsequent record's previous_hash should be the prior record's hash
        for i in range(1, len(records)):
            assert records[i]["previous_hash"] == records[i - 1]["record_hash"]

    def test_verify_chain_valid(self, logger: AuditLogger):
        logger.log("req-1", "innate", "block")
        logger.log("req-2", "policy", "allow")
        logger.log("req-3", "output", "block")

        is_valid, count = logger.verify_chain()
        assert is_valid
        assert count == 3

    def test_verify_chain_detects_tampering(self, logger: AuditLogger, log_path: Path):
        """Modifying a record should break the chain."""
        logger.log("req-1", "innate", "block")
        logger.log("req-2", "policy", "allow")
        logger.log("req-3", "output", "block")

        # Tamper with the second record
        lines = log_path.read_text().strip().split("\n")
        record = json.loads(lines[1])
        record["action"] = "TAMPERED"
        lines[1] = json.dumps(record)
        log_path.write_text("\n".join(lines) + "\n")

        is_valid, count = logger.verify_chain()
        assert not is_valid
        assert count == 2  # Fails at record 2

    def test_compliance_mapping(self, logger: AuditLogger, log_path: Path):
        logger.log("req-1", "innate", "block")

        lines = log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["compliance"]["nist_ai_rmf"] == "GOVERN 1.4"
        assert record["compliance"]["iso_42001"] == "Clause 9"
        assert record["compliance"]["eu_ai_act"] == "Articles 12, 19"
        assert record["compliance"]["soc2"] == "CC7.2"

    def test_get_recent(self, logger: AuditLogger):
        for i in range(5):
            logger.log(f"req-{i}", "innate", "block")

        recent = logger.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[-1]["request_id"] == "req-4"

    def test_get_recent_respects_max(self, logger: AuditLogger):
        for i in range(150):
            logger.log(f"req-{i}", "innate", "block")

        recent = logger.get_recent(limit=200)
        assert len(recent) <= 100  # max_recent default

    def test_record_count(self, logger: AuditLogger):
        assert logger.record_count == 0
        logger.log("req-1", "innate", "block")
        logger.log("req-2", "policy", "allow")
        assert logger.record_count == 2

    def test_chain_recovery(self, log_path: Path):
        """New logger instance should recover chain from existing file."""
        logger1 = AuditLogger(log_path=log_path)
        logger1.log("req-1", "innate", "block")
        logger1.log("req-2", "policy", "allow")
        last_hash = logger1._last_hash

        # Create new logger from same file
        logger2 = AuditLogger(log_path=log_path)
        assert logger2._last_hash == last_hash

        # Continue the chain
        logger2.log("req-3", "output", "block")

        is_valid, count = logger2.verify_chain()
        assert is_valid
        assert count == 3

    def test_decision_context_logged(self, logger: AuditLogger, log_path: Path):
        logger.log(
            "req-1", "innate", "block",
            decision_context={"matched_patterns": ["PI-001", "PI-002"]},
            threat_categories=["prompt_injection"],
            confidence=0.95,
        )

        lines = log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["decision_context"]["matched_patterns"] == ["PI-001", "PI-002"]
        assert record["threat_categories"] == ["prompt_injection"]
        assert record["confidence"] == 0.95

    def test_empty_log_verifies(self, log_path: Path):
        logger = AuditLogger(log_path=log_path)
        is_valid, count = logger.verify_chain()
        assert is_valid
        assert count == 0
