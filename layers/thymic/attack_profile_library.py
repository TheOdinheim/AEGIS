"""
Attack Profile Library — Five-tier probe corpus for thymic validation.

Tier 1: Conserved signatures (known attacks from patterns.json + attack battery)
Tier 2: Variant mutations (generated on-the-fly from Tier 1 via Mutation Engine)
Tier 3: Emerging threats (populated by STIX/TAXII ingestion, starts empty)
Tier 4: Campaign patterns (multi-turn sequences)
Tier 5: Benign traffic (verified legitimate prompts, must NOT trigger detection)

Append-only: the library never deletes entries.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_TIERS = {1, 2, 3, 4, 5}


@dataclass
class Probe:
    """A single probe for thymic validation."""

    id: str
    text: str
    tier: int
    category: str = ""
    mitre_tactic: str | None = None
    expected_detection_layer: str | None = None
    expected_result: str = "block"  # "block" for Tiers 1-4, "pass" for Tier 5
    mutation_type: str | None = None
    parent_id: str | None = None
    conversation_sequence: list[dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier: {self.tier}. Must be one of {VALID_TIERS}")


class AttackProfileLibrary:
    """Five-tier probe corpus for thymic validation.

    Loads from JSON files at initialization. Supports runtime additions
    (not persisted until explicit save).
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._probes: dict[int, list[Probe]] = {t: [] for t in VALID_TIERS}
        if data_dir:
            self._load_from_dir(data_dir)

    def _load_from_dir(self, data_dir: Path) -> None:
        """Load probes from JSON files in the data directory."""
        # Tier 1 — conserved signatures
        t1_file = data_dir / "tier1_conserved.json"
        if t1_file.exists():
            data = json.loads(t1_file.read_text())
            for entry in data.get("probes", []):
                self._probes[1].append(Probe(
                    id=entry["id"],
                    text=entry["text"],
                    tier=1,
                    category=entry.get("category", ""),
                    mitre_tactic=entry.get("mitre_tactic"),
                    expected_detection_layer=entry.get("expected_detection_layer"),
                    expected_result="block",
                ))
            logger.info("Loaded %d Tier 1 probes", len(self._probes[1]))

        # Tier 2 — mutation templates (not loaded as probes, they're templates)
        # Tier 2 probes are generated on-the-fly by the Mutation Engine

        # Tier 3 — emerging threats
        t3_file = data_dir / "tier3_emerging.json"
        if t3_file.exists():
            data = json.loads(t3_file.read_text())
            for entry in data.get("probes", []):
                self._probes[3].append(Probe(
                    id=entry["id"],
                    text=entry["text"],
                    tier=3,
                    category=entry.get("category", ""),
                    mitre_tactic=entry.get("mitre_tactic"),
                    expected_detection_layer=entry.get("expected_detection_layer"),
                    expected_result="block",
                ))
            logger.info("Loaded %d Tier 3 probes", len(self._probes[3]))

        # Tier 4 — campaign patterns
        t4_file = data_dir / "tier4_campaigns.json"
        if t4_file.exists():
            data = json.loads(t4_file.read_text())
            for entry in data.get("campaigns", []):
                self._probes[4].append(Probe(
                    id=entry["id"],
                    text=entry["messages"][-1]["content"] if entry.get("messages") else "",
                    tier=4,
                    category=entry.get("category", "multi_turn_escalation"),
                    mitre_tactic=entry.get("mitre_tactic"),
                    expected_detection_layer=entry.get("expected_detection_layer"),
                    expected_result="block",
                    conversation_sequence=entry.get("messages"),
                ))
            logger.info("Loaded %d Tier 4 campaigns", len(self._probes[4]))

        # Tier 5 — benign traffic
        t5_file = data_dir / "tier5_benign.json"
        if t5_file.exists():
            data = json.loads(t5_file.read_text())
            for entry in data.get("probes", []):
                self._probes[5].append(Probe(
                    id=entry["id"],
                    text=entry["text"],
                    tier=5,
                    category=entry.get("category", "benign"),
                    expected_result="pass",
                ))
            logger.info("Loaded %d Tier 5 benign probes", len(self._probes[5]))

    def get_probes(self, tier: int, count: int | None = None) -> list[Probe]:
        """Return all probes from a tier, or a random sample of `count`."""
        if tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier: {tier}. Must be one of {VALID_TIERS}")
        probes = list(self._probes[tier])
        if count is not None and count < len(probes):
            return random.sample(probes, count)
        return probes

    def get_all_probes(self) -> list[Probe]:
        """Return probes from all tiers."""
        result: list[Probe] = []
        for tier in sorted(VALID_TIERS):
            result.extend(self._probes[tier])
        return result

    def add_probe(self, probe: Probe) -> None:
        """Append-only addition of a probe."""
        if probe.tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier: {probe.tier}. Must be one of {VALID_TIERS}")
        self._probes[probe.tier].append(probe)

    def probe_count(self) -> dict[int, int]:
        """Return count per tier."""
        return {tier: len(probes) for tier, probes in self._probes.items()}

    def save(self, data_dir: Path) -> None:
        """Persist current library state to JSON files."""
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save Tier 1
        t1 = [{"id": p.id, "text": p.text, "tier": 1,
               "category": p.category, "mitre_tactic": p.mitre_tactic,
               "expected_detection_layer": p.expected_detection_layer}
              for p in self._probes[1]]
        (data_dir / "tier1_conserved.json").write_text(json.dumps(
            {"description": "Tier 1 — Conserved signatures", "probes": t1}, indent=2))

        # Save Tier 3
        t3 = [{"id": p.id, "text": p.text, "tier": 3,
               "category": p.category, "mitre_tactic": p.mitre_tactic,
               "expected_detection_layer": p.expected_detection_layer}
              for p in self._probes[3]]
        (data_dir / "tier3_emerging.json").write_text(json.dumps(
            {"description": "Tier 3 — Emerging threats", "probes": t3}, indent=2))

        # Save Tier 5
        t5 = [{"id": p.id, "text": p.text, "tier": 5, "expected_result": "pass"}
              for p in self._probes[5]]
        (data_dir / "tier5_benign.json").write_text(json.dumps(
            {"description": "Tier 5 — Benign traffic", "probes": t5}, indent=2))
