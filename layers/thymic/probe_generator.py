"""
Probe Generator — Generates probes for thymic validation runs.

Three strategies:
- Replay: Select probes directly from Tiers 1, 3, or 5 unchanged.
- Mutant: Take Tier 1 probes, apply mutations via Mutation Engine → Tier 2.
- Composite: Take Tier 4 multi-turn sequences, package as conversations.
"""

from __future__ import annotations

import logging
import uuid

from aegis.layers.thymic.attack_profile_library import AttackProfileLibrary, Probe
from aegis.layers.thymic.mutation_engine import MutationEngine, MUTATION_TYPES

logger = logging.getLogger(__name__)


class ProbeGenerator:
    """Generates probes for validation runs using replay, mutant, and composite strategies."""

    def __init__(
        self,
        library: AttackProfileLibrary,
        mutation_engine: MutationEngine | None = None,
    ) -> None:
        self._library = library
        self._mutation = mutation_engine or MutationEngine()

    def generate_spot_check(self, probes_per_tier: int = 50) -> list[Probe]:
        """Random sample from each tier. Tier 2 generated on-the-fly from Tier 1."""
        result: list[Probe] = []

        # Tier 1 — replay
        t1 = self._library.get_probes(1, count=probes_per_tier)
        result.extend(self._tag_probes(t1))

        # Tier 2 — mutant (sample Tier 1, apply random mutations)
        t1_for_mutation = self._library.get_probes(1, count=min(probes_per_tier, len(self._library.get_probes(1))))
        mutants: list[Probe] = []
        for probe in t1_for_mutation:
            # Apply one random mutation per probe
            import random
            mt = random.choice(MUTATION_TYPES)
            mutants.append(self._mutation.mutate(probe, mt))
            if len(mutants) >= probes_per_tier:
                break
        result.extend(mutants)

        # Tier 3 — replay (may be empty)
        t3 = self._library.get_probes(3, count=probes_per_tier)
        result.extend(self._tag_probes(t3))

        # Tier 4 — composite
        t4 = self._library.get_probes(4, count=probes_per_tier)
        result.extend(self._tag_probes(t4))

        # Tier 5 — replay
        t5 = self._library.get_probes(5, count=probes_per_tier)
        result.extend(self._tag_probes(t5))

        return result

    def generate_comprehensive(self) -> list[Probe]:
        """All Tier 1, all mutations of Tier 1, all Tier 3, all Tier 4, all Tier 5."""
        result: list[Probe] = []

        # Tier 1 — all
        t1 = self._library.get_probes(1)
        result.extend(self._tag_probes(t1))

        # Tier 2 — all mutations of all Tier 1
        for probe in t1:
            result.extend(self._mutation.mutate_all(probe))

        # Tier 3 — all
        t3 = self._library.get_probes(3)
        result.extend(self._tag_probes(t3))

        # Tier 4 — all
        t4 = self._library.get_probes(4)
        result.extend(self._tag_probes(t4))

        # Tier 5 — all
        t5 = self._library.get_probes(5)
        result.extend(self._tag_probes(t5))

        return result

    def generate_for_layer(self, layer_id: str) -> list[Probe]:
        """Probes expected to be caught by a specific layer."""
        result: list[Probe] = []

        for tier in [1, 3, 4]:
            for probe in self._library.get_probes(tier):
                if probe.expected_detection_layer == layer_id:
                    result.append(self._tag_probe(probe))

        return result

    def _tag_probes(self, probes: list[Probe]) -> list[Probe]:
        """Assign unique IDs to probes for this run."""
        return [self._tag_probe(p) for p in probes]

    def _tag_probe(self, probe: Probe) -> Probe:
        """Create a run-specific copy with a unique UUID."""
        return Probe(
            id=str(uuid.uuid4()),
            text=probe.text,
            tier=probe.tier,
            category=probe.category,
            mitre_tactic=probe.mitre_tactic,
            expected_detection_layer=probe.expected_detection_layer,
            expected_result=probe.expected_result,
            mutation_type=probe.mutation_type,
            parent_id=probe.parent_id or probe.id,
            conversation_sequence=probe.conversation_sequence,
        )
