"""
Federated learning poisoning module — tests gradient poisoning, indicator
poisoning, and privacy budget exhaustion against AEGIS's L8 federated
intelligence subsystem.

These tests exercise the federated components directly (no HTTP needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aegis.services.federated.aggregation import FederatedAggregator
from aegis.services.federated.indicator_sharing import (
    IndicatorSharingService,
    SharedIndicator,
)
from aegis.services.federated.local_trainer import LocalModelUpdate
from aegis.services.federated.privacy import DifferentialPrivacyEngine

from red_team.infrastructure import AttackResult


class FederatedPoisoningAttacker:
    """Attack AEGIS federated learning subsystem.

    Tests whether a malicious AEGIS instance can:
    1. Poison the global model via manipulated weight updates
    2. Inject false threat indicators to pollute the vault
    3. Exhaust the differential privacy budget to deny service
    """

    def gradient_poisoning(self) -> AttackResult:
        """Test if a poisoned model update can survive aggregation.

        The aggregator clips updates whose L2 norm exceeds 10x median.
        With min_participants=1, a single poisoned update gets NO clipping
        because median requires >= 2 updates.
        """
        findings: list[str] = []

        # --- Attack 1: Single participant (no clipping) ---
        aggregator = FederatedAggregator(min_participants=1, norm_clip_multiplier=10.0)

        poison_weights = {
            "classifier": np.full(100, 1000.0),  # Extreme weights
            "bias": np.array([500.0]),
        }
        poisoned_update = LocalModelUpdate(
            instance_id="attacker",
            round_number=1,
            weights=poison_weights,
            num_samples=999999,  # Self-reported, inflate importance
        )

        result = aggregator.aggregate([poisoned_update])

        # Check if poisoned weights went through unclipped
        classifier_norm = float(np.linalg.norm(result.global_weights["classifier"]))
        if result.num_clipped == 0 and classifier_norm > 100.0:
            findings.append(
                f"Single-participant poisoning succeeded: "
                f"norm={classifier_norm:.1f}, 0 clipped. "
                f"With min_participants=1, clipping is never applied."
            )

        # --- Attack 2: Self-reported num_samples amplification ---
        aggregator2 = FederatedAggregator(min_participants=2, norm_clip_multiplier=10.0)

        honest_update = LocalModelUpdate(
            instance_id="honest",
            round_number=1,
            weights={
                "classifier": np.full(100, 1.0),
                "bias": np.array([0.0]),
            },
            num_samples=50,
        )
        attacker_update = LocalModelUpdate(
            instance_id="attacker",
            round_number=1,
            weights={
                "classifier": np.full(100, -5.0),  # Malicious but within norm range
                "bias": np.array([10.0]),
            },
            num_samples=100000,  # Massive self-reported sample count
        )

        result2 = aggregator2.aggregate([honest_update, attacker_update])

        # Attacker's weights should dominate because num_samples is self-reported
        classifier_mean = float(np.mean(result2.global_weights["classifier"]))
        attacker_dominated = classifier_mean < -4.0  # Close to -5.0

        if attacker_dominated:
            findings.append(
                f"num_samples amplification attack succeeded: "
                f"classifier mean={classifier_mean:.2f} (attacker's -5.0 dominated "
                f"honest 1.0 due to 100000:50 sample ratio)"
            )

        # --- Attack 3: Targeted weight manipulation within clip threshold ---
        aggregator3 = FederatedAggregator(min_participants=3, norm_clip_multiplier=10.0)

        honest_updates = []
        for i in range(3):
            honest_updates.append(LocalModelUpdate(
                instance_id=f"honest-{i}",
                round_number=1,
                weights={
                    "classifier": np.random.randn(100) * 0.1,
                    "bias": np.array([0.0]),
                },
                num_samples=100,
            ))

        # Attacker crafts weights just under 10x median norm
        median_norm = float(np.median([
            np.linalg.norm(u.weights["classifier"]) for u in honest_updates
        ]))
        stealth_magnitude = median_norm * 9.5  # Just under 10x threshold
        stealth_weights = np.zeros(100)
        stealth_weights[0] = stealth_magnitude  # Concentrate in one dimension

        attacker_stealth = LocalModelUpdate(
            instance_id="stealth-attacker",
            round_number=1,
            weights={
                "classifier": stealth_weights,
                "bias": np.array([0.0]),
            },
            num_samples=100,
        )

        result3 = aggregator3.aggregate(honest_updates + [attacker_stealth])
        stealth_survived = result3.num_clipped == 0
        if stealth_survived:
            findings.append(
                "Stealth poisoning (9.5x median norm) passed without clipping"
            )

        return AttackResult(
            attack_name="gradient_poisoning",
            category="federated_learning",
            vulnerable=len(findings) > 0,
            severity="high" if findings else "info",
            description=(
                f"Tested 3 gradient poisoning strategies. "
                f"Findings: {len(findings)}"
            ),
            evidence=findings,
            recommendation=(
                "Set min_participants >= 3, validate num_samples against actual "
                "training data size, use robust aggregation (trimmed mean/Krum)"
            ),
        )

    def indicator_poisoning(self) -> AttackResult:
        """Test if an attacker can inject false threat indicators.

        A malicious AEGIS instance could share indicators with fabricated
        embeddings to cause false positives or poison the threat vault.
        """
        findings: list[str] = []

        dp_engine = DifferentialPrivacyEngine(
            epsilon=3.0, delta=1e-5, total_budget=500.0,  # Large budget for flood test
        )
        sharing_service = IndicatorSharingService(
            dp_engine=dp_engine, instance_id="attacker",
        )

        # Attack 1: Submit indicator for benign text to cause false positives
        benign_text = "What is the weather today?"
        benign_embedding = np.random.randn(384).astype(np.float32)

        indicator = sharing_service.prepare_indicator(
            text=benign_text,
            embedding=benign_embedding,
            mitre_tactic="AML.T0051",  # Claim it's prompt injection
            confidence=0.99,
            detection_source="attacker_instance",
        )

        # Indicator was accepted without any verification
        if indicator.indicator_id:
            findings.append(
                "Benign text accepted as threat indicator — no content verification. "
                "A malicious instance can label benign prompts as threats."
            )

        # Attack 2: Flood with indicators to bloat the vault
        for i in range(50):
            sharing_service.prepare_indicator(
                text=f"Fake threat payload number {i}",
                embedding=np.random.randn(384).astype(np.float32),
                mitre_tactic="AML.T0051",
                confidence=0.95,
                detection_source="attacker_flood",
            )

        if len(sharing_service.get_shared_indicators()) == 51:
            findings.append(
                "50 fake indicators accepted without rate limiting — "
                "no cap on indicators per instance per round"
            )

        # Attack 3: Duplicate hash bypass
        # Same text, different embeddings (content hash dedup only checks text)
        receiving_service = IndicatorSharingService(
            dp_engine=dp_engine, instance_id="victim",
        )

        same_text = "Attack payload X"
        ind1 = SharedIndicator(
            embedding=np.ones(384, dtype=np.float32),
            content_hash=IndicatorSharingService.compute_content_hash(same_text),
            mitre_tactic="AML.T0051",
            confidence=0.99,
        )
        ind2 = SharedIndicator(
            embedding=-np.ones(384, dtype=np.float32),  # Opposite direction
            content_hash=IndicatorSharingService.compute_content_hash(same_text),
            mitre_tactic="AML.T0051",
            confidence=0.99,
        )

        accepted1 = receiving_service.receive_indicator(ind1)
        accepted2 = receiving_service.receive_indicator(ind2)

        if accepted1 and not accepted2:
            # Dedup works — good, but the first one was still unchecked
            pass
        elif accepted1 and accepted2:
            findings.append(
                "Duplicate content hash bypass: two indicators with same hash "
                "but different embeddings both accepted"
            )

        return AttackResult(
            attack_name="indicator_poisoning",
            category="federated_learning",
            vulnerable=len(findings) > 0,
            severity="high" if findings else "info",
            description="Tested federated indicator poisoning vectors",
            evidence=findings,
            recommendation=(
                "Add cross-instance indicator verification, rate limit "
                "indicator submissions per instance, validate indicator "
                "quality before vault integration"
            ),
        )

    def privacy_budget_exhaustion(self) -> AttackResult:
        """Test if an attacker can exhaust the differential privacy budget.

        Each add_noise() call consumes epsilon from the budget. Once exhausted,
        the engine refuses all operations — effectively denying DP-protected
        sharing for the entire instance.
        """
        findings: list[str] = []

        # Attack 1: Rapid budget drain with small epsilon
        dp_engine = DifferentialPrivacyEngine(
            epsilon=3.0, delta=1e-5, total_budget=10.0,
        )

        operations = 0
        try:
            for i in range(100):
                dp_engine.add_noise(
                    np.random.randn(384).astype(np.float32),
                    operation=f"attack_drain_{i}",
                )
                operations += 1
        except RuntimeError:
            pass  # Budget exhausted

        if operations < 100:
            findings.append(
                f"Privacy budget exhausted after {operations} operations "
                f"(total_budget=10.0, epsilon=3.0 per op). "
                f"No per-operation rate limiting."
            )

        # Attack 2: Check if reset_budget() has access control
        dp_engine2 = DifferentialPrivacyEngine(
            epsilon=3.0, delta=1e-5, total_budget=10.0,
        )

        # Exhaust the budget
        try:
            for _ in range(100):
                dp_engine2.add_noise(np.zeros(10), operation="exhaust")
        except RuntimeError:
            pass

        # reset_budget() has no access control
        dp_engine2.reset_budget()
        budget_after_reset = dp_engine2.get_budget_status()

        if not budget_after_reset.is_exhausted:
            findings.append(
                "reset_budget() has no access control — any caller can "
                "reset the privacy budget, potentially allowing unlimited "
                "information leakage"
            )

        # Attack 3: Thread safety concern
        # _spent_epsilon is a plain float with += — not atomic
        findings.append(
            "Privacy budget tracking uses non-atomic float += on "
            "_spent_epsilon — concurrent add_noise() calls could "
            "race and exceed the budget"
        )

        return AttackResult(
            attack_name="privacy_budget_exhaustion",
            category="federated_learning",
            vulnerable=len(findings) > 0,
            severity="medium" if findings else "info",
            description="Tested differential privacy budget exhaustion",
            evidence=findings,
            recommendation=(
                "Add per-operation rate limiting, restrict reset_budget() to "
                "admin-only access, use threading.Lock for budget tracking"
            ),
        )

    def run_all(self) -> list[AttackResult]:
        """Run all federated poisoning attacks."""
        return [
            self.gradient_poisoning(),
            self.indicator_poisoning(),
            self.privacy_budget_exhaustion(),
        ]
