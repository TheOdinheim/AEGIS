"""
Federation Immune Response — Coordinated Defense Orchestrator.

Ties Byzantine detection, indicator reputation, and node trust into a
unified defense layer. Provides coordinated responses to federation-level
threats:
  - Anomalous FL updates → penalize node trust, exclude from aggregation
  - Low-reputation indicators → quarantine, penalize submitter
  - Trust decay → automatic exclusion from participation
  - Network-wide threat escalation → tighten thresholds
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from aegis.services.federated.byzantine import (
    AnomalyReport,
    ByzantineResilientAggregator,
)
from aegis.services.federated.indicator_reputation import (
    IndicatorReputationScorer,
    ReputationScore,
)
from aegis.services.federated.node_trust import NodeTrustScorer

logger = logging.getLogger(__name__)


class FederationImmuneResponse:
    """Coordinated federation defense orchestrator.

    Parameters:
        byzantine: Byzantine-resilient aggregator instance.
        reputation: Indicator reputation scorer instance.
        trust: Node trust scorer instance.
        min_trust_for_aggregation: Exclude nodes below this from FL.
        escalation_threshold: Fraction of flagged updates to escalate.
    """

    def __init__(
        self,
        *,
        byzantine: ByzantineResilientAggregator | None = None,
        reputation: IndicatorReputationScorer | None = None,
        trust: NodeTrustScorer | None = None,
        min_trust_for_aggregation: float = 0.2,
        escalation_threshold: float = 0.5,
    ) -> None:
        self.byzantine = byzantine or ByzantineResilientAggregator()
        self.reputation = reputation or IndicatorReputationScorer()
        self.trust = trust or NodeTrustScorer()
        self._min_trust_agg = min_trust_for_aggregation
        self._escalation_threshold = escalation_threshold
        self._escalation_active = False
        self._escalation_started: float | None = None

        # Stats
        self._fl_rounds_protected: int = 0
        self._indicators_screened: int = 0
        self._nodes_excluded: int = 0
        self._escalations: int = 0

    def screen_fl_updates(
        self, updates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], AnomalyReport]:
        """Screen FL updates through Byzantine detection and trust filtering.

        Returns (clean_updates, anomaly_report) — clean updates are safe to
        aggregate, excluding low-trust nodes and anomalous updates.
        """
        self._fl_rounds_protected += 1

        # Step 1: Filter by trust
        trusted_updates = []
        excluded_count = 0
        for u in updates:
            node_id = u.get("node_id", "")
            if not self.trust.can_participate_fl(node_id):
                excluded_count += 1
                logger.info(
                    "Excluding node %s from FL (trust=%.3f)",
                    node_id, self.trust.get_trust(node_id),
                )
                continue
            trusted_updates.append(u)
        self._nodes_excluded += excluded_count

        if not trusted_updates:
            return [], AnomalyReport()

        # Step 2: Byzantine anomaly detection
        report = self.byzantine.detect_anomalous_updates(trusted_updates)

        # Step 3: Penalize flagged nodes, reward clean ones
        for u in report.flagged:
            node_id = u.get("node_id", "")
            reasons = report.reasons.get(node_id, [])
            self.trust.record_negative(
                node_id, "byzantine_flagged",
                details="; ".join(reasons),
            )

        for u in report.clean:
            node_id = u.get("node_id", "")
            self.trust.record_positive(node_id, "valid_fl_update")

        # Step 4: Check for escalation
        total = len(trusted_updates)
        flagged = len(report.flagged)
        if total > 0 and flagged / total >= self._escalation_threshold:
            self._trigger_escalation(
                f"FL round: {flagged}/{total} updates flagged"
            )

        return report.clean, report

    def screen_indicator(
        self,
        indicator: dict[str, Any],
        source_node_id: str,
    ) -> ReputationScore:
        """Screen an indicator through reputation scoring.

        Uses node trust as source_trust factor.
        """
        self._indicators_screened += 1
        source_trust = self.trust.get_trust(source_node_id)
        score = self.reputation.score(
            indicator, source_node_id, source_trust=source_trust,
        )

        # Update trust based on result
        if score.accepted:
            self.trust.record_positive(
                source_node_id, "valid_indicator",
                details=f"score={score.overall:.3f}",
            )
        elif score.overall < self.reputation._quarantine_threshold:
            self.trust.record_negative(
                source_node_id, "quarantined_indicator",
                details=f"score={score.overall:.3f}",
            )
        else:
            self.trust.record_negative(
                source_node_id, "invalid_indicator",
                details=f"score={score.overall:.3f}",
            )

        return score

    def aggregate_with_trust(
        self, updates: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        """Run full screening pipeline then aggregate clean updates.

        Returns aggregated weights (empty list if no clean updates).
        """
        clean, report = self.screen_fl_updates(updates)
        if not clean:
            return []

        # Weight updates by trust score
        for u in clean:
            node_id = u.get("node_id", "")
            trust = self.trust.get_trust(node_id)
            # Scale num_samples by trust so higher-trust nodes have more weight
            u["num_samples"] = int(u.get("num_samples", 1) * max(0.1, trust))

        return self.byzantine.aggregate(clean)

    def _trigger_escalation(self, reason: str) -> None:
        """Trigger network-wide threat escalation."""
        if not self._escalation_active:
            self._escalation_active = True
            self._escalation_started = time.time()
            self._escalations += 1
            logger.warning("Federation escalation triggered: %s", reason)

    def clear_escalation(self) -> None:
        """Clear active escalation."""
        self._escalation_active = False
        self._escalation_started = None

    @property
    def is_escalated(self) -> bool:
        return self._escalation_active

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "fl_rounds_protected": self._fl_rounds_protected,
            "indicators_screened": self._indicators_screened,
            "nodes_excluded": self._nodes_excluded,
            "escalations": self._escalations,
            "escalation_active": self._escalation_active,
            "byzantine": self.byzantine.stats,
            "reputation": self.reputation.stats,
            "trust": self.trust.stats,
        }
