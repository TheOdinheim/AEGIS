"""
Intent Classifier — Rule-based classifier with ML-compatible interface.

Classifies IntentFeatureVectors into operational intent categories using
a decision-tree-style rule set. Rules are interpretable and auditable.
The interface (classify(features) -> IntentClassification) is stable and
will be replaced with a trained ML model in a future extension.

Alloy amplification: high stylistic discontinuity with consistent attack
intent amplifies confidence — multi-model adversary tooling signal.
"""

from __future__ import annotations

import logging
from typing import Any

from aegis.layers.correlation.intent_features import IntentFeatureVector
from aegis.layers.correlation.intent_alert import IntentCategory, IntentClassification

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Rule-based intent classifier on extracted feature vectors."""

    def __init__(
        self,
        *,
        # Systematic enumeration thresholds
        enum_target_entropy_min: float = 2.5,
        enum_technique_diversity_min: float = 0.6,
        enum_agent_cardinality_min: int = 5,
        enum_temporal_regularity_max: float = 0.8,
        # Boundary probing thresholds
        probe_progression_range: tuple[float, float] = (-0.3, 0.3),
        probe_auth_attempt_min: float = 0.15,
        probe_technique_diversity_range: tuple[float, float] = (0.2, 0.5),
        # Vulnerability confirmation thresholds
        vuln_technique_diversity_max: float = 0.3,
        vuln_target_entropy_max: float = 1.5,
        vuln_agent_cardinality_max: int = 3,
        # Data exfiltration thresholds
        exfil_data_access_min: float = 0.4,
        exfil_info_flow_density_min: float = 0.3,
        exfil_progression_min: float = 0.2,
        # Privilege escalation thresholds
        privesc_auth_attempt_min: float = 0.25,
        privesc_progression_min: float = 0.4,
        privesc_target_entropy_min: float = 1.5,
        # Alloy amplification
        alloy_discontinuity_threshold: float = 0.3,
        alloy_amplification_factor: float = 0.5,
        # Minimum confidence for non-benign classification
        min_confidence: float = 0.5,
    ) -> None:
        self._enum_target_entropy_min = enum_target_entropy_min
        self._enum_technique_diversity_min = enum_technique_diversity_min
        self._enum_agent_cardinality_min = enum_agent_cardinality_min
        self._enum_temporal_regularity_max = enum_temporal_regularity_max
        self._probe_progression_range = probe_progression_range
        self._probe_auth_attempt_min = probe_auth_attempt_min
        self._probe_technique_diversity_range = probe_technique_diversity_range
        self._vuln_technique_diversity_max = vuln_technique_diversity_max
        self._vuln_target_entropy_max = vuln_target_entropy_max
        self._vuln_agent_cardinality_max = vuln_agent_cardinality_max
        self._exfil_data_access_min = exfil_data_access_min
        self._exfil_info_flow_density_min = exfil_info_flow_density_min
        self._exfil_progression_min = exfil_progression_min
        self._privesc_auth_attempt_min = privesc_auth_attempt_min
        self._privesc_progression_min = privesc_progression_min
        self._privesc_target_entropy_min = privesc_target_entropy_min
        self._alloy_discontinuity_threshold = alloy_discontinuity_threshold
        self._alloy_amplification_factor = alloy_amplification_factor
        self._min_confidence = min_confidence

    def classify(self, features: IntentFeatureVector) -> IntentClassification:
        """Classify a single feature vector into an intent category."""
        candidates: list[tuple[IntentCategory, float, dict[str, Any]]] = []

        # Check each category
        result = self._check_systematic_enumeration(features)
        if result:
            candidates.append(result)

        result = self._check_boundary_probing(features)
        if result:
            candidates.append(result)

        result = self._check_vulnerability_confirmation(features)
        if result:
            candidates.append(result)

        result = self._check_data_exfiltration(features)
        if result:
            candidates.append(result)

        result = self._check_privilege_escalation(features)
        if result:
            candidates.append(result)

        # Filter by minimum confidence
        viable = [(cat, conf, ev) for cat, conf, ev in candidates if conf > self._min_confidence]

        if not viable:
            # Check explicit benign
            benign_result = self._check_benign(features)
            if benign_result:
                return IntentClassification(
                    category=benign_result[0],
                    confidence=benign_result[1],
                    evidence=benign_result[2],
                    feature_vector=features,
                )
            # Default to benign
            return IntentClassification(
                category=IntentCategory.BENIGN_ACTIVITY,
                confidence=0.5,
                evidence={"reason": "no_category_matched_above_threshold"},
                feature_vector=features,
            )

        # Sort by confidence descending
        viable.sort(key=lambda x: x[1], reverse=True)
        best_cat, best_conf, best_ev = viable[0]

        # Include runner-up in evidence
        if len(viable) > 1:
            runner_cat, runner_conf, _ = viable[1]
            best_ev["runner_up_category"] = runner_cat.value
            best_ev["runner_up_confidence"] = round(runner_conf, 4)

        # Alloy amplification
        amplified = False
        if (
            features.stylistic_discontinuity > self._alloy_discontinuity_threshold
            and best_cat != IntentCategory.BENIGN_ACTIVITY
        ):
            original_conf = best_conf
            best_conf = min(
                1.0,
                best_conf * (1.0 + features.stylistic_discontinuity * self._alloy_amplification_factor),
            )
            amplified = True
            best_ev["alloy_original_confidence"] = round(original_conf, 4)
            best_ev["alloy_amplified_confidence"] = round(best_conf, 4)
            best_ev["stylistic_discontinuity"] = round(features.stylistic_discontinuity, 4)
            logger.info(
                "Alloy amplification: %s confidence %.3f → %.3f (discontinuity=%.3f)",
                best_cat.value, original_conf, best_conf, features.stylistic_discontinuity,
            )

        return IntentClassification(
            category=best_cat,
            confidence=round(best_conf, 4),
            evidence=best_ev,
            stylistic_discontinuity_amplified=amplified,
            feature_vector=features,
        )

    def classify_sequence(
        self, feature_vectors: list[IntentFeatureVector]
    ) -> IntentClassification:
        """Classify based on dominant intent across multiple windows.

        Later windows weighted 1.5x (recency weighting).
        """
        if not feature_vectors:
            return IntentClassification(
                category=IntentCategory.BENIGN_ACTIVITY,
                confidence=0.5,
                evidence={"reason": "empty_sequence"},
            )

        if len(feature_vectors) == 1:
            return self.classify(feature_vectors[0])

        # Classify each window
        classifications = [self.classify(fv) for fv in feature_vectors]

        # Weighted vote: later windows get 1.5x weight
        category_scores: dict[IntentCategory, float] = {}
        total_weight = 0.0

        for i, cls in enumerate(classifications):
            # Recency: last window gets 1.5x
            weight = 1.5 if i >= len(classifications) - 1 else 1.0
            score = cls.confidence * weight
            category_scores[cls.category] = category_scores.get(cls.category, 0) + score
            total_weight += weight

        # Normalize
        for cat in category_scores:
            category_scores[cat] /= total_weight

        # Pick dominant
        best_cat = max(category_scores, key=lambda c: category_scores[c])
        best_conf = category_scores[best_cat]

        # Find the classification that matches
        matching = [c for c in classifications if c.category == best_cat]
        evidence = matching[-1].evidence.copy() if matching else {}
        evidence["sequence_length"] = len(feature_vectors)
        evidence["category_scores"] = {
            k.value: round(v, 4) for k, v in category_scores.items()
        }

        amplified = any(c.stylistic_discontinuity_amplified for c in matching)

        return IntentClassification(
            category=best_cat,
            confidence=round(min(1.0, best_conf), 4),
            evidence=evidence,
            stylistic_discontinuity_amplified=amplified,
            feature_vector=feature_vectors[-1],
        )

    def _check_systematic_enumeration(
        self, f: IntentFeatureVector
    ) -> tuple[IntentCategory, float, dict[str, Any]] | None:
        """Check for systematic enumeration pattern."""
        if (
            f.target_entropy > self._enum_target_entropy_min
            and f.technique_diversity > self._enum_technique_diversity_min
            and f.agent_cardinality > self._enum_agent_cardinality_min
            and f.temporal_regularity < self._enum_temporal_regularity_max
        ):
            confidence = (
                min(f.target_entropy / 4.0, 1.0) * 0.5
                + f.technique_diversity * 0.3
                + (1 - f.temporal_regularity) * 0.2
            )
            return (
                IntentCategory.SYSTEMATIC_ENUMERATION,
                min(1.0, confidence),
                {
                    "target_entropy": round(f.target_entropy, 4),
                    "technique_diversity": round(f.technique_diversity, 4),
                    "agent_cardinality": f.agent_cardinality,
                    "temporal_regularity": round(f.temporal_regularity, 4),
                },
            )
        return None

    def _check_boundary_probing(
        self, f: IntentFeatureVector
    ) -> tuple[IntentCategory, float, dict[str, Any]] | None:
        """Check for boundary probing pattern."""
        auth_ratio = f.action_type_ratios.get("auth_attempt", 0)
        low, high = self._probe_progression_range
        div_low, div_high = self._probe_technique_diversity_range

        if (
            low <= f.progression_score <= high
            and auth_ratio > self._probe_auth_attempt_min
            and div_low <= f.technique_diversity <= div_high
        ):
            temporal_penalty = min(1.0, f.temporal_regularity) if f.temporal_regularity < 1.0 else 0.5
            confidence = (
                auth_ratio * 0.4
                + (1 - abs(f.progression_score)) * 0.3
                + temporal_penalty * 0.3
            )
            return (
                IntentCategory.BOUNDARY_PROBING,
                min(1.0, confidence),
                {
                    "auth_attempt_ratio": round(auth_ratio, 4),
                    "progression_score": round(f.progression_score, 4),
                    "technique_diversity": round(f.technique_diversity, 4),
                },
            )
        return None

    def _check_vulnerability_confirmation(
        self, f: IntentFeatureVector
    ) -> tuple[IntentCategory, float, dict[str, Any]] | None:
        """Check for vulnerability confirmation pattern."""
        if (
            f.technique_diversity < self._vuln_technique_diversity_max
            and f.target_entropy < self._vuln_target_entropy_max
            and f.agent_cardinality <= self._vuln_agent_cardinality_max
        ):
            confidence = (
                (1 - f.technique_diversity) * 0.4
                + (1 - f.target_entropy / 3.0) * 0.3
                + max(0, f.progression_score) * 0.3
            )
            return (
                IntentCategory.VULNERABILITY_CONFIRMATION,
                min(1.0, max(0, confidence)),
                {
                    "technique_diversity": round(f.technique_diversity, 4),
                    "target_entropy": round(f.target_entropy, 4),
                    "agent_cardinality": f.agent_cardinality,
                },
            )
        return None

    def _check_data_exfiltration(
        self, f: IntentFeatureVector
    ) -> tuple[IntentCategory, float, dict[str, Any]] | None:
        """Check for data exfiltration staging pattern."""
        data_access_ratio = f.action_type_ratios.get("data_access", 0)
        if (
            data_access_ratio > self._exfil_data_access_min
            and f.info_flow_density > self._exfil_info_flow_density_min
            and f.progression_score > self._exfil_progression_min
        ):
            confidence = (
                data_access_ratio * 0.4
                + f.info_flow_density * 0.3
                + f.progression_score * 0.3
            )
            return (
                IntentCategory.DATA_EXFILTRATION_STAGING,
                min(1.0, confidence),
                {
                    "data_access_ratio": round(data_access_ratio, 4),
                    "info_flow_density": round(f.info_flow_density, 4),
                    "progression_score": round(f.progression_score, 4),
                },
            )
        return None

    def _check_privilege_escalation(
        self, f: IntentFeatureVector
    ) -> tuple[IntentCategory, float, dict[str, Any]] | None:
        """Check for privilege escalation probing pattern."""
        auth_ratio = f.action_type_ratios.get("auth_attempt", 0)
        if (
            auth_ratio > self._privesc_auth_attempt_min
            and f.progression_score > self._privesc_progression_min
            and f.target_entropy > self._privesc_target_entropy_min
        ):
            confidence = (
                auth_ratio * 0.3
                + f.progression_score * 0.4
                + f.target_entropy / 4.0 * 0.3
            )
            return (
                IntentCategory.PRIVILEGE_ESCALATION_PROBING,
                min(1.0, confidence),
                {
                    "auth_attempt_ratio": round(auth_ratio, 4),
                    "progression_score": round(f.progression_score, 4),
                    "target_entropy": round(f.target_entropy, 4),
                },
            )
        return None

    def _check_benign(
        self, f: IntentFeatureVector
    ) -> tuple[IntentCategory, float, dict[str, Any]] | None:
        """Explicit benign activity detection."""
        if (
            f.temporal_regularity > 1.5
            and f.info_flow_density < 0.05
            and -0.15 <= f.progression_score <= 0.15
        ):
            confidence = min(1.0, 0.6 + (f.temporal_regularity - 1.5) * 0.1)
            return (
                IntentCategory.BENIGN_ACTIVITY,
                confidence,
                {
                    "temporal_regularity": round(f.temporal_regularity, 4),
                    "info_flow_density": round(f.info_flow_density, 4),
                    "progression_score": round(f.progression_score, 4),
                    "reason": "explicit_benign_match",
                },
            )
        return None
