"""
Tests for intent classifier — rule-based classification of feature vectors.
"""

from __future__ import annotations

import pytest

from aegis.layers.correlation.intent_features import IntentFeatureVector
from aegis.layers.correlation.intent_classifier import IntentClassifier
from aegis.layers.correlation.intent_alert import IntentCategory


def _make_features(**kwargs) -> IntentFeatureVector:
    """Build a feature vector with sensible defaults."""
    defaults = dict(
        target_entropy=0.5,
        action_type_ratios={},
        technique_diversity=0.1,
        temporal_regularity=1.0,
        progression_score=0.0,
        info_flow_density=0.0,
        agent_cardinality=1,
        stylistic_discontinuity=0.0,
        window_event_count=20,
    )
    defaults.update(kwargs)
    return IntentFeatureVector(**defaults)


class TestSystematicEnumeration:
    """Tests for systematic enumeration intent detection."""

    def test_detect_enumeration(self):
        """High entropy, high diversity, many agents, low regularity → SYSTEMATIC_ENUMERATION."""
        classifier = IntentClassifier()
        features = _make_features(
            target_entropy=3.5,
            technique_diversity=0.7,
            agent_cardinality=10,
            temporal_regularity=0.3,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.SYSTEMATIC_ENUMERATION
        assert result.confidence > 0.6


class TestBoundaryProbing:
    """Tests for boundary probing intent detection."""

    def test_detect_probing(self):
        """Oscillating progression, high auth ratio, moderate diversity → BOUNDARY_PROBING."""
        classifier = IntentClassifier()
        features = _make_features(
            progression_score=0.05,
            action_type_ratios={"auth_attempt": 0.35, "probe": 0.3, "api_call": 0.35},
            technique_diversity=0.35,
            temporal_regularity=0.5,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.BOUNDARY_PROBING
        assert result.confidence > 0.5


class TestVulnerabilityConfirmation:
    """Tests for vulnerability confirmation intent detection."""

    def test_detect_confirmation(self):
        """Low diversity, focused targets, few agents → VULNERABILITY_CONFIRMATION."""
        classifier = IntentClassifier()
        features = _make_features(
            technique_diversity=0.1,
            target_entropy=0.5,
            agent_cardinality=2,
            progression_score=0.5,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.VULNERABILITY_CONFIRMATION
        assert result.confidence > 0.5


class TestDataExfiltration:
    """Tests for data exfiltration staging intent detection."""

    def test_detect_exfiltration(self):
        """High DATA_ACCESS, high info flow, positive progression → DATA_EXFILTRATION_STAGING."""
        classifier = IntentClassifier()
        features = _make_features(
            action_type_ratios={"data_access": 0.6, "api_call": 0.4},
            info_flow_density=0.5,
            progression_score=0.4,
            # Set values that exclude vulnerability_confirmation
            technique_diversity=0.5,
            target_entropy=2.0,
            agent_cardinality=8,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.DATA_EXFILTRATION_STAGING
        assert result.confidence > 0.5


class TestPrivilegeEscalation:
    """Tests for privilege escalation probing intent detection."""

    def test_detect_privesc(self):
        """High auth ratio, high progression, moderate entropy → PRIVILEGE_ESCALATION_PROBING."""
        classifier = IntentClassifier()
        features = _make_features(
            action_type_ratios={"auth_attempt": 0.4, "probe": 0.3, "tool_invocation": 0.3},
            progression_score=0.6,
            target_entropy=2.0,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.PRIVILEGE_ESCALATION_PROBING
        assert result.confidence > 0.5


class TestBenignActivity:
    """Tests for benign activity detection."""

    def test_detect_benign(self):
        """High temporal variance, low info flow, no progression → BENIGN_ACTIVITY."""
        classifier = IntentClassifier()
        features = _make_features(
            temporal_regularity=2.0,
            info_flow_density=0.01,
            progression_score=0.05,
            # Exclude vulnerability_confirmation match
            technique_diversity=0.5,
            target_entropy=2.0,
            agent_cardinality=10,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.BENIGN_ACTIVITY

    def test_default_benign(self):
        """No category matches above threshold → default BENIGN_ACTIVITY."""
        classifier = IntentClassifier()
        # Set values that exclude all attack categories
        features = _make_features(
            technique_diversity=0.5,
            target_entropy=2.0,
            agent_cardinality=8,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.BENIGN_ACTIVITY


class TestAlloyAmplification:
    """Tests for model alloy (stylistic discontinuity) amplification."""

    def test_amplifies_attack_intent(self):
        """Enumeration at 0.6 + discontinuity 0.5 → amplified to 0.75."""
        classifier = IntentClassifier()
        features = _make_features(
            target_entropy=3.5,
            technique_diversity=0.7,
            agent_cardinality=10,
            temporal_regularity=0.3,
            stylistic_discontinuity=0.5,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.SYSTEMATIC_ENUMERATION
        assert result.stylistic_discontinuity_amplified is True
        # Confidence should be amplified
        assert result.confidence > 0.6

    def test_does_not_amplify_benign(self):
        """Benign with high discontinuity → no amplification."""
        classifier = IntentClassifier()
        features = _make_features(
            temporal_regularity=2.0,
            info_flow_density=0.01,
            progression_score=0.05,
            stylistic_discontinuity=0.8,
            technique_diversity=0.5,
            target_entropy=2.0,
            agent_cardinality=10,
        )
        result = classifier.classify(features)
        assert result.category == IntentCategory.BENIGN_ACTIVITY
        assert result.stylistic_discontinuity_amplified is False


class TestSequenceClassification:
    """Tests for multi-window sequence classification."""

    def test_recency_weighting(self):
        """First window benign, last two enumeration → sequence = SYSTEMATIC_ENUMERATION."""
        classifier = IntentClassifier()

        benign = _make_features(
            temporal_regularity=2.0,
            info_flow_density=0.01,
            progression_score=0.05,
        )
        enum_features = _make_features(
            target_entropy=3.5,
            technique_diversity=0.7,
            agent_cardinality=10,
            temporal_regularity=0.3,
        )

        result = classifier.classify_sequence([benign, enum_features, enum_features])
        assert result.category == IntentCategory.SYSTEMATIC_ENUMERATION

    def test_empty_sequence(self):
        """Empty sequence → BENIGN_ACTIVITY."""
        classifier = IntentClassifier()
        result = classifier.classify_sequence([])
        assert result.category == IntentCategory.BENIGN_ACTIVITY


class TestAmbiguousInput:
    """Tests for ambiguous classification."""

    def test_highest_confidence_wins(self):
        """Features partially matching two categories → highest wins with runner-up."""
        classifier = IntentClassifier()
        # Features that match both priv-esc (auth>0.25, prog>0.4, entropy>1.5)
        # and boundary probing (prog in [-0.3,0.3], auth>0.15, div in [0.2,0.5])
        # Trick: use a progression_score that's JUST inside probing range AND above privesc
        # Actually these rules are mutually exclusive on progression_score.
        # Instead: match both data exfiltration and priv-esc.
        # data_exfil: data_access>0.4, info_flow>0.3, prog>0.2
        # priv_esc: auth_attempt>0.25, prog>0.4, entropy>1.5
        features = _make_features(
            action_type_ratios={"auth_attempt": 0.4, "data_access": 0.6},
            progression_score=0.7,
            target_entropy=3.0,
            info_flow_density=0.5,
            technique_diversity=0.5,
            agent_cardinality=8,
        )
        result = classifier.classify(features)
        assert result.category in (
            IntentCategory.PRIVILEGE_ESCALATION_PROBING,
            IntentCategory.DATA_EXFILTRATION_STAGING,
        )
        # Runner-up should be in evidence since both matched
        assert "runner_up_category" in result.evidence
        assert result.evidence["runner_up_confidence"] > 0
