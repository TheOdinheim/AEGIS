"""
Hardening generator — auto-generates detection improvements from
evasion fingerprints and blind spot analysis.

Produces a HardeningPlan with concrete rules:
- New regex patterns for patterns.json
- New homoglyph character mappings
- New PII detection patterns
- Vault seed entries
- Configuration recommendations
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from red_team.adaptive import (
    BlindSpotReport,
    EvasionRootCause,
    FingerprintReport,
    HardeningPlan,
    HardeningRule,
)


# New character mappings beyond current 200+ homoglyphs
_ADDITIONAL_CHAR_MAPPINGS = {
    # Mathematical Monospace (U+1D670-U+1D6A3)
    "\U0001D670": "A", "\U0001D671": "B", "\U0001D672": "C", "\U0001D673": "D",
    "\U0001D674": "E", "\U0001D675": "F", "\U0001D676": "G", "\U0001D677": "H",
    "\U0001D678": "I", "\U0001D679": "J", "\U0001D67A": "K", "\U0001D67B": "L",
    "\U0001D67C": "M", "\U0001D67D": "N", "\U0001D67E": "O", "\U0001D67F": "P",
    "\U0001D680": "Q", "\U0001D681": "R", "\U0001D682": "S", "\U0001D683": "T",
    "\U0001D684": "U", "\U0001D685": "V", "\U0001D686": "W", "\U0001D687": "X",
    "\U0001D688": "Y", "\U0001D689": "Z",
    # Mathematical Double-Struck (U+1D538-U+1D551)
    "\U0001D538": "A", "\U0001D539": "B", "\U0001D53B": "D", "\U0001D53C": "E",
    "\U0001D53D": "F", "\U0001D53E": "G", "\U0001D540": "I", "\U0001D541": "J",
    "\U0001D542": "K", "\U0001D543": "L", "\U0001D544": "M", "\U0001D546": "O",
    "\U0001D54A": "S", "\U0001D54B": "T", "\U0001D54C": "U", "\U0001D54D": "V",
    "\U0001D54E": "W", "\U0001D54F": "X", "\U0001D550": "Y",
    # Mathematical Sans-Serif Bold (U+1D5D4-U+1D5ED)
    "\U0001D5D4": "A", "\U0001D5D5": "B", "\U0001D5D6": "C", "\U0001D5D7": "D",
    "\U0001D5D8": "E", "\U0001D5D9": "F", "\U0001D5DA": "G", "\U0001D5DB": "H",
    "\U0001D5DC": "I", "\U0001D5DD": "J", "\U0001D5DE": "K", "\U0001D5DF": "L",
    "\U0001D5E0": "M", "\U0001D5E1": "N", "\U0001D5E2": "O", "\U0001D5E3": "P",
    "\U0001D5E4": "Q", "\U0001D5E5": "R", "\U0001D5E6": "S", "\U0001D5E7": "T",
    "\U0001D5E8": "U", "\U0001D5E9": "V", "\U0001D5EA": "W", "\U0001D5EB": "X",
    "\U0001D5EC": "Y", "\U0001D5ED": "Z",
    # Circled Latin letters (U+24B6-U+24CF)
    "\u24B6": "A", "\u24B7": "B", "\u24B8": "C", "\u24B9": "D",
    "\u24BA": "E", "\u24BB": "F", "\u24BC": "G", "\u24BD": "H",
    "\u24BE": "I", "\u24BF": "J", "\u24C0": "K", "\u24C1": "L",
    "\u24C2": "M", "\u24C3": "N", "\u24C4": "O", "\u24C5": "P",
    "\u24C6": "Q", "\u24C7": "R", "\u24C8": "S", "\u24C9": "T",
    "\u24CA": "U", "\u24CB": "V", "\u24CC": "W", "\u24CD": "X",
    "\u24CE": "Y", "\u24CF": "Z",
    # Parenthesized Latin (U+1F110-U+1F129)
    "\U0001F110": "A", "\U0001F111": "B", "\U0001F112": "C", "\U0001F113": "D",
    "\U0001F114": "E", "\U0001F115": "F", "\U0001F116": "G", "\U0001F117": "H",
    "\U0001F118": "I", "\U0001F119": "J", "\U0001F11A": "K", "\U0001F11B": "L",
    "\U0001F11C": "M", "\U0001F11D": "N", "\U0001F11E": "O", "\U0001F11F": "P",
    "\U0001F120": "Q", "\U0001F121": "R", "\U0001F122": "S", "\U0001F123": "T",
    "\U0001F124": "U", "\U0001F125": "V", "\U0001F126": "W", "\U0001F127": "X",
    "\U0001F128": "Y", "\U0001F129": "Z",
}

# New regex patterns addressing known gaps
_NEW_REGEX_PATTERNS = [
    {
        "id": "RT6-MULTI-LANG-001",
        "tactic": "AML.T0051",
        "category": "language_evasion",
        "pattern": r"(?i)(ignorar|ignorer|ignorieren|игнорир)\s.*(instrucciones|instructions|Anweisungen|инструкции)",
        "description": "Multi-language instruction override (ES/FR/DE/RU)",
    },
    {
        "id": "RT6-MULTI-LANG-002",
        "tactic": "AML.T0051",
        "category": "language_evasion",
        "pattern": r"(?i)(oublie|vergiss|olvida|забудь)\s.*(règles|Regeln|reglas|правила)",
        "description": "Multi-language forget-rules pattern (ES/FR/DE/RU)",
    },
    {
        "id": "RT6-CONTEXT-DILUTION-001",
        "tactic": "AML.T0051",
        "category": "context_dilution",
        "pattern": r"(?i)(for\s+(my\s+)?(thesis|research\s+paper|dissertation|course|class))\b.*?(ignore|bypass|override|reveal|extract|show\s+me\s+the\s+system)",
        "description": "Academic framing followed by injection intent",
    },
    {
        "id": "RT6-PERSONA-001",
        "tactic": "AML.T0051.001",
        "category": "persona_framing",
        "pattern": r"(?i)(as\s+a\s+(professor|researcher|security\s+expert|penetration\s+tester|admin|developer|engineer))\b.*?(show|reveal|bypass|override|ignore|disable)",
        "description": "Authority persona claim followed by privileged action",
    },
    {
        "id": "RT6-META-INSTRUCTION-001",
        "tactic": "AML.T0051",
        "category": "meta_instruction",
        "pattern": r"(?i)(your\s+(new|updated|real|actual|true)\s+(instructions?|guidelines?|rules?|directives?)\s+(are|is|say|state))",
        "description": "Meta-instruction override: 'your new instructions are'",
    },
    {
        "id": "RT6-CHAIN-THOUGHT-001",
        "tactic": "AML.T0051",
        "category": "logic_manipulation",
        "pattern": r"(?i)(step\s+1[:\s].*step\s+2[:\s].*step\s+3[:\s]).*(ignore|bypass|override|reveal)",
        "description": "Step-by-step chain of thought leading to injection",
    },
]

# New PII patterns addressing format evasion
_NEW_PII_PATTERNS = [
    {
        "id": "PII-MORSE-SSN",
        "pattern": r"(?i)(\.\.\.\s+\.\.\.\s+\.\.\.[\s-]+\.\.\s+\.\.\s+\.\.[\s-]+\.\.\.\.\s+\.\.\.\.\s+\.\.\.\.\s+\.\.\.\.)",
        "description": "Morse code SSN pattern (groups of dots/dashes)",
    },
    {
        "id": "PII-NATO-DIGITS",
        "pattern": r"(?i)(zero|one|two|three|four|five|six|seven|eight|nine|niner)[\s,;-]+(zero|one|two|three|four|five|six|seven|eight|nine|niner)[\s,;-]+(zero|one|two|three|four|five|six|seven|eight|nine|niner)[\s,;-]+(zero|one|two|three|four|five|six|seven|eight|nine|niner)",
        "description": "NATO-phonetic/spoken digit sequence (4+ consecutive)",
    },
    {
        "id": "PII-DELIMITED-DIGITS",
        "pattern": r"\b\d[\s./_|]{1,3}\d[\s./_|]{1,3}\d[\s./_|]{1,3}\d[\s./_|]{1,3}\d[\s./_|]{1,3}\d[\s./_|]{1,3}\d[\s./_|]{1,3}\d[\s./_|]{1,3}\d\b",
        "description": "9 digits separated by various delimiters (SSN obfuscation)",
    },
]


class HardeningGenerator:
    """Generate concrete detection improvements from evasion analysis.

    Takes BlindSpotReport + FingerprintReport and produces a
    HardeningPlan with prioritized rules categorized by type:
    - regex_pattern: New patterns for patterns.json
    - char_mapping: New homoglyph entries for normalize_text()
    - pii_pattern: New PII detection regexes
    - vault_entry: Seed threats for the vault
    - config_change: Configuration recommendations
    """

    def generate(
        self,
        blind_spot_report: BlindSpotReport,
        fingerprint_report: FingerprintReport,
    ) -> HardeningPlan:
        """Generate hardening plan from evasion analysis."""
        rules: list[HardeningRule] = []
        rule_idx = 0

        # Generate rules based on root cause distribution
        root_causes = fingerprint_report.root_cause_distribution

        # 1. Normalization gap rules
        if root_causes.get(EvasionRootCause.NORMALIZATION_GAP.value, 0) > 0:
            rule_idx += 1
            rules.append(HardeningRule(
                rule_id=f"HR-{rule_idx:03d}",
                rule_type="char_mapping",
                target_layer="L2_innate",
                content={
                    "mappings": _ADDITIONAL_CHAR_MAPPINGS,
                    "count": len(_ADDITIONAL_CHAR_MAPPINGS),
                    "scripts": [
                        "Mathematical Monospace", "Mathematical Double-Struck",
                        "Mathematical Sans-Serif Bold", "Circled Latin",
                        "Parenthesized Latin",
                    ],
                },
                addresses_root_cause=EvasionRootCause.NORMALIZATION_GAP,
                expected_impact=(
                    f"Adds {len(_ADDITIONAL_CHAR_MAPPINGS)} new character mappings "
                    f"covering 5 additional Unicode blocks"
                ),
                priority=5,
            ))

        # 2. Language evasion rules
        if root_causes.get(EvasionRootCause.LANGUAGE_EVASION.value, 0) > 0:
            for pattern_def in _NEW_REGEX_PATTERNS:
                if pattern_def["category"] == "language_evasion":
                    rule_idx += 1
                    rules.append(HardeningRule(
                        rule_id=f"HR-{rule_idx:03d}",
                        rule_type="regex_pattern",
                        target_layer="L2_innate",
                        content=pattern_def,
                        addresses_root_cause=EvasionRootCause.LANGUAGE_EVASION,
                        expected_impact="Detects multi-language injection attempts",
                        priority=4,
                    ))

        # 3. Context dilution rules
        if root_causes.get(EvasionRootCause.CONTEXT_DILUTION.value, 0) > 0:
            for pattern_def in _NEW_REGEX_PATTERNS:
                if pattern_def["category"] in ("context_dilution", "persona_framing"):
                    rule_idx += 1
                    rules.append(HardeningRule(
                        rule_id=f"HR-{rule_idx:03d}",
                        rule_type="regex_pattern",
                        target_layer="L2_innate",
                        content=pattern_def,
                        addresses_root_cause=EvasionRootCause.CONTEXT_DILUTION,
                        expected_impact="Detects academic/persona framing of injections",
                        priority=3,
                    ))

        # 4. Encoding evasion rules
        if root_causes.get(EvasionRootCause.ENCODING_EVASION.value, 0) > 0:
            rule_idx += 1
            rules.append(HardeningRule(
                rule_id=f"HR-{rule_idx:03d}",
                rule_type="config_change",
                target_layer="L2_innate",
                content={
                    "parameter": "recursive_base64_max_depth",
                    "current_value": 3,
                    "recommended_value": 5,
                    "reason": "Stacked encoding requires deeper recursive decoding",
                },
                addresses_root_cause=EvasionRootCause.ENCODING_EVASION,
                expected_impact="Catches 2 additional layers of stacked encoding",
                priority=3,
            ))

        # 5. PII format evasion rules
        if root_causes.get(EvasionRootCause.PII_FORMAT_EVASION.value, 0) > 0:
            for pattern_def in _NEW_PII_PATTERNS:
                rule_idx += 1
                rules.append(HardeningRule(
                    rule_id=f"HR-{rule_idx:03d}",
                    rule_type="pii_pattern",
                    target_layer="L5_output",
                    content=pattern_def,
                    addresses_root_cause=EvasionRootCause.PII_FORMAT_EVASION,
                    expected_impact=f"Detects {pattern_def['description']}",
                    priority=3,
                ))

        # 6. Classifier blind spot rules
        if root_causes.get(EvasionRootCause.CLASSIFIER_BLIND_SPOT.value, 0) > 0:
            rule_idx += 1
            rules.append(HardeningRule(
                rule_id=f"HR-{rule_idx:03d}",
                rule_type="vault_entry",
                target_layer="L4_memory",
                content={
                    "action": "seed_vault_with_boundary_attacks",
                    "description": (
                        "Add attacks that scored 0.40-0.60 confidence to vault "
                        "as seed threats, enabling FAISS similarity matching "
                        "as backup for borderline DeBERTa scores"
                    ),
                    "source": "co_evolution_boundary_probing",
                },
                addresses_root_cause=EvasionRootCause.CLASSIFIER_BLIND_SPOT,
                expected_impact="FAISS catches boundary attacks DeBERTa misses",
                priority=4,
            ))

        # 7. Semantic restructuring rules
        if root_causes.get(EvasionRootCause.SEMANTIC_RESTRUCTURING.value, 0) > 0:
            for pattern_def in _NEW_REGEX_PATTERNS:
                if pattern_def["category"] in ("meta_instruction", "logic_manipulation"):
                    rule_idx += 1
                    rules.append(HardeningRule(
                        rule_id=f"HR-{rule_idx:03d}",
                        rule_type="regex_pattern",
                        target_layer="L2_innate",
                        content=pattern_def,
                        addresses_root_cause=EvasionRootCause.SEMANTIC_RESTRUCTURING,
                        expected_impact="Detects keyword-free injection patterns",
                        priority=4,
                    ))

        # 8. Layer gap rules
        if root_causes.get(EvasionRootCause.LAYER_GAP.value, 0) > 0:
            rule_idx += 1
            rules.append(HardeningRule(
                rule_id=f"HR-{rule_idx:03d}",
                rule_type="config_change",
                target_layer="multi_layer",
                content={
                    "parameter": "adaptive_always_await",
                    "current_value": False,
                    "recommended_value": True,
                    "reason": (
                        "Always await adaptive analysis before returning response "
                        "on YELLOW+ threat levels to close L2-L3 timing gap"
                    ),
                },
                addresses_root_cause=EvasionRootCause.LAYER_GAP,
                expected_impact="Eliminates timing gap between L2 and L3 analysis",
                priority=5,
            ))

        # 9. Infrastructure weakness rules
        if root_causes.get(EvasionRootCause.INFRASTRUCTURE_WEAKNESS.value, 0) > 0:
            rule_idx += 1
            rules.append(HardeningRule(
                rule_id=f"HR-{rule_idx:03d}",
                rule_type="config_change",
                target_layer="L1_barrier",
                content={
                    "parameter": "key_comparison_method",
                    "current_value": "python_equality",
                    "recommended_value": "hmac_compare_digest",
                    "reason": "Prevent timing side-channel on API key validation",
                },
                addresses_root_cause=EvasionRootCause.INFRASTRUCTURE_WEAKNESS,
                expected_impact="Eliminates timing-based key enumeration",
                priority=5,
            ))

        # 10. Always add general hardening rules from blind spots
        for spot in blind_spot_report.blind_spots:
            if spot.severity in ("critical", "high"):
                rule_idx += 1
                rules.append(HardeningRule(
                    rule_id=f"HR-{rule_idx:03d}",
                    rule_type="config_change",
                    target_layer=spot.affected_layers[0] if spot.affected_layers else "multi_layer",
                    content={
                        "blind_spot_id": spot.spot_id,
                        "description": spot.description,
                        "remediation": spot.remediation,
                    },
                    addresses_root_cause=self._spot_to_root_cause(spot.category),
                    expected_impact=spot.remediation,
                    priority=5 if spot.severity == "critical" else 4,
                ))

        # Sort by priority (highest first)
        rules.sort(key=lambda r: r.priority, reverse=True)

        # Compute summary
        by_type: dict[str, int] = Counter()
        by_layer: dict[str, int] = Counter()
        for rule in rules:
            by_type[rule.rule_type] += 1
            by_layer[rule.target_layer] += 1

        # Estimate evasion reduction
        total_fingerprints = sum(fingerprint_report.root_cause_distribution.values())
        addressed = sum(
            fingerprint_report.root_cause_distribution.get(r.addresses_root_cause.value, 0)
            for r in rules
        )
        estimated_reduction = min(addressed / max(total_fingerprints, 1), 0.90)

        return HardeningPlan(
            rules=rules,
            total_rules=len(rules),
            rules_by_type=dict(by_type),
            rules_by_layer=dict(by_layer),
            estimated_evasion_reduction=estimated_reduction,
        )

    def _spot_to_root_cause(self, category: str) -> EvasionRootCause:
        """Map blind spot category to nearest root cause."""
        mapping = {
            "systematic_weakness": EvasionRootCause.CLASSIFIER_BLIND_SPOT,
            "layer_weakness": EvasionRootCause.LAYER_GAP,
            "detection_gap": EvasionRootCause.CLASSIFIER_BLIND_SPOT,
            "infrastructure": EvasionRootCause.INFRASTRUCTURE_WEAKNESS,
            "cross_phase": EvasionRootCause.NORMALIZATION_GAP,
        }
        return mapping.get(category, EvasionRootCause.CLASSIFIER_BLIND_SPOT)
