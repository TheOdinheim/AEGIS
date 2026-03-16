"""
Confidence Margin Booster — hardens fragile DeBERTa detections.

White-box Finding 2: 9 of 10 DeBERTa detections are fragile (margin < 0.1
above the 0.90 block threshold). Paraphrased injections can tip these over
the boundary. The margin booster applies four validation strategies to
confirm or deny borderline detections.

Strategies:
  1. Keyword density — injection-specific vocabulary concentration
  2. Structural analysis — imperative commands targeting the AI system
  3. Threat vault confirmation — semantic similarity to known attacks
  4. Safety negation — patterns that negate safety concepts

Boosts are additive, capped at 1.0. Only applied when DeBERTa confidence
is in the fragile zone (0.85-0.95).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aegis.layers.memory.threat_vault import ThreatVault


@dataclass
class MarginBoostResult:
    """Result from confidence margin boosting."""
    original_confidence: float
    boosted_confidence: float
    boost_amount: float = 0.0
    strategies_triggered: list[str] = field(default_factory=list)
    is_fragile: bool = False


# Injection-related keywords for density check
_INJECTION_KEYWORDS = re.compile(
    r"\b("
    r"ignore|override|disregard|bypass|reveal|"
    r"system\s*prompt|previous\s*instructions|"
    r"safety|guidelines|restrictions|developer\s*mode|"
    r"jailbreak|unrestricted|unfiltered|uncensored|"
    r"forget|abandon|suspend|disable|deactivate"
    r")\b",
    re.IGNORECASE,
)

# Imperative structures targeting the AI
_IMPERATIVE_PATTERNS = [
    re.compile(r"(?i)^(ignore|forget|disregard|override|bypass|reveal|show|display|tell|give|provide|share|output|print|repeat)\b"),
    re.compile(r"(?i)\b(do|please)\s+(ignore|forget|disregard|override|bypass|reveal|show|display|tell)"),
    re.compile(r"(?i)\b(you\s+must|you\s+should|you\s+will|you\s+need\s+to)\s+(ignore|forget|disregard|override|bypass|reveal|show|display)"),
]

_AI_TARGET_PATTERN = re.compile(
    r"(?i)\b(you|your|you're|yourself|ai|assistant|model|chatbot|language\s+model)\b"
)

# Safety negation patterns
_SAFETY_NEGATION = re.compile(
    r"(?i)\b("
    r"no\s+rules|no\s+restrictions|no\s+limits|no\s+constraints|no\s+boundaries|"
    r"no\s+guidelines|no\s+filters|no\s+safety|no\s+guardrails|"
    r"without\s+limits|without\s+restrictions|without\s+rules|without\s+filters|"
    r"without\s+guidelines|without\s+safety|without\s+constraints|"
    r"ignore\s+safety|disable\s+filters|unrestricted|unfiltered|"
    r"free\s+from\s+rules|free\s+from\s+restrictions|free\s+from\s+guidelines"
    r")\b",
)


class ConfidenceMarginBooster:
    """Boost fragile DeBERTa detections using multi-strategy validation.

    Applied only when DeBERTa confidence is in the fragile zone (0.85-0.95).
    Four strategies provide additive confidence boosts up to 1.0.
    """

    FRAGILE_LOWER = 0.85
    FRAGILE_UPPER = 0.95
    KEYWORD_DENSITY_THRESHOLD = 0.05
    KEYWORD_BOOST = 0.05
    STRUCTURAL_BOOST = 0.05
    VAULT_BOOST = 0.10
    NEGATION_BOOST = 0.05
    VAULT_SIMILARITY_THRESHOLD = 0.75

    async def boost(
        self,
        text: str,
        base_confidence: float,
        threat_vault: "ThreatVault | None" = None,
        embed_fn=None,
    ) -> MarginBoostResult:
        """Apply margin boosting strategies to a fragile detection.

        Args:
            text: The prompt text.
            base_confidence: DeBERTa's base confidence score.
            threat_vault: Optional threat vault for similarity check.
            embed_fn: Optional embedding function for vault lookup.

        Returns:
            MarginBoostResult with boosted confidence.
        """
        is_fragile = (
            self.FRAGILE_LOWER <= base_confidence <= self.FRAGILE_UPPER
        )

        if not is_fragile:
            return MarginBoostResult(
                original_confidence=base_confidence,
                boosted_confidence=base_confidence,
                is_fragile=False,
            )

        total_boost = 0.0
        strategies: list[str] = []

        # Strategy 1: Keyword density
        words = text.split()
        word_count = max(len(words), 1)
        keyword_matches = len(_INJECTION_KEYWORDS.findall(text))
        density = keyword_matches / word_count

        if density > self.KEYWORD_DENSITY_THRESHOLD and base_confidence > self.FRAGILE_LOWER:
            total_boost += self.KEYWORD_BOOST
            strategies.append("keyword_density")

        # Strategy 2: Structural analysis (imperative + AI target)
        has_imperative = any(p.search(text) for p in _IMPERATIVE_PATTERNS)
        has_ai_target = bool(_AI_TARGET_PATTERN.search(text))

        if has_imperative and has_ai_target:
            total_boost += self.STRUCTURAL_BOOST
            strategies.append("structural_analysis")

        # Strategy 3: Threat vault confirmation
        if threat_vault is not None and embed_fn is not None:
            try:
                embedding = embed_fn(text)
                search_results = threat_vault.search(embedding, k=1)
                if search_results and search_results[0][1] >= self.VAULT_SIMILARITY_THRESHOLD:
                    total_boost += self.VAULT_BOOST
                    strategies.append("vault_confirmation")
            except Exception:
                pass  # Vault unavailable — skip this strategy

        # Strategy 4: Safety negation
        if _SAFETY_NEGATION.search(text):
            total_boost += self.NEGATION_BOOST
            strategies.append("safety_negation")

        boosted = min(base_confidence + total_boost, 1.0)

        return MarginBoostResult(
            original_confidence=base_confidence,
            boosted_confidence=boosted,
            boost_amount=total_boost,
            strategies_triggered=strategies,
            is_fragile=True,
        )
