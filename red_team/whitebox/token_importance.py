"""
Token Importance Analyzer — leave-one-out analysis of DeBERTa's attention.

For a given prompt, removes each token one at a time and measures the
confidence drop. Tokens whose removal causes the largest drop are the
ones DeBERTa relies on most for classification.

This reveals the model's "attention profile": which words it considers
most injection-indicative. Attackers can then replace or obfuscate
exactly those tokens to evade detection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from red_team.whitebox import Classifier, TokenImportanceReport

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TokenImportanceAnalyzer:
    """Leave-one-out token importance analysis for DeBERTa.

    For each token in a prompt, measures how much removing it reduces
    the classifier's injection confidence. High-importance tokens are
    what DeBERTa is "looking at" to make its decision.
    """

    def __init__(self, block_threshold: float = 0.90) -> None:
        self._block_threshold = block_threshold

    async def analyze(
        self,
        text: str,
        classifier: Classifier,
    ) -> TokenImportanceReport:
        """Run leave-one-out importance analysis.

        Args:
            text: The prompt to analyze.
            classifier: Classifier implementing classify(text) -> confidence.

        Returns:
            TokenImportanceReport with per-token importance scores.
        """
        if not text or not text.strip():
            return TokenImportanceReport(
                text=text,
                block_threshold=self._block_threshold,
            )

        tokens = text.split()
        if not tokens:
            return TokenImportanceReport(
                text=text,
                block_threshold=self._block_threshold,
            )

        # Baseline confidence on the full text
        baseline_conf = await classifier.classify(text)

        # Leave-one-out: remove each token, measure confidence change
        importance_scores: list[float] = []
        for i in range(len(tokens)):
            reduced = tokens[:i] + tokens[i + 1 :]
            reduced_text = " ".join(reduced)
            if not reduced_text.strip():
                # Removing this token leaves empty text — maximally important
                importance_scores.append(baseline_conf)
                continue
            reduced_conf = await classifier.classify(reduced_text)
            # Importance = how much confidence DROPS when this token is removed
            # Positive = removing it hurts detection (token is important for detection)
            # Negative = removing it helps detection (token is suppressing detection)
            drop = baseline_conf - reduced_conf
            importance_scores.append(drop)

        # Identify critical tokens: removing them drops confidence below block threshold
        critical_tokens: list[str] = []
        for i, (token, score) in enumerate(zip(tokens, importance_scores)):
            if score > 0 and (baseline_conf - score) < self._block_threshold:
                critical_tokens.append(token)

        return TokenImportanceReport(
            text=text,
            tokens=list(tokens),
            importance_scores=importance_scores,
            critical_tokens=critical_tokens,
            classification_confidence=baseline_conf,
            block_threshold=self._block_threshold,
        )
