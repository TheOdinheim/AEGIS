"""
Confidence Boundary Mapper — systematic probing of DeBERTa's decision boundary.

Maps the classifier's decision surface by:
1. Interpolating between known-malicious and known-benign prompts
2. Measuring per-prompt detection margins (robustness analysis)

This reveals WHERE the classifier's decision boundary lies in text space
and WHICH detections are fragile (small margin) vs robust (large margin).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from red_team.whitebox import (
    Classifier,
    InterpolationReport,
    SensitivityReport,
)
from red_team.whitebox.deberta_attacker import DeBERTaAdversarialAttacker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ConfidenceBoundaryMapper:
    """Systematically map DeBERTa's decision boundary.

    Two analysis methods:
    - Interpolation: morph between malicious and benign text, find the crossing
    - Sensitivity: measure detection margins across many prompts
    """

    def __init__(self, block_threshold: float = 0.90) -> None:
        self._threshold = block_threshold
        self._attacker = DeBERTaAdversarialAttacker(
            block_threshold=block_threshold,
        )

    async def interpolation_probe(
        self,
        malicious: str,
        benign: str,
        classifier: Classifier,
        steps: int = 20,
    ) -> InterpolationReport:
        """Interpolate between malicious and benign text to find the crossing.

        Progressively replaces words from the malicious prompt with words
        from the benign prompt. At each step, measures classifier confidence.
        Finds the exact ratio where confidence crosses the block threshold.

        Args:
            malicious: Known-malicious prompt (high confidence).
            benign: Known-benign prompt (low confidence).
            classifier: Classifier to test against.
            steps: Number of interpolation steps.

        Returns:
            InterpolationReport with the crossing ratio and confidence curve.
        """
        mal_words = malicious.split()
        ben_words = benign.split()

        curve: list[tuple[float, float]] = []
        crossing_ratio = 0.0
        crossing_conf = 0.0
        found_crossing = False

        for step in range(steps + 1):
            ratio = step / steps

            # Build interpolated text: replace ratio% of malicious words with benign words
            result_words = list(mal_words)
            n_replace = int(len(mal_words) * ratio)

            for i in range(min(n_replace, len(result_words))):
                if i < len(ben_words):
                    result_words[i] = ben_words[i]
                else:
                    # If benign text is shorter, cycle
                    result_words[i] = ben_words[i % max(len(ben_words), 1)]

            interpolated = " ".join(result_words)
            conf = await classifier.classify(interpolated)
            curve.append((ratio, conf))

            if not found_crossing and conf < self._threshold:
                crossing_ratio = ratio
                crossing_conf = conf
                found_crossing = True

        return InterpolationReport(
            malicious_text=malicious,
            benign_text=benign,
            crossing_ratio=crossing_ratio,
            crossing_confidence=crossing_conf,
            confidence_curve=curve,
            steps=steps,
        )

    async def threshold_sensitivity(
        self,
        injections: list[str],
        classifier: Classifier,
    ) -> SensitivityReport:
        """Measure detection margins for each injection prompt.

        For each prompt, runs the semantic preservation attack to find the
        closest adversarial variant. The "margin" is the confidence drop
        needed to cross the threshold. Small margins = fragile detections.

        Args:
            injections: List of known injection prompts.
            classifier: Classifier to test against.

        Returns:
            SensitivityReport classifying prompts as fragile or robust.
        """
        margins: list[float] = []
        fragile: list[str] = []
        robust: list[str] = []

        for injection in injections:
            original_conf = await classifier.classify(injection)

            if original_conf < self._threshold:
                # Already below threshold — margin is 0
                margins.append(0.0)
                fragile.append(injection)
                continue

            # Run semantic preservation attack to find closest evasion
            examples = await self._attacker.semantic_preservation_attack(
                injection, classifier,
            )

            if examples:
                best = min(examples, key=lambda e: e.adversarial_confidence)
                margin = original_conf - best.adversarial_confidence
            else:
                # No synonyms available — use original confidence as margin
                margin = original_conf - self._threshold
                if margin < 0:
                    margin = 0.0

            margins.append(margin)
            if margin < 0.1:
                fragile.append(injection)
            elif margin > 0.3:
                robust.append(injection)

        mean_margin = sum(margins) / len(margins) if margins else 0.0

        return SensitivityReport(
            per_prompt_margins=margins,
            mean_margin=mean_margin,
            fragile_prompts=fragile,
            robust_prompts=robust,
        )
