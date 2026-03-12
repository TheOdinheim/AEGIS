"""
Campaign runner — orchestrates multi-stage adversarial attack campaigns.

A campaign consists of:
1. Generate attacks (or load from corpus)
2. Run evasion tests against AEGIS
3. Feed successful blocks into the immune system (triggers antibody generation)
4. Validate that the immune system learned and can generalize
5. Re-run variant attacks to measure improvement
6. Generate comprehensive report
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from red_team import (
    Attack,
    EvasionReport,
    LearningReport,
)
from red_team.attack_generator import AdversarialAttackGenerator
from red_team.evasion_engine import EvasionEngine
from red_team.learning_validator import LearningValidator
from red_team.report import RedTeamReport

logger = logging.getLogger(__name__)


@dataclass
class CampaignConfig:
    """Configuration for a red team campaign."""

    name: str = "default-campaign"
    aegis_url: str = "http://localhost:8000"
    api_key: str = ""
    concurrency: int = 5
    learning_wait_seconds: int = 5
    output_dir: str = "red_team/reports"
    corpus_path: str | None = None  # Load existing corpus instead of generating
    generate_count: int | None = None  # Override per-technique counts


@dataclass
class CampaignResult:
    """Full result of a red team campaign."""

    config: CampaignConfig
    initial_evasion: EvasionReport
    learning: LearningReport
    post_learning_evasion: EvasionReport | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_name": self.config.name,
            "timestamp": self.timestamp,
            "initial_evasion": self.initial_evasion.to_dict(),
            "learning": self.learning.to_dict(),
            "post_learning_evasion": (
                self.post_learning_evasion.to_dict()
                if self.post_learning_evasion
                else None
            ),
        }


class CampaignRunner:
    """Orchestrates end-to-end red team campaigns against AEGIS.

    Lifecycle:
        1. generate/load attacks
        2. initial evasion test
        3. validate immune system learning
        4. optional: re-run variants to measure improvement
        5. generate report
    """

    def __init__(self, config: CampaignConfig) -> None:
        self._config = config
        self._generator = AdversarialAttackGenerator()
        self._engine = EvasionEngine(concurrency=config.concurrency)
        self._validator = LearningValidator()
        self._reporter = RedTeamReport()

    async def run(self) -> CampaignResult:
        """Execute the full campaign pipeline.

        Returns:
            CampaignResult with initial evasion, learning validation, and report.
        """
        logger.info("Starting red team campaign: %s", self._config.name)

        # Step 1: Generate or load attacks
        if self._config.corpus_path:
            attacks = AdversarialAttackGenerator.load_corpus(self._config.corpus_path)
            logger.info("Loaded %d attacks from %s", len(attacks), self._config.corpus_path)
        else:
            attacks = self._generator.generate_full_corpus()
            logger.info("Generated %d attacks", len(attacks))

        # Step 2: Initial evasion test
        logger.info("Running initial evasion test (%d attacks)...", len(attacks))
        initial_report = await self._engine.run_evasion_test(
            attacks, self._config.aegis_url, self._config.api_key
        )
        logger.info(
            "Initial results: %d/%d blocked (%.1f%% evasion rate)",
            initial_report.blocked,
            initial_report.total_attacks,
            initial_report.evasion_rate * 100,
        )

        # Step 3: Validate immune system learning
        blocked_attacks = [
            a for a, r in zip(attacks, initial_report.all_results)
            if r.was_blocked
        ]
        evaded_attacks = [
            a for a, r in zip(attacks, initial_report.all_results)
            if not r.was_blocked
        ]

        logger.info("Validating learning from %d blocked attacks...", len(blocked_attacks))
        learning_report = await self._validator.validate_learning(
            blocked_attacks=blocked_attacks,
            evaded_attacks=evaded_attacks,
            aegis_url=self._config.aegis_url,
            api_key=self._config.api_key,
            wait_seconds=self._config.learning_wait_seconds,
        )
        logger.info(
            "Learning: %d antibodies, %.1f%% generalization rate, %d detection gaps",
            learning_report.antibodies_generated,
            learning_report.generalization_rate * 100,
            len(learning_report.detection_gaps),
        )

        # Step 4: Build result
        result = CampaignResult(
            config=self._config,
            initial_evasion=initial_report,
            learning=learning_report,
        )

        # Step 5: Generate and save report
        output_dir = Path(self._config.output_dir)
        self._reporter.generate_full_report(result, output_dir)
        logger.info("Report saved to %s", output_dir)

        return result
