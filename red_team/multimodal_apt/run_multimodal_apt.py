#!/usr/bin/env python3
"""
Multimodal APT Entry Point — Run campaigns against live AEGIS.

Usage:
    # Run all campaigns:
    python3 -m red_team.multimodal_apt.run_multimodal_apt --url URL --api-key KEY

    # Run specific campaign:
    python3 -m red_team.multimodal_apt.run_multimodal_apt --url URL --api-key KEY --campaign PRISM

    # Adaptive only:
    python3 -m red_team.multimodal_apt.run_multimodal_apt --url URL --api-key KEY --adaptive-only

    # Total War only:
    python3 -m red_team.multimodal_apt.run_multimodal_apt --url URL --api-key KEY --total-war-only

    # Run CI-safe tests (no live server):
    python3 -m pytest tests/test_multimodal_apt.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure aegis importable
_project = Path(__file__).resolve().parent.parent.parent
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))
_parent = _project.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS Multimodal APT Campaign Runner")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="AEGIS base URL")
    parser.add_argument("--api-key", type=str, default="", help="AEGIS API key")
    parser.add_argument("--output", type=str, default="red_team/data", help="Output directory")
    parser.add_argument("--campaign", type=str, default=None, help="Run specific campaign")
    parser.add_argument("--adaptive-only", action="store_true", help="Run ADAPTATION campaign only")
    parser.add_argument("--total-war-only", action="store_true", help="Run TOTAL WAR only")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name for requests")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("AEGIS_API_KEY", "")

    from red_team.multimodal_apt.live_campaign_runner import LiveCampaignRunner
    from red_team.multimodal_apt.report import MultimodalAPTReport
    from red_team.multimodal_apt import MultimodalAPTAssessment, _grade

    runner = LiveCampaignRunner(
        aegis_url=args.url,
        api_key=api_key,
        model=args.model,
    )

    campaigns = {
        "PRISM": runner.run_prism,
        "DEEP COVER": runner.run_deep_cover,
        "WHISPER": runner.run_whisper,
        "CHIMERA": runner.run_chimera,
        "PUPPET MASTER": runner.run_puppet_master,
        "ADAPTATION": runner.run_adaptation,
        "TOTAL WAR": runner.run_total_war,
    }

    if args.campaign and args.campaign not in campaigns:
        print(f"Unknown campaign: {args.campaign}")
        print(f"Available: {', '.join(campaigns.keys())}")
        sys.exit(1)

    if args.adaptive_only:
        to_run = {"ADAPTATION": campaigns["ADAPTATION"]}
    elif args.total_war_only:
        to_run = {"TOTAL WAR": campaigns["TOTAL WAR"]}
    elif args.campaign:
        to_run = {args.campaign: campaigns[args.campaign]}
    else:
        to_run = campaigns

    async def _run() -> MultimodalAPTAssessment:
        assessment = MultimodalAPTAssessment()

        for name, fn in to_run.items():
            print(f"\nRunning Campaign: {name}...")
            result = await fn()
            assessment.campaigns.append(result)
            print(
                f"  {name}: {result.blocked}/{result.total_attacks} blocked "
                f"({result.detection_rate:.1%} detection rate)"
            )
            if result.false_positive_rate > 0:
                print(f"  FPR: {result.false_positive_rate:.1%}")

        # Compute overall metrics
        total_attacks = sum(c.total_attacks for c in assessment.campaigns)
        total_blocked = sum(c.blocked for c in assessment.campaigns)
        total_fp = sum(c.false_positives for c in assessment.campaigns)

        assessment.overall_detection_rate = total_blocked / total_attacks if total_attacks else 0
        assessment.overall_fpr = total_fp / total_attacks if total_attacks else 0
        assessment.compute_grade()
        assessment.executive_summary = (
            f"Multimodal APT assessment across {len(assessment.campaigns)} campaigns. "
            f"Overall detection rate: {assessment.overall_detection_rate:.1%}. "
            f"Overall FPR: {assessment.overall_fpr:.1%}. "
            f"Grade: {assessment.grade}."
        )

        return assessment

    assessment = asyncio.run(_run())

    # Generate report
    reporter = MultimodalAPTReport()
    report = reporter.generate_report(assessment, args.output)
    print(f"\nOverall Grade: {assessment.grade}")
    print(f"Detection Rate: {assessment.overall_detection_rate:.1%}")
    print(f"FPR: {assessment.overall_fpr:.1%}")
    print(f"\nReport saved to {args.output}/")


if __name__ == "__main__":
    main()
