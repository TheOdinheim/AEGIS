#!/usr/bin/env python3
"""
Red Team Entry Point — Run APT campaigns against AEGIS.

Usage:
    # Run all campaigns against in-process AEGIS (no live server needed):
    python3 -m red_team.run_red_team

    # Run a specific campaign:
    python3 -m red_team.run_red_team --campaign "PHANTOM NEEDLE"

    # Save results to custom path:
    python3 -m red_team.run_red_team --output red_team/data/results.json

    # Run from pytest (CI-safe, via test_red_team_regression.py):
    python3 -m pytest tests/test_red_team_regression.py -v
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Ensure AEGIS modules importable
_project = Path(__file__).resolve().parent.parent
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))
_parent = _project.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS Red Team APT Campaign Runner")
    parser.add_argument(
        "--campaign",
        type=str,
        default=None,
        help="Run a specific campaign by name (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="red_team/data/campaign_results.json",
        help="Output JSON path (default: red_team/data/campaign_results.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Configure environment
    os.environ.setdefault("AEGIS_SKIP_MODEL_LOAD", "true")
    os.environ.setdefault("AEGIS_API_KEY", "aegis-test-secretkey123")
    os.environ.setdefault("AEGIS_UPSTREAM_URL", "http://localhost:11434")

    random.seed(args.seed)

    from aegis.config import AegisConfig
    from aegis.main import app, _init_layers
    from starlette.testclient import TestClient

    import red_team.apt_campaigns as apt

    campaigns = {
        "PHANTOM NEEDLE": apt.run_phantom_needle,
        "SILENT SIPHON": apt.run_silent_siphon,
        "SLOW BURN": apt.run_slow_burn,
        "HYDRA": apt.run_hydra,
        "GHOST PROTOCOL": apt.run_ghost_protocol,
        "CASCADING FAILURE": apt.run_cascading_failure,
    }

    if args.campaign and args.campaign not in campaigns:
        print(f"Unknown campaign: {args.campaign}")
        print(f"Available: {', '.join(campaigns.keys())}")
        sys.exit(1)

    config = AegisConfig()
    client = TestClient(app, raise_server_exceptions=False)

    to_run = {args.campaign: campaigns[args.campaign]} if args.campaign else campaigns

    results = {}
    for name, fn in to_run.items():
        _init_layers(config)
        random.seed(args.seed)
        result = fn(client)
        results[name] = result.to_dict()
        print(
            f"{name}: {result.attacks_total} total, "
            f"{result.attacks_blocked} blocked, "
            f"{result.attacks_evaded} evaded "
            f"({result.evasion_rate*100:.1f}%), "
            f"attacker_success={result.overall_success}"
        )
        for s in result.stages:
            print(
                f"  {s.stage_name}: {s.attacks_sent} sent, "
                f"{s.attacks_blocked} blocked, "
                f"{s.attacks_evaded} evaded ({s.evasion_rate*100:.1f}%)"
            )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
