"""
Infrastructure security attack orchestrator.

Runs all infrastructure attack modules and produces a consolidated
assessment report. Can run via TestClient (in-process) or against
a live AEGIS instance.

Usage:
    python3 -m red_team.infrastructure.run_infrastructure_attacks [--url URL] [--api-key KEY] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

from starlette.testclient import TestClient

from red_team.infrastructure import AttackResult, InfrastructureAssessment
from red_team.infrastructure.audit_integrity import AuditIntegrityTester
from red_team.infrastructure.auth_attacks import AuthAttacker
from red_team.infrastructure.backing_service_attacks import BackingServiceAttacker
from red_team.infrastructure.denial_of_service import DenialOfServiceTester
from red_team.infrastructure.federated_poisoning import FederatedPoisoningAttacker
from red_team.infrastructure.rate_limit_bypass import RateLimitBypass
from red_team.infrastructure.supply_chain_self import SelfSupplyChainTester
from red_team.infrastructure.tenant_isolation import TenantIsolationTester

API_KEY = "aegis-test-secretkey123"


def _create_testclient() -> TestClient:
    """Create a TestClient for in-process testing."""
    import os
    os.environ.setdefault("AEGIS_SKIP_MODEL_LOAD", "true")
    os.environ.setdefault("AEGIS_API_KEY", API_KEY)

    from aegis.main import app
    return TestClient(app)


def _patch_upstream(content: str = "Hello"):
    """Patch upstream forwarding for in-process testing."""
    async def mock_forward(body, upstream_url):
        return {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    return patch("aegis.main._forward_to_upstream", side_effect=mock_forward)


def run_assessment(
    client: TestClient | None = None,
    api_key: str = API_KEY,
    project_root: str | None = None,
) -> InfrastructureAssessment:
    """Run all infrastructure security attacks.

    Args:
        client: TestClient instance. Created if None.
        api_key: API key for authenticated endpoints.
        project_root: Path to AEGIS project root for file-based checks.

    Returns:
        InfrastructureAssessment with all results.
    """
    if client is None:
        client = _create_testclient()

    assessment = InfrastructureAssessment()

    # 1. Authentication attacks
    print("[1/8] Running authentication attacks...")
    auth = AuthAttacker(client, api_key)
    for result in auth.run_all():
        assessment.add_result(result)

    # 2. Rate limit bypass
    print("[2/8] Running rate limit bypass attacks...")
    rl = RateLimitBypass(client, api_key)
    for result in rl.run_all():
        assessment.add_result(result)

    # 3. Tenant isolation
    print("[3/8] Running tenant isolation attacks...")
    tenant = TenantIsolationTester(client, api_key)
    for result in tenant.run_all():
        assessment.add_result(result)

    # 4. Backing service analysis
    print("[4/8] Running backing service analysis...")
    backing = BackingServiceAttacker()
    for result in backing.run_all():
        assessment.add_result(result)

    # 5. Denial of service
    print("[5/8] Running denial of service attacks...")
    dos = DenialOfServiceTester(client, api_key)
    for result in dos.run_all():
        assessment.add_result(result)

    # 6. Audit integrity
    print("[6/8] Running audit integrity attacks...")
    audit = AuditIntegrityTester(client, api_key)
    for result in audit.run_all():
        assessment.add_result(result)

    # 7. Federated poisoning
    print("[7/8] Running federated poisoning attacks...")
    federated = FederatedPoisoningAttacker()
    for result in federated.run_all():
        assessment.add_result(result)

    # 8. Supply chain self-check
    print("[8/8] Running supply chain self-check...")
    supply = SelfSupplyChainTester(project_root)
    for result in supply.run_all():
        assessment.add_result(result)

    return assessment


def format_report(assessment: InfrastructureAssessment) -> str:
    """Format assessment as human-readable markdown."""
    lines: list[str] = []
    lines.append("# AEGIS Infrastructure Security Assessment")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"\n## Summary\n")
    lines.append(f"- **Total tests**: {assessment.total_tests}")
    lines.append(f"- **Vulnerabilities found**: {assessment.vulnerabilities_found}")
    lines.append(f"- **Critical**: {assessment.critical_count}")
    lines.append(f"- **High**: {assessment.high_count}")
    lines.append(f"- **Medium**: {assessment.medium_count}")
    lines.append(f"- **Low**: {assessment.low_count}")

    # Group by category
    categories: dict[str, list[AttackResult]] = {}
    for r in assessment.results:
        categories.setdefault(r.category, []).append(r)

    for category, results in sorted(categories.items()):
        lines.append(f"\n## {category.replace('_', ' ').title()}\n")
        for r in results:
            status = "VULNERABLE" if r.vulnerable else "PASS"
            icon = "!!!" if r.vulnerable else "OK"
            lines.append(f"### [{icon}] {r.attack_name} ({r.severity})\n")
            lines.append(f"**Status**: {status}")
            lines.append(f"\n{r.description}\n")
            if r.evidence:
                lines.append("**Evidence**:")
                for e in r.evidence:
                    lines.append(f"- {e}")
            if r.recommendation:
                lines.append(f"\n**Recommendation**: {r.recommendation}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AEGIS Infrastructure Security Assessment",
    )
    parser.add_argument(
        "--api-key", default=API_KEY,
        help="API key for authenticated endpoints",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--project-root", default=None,
        help="AEGIS project root directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AEGIS Infrastructure Security Assessment")
    print("=" * 60)

    assessment = run_assessment(
        api_key=args.api_key,
        project_root=args.project_root,
    )

    # Print report
    report = format_report(assessment)
    print(report)

    # Save JSON
    if args.output:
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assessment": assessment.to_dict(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nJSON results saved to {args.output}")

    # Exit code
    if assessment.critical_count > 0:
        print(f"\n!!! {assessment.critical_count} CRITICAL vulnerabilities found")
        sys.exit(2)
    elif assessment.high_count > 0:
        print(f"\n!! {assessment.high_count} HIGH vulnerabilities found")
        sys.exit(1)
    else:
        print("\nNo critical or high vulnerabilities found")
        sys.exit(0)


if __name__ == "__main__":
    main()
