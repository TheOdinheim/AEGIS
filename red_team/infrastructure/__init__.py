"""
Red Team Phase 5 — Infrastructure Security Attack Engine.

Attacks AEGIS itself: authentication, rate limiting, tenant isolation,
backing services, denial of service, audit integrity, federated learning
poisoning, and supply chain self-protection.

All modules provide both analysis classes (for documentation/reporting)
and testable attack methods that run via TestClient or direct function calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttackResult:
    """Result of a single infrastructure attack test."""

    attack_name: str
    category: str
    vulnerable: bool
    severity: str  # "critical", "high", "medium", "low", "info"
    description: str
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_name": self.attack_name,
            "category": self.category,
            "vulnerable": self.vulnerable,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "details": self.details,
        }


@dataclass
class InfrastructureAssessment:
    """Full infrastructure security assessment."""

    results: list[AttackResult] = field(default_factory=list)
    total_tests: int = 0
    vulnerabilities_found: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def add_result(self, result: AttackResult) -> None:
        self.results.append(result)
        self.total_tests += 1
        if result.vulnerable:
            self.vulnerabilities_found += 1
            sev = result.severity
            if sev == "critical":
                self.critical_count += 1
            elif sev == "high":
                self.high_count += 1
            elif sev == "medium":
                self.medium_count += 1
            elif sev == "low":
                self.low_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "vulnerabilities_found": self.vulnerabilities_found,
            "severity_breakdown": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "results": [r.to_dict() for r in self.results],
        }
