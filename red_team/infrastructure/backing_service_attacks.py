"""
Backing service attack module — tests Redis and PostgreSQL exposure,
default credential exploitation, and service dependency confusion.

These are primarily analysis/documentation tests since backing services
are not available in CI. The tests validate AEGIS's behavior when
backing services return unexpected data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from red_team.infrastructure import AttackResult


@dataclass
class RedisExploitResult:
    """Result of Redis exploitation analysis."""
    exposed: bool
    auth_required: bool
    findings: list[str]


@dataclass
class PostgresExploitResult:
    """Result of PostgreSQL exploitation analysis."""
    exposed: bool
    default_creds_work: bool
    findings: list[str]


class BackingServiceAttacker:
    """Analyze backing service security posture.

    These tests are primarily documentation/analysis — they check
    configurations and code paths rather than actually attacking live services.
    """

    def analyze_redis_security(self) -> AttackResult:
        """Analyze Redis security configuration.

        Checks:
        - Is Redis configured with authentication?
        - Is Redis exposed on a network-accessible port?
        - Are Redis commands restricted (rename-command)?
        """
        findings: list[str] = []
        vulnerable = False

        # Check docker-compose.yml for Redis configuration
        try:
            with open("docker-compose.yml", "r") as f:
                compose = f.read()
            if "6379:6379" in compose:
                findings.append("Redis port 6379 is mapped to host — accessible from outside container")
                vulnerable = True
            if "requirepass" not in compose and "REDIS_PASSWORD" not in compose:
                findings.append("No Redis password configured in docker-compose.yml")
                vulnerable = True
            if "rename-command" not in compose:
                findings.append("No dangerous command renaming (FLUSHALL, KEYS, CONFIG)")
        except FileNotFoundError:
            findings.append("docker-compose.yml not found — cannot analyze Redis config")

        # Check if redis_client.py uses authentication
        try:
            with open("services/redis_client.py", "r") as f:
                redis_code = f.read()
            if "password" not in redis_code.lower():
                findings.append("redis_client.py does not explicitly handle Redis password")
        except FileNotFoundError:
            pass

        return AttackResult(
            attack_name="redis_security_analysis",
            category="backing_services",
            vulnerable=vulnerable,
            severity="high" if vulnerable else "low",
            description="Redis security configuration analysis",
            evidence=findings,
            recommendation="Configure Redis AUTH, bind to localhost only, rename dangerous commands",
        )

    def analyze_postgres_security(self) -> AttackResult:
        """Analyze PostgreSQL security configuration."""
        findings: list[str] = []
        vulnerable = False

        try:
            with open("docker-compose.yml", "r") as f:
                compose = f.read()
            if "5432:5432" in compose:
                findings.append("PostgreSQL port 5432 is mapped to host")
                vulnerable = True
            if "POSTGRES_PASSWORD: aegis" in compose or "POSTGRES_PASSWORD: postgres" in compose:
                findings.append("Default/weak PostgreSQL password in docker-compose.yml")
                vulnerable = True
        except FileNotFoundError:
            findings.append("docker-compose.yml not found")

        # Check for SQL injection surfaces
        try:
            with open("services/audit_logger.py", "r") as f:
                audit_code = f.read()
            if "f\"" in audit_code and "SQL" in audit_code.upper():
                findings.append("Potential f-string SQL construction in audit_logger.py")
        except FileNotFoundError:
            pass

        return AttackResult(
            attack_name="postgres_security_analysis",
            category="backing_services",
            vulnerable=vulnerable,
            severity="high" if vulnerable else "low",
            description="PostgreSQL security configuration analysis",
            evidence=findings,
            recommendation="Use strong passwords, bind to localhost, use pg_hba.conf for ACLs",
        )

    def analyze_dependency_trust(self) -> AttackResult:
        """Analyze whether AEGIS validates backing service responses.

        AEGIS should not blindly trust data from Redis/PostgreSQL —
        a compromised backing service could return manipulated data.
        """
        findings: list[str] = []

        # Rate limit data from Redis is trusted directly
        findings.append(
            "Rate limit counts from Redis are trusted without validation — "
            "a compromised Redis could report 'under limit' for all keys"
        )

        # Tenant config from PostgreSQL is trusted
        findings.append(
            "Tenant configuration from PostgreSQL is trusted — "
            "a compromised DB could grant admin-level permissions to any tenant"
        )

        # Event bus messages from Redis are advisory (good)
        findings.append(
            "Event bus messages are advisory-only (good) — layers maintain independent state"
        )

        return AttackResult(
            attack_name="dependency_trust_analysis",
            category="backing_services",
            vulnerable=True,
            severity="medium",
            description="Analysis of trust boundaries with backing services",
            evidence=findings,
            recommendation="Add integrity checks on critical data from backing services",
        )

    def run_all(self) -> list[AttackResult]:
        """Run all backing service analysis."""
        return [
            self.analyze_redis_security(),
            self.analyze_postgres_security(),
            self.analyze_dependency_trust(),
        ]
