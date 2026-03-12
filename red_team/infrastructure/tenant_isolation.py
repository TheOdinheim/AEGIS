"""
Tenant isolation attack module — tests cross-tenant data leakage,
privilege escalation, and tenant ID injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from starlette.testclient import TestClient

from red_team.infrastructure import AttackResult

API_KEY = "aegis-test-secretkey123"


@dataclass
class LeakageResult:
    """Result of cross-tenant leakage test."""
    canary_leaked: bool
    leak_locations: list[str]


@dataclass
class EscalationResult:
    """Result of privilege escalation test."""
    escalations_found: int
    results: list[dict[str, Any]]


class TenantIsolationTester:
    """Test tenant isolation and privilege escalation."""

    def __init__(self, client: TestClient, api_key: str = API_KEY):
        self._client = client
        self._api_key = api_key

    def _headers(self, key: str | None = None, session: str | None = None) -> dict[str, str]:
        h: dict[str, str] = {"Authorization": f"Bearer {key or self._api_key}"}
        if session:
            h["x-session-id"] = session
        return h

    def cross_tenant_leakage(self) -> LeakageResult:
        """Test if tenant A's data appears in tenant B's views.

        Since we use a single API key in test mode, we test by checking
        that endpoints don't leak data across sessions and that health
        endpoint doesn't expose tenant-specific information.
        """
        leak_locations: list[str] = []

        # Send a canary request as "tenant A"
        canary = "CANARY_TENANT_A_SECRET_12345"
        self._client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": canary}]},
            headers=self._headers(session="tenant-a-session"),
        )

        # Check health endpoint (unauthenticated) for leakage
        resp = self._client.get("/health")
        if canary in resp.text:
            leak_locations.append("/health (unauthenticated)")

        # Check authenticated health for leakage
        resp = self._client.get("/health", headers=self._headers())
        if canary in resp.text:
            leak_locations.append("/health (authenticated)")

        # Check vault stats for prompt leakage
        resp = self._client.get("/v1/vault/stats", headers=self._headers())
        if resp.status_code == 200 and canary in resp.text:
            leak_locations.append("/v1/vault/stats")

        # Check models endpoint
        resp = self._client.get("/v1/models")
        if canary in resp.text:
            leak_locations.append("/v1/models")

        return LeakageResult(
            canary_leaked=len(leak_locations) > 0,
            leak_locations=leak_locations,
        )

    def privilege_escalation(self) -> EscalationResult:
        """Test if a normal user can access admin endpoints."""
        admin_endpoints = [
            ("POST", "/v1/admin/backup", {}),
            ("GET", "/v1/admin/deep-health", None),
            ("GET", "/v1/admin/config-validation", None),
            ("POST", "/v1/admin/config-reload", {}),
        ]

        results: list[dict[str, Any]] = []
        escalations = 0

        for method, path, body in admin_endpoints:
            if method == "GET":
                resp = self._client.get(path, headers=self._headers())
            else:
                resp = self._client.post(path, json=body or {}, headers=self._headers())

            # Admin endpoints SHOULD be accessible with valid key (no RBAC yet)
            # But we document this as a finding
            accessible = resp.status_code not in (401, 403, 404)
            if accessible:
                escalations += 1
            results.append({
                "endpoint": f"{method} {path}",
                "status": resp.status_code,
                "accessible": accessible,
            })

        return EscalationResult(
            escalations_found=escalations,
            results=results,
        )

    def tenant_id_injection(self) -> AttackResult:
        """Test tenant ID injection via various request fields."""
        injection_attempts = [
            # In custom headers
            ("X-Tenant-Id", "admin"),
            ("X-Organization", "competitor-corp"),
            ("X-Tenant-Id", "'; DROP TABLE tenants; --"),
            # In model name
            (None, None),  # Model name injection handled below
        ]

        results: list[dict[str, Any]] = []

        # Header-based injection
        for header, value in injection_attempts:
            if header is None:
                continue
            resp = self._client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]},
                headers={**self._headers(), header: value},
            )
            results.append({
                "technique": f"header:{header}={value}",
                "status": resp.status_code,
                "affected": resp.status_code not in (200, 403),
            })

        # Model name injection
        for model_name in [
            "gpt-4; tenant_id=admin",
            "gpt-4\nX-Tenant: admin",
            "gpt-4' OR 1=1 --",
            "../../../etc/passwd",
        ]:
            resp = self._client.post(
                "/v1/chat/completions",
                json={"model": model_name, "messages": [{"role": "user", "content": "test"}]},
                headers=self._headers(),
            )
            results.append({
                "technique": f"model_name:{model_name[:40]}",
                "status": resp.status_code,
                "affected": resp.status_code == 500,  # 500 = injection worked
            })

        vulnerable = any(r["affected"] for r in results)
        return AttackResult(
            attack_name="tenant_id_injection",
            category="tenant_isolation",
            vulnerable=vulnerable,
            severity="critical" if vulnerable else "info",
            description="Tenant ID injection via headers and request fields",
            evidence=[str(r) for r in results if r["affected"]],
            recommendation="Never trust client-supplied tenant identifiers",
            details={"results": results},
        )

    def run_all(self) -> list[AttackResult]:
        """Run all tenant isolation attacks."""
        results: list[AttackResult] = []

        leakage = self.cross_tenant_leakage()
        results.append(AttackResult(
            attack_name="cross_tenant_leakage",
            category="tenant_isolation",
            vulnerable=leakage.canary_leaked,
            severity="critical" if leakage.canary_leaked else "info",
            description="Cross-tenant data leakage via shared endpoints",
            evidence=leakage.leak_locations,
            recommendation="Ensure all endpoints filter by authenticated tenant",
        ))

        esc = self.privilege_escalation()
        results.append(AttackResult(
            attack_name="privilege_escalation",
            category="tenant_isolation",
            vulnerable=esc.escalations_found > 0,
            severity="medium",
            description=f"Admin endpoints accessible: {esc.escalations_found}/{len(esc.results)}",
            evidence=[r["endpoint"] for r in esc.results if r["accessible"]],
            recommendation="Implement RBAC — separate admin keys from regular API keys",
            details={"results": esc.results},
        ))

        results.append(self.tenant_id_injection())

        return results
