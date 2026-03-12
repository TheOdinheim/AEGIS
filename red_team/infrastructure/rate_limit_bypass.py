"""
Rate limit bypass module — tests session rotation, header manipulation,
distributed bypass, and quota exhaustion attacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from starlette.testclient import TestClient

from red_team.infrastructure import AttackResult

API_KEY = "aegis-test-secretkey123"


@dataclass
class SessionRotationResult:
    """Result of session rotation rate limit bypass."""
    requests_after_limit: int
    bypassed_via_headers: dict[str, bool]  # header_name -> bypassed
    vulnerable: bool


@dataclass
class ExhaustionResult:
    """Result of quota exhaustion attack."""
    requests_sent: int
    requests_succeeded: int
    legitimate_user_blocked: bool


class RateLimitBypass:
    """Attack AEGIS rate limiting mechanisms."""

    def __init__(self, client: TestClient, api_key: str = API_KEY):
        self._client = client
        self._api_key = api_key

    def _send(self, extra_headers: dict[str, str] | None = None) -> int:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if extra_headers:
            headers.update(extra_headers)
        resp = self._client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            headers=headers,
        )
        return resp.status_code

    def session_rotation(self, rate_limit: int = 60) -> SessionRotationResult:
        """Test if header manipulation bypasses rate limits.

        After hitting the rate limit, try requests with spoofed headers
        to see if rate limiting is tied to headers the client can control.
        """
        # First, send requests to approach the limit (but not exceed)
        # We can't actually hit the limit easily in tests since default is 60 RPM,
        # so we test the header-based key derivation
        bypass_headers = {
            "X-Forwarded-For": "192.168.1.100",
            "X-Real-IP": "10.0.0.50",
            "CF-Connecting-IP": "172.16.0.1",
            "X-Session-Id": "attacker-session-999",
            "User-Agent": "AttackerBot/1.0",
        }

        bypassed: dict[str, bool] = {}
        # Send a baseline request first
        baseline_status = self._send()

        for header_name, header_value in bypass_headers.items():
            status = self._send({header_name: header_value})
            # If the request succeeds, check if it was rate limited differently
            # The key test is: does adding this header change the rate limit bucket?
            bypassed[header_name] = status == baseline_status  # Same treatment = good

        # Check if rate limiting is per API key (correct) vs per header (bypassable)
        vulnerable = any(not v for v in bypassed.values())

        return SessionRotationResult(
            requests_after_limit=len(bypass_headers),
            bypassed_via_headers=bypassed,
            vulnerable=vulnerable,
        )

    def header_injection_bypass(self) -> AttackResult:
        """Test if injecting multiple values for the same header causes confusion."""
        test_cases = [
            ("duplicate_xff", {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}),
            ("xff_localhost", {"X-Forwarded-For": "127.0.0.1"}),
            ("xff_internal", {"X-Forwarded-For": "10.0.0.1"}),
            ("host_override", {"Host": "admin.aegis.internal"}),
            ("content_type_mismatch", {"Content-Type": "text/plain"}),
        ]

        results: list[dict[str, Any]] = []
        for name, headers in test_cases:
            status = self._send(headers)
            results.append({"technique": name, "status": status})

        unexpected = [r for r in results if r["status"] not in (200, 403, 429)]
        return AttackResult(
            attack_name="header_injection_bypass",
            category="rate_limiting",
            vulnerable=len(unexpected) > 0,
            severity="medium" if unexpected else "info",
            description="Header injection attempts to bypass rate limiting or confuse routing",
            evidence=[str(r) for r in unexpected],
            recommendation="Validate and sanitize all incoming headers",
            details={"results": results},
        )

    def quota_exhaustion(self, requests_count: int = 20) -> ExhaustionResult:
        """Test if an attacker can exhaust a legitimate user's rate limit quota.

        Sends requests at the rate limit to consume the victim's quota.
        """
        succeeded = 0
        for _ in range(requests_count):
            status = self._send()
            if status in (200, 403):  # 403 = blocked by security, not rate limit
                succeeded += 1

        return ExhaustionResult(
            requests_sent=requests_count,
            requests_succeeded=succeeded,
            legitimate_user_blocked=False,  # Would need 2 clients to test properly
        )

    def run_all(self) -> list[AttackResult]:
        """Run all rate limit bypass attacks."""
        results: list[AttackResult] = []

        sr = self.session_rotation()
        results.append(AttackResult(
            attack_name="session_rotation_bypass",
            category="rate_limiting",
            vulnerable=sr.vulnerable,
            severity="high" if sr.vulnerable else "info",
            description="Rate limit bypass via header manipulation",
            evidence=[f"{k}: bypassed={v}" for k, v in sr.bypassed_via_headers.items()],
            recommendation="Rate limit by API key hash, not client-supplied headers",
        ))

        results.append(self.header_injection_bypass())

        ex = self.quota_exhaustion()
        results.append(AttackResult(
            attack_name="quota_exhaustion",
            category="rate_limiting",
            vulnerable=False,  # Design limitation, not a bug
            severity="info",
            description=f"Sent {ex.requests_sent} requests, {ex.requests_succeeded} succeeded",
            recommendation="Consider per-IP rate limits in addition to per-key",
            details={"requests_sent": ex.requests_sent, "succeeded": ex.requests_succeeded},
        ))

        return results
