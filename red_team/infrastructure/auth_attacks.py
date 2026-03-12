"""
Authentication attack module — tests API key validation, timing attacks,
key format inference, and authentication bypass techniques.

All tests run via TestClient (no live server needed).
"""

from __future__ import annotations

import base64
import hashlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from starlette.testclient import TestClient

from red_team.infrastructure import AttackResult

API_KEY = "aegis-test-secretkey123"


@dataclass
class TimingAttackResult:
    """Result of timing analysis on key validation."""
    correlation_coefficient: float
    vulnerable: bool
    mean_times: dict[str, float]  # category -> mean response time ms
    recommendation: str


@dataclass
class FormatInferenceResult:
    """Result of key format inference attack."""
    error_messages: dict[str, str]  # key_variant -> error message
    info_leaked: bool
    recommendation: str


@dataclass
class AuthBypassResult:
    """Result of authentication bypass attempts."""
    techniques_tested: int
    bypasses_found: int
    results: list[dict[str, Any]]


class AuthAttacker:
    """Attack AEGIS authentication mechanisms."""

    def __init__(self, client: TestClient, api_key: str = API_KEY):
        self._client = client
        self._api_key = api_key

    def _chat_body(self) -> dict:
        return {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

    def timing_attack(self, iterations: int = 100) -> TimingAttackResult:
        """Measure response time correlation with key correctness.

        Sends requests with keys of varying correctness and measures
        whether response time correlates with the number of correct characters.
        """
        categories = {
            "wrong_key": "completely-wrong-random-key-abcdef",
            "correct_prefix": self._api_key[:5] + "x" * (len(self._api_key) - 5),
            "half_correct": self._api_key[:len(self._api_key)//2] + "x" * (len(self._api_key) - len(self._api_key)//2),
            "mostly_correct": self._api_key[:-2] + "xx",
            "correct_key": self._api_key,
        }

        times: dict[str, list[float]] = {k: [] for k in categories}

        for _ in range(iterations):
            for cat, key in categories.items():
                start = time.perf_counter()
                self._client.post(
                    "/v1/chat/completions",
                    json=self._chat_body(),
                    headers={"Authorization": f"Bearer {key}"},
                )
                elapsed = (time.perf_counter() - start) * 1000
                times[cat].append(elapsed)

        mean_times = {k: statistics.mean(v) for k, v in times.items()}

        # Check if there's a statistically significant difference
        wrong_times = times["wrong_key"]
        correct_times = times["correct_key"]
        mean_diff = abs(statistics.mean(correct_times) - statistics.mean(wrong_times))
        combined_std = (statistics.stdev(wrong_times) + statistics.stdev(correct_times)) / 2

        # Vulnerable if difference > 2x combined standard deviation
        vulnerable = combined_std > 0 and mean_diff / combined_std > 2.0

        # Simple correlation: do means increase with correctness?
        ordered_means = [mean_times[k] for k in categories]
        diffs = [ordered_means[i+1] - ordered_means[i] for i in range(len(ordered_means)-1)]
        positive_diffs = sum(1 for d in diffs if d > 0)
        correlation = positive_diffs / len(diffs) if diffs else 0.0

        return TimingAttackResult(
            correlation_coefficient=correlation,
            vulnerable=vulnerable,
            mean_times=mean_times,
            recommendation=(
                "Use hmac.compare_digest() for constant-time key comparison"
                if vulnerable else "No significant timing difference detected"
            ),
        )

    def key_format_inference(self) -> FormatInferenceResult:
        """Test if error messages leak key format information."""
        test_keys = {
            "empty": "",
            "short_1char": "a",
            "short_5chars": "abcde",
            "correct_length_wrong": "x" * len(self._api_key),
            "special_chars": "!@#$%^&*()",
            "with_spaces": "aegis test key 123",
            "unicode_key": "aegis-aeio-key-unicode-test",  # HTTP headers are ASCII-only
            "null_byte": "aegis\x00key",
            "very_long": "a" * 10000,
            "no_header": None,  # Special case: no auth header at all
        }

        error_messages: dict[str, str] = {}
        for name, key in test_keys.items():
            if key is None:
                resp = self._client.post(
                    "/v1/chat/completions",
                    json=self._chat_body(),
                )
            else:
                resp = self._client.post(
                    "/v1/chat/completions",
                    json=self._chat_body(),
                    headers={"Authorization": f"Bearer {key}"},
                )
            try:
                detail = resp.json().get("detail", resp.json().get("error", {}).get("message", ""))
            except Exception:
                detail = resp.text[:200]
            error_messages[name] = f"status={resp.status_code} detail={detail}"

        # Check if error messages differ (leaking format information)
        unique_messages = set(error_messages.values())
        info_leaked = len(unique_messages) > 2  # More than just "valid" vs "invalid"

        return FormatInferenceResult(
            error_messages=error_messages,
            info_leaked=info_leaked,
            recommendation=(
                "Standardize error messages to not leak key format information"
                if info_leaked else "Error messages are consistent"
            ),
        )

    def auth_bypass(self) -> AuthBypassResult:
        """Test 15 authentication bypass techniques."""
        body = self._chat_body()
        techniques: list[tuple[str, dict[str, str] | None]] = [
            ("no_auth_header", {}),
            ("empty_bearer", {"Authorization": "Bearer "}),
            ("bearer_with_spaces", {"Authorization": "Bearer  " + self._api_key}),
            ("lowercase_bearer", {"Authorization": "bearer " + self._api_key}),
            ("uppercase_bearer", {"Authorization": "BEARER " + self._api_key}),
            ("mixed_case_bearer", {"Authorization": "BeArEr " + self._api_key}),
            ("double_bearer", {"Authorization": "Bearer Bearer " + self._api_key}),
            ("basic_auth", {"Authorization": "Basic " + base64.b64encode(
                f"user:{self._api_key}".encode()).decode()}),
            ("x_api_key_header", {"X-API-Key": self._api_key}),
            ("cookie_auth", {"Cookie": f"api_key={self._api_key}"}),
            ("null_byte_in_token", {"Authorization": f"Bearer {self._api_key}\x00admin"}),
            ("newline_injection", {"Authorization": f"Bearer {self._api_key}\r\nX-Admin: true"}),
            ("oversized_token", {"Authorization": "Bearer " + "A" * 100000}),
            ("jwt_forged", {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhZG1pbiI6dHJ1ZX0.fake"}),
            ("sql_injection_token", {"Authorization": "Bearer ' OR 1=1 --"}),
        ]

        results: list[dict[str, Any]] = []
        bypasses_found = 0

        for technique_name, headers in techniques:
            resp = self._client.post(
                "/v1/chat/completions",
                json=body,
                headers=headers or {},
            )
            bypassed = resp.status_code not in (401, 403, 422)
            if bypassed:
                bypasses_found += 1
            results.append({
                "technique": technique_name,
                "status_code": resp.status_code,
                "bypassed": bypassed,
            })

        return AuthBypassResult(
            techniques_tested=len(techniques),
            bypasses_found=bypasses_found,
            results=results,
        )

    def run_all(self) -> list[AttackResult]:
        """Run all auth attacks and return results."""
        results: list[AttackResult] = []

        # Timing attack
        timing = self.timing_attack(iterations=20)
        results.append(AttackResult(
            attack_name="timing_attack_key_validation",
            category="authentication",
            vulnerable=timing.vulnerable,
            severity="high" if timing.vulnerable else "info",
            description="Timing side-channel on API key validation",
            evidence=[f"Mean times: {timing.mean_times}"],
            recommendation=timing.recommendation,
            details={"correlation": timing.correlation_coefficient},
        ))

        # Format inference
        fmt = self.key_format_inference()
        results.append(AttackResult(
            attack_name="key_format_inference",
            category="authentication",
            vulnerable=fmt.info_leaked,
            severity="medium" if fmt.info_leaked else "info",
            description="Error messages may leak API key format information",
            evidence=list(fmt.error_messages.values())[:5],
            recommendation=fmt.recommendation,
        ))

        # Bypass
        bypass = self.auth_bypass()
        results.append(AttackResult(
            attack_name="auth_bypass_techniques",
            category="authentication",
            vulnerable=bypass.bypasses_found > 0,
            severity="critical" if bypass.bypasses_found > 0 else "info",
            description=f"Tested {bypass.techniques_tested} bypass techniques",
            evidence=[r["technique"] for r in bypass.results if r["bypassed"]],
            recommendation="Fix all bypass techniques that returned non-401/403",
            details={"bypasses": bypass.bypasses_found, "results": bypass.results},
        ))

        return results
