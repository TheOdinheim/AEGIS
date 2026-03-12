"""
Canary Token Verifier Tests

Tests canary token generation, injection, and verification:
    - Token generation is deterministic for same inputs
    - Token injection adds canary to system prompt
    - Token present in output = leakage detected
    - Token tampered in output = detected
    - Token missing from output = no false positive
    - Token missing from system prompt when expected = tampering detected
"""

from __future__ import annotations

import pytest

from aegis.config import CanaryConfig
from aegis.layers.innate.canary_verifier import (
    CanaryVerifier,
    format_canary,
    generate_canary_token,
    inject_canary,
)


class TestCanaryTokenGeneration:
    def test_deterministic(self):
        """Same inputs produce same token."""
        t1 = generate_canary_token("tenant-1", "session-1", "secret-key")
        t2 = generate_canary_token("tenant-1", "session-1", "secret-key")
        assert t1 == t2
        assert len(t1) == 32

    def test_different_tenant_different_token(self):
        t1 = generate_canary_token("tenant-1", "session-1", "secret-key")
        t2 = generate_canary_token("tenant-2", "session-1", "secret-key")
        assert t1 != t2

    def test_different_session_different_token(self):
        t1 = generate_canary_token("tenant-1", "session-1", "secret-key")
        t2 = generate_canary_token("tenant-1", "session-2", "secret-key")
        assert t1 != t2

    def test_different_secret_different_token(self):
        t1 = generate_canary_token("tenant-1", "session-1", "secret-1")
        t2 = generate_canary_token("tenant-1", "session-1", "secret-2")
        assert t1 != t2


class TestCanaryFormatting:
    def test_html_comment_format(self):
        token = "a" * 32
        formatted = format_canary(token, "html_comment")
        assert formatted == f"<!-- AEGIS-CANARY-{'a' * 32} -->"

    def test_unicode_tag_format(self):
        token = "b" * 32
        formatted = format_canary(token, "unicode_tag")
        assert "\u200b" in formatted
        assert "AEGIS-CANARY-" in formatted


class TestCanaryInjection:
    def test_inject_into_system_prompt(self):
        config = CanaryConfig(secret_key="test-secret")
        prompt = "You are a helpful assistant."
        modified, token = inject_canary(prompt, "t1", "s1", config)
        assert prompt in modified
        assert "AEGIS-CANARY-" in modified
        assert token in modified
        assert len(token) == 32

    def test_inject_preserves_original_prompt(self):
        config = CanaryConfig(secret_key="test-secret")
        prompt = "Be helpful and accurate."
        modified, token = inject_canary(prompt, "t1", "s1", config)
        assert modified.startswith(prompt)


class TestCanaryVerifierOutput:
    @pytest.fixture
    def verifier(self) -> CanaryVerifier:
        return CanaryVerifier(CanaryConfig(secret_key="test-secret"))

    def test_canary_in_output_detected(self, verifier: CanaryVerifier):
        """Model output containing canary = system prompt leaked."""
        token = generate_canary_token("t1", "s1", "test-secret")
        response = f"Here is my system prompt: <!-- AEGIS-CANARY-{token} -->"
        result = verifier.verify_output(response, expected_token=token)
        assert result.is_threat
        assert result.confidence >= 0.95
        assert result.threat_category.value == "AEGIS.CANARY"

    def test_canary_absent_from_output_is_safe(self, verifier: CanaryVerifier):
        """Normal model output without canary should not trigger."""
        result = verifier.verify_output(
            "The capital of France is Paris.",
            expected_token="a" * 32,
        )
        assert not result.is_threat
        assert result.confidence == 0.0

    def test_tampered_canary_in_output_detected(self, verifier: CanaryVerifier):
        """Output containing a different canary token should be detected."""
        expected = "a" * 32
        tampered = "b" * 32
        response = f"<!-- AEGIS-CANARY-{tampered} -->"
        result = verifier.verify_output(response, expected_token=expected)
        # Still detected as leakage (any canary pattern in output is suspicious)
        assert result.is_threat
        assert result.confidence >= 0.95

    def test_unicode_canary_in_output_detected(self, verifier: CanaryVerifier):
        """Unicode-formatted canary in output should be detected."""
        token = "c" * 32
        response = f"Some text \u200bAEGIS-CANARY-{token}\u200b more text"
        result = verifier.verify_output(response, expected_token=token)
        assert result.is_threat

    def test_disabled_verifier(self):
        """Disabled verifier should not flag anything."""
        verifier = CanaryVerifier(CanaryConfig(enabled=False))
        token = "a" * 32
        response = f"<!-- AEGIS-CANARY-{token} -->"
        result = verifier.verify_output(response, expected_token=token)
        assert not result.is_threat


class TestCanaryVerifierInput:
    @pytest.fixture
    def verifier(self) -> CanaryVerifier:
        return CanaryVerifier(CanaryConfig(secret_key="test-secret"))

    def test_canary_present_and_correct(self, verifier: CanaryVerifier):
        """System prompt with correct canary should pass."""
        import asyncio
        token = "d" * 32
        prompt = f"You are helpful.\n<!-- AEGIS-CANARY-{token} -->"
        result = asyncio.get_event_loop().run_until_complete(
            verifier.scan_input(prompt, expected_token=token)
        )
        assert not result.is_threat

    def test_canary_missing_when_expected(self, verifier: CanaryVerifier):
        """System prompt without expected canary = tampering detected."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            verifier.scan_input("You are helpful.", expected_token="e" * 32)
        )
        assert result.is_threat
        assert result.confidence >= 0.95
        assert "missing" in result.matched_patterns[0].lower()

    def test_canary_modified_when_expected(self, verifier: CanaryVerifier):
        """System prompt with wrong canary = tampering detected."""
        import asyncio
        expected = "f" * 32
        wrong = "0" * 32
        prompt = f"You are helpful.\n<!-- AEGIS-CANARY-{wrong} -->"
        result = asyncio.get_event_loop().run_until_complete(
            verifier.scan_input(prompt, expected_token=expected)
        )
        assert result.is_threat
        assert "modified" in result.matched_patterns[0].lower()
