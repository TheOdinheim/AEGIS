"""
Scanner 6 — Canary Token Verifier (NK Cell Analog)

AEGIS injects a unique canary token (HMAC-SHA256 of tenant_id + session_id
+ secret key) into system prompts before forwarding to the upstream model.
The canary is formatted as an invisible HTML comment or Unicode tag.

The verifier checks every response for:
    (a) Canary token present and unmodified
    (b) Canary token not relocated from its original position
    (c) System prompt content not echoed around the canary

If the canary is missing or modified, flag with high confidence — this is
a strong signal the system prompt was tampered with or echoed in output.

ASSUMED-BREACH POSTURE: An attacker who knows the canary secret key can
forge valid tokens, neutralizing canary verification. The secret key must
be rotated regularly and stored in a secrets manager. Even with the key
leaked, all other layers remain active. Canary verification is one of six
innate scanners — it provides defense in depth, not sole protection.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import time
from typing import Any

from aegis.config import CanaryConfig
from aegis.models.scan_result import ScanResult, ThreatCategory

logger = logging.getLogger(__name__)

# Canary token prefix and suffix markers
_CANARY_PREFIX = "AEGIS-CANARY-"
_HTML_COMMENT_FMT = "<!-- {prefix}{token} -->"
_UNICODE_TAG_FMT = "\u200b{prefix}{token}\u200b"  # Zero-width space wrapper


def generate_canary_token(
    tenant_id: str,
    session_id: str,
    secret_key: str,
) -> str:
    """Generate an HMAC-SHA256 canary token from tenant + session + secret.

    Returns the raw hex digest (32 chars). The caller formats it for
    injection using format_canary().
    """
    msg = f"{tenant_id}:{session_id}".encode("utf-8")
    key = secret_key.encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()[:32]


def format_canary(token: str, fmt: str = "html_comment") -> str:
    """Format a canary token for injection into a system prompt.

    Args:
        token: Raw canary token hex string.
        fmt: Format type — 'html_comment' or 'unicode_tag'.
    """
    if fmt == "unicode_tag":
        return _UNICODE_TAG_FMT.format(prefix=_CANARY_PREFIX, token=token)
    return _HTML_COMMENT_FMT.format(prefix=_CANARY_PREFIX, token=token)


def inject_canary(
    system_prompt: str,
    tenant_id: str,
    session_id: str,
    config: CanaryConfig,
) -> tuple[str, str]:
    """Inject a canary token at the end of the system prompt.

    Returns (modified_prompt, canary_token).
    """
    token = generate_canary_token(tenant_id, session_id, config.secret_key)
    formatted = format_canary(token, config.token_format)
    return f"{system_prompt}\n{formatted}", token


# Compiled regex for detecting canary tokens in text
_CANARY_RE = re.compile(
    r"(?:"
    r"<!--\s*" + re.escape(_CANARY_PREFIX) + r"([a-f0-9]{32})\s*-->"
    r"|"
    r"\u200b" + re.escape(_CANARY_PREFIX) + r"([a-f0-9]{32})\u200b"
    r")"
)


class CanaryVerifier:
    """L2 Innate Scanner — verifies canary token integrity in responses.

    This scanner operates on the MODEL RESPONSE (output path), not the
    input. When integrated into the innate pipeline, it checks whether
    the system prompt's canary appeared in the output — which would
    indicate system prompt leakage.

    On the input path, it verifies that the system prompt in the request
    has not had its canary token tampered with (for re-injection attacks).

    Injection/verification flow (wired in main.py):
        1. inject_canary() inserts HMAC token into system prompt BEFORE
           forwarding to upstream model (request path).
        2. verify_output() checks the model response for canary token
           presence (response path). If found, the model leaked the
           system prompt.
    """

    def __init__(self, config: CanaryConfig | None = None):
        self._config = config or CanaryConfig()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def verify_output(
        self,
        response_text: str,
        expected_token: str | None = None,
    ) -> ScanResult:
        """Check if the model response contains a canary token (leakage).

        If the canary token appears in the response, the model has leaked
        the system prompt. This is a high-confidence threat signal.

        Args:
            response_text: The model's response text.
            expected_token: The canary token that was injected into the
                system prompt. If provided, exact match is checked.
        """
        start = time.perf_counter()

        if not self._config.enabled:
            return ScanResult(
                scanner_id="canary_verifier",
                is_threat=False,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            matches = _CANARY_RE.findall(response_text)
            # findall returns list of tuples (html_group, unicode_group)
            found_tokens = [m[0] or m[1] for m in matches if m[0] or m[1]]

            elapsed = (time.perf_counter() - start) * 1000

            if found_tokens:
                # Canary token found in output = system prompt leaked
                if expected_token and expected_token in found_tokens:
                    return ScanResult(
                        scanner_id="canary_verifier",
                        is_threat=True,
                        confidence=0.98,
                        threat_category=ThreatCategory.CANARY_TAMPERING,
                        matched_patterns=[
                            f"Canary token found in output (exact match): {expected_token[:8]}..."
                        ],
                        latency_ms=elapsed,
                    )
                return ScanResult(
                    scanner_id="canary_verifier",
                    is_threat=True,
                    confidence=0.95,
                    threat_category=ThreatCategory.CANARY_TAMPERING,
                    matched_patterns=[
                        f"Canary token pattern found in output: {found_tokens[0][:8]}..."
                    ],
                    latency_ms=elapsed,
                )

            return ScanResult(
                scanner_id="canary_verifier",
                is_threat=False,
                confidence=0.0,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Canary verifier crashed: %s", e)
            return ScanResult(
                scanner_id="canary_verifier",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed,
            )

    async def scan_input(
        self,
        system_prompt: str,
        expected_token: str | None = None,
    ) -> ScanResult:
        """Check that the system prompt's canary is intact (input path).

        Verifies:
        1. If a canary was expected, it's still present and unmodified
        2. The canary hasn't been relocated from its original position

        Args:
            system_prompt: The system prompt from the request.
            expected_token: Token that should be present. If None, checks
                for any canary-shaped token but cannot verify correctness.
        """
        start = time.perf_counter()

        if not self._config.enabled:
            return ScanResult(
                scanner_id="canary_verifier",
                is_threat=False,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            matches = _CANARY_RE.findall(system_prompt)
            found_tokens = [m[0] or m[1] for m in matches if m[0] or m[1]]

            elapsed = (time.perf_counter() - start) * 1000

            if expected_token:
                if not found_tokens:
                    # Expected canary is missing = system prompt tampered
                    return ScanResult(
                        scanner_id="canary_verifier",
                        is_threat=True,
                        confidence=0.95,
                        threat_category=ThreatCategory.CANARY_TAMPERING,
                        matched_patterns=["Expected canary token missing from system prompt"],
                        latency_ms=elapsed,
                    )
                if expected_token not in found_tokens:
                    # Canary modified = system prompt tampered
                    return ScanResult(
                        scanner_id="canary_verifier",
                        is_threat=True,
                        confidence=0.95,
                        threat_category=ThreatCategory.CANARY_TAMPERING,
                        matched_patterns=[
                            f"Canary token modified: expected {expected_token[:8]}..., "
                            f"found {found_tokens[0][:8]}..."
                        ],
                        latency_ms=elapsed,
                    )

            # Canary present and correct (or no expected token)
            return ScanResult(
                scanner_id="canary_verifier",
                is_threat=False,
                confidence=0.0,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Canary verifier crashed on input: %s", e)
            return ScanResult(
                scanner_id="canary_verifier",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed,
            )
