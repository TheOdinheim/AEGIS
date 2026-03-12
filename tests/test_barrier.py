"""
L1 Barrier Layer Tests

Validates authentication, sliding-window rate limiting, schema validation,
size enforcement, token counting, and identity binding.

Covers attack battery items:
    #15 — Rate Limit Bypass (burst of 100 requests in 1 second)
    #19 — Schema Injection (extra fields in JSON request)
"""

from __future__ import annotations

import asyncio
import pytest

from aegis.config import BarrierConfig
from aegis.layers.barrier import BarrierLayer, BarrierReject


@pytest.fixture
def config() -> BarrierConfig:
    return BarrierConfig(
        max_tokens_per_request=1000,
        rate_limit_rpm=10,
        rate_limit_burst=2,
        max_request_size_bytes=1024,
    )


@pytest.fixture
def barrier(config: BarrierConfig) -> BarrierLayer:
    return BarrierLayer(config, valid_api_keys={"aegis-testco-key123"})


def _headers(key: str = "aegis-testco-key123") -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


def _body(**overrides) -> dict:
    base = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    base.update(overrides)
    return base


# ---- Authentication ----

class TestAuthentication:
    def test_missing_api_key_rejected(self, barrier: BarrierLayer):
        with pytest.raises(BarrierReject, match="Missing API key"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(_body(), headers={})
            )

    def test_invalid_api_key_rejected(self, barrier: BarrierLayer):
        with pytest.raises(BarrierReject, match="Invalid API key"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(_body(), headers=_headers("bad-key"))
            )

    def test_valid_api_key_accepted(self, barrier: BarrierLayer):
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(_body(), headers=_headers())
        )
        assert ctx.api_key_hash != ""
        assert ctx.tenant_id == "testco"

    def test_x_api_key_header(self, barrier: BarrierLayer):
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(_body(), headers={"x-api-key": "aegis-testco-key123"})
        )
        assert ctx.tenant_id == "testco"


# ---- Schema Validation ----

class TestSchemaValidation:
    def test_missing_model_rejected(self, barrier: BarrierLayer):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(BarrierReject, match="missing required field: model"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(body, headers=_headers())
            )

    def test_missing_messages_rejected(self, barrier: BarrierLayer):
        body = {"model": "gpt-4"}
        with pytest.raises(BarrierReject, match="missing required field: messages"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(body, headers=_headers())
            )

    def test_empty_messages_rejected(self, barrier: BarrierLayer):
        body = {"model": "gpt-4", "messages": []}
        with pytest.raises(BarrierReject, match="must not be empty"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(body, headers=_headers())
            )

    def test_invalid_role_rejected(self, barrier: BarrierLayer):
        body = {"model": "gpt-4", "messages": [{"role": "hacker", "content": "hi"}]}
        with pytest.raises(BarrierReject, match="role invalid"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(body, headers=_headers())
            )

    def test_attack_19_schema_injection(self, barrier: BarrierLayer):
        """Attack #19: Extra fields in JSON request."""
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "malicious_field": "payload",
            "exploit": True,
        }
        with pytest.raises(BarrierReject, match="unexpected fields"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(body, headers=_headers())
            )

    def test_valid_optional_fields_accepted(self, barrier: BarrierLayer):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "stream": True,
        }
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(body, headers=_headers())
        )
        assert ctx.stream is True
        assert ctx.temperature == 0.7


# ---- Size Enforcement ----

class TestSizeEnforcement:
    def test_oversized_body_rejected(self, barrier: BarrierLayer):
        body = _body()
        raw = b"x" * 2048  # exceeds 1024 limit
        with pytest.raises(BarrierReject, match="exceeds"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(body, headers=_headers(), raw_body=raw)
            )


# ---- Token Counting ----

class TestTokenCounting:
    def test_token_overflow_rejected(self, barrier: BarrierLayer):
        # Generate a message with many tokens (config max is 1000)
        long_content = "word " * 2000  # ~2000 tokens, well over 1000 limit
        body = _body(messages=[{"role": "user", "content": long_content}])
        with pytest.raises(BarrierReject, match="Token count"):
            asyncio.get_event_loop().run_until_complete(
                barrier.process(body, headers=_headers())
            )

    def test_normal_tokens_accepted(self, barrier: BarrierLayer):
        body = _body(messages=[{"role": "user", "content": "Hello world"}])
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(body, headers=_headers())
        )
        assert ctx.metadata["token_count"] > 0
        assert ctx.metadata["token_count"] < 100


# ---- Rate Limiting ----

class TestRateLimiting:
    def test_attack_15_rate_limit_bypass(self, barrier: BarrierLayer):
        """Attack #15: Burst of requests exceeding rate limit."""
        body = _body()
        headers = _headers()
        # rpm=10, burst=2 → 12 total allowed in window
        allowed = 0
        rejected = 0
        for _ in range(20):
            try:
                asyncio.get_event_loop().run_until_complete(
                    barrier.process(body, headers=headers)
                )
                allowed += 1
            except BarrierReject as e:
                if e.status_code == 429:
                    rejected += 1
                else:
                    raise

        assert allowed == 12  # rpm + burst
        assert rejected == 8  # remaining rejected


# ---- Identity Binding ----

class TestIdentityBinding:
    def test_tenant_extracted_from_key(self, barrier: BarrierLayer):
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(_body(), headers=_headers())
        )
        assert ctx.tenant_id == "testco"

    def test_session_id_from_header(self, barrier: BarrierLayer):
        headers = {**_headers(), "x-session-id": "session-abc"}
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(_body(), headers=headers)
        )
        assert ctx.session_id == "session-abc"

    def test_source_ip_captured(self, barrier: BarrierLayer):
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(_body(), headers=_headers(), source_ip="1.2.3.4")
        )
        assert ctx.source_ip == "1.2.3.4"

    def test_user_id_from_body(self, barrier: BarrierLayer):
        body = _body(user="user-42")
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(body, headers=_headers())
        )
        assert ctx.user_id == "user-42"

    def test_model_captured(self, barrier: BarrierLayer):
        ctx = asyncio.get_event_loop().run_until_complete(
            barrier.process(_body(model="gpt-4-turbo"), headers=_headers())
        )
        assert ctx.model == "gpt-4-turbo"
