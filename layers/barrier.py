"""
L1 — Barrier Layer (Skin / Mucous Membranes)

The outermost defense. Enforces structural rules only — does NOT inspect
content. Handles API key validation, sliding-window rate limiting, request
schema validation against the OpenAI API spec, request size enforcement,
token counting (tiktoken cl100k_base), and identity binding.

Target latency: <1ms for cached decisions. 10,000+ RPS per node.

ASSUMED-BREACH POSTURE: This layer assumes the network is hostile, the
client is adversarial, and TLS termination may be intercepted. API keys
may be stolen. Rate limit state may be tampered with. This layer's output
(RequestContext) should NOT be trusted by downstream layers — a compromised
Barrier could emit valid-looking contexts for invalid requests.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import tiktoken

from aegis.config import BarrierConfig
from aegis.models.request_context import (
    ChatMessage,
    RateLimitStatus,
    RequestContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sliding-window rate limiter (in-memory with optional Redis upgrade)
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Per-key sliding-window rate limiter.

    Tracks timestamps of requests in the current window. On each call,
    prunes expired entries and checks count against limit + burst.

    ASSUMED-BREACH POSTURE: An attacker rotating API keys or spoofing
    identifiers can bypass per-key limiting. In production, rate limiting
    also applies per source-IP as a secondary key.
    """

    def __init__(self, rpm: int, burst: int, window_seconds: int = 60):
        self._rpm = rpm
        self._burst = burst
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> RateLimitStatus:
        """Check rate limit for *key*. Returns status with remaining count."""
        now = time.monotonic()
        cutoff = now - self._window
        # Prune expired entries
        timestamps = self._requests[key]
        self._requests[key] = timestamps = [
            t for t in timestamps if t > cutoff
        ]
        effective_limit = self._rpm + self._burst
        remaining = max(0, effective_limit - len(timestamps))
        is_throttled = remaining == 0
        if not is_throttled:
            timestamps.append(now)
        return RateLimitStatus(
            requests_remaining=remaining,
            window_reset_at=datetime.now(timezone.utc),
            is_throttled=is_throttled,
        )


class RedisSlidingWindowRateLimiter:
    """Redis-backed sliding-window rate limiter using sorted sets.

    Each key gets a Redis sorted set ``aegis:ratelimit:{api_key_hash}``
    where members are unique request IDs scored by timestamp. On each
    check:
        1. ZREMRANGEBYSCORE to prune expired entries
        2. ZCARD to count current window
        3. If under limit, ZADD the new request
        4. EXPIRE to auto-clean idle keys

    Falls back to an in-memory ``SlidingWindowRateLimiter`` if Redis
    is unavailable. AEGIS never crashes due to Redis being down.
    """

    KEY_PREFIX = "aegis:ratelimit:"

    def __init__(
        self,
        redis_client: Any,
        rpm: int,
        burst: int,
        window_seconds: int = 60,
    ):
        self._redis = redis_client
        self._rpm = rpm
        self._burst = burst
        self._window = window_seconds
        # In-memory fallback for when Redis goes away mid-operation
        self._fallback = SlidingWindowRateLimiter(rpm, burst, window_seconds)

    async def check(self, key: str) -> RateLimitStatus:
        """Check rate limit using Redis sorted set.

        Falls back to in-memory if Redis operation fails.
        """
        try:
            return await self._check_redis(key)
        except Exception as e:
            logger.warning("Redis rate limit failed (%s) — using in-memory fallback", e)
            return self._fallback.check(key)

    # Lua script for atomic rate limiting: prune + count + conditionally add
    _LUA_RATE_LIMIT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
local ttl = tonumber(ARGV[5])
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, ttl)
    return count
else
    return -1
end
"""

    async def _check_redis(self, key: str) -> RateLimitStatus:
        redis_key = f"{self.KEY_PREFIX}{key}"
        now = time.time()
        cutoff = now - self._window
        effective_limit = self._rpm + self._burst
        member = f"{now}:{uuid.uuid4().hex[:8]}"
        ttl = self._window + 10

        result = await self._redis.eval(
            self._LUA_RATE_LIMIT, 1, redis_key,
            str(now), str(cutoff), str(effective_limit), member, str(ttl),
        )

        if result == -1:
            # Over limit
            return RateLimitStatus(
                requests_remaining=0,
                window_reset_at=datetime.now(timezone.utc),
                is_throttled=True,
            )
        else:
            count = int(result)
            remaining = max(0, effective_limit - count - 1)  # -1 because we just added
            return RateLimitStatus(
                requests_remaining=remaining,
                window_reset_at=datetime.now(timezone.utc),
                is_throttled=False,
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_VALID_ROLES = {"system", "user", "assistant", "tool", "function"}
_REQUIRED_CHAT_FIELDS = {"model", "messages"}
_ALLOWED_CHAT_FIELDS = {
    "model", "messages", "stream", "temperature", "top_p", "n",
    "max_tokens", "max_completion_tokens", "presence_penalty",
    "frequency_penalty", "logit_bias", "user", "stop", "seed",
    "tools", "tool_choice", "response_format", "logprobs",
    "top_logprobs", "parallel_tool_calls", "service_tier",
    "store", "metadata", "stream_options",
}


def validate_chat_schema(body: dict[str, Any]) -> list[str]:
    """Validate request body against the OpenAI Chat Completions schema.

    Returns a list of violation descriptions (empty = valid).
    """
    violations: list[str] = []

    # Required fields
    for field in _REQUIRED_CHAT_FIELDS:
        if field not in body:
            violations.append(f"missing required field: {field}")

    # Unexpected fields
    extra = set(body.keys()) - _ALLOWED_CHAT_FIELDS
    if extra:
        violations.append(f"unexpected fields: {sorted(extra)}")

    # Messages must be a list
    messages = body.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            violations.append("messages must be a list")
        elif len(messages) == 0:
            violations.append("messages must not be empty")
        else:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    violations.append(f"messages[{i}] must be an object")
                    continue
                role = msg.get("role")
                if role not in _VALID_ROLES:
                    violations.append(
                        f"messages[{i}].role invalid: {role!r}"
                    )
                if "content" not in msg and "tool_calls" not in msg:
                    violations.append(
                        f"messages[{i}] must have content or tool_calls"
                    )

    # Model must be a string
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        violations.append("model must be a string")

    # Temperature range
    temp = body.get("temperature")
    if temp is not None and (not isinstance(temp, (int, float)) or temp < 0 or temp > 2):
        violations.append("temperature must be a number between 0 and 2")

    return violations


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

# Lazy singleton — tiktoken encoding is expensive to load.
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(messages: list[ChatMessage]) -> int:
    """Count tokens across all messages using cl100k_base."""
    enc = _get_encoding()
    total = 0
    for msg in messages:
        # Per OpenAI: every message has overhead tokens
        total += 4  # <|im_start|>{role}\n ... <|im_end|>\n
        if msg.content:
            total += len(enc.encode(msg.content))
        if msg.name:
            total += len(enc.encode(msg.name)) + 1
    total += 2  # reply priming
    return total


# ---------------------------------------------------------------------------
# Barrier Layer
# ---------------------------------------------------------------------------

class BarrierLayer:
    """L1 Barrier Layer — the outermost immune defense.

    Processes raw request dicts into validated RequestContext objects,
    enforcing authentication, rate limits, schema, size, and token limits.

    When a TenantManager is provided, per-tenant rate limits, token limits,
    and allowed model lists are enforced. Without a TenantManager, global
    defaults from BarrierConfig are used (backward compatible).

    Usage:
        barrier = BarrierLayer(config, valid_api_keys={"aegis-key-1"})
        ctx_or_error = await barrier.process(body, headers, source_ip)
    """

    def __init__(
        self,
        config: BarrierConfig,
        valid_api_keys: set[str] | None = None,
        redis_client: Any | None = None,
        tenant_manager: Any | None = None,
    ):
        self._config = config
        self._valid_keys = valid_api_keys or set()
        self._tenant_manager = tenant_manager
        self._redis_client = redis_client
        if redis_client is not None:
            self._limiter: SlidingWindowRateLimiter | RedisSlidingWindowRateLimiter = (
                RedisSlidingWindowRateLimiter(
                    redis_client=redis_client,
                    rpm=config.rate_limit_rpm,
                    burst=config.rate_limit_burst,
                )
            )
            self._use_redis_limiter = True
        else:
            self._limiter = SlidingWindowRateLimiter(
                rpm=config.rate_limit_rpm,
                burst=config.rate_limit_burst,
            )
            self._use_redis_limiter = False
        # Per-tenant limiters (created on demand)
        self._tenant_limiters: dict[str, SlidingWindowRateLimiter | RedisSlidingWindowRateLimiter] = {}

    @property
    def tenant_manager(self) -> Any | None:
        return self._tenant_manager

    @tenant_manager.setter
    def tenant_manager(self, value: Any | None) -> None:
        self._tenant_manager = value

    def _get_tenant_limiter(
        self, tenant_id: str, rpm: int, burst: int,
    ) -> SlidingWindowRateLimiter | RedisSlidingWindowRateLimiter:
        """Get or create a rate limiter for a specific tenant.

        Uses setdefault() to avoid check-then-act race under concurrent access.
        """
        key = f"{tenant_id}:{rpm}:{burst}"
        if key in self._tenant_limiters:
            return self._tenant_limiters[key]
        if self._redis_client is not None:
            new_limiter = RedisSlidingWindowRateLimiter(
                redis_client=self._redis_client,
                rpm=rpm,
                burst=burst,
            )
        else:
            new_limiter = SlidingWindowRateLimiter(
                rpm=rpm,
                burst=burst,
            )
        return self._tenant_limiters.setdefault(key, new_limiter)

    async def process(
        self,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
        source_ip: str = "0.0.0.0",
        raw_body: bytes = b"",
    ) -> RequestContext:
        """Validate and convert a raw request into a RequestContext.

        Raises BarrierReject on any structural violation.
        """
        headers = headers or {}

        # --- Authentication ---
        api_key = self._extract_api_key(headers)
        if not api_key:
            raise BarrierReject(401, "Missing API key")
        if self._valid_keys and api_key not in self._valid_keys:
            raise BarrierReject(401, "Invalid API key")
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # --- Tenant resolution ---
        tenant_config = None
        tenant_id = self._tenant_from_key(api_key)
        if self._tenant_manager:
            try:
                tenant_config = await self._tenant_manager.resolve_tenant(api_key_hash)
                if tenant_config:
                    tenant_id = tenant_config.tenant_id
            except Exception as e:
                logger.warning("Tenant resolution failed (%s) — using defaults", e)

        # --- Size enforcement ---
        body_size = len(raw_body) if raw_body else 0
        if body_size > self._config.max_request_size_bytes:
            raise BarrierReject(
                413,
                f"Request body {body_size} bytes exceeds "
                f"{self._config.max_request_size_bytes} byte limit",
            )

        # --- Schema validation ---
        violations = validate_chat_schema(body)
        if violations:
            raise BarrierReject(400, f"Schema violation: {'; '.join(violations)}")

        # --- Allowed model check (per-tenant) ---
        requested_model = body.get("model", "")
        if tenant_config and tenant_config.allowed_models and requested_model:
            if requested_model not in tenant_config.allowed_models:
                raise BarrierReject(
                    403,
                    f"Model '{requested_model}' is not authorized for tenant '{tenant_config.name}'",
                )

        # --- Parse messages ---
        messages = [
            ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content"),
                name=m.get("name"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in body.get("messages", [])
        ]

        # --- Token counting ---
        max_tokens = (
            tenant_config.max_tokens_per_request
            if tenant_config
            else self._config.max_tokens_per_request
        )
        token_count = count_tokens(messages)
        if token_count > max_tokens:
            raise BarrierReject(
                400,
                f"Token count {token_count} exceeds "
                f"{max_tokens} limit",
            )

        # --- Rate limiting (per-tenant or global) ---
        if tenant_config:
            limiter = self._get_tenant_limiter(
                tenant_config.tenant_id,
                tenant_config.rate_limit_rpm,
                tenant_config.rate_limit_burst,
            )
        else:
            limiter = self._limiter

        if isinstance(limiter, RedisSlidingWindowRateLimiter):
            rate_status = await limiter.check(api_key_hash)
        else:
            rate_status = limiter.check(api_key_hash)
        if rate_status.is_throttled:
            raise BarrierReject(429, "Rate limit exceeded")

        # --- Identity binding ---
        session_id = (
            headers.get("x-session-id")
            or body.get("metadata", {}).get("session_id", "")  # type: ignore[union-attr]
            or str(uuid.uuid4())
        )
        user_id = body.get("user", "anonymous")
        if not isinstance(user_id, str):
            user_id = str(user_id)

        return RequestContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            api_key_hash=api_key_hash,
            source_ip=source_ip,
            model=body.get("model", ""),
            messages=messages,
            stream=bool(body.get("stream", False)),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            raw_body=raw_body,
            rate_limit=rate_status,
            metadata={"token_count": token_count},
        )

    # --- Private helpers ---

    @staticmethod
    def _extract_api_key(headers: dict[str, str]) -> str | None:
        """Extract API key from Authorization header (Bearer token)."""
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        # Also accept x-api-key header
        return headers.get("x-api-key") or headers.get("X-Api-Key") or None

    @staticmethod
    def _tenant_from_key(api_key: str) -> str:
        """Derive tenant identifier from API key prefix.

        Convention: keys are formatted as "aegis-{tenant}-{secret}".
        Falls back to "default" if the key doesn't follow the convention.
        """
        parts = api_key.split("-", 2)
        if len(parts) >= 2 and parts[0] == "aegis":
            return parts[1]
        return "default"


class BarrierReject(Exception):
    """Raised when L1 Barrier rejects a request.

    Contains HTTP status code and reason for the rejection.
    """

    def __init__(self, status_code: int, reason: str):
        self.status_code = status_code
        self.reason = reason
        super().__init__(f"L1 Barrier rejected ({status_code}): {reason}")
