"""
RequestContext — The canonical data object flowing through all AEGIS layers.

Built by L1 Barrier from the raw HTTP request. Contains validated identity
metadata, the original payload, rate limit status, and timestamps. Every
downstream layer receives this object and uses it for its analysis.

ASSUMED-BREACH POSTURE: The RequestContext is constructed from a potentially
adversarial HTTP request. Fields like tenant_id and user_id are derived from
authenticated tokens, but those tokens may be stolen. The session_id may be
replayed. The raw messages content is UNTRUSTED and must be independently
validated by every layer that consumes it. A compromised L1 could forge any
field in this context — downstream layers must not treat RequestContext
fields as proof of safety, only as routing metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the OpenAI chat completions format."""
    role: str = Field(description="Message role: system, user, assistant, tool")
    content: str | None = Field(default=None, description="Message content")
    name: str | None = Field(default=None, description="Optional sender name")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls (assistant role)"
    )
    tool_call_id: str | None = Field(
        default=None, description="Tool call ID (tool role)"
    )


class RateLimitStatus(BaseModel):
    """Current rate limit state for this API key at time of request."""
    requests_remaining: int = Field(description="Requests remaining in current window")
    window_reset_at: datetime = Field(description="When the current window resets")
    is_throttled: bool = Field(
        default=False, description="Whether this request is being throttled"
    )


class RequestContext(BaseModel):
    """Canonical request object flowing through the entire AEGIS pipeline.

    Created by L1 Barrier. Consumed by L2-L7. Immutable after construction —
    layers append their analysis results to separate report objects, never
    mutate the RequestContext.

    Fields:
        request_id: Unique identifier for this request (UUID4).
        timestamp: UTC timestamp when the request entered AEGIS.
        tenant_id: Tenant identifier from authenticated API key.
        user_id: User identifier (may be anonymous).
        session_id: Session identifier for multi-turn behavioral tracking.
        api_key_hash: SHA-256 hash of the API key (never store raw keys).
        source_ip: Client IP address (may be behind proxy).
        model: Requested model identifier (e.g., "gpt-4").
        messages: The chat messages array from the request body.
        stream: Whether the client requested streaming response.
        temperature: Sampling temperature from the request.
        max_tokens: Max tokens requested for the response.
        raw_body: The complete raw request body for forensic analysis.
        rate_limit: Current rate limit state for this key.
        metadata: Arbitrary metadata for downstream layers.
    """
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier (UUID4)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when request entered AEGIS",
    )

    # --- Identity (from authenticated tokens — may be stolen) ---
    tenant_id: str = Field(default="default", description="Tenant identifier")
    user_id: str = Field(default="anonymous", description="User identifier")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for multi-turn tracking",
    )
    api_key_hash: str = Field(default="", description="SHA-256 hash of the API key")
    source_ip: str = Field(default="0.0.0.0", description="Client IP address")

    # --- Request payload (UNTRUSTED) ---
    model: str = Field(default="", description="Requested model identifier")
    messages: list[ChatMessage] = Field(
        default_factory=list, description="Chat messages array"
    )
    stream: bool = Field(default=False, description="Streaming response requested")
    temperature: float | None = Field(default=None, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Max response tokens")
    raw_body: bytes = Field(
        default=b"", description="Complete raw request body for forensic logging"
    )

    # --- Rate limit state ---
    rate_limit: RateLimitStatus | None = Field(
        default=None, description="Rate limit status at time of request"
    )

    # --- Agent identity ---
    agent_trust_level: float = Field(
        default=1.0,
        description="Agent trust level (1.0 for non-agent requests, 0.0-1.0 for agents)",
    )

    # --- Extensibility ---
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for downstream layer use",
    )

    @property
    def prompt_text(self) -> str:
        """Extract the concatenated user-visible prompt text for scanning.

        Combines all message contents for layers that need the full text.
        System messages are included because they can be injection targets.
        """
        parts = []
        for msg in self.messages:
            if msg.content:
                parts.append(msg.content)
        return "\n".join(parts)

    @property
    def last_user_message(self) -> str | None:
        """Extract the most recent user message content."""
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return None
