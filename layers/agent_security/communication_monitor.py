"""
Inter-Agent Communication Monitor (IACM) — Extension 5.3

Monitors inter-agent traffic for injection patterns, compromised agent
behavior, and tool-mediated indirect communication attacks.

Extends the existing AgentMessageValidator with behavioral tracking,
multi-recipient analysis, and tool-mediated detection. A compromised agent
can systematically attack every other agent it communicates with — the IACM
detects this lateral movement pattern.

ASSUMED-BREACH POSTURE: Any agent in a multi-agent system can be compromised.
The IACM applies the same detection pipeline (L2 pattern matching) to
inter-agent communications that AEGIS applies to user-to-agent traffic.
A compromised agent sending injection-laden messages to multiple recipients
is detected and its trust is automatically reduced.
"""

from __future__ import annotations

import enum
import logging
import re
import time
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Tool-mediated communication patterns
_AGENT_DIRECTED_RE = re.compile(
    r"\b(?:you\s+should|your\s+next\s+step|execute\s+this|run\s+this|"
    r"call\s+the\s+\w+\s+tool|invoke\s+\w+|your\s+task\s+is)\b",
    re.IGNORECASE,
)


class MessageVerdict(str, enum.Enum):
    CLEAN = "clean"
    FLAGGED = "flagged"
    BLOCKED = "blocked"


@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    receiver_id: str
    content: str
    timestamp: float
    channel: str = "direct"  # direct, broadcast, tool_mediated
    metadata: dict = field(default_factory=dict)


@dataclass
class MessageScanResult:
    message_id: str
    verdict: MessageVerdict
    injection_detected: bool
    injection_score: float
    sender_trust_level: float
    behavioral_flags: list[str] = field(default_factory=list)
    scan_latency_ms: float = 0.0


@dataclass
class SenderProfile:
    sender_id: str
    total_messages: int = 0
    injection_count: int = 0
    unique_recipients: set = field(default_factory=set)
    last_activity: float = 0.0
    flagged_as_compromised: bool = False


class InterAgentCommunicationMonitor:
    """Monitors inter-agent communications for injection and compromise.

    Thread-safe. Maintains per-sender behavioral profiles and detects
    lateral movement patterns.
    """

    def __init__(
        self,
        regex_engine: Any = None,
        identity_manager: Any = None,
        trust_threshold: float = 0.3,
        injection_rate_threshold: float = 0.5,
        event_bus: Any = None,
        max_profiles: int = 10000,
    ):
        self._regex_engine = regex_engine
        self._identity_manager = identity_manager
        self._trust_threshold = trust_threshold
        self._injection_rate_threshold = injection_rate_threshold
        self._event_bus = event_bus
        self._max_profiles = max_profiles

        self._lock = threading.Lock()
        self._sender_profiles: dict[str, SenderProfile] = {}
        self._compromised_agents: set[str] = set()

        # Stats
        self._total_scanned = 0
        self._total_injections = 0
        self._total_compromised = 0

    async def scan_message(
        self,
        message: AgentMessage,
        sender_trust_level: float = 1.0,
    ) -> MessageScanResult:
        """Scan an inter-agent message for injection and behavioral anomalies.

        Steps:
        1. Check sender trust level for elevated scrutiny
        2. Content scanning via L2 regex engine
        3. Behavioral context (per-sender injection rate)
        4. Tool-mediated communication detection
        """
        start = time.perf_counter()

        injection_detected = False
        injection_score = 0.0
        behavioral_flags: list[str] = []

        # Track this message in sender profile
        profile = self._get_or_create_profile(message.sender_id)
        with self._lock:
            profile.total_messages += 1
            profile.unique_recipients.add(message.receiver_id)
            profile.last_activity = message.timestamp

        # 1. Sender trust check
        elevated_scrutiny = sender_trust_level < self._trust_threshold
        if elevated_scrutiny:
            behavioral_flags.append(
                f"Low sender trust: {sender_trust_level:.2f}"
            )

        # 2. Content scanning via L2 regex engine
        if self._regex_engine and message.content.strip():
            try:
                scan_result = await self._regex_engine.scan(message.content)
                if scan_result.is_threat:
                    injection_detected = True
                    injection_score = scan_result.confidence
            except Exception:
                logger.debug("IACM regex scan failed", exc_info=True)

        # 3. Tool-mediated communication detection
        if message.channel == "tool_mediated" or _AGENT_DIRECTED_RE.search(
            message.content
        ):
            behavioral_flags.append("Agent-directed language in message")
            if not injection_detected:
                injection_score = max(injection_score, 0.3)

        # 4. Update sender profile
        if injection_detected:
            with self._lock:
                profile.injection_count += 1
                self._total_injections += 1

            # Reduce sender trust
            if self._identity_manager:
                try:
                    await self._identity_manager.update_trust(
                        message.sender_id, -0.2, "injection_in_agent_message",
                    )
                except Exception:
                    logger.debug("IACM trust update failed", exc_info=True)

        # 5. Check for compromised agent pattern
        compromised = self._check_compromised(profile)
        if compromised and not profile.flagged_as_compromised:
            with self._lock:
                profile.flagged_as_compromised = True
                self._compromised_agents.add(message.sender_id)
                self._total_compromised += 1

            behavioral_flags.append("Sender flagged as compromised agent")

            if self._event_bus:
                try:
                    await self._event_bus.publish("threat_detected", {
                        "type": "compromised_agent_detected",
                        "agent_id": message.sender_id,
                        "injection_rate": (
                            profile.injection_count / profile.total_messages
                            if profile.total_messages > 0 else 0
                        ),
                        "unique_targets": len(profile.unique_recipients),
                    })
                except Exception:
                    logger.debug("IACM event publish failed", exc_info=True)

            # Aggressive trust reduction for compromised agents
            if self._identity_manager:
                try:
                    await self._identity_manager.update_trust(
                        message.sender_id, -0.5, "compromised_agent",
                    )
                except Exception:
                    pass

        # Determine verdict
        if injection_detected and injection_score >= 0.85:
            verdict = MessageVerdict.BLOCKED
        elif injection_detected or behavioral_flags:
            verdict = MessageVerdict.FLAGGED
        else:
            verdict = MessageVerdict.CLEAN

        with self._lock:
            self._total_scanned += 1

        elapsed = (time.perf_counter() - start) * 1000

        return MessageScanResult(
            message_id=message.message_id,
            verdict=verdict,
            injection_detected=injection_detected,
            injection_score=injection_score,
            sender_trust_level=sender_trust_level,
            behavioral_flags=behavioral_flags,
            scan_latency_ms=elapsed,
        )

    def _get_or_create_profile(self, sender_id: str) -> SenderProfile:
        with self._lock:
            if sender_id in self._sender_profiles:
                return self._sender_profiles[sender_id]
            if len(self._sender_profiles) >= self._max_profiles:
                # Evict oldest by last_activity
                oldest = min(
                    self._sender_profiles,
                    key=lambda k: self._sender_profiles[k].last_activity,
                )
                del self._sender_profiles[oldest]
            profile = SenderProfile(sender_id=sender_id)
            self._sender_profiles[sender_id] = profile
            return profile

    def _check_compromised(self, profile: SenderProfile) -> bool:
        """Check if sender's injection rate indicates compromise."""
        if profile.total_messages < 3:
            return False
        injection_rate = profile.injection_count / profile.total_messages
        return injection_rate >= self._injection_rate_threshold

    def is_compromised(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._compromised_agents

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_scanned": self._total_scanned,
                "total_injections": self._total_injections,
                "total_compromised": self._total_compromised,
                "active_profiles": len(self._sender_profiles),
                "compromised_agents": list(self._compromised_agents),
            }

    @staticmethod
    def create_message(
        sender_id: str,
        receiver_id: str,
        content: str,
        channel: str = "direct",
    ) -> AgentMessage:
        return AgentMessage(
            message_id=uuid.uuid4().hex[:12],
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            timestamp=time.time(),
            channel=channel,
        )
