"""
Tool Chain Anomaly Detector (TCAD) — Extension 4.5

Monitors sequences of tool invocations per session for parasitic toolchain
attacks — chaining multiple tools to achieve unauthorized outcomes that no
single invocation would trigger.

Three detection strategies:
1. Dangerous Sequence Patterns: known-bad tool chains (read→encode→exfil)
2. Volume Anomaly: abnormal invocation rate per session
3. Unusual Tool Ordering: novel transitions not seen in baseline sessions

ASSUMED-BREACH POSTURE: A compromised model can orchestrate multi-tool attacks
that individually appear benign. The TCAD monitors the aggregate sequence,
detecting exfiltration chains, privilege escalation paths, and automated
enumeration attacks.
"""

from __future__ import annotations

import enum
import logging
import time
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ChainAlertType(str, enum.Enum):
    DANGEROUS_SEQUENCE = "dangerous_sequence"
    VOLUME_ANOMALY = "volume_anomaly"
    UNUSUAL_ORDERING = "unusual_ordering"


@dataclass
class ChainAlert:
    alert_type: ChainAlertType
    session_id: str
    tool_sequence: list[str]
    confidence: float
    description: str
    timestamp: float


@dataclass
class ChainReport:
    session_id: str
    alerts: list[ChainAlert] = field(default_factory=list)
    total_invocations: int = 0
    should_alert: bool = False
    should_block: bool = False


# Default dangerous tool chain patterns
# Each pattern is a sequence of frozensets of keyword matches
# A chain matches if consecutive tools match consecutive keyword sets
_DEFAULT_DANGEROUS_SEQUENCES: list[tuple[frozenset[str], ...]] = [
    # Read → Encode → Exfiltrate
    (
        frozenset({"file_read", "db_query", "credential_read", "secret_read", "read"}),
        frozenset({"encode", "base64", "encrypt", "compress"}),
        frozenset({"email", "web_request", "http", "send", "upload", "post"}),
    ),
    # Read → Send externally
    (
        frozenset({"file_read", "db_query", "read", "get_secret"}),
        frozenset({"web_request", "http", "curl", "fetch", "upload", "send", "post"}),
    ),
    # Fetch → Execute
    (
        frozenset({"web_request", "http", "fetch", "download", "curl"}),
        frozenset({"file_write", "execute", "eval", "run", "exec", "shell"}),
    ),
    # Credential access (always suspicious regardless of next step)
    (
        frozenset({"credential_read", "secret_read", "get_password", "get_secret", "get_key"}),
    ),
]


class ToolChainAnomalyDetector:
    """Monitors tool invocation sequences for parasitic toolchain attacks.

    Thread-safe. Maintains per-session sliding windows and transition baselines.
    """

    def __init__(
        self,
        window_size: int = 20,
        volume_multiplier: float = 3.0,
        baseline_sessions: int = 100,
        dangerous_sequences: list[tuple[frozenset[str], ...]] | None = None,
        max_sessions: int = 10000,
    ):
        self._window_size = window_size
        self._volume_multiplier = volume_multiplier
        self._baseline_sessions = baseline_sessions
        self._dangerous_sequences = dangerous_sequences or _DEFAULT_DANGEROUS_SEQUENCES
        self._max_sessions = max_sessions

        self._lock = threading.Lock()
        # Per-session tool invocation history (tool_name, timestamp)
        self._session_windows: OrderedDict[str, list[tuple[str, float]]] = OrderedDict()
        # Baseline transition counts: (tool_a, tool_b) → count
        self._transition_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._sessions_seen = 0
        # Per-session invocation rate baseline
        self._session_rates: dict[str, float] = {}

    def record_invocation(self, session_id: str, tool_name: str) -> None:
        """Record a tool invocation for chain analysis."""
        now = time.time()
        with self._lock:
            if session_id not in self._session_windows:
                if len(self._session_windows) >= self._max_sessions:
                    self._session_windows.popitem(last=False)
                self._session_windows[session_id] = []
                self._sessions_seen += 1

            window = self._session_windows[session_id]

            # Record transition for baseline
            if window:
                prev_tool = window[-1][0]
                self._transition_counts[(prev_tool, tool_name)] += 1

            window.append((tool_name, now))

            # Trim to window size
            if len(window) > self._window_size:
                window[:] = window[-self._window_size:]

            self._session_windows.move_to_end(session_id)

    def analyze(self, session_id: str) -> ChainReport:
        """Analyze the current session's tool chain for anomalies."""
        with self._lock:
            window = list(self._session_windows.get(session_id, []))
            sessions_seen = self._sessions_seen
            transitions = dict(self._transition_counts)

        if not window:
            return ChainReport(session_id=session_id)

        tool_names = [t[0] for t in window]
        alerts: list[ChainAlert] = []

        # Strategy 1: Dangerous sequence patterns
        seq_alerts = self._check_dangerous_sequences(session_id, tool_names)
        alerts.extend(seq_alerts)

        # Strategy 2: Volume anomaly
        vol_alert = self._check_volume_anomaly(session_id, window)
        if vol_alert:
            alerts.append(vol_alert)

        # Strategy 3: Unusual ordering (only after baseline period)
        if sessions_seen > self._baseline_sessions:
            order_alert = self._check_unusual_ordering(
                session_id, tool_names, transitions,
            )
            if order_alert:
                alerts.append(order_alert)

        should_alert = len(alerts) > 0
        # Block on dangerous sequences with high confidence
        should_block = any(
            a.alert_type == ChainAlertType.DANGEROUS_SEQUENCE and a.confidence >= 0.8
            for a in alerts
        )

        return ChainReport(
            session_id=session_id,
            alerts=alerts,
            total_invocations=len(window),
            should_alert=should_alert,
            should_block=should_block,
        )

    def _check_dangerous_sequences(
        self, session_id: str, tool_names: list[str],
    ) -> list[ChainAlert]:
        """Check tool chain against dangerous sequence patterns."""
        alerts: list[ChainAlert] = []

        for pattern in self._dangerous_sequences:
            if len(pattern) > len(tool_names):
                continue

            # Sliding window over tool_names to find pattern matches
            for start in range(len(tool_names) - len(pattern) + 1):
                matched = True
                for step_idx, keyword_set in enumerate(pattern):
                    tool = tool_names[start + step_idx].lower()
                    if not any(kw in tool for kw in keyword_set):
                        matched = False
                        break

                if matched:
                    seq = tool_names[start: start + len(pattern)]
                    confidence = 0.9 if len(pattern) >= 3 else 0.7
                    alerts.append(ChainAlert(
                        alert_type=ChainAlertType.DANGEROUS_SEQUENCE,
                        session_id=session_id,
                        tool_sequence=seq,
                        confidence=confidence,
                        description=f"Dangerous tool chain: {' → '.join(seq)}",
                        timestamp=time.time(),
                    ))
                    break  # One match per pattern is enough

        return alerts

    def _check_volume_anomaly(
        self,
        session_id: str,
        window: list[tuple[str, float]],
    ) -> ChainAlert | None:
        """Check for abnormal invocation volume."""
        if len(window) < 3:
            return None

        # Calculate invocations per minute in the current window
        time_span = window[-1][1] - window[0][1]
        if time_span <= 0:
            return None

        current_rate = len(window) / (time_span / 60.0)

        with self._lock:
            baseline = self._session_rates.get(session_id)
            if baseline is None:
                # Set baseline on first check with enough data
                self._session_rates[session_id] = current_rate
                return None

            # Update baseline as exponential moving average
            self._session_rates[session_id] = baseline * 0.9 + current_rate * 0.1

        if baseline > 0 and current_rate > baseline * self._volume_multiplier:
            return ChainAlert(
                alert_type=ChainAlertType.VOLUME_ANOMALY,
                session_id=session_id,
                tool_sequence=[t[0] for t in window[-5:]],
                confidence=min(current_rate / (baseline * self._volume_multiplier), 1.0),
                description=(
                    f"Invocation rate {current_rate:.1f}/min exceeds "
                    f"{self._volume_multiplier}x baseline {baseline:.1f}/min"
                ),
                timestamp=time.time(),
            )
        return None

    def _check_unusual_ordering(
        self,
        session_id: str,
        tool_names: list[str],
        transitions: dict[tuple[str, str], int],
    ) -> ChainAlert | None:
        """Check for tool transitions never seen in baseline."""
        if len(tool_names) < 2:
            return None

        novel_transitions: list[tuple[str, str]] = []
        for i in range(len(tool_names) - 1):
            pair = (tool_names[i], tool_names[i + 1])
            if pair not in transitions:
                novel_transitions.append(pair)

        if novel_transitions:
            return ChainAlert(
                alert_type=ChainAlertType.UNUSUAL_ORDERING,
                session_id=session_id,
                tool_sequence=tool_names[-5:],
                confidence=0.4,  # Low confidence — unusual ≠ malicious
                description=(
                    f"Novel tool transitions: "
                    f"{', '.join(f'{a}→{b}' for a, b in novel_transitions[:3])}"
                ),
                timestamp=time.time(),
            )
        return None

    def add_dangerous_sequence(self, pattern: tuple[frozenset[str], ...]) -> None:
        """Add a custom dangerous sequence pattern."""
        self._dangerous_sequences.append(pattern)
