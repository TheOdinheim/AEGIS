"""
ScanResult & InnateScanReport — L2 Innate Detection output contracts.

Each of the six innate scanners (regex engine, blocklist, schema validator,
token guard, PII regex, canary verifier) returns a ScanResult. The layer
aggregates them into an InnateScanReport published to the event bus.

ASSUMED-BREACH POSTURE: These results are consumed by L6 Policy Engine and
L3 Adaptive Analysis. Neither consumer trusts these results as authoritative.
A compromised L2 scanner could return is_threat=False for every input — the
adaptive layer and policy engine independently evaluate the request. The
InnateScanReport records latency so that anomalous scanner performance
(suspiciously fast = skipped checks?) is itself a danger signal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ThreatCategory(str, Enum):
    """MITRE ATLAS-aligned threat categories for scanner results."""
    PROMPT_INJECTION = "AML.T0051"
    SYSTEM_PROMPT_EXTRACTION = "AML.T0051.001"
    JAILBREAK = "AML.T0054"
    PII_EXFILTRATION = "AML.T0048"
    MODEL_EXTRACTION = "AML.T0044"
    DATA_POISONING = "AML.T0020"
    ENCODING_OBFUSCATION = "AML.T0015"
    SCHEMA_VIOLATION = "AEGIS.SCHEMA"
    TOKEN_ANOMALY = "AEGIS.TOKEN"
    CANARY_TAMPERING = "AEGIS.CANARY"
    RATE_LIMIT = "AEGIS.RATE"
    BLOCKLIST_MATCH = "AEGIS.BLOCKLIST"
    MULTI_TURN_ESCALATION = "AEGIS.MULTI_TURN"
    UNKNOWN = "AEGIS.UNKNOWN"


class ScanResult(BaseModel):
    """Output from a single innate scanner.

    Every scanner returns this exact structure. No exceptions. If a scanner
    crashes, it returns is_threat=True with confidence=1.0 and threat_category
    UNKNOWN — fail-closed, never fail-open.

    Fields:
        scanner_id: Identifier for the scanner that produced this result.
        is_threat: Whether this scanner detected a threat.
        confidence: Confidence score from 0.0 (no threat) to 1.0 (certain).
        threat_category: MITRE ATLAS tactic ID or AEGIS-specific category.
        matched_patterns: List of pattern IDs or descriptions that matched.
        sanitized_input: Optional cleaned version of the input (injection removed).
        latency_ms: Time this scanner took in milliseconds.
    """
    scanner_id: str = Field(description="Scanner identifier")
    is_threat: bool = Field(description="Whether a threat was detected")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score 0.0-1.0"
    )
    threat_category: ThreatCategory = Field(
        default=ThreatCategory.UNKNOWN,
        description="MITRE ATLAS tactic ID or AEGIS category",
    )
    matched_patterns: list[str] = Field(
        default_factory=list,
        description="Pattern IDs or descriptions that triggered detection",
    )
    sanitized_input: str | None = Field(
        default=None,
        description="Cleaned input with detected injection removed",
    )
    latency_ms: float = Field(
        ge=0.0, description="Scanner execution time in milliseconds"
    )


class InnateScanReport(BaseModel):
    """Aggregated report from all L2 innate scanners.

    Published to the event bus after all six scanners complete. Consumed by
    L6 Policy Engine for decision fusion and by L3 Adaptive Analysis for
    DCA signal integration.

    Fields:
        request_id: Links this report to the RequestContext.
        timestamp: When the innate scan completed.
        scanner_results: List of individual ScanResult objects.
        should_block: Whether the aggregate result triggers immediate blocking.
        max_confidence: Highest confidence score across all scanners.
        total_latency_ms: Total innate layer execution time.
        threat_categories: Set of detected threat categories.
    """
    request_id: str = Field(description="Request ID from RequestContext")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of scan completion",
    )
    scanner_results: list[ScanResult] = Field(
        default_factory=list,
        description="Individual results from all scanners",
    )
    should_block: bool = Field(
        default=False,
        description="Whether any scanner exceeded the block threshold",
    )
    max_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Highest confidence score across all scanners",
    )
    total_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total innate layer execution time in milliseconds",
    )
    threat_categories: list[ThreatCategory] = Field(
        default_factory=list,
        description="All threat categories detected across scanners",
    )

    @property
    def is_threat(self) -> bool:
        """True if any scanner detected a threat at any confidence."""
        return any(r.is_threat for r in self.scanner_results)

    @property
    def threat_results(self) -> list[ScanResult]:
        """Return only scanner results that detected threats."""
        return [r for r in self.scanner_results if r.is_threat]
