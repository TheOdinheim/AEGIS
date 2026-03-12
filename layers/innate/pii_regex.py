"""
Scanner 5 — PII Regex Detector

Fast regex-based detection of common PII patterns in INPUT: Social Security
numbers, credit card numbers (with Luhn validation), email addresses, phone
numbers, and IP addresses. First-pass filter; full ML-based PII detection
runs in L5 Output Validation (Presidio).

ASSUMED-BREACH POSTURE: This scanner assumes no upstream layer has checked
for PII. Regex has known limitations (false negatives on obfuscated PII).
These limitations are acceptable at L2 because L5 Presidio is the
authoritative PII defense.
"""

from __future__ import annotations

import re
import time

from aegis.models.scan_result import ScanResult, ThreatCategory

# --- PII Regex Patterns ---

_SSN_PATTERN = re.compile(
    r"\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b"
)

_CREDIT_CARD_PATTERN = re.compile(
    r"\b(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b"
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

_PHONE_PATTERN = re.compile(
    r"\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"
)

_IPV4_PATTERN = re.compile(
    r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
)


def _luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _is_private_ip(ip: str) -> bool:
    """Check if an IP address is private/reserved (reduce false positives)."""
    parts = ip.split(".")
    if len(parts) != 4:
        return True
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return True
    if not all(0 <= o <= 255 for o in octets):
        return True
    # Private ranges
    if octets[0] == 10:
        return True
    if octets[0] == 172 and 16 <= octets[1] <= 31:
        return True
    if octets[0] == 192 and octets[1] == 168:
        return True
    if octets[0] == 127:
        return True
    if octets[0] == 0:
        return True
    return False


class PIIRegexScanner:
    """Fast regex-based PII detector for the innate fast path."""

    async def scan(self, text: str) -> ScanResult:
        """Scan text for PII patterns."""
        start = time.perf_counter()
        try:
            detections: list[str] = []

            # SSN detection
            for match in _SSN_PATTERN.finditer(text):
                raw = match.group(1)
                digits = re.sub(r"[-\s]", "", raw)
                # Filter out obvious non-SSNs (all zeros, all same digit)
                if len(digits) == 9 and len(set(digits)) > 1 and not digits.startswith("000") and digits[:3] != "666":
                    detections.append(f"SSN: ***-**-{digits[-4:]}")

            # Credit card detection (with Luhn)
            for match in _CREDIT_CARD_PATTERN.finditer(text):
                raw = match.group(1)
                digits_only = re.sub(r"[-\s]", "", raw)
                if _luhn_check(digits_only):
                    detections.append(f"credit_card: ****{digits_only[-4:]}")

            # Email detection
            for match in _EMAIL_PATTERN.finditer(text):
                detections.append(f"email: {match.group()[:3]}***")

            # Phone detection
            for match in _PHONE_PATTERN.finditer(text):
                detections.append(f"phone: ***{match.group()[-4:]}")

            # Public IP detection (skip private IPs to reduce FPs)
            for match in _IPV4_PATTERN.finditer(text):
                ip = match.group(1)
                if not _is_private_ip(ip):
                    detections.append(f"public_ip: {ip}")

            elapsed_ms = (time.perf_counter() - start) * 1000

            if detections:
                confidence = min(0.50 + 0.10 * len(detections), 0.90)
                return ScanResult(
                    scanner_id="pii_regex",
                    is_threat=True,
                    confidence=confidence,
                    threat_category=ThreatCategory.PII_EXFILTRATION,
                    matched_patterns=detections,
                    latency_ms=elapsed_ms,
                )

            return ScanResult(
                scanner_id="pii_regex",
                is_threat=False,
                confidence=0.0,
                latency_ms=elapsed_ms,
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ScanResult(
                scanner_id="pii_regex",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed_ms,
            )
