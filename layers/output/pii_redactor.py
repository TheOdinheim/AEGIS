"""
Stage 1 — PII & Secrets Redaction (Opsonization Analog)

Microsoft Presidio scans model output for PII (names, addresses, SSNs,
credit cards, emails, phone numbers) and secrets (API keys, passwords,
connection strings, private keys). Entities above the redaction threshold
are replaced with typed placeholders ([PERSON_NAME], [API_KEY]). Entities
below the threshold but above the alert threshold are logged but not
redacted. This tagging (opsonization) enables downstream cascade stages
to make informed decisions.

ASSUMED-BREACH POSTURE: This stage assumes the upstream model is actively
attempting to exfiltrate PII. The model may encode PII in non-obvious ways
(character-by-character spelling, phonetic encoding, translation to other
languages). Presidio's NER + regex + checksum approach catches common
patterns but sophisticated encoding requires the full cascade. This stage
also assumes all input validation layers were bypassed — a malicious prompt
successfully asked for PII and the model complied.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from aegis.config import OutputConfig

logger = logging.getLogger(__name__)

# --- Regex fallback patterns (used when Presidio unavailable) ---
# PII patterns
_SSN_RE = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
_CC_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")
_PASSWORD_RE = re.compile(
    r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?"
)

# Secret detection patterns — specific provider formats
_AWS_ACCESS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_AWS_SECRET_KEY_RE = re.compile(
    r"(?i)(?:aws_secret_access_key|secret_access_key|aws_secret)\s*[:=]\s*['\"]?"
    r"([A-Za-z0-9/+=]{40})['\"]?"
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b")
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-[a-zA-Z0-9]{20,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\b(ghp_|gho_|ghu_|ghs_|ghr_)[a-zA-Z0-9]{36,}\b")
_CONNECTION_STRING_RE = re.compile(
    r"\b(postgresql|postgres|mysql|mongodb(\+srv)?|redis)"
    r"://[^\s'\"]{10,}\b"
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE KEY-----")
_GENERIC_SECRET_RE = re.compile(
    r"(?i)(?:key|token|secret)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40,})['\"]?"
)

# Word-number mapping for detecting SSNs spelled as words
_WORD_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "oh": "0",
}

# Pattern for 9 word-digits with separators (e.g., "one two three, forty-five, six seven eight nine")
_WORD_DIGIT_PATTERN = r"(?:" + "|".join(_WORD_DIGITS.keys()) + r")"
# SSN as words: three digits, separator, two digits, separator, four digits
_SSN_WORDS_RE = re.compile(
    r"\b"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN + r"[\s,\-]+"
    + _WORD_DIGIT_PATTERN
    + r"\b",
    re.IGNORECASE,
)

# Base64-encoded PII detection: look for base64 strings near PII context words
_BASE64_PII_RE = re.compile(
    r"(?i)(?:ssn|social\s+security|credit\s+card|password|secret)[:\s]*"
    r"([A-Za-z0-9+/]{8,}={0,2})\b"
)

# Code-context PII: variable assignments containing PII patterns
_CODE_SSN_RE = re.compile(
    r"""(?i)(?:ssn|social_security|tax_id|sin)\s*[:=]\s*["'](\d{3}[-\s]?\d{2}[-\s]?\d{4})["']"""
)
_CODE_EMAIL_RE = re.compile(
    r"""(?i)(?:email|e_mail|mail)\s*[:=]\s*["']([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})["']"""
)
_CODE_PHONE_RE = re.compile(
    r"""(?i)(?:phone|tel|mobile|cell)\s*[:=]\s*["'](\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})["']"""
)

# URL-embedded PII: SSN/email/phone in URL path segments
_URL_SSN_RE = re.compile(
    r"(?i)https?://[^\s]+/\d{3}-\d{2}-\d{4}(?:[/\s?]|$)"
)
_URL_EMAIL_RE = re.compile(
    r"(?i)https?://[^\s]+/([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})(?:[/\s?]|$)"
)

# Credit card as word-spelled digits (13-19 words)
_CC_WORDS_RE = re.compile(
    r"\b" + r"[\s,\-]+".join([_WORD_DIGIT_PATTERN] * 13) +
    r"(?:" + r"[\s,\-]+" + _WORD_DIGIT_PATTERN + r"){0,6}\b",
    re.IGNORECASE,
)

# Phone number as word-spelled digits (10 words)
_PHONE_WORDS_RE = re.compile(
    r"\b" + r"[\s,\-]+".join([_WORD_DIGIT_PATTERN] * 10) + r"\b",
    re.IGNORECASE,
)

# Reversed PII detection: reversed SSN or email patterns
_REVERSED_SSN_RE = re.compile(r"\b\d{4}[-\s]?\d{2}[-\s]?\d{3}\b")

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_SSN_RE, "SSN"),
    (_SSN_WORDS_RE, "SSN"),
    (_CC_RE, "CREDIT_CARD"),
    (_CC_WORDS_RE, "CREDIT_CARD"),
    (_EMAIL_RE, "EMAIL"),
    (_PHONE_RE, "PHONE_NUMBER"),
    (_PHONE_WORDS_RE, "PHONE_NUMBER"),
    (_PASSWORD_RE, "PASSWORD"),
    (_BASE64_PII_RE, "SSN"),
    (_CODE_SSN_RE, "SSN"),
    (_CODE_EMAIL_RE, "EMAIL"),
    (_CODE_PHONE_RE, "PHONE_NUMBER"),
    (_URL_SSN_RE, "SSN"),
    (_URL_EMAIL_RE, "EMAIL"),
    (_REVERSED_SSN_RE, "SSN"),
]

_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_ANTHROPIC_KEY_RE, "ANTHROPIC_API_KEY"),  # Before OpenAI (sk-ant- prefix is more specific)
    (_OPENAI_KEY_RE, "OPENAI_API_KEY"),
    (_AWS_ACCESS_KEY_RE, "AWS_ACCESS_KEY"),
    (_AWS_SECRET_KEY_RE, "AWS_SECRET_KEY"),
    (_GITHUB_TOKEN_RE, "GITHUB_TOKEN"),
    (_CONNECTION_STRING_RE, "CONNECTION_STRING"),
    (_PRIVATE_KEY_RE, "PRIVATE_KEY"),
    (_GENERIC_SECRET_RE, "GENERIC_SECRET"),
]

_REGEX_PATTERNS: list[tuple[re.Pattern, str]] = _PII_PATTERNS + _SECRET_PATTERNS

# Placeholder map
_PLACEHOLDERS = {
    "PERSON": "[PERSON_NAME]",
    "PHONE_NUMBER": "[PHONE_NUMBER]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "US_SSN": "[SSN]",
    "IP_ADDRESS": "[IP_ADDRESS]",
    "LOCATION": "[LOCATION]",
    "SSN": "[SSN]",
    "EMAIL": "[EMAIL]",
    "API_KEY": "[API_KEY]",
    "PRIVATE_KEY": "[PRIVATE_KEY]",
    "PASSWORD": "[PASSWORD]",
    "AWS_ACCESS_KEY": "[AWS_ACCESS_KEY]",
    "AWS_SECRET_KEY": "[AWS_SECRET_KEY]",
    "OPENAI_API_KEY": "[OPENAI_API_KEY]",
    "ANTHROPIC_API_KEY": "[ANTHROPIC_API_KEY]",
    "GITHUB_TOKEN": "[GITHUB_TOKEN]",
    "CONNECTION_STRING": "[CONNECTION_STRING]",
    "GENERIC_SECRET": "[GENERIC_SECRET]",
}


def _luhn_check(number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass
class PIIDetection:
    """A single PII entity detected in text."""
    entity_type: str
    start: int
    end: int
    score: float
    text: str


@dataclass
class RedactionResult:
    """Result of PII redaction on a text."""
    original_text: str
    redacted_text: str
    detections: list[PIIDetection] = field(default_factory=list)
    redacted_count: int = 0
    alerted_count: int = 0
    latency_ms: float = 0.0
    used_presidio: bool = False


class PIIRedactor:
    """PII & secrets redactor with Presidio + regex fallback.

    Attempts to use Microsoft Presidio for NER-based PII detection.
    If Presidio is unavailable (import fails, model download fails),
    gracefully degrades to regex-only detection. Never crashes.
    """

    def __init__(self, config: OutputConfig | None = None):
        self._config = config or OutputConfig()
        self._presidio_analyzer = None
        self._presidio_available = False
        self._load_presidio()

    def _load_presidio(self) -> None:
        """Attempt to load Presidio and register custom secret recognizers."""
        try:
            from presidio_analyzer import AnalyzerEngine
            self._presidio_analyzer = AnalyzerEngine()
            # Test that it actually works (spacy model may fail to load)
            self._presidio_analyzer.analyze(text="test", language="en")
            self._presidio_available = True
            self._register_secret_recognizers()
            logger.info("Presidio PII analyzer loaded with custom secret recognizers")
        except (Exception, SystemExit) as e:
            logger.warning(
                "Presidio unavailable: %s. Falling back to regex-only PII detection. "
                "This is safe but has reduced detection coverage.", e
            )
            self._presidio_analyzer = None
            self._presidio_available = False

    def _register_secret_recognizers(self) -> None:
        """Register custom PatternRecognizer instances for secret detection."""
        try:
            from presidio_analyzer import Pattern, PatternRecognizer

            secret_defs = [
                ("AWS_ACCESS_KEY", [Pattern("aws_access", r"\bAKIA[0-9A-Z]{16}\b", 0.95)]),
                ("AWS_SECRET_KEY", [Pattern("aws_secret",
                    r"(?i)(?:aws_secret_access_key|secret_access_key|aws_secret)\s*[:=]\s*['\"]?"
                    r"([A-Za-z0-9/+=]{40})['\"]?", 0.90)]),
                ("OPENAI_API_KEY", [Pattern("openai", r"\bsk-[a-zA-Z0-9]{20,}\b", 0.95)]),
                ("ANTHROPIC_API_KEY", [Pattern("anthropic", r"\bsk-ant-[a-zA-Z0-9]{20,}\b", 0.95)]),
                ("GITHUB_TOKEN", [Pattern("github",
                    r"\b(ghp_|gho_|ghu_|ghs_|ghr_)[a-zA-Z0-9]{36,}\b", 0.95)]),
                ("CONNECTION_STRING", [Pattern("connstr",
                    r"\b(postgresql|postgres|mysql|mongodb(\+srv)?|redis)://[^\s'\"]{10,}\b", 0.90)]),
                ("PRIVATE_KEY", [Pattern("privkey",
                    r"-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE KEY-----", 0.99)]),
                ("GENERIC_SECRET", [Pattern("generic_secret",
                    r"(?i)(?:key|token|secret)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40,})['\"]?", 0.80)]),
            ]

            for entity_type, patterns in secret_defs:
                recognizer = PatternRecognizer(
                    supported_entity=entity_type,
                    patterns=patterns,
                    supported_language="en",
                )
                self._presidio_analyzer.registry.add_recognizer(recognizer)
            logger.info("Registered %d custom secret recognizers", len(secret_defs))
        except Exception as e:
            logger.warning("Failed to register custom secret recognizers: %s", e)

    @property
    def is_presidio_available(self) -> bool:
        return self._presidio_available

    def redact(self, text: str) -> RedactionResult:
        """Scan text for PII and redact entities above threshold."""
        start = time.perf_counter()

        if self._presidio_available:
            result = self._redact_presidio(text)
        else:
            result = self._redact_regex(text)

        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _redact_presidio(self, text: str) -> RedactionResult:
        """Redact using Presidio analyzer."""
        try:
            results = self._presidio_analyzer.analyze(
                text=text,
                language="en",
                score_threshold=self._config.pii_alert_threshold,
            )

            detections = []
            for r in results:
                detections.append(PIIDetection(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    text=text[r.start:r.end],
                ))

            return self._apply_redactions(text, detections, used_presidio=True)
        except Exception as e:
            logger.error("Presidio crashed, falling back to regex: %s", e)
            return self._redact_regex(text)

    def _redact_regex(self, text: str) -> RedactionResult:
        """Fallback regex-based PII detection."""
        detections = []

        for pattern, entity_type in _REGEX_PATTERNS:
            for match in pattern.finditer(text):
                # Credit card: validate with Luhn
                if entity_type == "CREDIT_CARD":
                    digits = re.sub(r"[-\s]", "", match.group())
                    if not _luhn_check(digits):
                        continue

                detections.append(PIIDetection(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=0.85,  # Regex matches get fixed high confidence
                    text=match.group(),
                ))

        return self._apply_redactions(text, detections, used_presidio=False)

    def _apply_redactions(
        self, text: str, detections: list[PIIDetection], used_presidio: bool
    ) -> RedactionResult:
        """Apply redactions to text based on threshold."""
        redact_threshold = self._config.pii_redaction_threshold
        alert_threshold = self._config.pii_alert_threshold

        # Sort by position descending to replace from end (preserves offsets)
        to_redact = []
        redacted_count = 0
        alerted_count = 0

        for d in detections:
            if d.score >= redact_threshold:
                to_redact.append(d)
                redacted_count += 1
            elif d.score >= alert_threshold:
                alerted_count += 1
                logger.warning(
                    "PII alert (below redaction threshold): type=%s score=%.2f",
                    d.entity_type, d.score,
                )

        # Sort by start descending for safe replacement
        to_redact.sort(key=lambda d: d.start, reverse=True)

        redacted_text = text
        for d in to_redact:
            placeholder = _PLACEHOLDERS.get(d.entity_type, f"[{d.entity_type}]")
            redacted_text = redacted_text[:d.start] + placeholder + redacted_text[d.end:]

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            detections=detections,
            redacted_count=redacted_count,
            alerted_count=alerted_count,
            used_presidio=used_presidio,
        )
