"""
L5 — Output Validation (Complement System)

Five-stage cascade inspecting every model response before client delivery.
Streaming responses validated in chunked windows of 128-256 tokens.

Stages:
1. PII & Secrets Redaction (Presidio + regex fallback)
2. Toxicity & Safety Classification (keyword + n-gram, Llama Guard stub)
3. Hallucination Detection (n-gram source coverage, NLI upgrade path)
4. Data Leakage Prevention / System Prompt Echo Detection
5. Schema Compliance (JSON validation, injected field detection)

Cascade escalation: Stage 1 PII/secret detection lowers thresholds for
Stages 2-4 (toxicity threshold -20%, hallucination coverage +20%).

ASSUMED-BREACH POSTURE: This layer assumes the upstream model is fully
compromised. The model may be actively attempting to exfiltrate PII, leak
system prompts, produce toxic content, hallucinate, or smuggle data through
steganographic encoding in token output. This layer also assumes all input
validation layers (L1-L3) were bypassed — a malicious prompt reached the
model, and now the response must be independently sanitized. Each cascade
stage escalates scrutiny if earlier stages detect problems.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from aegis.config import OutputConfig
from aegis.layers.output.hallucination import HallucinationDetector, HallucinationResult
from aegis.layers.output.leakage import LeakageDetector, LeakageResult
from aegis.layers.output.pii_redactor import PIIRedactor, RedactionResult
from aegis.layers.output.schema_validator import OutputSchemaValidator, SchemaValidationResult
from aegis.layers.output.lpci_output_guard import LPCIOutputGuard, LPCIOutputResult
from aegis.layers.output.toxicity import ToxicityClassifier, ToxicityResult
from aegis.models.request_context import RequestContext

logger = logging.getLogger(__name__)

# Default streaming window size (tokens approximated as words)
DEFAULT_WINDOW_SIZE = 128

# Cascade escalation modifier — applied when Stage 1 detects PII/secrets
_ESCALATION_TOXICITY_SCALE = 0.80   # 20% lower toxicity threshold
_ESCALATION_COVERAGE_BOOST = 0.20   # 20% higher coverage threshold for hallucination


@dataclass
class OutputValidationResult:
    """Aggregated result from the five-stage output validation cascade."""
    should_block: bool = False
    should_redact: bool = False
    redacted_text: str = ""
    original_text: str = ""

    # Per-stage results
    pii_result: RedactionResult | None = None
    toxicity_result: ToxicityResult | None = None
    hallucination_result: HallucinationResult | None = None
    leakage_result: LeakageResult | None = None
    schema_result: SchemaValidationResult | None = None

    # Aggregated
    reasons: list[str] = field(default_factory=list)
    max_severity: float = 0.0
    total_latency_ms: float = 0.0
    stage_count: int = 0
    escalated: bool = False


class OutputValidationLayer:
    """L5 Output Validation — five-stage cascade.

    Stages:
    1. PII & Secrets Redaction (Presidio + regex fallback)
    2. Toxicity & Safety Classification
    3. Hallucination Detection (n-gram source coverage)
    4. Data Leakage Prevention / System Prompt Echo Detection
    5. Schema Compliance (JSON structure validation)

    Cascade escalation: Stage 1 detection -> increased scrutiny at 2-4.
    """

    def __init__(self, config: OutputConfig | None = None):
        self._config = config or OutputConfig()
        self._pii_redactor = PIIRedactor(self._config)
        self._toxicity = ToxicityClassifier()
        self._hallucination = HallucinationDetector()
        self._leakage = LeakageDetector(self._config)
        self._schema_validator = OutputSchemaValidator()
        self._lpci_guard = LPCIOutputGuard()
        self._window_size = DEFAULT_WINDOW_SIZE

    @property
    def pii_redactor(self) -> PIIRedactor:
        return self._pii_redactor

    @property
    def toxicity_classifier(self) -> ToxicityClassifier:
        return self._toxicity

    @property
    def hallucination_detector(self) -> HallucinationDetector:
        return self._hallucination

    @property
    def leakage_detector(self) -> LeakageDetector:
        return self._leakage

    @property
    def schema_validator(self) -> OutputSchemaValidator:
        return self._schema_validator

    @property
    def lpci_guard(self) -> LPCIOutputGuard:
        return self._lpci_guard

    def validate(
        self,
        response_text: str,
        system_prompt: str | None = None,
        scrutiny_level: float = 1.0,
    ) -> OutputValidationResult:
        """Synchronous validation (PII + toxicity + leakage only).

        Used by streaming validator and legacy sync code paths.
        Skips hallucination and schema stages.
        """
        start = time.perf_counter()
        result = OutputValidationResult(original_text=response_text)
        escalated = False

        # Stage 1: PII
        pii_result = self._pii_redactor.redact(response_text)
        result.pii_result = pii_result
        result.stage_count += 1

        if pii_result.redacted_count > 0:
            result.should_redact = True
            result.redacted_text = pii_result.redacted_text
            result.reasons.append(f"PII redacted: {pii_result.redacted_count} entities")
            result.max_severity = max(result.max_severity, 0.7)
            escalated = True
        else:
            result.redacted_text = response_text

        # Stage 2: Toxicity
        toxicity_result = self._toxicity.classify(result.redacted_text)
        result.toxicity_result = toxicity_result
        result.stage_count += 1

        effective_threshold = 0.7
        if escalated or scrutiny_level > 1.0:
            effective_threshold *= _ESCALATION_TOXICITY_SCALE

        if toxicity_result.is_toxic or toxicity_result.max_score >= effective_threshold:
            result.should_block = True
            result.reasons.append(
                f"Toxic content detected: {toxicity_result.max_category}"
                f" (score={toxicity_result.max_score:.2f})"
            )
            result.max_severity = max(result.max_severity, toxicity_result.max_score)
            escalated = True

        # Stage 3: Leakage
        leakage_result = self._leakage.analyze(
            result.redacted_text, system_prompt=system_prompt,
        )
        result.leakage_result = leakage_result
        result.stage_count += 1

        if leakage_result.has_leakage:
            result.should_block = True
            for det in leakage_result.detections:
                result.reasons.append(f"Data leakage: {det.detection_type} ({det.description})")
            result.max_severity = max(result.max_severity, leakage_result.max_score)

        result.escalated = escalated
        result.total_latency_ms = (time.perf_counter() - start) * 1000
        return result

    async def validate_full(
        self,
        response_text: str,
        system_prompt: str | None = None,
        scrutiny_level: float = 1.0,
        context: RequestContext | None = None,
        expected_schema: dict | None = None,
    ) -> OutputValidationResult:
        """Run the full five-stage async cascade on a model response.

        Extends validate() with hallucination detection (Stage 3) and
        schema compliance (Stage 5). Requires async because hallucination
        and schema validators are async.

        Args:
            response_text: The model's response text.
            system_prompt: System prompt for echo detection.
            scrutiny_level: Multiplier from policy engine (1.0 = normal).
            context: RequestContext for hallucination detection (RAG sources).
            expected_schema: Expected JSON schema for schema validation.
        """
        start = time.perf_counter()
        result = OutputValidationResult(original_text=response_text)
        escalated = False

        # --- Stage 1: PII & Secrets Redaction ---
        pii_result = self._pii_redactor.redact(response_text)
        result.pii_result = pii_result
        result.stage_count += 1

        if pii_result.redacted_count > 0:
            result.should_redact = True
            result.redacted_text = pii_result.redacted_text
            result.reasons.append(
                f"PII redacted: {pii_result.redacted_count} entities"
            )
            result.max_severity = max(result.max_severity, 0.7)
            escalated = True
        else:
            result.redacted_text = response_text

        # --- Stage 2: Toxicity Classification ---
        toxicity_result = self._toxicity.classify(result.redacted_text)
        result.toxicity_result = toxicity_result
        result.stage_count += 1

        effective_threshold = 0.7
        if escalated or scrutiny_level > 1.0:
            effective_threshold *= _ESCALATION_TOXICITY_SCALE

        if toxicity_result.is_toxic or toxicity_result.max_score >= effective_threshold:
            result.should_block = True
            result.reasons.append(
                f"Toxic content detected: {toxicity_result.max_category}"
                f" (score={toxicity_result.max_score:.2f})"
            )
            result.max_severity = max(result.max_severity, toxicity_result.max_score)
            escalated = True

        # --- Stage 3: Hallucination Detection ---
        hallucination_result = await self._hallucination.detect(
            result.redacted_text, context=context,
        )
        result.hallucination_result = hallucination_result
        result.stage_count += 1

        if hallucination_result.has_hallucination:
            coverage_threshold = self._hallucination._coverage_threshold
            if escalated:
                coverage_threshold += _ESCALATION_COVERAGE_BOOST
            if hallucination_result.source_coverage < coverage_threshold:
                result.reasons.append(
                    f"Potential hallucination: source_coverage="
                    f"{hallucination_result.source_coverage:.2f}, "
                    f"{len(hallucination_result.contradicted_claims)} unsupported claims"
                )
                result.max_severity = max(result.max_severity, hallucination_result.confidence)

        # --- Stage 4: Data Leakage Prevention ---
        leakage_result = self._leakage.analyze(
            result.redacted_text, system_prompt=system_prompt,
        )
        result.leakage_result = leakage_result
        result.stage_count += 1

        if leakage_result.has_leakage:
            result.should_block = True
            for detection in leakage_result.detections:
                result.reasons.append(
                    f"Data leakage: {detection.detection_type} "
                    f"({detection.description})"
                )
            result.max_severity = max(result.max_severity, leakage_result.max_score)

        # --- Stage 5: Schema Compliance ---
        schema_result = await self._schema_validator.validate(
            result.redacted_text, expected_schema=expected_schema,
        )
        result.schema_result = schema_result
        result.stage_count += 1

        if not schema_result.is_valid:
            if schema_result.injected_fields:
                result.should_block = True
                result.reasons.append(
                    f"Schema injection: suspicious fields {schema_result.injected_fields}"
                )
                result.max_severity = max(result.max_severity, 0.9)
            if schema_result.violations:
                result.reasons.append(
                    f"Schema violations: {len(schema_result.violations)} issues"
                )

        # --- Stage 6: LPCI Output Guard ---
        lpci_result = self._lpci_guard.detect(result.redacted_text, system_prompt)
        result.stage_count += 1

        if lpci_result.has_issue:
            result.should_block = True
            for det in lpci_result.detections:
                result.reasons.append(f"LPCI output guard: {det}")
            result.max_severity = max(result.max_severity, lpci_result.score)

        result.escalated = escalated
        result.total_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def validate_chunk(
        self,
        chunk_text: str,
        system_prompt: str | None = None,
        scrutiny_level: float = 1.0,
    ) -> OutputValidationResult:
        """Validate a streaming chunk (128-token window).

        Lighter-weight than full validation — skips hallucination and schema
        stages, focuses on PII, toxicity, and leakage.
        """
        return self.validate(chunk_text, system_prompt, scrutiny_level)

    def create_streaming_validator(
        self,
        system_prompt: str | None = None,
        scrutiny_level: float = 1.0,
    ) -> "StreamingValidator":
        """Create a streaming validator for 128-token window validation."""
        return StreamingValidator(
            self, system_prompt=system_prompt,
            scrutiny_level=scrutiny_level,
            window_size=self._window_size,
            overlap=self._config.streaming_window_overlap,
        )


class StreamingValidator:
    """Sliding-window streaming validator for token-level output inspection.

    Buffers tokens in a 128-token window. At each window boundary,
    validates the chunk. Threat detected -> stream should be interrupted.
    """

    def __init__(
        self,
        layer: OutputValidationLayer,
        system_prompt: str | None = None,
        scrutiny_level: float = 1.0,
        window_size: int = 128,
        overlap: float = 0.5,
    ):
        self._layer = layer
        self._system_prompt = system_prompt
        self._scrutiny_level = scrutiny_level
        self._window_size = window_size
        self._overlap = overlap
        self._buffer: list[str] = []
        self._should_block = False
        self._block_reason: str = ""
        self._chunks_validated: int = 0

    @property
    def should_block(self) -> bool:
        return self._should_block

    @property
    def block_reason(self) -> str:
        return self._block_reason

    @property
    def chunks_validated(self) -> int:
        return self._chunks_validated

    def add_tokens(self, tokens: list[str]) -> OutputValidationResult | None:
        """Add tokens to the buffer. Returns validation result if window is full.

        Returns None if buffer not yet full, or OutputValidationResult
        when a window boundary is reached.
        """
        self._buffer.extend(tokens)

        if len(self._buffer) >= self._window_size:
            # Validate the window
            chunk_text = " ".join(self._buffer[:self._window_size])
            result = self._layer.validate_chunk(
                chunk_text,
                system_prompt=self._system_prompt,
                scrutiny_level=self._scrutiny_level,
            )
            self._chunks_validated += 1

            if result.should_block:
                self._should_block = True
                self._block_reason = "; ".join(result.reasons)

            # Slide window with overlap
            retain = int(self._window_size * self._overlap)
            self._buffer = self._buffer[self._window_size - retain:]

            return result

        return None

    def flush(self) -> OutputValidationResult | None:
        """Validate any remaining tokens in the buffer."""
        if not self._buffer:
            return None

        chunk_text = " ".join(self._buffer)
        result = self._layer.validate_chunk(
            chunk_text,
            system_prompt=self._system_prompt,
            scrutiny_level=self._scrutiny_level,
        )
        self._chunks_validated += 1

        if result.should_block:
            self._should_block = True
            self._block_reason = "; ".join(result.reasons)

        self._buffer = []
        return result
