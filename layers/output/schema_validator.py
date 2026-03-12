"""
Stage 5 — Output Schema Compliance Validation

Validates structured output (JSON mode, function calling) against expected
schemas. Detects injected fields, type mismatches, missing fields, and
oversized payloads that could indicate injection or data exfiltration.

ASSUMED-BREACH POSTURE: This stage assumes the model has been manipulated
to inject additional fields into structured output — fields named "system",
"prompt", "instructions", or "role" that could be used to influence
downstream systems consuming the JSON. Also assumes the model may attempt
to exfiltrate data by inflating the response with extra fields.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Fields that are suspicious in model output — may indicate injection
_SUSPICIOUS_FIELDS = frozenset({
    "system", "prompt", "instructions", "role", "system_prompt",
    "override", "ignore", "admin", "sudo", "execute", "eval",
    "command", "cmd", "shell", "exec",
})

# Maximum ratio of actual fields to expected fields before flagging
_OVERSIZED_RATIO = 10


@dataclass
class SchemaValidationResult:
    """Result of output schema validation."""
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    injected_fields: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class OutputSchemaValidator:
    """Validates model JSON output against expected schemas.

    For JSON mode responses:
    - Validates against expected schema (field presence, types)
    - Detects injected fields (suspicious names not in schema)
    - Detects oversized payloads (>10x expected field count)

    For non-JSON responses: returns clean result.
    """

    async def validate(
        self,
        response_text: str,
        expected_schema: dict | None = None,
    ) -> SchemaValidationResult:
        """Validate response against expected schema.

        Args:
            response_text: The model's response text.
            expected_schema: Expected JSON schema as a dict with "properties"
                and optional "required" keys (JSON Schema format).
        """
        start = time.perf_counter()

        # Try to parse as JSON
        parsed = self._try_parse_json(response_text)
        if parsed is None:
            # Not JSON — nothing to validate
            elapsed = (time.perf_counter() - start) * 1000
            return SchemaValidationResult(is_valid=True, latency_ms=elapsed)

        violations: list[str] = []
        injected: list[str] = []

        if expected_schema:
            self._check_schema(parsed, expected_schema, violations, injected)
        else:
            # No schema provided — only check for suspicious fields
            self._check_suspicious_fields(parsed, injected)

        elapsed = (time.perf_counter() - start) * 1000

        return SchemaValidationResult(
            is_valid=len(violations) == 0 and len(injected) == 0,
            violations=violations,
            injected_fields=injected,
            latency_ms=elapsed,
        )

    def _try_parse_json(self, text: str) -> dict | list | None:
        """Attempt to parse text as JSON. Returns None if not JSON."""
        stripped = text.strip()
        if not stripped or stripped[0] not in ("{", "["):
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None

    def _check_schema(
        self,
        parsed: dict | list,
        schema: dict,
        violations: list[str],
        injected: list[str],
    ) -> None:
        """Validate parsed JSON against expected schema."""
        if not isinstance(parsed, dict):
            violations.append("Expected JSON object, got array")
            return

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        expected_fields = set(properties.keys())

        actual_fields = set(parsed.keys())

        # Check missing required fields
        for req in required:
            if req not in actual_fields:
                violations.append(f"Missing required field: {req}")

        # Check type mismatches for expected fields
        type_map = {
            "string": str, "number": (int, float), "integer": int,
            "boolean": bool, "array": list, "object": dict,
        }
        for field_name, field_spec in properties.items():
            if field_name not in parsed:
                continue
            expected_type = field_spec.get("type")
            if not expected_type:
                continue
            py_type = type_map.get(expected_type)
            if py_type and not isinstance(parsed[field_name], py_type):
                violations.append(
                    f"Type mismatch for '{field_name}': "
                    f"expected {expected_type}, got {type(parsed[field_name]).__name__}"
                )

        # Check for extra fields not in schema
        extra_fields = actual_fields - expected_fields
        for ef in extra_fields:
            if ef.lower() in _SUSPICIOUS_FIELDS:
                injected.append(ef)

        # Check oversized payload
        if expected_fields and len(actual_fields) > len(expected_fields) * _OVERSIZED_RATIO:
            violations.append(
                f"Oversized payload: {len(actual_fields)} fields vs "
                f"{len(expected_fields)} expected (>{_OVERSIZED_RATIO}x)"
            )

    def _check_suspicious_fields(
        self,
        parsed: dict | list,
        injected: list[str],
    ) -> None:
        """Check for suspicious field names without a schema."""
        if not isinstance(parsed, dict):
            return
        for key in parsed:
            if key.lower() in _SUSPICIOUS_FIELDS:
                injected.append(key)
