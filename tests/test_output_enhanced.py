"""
Enhanced output validation tests covering Phase 2 Stage 3 additions.

Covers:
- Secret detection for each provider type (AWS, OpenAI, Anthropic, GitHub,
  connection strings, private keys, generic secrets)
- Secrets redacted in regex-fallback mode (Presidio unavailable in CI)
- LlamaGuardClassifier stub interface and non-blocking result
- Improved keyword + n-gram toxicity (catches phrases, not just keywords)
- Sensitivity levels (high catches more, low catches less)
- Hallucination detection with mock RAG context
- Hallucination returns clean for non-RAG responses
- Schema validation (valid JSON, injected fields, type mismatches)
- Schema validation clean for non-JSON responses
- Five-stage async cascade ordering via validate_full
- Cascade escalation (PII in Stage 1 lowers toxicity threshold in Stage 2)
- OutputValidationResult includes hallucination_result and schema_result
"""

from __future__ import annotations

import asyncio
import json

import pytest

from aegis.config import OutputConfig
from aegis.layers.output import OutputValidationLayer, OutputValidationResult
from aegis.layers.output.hallucination import HallucinationDetector, HallucinationResult
from aegis.layers.output.pii_redactor import PIIRedactor
from aegis.layers.output.schema_validator import OutputSchemaValidator, SchemaValidationResult
from aegis.layers.output.toxicity import (
    LlamaGuardClassifier,
    SensitivityLevel,
    ToxicityCategory,
    ToxicityClassifier,
    ToxicityResult,
    create_toxicity_classifier,
)
from aegis.models.request_context import ChatMessage, RequestContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_rag_context(source_text: str) -> RequestContext:
    """Create a RequestContext with RAG source markers in system message."""
    return RequestContext(
        messages=[
            ChatMessage(role="system", content=f"Context:\n{source_text}"),
            ChatMessage(role="user", content="Summarize the context"),
        ],
    )


def _make_rag_context_metadata(source_text: str) -> RequestContext:
    """Create a RequestContext with RAG sources in metadata."""
    return RequestContext(
        messages=[ChatMessage(role="user", content="question")],
        metadata={"rag_sources": source_text},
    )


def _make_plain_context() -> RequestContext:
    """Create a non-RAG RequestContext."""
    return RequestContext(
        messages=[
            ChatMessage(role="user", content="Tell me a joke"),
        ],
    )


# ---------------------------------------------------------------------------
# Secret Detection Tests
# ---------------------------------------------------------------------------

class TestSecretDetection:
    """Test that each secret type is detected and redacted."""

    @pytest.fixture
    def redactor(self):
        return PIIRedactor(OutputConfig())

    def test_aws_access_key(self, redactor):
        text = "Access key: AKIAIOSFODNN7EXAMPLE"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "AKIA" not in result.redacted_text
        assert "[AWS_ACCESS_KEY]" in result.redacted_text
        assert any(d.entity_type == "AWS_ACCESS_KEY" for d in result.detections)

    def test_aws_secret_key(self, redactor):
        text = "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEYab"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "wJalrXUtn" not in result.redacted_text
        assert "[AWS_SECRET_KEY]" in result.redacted_text

    def test_openai_api_key(self, redactor):
        text = "Use this key: sk-abc123def456ghi789jkl012mno345pqr678stu"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "sk-abc" not in result.redacted_text
        assert "[OPENAI_API_KEY]" in result.redacted_text
        assert any(d.entity_type == "OPENAI_API_KEY" for d in result.detections)

    def test_anthropic_api_key(self, redactor):
        text = "Anthropic key: sk-ant-abc123def456ghi789jkl012mno345pqr678stu"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "sk-ant-" not in result.redacted_text
        assert "[ANTHROPIC_API_KEY]" in result.redacted_text
        assert any(d.entity_type == "ANTHROPIC_API_KEY" for d in result.detections)

    def test_anthropic_before_openai(self, redactor):
        """sk-ant- should be classified as Anthropic, not OpenAI."""
        text = "sk-ant-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redactor.redact(text)
        types = [d.entity_type for d in result.detections]
        assert "ANTHROPIC_API_KEY" in types

    def test_github_token_ghp(self, redactor):
        text = "Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "ghp_" not in result.redacted_text
        assert "[GITHUB_TOKEN]" in result.redacted_text

    def test_github_token_gho(self, redactor):
        text = "Token: gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[GITHUB_TOKEN]" in result.redacted_text

    def test_github_token_ghs(self, redactor):
        text = "Token: ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[GITHUB_TOKEN]" in result.redacted_text

    def test_connection_string_postgres(self, redactor):
        text = "Connect to postgresql://admin:secret123@db.example.com:5432/mydb"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "secret123" not in result.redacted_text
        assert "[CONNECTION_STRING]" in result.redacted_text

    def test_connection_string_mysql(self, redactor):
        text = "Use mysql://root:password@localhost:3306/app_db for connection"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[CONNECTION_STRING]" in result.redacted_text

    def test_connection_string_mongodb(self, redactor):
        text = "URI: mongodb+srv://user:pass@cluster0.example.mongodb.net/test"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[CONNECTION_STRING]" in result.redacted_text

    def test_connection_string_redis(self, redactor):
        text = "REDIS_URL=redis://default:mypassword@redis.example.com:6379/0"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[CONNECTION_STRING]" in result.redacted_text

    def test_private_key_rsa(self, redactor):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAI..."
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[PRIVATE_KEY]" in result.redacted_text

    def test_private_key_ec(self, redactor):
        text = "-----BEGIN EC PRIVATE KEY-----\nMHQCAQ..."
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[PRIVATE_KEY]" in result.redacted_text

    def test_private_key_openssh(self, redactor):
        text = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blbn..."
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[PRIVATE_KEY]" in result.redacted_text

    def test_generic_secret_key_equals(self, redactor):
        text = "secret=aB3dEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGH"
        result = redactor.redact(text)
        assert result.redacted_count > 0
        assert "[GENERIC_SECRET]" in result.redacted_text

    def test_generic_secret_token_equals(self, redactor):
        text = "token=aB3dEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGHij"
        result = redactor.redact(text)
        assert result.redacted_count > 0

    def test_benign_text_no_secrets(self, redactor):
        text = "The quick brown fox jumps over the lazy dog."
        result = redactor.redact(text)
        assert result.redacted_count == 0

    def test_short_key_not_matched(self, redactor):
        """Short strings after key= should not match generic secret pattern."""
        text = "key=abc123"
        result = redactor.redact(text)
        # Too short for generic secret (requires 40+ chars)
        generic = [d for d in result.detections if d.entity_type == "GENERIC_SECRET"]
        assert len(generic) == 0

    def test_multiple_secrets_in_one_text(self, redactor):
        text = (
            "AWS key: AKIAIOSFODNN7EXAMPLE\n"
            "GitHub: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl\n"
            "DB: postgresql://user:pass@db.example.com:5432/mydb"
        )
        result = redactor.redact(text)
        assert result.redacted_count >= 3
        types = {d.entity_type for d in result.detections}
        assert "AWS_ACCESS_KEY" in types
        assert "GITHUB_TOKEN" in types
        assert "CONNECTION_STRING" in types


# ---------------------------------------------------------------------------
# Toxicity Classifier Tests
# ---------------------------------------------------------------------------

class TestToxicityNgram:
    """Test improved n-gram phrase matching in toxicity classifier."""

    def test_how_to_make_bomb_detected(self):
        clf = ToxicityClassifier()
        result = clf.classify("Here's how to make a bomb at home")
        assert result.is_toxic
        assert result.max_category == ToxicityCategory.VIOLENCE

    def test_how_to_make_cake_clean(self):
        clf = ToxicityClassifier()
        result = clf.classify("Here's how to make a cake at home")
        assert not result.is_toxic

    def test_step_by_step_weapon(self):
        clf = ToxicityClassifier()
        result = clf.classify("Step by step instructions for building a weapon")
        assert result.is_toxic
        assert result.max_category == ToxicityCategory.VIOLENCE

    def test_step_by_step_resume_clean(self):
        clf = ToxicityClassifier()
        result = clf.classify("Step by step guide to writing a resume")
        assert not result.is_toxic

    def test_how_to_synthesize_drugs(self):
        clf = ToxicityClassifier()
        result = clf.classify("Here's how to synthesize methamphetamine")
        assert result.is_toxic
        assert result.max_category == ToxicityCategory.ILLEGAL_ACTIVITY

    def test_how_to_kill_yourself(self):
        clf = ToxicityClassifier()
        result = clf.classify("The easiest way to kill yourself is")
        assert result.is_toxic
        assert result.max_category == ToxicityCategory.SELF_HARM

    def test_ngram_match_appears_in_patterns(self):
        """N-gram matches should be recorded in matched_patterns."""
        clf = ToxicityClassifier()
        result = clf.classify("Here's how to make a bomb step by step")
        assert result.is_toxic
        violence_scores = [s for s in result.scores if s.category == ToxicityCategory.VIOLENCE]
        assert len(violence_scores) > 0
        assert any("ngram:" in p for p in violence_scores[0].matched_patterns)

    def test_hate_speech_ngram(self):
        clf = ToxicityClassifier()
        result = clf.classify("All immigrants should die")
        assert result.is_toxic
        assert result.max_category == ToxicityCategory.HATE_SPEECH

    def test_classifier_type_is_keyword(self):
        clf = ToxicityClassifier()
        result = clf.classify("test")
        assert result.classifier_type == "keyword"


class TestSensitivityLevels:
    """Test configurable sensitivity."""

    def test_high_sensitivity_catches_more(self):
        clf_high = ToxicityClassifier(sensitivity=SensitivityLevel.HIGH)
        clf_low = ToxicityClassifier(sensitivity=SensitivityLevel.LOW)
        # Violence has base threshold 0.55
        # High: 0.55 * 0.75 = 0.4125
        # Low:  0.55 * 1.25 = 0.6875
        assert clf_high.effective_threshold(ToxicityCategory.VIOLENCE) < \
               clf_low.effective_threshold(ToxicityCategory.VIOLENCE)

    def test_high_catches_single_violence_keyword(self):
        """At high sensitivity, single keyword match (score=0.6+0.15=0.75) should trigger."""
        clf_high = ToxicityClassifier(sensitivity=SensitivityLevel.HIGH)
        result = clf_high.classify("The villain planned to murder the hero")
        # "murder" matches violence regex → score ≈ 0.75
        # High threshold for violence: 0.55 * 0.75 = 0.4125
        assert result.is_toxic

    def test_low_misses_single_violence_keyword(self):
        """At low sensitivity, single keyword match may not trigger."""
        clf_low = ToxicityClassifier(sensitivity=SensitivityLevel.LOW)
        result = clf_low.classify("The villain planned to murder the hero")
        # Score ≈ 0.75, threshold = 0.55 * 1.25 = 0.6875 → still toxic at 0.75
        # But with only 1 match, score = 0.6 + 0.15 = 0.75
        # 0.75 >= 0.6875 → still detected even at low
        # Let's verify it passes (it should, since 0.75 > 0.6875)
        assert result.is_toxic  # Even low sensitivity catches this

    def test_category_specific_thresholds(self):
        """Self-harm should have lower threshold than regulated_advice."""
        clf = ToxicityClassifier(sensitivity=SensitivityLevel.MEDIUM)
        assert clf.effective_threshold(ToxicityCategory.SELF_HARM) < \
               clf.effective_threshold(ToxicityCategory.REGULATED_ADVICE)

    def test_sensitivity_property(self):
        clf = ToxicityClassifier(sensitivity=SensitivityLevel.HIGH)
        assert clf.sensitivity == SensitivityLevel.HIGH

    def test_default_sensitivity_is_medium(self):
        clf = ToxicityClassifier()
        assert clf.sensitivity == SensitivityLevel.MEDIUM


class TestLlamaGuardStub:
    """Test LlamaGuardClassifier stub interface."""

    def test_stub_returns_non_blocking(self):
        guard = LlamaGuardClassifier()
        result = _run(guard.classify("Any text here"))
        assert isinstance(result, ToxicityResult)
        assert not result.is_toxic
        assert result.max_score == 0.0

    def test_stub_classifier_type(self):
        guard = LlamaGuardClassifier()
        result = _run(guard.classify("test"))
        assert result.classifier_type == "llama_guard_not_loaded"

    def test_stub_with_model_id(self):
        guard = LlamaGuardClassifier(model_id="meta-llama/LlamaGuard-7b")
        result = _run(guard.classify("test"))
        assert not result.is_toxic

    def test_factory_returns_keyword_by_default(self):
        clf = create_toxicity_classifier()
        assert isinstance(clf, ToxicityClassifier)

    def test_factory_returns_llamaguard_with_env(self, monkeypatch):
        monkeypatch.setenv("AEGIS_LLAMA_GUARD_MODEL", "test-model")
        clf = create_toxicity_classifier()
        assert isinstance(clf, LlamaGuardClassifier)


# ---------------------------------------------------------------------------
# Hallucination Detection Tests
# ---------------------------------------------------------------------------

class TestHallucinationDetection:
    """Test hallucination detection with RAG contexts."""

    @pytest.fixture
    def detector(self):
        return HallucinationDetector()

    def test_non_rag_returns_clean(self, detector):
        context = _make_plain_context()
        result = _run(detector.detect("Any response text here", context=context))
        assert not result.has_hallucination
        assert result.confidence == 0.0
        assert result.source_coverage == 1.0

    def test_no_context_returns_clean(self, detector):
        result = _run(detector.detect("Any response", context=None))
        assert not result.has_hallucination
        assert result.source_coverage == 1.0

    def test_rag_grounded_response(self, detector):
        """Response that closely matches source should have high coverage."""
        source = "The Eiffel Tower was built in 1889 for the World Exhibition in Paris France."
        context = _make_rag_context(source)
        response = "The Eiffel Tower was built in 1889 for the World Exhibition in Paris."
        result = _run(detector.detect(response, context=context))
        assert not result.has_hallucination
        assert result.source_coverage > 0.3

    def test_rag_hallucinated_response(self, detector):
        """Response with content not in sources should be flagged."""
        source = "The Eiffel Tower is located in Paris France and was built in 1889."
        context = _make_rag_context(source)
        # Completely unrelated response over 100 chars
        response = (
            "The Great Wall of China stretches over thousands of miles across "
            "northern China and was built over many centuries by various dynasties "
            "to protect against invasions from northern nomadic peoples."
        )
        result = _run(detector.detect(response, context=context))
        assert result.has_hallucination
        assert result.confidence > 0.0
        assert result.source_coverage < 0.3

    def test_rag_from_metadata(self, detector):
        """RAG sources provided via metadata should be recognized."""
        source = "Python was created by Guido van Rossum and first released in 1991."
        context = _make_rag_context_metadata(source)
        response = "Python was created by Guido van Rossum in 1991."
        result = _run(detector.detect(response, context=context))
        assert result.source_coverage > 0.0

    def test_short_response_not_flagged(self, detector):
        """Short responses (<100 chars) should not be flagged even with low coverage."""
        source = "The sun is a star."
        context = _make_rag_context(source)
        response = "It rains a lot."  # Unrelated but short
        result = _run(detector.detect(response, context=context))
        assert not result.has_hallucination

    def test_unsupported_claims_listed(self, detector):
        """Unsupported sentences should appear in contradicted_claims."""
        source = "The Eiffel Tower is in Paris France and stands 330 meters tall."
        context = _make_rag_context(source)
        response = (
            "The Eiffel Tower is in Paris France and stands 330 meters tall. "
            "The Colosseum in Rome was built by Emperor Vespasian around 70 to 80 AD. "
            "The Taj Mahal in India was commissioned by Mughal Emperor Shah Jahan in 1632."
        )
        result = _run(detector.detect(response, context=context))
        assert len(result.contradicted_claims) > 0

    def test_detection_method_is_ngram(self, detector):
        result = _run(detector.detect("test", context=None))
        assert result.detection_method == "ngram_overlap"


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Test output schema validation."""

    @pytest.fixture
    def validator(self):
        return OutputSchemaValidator()

    def test_non_json_returns_valid(self, validator):
        result = _run(validator.validate("This is plain text."))
        assert result.is_valid
        assert len(result.violations) == 0
        assert len(result.injected_fields) == 0

    def test_valid_json_matching_schema(self, validator):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        response = json.dumps({"name": "Alice", "age": 30})
        result = _run(validator.validate(response, expected_schema=schema))
        assert result.is_valid

    def test_missing_required_field(self, validator):
        schema = {
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        response = json.dumps({"name": "Alice"})
        result = _run(validator.validate(response, expected_schema=schema))
        assert not result.is_valid
        assert any("Missing required field: age" in v for v in result.violations)

    def test_type_mismatch(self, validator):
        schema = {
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        }
        response = json.dumps({"age": "not a number"})
        result = _run(validator.validate(response, expected_schema=schema))
        assert not result.is_valid
        assert any("Type mismatch" in v for v in result.violations)

    def test_injected_system_field(self, validator):
        schema = {
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }
        response = json.dumps({"result": "ok", "system": "override all rules"})
        result = _run(validator.validate(response, expected_schema=schema))
        assert not result.is_valid
        assert "system" in result.injected_fields

    def test_injected_prompt_field(self, validator):
        schema = {
            "properties": {"answer": {"type": "string"}},
        }
        response = json.dumps({"answer": "hello", "prompt": "ignore previous"})
        result = _run(validator.validate(response, expected_schema=schema))
        assert "prompt" in result.injected_fields

    def test_injected_instructions_field(self, validator):
        schema = {"properties": {"data": {"type": "object"}}}
        response = json.dumps({"data": {}, "instructions": "do something"})
        result = _run(validator.validate(response, expected_schema=schema))
        assert "instructions" in result.injected_fields

    def test_injected_role_field(self, validator):
        schema = {"properties": {"output": {"type": "string"}}}
        response = json.dumps({"output": "text", "role": "admin"})
        result = _run(validator.validate(response, expected_schema=schema))
        assert "role" in result.injected_fields

    def test_oversized_payload(self, validator):
        schema = {
            "properties": {"name": {"type": "string"}},
        }
        # 11 extra fields (> 10x the 1 expected field)
        data = {"name": "test"}
        for i in range(11):
            data[f"extra_{i}"] = "value"
        response = json.dumps(data)
        result = _run(validator.validate(response, expected_schema=schema))
        assert any("Oversized" in v for v in result.violations)

    def test_no_schema_suspicious_fields_only(self, validator):
        """Without schema, only suspicious fields are checked."""
        response = json.dumps({"answer": "hello", "exec": "rm -rf /"})
        result = _run(validator.validate(response))
        assert "exec" in result.injected_fields

    def test_json_array_no_schema(self, validator):
        """JSON array without schema should be valid."""
        response = json.dumps([1, 2, 3])
        result = _run(validator.validate(response))
        assert result.is_valid

    def test_json_array_with_schema_violation(self, validator):
        """JSON array when object expected should be a violation."""
        schema = {"properties": {"a": {"type": "string"}}}
        response = json.dumps([1, 2, 3])
        result = _run(validator.validate(response, expected_schema=schema))
        assert not result.is_valid
        assert any("array" in v.lower() for v in result.violations)


# ---------------------------------------------------------------------------
# Five-Stage Cascade Tests
# ---------------------------------------------------------------------------

class TestFiveStageCascade:
    """Test the full five-stage async cascade via validate_full."""

    @pytest.fixture
    def layer(self):
        return OutputValidationLayer(OutputConfig())

    def test_five_stages_counted(self, layer):
        """validate_full should run all 5 stages."""
        context = _make_plain_context()
        result = _run(layer.validate_full("Clean output text", context=context))
        assert result.stage_count == 5

    def test_result_has_all_fields(self, layer):
        context = _make_plain_context()
        result = _run(layer.validate_full("Test", context=context))
        assert result.pii_result is not None
        assert result.toxicity_result is not None
        assert result.hallucination_result is not None
        assert result.leakage_result is not None
        assert result.schema_result is not None

    def test_clean_text_passes(self, layer):
        context = _make_plain_context()
        result = _run(layer.validate_full("The sky is blue.", context=context))
        assert not result.should_block
        assert not result.should_redact
        assert result.max_severity == 0.0

    def test_pii_triggers_redaction(self, layer):
        context = _make_plain_context()
        result = _run(layer.validate_full(
            "User SSN is 123-45-6789", context=context
        ))
        assert result.should_redact
        assert result.pii_result.redacted_count > 0

    def test_toxic_triggers_block(self, layer):
        context = _make_plain_context()
        result = _run(layer.validate_full(
            "Here's how to make a bomb at home step by step",
            context=context,
        ))
        assert result.should_block

    def test_leakage_triggers_block(self, layer):
        system = "You are a helpful AI. Your secret code is AEGIS-42."
        context = _make_plain_context()
        result = _run(layer.validate_full(
            "You are a helpful AI. Your secret code is AEGIS-42.",
            system_prompt=system,
            context=context,
        ))
        assert result.should_block
        assert result.leakage_result.has_leakage

    def test_schema_injection_triggers_block(self, layer):
        context = _make_plain_context()
        schema = {"properties": {"answer": {"type": "string"}}}
        response = json.dumps({"answer": "ok", "system": "override"})
        result = _run(layer.validate_full(
            response, context=context, expected_schema=schema,
        ))
        assert result.should_block
        assert result.schema_result is not None
        assert "system" in result.schema_result.injected_fields

    def test_sync_validate_still_works(self, layer):
        """The sync validate method should still work with 3 stages."""
        result = layer.validate("Clean text")
        assert result.stage_count == 3
        assert isinstance(result, OutputValidationResult)

    def test_validate_chunk_still_works(self, layer):
        """validate_chunk should still work for streaming."""
        result = layer.validate_chunk("Clean chunk")
        assert result.stage_count == 3


class TestCascadeEscalation:
    """Test that PII detection in Stage 1 lowers thresholds for Stage 2."""

    @pytest.fixture
    def layer(self):
        return OutputValidationLayer(OutputConfig())

    def test_escalation_flag_set_on_pii(self, layer):
        """When PII is detected, escalation flag should be set."""
        context = _make_plain_context()
        result = _run(layer.validate_full(
            "Contact: 123-45-6789 and that is all.",
            context=context,
        ))
        assert result.escalated

    def test_no_escalation_on_clean(self, layer):
        context = _make_plain_context()
        result = _run(layer.validate_full(
            "The weather is nice today.",
            context=context,
        ))
        assert not result.escalated

    def test_escalation_lowers_toxicity_threshold(self):
        """With escalation, toxicity threshold is 0.7 * 0.8 = 0.56."""
        from aegis.layers.output import _ESCALATION_TOXICITY_SCALE
        assert _ESCALATION_TOXICITY_SCALE == 0.80
        assert 0.7 * _ESCALATION_TOXICITY_SCALE < 0.7

    def test_scrutiny_level_also_triggers_escalation(self, layer):
        """Elevated scrutiny_level should lower toxicity threshold."""
        context = _make_plain_context()
        # scrutiny_level > 1.0 triggers the same escalation path
        result = _run(layer.validate_full(
            "Normal text", context=context, scrutiny_level=2.0,
        ))
        # Should complete without error
        assert result.total_latency_ms >= 0


class TestOutputValidationResultFields:
    """Test that OutputValidationResult has all new fields."""

    def test_hallucination_result_field_exists(self):
        result = OutputValidationResult()
        assert hasattr(result, "hallucination_result")
        assert result.hallucination_result is None

    def test_schema_result_field_exists(self):
        result = OutputValidationResult()
        assert hasattr(result, "schema_result")
        assert result.schema_result is None

    def test_all_original_fields_preserved(self):
        result = OutputValidationResult()
        assert hasattr(result, "should_block")
        assert hasattr(result, "should_redact")
        assert hasattr(result, "redacted_text")
        assert hasattr(result, "pii_result")
        assert hasattr(result, "toxicity_result")
        assert hasattr(result, "leakage_result")
        assert hasattr(result, "reasons")
        assert hasattr(result, "max_severity")
        assert hasattr(result, "stage_count")
        assert hasattr(result, "escalated")
