"""
AEGIS Configuration — Every parameter from the architecture specification.

ASSUMED-BREACH POSTURE: Configuration is loaded from environment variables,
never from the request path or client-supplied headers. A compromised
environment variable source (e.g., a leaked .env file) would allow an
attacker to weaken thresholds, disable layers, or widen rate limits. In
production, secrets must come from a vault (AWS Secrets Manager, HashiCorp
Vault) and config should be immutable after startup. The validate() method
checks for obviously dangerous configurations (thresholds of 0, disabled
layers) and refuses to start.

All defaults match the architecture specification. Override via environment
variables prefixed with AEGIS_.
"""

from __future__ import annotations

import enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ThreatLevel(int, enum.Enum):
    """Threat Level Indicator (TLI) — 5-level scale from CLAUDE.md Section 3 L6.

    GREEN  = Normal operations, standard thresholds
    BLUE   = Elevated detection rate (>2x baseline), lower block thresholds 10%
    YELLOW = Confirmed novel attack, full slow-path on all requests
    ORANGE = Active campaign across tenants, maximum detection sensitivity
    RED    = Critical threat or zero-day, fail-closed pending human clearance
    """
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    ORANGE = 4
    RED = 5


class CircuitBreakerState(str, enum.Enum):
    """Circuit breaker states per L7 Self-Healing spec."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# L1 — Barrier Layer Configuration
# ---------------------------------------------------------------------------

class BarrierConfig(BaseSettings):
    """L1 Barrier Layer — Skin / Mucous Membranes.

    Defends against: Unauthenticated access, volumetric DoS, malformed
    payloads, oversized requests, expired sessions.

    Assumes compromised: Nothing upstream — this is the outermost layer.
    Assumes the network, the client, and the TLS termination proxy may
    all be adversarial.
    """
    model_config = {"env_prefix": "AEGIS_BARRIER_"}

    max_tokens_per_request: int = Field(
        default=128_000,
        description="Maximum input token count before rejection",
    )
    rate_limit_rpm: int = Field(
        default=60,
        description="Requests per minute per API key (configurable per tenant)",
    )
    rate_limit_burst: int = Field(
        default=10,
        description="Burst allowance above steady-state rate limit",
    )
    max_request_size_bytes: int = Field(
        default=10_485_760,  # 10 MB
        description="Maximum request payload size in bytes",
    )
    tls_min_version: str = Field(
        default="1.3",
        description="Minimum TLS version accepted",
    )
    mtls_required: bool = Field(
        default=False,
        description="Require mutual TLS client certificate (enterprise)",
    )
    ip_reputation_enabled: bool = Field(
        default=True,
        description="Check source IP against threat intelligence feeds",
    )
    session_timeout_seconds: int = Field(
        default=3600,
        description="Session expiry for behavioral tracking",
    )


# ---------------------------------------------------------------------------
# L2 — Innate Detection Configuration
# ---------------------------------------------------------------------------

class InnateConfig(BaseSettings):
    """L2 Innate Detection — Pattern Recognition Receptors / NK Cells.

    Defends against: Known injection patterns, blocklisted payloads,
    malformed schemas, token stuffing, PII in input, canary tampering.

    Assumes compromised: L1 Barrier. Malformed requests, missing auth,
    exceeded rate limits may all be present.
    """
    model_config = {"env_prefix": "AEGIS_INNATE_"}

    block_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for immediate blocking",
    )
    alert_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for priority slow-path flagging",
    )
    max_scanner_latency_ms: float = Field(
        default=5.0,
        description="Maximum allowed latency for the innate fast path",
    )
    regex_pattern_file: str = Field(
        default="data/patterns.json",
        description="Path to MITRE ATLAS-tagged regex pattern library",
    )
    blocklist_file: str = Field(
        default="data/blocklist.txt",
        description="Path to curated malicious payload blocklist",
    )
    bloom_filter_fp_rate: float = Field(
        default=0.0001,
        description="Bloom filter false positive rate for blocklist (0.01%)",
    )
    canary_token_enabled: bool = Field(
        default=True,
        description="Enable canary token verification in system prompts",
    )

    @field_validator("block_threshold")
    @classmethod
    def block_above_alert(cls, v: float, info) -> float:
        alert = info.data.get("alert_threshold", 0.50)
        if v <= alert:
            raise ValueError(
                f"block_threshold ({v}) must be greater than "
                f"alert_threshold ({alert})"
            )
        return v


# ---------------------------------------------------------------------------
# Canary Token Configuration
# ---------------------------------------------------------------------------

class CanaryConfig(BaseSettings):
    """Canary Token Verifier — NK Cell analog.

    AEGIS injects HMAC-SHA256 canary tokens into system prompts before
    forwarding. The verifier checks responses for canary presence and
    integrity. Missing/modified canary = system prompt tampering.

    ASSUMED-BREACH POSTURE: The canary secret key could be leaked. An
    attacker who knows the key can forge valid canary tokens. However, this
    only neutralizes canary verification — all other layers remain active.
    The secret key is never included in API responses or logs.
    """
    model_config = {"env_prefix": "AEGIS_CANARY_"}

    enabled: bool = Field(
        default=True,
        description="Enable canary token injection and verification",
    )
    # REQUIRED IN PRODUCTION: Set via AEGIS_CANARY_SECRET_KEY env var.
    # The default value is intentionally weak — it MUST be replaced with
    # a cryptographically random secret stored in a secrets manager.
    secret_key: str = Field(
        default="aegis-canary-default-secret-change-in-production",
        description="HMAC-SHA256 secret key for canary token generation",
    )
    token_format: str = Field(
        default="html_comment",
        description="Canary format: 'html_comment' or 'unicode_tag'",
    )
    verification_on_output: bool = Field(
        default=True,
        description="Check model responses for canary token presence/tampering",
    )


# ---------------------------------------------------------------------------
# L3 — Adaptive Analysis Configuration
# ---------------------------------------------------------------------------

class AdaptiveConfig(BaseSettings):
    """L3 Adaptive Analysis — T-Cells / B-Cells / Clonal Selection.

    Defends against: Paraphrased injections, semantic attacks, behavioral
    anomalies, multi-turn escalation, novel zero-day patterns.

    Assumes compromised: L1 Barrier AND L2 Innate. Attacks evading regex,
    blocklists, and schema validation are expected — that is this layer's
    entire purpose.
    """
    model_config = {"env_prefix": "AEGIS_ADAPTIVE_"}

    injection_model: str = Field(
        default="ProtectAI/deberta-v3-base-prompt-injection-v2",
        description="HuggingFace model ID for prompt injection classification",
    )
    injection_model_onnx: bool = Field(
        default=True,
        description="Use ONNX-optimized inference (~20ms vs ~80ms)",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model for semantic embeddings (384-dim)",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for Threat Vault match",
    )
    behavioral_psi_threshold: float = Field(
        default=0.25,
        description="Population Stability Index threshold for significant drift",
    )
    multi_turn_window: int = Field(
        default=10,
        description="Sliding window size (turns) for multi-turn sequence analysis",
    )
    mcav_anomaly_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="DCA Mature Context Antigen Value threshold for anomalous",
    )
    mcav_threat_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="DCA MCAV threshold for definitive threat",
    )
    max_latency_ms: float = Field(
        default=50.0,
        description="Maximum allowed latency for adaptive slow path",
    )


# ---------------------------------------------------------------------------
# L4 — Immune Memory Configuration
# ---------------------------------------------------------------------------

class MemoryConfig(BaseSettings):
    """L4 Immune Memory — Threat Vault / Antibody Library.

    Defends against: Repeated attacks, variant attacks semantically similar
    to known threats, attacks matching STIX/TAXII intelligence feeds.

    Assumes compromised: The vault itself could be poisoned by a compromised
    L3 that injects false embeddings. Unconfirmed entries carry reduced
    weight. Provenance is tracked for every entry.
    """
    model_config = {"env_prefix": "AEGIS_MEMORY_"}

    faiss_index_type: str = Field(
        default="HNSW",
        description="FAISS index type — HNSW for sub-ms ANN at 1M+ vectors",
    )
    faiss_dimension: int = Field(
        default=384,
        description="Embedding dimension (must match embedding_model output)",
    )
    faiss_hnsw_m: int = Field(
        default=32,
        description="HNSW M parameter — connections per node (higher = better recall, more memory)",
    )
    faiss_ef_search: int = Field(
        default=64,
        description="HNSW ef_search — search-time accuracy/speed tradeoff",
    )
    acute_memory_days: int = Field(
        default=30,
        description="Days before acute memory promotes to persistent",
    )
    dormant_memory_days: int = Field(
        default=180,
        description="Days unseen before persistent memory becomes dormant",
    )
    seed_threats_file: str = Field(
        default="data/seed_threats.json",
        description="Path to initial threat vault seed data",
    )
    faiss_index_path: str = Field(
        default="data/threat_vault.faiss",
        description="Path to persist FAISS index on disk",
    )
    metadata_path: str = Field(
        default="data/threat_vault_meta.json",
        description="Path to persist indicator metadata sidecar JSON",
    )
    persist_every_n_updates: int = Field(
        default=5,
        description="Batch persistence: save index to disk every N additions",
    )
    maintenance_interval_hours: int = Field(
        default=6,
        description="Hours between maintenance cycle runs (lifecycle transitions)",
    )
    promote_min_sources: int = Field(
        default=3,
        description="Minimum distinct sources/tenants before promoting acute to persistent",
    )
    benign_prompts_file: str = Field(
        default="data/benign_prompts.json",
        description="Path to benign prompts dataset for clonal selection affinity testing",
    )
    signature_affinity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum affinity score for a generated signature to be promoted",
    )
    signature_max_fpr: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum false positive rate before auto-deprecating a signature",
    )
    signature_deprecation_min_matches: int = Field(
        default=20,
        description="Minimum match count before evaluating a signature for deprecation",
    )
    signature_num_candidates: int = Field(
        default=15,
        description="Number of candidate regex patterns to generate per attack (10-20)",
    )
    signature_max_promoted: int = Field(
        default=3,
        description="Maximum number of candidate signatures promoted per attack",
    )
    dp_epsilon: float = Field(
        default=3.0,
        description="Differential privacy epsilon for shared threat embeddings",
    )


# ---------------------------------------------------------------------------
# L5 — Output Validation Configuration
# ---------------------------------------------------------------------------

class OutputConfig(BaseSettings):
    """L5 Output Validation — Complement System Cascade.

    Defends against: PII leakage in model responses, toxic content,
    hallucinations, system prompt echo, data exfiltration, schema violations.

    Assumes compromised: The upstream model itself. The model may be actively
    exfiltrating data, leaking system prompts, or producing steganographic
    output. Also assumes all input layers (L1-L3) were bypassed.
    """
    model_config = {"env_prefix": "AEGIS_OUTPUT_"}

    pii_redaction_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Presidio confidence threshold for PII redaction",
    )
    pii_alert_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Presidio confidence threshold for PII logging (no redaction)",
    )
    streaming_window_size: int = Field(
        default=128,
        description="Token window size for streaming chunk validation (128-256)",
    )
    streaming_window_overlap: float = Field(
        default=0.5,
        description="Fraction of window to retain as overlap for context",
    )
    toxicity_enabled: bool = Field(
        default=True,
        description="Enable toxicity/safety classification on output",
    )
    hallucination_enabled: bool = Field(
        default=False,
        description="Enable NLI hallucination detection (requires RAG context)",
    )
    leakage_ngram_size: int = Field(
        default=4,
        description="N-gram size for system prompt echo detection",
    )
    leakage_overlap_threshold: float = Field(
        default=0.3,
        description="N-gram overlap ratio to flag system prompt leakage",
    )


# ---------------------------------------------------------------------------
# L6 — Policy Engine Configuration
# ---------------------------------------------------------------------------

class PolicyConfig(BaseSettings):
    """L6 Policy Engine — Regulatory T-Cells / Immune Tolerance.

    Defends against: Autoimmune overreaction (blocking legitimate traffic),
    under-reaction (missing attacks), inconsistent enforcement across tenants.

    Assumes compromised: All detection layers. The policy engine receives
    threat scores from L2 and L3 but independently evaluates context,
    tenant policy, and threat level before rendering a decision. A
    compromised detection layer reporting false all-clear does not override
    policy rules (e.g., hard safety limits still apply).
    """
    model_config = {"env_prefix": "AEGIS_POLICY_"}

    default_threat_level: ThreatLevel = Field(
        default=ThreatLevel.GREEN,
        description="Initial threat level indicator on startup",
    )
    blue_escalation_multiplier: float = Field(
        default=2.0,
        description="Detection rate multiplier (vs baseline) to trigger BLUE",
    )
    block_threshold_reduction_blue: float = Field(
        default=0.10,
        description="Fraction to lower block thresholds when BLUE",
    )
    enable_human_review_at: ThreatLevel = Field(
        default=ThreatLevel.YELLOW,
        description="TLI level at which human review queue is activated",
    )
    fail_closed_at: ThreatLevel = Field(
        default=ThreatLevel.RED,
        description="TLI level at which all requests are blocked pending human review",
    )
    backend: str = Field(
        default="python",
        description="Policy engine backend: 'python' (built-in) or 'opa' (Open Policy Agent)",
    )
    opa_url: str = Field(
        default="http://localhost:8181",
        description="OPA server URL (used when backend='opa')",
    )


# ---------------------------------------------------------------------------
# L7 — Self-Healing Configuration
# ---------------------------------------------------------------------------

class HealingConfig(BaseSettings):
    """L7 Self-Healing — Wound Healing / Tissue Repair.

    Defends against: Sustained attacks degrading model availability,
    compromised model endpoints, cascading failures.

    Assumes compromised: The primary model endpoint. The circuit breaker
    treats the upstream model as potentially hostile — elevated error rates
    or threat detections trigger automatic fallback. The healing layer does
    not trust that the model is merely "having a bad day"; it assumes active
    compromise until probes confirm otherwise.
    """
    model_config = {"env_prefix": "AEGIS_HEALING_"}

    circuit_breaker_threshold: float = Field(
        default=0.50,
        description="Failure rate (0-1) over sliding window to trip breaker",
    )
    circuit_breaker_window_seconds: int = Field(
        default=60,
        description="Sliding window duration for failure rate calculation",
    )
    cooldown_seconds: int = Field(
        default=30,
        description="Cooldown period in OPEN state before probing",
    )
    max_cooldown_seconds: int = Field(
        default=300,
        description="Maximum cooldown after exponential backoff (5 minutes)",
    )
    probe_count: int = Field(
        default=5,
        description="Number of probe requests in HALF_OPEN state",
    )
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Priority-ordered list of fallback model endpoint URLs",
    )
    sanitization_enabled: bool = Field(
        default=True,
        description="Enable prompt sanitization for low-confidence detections",
    )
    quarantine_threshold: int = Field(
        default=3,
        description="Adversarial events per session before quarantine",
    )


# ---------------------------------------------------------------------------
# L8 — Federated Intelligence Configuration (production only)
# ---------------------------------------------------------------------------

class FederatedConfig(BaseSettings):
    """L8 Federated Threat Intelligence — Herd Immunity.

    Defends against: Novel attacks not yet seen by this instance, global
    threat campaigns, zero-days discovered by other AEGIS deployments.

    Assumes compromised: Other AEGIS instances in the federation. Gradient
    updates are protected with differential privacy (epsilon=3) to prevent
    gradient inversion attacks. Shared indicators are anonymized. A
    compromised peer cannot poison the global model beyond the DP bound.

    NOTE: Deferred to production. Bootstrap implements L1-L7 only.
    """
    model_config = {"env_prefix": "AEGIS_FEDERATED_"}

    enabled: bool = Field(
        default=False,
        description="Enable federated intelligence (requires multiple deployments)",
    )
    aggregation_url: str = Field(
        default="",
        description="URL of the FedAvg aggregation server",
    )
    model_update_interval_hours: int = Field(
        default=6,
        description="Hours between federated model weight updates",
    )
    indicator_share_interval_minutes: int = Field(
        default=15,
        description="Minutes between STIX/TAXII indicator distribution",
    )
    dp_epsilon: float = Field(
        default=3.0,
        description="Differential privacy epsilon for gradient updates",
    )


# ---------------------------------------------------------------------------
# Multimodal Security Configuration
# ---------------------------------------------------------------------------

class MultimodalConfig(BaseSettings):
    """Multimodal Security — Image scanning, document scanning, text extraction.

    Defends against: Prompt injections embedded in images or documents,
    steganographic payloads, metadata-based attacks, adversarial image
    perturbations, hidden text injection in documents, polyglot files,
    macro-enabled documents.

    Assumes compromised: L1 Barrier. Images and documents in multimodal
    requests may contain injections that bypass text-only L2 scanners.
    This layer extracts text from all media and feeds it through the
    existing pipeline.
    """
    model_config = {"env_prefix": "AEGIS_MULTIMODAL_"}

    enabled: bool = Field(
        default=False,
        description="Enable multimodal image scanning (requires Pillow)",
    )
    max_image_size_mb: float = Field(
        default=20.0,
        ge=0.1,
        description="Maximum image size in megabytes before rejection",
    )
    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR text extraction from images",
    )
    steg_enabled: bool = Field(
        default=True,
        description="Enable steganographic analysis on images",
    )
    max_images_per_request: int = Field(
        default=10,
        ge=1,
        description="Maximum number of images per request",
    )

    # --- Document scanning ---
    document_scanning_enabled: bool = Field(
        default=True,
        description="Enable document scanning for base64-encoded file attachments",
    )
    document_max_size_mb: float = Field(
        default=50.0,
        ge=0.1,
        description="Maximum document size in megabytes before rejection",
    )
    document_block_macros: bool = Field(
        default=True,
        description="Block documents containing VBA macros or PDF JavaScript",
    )
    document_block_scripts: bool = Field(
        default=True,
        description="Block documents containing embedded script bodies (e.g. <script> tags)",
    )

    # --- Audio scanning ---
    audio_scanning_enabled: bool = Field(
        default=True,
        description="Enable audio scanning for voice-enabled AI security",
    )
    audio_max_size_mb: float = Field(
        default=100.0,
        ge=0.1,
        description="Maximum audio size in megabytes before rejection",
    )
    audio_max_duration_seconds: float = Field(
        default=1800.0,
        ge=1.0,
        description="Maximum audio duration in seconds (default 30 minutes)",
    )

    # --- Cross-modal correlation ---
    cross_modal_enabled: bool = Field(
        default=True,
        description="Enable cross-modal correlation engine (laundering, inconsistency, escalation, volume)",
    )

    # --- Tool use security ---
    tool_definition_scanning: bool = Field(
        default=True,
        description="Scan function/tool definitions for injection patterns",
    )
    tool_output_scanning: bool = Field(
        default=True,
        description="Scan tool output messages for injection patterns",
    )


# ---------------------------------------------------------------------------
# Global Platform Configuration
# ---------------------------------------------------------------------------

class AegisConfig(BaseSettings):
    """Root configuration aggregating all layer configs.

    ASSUMED-BREACH POSTURE: If this configuration is leaked, an attacker
    learns every threshold, every model path, and every fallback endpoint.
    In production, sensitive fields (API keys, upstream URLs) MUST be
    sourced from a secrets manager, not .env files. The config object is
    frozen after startup — runtime modification is prohibited.
    """
    model_config = {"env_prefix": "AEGIS_"}

    # --- Core platform ---
    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, description="Bind port")
    debug: bool = Field(default=False, description="Debug mode (never in production)")
    log_level: str = Field(default="INFO", description="Logging level")

    # --- Upstream model ---
    upstream_url: str = Field(
        default="https://api.openai.com",
        description="Base URL of the upstream AI model provider",
    )
    upstream_api_key: str = Field(
        default="",
        description="API key for the upstream model provider",
    )
    api_key: str = Field(
        default="",
        description="AEGIS gateway API key required from clients",
    )

    # --- Supply chain ---
    supply_chain_model_base: str = Field(
        default="/tmp/aegis-models",
        description="Allowed base directory for supply chain model verification. "
        "model_path must resolve within this directory.",
    )

    # --- Event bus ---
    event_bus_type: str = Field(
        default="memory",
        description="Event bus backend: 'memory' (dev) or 'redis' (prod)",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for event bus and rate limiting",
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql://aegis:aegis@localhost:5432/aegis",
        description="PostgreSQL connection URL for audit logs and signatures",
    )

    # --- Distillation defense ---
    distillation_defense_enabled: bool = Field(
        default=True,
        description="Enable cross-session distillation attack detection",
    )
    distillation_window_hours: float = Field(
        default=24.0,
        ge=1.0,
        description="Hours of history to retain for distillation detection",
    )
    distillation_max_history: int = Field(
        default=10000,
        ge=100,
        description="Maximum interaction records per API key",
    )
    reasoning_trace_mode: str = Field(
        default="monitor",
        description="Reasoning trace sanitizer mode: 'monitor', 'redact', or 'summarize'",
    )

    # --- Manipulation detection (MTMD) ---
    manipulation_detection_enabled: bool = Field(
        default=True,
        description="Enable multi-turn manipulation detector (MTMD)",
    )
    mtmd_window_size: int = Field(
        default=20,
        ge=5,
        description="MTMD sliding window size (turns per source)",
    )
    mtmd_max_history: int = Field(
        default=100,
        ge=10,
        description="Maximum turn records per source before LRU eviction",
    )
    mtmd_block_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MTMD manipulation_score threshold for blocking",
    )
    mtmd_alert_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="MTMD manipulation_score threshold for alerting",
    )

    # --- Source behavioral profiling ---
    profiler_enabled: bool = Field(
        default=True,
        description="Enable source behavioral profiling",
    )
    profiler_max_profiles: int = Field(
        default=10000,
        ge=100,
        description="Maximum source profiles before LRU eviction",
    )

    # --- Adaptive rate limiting ---
    adaptive_rate_limit_enabled: bool = Field(
        default=True,
        description="Enable adaptive rate limiting based on manipulation risk",
    )
    adaptive_rate_limit_hard_stop_threshold: int = Field(
        default=5,
        ge=2,
        description="Manipulation attempts before hard stop",
    )
    adaptive_rate_limit_hard_stop_duration: int = Field(
        default=300,
        ge=30,
        description="Hard stop duration in seconds",
    )
    adaptive_rate_limit_max_states: int = Field(
        default=10000,
        ge=100,
        description="Maximum adaptive rate limit states",
    )

    # --- Jailbreak taxonomy ---
    jailbreak_taxonomy_enabled: bool = Field(
        default=True,
        description="Enable jailbreak attempt taxonomy logging",
    )
    jailbreak_taxonomy_max_attempts: int = Field(
        default=50000,
        ge=1000,
        description="Maximum taxonomy log entries before FIFO eviction",
    )

    # --- Tool Proxy (TIP/TIPE) ---
    tool_proxy_enabled: bool = Field(
        default=True,
        description="Enable Tool Invocation Proxy for agent-to-tool security",
    )
    tool_proxy_default_rpm: int = Field(
        default=60,
        ge=1,
        description="Default per-tool rate limit (RPM per source)",
    )
    tool_proxy_max_param_size: int = Field(
        default=10000,
        ge=100,
        description="Maximum parameter size in bytes",
    )
    tool_proxy_max_invocation_log: int = Field(
        default=10000,
        ge=100,
        description="Maximum invocation log entries before FIFO eviction",
    )
    tool_proxy_block_internal_urls: bool = Field(
        default=True,
        description="Block URLs targeting internal/private IP ranges",
    )
    tool_proxy_require_https: bool = Field(
        default=False,
        description="Require HTTPS for all URL parameters",
    )

    # --- Tool Description Integrity Validator (TDIV) ---
    tdiv_enabled: bool = Field(
        default=True,
        description="Enable Tool Description Integrity Validator",
    )
    tdiv_max_description_length: int = Field(
        default=2000,
        ge=100,
        description="Max single field length before flagging as suspicious",
    )

    # --- Tool Response Sanitizer (TRS) ---
    trs_enabled: bool = Field(
        default=True,
        description="Enable Tool Response Sanitizer",
    )
    trs_default_max_output_size: int = Field(
        default=102400,
        ge=1024,
        description="Default max tool output size in bytes (100KB)",
    )
    trs_block_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Injection score threshold for blocking tool responses",
    )

    # --- Tool Chain Anomaly Detector (TCAD) ---
    tcad_enabled: bool = Field(
        default=True,
        description="Enable Tool Chain Anomaly Detector",
    )
    tcad_window_size: int = Field(
        default=20,
        ge=5,
        description="Per-session tool invocation sliding window size",
    )
    tcad_volume_multiplier: float = Field(
        default=3.0,
        ge=1.5,
        description="Volume anomaly threshold multiplier over baseline",
    )
    tcad_baseline_sessions: int = Field(
        default=100,
        ge=10,
        description="Sessions needed before unusual ordering detection activates",
    )

    # --- Inter-Agent Communication Monitor (IACM) ---
    iacm_enabled: bool = Field(
        default=True,
        description="Enable Inter-Agent Communication Monitor",
    )
    iacm_trust_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sender trust below this triggers elevated scrutiny",
    )
    iacm_injection_rate_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Injection rate above this flags sender as compromised",
    )

    # --- CoT Defense (Extensions 1.2-1.4) ---
    cot_defense_enabled: bool = Field(
        default=True,
        description="Enable chain-of-thought hijacking defense",
    )
    cot_coherence_window_size: int = Field(
        default=200,
        ge=50,
        description="Tokens per coherence analysis window",
    )
    cot_coherence_pivot_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Coherence score below this is a pivot",
    )
    cot_length_anomaly_multiplier: float = Field(
        default=3.0,
        ge=1.5,
        description="Baseline multiplier for length anomaly detection",
    )
    cot_length_anomaly_default_baseline: int = Field(
        default=500,
        ge=50,
        description="Default reasoning trace length baseline (tokens)",
    )
    cot_length_anomaly_max_baselines: int = Field(
        default=10000,
        ge=100,
        description="Max per-(tenant, session) baselines to track",
    )
    cot_alignment_misalign_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Alignment score below this is MISALIGNED",
    )
    cot_defense_block_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Combined CoT defense score for blocking",
    )

    # --- Model Provenance Validator (MPV) — Extension 3.1 ---
    provenance_validator_enabled: bool = Field(
        default=True,
        description="Enable Model Provenance Validator",
    )
    provenance_trusted_registries: list[str] = Field(
        default_factory=lambda: ["huggingface.co", "pytorch.org", "tensorflow.org", "onnx.ai"],
        description="Trusted model registries (hostnames)",
    )
    provenance_trusted_orgs: list[str] = Field(
        default_factory=lambda: [
            "meta-llama", "google", "microsoft", "openai",
            "mistralai", "anthropic", "ProtectAI", "sentence-transformers",
        ],
        description="Trusted model organizations",
    )
    provenance_block_untrusted: bool = Field(
        default=False,
        description="Block unverified (not just rejected) models",
    )

    # --- Skill/Plugin Auditor (SPA) — Extension 3.2 ---
    skill_auditor_enabled: bool = Field(
        default=True,
        description="Enable Skill/Plugin Auditor",
    )
    skill_auditor_block_lethal_trifecta: bool = Field(
        default=True,
        description="Block skills with file + network + execution permissions",
    )

    # --- Dependency Chain Analyzer (DCA) — Extension 3.3 ---
    dependency_analyzer_enabled: bool = Field(
        default=True,
        description="Enable Dependency Chain Analyzer",
    )

    # --- Supply Chain Validation Cache — Extension 3.4 ---
    supply_chain_cache_max_entries: int = Field(
        default=10000,
        ge=100,
        description="Max cached validation results before LRU eviction",
    )
    supply_chain_cache_default_ttl: int = Field(
        default=604800,
        ge=3600,
        description="Default cache TTL in seconds (7 days)",
    )

    # --- Re-Validation Scheduler — Extension 3.5 ---
    revalidation_enabled: bool = Field(
        default=True,
        description="Enable continuous supply chain re-validation scheduler",
    )
    revalidation_active_interval_hours: float = Field(
        default=6.0,
        ge=0.5,
        description="Re-validation interval for active components (hours)",
    )
    revalidation_inactive_interval_hours: float = Field(
        default=24.0,
        ge=1.0,
        description="Re-validation interval for inactive components (hours)",
    )

    # --- Temporal Defense (Extension 2) ---
    temporal_defense_enabled: bool = Field(
        default=True,
        description="Enable temporal threat detection infrastructure",
    )
    bbe_enabled: bool = Field(
        default=True,
        description="Enable Behavioral Baseline Engine",
    )
    bbe_warning_threshold_sigma: float = Field(
        default=2.0,
        ge=1.0,
        description="BBE warning alert threshold in standard deviations",
    )
    bbe_critical_threshold_sigma: float = Field(
        default=3.0,
        ge=1.5,
        description="BBE critical alert threshold in standard deviations",
    )
    bbe_production_transition_count: int = Field(
        default=500,
        ge=10,
        description="Real interactions before transitioning to production tier",
    )
    bbe_max_baselines: int = Field(
        default=10000,
        ge=100,
        description="Maximum per-(tenant, category) baselines before LRU eviction",
    )
    bbe_short_window: int = Field(
        default=100,
        ge=10,
        description="Short-term window size (interactions) for drift detection",
    )
    bbe_medium_window: int = Field(
        default=1000,
        ge=50,
        description="Medium-term window size (interactions) for drift detection",
    )
    synthetic_generator_enabled: bool = Field(
        default=True,
        description="Enable Synthetic Traffic Generator",
    )
    canary_injection_enabled: bool = Field(
        default=True,
        description="Enable Canary Injection System for temporal threat detection",
    )
    canary_profile: str = Field(
        default="general_enterprise",
        description="Canary query profile to load (general_enterprise or public_safety)",
    )
    canary_injections_per_hour: float = Field(
        default=6.0,
        ge=0.1,
        description="Target canary injections per hour",
    )
    canary_startup_delay_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Delay before first canary injection after startup",
    )
    canary_keyword_pass_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Keyword hit rate threshold for PASS verdict",
    )
    canary_keyword_fail_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Keyword hit rate threshold below which verdict is FAIL",
    )
    canary_semantic_pass_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Semantic similarity threshold for PASS verdict",
    )
    canary_consecutive_fail_critical: int = Field(
        default=3,
        ge=2,
        description="Consecutive failures before CRITICAL alert",
    )
    mpr_enabled: bool = Field(
        default=True,
        description="Enable Memory Provenance Registry",
    )
    mpr_max_records: int = Field(
        default=100000,
        ge=100,
        description="Maximum provenance records before FIFO eviction",
    )
    tce_enabled: bool = Field(
        default=True,
        description="Enable Temporal Correlation Engine",
    )
    tce_default_correlation_window_hours: float = Field(
        default=72.0,
        ge=1.0,
        description="Default lookback window for correlation analysis (hours)",
    )
    tce_max_candidates: int = Field(
        default=5,
        ge=1,
        description="Maximum correlation candidates returned per report",
    )
    tce_auto_correlate_on_critical: bool = Field(
        default=True,
        description="Auto-trigger correlation on critical drift/canary alerts",
    )

    # --- Layer configs ---
    barrier: BarrierConfig = Field(default_factory=BarrierConfig)
    innate: InnateConfig = Field(default_factory=InnateConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    healing: HealingConfig = Field(default_factory=HealingConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)

    # --- Performance budget ---
    total_gateway_overhead_target_ms: float = Field(
        default=150.0,
        description="Maximum acceptable total gateway overhead in ms",
    )
    throughput_target_rps: int = Field(
        default=10_000,
        description="Target requests per second per gateway node",
    )


@lru_cache(maxsize=1)
def get_config() -> AegisConfig:
    """Return the singleton AegisConfig, loaded from environment.

    Cached so all layers share the same config instance.
    """
    return AegisConfig()
