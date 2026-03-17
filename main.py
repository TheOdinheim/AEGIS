"""
AEGIS — Adaptive Enterprise Guard for Intelligent Systems

FastAPI entry point. Wires L1 through L7 into the dual-path pipeline:

    Request → L1 Barrier → L2 Innate (sync, <5ms)
                          → L3 Adaptive (async, parallel with model call)
            → L6 Policy Engine (merges both scores)
            → Forward to upstream model (with L7 circuit breaker)
            → L5 Output Validation on response
            → Return to client

For streaming (SSE), tokens are buffered in 128-token sliding windows and
validated through L5's StreamingValidator. Threat detection terminates the
stream with a safety message.

The antibody learning loop fires automatically inside L3: when adaptive
catches a threat that innate missed, the attack is embedded and stored in
L4's threat vault.

ASSUMED-BREACH POSTURE: The proxy assumes every upstream response is hostile
and every client request is adversarial. Every layer operates independently.
A compromised upstream model, spoofed client, or poisoned vault cannot
disable the cascade.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from aegis.config import AegisConfig, CircuitBreakerState, get_config
from aegis.layers.barrier import BarrierLayer, BarrierReject
from aegis.layers.innate import InnateDetectionLayer
from aegis.layers.innate.canary_verifier import inject_canary
from aegis.layers.adaptive import AdaptiveAnalysisLayer
from aegis.layers.output import OutputValidationLayer
from aegis.layers.output.streaming import StreamingInterceptor, format_safety_termination
from aegis.layers.policy import PolicyEngine
from aegis.layers.policy.opa_engine import OPAPolicyEngine
from aegis.layers.healing import HealingLayer
from aegis.layers.audit import AuditLogger, get_audit_logger, reset_audit_logger
from aegis.layers.memory.threat_vault import ThreatVault
from aegis.layers.memory.signatures import SignatureGenerator, SignatureStore
from aegis.layers.supply_chain import SupplyChainVerifier
from aegis.layers.agent_security import AgentSecurityLayer, AgentSecurityResult
from aegis.services.compliance import ComplianceEngine
from aegis.services.federated import FederatedIntelligenceManager
from aegis.services.deployment import (
    ConfigValidator, ConfigValidationResult,
    VaultBackupManager, DeepHealthMonitor,
)
from aegis.services.redis_client import init_redis, close_redis, redis_health, get_redis
from aegis.services.db import init_db, close_db, db_health, get_session_factory
from aegis.services.audit_logger import (
    PgAuditLogger, PgAuditRecord, compute_prompt_hash,
    get_pg_audit_logger, reset_pg_audit_logger,
    log_circuit_breaker_event,
)
from aegis.services.tenant_manager import TenantManager, TenantConfig
from aegis.services.threat_intel import ThreatIntelManager
from aegis.services.event_bus import (
    EventBus, InMemoryEventBus, create_event_bus,
    CHANNEL_THREAT_DETECTED, CHANNEL_CIRCUIT_BREAKER,
    CHANNEL_ANTIBODY_GENERATED, CHANNEL_POLICY_ESCALATION,
    Event,
)
from aegis.middleware.metrics import (
    ACTIVE_CONNECTIONS,
    ADAPTIVE_ANALYZER_LATENCY,
    ANTIBODY_GENERATIONS,
    BLOCKS_TOTAL,
    BUILD_INFO,
    CIRCUIT_BREAKER_STATE,
    CIRCUIT_BREAKER_TRIPS,
    EVENT_BUS_EVENTS,
    INNATE_SCANNER_LATENCY,
    LAYER_LATENCY,
    OUTPUT_CASCADE_STAGE,
    PII_REDACTIONS,
    POLICY_DECISIONS,
    QUARANTINED_SESSIONS,
    RATE_LIMIT_TRIGGERS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    TENANT_REQUESTS,
    THREAT_LEVEL,
    THREATS_DETECTED,
    TOXICITY_DETECTIONS,
    UPSTREAM_ERRORS,
    UPSTREAM_LATENCY,
    VAULT_SIZE,
    STREAM_INTERRUPTIONS,
    STREAM_WINDOWS_EVALUATED,
    STREAM_TOKENS_PROCESSED,
    MULTIMODAL_IMAGES_SCANNED,
    MULTIMODAL_OCR_TEXT_EXTRACTED,
    MULTIMODAL_IMAGE_THREATS,
    MULTIMODAL_SCAN_LATENCY,
    MULTIMODAL_DOCUMENTS_SCANNED,
    MULTIMODAL_DOCUMENT_THREATS,
    MULTIMODAL_HIDDEN_CONTENT_DETECTED,
    MULTIMODAL_AUDIO_SCANNED,
    MULTIMODAL_AUDIO_THREATS,
    DISTILLATION_SIGNALS,
    DISTILLATION_ALERTS,
    DISTILLATION_BLOCKS,
    REASONING_TRACES_DETECTED,
    GOVERNANCE_DISCLOSURES_DETECTED,
    get_metrics_text,
    track_latency,
)
from aegis.middleware.request_enrichment import (
    extract_headers,
    extract_source_ip,
    get_raw_body,
    parse_request_body,
)
from aegis.layers.multimodal import MultimodalPreprocessor
from aegis.layers.multimodal.cross_modal_engine import CrossModalCorrelationEngine
from aegis.layers.adaptive.distillation_defense import DistillationDefenseAnalyzer
from aegis.layers.adaptive.distillation_models import InteractionRecord
from aegis.layers.output.reasoning_sanitizer import ReasoningTraceSanitizer
from aegis.models.policy_decision import PolicyAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level layer singletons (initialized in lifespan)
# ---------------------------------------------------------------------------

_barrier: BarrierLayer | None = None
_innate: InnateDetectionLayer | None = None
_adaptive: AdaptiveAnalysisLayer | None = None
_output: OutputValidationLayer | None = None
_policy: PolicyEngine | OPAPolicyEngine | None = None
_healing: HealingLayer | None = None
_vault: ThreatVault | None = None
_signature_store: SignatureStore | None = None
_signature_generator: SignatureGenerator | None = None
_supply_chain: SupplyChainVerifier | None = None
_config: AegisConfig | None = None
_http_client: httpx.AsyncClient | None = None
_audit: AuditLogger | None = None
_event_bus: EventBus | None = None
_redis_client: Any | None = None
_pg_audit: PgAuditLogger | None = None
_db_engine: Any | None = None
_tenant_manager: TenantManager | None = None
_threat_intel: ThreatIntelManager | None = None
_agent_security: AgentSecurityLayer | None = None
_compliance: ComplianceEngine | None = None
_federated: FederatedIntelligenceManager | None = None
_deep_health: DeepHealthMonitor | None = None
_config_validation: ConfigValidationResult | None = None
_vault_backup: VaultBackupManager | None = None
_multimodal: MultimodalPreprocessor | None = None
_cross_modal: CrossModalCorrelationEngine | None = None
_distillation: DistillationDefenseAnalyzer | None = None
_reasoning_sanitizer: ReasoningTraceSanitizer | None = None


def _init_layers(
    config: AegisConfig | None = None,
    redis_client: Any | None = None,
    session_factory: Any | None = None,
    tenant_manager: TenantManager | None = None,
) -> None:
    """Initialize all layers. Called during app startup."""
    global _barrier, _innate, _adaptive, _output, _policy, _healing, _vault, _config, _http_client, _audit, _threat_intel
    global _signature_store, _signature_generator, _supply_chain, _agent_security, _compliance, _federated, _multimodal
    global _distillation, _reasoning_sanitizer, _cross_modal

    _config = config or get_config()
    data_dir = Path(__file__).parent / "data"

    # L1 Barrier (with optional Redis-backed rate limiter and tenant manager)
    _barrier = BarrierLayer(
        config=_config.barrier,
        valid_api_keys={_config.api_key} if _config.api_key else None,
        redis_client=redis_client,
        tenant_manager=tenant_manager,
    )

    # L4 Memory (Threat Vault) — must be created before L3
    # Try loading persisted vault first; fall back to seed data
    _vault = ThreatVault(config=_config.memory, session_factory=session_factory)
    loaded = _vault.load_from_disk()
    if loaded == 0:
        seed_file = data_dir / "seed_threats.json"
        if seed_file.exists():
            _vault.load_seed(seed_file)

    # L4 Tier 2 — Signature Database & Clonal Selection
    _signature_store = SignatureStore(config=_config.memory, session_factory=session_factory)
    _signature_generator = SignatureGenerator(
        config=_config.memory,
        signature_store=_signature_store,
    )
    # Load benign prompts for affinity testing
    benign_file = data_dir / _config.memory.benign_prompts_file.split("/")[-1]
    if benign_file.exists():
        _signature_generator.load_benign_prompts(benign_file)
    # Set positive set from vault's known attacks
    attack_texts = [
        ind.payload_summary for ind in _vault.get_indicators()
        if ind.payload_summary
    ]
    if attack_texts:
        _signature_generator.set_positive_set(attack_texts)

    # Supply Chain Verifier (Mucosal Immunity)
    cve_file = data_dir / "known_cves.json"
    _supply_chain = SupplyChainVerifier(
        cve_file=cve_file if cve_file.exists() else None,
    )

    # L2 Innate (with canary verifier)
    _innate = InnateDetectionLayer(
        config=_config.innate, data_dir=data_dir, canary_config=_config.canary,
    )

    def _on_antibody_generated(attack_text: str, indicator_id: str) -> None:
        """Callback: fire-and-forget clonal selection via asyncio task.

        Uses create_task() so signature generation doesn't block the
        request pipeline. Incrementally appends the new attack to the
        positive set instead of rebuilding from all vault contents.
        """
        if not _signature_generator or not _innate:
            return

        async def _run_clonal_selection() -> None:
            try:
                # Incremental positive set update — append new attack only
                if attack_text and attack_text not in _signature_generator._positive_set:
                    _signature_generator._positive_set.append(attack_text)
                # Generate and select signatures
                promoted = _signature_generator.generate_and_select(attack_text, indicator_id)
                # Add promoted signatures to L2 regex engine for immediate detection
                for sig in promoted:
                    ok = _innate.regex_engine.add_dynamic_pattern(
                        pattern_str=sig.pattern,
                        pattern_id=sig.signature_id,
                        description=f"Clonal selection from {indicator_id}",
                    )
                    logger.info(
                        "Clonal selection promoted %s to L2 fast path (ok=%s): %s",
                        sig.signature_id, ok, sig.pattern[:80],
                    )
                logger.info(
                    "Clonal selection complete for %s: %d promoted, L2 now has %d patterns",
                    indicator_id, len(promoted), _innate.regex_engine.pattern_count,
                )
            except Exception as e:
                logger.error("Async clonal selection failed: %s", e)

        try:
            asyncio.create_task(_run_clonal_selection())
        except RuntimeError:
            # No event loop — fall back to synchronous execution
            if attack_text and attack_text not in _signature_generator._positive_set:
                _signature_generator._positive_set.append(attack_text)
            promoted = _signature_generator.generate_and_select(attack_text, indicator_id)
            for sig in promoted:
                _innate.regex_engine.add_dynamic_pattern(
                    pattern_str=sig.pattern,
                    pattern_id=sig.signature_id,
                    description=f"Clonal selection from {indicator_id}",
                )

    # L3 Adaptive (with learning loop wired to vault and clonal selection)
    # skip_model_load=True skips DeBERTa/sentence-transformers download.
    # Set AEGIS_SKIP_MODEL_LOAD=true to skip (e.g., tests, CI).
    # Default: load models (False) for production and demo use.
    _skip_ml = os.environ.get("AEGIS_SKIP_MODEL_LOAD", "").lower() in ("true", "1", "yes")
    _adaptive = AdaptiveAnalysisLayer(
        config=_config.adaptive,
        threat_vault=_vault,
        skip_model_load=_skip_ml,
        on_antibody=_on_antibody_generated,
    )

    # Threat Intel Manager (STIX/TAXII ingestion, L4 Tier 3)
    embed_fn = None
    if hasattr(_adaptive, '_semantic') and hasattr(_adaptive._semantic, 'embed'):
        embed_fn = _adaptive._semantic.embed
    _threat_intel = ThreatIntelManager(
        threat_vault=_vault,
        embed_fn=embed_fn,
        session_factory=session_factory,
    )

    # L5 Output Validation
    _output = OutputValidationLayer(config=_config.output)

    # L6 Policy Engine (with optional tenant manager for per-tenant thresholds)
    _python_policy = PolicyEngine(config=_config.policy, tenant_manager=tenant_manager)
    if _config.policy.backend == "opa":
        _policy = OPAPolicyEngine(
            opa_url=_config.policy.opa_url,
            fallback=_python_policy,
            config=_config.policy,
            tenant_manager=tenant_manager,
        )
        logger.info("Policy backend: OPA at %s (Python fallback active)", _config.policy.opa_url)
    else:
        _policy = _python_policy
        logger.info("Policy backend: Python")

    # L7 Self-Healing
    _healing = HealingLayer(config=_config.healing)

    # Multi-Agent Security (MHC identity verification)
    _agent_security = AgentSecurityLayer(
        innate_layer=_innate,
        event_bus=_event_bus,
    )

    # Compliance Engine (cross-framework regulatory reporting)
    _compliance = ComplianceEngine(
        audit_logger=get_audit_logger(),
        config=_config,
        vault=_vault,
        policy=_policy,
        healing=_healing,
        supply_chain=_supply_chain,
    )

    # L8 Federated Threat Intelligence (Herd Immunity)
    _federated = FederatedIntelligenceManager(
        instance_id=os.environ.get("AEGIS_INSTANCE_ID", "local"),
        epsilon=float(os.environ.get("AEGIS_DP_EPSILON", "3.0")),
        delta=float(os.environ.get("AEGIS_DP_DELTA", "1e-5")),
        privacy_budget=float(os.environ.get("AEGIS_DP_BUDGET", "100.0")),
    )

    # Multimodal preprocessor (config-gated, zero overhead when disabled)
    _multimodal = MultimodalPreprocessor(
        enabled=_config.multimodal.enabled,
        max_image_size_mb=_config.multimodal.max_image_size_mb,
        ocr_enabled=_config.multimodal.ocr_enabled,
        steg_enabled=_config.multimodal.steg_enabled,
        max_images_per_request=_config.multimodal.max_images_per_request,
        document_scanning_enabled=_config.multimodal.document_scanning_enabled,
        document_max_size_mb=_config.multimodal.document_max_size_mb,
        document_block_macros=_config.multimodal.document_block_macros,
        document_block_scripts=_config.multimodal.document_block_scripts,
        audio_scanning_enabled=_config.multimodal.audio_scanning_enabled,
        audio_max_size_mb=_config.multimodal.audio_max_size_mb,
        audio_max_duration_seconds=_config.multimodal.audio_max_duration_seconds,
        regex_engine=_innate.regex_engine if _innate else None,
    )

    # Cross-modal correlation engine
    if _config.multimodal.cross_modal_enabled:
        _cross_modal = CrossModalCorrelationEngine(
            regex_engine=_innate.regex_engine if _innate else None,
        )
    else:
        _cross_modal = None

    # Distillation defense (config-gated)
    if _config.distillation_defense_enabled:
        _distillation = DistillationDefenseAnalyzer(
            window_hours=_config.distillation_window_hours,
            max_history=_config.distillation_max_history,
        )
        _reasoning_sanitizer = ReasoningTraceSanitizer(
            mode=_config.reasoning_trace_mode,
        )
    else:
        _distillation = None
        _reasoning_sanitizer = None

    # Audit logger
    _audit = get_audit_logger()

    # HTTP client (if not already created by lifespan)
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifespan: initialize layers, Redis, PostgreSQL, tenant manager, event bus, HTTP client."""
    global _http_client, _event_bus, _redis_client, _pg_audit, _db_engine, _tenant_manager
    global _deep_health, _config_validation, _vault_backup

    # --- Configuration validation (before anything else) ---
    validator = ConfigValidator()
    _config_validation = validator.validate(get_config())
    if not _config_validation.valid:
        for err in _config_validation.errors:
            logger.error("FATAL: %s", err)
        raise SystemExit(1)

    # --- Redis (optional — graceful degradation if unavailable) ---
    redis_url = os.environ.get("REDIS_URL", "")
    _redis_client = await init_redis(redis_url) if redis_url else None

    # --- PostgreSQL (optional — graceful degradation if unavailable) ---
    db_url = os.environ.get("DATABASE_URL", "")
    _db_engine = await init_db(db_url) if db_url else None
    session_factory = get_session_factory()

    # --- PostgreSQL audit logger ---
    _pg_audit = get_pg_audit_logger()
    _pg_audit.session_factory = session_factory
    _pg_audit.start_flush_task()

    # --- Tenant Manager (loads from PostgreSQL, falls back to env-var defaults) ---
    default_tenant = TenantConfig(
        tenant_id="default",
        name="default",
        rate_limit_rpm=get_config().barrier.rate_limit_rpm,
        rate_limit_burst=get_config().barrier.rate_limit_burst,
        max_tokens_per_request=get_config().barrier.max_tokens_per_request,
        block_threshold=get_config().innate.block_threshold,
        alert_threshold=get_config().innate.alert_threshold,
    )
    _tenant_manager = TenantManager(
        session_factory=session_factory,
        default_config=default_tenant,
    )
    if session_factory:
        await _tenant_manager.load_all()
        _tenant_manager.start_refresh()

    # --- Initialize all security layers ---
    _init_layers(
        redis_client=_redis_client,
        session_factory=session_factory,
        tenant_manager=_tenant_manager if session_factory else None,
    )

    # --- Sync vault indicators from PostgreSQL (if available) ---
    if _vault and session_factory:
        await _vault.load_from_postgres()

    # --- Event bus (Redis Streams or in-memory fallback) ---
    _event_bus = await create_event_bus(_redis_client)
    await _wire_event_bus_subscriptions()
    await _event_bus.start()

    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
    )

    # Auto-ingest STIX threat feeds (vaccine loading)
    if _threat_intel:
        stix_feeds_dir = Path(__file__).parent / "data" / "stix_feeds"
        if stix_feeds_dir.is_dir():
            try:
                feed_count = await _threat_intel.ingest_directory(stix_feeds_dir)
                logger.info("Auto-ingested %d indicators from STIX feeds", feed_count)
            except Exception as e:
                logger.error("STIX feed auto-ingestion failed: %s", e)

    # Start vault maintenance background task (lifecycle transitions every 6h)
    if _vault:
        _vault.start_maintenance()

    # Start federated intelligence scheduler (rounds every 6h)
    if _federated:
        _federated.start_scheduler()

    # Deep health monitor
    _deep_health = DeepHealthMonitor()
    _deep_health.set_components(
        vault=_vault,
        healing=_healing,
        adaptive=_adaptive,
        event_bus=_event_bus,
    )
    if os.environ.get("AEGIS_DEEP_HEALTH_ENABLED", "").lower() in ("true", "1", "yes"):
        _deep_health.start_monitor()

    # Vault backup manager
    _vault_backup = VaultBackupManager()

    # Create backup directory
    backup_dir = Path(os.environ.get("AEGIS_BACKUP_DIR", "/tmp/aegis-backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize metrics ---
    BUILD_INFO.info({"version": "0.1.0", "component": "aegis"})
    if _policy:
        THREAT_LEVEL.set(_policy.threat_level.value)
    if _vault:
        stats = _vault.get_stats()
        for phase in ("acute", "persistent", "dormant"):
            VAULT_SIZE.labels(phase=phase).set(
                stats.get("phase_distribution", {}).get(phase, 0)
            )

    logger.info("AEGIS proxy initialized — all layers active (event bus: %s)", _event_bus.backend)
    yield

    # Save vault state on shutdown
    if _vault:
        _vault.stop_maintenance()
        try:
            _vault.save_to_disk()
        except Exception as e:
            logger.error("Vault save on shutdown failed: %s", e)

    # Stop deep health monitor
    if _deep_health:
        _deep_health.stop_monitor()

    # Stop federated scheduler
    if _federated:
        _federated.stop_scheduler()

    # Shut down event bus, tenant manager, Redis, and PostgreSQL
    if _event_bus:
        await _event_bus.stop()
    if _tenant_manager:
        _tenant_manager.stop_refresh()
    if _pg_audit:
        _pg_audit.stop_flush_task()
        # Final buffer flush attempt
        await _pg_audit.flush_buffer()
    if isinstance(_policy, OPAPolicyEngine):
        await _policy.close()
    await _http_client.aclose()
    await close_redis()
    await close_db()
    logger.info("AEGIS proxy shutdown")


async def _wire_event_bus_subscriptions() -> None:
    """Wire event bus subscriptions for inter-layer signaling.

    This is the cytokine cascade — events flow between layers:
      - L3 → "threat_detected" → L6 (policy escalation)
      - L3 → "antibody_generated" → (informational)
      - L7 → "circuit_breaker" → L6 (policy escalation)
      - L6 → "policy_escalation" → (informational, audit)
    """
    if not _event_bus:
        return

    # L6 subscribes to threat_detected: escalate TLI on sustained threats
    async def _on_threat_for_policy(event: Event) -> None:
        if _policy and event.payload.get("should_block"):
            _policy.escalate_threat_level()
            logger.info("Policy escalated TLI to %s via event bus", _policy.threat_level.name)

    # L6 subscribes to circuit_breaker: escalate on trips
    async def _on_circuit_breaker_for_policy(event: Event) -> None:
        if _policy and event.payload.get("new_state") == "open":
            _policy.escalate_threat_level()
            logger.info("Policy escalated TLI to %s on circuit breaker trip", _policy.threat_level.name)

    await _event_bus.subscribe(CHANNEL_THREAT_DETECTED, _on_threat_for_policy)
    await _event_bus.subscribe(CHANNEL_CIRCUIT_BREAKER, _on_circuit_breaker_for_policy)


app = FastAPI(
    title="AEGIS",
    description="Adaptive Enterprise Guard for Intelligent Systems",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health / Metrics / Models endpoints
# ---------------------------------------------------------------------------


def _is_authenticated(request: Request) -> bool:
    """Check if the request has a valid API key."""
    if not _config or not _config.api_key:
        return False
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return hmac.compare_digest(auth[7:], _config.api_key)
    return False


@app.get("/health")
async def health(request: Request) -> Response:
    """Health check endpoint.

    Unauthenticated: returns only {"status": "ok"} (200) or
    {"status": "degraded"} (503) — no internal details exposed.
    Authenticated (valid API key): returns structured JSON with per-component
    health status (healthy/degraded/unhealthy), overall status derived from
    worst component, version, uptime, and component details.
    """
    all_layers_up = all([_barrier, _innate, _adaptive, _output, _policy, _healing])

    if not _is_authenticated(request):
        status_code = 200 if all_layers_up else 503
        status_text = "ok" if all_layers_up else "degraded"
        return JSONResponse(
            status_code=status_code,
            content={"status": status_text},
        )

    # --- Build per-component health ---
    components: dict[str, dict] = {}

    # Security layers
    layer_names = {
        "barrier": _barrier, "innate": _innate, "adaptive": _adaptive,
        "output": _output, "policy": _policy, "healing": _healing,
    }
    for name, layer in layer_names.items():
        components[name] = {
            "status": "healthy" if layer is not None else "unhealthy",
            "type": "security_layer",
        }

    # Circuit breaker
    breaker_state = "unknown"
    if _healing:
        breaker = _healing.get_breaker("primary")
        breaker_state = breaker.state.value
        cb_status = "healthy" if breaker_state == "closed" else (
            "degraded" if breaker_state == "half_open" else "unhealthy"
        )
        components["circuit_breaker"] = {
            "status": cb_status,
            "type": "resilience",
            "state": breaker_state,
            "consecutive_trips": breaker.consecutive_trips,
        }

    # Redis
    redis_status = await redis_health()
    redis_up = redis_status.get("status") == "connected" if isinstance(redis_status, dict) else False
    components["redis"] = {
        "status": "healthy" if redis_up else "degraded",
        "type": "backing_service",
        "details": redis_status,
    }

    # PostgreSQL
    pg_status = await db_health()
    pg_up = pg_status.get("status") == "connected" if isinstance(pg_status, dict) else False
    components["postgres"] = {
        "status": "healthy" if pg_up else "degraded",
        "type": "backing_service",
        "details": pg_status,
    }

    # Tenant manager
    tenant_status = _tenant_manager.stats if _tenant_manager else {"status": "not_initialized"}
    components["tenant_manager"] = {
        "status": "healthy" if _tenant_manager else "degraded",
        "type": "service",
        "details": tenant_status,
    }

    # Event bus
    components["event_bus"] = {
        "status": "healthy" if _event_bus else "unhealthy",
        "type": "messaging",
        "backend": _event_bus.backend if _event_bus else "none",
    }

    # Threat vault
    if _vault:
        vault_stats = _vault.get_stats()
        components["threat_vault"] = {
            "status": "healthy",
            "type": "memory",
            "total_indicators": vault_stats.get("total_indicators", 0),
        }

    # Compliance engine
    if _compliance:
        components["compliance"] = {
            "status": "healthy",
            "type": "service",
            "details": _compliance.stats,
        }

    # Deep health monitor
    if _deep_health:
        components["deep_health_monitor"] = {
            "status": "healthy" if _deep_health._running or True else "degraded",
            "type": "service",
            "details": _deep_health.stats,
        }

    # Federated intelligence (L8)
    if _federated:
        fed_stats = _federated.stats
        fed_status = "degraded" if fed_stats.get("privacy_budget_exhausted") else "healthy"
        components["federated"] = {
            "status": fed_status,
            "type": "service",
            "details": fed_stats,
        }

    # --- Derive overall status from worst component ---
    statuses = [c["status"] for c in components.values()]
    if "unhealthy" in statuses and not all_layers_up:
        overall = "unhealthy"
        status_code = 503
    elif "degraded" in statuses or "unhealthy" in statuses:
        overall = "degraded"
        status_code = 200
    else:
        overall = "healthy"
        status_code = 200

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "version": app.version,
            "threat_level": _policy.threat_level.name if _policy else "unknown",
            "components": components,
        },
    )


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus metrics endpoint. Requires authentication to prevent
    information leakage (detection rates, latencies, block counts)."""
    if not _is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"},
        )
    return Response(
        content=get_metrics_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/v1/audit/recent")
async def audit_recent(request: Request, limit: int = 20) -> Response:
    """Return recent audit records. Requires authentication to prevent
    information leakage (block reasons, threat categories, confidence scores)."""
    if not _is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"},
        )
    if not _audit:
        return JSONResponse(content={"records": [], "count": 0})
    limit = min(limit, 100)  # Cap at 100
    records = _audit.get_recent(limit)
    return JSONResponse(content={
        "records": records,
        "count": len(records),
        "total": _audit.record_count,
        "compliance": {
            "nist_ai_rmf": "GOVERN 1.4",
            "iso_42001": "Clause 9",
            "eu_ai_act": "Articles 12, 19",
            "soc2": "CC7.2",
        },
    })


@app.get("/v1/vault/stats")
async def vault_stats(request: Request) -> Response:
    """Immune system health dashboard — Threat Vault statistics.

    Requires authentication to prevent information leakage (vault size,
    phase distribution, category breakdown, top matched indicators).
    """
    if not _is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"},
        )
    if not _vault:
        return JSONResponse(content={"error": "Vault not initialized", "total_indicators": 0})
    return JSONResponse(content=_vault.get_stats())


# ---------------------------------------------------------------------------
# Threat Intelligence Endpoints (L4 Tier 3 — STIX/TAXII)
# ---------------------------------------------------------------------------


@app.post("/v1/threat-intel/ingest")
async def threat_intel_ingest(request: Request) -> Response:
    """Ingest a STIX 2.1 threat intelligence bundle.

    Accepts a JSON body containing a valid STIX 2.1 bundle with indicator
    objects. Each indicator is embedded and stored in the Threat Vault.
    Deduplication by payload_hash prevents duplicates.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"},
        )
    if not _threat_intel:
        return JSONResponse(
            status_code=503,
            content={"error": "Threat intel manager not initialized"},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body"},
        )

    try:
        count = await _threat_intel.ingest_stix_bundle(body)
        return JSONResponse(content={
            "ingested": count,
            "status": "ok",
        })
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )
    except Exception as e:
        logger.error("Threat intel ingestion failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Ingestion failed"},
        )


@app.get("/v1/threat-intel/export")
async def threat_intel_export(request: Request) -> Response:
    """Export threat indicators as a STIX 2.1 bundle.

    Optional query parameter ?since=<ISO datetime> to export only
    indicators created after a given timestamp. Embeddings are
    anonymized with differential privacy noise (epsilon=3.0).

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"},
        )
    if not _threat_intel:
        return JSONResponse(
            status_code=503,
            content={"error": "Threat intel manager not initialized"},
        )

    since = None
    since_str = request.query_params.get("since")
    if since_str:
        try:
            since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid 'since' parameter — use ISO 8601 format"},
            )

    bundle = await _threat_intel.export_stix_bundle(since=since)
    return JSONResponse(content=bundle)


@app.get("/v1/threat-intel/stats")
async def threat_intel_stats(request: Request) -> Response:
    """Return threat intelligence feed statistics.

    Includes total ingested, deduplicated, exported counts, last ingestion
    time, source breakdown, and vault size.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"},
        )
    if not _threat_intel:
        return JSONResponse(
            status_code=503,
            content={"error": "Threat intel manager not initialized"},
        )

    return JSONResponse(content=_threat_intel.stats)


@app.post("/v1/supply-chain/verify")
async def supply_chain_verify(request: Request) -> dict[str, Any]:
    """Run the four-stage model supply chain verification pipeline.

    Accepts a JSON body with:
        - model_id (required): Unique identifier for the model
        - model_path (optional): Path to model directory on disk
        - manifest (optional): {filename: sha256_hash} for integrity check
        - dependencies (optional): {package: version} for CVE audit
        - source (optional): Declared model source/publisher
        - behavioral_responses (optional): {probe_id: response} for offline probing
    """
    if not _supply_chain:
        return {"error": "Supply chain verifier not initialized"}

    body = await request.json()
    model_id = body.get("model_id")
    if not model_id:
        return JSONResponse(
            status_code=400,
            content={"error": "model_id is required"},
        )

    model_path = None
    if body.get("model_path"):
        model_path = Path(body["model_path"])
        # Path traversal protection: resolve and validate against allowed base
        path_error = _validate_model_path(model_path)
        if path_error:
            return JSONResponse(
                status_code=403,
                content={"error": path_error},
            )

    report = _supply_chain.verify(
        model_id=model_id,
        model_path=model_path,
        manifest=body.get("manifest"),
        dependencies=body.get("dependencies"),
        source=body.get("source"),
        behavioral_responses=body.get("behavioral_responses"),
    )

    return report.model_dump(mode="json")


@app.get("/v1/supply-chain/report/{model_id}")
async def supply_chain_report(model_id: str) -> dict[str, Any]:
    """Retrieve a previously generated supply chain verification report."""
    if not _supply_chain:
        return {"error": "Supply chain verifier not initialized"}

    report = _supply_chain.get_report(model_id)
    if not report:
        return JSONResponse(
            status_code=404,
            content={"error": f"No report found for model_id: {model_id}"},
        )
    return report.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Multi-Agent Security Endpoints (MHC Identity Verification)
# ---------------------------------------------------------------------------


@app.post("/v1/agents/register")
async def agent_register(request: Request) -> Response:
    """Register a new agent and return its signed identity with JWT token.

    Requires authentication. Body: {parent_agent_id, capabilities, scope, ttl_hours}.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _agent_security:
        return JSONResponse(status_code=503, content={"error": "Agent security not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    agent = await _agent_security.identity_manager.register_agent(
        parent_id=body.get("parent_agent_id"),
        capabilities=body.get("capabilities"),
        scope=body.get("scope"),
        ttl_hours=body.get("ttl_hours", 24),
    )

    result = agent.to_dict()
    result["token"] = agent.token
    return JSONResponse(content=result)


@app.get("/v1/agents/stats")
async def agent_stats(request: Request) -> Response:
    """Return multi-agent security statistics.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _agent_security:
        return JSONResponse(status_code=503, content={"error": "Agent security not initialized"})

    return JSONResponse(content=_agent_security.stats())


@app.post("/v1/agents/message/validate")
async def agent_message_validate(request: Request) -> Response:
    """Validate an inter-agent message for injection and authorization.

    Requires authentication. Body: {sender_id, receiver_id, message}.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _agent_security:
        return JSONResponse(status_code=503, content={"error": "Agent security not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    sender_id = body.get("sender_id", "")
    receiver_id = body.get("receiver_id", "")
    message = body.get("message", {})

    if not sender_id or not receiver_id:
        return JSONResponse(
            status_code=400,
            content={"error": "sender_id and receiver_id are required"},
        )

    result = await _agent_security.validate_agent_message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message=message,
    )

    return JSONResponse(content={
        "valid": result.valid,
        "blocked_reason": result.blocked_reason,
        "injection_detected": result.injection_detected,
        "sender_trust": round(result.sender_trust, 4),
    })


@app.get("/v1/agents/{agent_id}")
async def agent_status(request: Request, agent_id: str) -> Response:
    """Get agent status, trust level, capabilities, and lineage.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _agent_security:
        return JSONResponse(status_code=503, content={"error": "Agent security not initialized"})

    agent = await _agent_security.identity_manager.verify_agent(agent_id)
    if agent is None:
        return JSONResponse(status_code=404, content={"error": f"Agent not found or inactive: {agent_id}"})

    return JSONResponse(content=agent.to_dict())


@app.post("/v1/agents/{agent_id}/authorize")
async def agent_authorize(request: Request, agent_id: str) -> Response:
    """Check if an agent is authorized for a specific tool/action.

    Requires authentication. Body: {tool_name, tool_params}.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _agent_security:
        return JSONResponse(status_code=503, content={"error": "Agent security not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    tool_name = body.get("tool_name", "")
    tool_params = body.get("tool_params", {})

    result = await _agent_security.process_agent_request(
        agent_id=agent_id,
        action=tool_name,
        params=tool_params,
    )

    return JSONResponse(content={
        "allowed": result.allowed,
        "reason": result.reason,
        "violations": result.violations,
        "agent_id": agent_id,
    })


# ---------------------------------------------------------------------------
# Compliance Dashboard Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/compliance/frameworks")
async def compliance_frameworks(request: Request) -> Response:
    """List available compliance frameworks with control counts.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _compliance:
        return JSONResponse(status_code=503, content={"error": "Compliance engine not initialized"})

    return JSONResponse(content={
        "frameworks": _compliance.get_frameworks(),
    })


@app.get("/v1/compliance/matrix")
async def compliance_matrix(request: Request) -> Response:
    """Return the full cross-framework coverage matrix.

    Shows how each AEGIS capability maps to controls across all frameworks.
    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _compliance:
        return JSONResponse(status_code=503, content={"error": "Compliance engine not initialized"})

    return JSONResponse(content=_compliance.get_coverage_matrix())


@app.post("/v1/compliance/report")
async def compliance_report(request: Request) -> Response:
    """Generate a cross-framework compliance report.

    Body: {frameworks: ["NIST_AI_RMF", "SOC_2"], time_range_hours: 24}
    If frameworks is omitted, generates for all frameworks.
    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _compliance:
        return JSONResponse(status_code=503, content={"error": "Compliance engine not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    frameworks = body.get("frameworks")
    time_range_hours = body.get("time_range_hours", 24)

    report = await _compliance.generate_report(
        frameworks=frameworks,
        time_range_hours=time_range_hours,
    )
    return JSONResponse(content=report.to_dict())


@app.get("/v1/compliance/report/{report_id}")
async def compliance_report_by_id(request: Request, report_id: str) -> Response:
    """Retrieve a previously generated compliance report.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _compliance:
        return JSONResponse(status_code=503, content={"error": "Compliance engine not initialized"})

    report = _compliance.get_report(report_id)
    if not report:
        return JSONResponse(
            status_code=404,
            content={"error": f"Report not found: {report_id}"},
        )
    return JSONResponse(content=report.to_dict())


@app.get("/v1/compliance/evidence/{capability}")
async def compliance_evidence(request: Request, capability: str) -> Response:
    """Collect and return evidence for a specific AEGIS capability.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _compliance:
        return JSONResponse(status_code=503, content={"error": "Compliance engine not initialized"})

    from aegis.services.compliance.framework_mappings import AEGIS_CAPABILITIES
    if capability not in AEGIS_CAPABILITIES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown capability: {capability}",
                "valid_capabilities": AEGIS_CAPABILITIES,
            },
        )

    time_range_hours = int(request.query_params.get("time_range_hours", "24"))
    evidence = await _compliance.collect_evidence(capability, time_range_hours)
    return JSONResponse(content=evidence.to_dict())


# ---------------------------------------------------------------------------
# Federated Threat Intelligence Endpoints (L8 — Herd Immunity)
# ---------------------------------------------------------------------------


@app.post("/v1/federated/round")
async def federated_round(request: Request) -> Response:
    """Trigger a federated learning round.

    Optionally accepts external model updates in the request body.
    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _federated:
        return JSONResponse(status_code=503, content={"error": "Federated intelligence not initialized"})

    # Parse optional external updates from body
    external_updates = None
    try:
        body = await request.json()
        if body and isinstance(body, dict) and "updates" in body:
            from aegis.services.federated.local_trainer import LocalModelUpdate
            external_updates = []
            for u in body["updates"]:
                weights = {k: np.array(v) for k, v in u.get("weights", {}).items()}
                external_updates.append(LocalModelUpdate(
                    instance_id=u.get("instance_id", "external"),
                    round_number=u.get("round_number", 0),
                    weights=weights,
                    num_samples=u.get("num_samples", 0),
                    metrics=u.get("metrics", {}),
                ))
    except Exception:
        pass  # No body or invalid — proceed without external updates

    result = await _federated.run_federated_round(external_updates)
    return JSONResponse(content=result.to_dict())


@app.get("/v1/federated/status")
async def federated_status(request: Request) -> Response:
    """Get federated intelligence status.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _federated:
        return JSONResponse(status_code=503, content={"error": "Federated intelligence not initialized"})

    return JSONResponse(content=_federated.get_status())


@app.post("/v1/federated/indicators/share")
async def federated_indicators_share(request: Request) -> Response:
    """Share a threat indicator with the federated network.

    Body: {"text": str, "embedding": list[float], "mitre_tactic": str,
           "affected_models": list[str], "confidence": float, "detection_source": str}

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _federated:
        return JSONResponse(status_code=503, content={"error": "Federated intelligence not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    text = body.get("text", "")
    embedding_list = body.get("embedding", [])
    if not text or not embedding_list:
        return JSONResponse(
            status_code=400,
            content={"error": "Both 'text' and 'embedding' are required"},
        )

    embedding = np.array(embedding_list, dtype=np.float64)
    indicator = _federated.share_indicator(
        text=text,
        embedding=embedding,
        mitre_tactic=body.get("mitre_tactic", ""),
        affected_models=body.get("affected_models", []),
        confidence=body.get("confidence", 0.0),
        detection_source=body.get("detection_source", ""),
    )

    if indicator is None:
        return JSONResponse(
            status_code=429,
            content={"error": "Privacy budget exhausted — cannot share indicator"},
        )

    return JSONResponse(content=indicator.to_dict())


@app.post("/v1/federated/indicators/receive")
async def federated_indicators_receive(request: Request) -> Response:
    """Receive threat indicators from other instances.

    Body: {"indicators": [{"indicator_id": str, "content_hash": str,
           "embedding": list[float], "mitre_tactic": str, ...}]}

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _federated:
        return JSONResponse(status_code=503, content={"error": "Federated intelligence not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    raw_indicators = body.get("indicators", [])
    if not raw_indicators:
        return JSONResponse(status_code=400, content={"error": "No indicators provided"})

    from aegis.services.federated.indicator_sharing import SharedIndicator
    indicators = []
    for raw in raw_indicators:
        embedding = np.array(raw.get("embedding", []), dtype=np.float64)
        indicators.append(SharedIndicator(
            indicator_id=raw.get("indicator_id", ""),
            embedding=embedding,
            content_hash=raw.get("content_hash", ""),
            mitre_tactic=raw.get("mitre_tactic", ""),
            affected_models=raw.get("affected_models", []),
            confidence=raw.get("confidence", 0.0),
            detection_source=raw.get("detection_source", ""),
        ))

    accepted = _federated.receive_indicators(indicators)
    return JSONResponse(content={
        "received": len(indicators),
        "accepted": accepted,
        "duplicates": len(indicators) - accepted,
    })


@app.get("/v1/federated/privacy-budget")
async def federated_privacy_budget(request: Request) -> Response:
    """Get current privacy budget status.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _federated:
        return JSONResponse(status_code=503, content={"error": "Federated intelligence not initialized"})

    budget = _federated.dp_engine.get_budget_status()
    return JSONResponse(content=budget.to_dict())


# ---------------------------------------------------------------------------
# Admin Endpoints (Enterprise Hardening)
# ---------------------------------------------------------------------------


@app.post("/v1/admin/backup")
async def admin_backup(request: Request) -> Response:
    """Trigger vault backup.

    Optional body: {"output_path": "/path/to/backup.json"}
    Default: /tmp/aegis-backups/vault_backup_{timestamp}.json

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _vault_backup or not _vault:
        return JSONResponse(status_code=503, content={"error": "Backup manager not initialized"})

    # Parse optional output path
    output_path = None
    try:
        body = await request.json()
        if body and isinstance(body, dict):
            output_path = body.get("output_path")
    except Exception:
        pass

    if not output_path:
        backup_dir = os.environ.get("AEGIS_BACKUP_DIR", "/tmp/aegis-backups")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"{backup_dir}/vault_backup_{ts}.json"

    result = await _vault_backup.backup_vault(
        _vault, Path(output_path), _signature_store,
    )
    status_code = 200 if result.success else 500
    return JSONResponse(status_code=status_code, content=result.to_dict())


@app.post("/v1/admin/restore")
async def admin_restore(request: Request) -> Response:
    """Restore vault from backup file.

    Body: {"backup_path": "/path/to/backup.json"}

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _vault_backup or not _vault:
        return JSONResponse(status_code=503, content={"error": "Backup manager not initialized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    backup_path = body.get("backup_path", "") if isinstance(body, dict) else ""
    if not backup_path:
        return JSONResponse(status_code=400, content={"error": "backup_path is required"})

    result = await _vault_backup.restore_vault(
        _vault, Path(backup_path), _signature_store,
    )
    status_code = 200 if result.success else 400
    return JSONResponse(status_code=status_code, content=result.to_dict())


@app.get("/v1/admin/deep-health")
async def admin_deep_health(request: Request) -> Response:
    """Run deep health check.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _deep_health:
        return JSONResponse(status_code=503, content={"error": "Deep health monitor not initialized"})

    result = await _deep_health.run_deep_health_check()
    return JSONResponse(content=result.to_dict())


@app.get("/v1/admin/config-validation")
async def admin_config_validation(request: Request) -> Response:
    """Return startup configuration validation result.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})
    if not _config_validation:
        return JSONResponse(status_code=503, content={"error": "Config validation not available"})

    return JSONResponse(content=_config_validation.to_dict())


@app.post("/v1/admin/config-reload")
async def admin_config_reload(request: Request) -> Response:
    """Reload dynamic configuration.

    Refreshes tenant cache, reloads STIX feeds, updates threat level.
    Does NOT restart the process or reinitialize models.

    Requires authentication.
    """
    if not _is_authenticated(request):
        return JSONResponse(status_code=401, content={"error": "Authentication required"})

    reloaded: dict[str, Any] = {}

    # Refresh tenant cache
    if _tenant_manager:
        try:
            await _tenant_manager.load_all()
            reloaded["tenant_cache"] = "refreshed"
        except Exception as e:
            reloaded["tenant_cache"] = f"failed: {e}"
    else:
        reloaded["tenant_cache"] = "not_available"

    # Refresh threat level
    if _policy:
        reloaded["threat_level"] = _policy.threat_level.name
    else:
        reloaded["threat_level"] = "not_available"

    # Reload STIX feeds
    if _threat_intel:
        stix_feeds_dir = Path(__file__).parent / "data" / "stix_feeds"
        if stix_feeds_dir.is_dir():
            try:
                count = await _threat_intel.ingest_directory(stix_feeds_dir)
                reloaded["stix_feeds"] = f"reloaded ({count} indicators)"
            except Exception as e:
                reloaded["stix_feeds"] = f"failed: {e}"
        else:
            reloaded["stix_feeds"] = "no_feeds_directory"
    else:
        reloaded["stix_feeds"] = "not_available"

    return JSONResponse(content={"status": "ok", "reloaded": reloaded})


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": "aegis-proxy",
                "object": "model",
                "created": 0,
                "owned_by": "aegis",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Core proxy endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """OpenAI-compatible chat completions endpoint.

    Implements the dual-path pipeline:
    1. L1 Barrier (auth, schema, rate limit, token count)
    2. L2 Innate (sync, <5ms) — block if threshold exceeded
    3. L3 Adaptive (async, parallel with model call)
    4. L6 Policy Engine (merge scores, render decision)
    5. Forward to upstream model (with L7 circuit breaker)
    6. L5 Output Validation on response
    7. Return to client
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    ACTIVE_CONNECTIONS.inc()

    # Fail-closed: if any layer is uninitialized, block immediately (T23)
    if not _barrier or not _innate or not _adaptive or not _output or not _policy or not _healing:
        logger.error("Security pipeline not initialized — blocking request %s", request_id)
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="503").inc()
        return _error_response(
            503,
            "Security pipeline failure — request blocked",
            request_id,
        )

    headers = extract_headers(request)
    try:
        response = await _process_request(request, request_id)
        # Agent trust reinforcement: +0.02 on allow, -0.1 on block
        if hasattr(response, "status_code"):
            if response.status_code == 200:
                await _update_agent_trust(headers, 0.02, "clean_interaction")
            elif response.status_code == 403:
                await _update_agent_trust(headers, -0.1, "request_blocked")
        return response
    except BarrierReject as e:
        BLOCKS_TOTAL.labels(layer="barrier", reason=e.reason).inc()
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status=str(e.status_code)).inc()
        if e.status_code == 429:
            RATE_LIMIT_TRIGGERS.labels(tenant_id="unknown").inc()
        if _audit:
            _audit.log(request_id, "barrier", "block", decision_context={"reason": e.reason})
        await _update_agent_trust(headers, -0.1, f"barrier_reject:{e.reason}")
        return _error_response(e.status_code, e.reason, request_id)
    except Exception as e:
        # Fail-closed: any unhandled exception blocks the request (T23)
        logger.exception("Security pipeline failure in request %s: %s", request_id, e)
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="503").inc()
        if _audit:
            _audit.log(
                request_id, "pipeline", "block",
                decision_context={"reason": "unhandled_exception", "error": str(e)},
            )
        return _error_response(
            503,
            "Security pipeline failure — request blocked",
            request_id,
        )
    finally:
        elapsed = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(endpoint="/v1/chat/completions").observe(elapsed)
        ACTIVE_CONNECTIONS.dec()


def _record_request_metrics(
    *,
    tenant_id: str = "default",
    barrier_ms: float = 0.0,
    innate_ms: float = 0.0,
    adaptive_ms: float = 0.0,
    policy_ms: float = 0.0,
    upstream_ms: float = 0.0,
    output_ms: float = 0.0,
    total_ms: float = 0.0,
    status: str = "allowed",
) -> None:
    """Record the full per-request latency breakdown and tenant-labeled metrics.

    Called at the end of every request cycle (both allow and block paths).
    Populates LAYER_LATENCY histograms and TENANT_REQUESTS counter. Also
    updates the THREAT_LEVEL gauge and QUARANTINED_SESSIONS gauge.
    """
    TENANT_REQUESTS.labels(tenant_id=tenant_id, status=status).inc()
    if _policy:
        THREAT_LEVEL.set(_policy.threat_level.value)
    if _healing:
        QUARANTINED_SESSIONS.set(len(_healing.quarantine.quarantined_sessions))


async def _update_agent_trust(headers: dict[str, str], delta: float, reason: str) -> None:
    """Update agent trust after request completion. Fire-and-forget."""
    agent_id = headers.get("x-aegis-agent-id")
    if agent_id and _agent_security:
        try:
            await _agent_security.identity_manager.update_trust(agent_id, delta, reason)
        except Exception as e:
            logger.debug("Agent trust update failed for %s: %s", agent_id, e)


async def _process_request(request: Request, request_id: str) -> Response:
    """Core dual-path pipeline processing."""
    assert _barrier and _innate and _adaptive and _output and _policy and _healing

    # --- Parse request ---
    body = await parse_request_body(request)
    headers = extract_headers(request)
    source_ip = extract_source_ip(request)
    raw_body = await get_raw_body(request)
    is_streaming = body.get("stream", False)

    # --- Flatten multimodal list-content messages for ChatMessage parsing ---
    # OpenAI multimodal format uses content: [{type: "text", ...}, {type: "image_url", ...}]
    # but ChatMessage expects content: str. Flatten BEFORE barrier so parsing doesn't crash.
    # The original body dict is preserved for the multimodal preprocessor below.
    original_messages = body.get("messages", [])
    flattened_messages = []
    for msg in original_messages:
        content = msg.get("content")
        if isinstance(content, list):
            # Extract text parts; non-text parts are handled by multimodal preprocessor
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            flat_msg = dict(msg)
            flat_msg["content"] = "\n".join(text_parts) if text_parts else ""
            flattened_messages.append(flat_msg)
        else:
            flattened_messages.append(msg)
    barrier_body = dict(body)
    barrier_body["messages"] = flattened_messages

    # --- L1 Barrier ---
    with_latency = time.perf_counter()
    context = await _barrier.process(barrier_body, headers, source_ip, raw_body)
    context.request_id = request_id
    LAYER_LATENCY.labels(layer="barrier").observe(time.perf_counter() - with_latency)

    # --- Agent identity verification (MHC check) ---
    agent_id_header = headers.get("x-aegis-agent-id")
    if agent_id_header and _agent_security:
        agent = await _agent_security.identity_manager.verify_agent(agent_id_header)
        if agent is None:
            BLOCKS_TOTAL.labels(layer="agent_security", reason="identity_verification_failed").inc()
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": "Agent identity verification failed",
                        "type": "security_block",
                        "code": "aegis_agent_rejected",
                        "request_id": request_id,
                    }
                },
            )
        context.agent_trust_level = agent.trust_level
        # Adjust scrutiny based on agent trust
        if agent.trust_level < 0.3:
            context.metadata["agent_scrutiny_multiplier"] = 2.0
        elif agent.trust_level < 0.5:
            context.metadata["agent_scrutiny_multiplier"] = 1.5
        else:
            context.metadata["agent_scrutiny_multiplier"] = 1.0

    # Check session quarantine
    if _healing.quarantine.is_quarantined(context.session_id):
        BLOCKS_TOTAL.labels(layer="healing", reason="quarantined_session").inc()
        return _error_response(
            403,
            "Session has been quarantined due to repeated adversarial behavior.",
            request_id,
        )

    # --- Multimodal preprocessing (config-gated) ---
    # Extract text from images, documents, audio and add to context for L2/L3 scanning
    mm_image_text = ""
    mm_document_text = ""
    mm_audio_text = ""
    mm_image_scans: list = []
    mm_document_scans: list = []
    mm_audio_scans: list = []
    mm_image_count = 0
    mm_document_count = 0
    mm_audio_count = 0

    if _multimodal and _multimodal.enabled:
        mm_start = time.perf_counter()
        raw_messages = body.get("messages", [])
        mm_report = _multimodal.preprocess_messages(raw_messages)
        mm_elapsed = time.perf_counter() - mm_start
        MULTIMODAL_SCAN_LATENCY.observe(mm_elapsed)

        mm_image_count = mm_report.images_found
        mm_image_scans = list(mm_report.scan_results)

        from aegis.models.request_context import ChatMessage

        if mm_report.has_images:
            MULTIMODAL_IMAGES_SCANNED.inc(mm_report.images_scanned)
            if mm_report.extracted_text:
                MULTIMODAL_OCR_TEXT_EXTRACTED.inc()
                mm_image_text = mm_report.extracted_text
                context.metadata["multimodal_extracted_text"] = mm_report.extracted_text
                context.messages.append(ChatMessage(
                    role="user",
                    content=f"[AEGIS_IMAGE_TEXT_EXTRACTION]\n{mm_report.extracted_text}",
                ))
            for sr in mm_report.scan_results:
                if sr.is_threat:
                    MULTIMODAL_IMAGE_THREATS.labels(threat_type=sr.threat_category.value).inc()
            if mm_report.should_block:
                BLOCKS_TOTAL.labels(layer="multimodal", reason="image_threat").inc()
                return _block_response(request_id, "Request blocked by AEGIS multimodal image scanning.")

        # --- Document scanning (async) ---
        if _multimodal.document_scanning_enabled:
            doc_scans = await _multimodal.scan_documents(raw_messages)
            mm_document_count = mm_report.documents_found
            mm_document_scans = doc_scans
            if doc_scans:
                MULTIMODAL_DOCUMENTS_SCANNED.inc(len(doc_scans))
                for sr in doc_scans:
                    if sr.is_threat:
                        MULTIMODAL_DOCUMENT_THREATS.labels(threat_type=sr.threat_category.value).inc()
                doc_should_block = any(
                    sr.is_threat and sr.confidence >= 0.85 for sr in doc_scans
                )
                if doc_should_block:
                    BLOCKS_TOTAL.labels(layer="multimodal", reason="document_threat").inc()
                    return _block_response(request_id, "Request blocked by AEGIS document scanning.")
            # Extract document text for L2/L3 scanning
            mm_document_text = _multimodal.extract_document_text(raw_messages)
            if mm_document_text.strip():
                context.metadata["multimodal_document_text"] = mm_document_text.strip()
                context.messages.append(ChatMessage(
                    role="user",
                    content=f"[AEGIS_DOCUMENT_TEXT_EXTRACTION]\n{mm_document_text.strip()}",
                ))

        # --- Audio scanning (async) ---
        if _multimodal.audio_scanning_enabled:
            audio_scans = await _multimodal.scan_audio(raw_messages)
            mm_audio_count = mm_report.audio_found
            mm_audio_scans = audio_scans
            if audio_scans:
                MULTIMODAL_AUDIO_SCANNED.inc(len(audio_scans))
                for sr in audio_scans:
                    if sr.is_threat:
                        MULTIMODAL_AUDIO_THREATS.labels(threat_type=sr.threat_category.value).inc()
                audio_should_block = any(
                    sr.is_threat and sr.confidence >= 0.85 for sr in audio_scans
                )
                if audio_should_block:
                    BLOCKS_TOTAL.labels(layer="multimodal", reason="audio_threat").inc()
                    return _block_response(request_id, "Request blocked by AEGIS audio scanning.")

        LAYER_LATENCY.labels(layer="multimodal").observe(
            time.perf_counter() - mm_start
        )

    # --- Cross-modal correlation (config-gated) ---
    if _cross_modal and (mm_image_count + mm_document_count + mm_audio_count) > 0:
        xm_start = time.perf_counter()
        # Build text_scan from the innate text (will be None here, pre-L2)
        xm_report = await _cross_modal.correlate(
            text_content=context.prompt_text,
            text_scan=None,  # L2 hasn't run yet; cross-modal uses its own checks
            image_scans=mm_image_scans,
            document_scans=mm_document_scans,
            audio_scans=mm_audio_scans,
            extracted_text=(mm_image_text + " " + mm_document_text + " " + mm_audio_text).strip(),
            session_id=context.session_id,
            image_count=mm_image_count,
            document_count=mm_document_count,
            audio_count=mm_audio_count,
            image_text=mm_image_text,
            document_text=mm_document_text,
            audio_text=mm_audio_text,
        )
        LAYER_LATENCY.labels(layer="cross_modal").observe(
            time.perf_counter() - xm_start
        )
        if xm_report.should_block:
            BLOCKS_TOTAL.labels(layer="multimodal", reason="cross_modal_threat").inc()
            return _block_response(request_id, "Request blocked by AEGIS cross-modal correlation.")

    # --- L2 Innate (sync, <5ms) ---
    with_latency = time.perf_counter()
    innate_report = await _innate.scan(context)
    LAYER_LATENCY.labels(layer="innate").observe(time.perf_counter() - with_latency)

    # Record per-scanner latency breakdown
    if innate_report.scanner_results:
        for sr in innate_report.scanner_results:
            INNATE_SCANNER_LATENCY.labels(scanner=sr.scanner_id).observe(
                sr.latency_ms / 1000.0
            )

    # --- Record signature matches for clonal-selection deprecation tracking ---
    _record_signature_matches(innate_report, context.prompt_text)

    if innate_report.should_block:
        for cat in innate_report.threat_categories:
            THREATS_DETECTED.labels(layer="innate", category=cat.value).inc()
        BLOCKS_TOTAL.labels(layer="innate", reason="threshold_exceeded").inc()
        _healing.quarantine.record_adversarial_event(context.session_id)
        # Record block for multi-turn rapid-fire detection
        if _adaptive:
            _adaptive.multi_turn.record_block(context.session_id)
        QUARANTINED_SESSIONS.set(len(_healing.quarantine.quarantined_sessions))
        if _audit:
            _audit.log(
                request_id, "innate", "block",
                tenant_id=context.tenant_id,
                source_ip=source_ip,
                session_id=context.session_id,
                threat_categories=[c.value for c in innate_report.threat_categories],
                confidence=innate_report.max_confidence,
                latency_ms=innate_report.total_latency_ms,
            )
        _fire_pg_audit(
            request_id, body=body, context=context,
            innate_report=innate_report, final_action="block",
            block_reason="innate_threshold_exceeded",
            latency_total_ms=(time.perf_counter() - time.perf_counter()) * 1000,
        )
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="403").inc()
        TENANT_REQUESTS.labels(tenant_id=context.tenant_id, status="blocked").inc()
        _record_distillation_interaction(context, was_blocked=True, block_reason="innate_threshold")
        return _block_response(request_id, "Request blocked by AEGIS innate detection.")

    # --- L3 Adaptive (async, parallel with model call) ---
    adaptive_task = asyncio.create_task(
        _run_adaptive(context, innate_report)
    )

    # --- L6 Quick policy check with innate-only scores ---
    # (adaptive result will be checked after model call)
    with_latency = time.perf_counter()
    if isinstance(_policy, OPAPolicyEngine):
        quick_decision = await _policy.evaluate(
            request_id=request_id,
            innate_report=innate_report,
            tenant_id=context.tenant_id,
            model=context.model,
        )
    else:
        quick_decision = _policy.evaluate(
            request_id=request_id,
            innate_report=innate_report,
            tenant_id=context.tenant_id,
            model=context.model,
        )
    LAYER_LATENCY.labels(layer="policy").observe(time.perf_counter() - with_latency)
    POLICY_DECISIONS.labels(
        action=quick_decision.action.value,
        tier=quick_decision.triggered_by.value,
    ).inc()

    if not quick_decision.is_allowed:
        adaptive_task.cancel()
        BLOCKS_TOTAL.labels(layer="policy", reason=quick_decision.action.value).inc()
        _healing.quarantine.record_adversarial_event(context.session_id)
        QUARANTINED_SESSIONS.set(len(_healing.quarantine.quarantined_sessions))
        if _audit:
            _audit.log(
                request_id, "policy", "block",
                tenant_id=context.tenant_id,
                source_ip=source_ip,
                session_id=context.session_id,
                decision_context={"action": quick_decision.action.value},
            )
        _fire_pg_audit(
            request_id, body=body, context=context,
            innate_report=innate_report, final_action="block",
            block_reason=f"policy_{quick_decision.action.value}",
        )
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="403").inc()
        TENANT_REQUESTS.labels(tenant_id=context.tenant_id, status="blocked").inc()
        _record_distillation_interaction(context, was_blocked=True, block_reason="policy")
        return _block_response(request_id, quick_decision.block_message)

    # --- L7 Circuit Breaker check ---
    breaker = _healing.get_breaker("primary")
    use_primary = breaker.should_allow_request()
    CIRCUIT_BREAKER_STATE.labels(endpoint="primary").set(
        {"closed": 0, "open": 1, "half_open": 2}.get(breaker.state.value, 0)
    )

    if not use_primary:
        fallback = _healing.get_fallback("primary")
        if not fallback:
            REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="503").inc()
            return _error_response(
                503,
                "Primary model unavailable and no fallback configured.",
                request_id,
            )

    # Determine upstream URL
    upstream_url = _config.upstream_url.rstrip("/") + "/v1/chat/completions"
    if not use_primary:
        fallback_url = _healing.get_fallback("primary")
        if fallback_url:
            upstream_url = fallback_url.rstrip("/") + "/v1/chat/completions"

    # Extract system prompt for output validation
    system_prompt = None
    for msg in body.get("messages", []):
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break

    # --- Canary token injection ---
    canary_token = None
    if _config.canary.enabled and system_prompt:
        modified_prompt, canary_token = inject_canary(
            system_prompt, context.tenant_id, context.session_id, _config.canary,
        )
        # Update body with canary-injected system prompt for upstream
        body = {**body}
        messages = list(body.get("messages", []))
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i] = {**msg, "content": modified_prompt}
                break
        body["messages"] = messages

    scrutiny_level = quick_decision.output_scrutiny_level
    # Apply agent trust-based scrutiny multiplier
    agent_multiplier = context.metadata.get("agent_scrutiny_multiplier", 1.0)
    if agent_multiplier > 1.0:
        scrutiny_level = min(scrutiny_level * agent_multiplier, 3.0)

    if is_streaming:
        return await _handle_streaming(
            body, upstream_url, request_id, adaptive_task,
            system_prompt, scrutiny_level, context, breaker,
            canary_token=canary_token,
        )
    else:
        return await _handle_non_streaming(
            body, upstream_url, request_id, adaptive_task,
            system_prompt, scrutiny_level, context, breaker,
            canary_token=canary_token,
        )


# ---------------------------------------------------------------------------
# Supply chain path validation
# ---------------------------------------------------------------------------

_FORBIDDEN_PREFIXES = ("/etc", "/proc", "/sys", "/dev", "/root", "/var/run", "/run")


def _validate_model_path(model_path: Path) -> str | None:
    """Validate model_path against path traversal attacks.

    Returns an error message if the path is invalid, or None if valid.
    """
    try:
        resolved = model_path.resolve(strict=False)
    except (OSError, ValueError) as e:
        return f"Invalid model path: {e}"

    # Check for forbidden system directories
    resolved_str = str(resolved)
    for prefix in _FORBIDDEN_PREFIXES:
        if resolved_str.startswith(prefix):
            return f"Access denied: model_path resolves to restricted directory {prefix}"

    # Validate against allowed base directory using Path.parents
    # (string startswith has false positives: /tmp/models-evil matches /tmp/models)
    if _config:
        base = Path(_config.supply_chain_model_base).resolve(strict=False)
        if resolved != base and base not in resolved.parents:
            return (
                f"Access denied: model_path must be within "
                f"{_config.supply_chain_model_base}"
            )

    # Check for symlinks pointing outside the base
    if model_path.is_symlink():
        link_target = model_path.resolve(strict=False)
        if _config:
            base = Path(_config.supply_chain_model_base).resolve(strict=False)
            if link_target != base and base not in link_target.parents:
                return "Access denied: symlink points outside allowed base directory"

    return None


_sig_match_request_count = 0


def _record_signature_matches(innate_report, prompt_text: str) -> None:
    """Record matches against clonal-selection-generated signatures.

    When L2 regex matches a pattern with an ID starting with "sig-" or "CS-"
    (generated by clonal selection), record the match on the corresponding
    SignatureEntry for deprecation tracking. Runs deprecation_check() every
    100 requests.
    """
    global _sig_match_request_count
    if not _signature_store or not innate_report.scanner_results:
        return

    _sig_match_request_count += 1

    # Check regex scanner results for clonal-selection pattern IDs
    for result in innate_report.scanner_results:
        if result.scanner_id != "regex_engine" or not result.matched_patterns:
            continue
        for pattern_desc in result.matched_patterns:
            # Pattern descriptions are formatted as "PATTERN_ID: description"
            pattern_id = pattern_desc.split(":")[0].strip()
            if pattern_id.startswith("sig-") or pattern_id.startswith("CS-"):
                sig = _signature_store.get_by_id(pattern_id)
                if sig:
                    sig.record_match(is_false_positive=False)

    # Periodic deprecation check
    if _sig_match_request_count % 100 == 0:
        deprecated = _signature_store.deprecation_check()
        if deprecated:
            logger.info("Auto-deprecated %d signatures: %s", len(deprecated), deprecated)


def _record_distillation_interaction(
    context, was_blocked: bool, block_reason: str = "", response_text: str = "",
) -> None:
    """Record an interaction for distillation defense cross-session tracking.

    Called on EVERY request path (blocked and allowed) to build per-key history.
    """
    if not _distillation:
        return
    api_key_hash = context.api_key_hash or compute_prompt_hash("unknown")
    _distillation.record_interaction(
        api_key_hash,
        InteractionRecord(
            topic_hash=_distillation._compute_topic_hash(context.prompt_text),
            query_text_summary=context.prompt_text[:200],
            response_length=len(response_text),
            was_blocked=was_blocked,
            block_reason=block_reason,
            complexity_score=_distillation._compute_complexity(context.prompt_text),
            is_reasoning_query=_distillation._is_reasoning_query(context.prompt_text),
        ),
    )


async def _run_adaptive(context, innate_report):
    """Run adaptive analysis and track latency."""
    with_latency = time.perf_counter()
    try:
        report = await _adaptive.analyze(context, innate_report)
        LAYER_LATENCY.labels(layer="adaptive").observe(time.perf_counter() - with_latency)
        if report.should_block:
            for r in report.analyzer_results:
                if r.is_threat:
                    THREATS_DETECTED.labels(layer="adaptive", category=r.threat_category.value).inc()
            # Publish threat_detected event (cytokine signal)
            if _event_bus:
                try:
                    await _event_bus.publish(CHANNEL_THREAT_DETECTED, {
                        "request_id": context.request_id,
                        "session_id": context.session_id,
                        "tenant_id": context.tenant_id,
                        "mcav_score": report.mcav_score,
                        "should_block": True,
                        "is_novel_attack": report.is_novel_attack,
                    })
                    EVENT_BUS_EVENTS.labels(channel=CHANNEL_THREAT_DETECTED).inc()
                except Exception as e:
                    logger.error("Event bus publish failed: %s", e)
            # Publish antibody_generated if novel attack
            if report.is_novel_attack and _event_bus:
                ANTIBODY_GENERATIONS.inc()
                try:
                    await _event_bus.publish(CHANNEL_ANTIBODY_GENERATED, {
                        "request_id": context.request_id,
                        "session_id": context.session_id,
                        "mcav_score": report.mcav_score,
                    })
                    EVENT_BUS_EVENTS.labels(channel=CHANNEL_ANTIBODY_GENERATED).inc()
                except Exception as e:
                    logger.error("Event bus publish failed: %s", e)
        return report
    except Exception as e:
        logger.error("Adaptive analysis failed: %s", e)
        LAYER_LATENCY.labels(layer="adaptive").observe(time.perf_counter() - with_latency)
        return None


async def _handle_non_streaming(
    body: dict,
    upstream_url: str,
    request_id: str,
    adaptive_task: asyncio.Task,
    system_prompt: str | None,
    scrutiny_level: float,
    context,
    breaker,
    canary_token: str | None = None,
) -> Response:
    """Handle non-streaming request: forward, wait for adaptive, validate output."""
    assert _output and _policy and _healing and _http_client and _config

    # --- Forward to upstream model ---
    upstream_start = time.perf_counter()
    try:
        upstream_response = await _forward_to_upstream(body, upstream_url)
        UPSTREAM_LATENCY.labels(endpoint="primary").observe(
            time.perf_counter() - upstream_start
        )
        breaker.record_success()
    except Exception as e:
        UPSTREAM_LATENCY.labels(endpoint="primary").observe(
            time.perf_counter() - upstream_start
        )
        UPSTREAM_ERRORS.labels(endpoint="primary", error_type=type(e).__name__).inc()
        old_state = breaker.state.value
        breaker.record_failure()
        if breaker.state == CircuitBreakerState.OPEN:
            CIRCUIT_BREAKER_TRIPS.labels(endpoint="primary").inc()
            await _publish_circuit_breaker_event(
                breaker.endpoint, old_state, "open", str(e),
            )
        logger.error("Upstream model error: %s", e)
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="502").inc()
        TENANT_REQUESTS.labels(tenant_id=context.tenant_id, status="error").inc()
        return _error_response(502, "Upstream model error", request_id)

    # --- Wait for adaptive result ---
    adaptive_report = await adaptive_task

    # Record per-analyzer latency breakdown
    if adaptive_report and adaptive_report.analyzer_results:
        for ar in adaptive_report.analyzer_results:
            ADAPTIVE_ANALYZER_LATENCY.labels(analyzer=ar.analyzer_id).observe(
                ar.latency_ms / 1000.0
            )

    # Re-evaluate policy with full scores (innate + adaptive)
    if adaptive_report and adaptive_report.should_block:
        BLOCKS_TOTAL.labels(layer="adaptive", reason="mcav_threshold").inc()
        _healing.quarantine.record_adversarial_event(context.session_id)
        # Record block for multi-turn rapid-fire detection
        if _adaptive:
            _adaptive.multi_turn.record_block(context.session_id)
        QUARANTINED_SESSIONS.set(len(_healing.quarantine.quarantined_sessions))
        if _audit:
            _audit.log(request_id, "adaptive", "block", tenant_id=context.tenant_id)
        _fire_pg_audit(
            request_id, body=body, context=context,
            adaptive_report=adaptive_report, final_action="block",
            block_reason="adaptive_mcav_threshold",
        )
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="403").inc()
        TENANT_REQUESTS.labels(tenant_id=context.tenant_id, status="blocked").inc()
        _record_distillation_interaction(context, was_blocked=True, block_reason="adaptive_mcav")
        return _block_response(request_id, "Request blocked by AEGIS adaptive analysis.")

    # --- Canary verification on output ---
    response_text = _extract_response_text(upstream_response)
    if canary_token and _innate:
        canary_result = _innate.canary_verifier.verify_output(
            response_text, expected_token=canary_token,
        )
        if canary_result.is_threat:
            BLOCKS_TOTAL.labels(layer="innate", reason="canary_leakage").inc()
            if _audit:
                _audit.log(
                    request_id, "canary", "block",
                    tenant_id=context.tenant_id,
                    decision_context={"reason": "canary_token_in_output"},
                )
            REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="403").inc()
            return _block_response(
                request_id,
                "Response blocked: system prompt leakage detected.",
            )

    # --- L5 Output Validation ---
    with_latency = time.perf_counter()
    output_result = await _output.validate_full(
        response_text,
        system_prompt=system_prompt,
        scrutiny_level=scrutiny_level,
        context=context,
    )
    LAYER_LATENCY.labels(layer="output").observe(time.perf_counter() - with_latency)

    # Track output cascade stages
    if output_result.pii_result:
        OUTPUT_CASCADE_STAGE.labels(
            stage="pii_redaction",
            result="detected" if output_result.pii_result.redacted_count > 0 else "clean",
        ).inc()
    if output_result.toxicity_result:
        OUTPUT_CASCADE_STAGE.labels(
            stage="toxicity",
            result="detected" if output_result.toxicity_result.is_toxic else "clean",
        ).inc()
    if output_result.leakage_result:
        OUTPUT_CASCADE_STAGE.labels(
            stage="leakage",
            result="detected" if output_result.leakage_result.has_leakage else "clean",
        ).inc()
    if output_result.hallucination_result:
        OUTPUT_CASCADE_STAGE.labels(
            stage="hallucination",
            result="detected" if output_result.hallucination_result.has_hallucination else "clean",
        ).inc()
    if output_result.schema_result:
        OUTPUT_CASCADE_STAGE.labels(
            stage="schema",
            result="detected" if not output_result.schema_result.is_valid else "clean",
        ).inc()

    # Track PII redactions
    if output_result.pii_result and output_result.pii_result.redacted_count > 0:
        for d in output_result.pii_result.detections:
            PII_REDACTIONS.labels(entity_type=d.entity_type).inc()

    # Track toxicity
    if output_result.toxicity_result and output_result.toxicity_result.is_toxic:
        if output_result.toxicity_result.max_category:
            TOXICITY_DETECTIONS.labels(
                category=output_result.toxicity_result.max_category.value
            ).inc()

    if output_result.should_block:
        BLOCKS_TOTAL.labels(layer="output", reason="cascade_block").inc()
        _healing.quarantine.record_adversarial_event(context.session_id)
        if _audit:
            _audit.log(
                request_id, "output", "block",
                tenant_id=context.tenant_id,
                decision_context={"reasons": output_result.reasons},
            )
        _fire_pg_audit(
            request_id, body=body, context=context,
            adaptive_report=adaptive_report,
            output_result=output_result,
            final_action="block",
            block_reason="output_cascade_block",
        )
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="403").inc()
        _record_distillation_interaction(context, was_blocked=True, block_reason="output_cascade", response_text=response_text)
        return _block_response(
            request_id,
            "Response blocked by AEGIS output validation.",
        )

    # Apply redactions if needed
    if output_result.should_redact:
        upstream_response = _apply_redactions_to_response(
            upstream_response, output_result.redacted_text
        )

    # --- Reasoning trace sanitization (L5 Stage 6) ---
    if _reasoning_sanitizer:
        sanitized_text, reasoning_result = _reasoning_sanitizer.scan_and_redact(
            _extract_response_text(upstream_response)
        )
        if reasoning_result.has_reasoning_trace:
            REASONING_TRACES_DETECTED.inc()
        if reasoning_result.has_governance_disclosure:
            GOVERNANCE_DISCLOSURES_DETECTED.inc()
        if reasoning_result.redacted_text and reasoning_result.redacted_text != _extract_response_text(upstream_response):
            upstream_response = _apply_redactions_to_response(
                upstream_response, reasoning_result.redacted_text
            )

    # --- Distillation defense analysis ---
    if _distillation:
        api_key_hash = context.api_key_hash or compute_prompt_hash("unknown")
        distill_report = await _distillation.analyze(
            api_key=api_key_hash,
            query_text=context.prompt_text,
            response_text=response_text,
            was_blocked=False,
        )
        # Record metrics for triggered signals
        for sig in distill_report.signals:
            if sig.triggered:
                DISTILLATION_SIGNALS.labels(strategy=sig.strategy.value).inc()
        if distill_report.should_alert:
            DISTILLATION_ALERTS.inc()
            if _audit:
                _audit.log(
                    request_id, "distillation", "alert",
                    tenant_id=context.tenant_id,
                    decision_context={
                        "combined_score": distill_report.combined_threat_score,
                        "total_queries": distill_report.total_queries,
                        "action": distill_report.recommended_action,
                    },
                )
        if distill_report.should_block:
            DISTILLATION_BLOCKS.inc()
            BLOCKS_TOTAL.labels(layer="distillation", reason="extraction_pattern").inc()
            if _audit:
                _audit.log(
                    request_id, "distillation", "block",
                    tenant_id=context.tenant_id,
                    decision_context={
                        "combined_score": distill_report.combined_threat_score,
                        "total_queries": distill_report.total_queries,
                    },
                )
            REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="403").inc()
            TENANT_REQUESTS.labels(tenant_id=context.tenant_id, status="blocked").inc()
            return _block_response(
                request_id,
                "Request blocked: anomalous query pattern detected.",
            )

    if _audit:
        _audit.log(
            request_id, "pipeline", "allow",
            tenant_id=context.tenant_id,
            decision_context={"redacted": output_result.should_redact},
        )
    _fire_pg_audit(
        request_id, body=body, context=context,
        adaptive_report=adaptive_report,
        output_result=output_result,
        final_action="allow",
    )
    REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="200").inc()
    TENANT_REQUESTS.labels(tenant_id=context.tenant_id, status="allowed").inc()
    # Update threat level gauge
    if _policy:
        THREAT_LEVEL.set(_policy.threat_level.value)
    return JSONResponse(content=upstream_response)


async def _handle_streaming(
    body: dict,
    upstream_url: str,
    request_id: str,
    adaptive_task: asyncio.Task,
    system_prompt: str | None,
    scrutiny_level: float,
    context,
    breaker,
    canary_token: str | None = None,
) -> Response:
    """Handle streaming (SSE) request with StreamingInterceptor.

    Uses StreamingInterceptor for production-grade token-level validation:
    - Hold buffer delays 32 tokens for PII scanning before transmission
    - Sliding 128-token window with 50% overlap evaluates PII/toxicity/leakage
    - Cumulative threat tracking catches slow-drip attacks across windows
    - PII is transparently redacted; toxicity/leakage terminates stream
    """
    assert _output and _healing and _http_client and _config

    interceptor = StreamingInterceptor(
        pii_redactor=_output.pii_redactor,
        toxicity_classifier=_output.toxicity_classifier,
        leakage_detector=_output.leakage_detector,
        system_prompt=system_prompt,
        scrutiny_level=scrutiny_level,
        window_size=_output._window_size,
        overlap=_config.output.streaming_window_overlap,
    )

    async def stream_generator() -> AsyncGenerator[str, None]:
        # Accumulator for tokens waiting to be packed into SSE events
        pending_tokens: list[str] = []

        def _yield_tokens_as_sse(tokens: list[str]) -> str | None:
            """Format released tokens as an SSE data line."""
            if not tokens:
                return None
            content = " ".join(tokens)
            data = {
                "choices": [{
                    "index": 0,
                    "delta": {"content": content},
                }],
            }
            return f"data: {json.dumps(data)}\n\n"

        try:
            async with _http_client.stream(
                "POST",
                upstream_url,
                json=body,
                headers=_upstream_headers(),
            ) as response:
                breaker.record_success()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            # Flush interceptor — release all held tokens
                            flushed, flush_result = interceptor.flush()
                            STREAM_WINDOWS_EVALUATED.inc(interceptor.stats.windows_evaluated)
                            STREAM_TOKENS_PROCESSED.inc(interceptor.stats.tokens_processed)

                            if flush_result and flush_result.should_terminate:
                                reason = _classify_termination_reason(flush_result.reason)
                                STREAM_INTERRUPTIONS.labels(reason=reason).inc()
                                if _event_bus:
                                    await _event_bus.publish(CHANNEL_THREAT_DETECTED, {
                                        "request_id": request_id,
                                        "stream_interrupted": True,
                                        "reason": flush_result.reason,
                                    })
                                yield format_safety_termination(request_id, flush_result.reason)
                                return

                            if flushed:
                                sse = _yield_tokens_as_sse(flushed)
                                if sse:
                                    yield sse
                            yield "data: [DONE]\n\n"
                            return

                        # Extract tokens and feed to interceptor
                        tokens = _extract_sse_tokens(data_str)
                        if tokens:
                            released, result = interceptor.add_tokens(tokens)

                            if result and result.should_terminate:
                                reason = _classify_termination_reason(result.reason)
                                STREAM_INTERRUPTIONS.labels(reason=reason).inc()
                                STREAM_WINDOWS_EVALUATED.inc(interceptor.stats.windows_evaluated)
                                STREAM_TOKENS_PROCESSED.inc(interceptor.stats.tokens_processed)
                                if _event_bus:
                                    await _event_bus.publish(CHANNEL_THREAT_DETECTED, {
                                        "request_id": request_id,
                                        "stream_interrupted": True,
                                        "reason": result.reason,
                                    })
                                yield format_safety_termination(request_id, result.reason)
                                return

                            # Yield released tokens (may be redacted)
                            if released:
                                sse = _yield_tokens_as_sse(released)
                                if sse:
                                    yield sse
                        else:
                            # Non-content SSE data line (e.g., metadata)
                            yield f"{line}\n\n"

        except Exception as e:
            breaker.record_failure()
            if breaker.state == CircuitBreakerState.OPEN:
                CIRCUIT_BREAKER_TRIPS.labels(endpoint="primary").inc()
            logger.error("Streaming upstream error: %s", e)
            yield _format_error_sse(request_id, "Upstream model error")
            return

        # Check adaptive result after stream completes
        if not adaptive_task.done():
            adaptive_report = await adaptive_task
        else:
            adaptive_report = adaptive_task.result() if not adaptive_task.cancelled() else None

        if adaptive_report and adaptive_report.should_block:
            _healing.quarantine.record_adversarial_event(context.session_id)
            if _adaptive:
                _adaptive.multi_turn.record_block(context.session_id)

    # --- Await adaptive result BEFORE streaming tokens to client ---
    # DeBERTa completes in ~20ms; upstream TTFT is 200-400ms, so the
    # adaptive task is almost always done before the first token arrives.
    # Checking here lets us block *before* any response bytes are sent.
    adaptive_report = await adaptive_task
    if adaptive_report and adaptive_report.should_block:
        BLOCKS_TOTAL.labels(layer="adaptive", reason="mcav_threshold").inc()
        _healing.quarantine.record_adversarial_event(context.session_id)
        # Record block for multi-turn rapid-fire detection
        if _adaptive:
            _adaptive.multi_turn.record_block(context.session_id)
        if _audit:
            _audit.log(request_id, "adaptive", "block", tenant_id=context.tenant_id)
        REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="403").inc()
        return _block_response(request_id, "Request blocked by AEGIS adaptive analysis.")

    REQUESTS_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="200").inc()
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


# ---------------------------------------------------------------------------
# PostgreSQL audit helper
# ---------------------------------------------------------------------------


def _fire_pg_audit(
    request_id: str,
    *,
    body: dict | None = None,
    context: Any = None,
    innate_report: Any = None,
    adaptive_report: Any = None,
    output_result: Any = None,
    final_action: str = "allow",
    block_reason: str = "",
    latency_total_ms: float = 0.0,
) -> None:
    """Fire-and-forget PostgreSQL audit log. Never blocks, never raises."""
    if not _pg_audit:
        return

    # Compute prompt hash from raw messages — NEVER store raw prompts
    messages = body.get("messages", []) if body else []
    prompt_hash = compute_prompt_hash(messages)

    # Extract layer results as JSONB
    innate_scanners: dict = {}
    innate_is_threat = False
    innate_confidence = 0.0
    latency_innate = 0.0
    if innate_report:
        innate_is_threat = innate_report.should_block
        innate_confidence = innate_report.max_confidence
        latency_innate = innate_report.total_latency_ms
        innate_scanners = {
            "scanner_count": len(innate_report.scanner_results) if innate_report.scanner_results else 0,
            "threat_categories": [c.value for c in innate_report.threat_categories] if innate_report.threat_categories else [],
        }

    adaptive_analyzers: dict = {}
    adaptive_is_threat = False
    adaptive_confidence = 0.0
    adaptive_mcav = 0.0
    latency_adaptive = 0.0
    if adaptive_report:
        adaptive_is_threat = adaptive_report.should_block
        adaptive_mcav = adaptive_report.mcav_score
        adaptive_confidence = max(
            (r.confidence for r in adaptive_report.analyzer_results if r.is_threat),
            default=0.0,
        )
        # Extract multi-turn details if available
        multi_turn_details: dict = {}
        for ar in adaptive_report.analyzer_results:
            if ar.analyzer_id == "multi_turn_sequence" and ar.details:
                multi_turn_details = {
                    "turn_count": ar.details.get("turns_analyzed", 0),
                    "escalation_score": ar.details.get("escalation_score", 0.0),
                    "pattern_type": ar.details.get("pattern_type", "none"),
                    "triggered_strategies": ar.details.get("triggered_strategies", []),
                }
                break
        adaptive_analyzers = {
            "mcav_score": adaptive_report.mcav_score,
            "is_novel": adaptive_report.is_novel_attack,
            **({"multi_turn": multi_turn_details} if multi_turn_details else {}),
        }

    output_validation: dict = {}
    output_pii = False
    output_toxicity = 0.0
    output_leakage = False
    latency_output = 0.0
    if output_result:
        output_pii = bool(output_result.pii_result and output_result.pii_result.redacted_count > 0)
        if output_result.toxicity_result:
            output_toxicity = output_result.toxicity_result.max_score
        output_leakage = bool(output_result.leakage_result and output_result.leakage_result.has_leakage)
        output_validation = {
            "should_block": output_result.should_block,
            "should_redact": output_result.should_redact,
            "reasons": output_result.reasons if output_result.reasons else [],
        }

    breaker_state = "closed"
    if _healing:
        breaker_state = _healing.get_breaker("primary").state.value

    threat_level = 1
    if _policy:
        threat_level = _policy.threat_level.value

    record = PgAuditRecord(
        request_id=request_id,
        tenant_id=context.tenant_id if context else "",
        user_id=context.user_id if context else "",
        session_id=context.session_id if context else "",
        source_ip=context.source_ip if context else "",
        model=context.model if context else "",
        prompt_hash=prompt_hash,
        token_count=context.metadata.get("token_count", 0) if context and context.metadata else 0,
        innate_is_threat=innate_is_threat,
        innate_confidence=innate_confidence,
        innate_scanners=innate_scanners,
        adaptive_is_threat=adaptive_is_threat,
        adaptive_confidence=adaptive_confidence,
        adaptive_mcav=adaptive_mcav,
        adaptive_analyzers=adaptive_analyzers,
        output_pii_found=output_pii,
        output_toxicity=output_toxicity,
        output_leakage=output_leakage,
        output_validation=output_validation,
        final_action=final_action,
        block_reason=block_reason,
        threat_level=threat_level,
        latency_innate_ms=latency_innate,
        latency_adaptive_ms=latency_adaptive,
        latency_output_ms=latency_output,
        latency_total_ms=latency_total_ms,
        circuit_state=breaker_state,
    )
    _pg_audit.log(record)


# ---------------------------------------------------------------------------
# Upstream communication helpers
# ---------------------------------------------------------------------------


async def _publish_circuit_breaker_event(
    endpoint: str, old_state: str, new_state: str, reason: str,
    failure_rate: float = 0.0,
) -> None:
    """Publish a circuit breaker state transition to the event bus and PostgreSQL."""
    # Event bus publish
    if _event_bus:
        try:
            await _event_bus.publish(CHANNEL_CIRCUIT_BREAKER, {
                "endpoint": endpoint,
                "previous_state": old_state,
                "new_state": new_state,
                "trigger_reason": reason,
            })
            EVENT_BUS_EVENTS.labels(channel=CHANNEL_CIRCUIT_BREAKER).inc()
        except Exception as e:
            logger.error("Event bus circuit_breaker publish failed: %s", e)

    # PostgreSQL persist (fire-and-forget)
    session_factory = get_session_factory()
    if session_factory:
        try:
            asyncio.create_task(log_circuit_breaker_event(
                session_factory=session_factory,
                endpoint=endpoint,
                previous_state=old_state,
                new_state=new_state,
                trigger_reason=reason,
                failure_rate=failure_rate,
                window_seconds=_config.healing.circuit_breaker_window_seconds if _config else 60,
                cooldown_seconds=_config.healing.cooldown_seconds if _config else 30,
            ))
        except RuntimeError:
            pass


async def _forward_to_upstream(body: dict, upstream_url: str) -> dict:
    """Forward request to upstream model and return JSON response."""
    assert _http_client and _config

    # Don't stream for non-streaming requests
    forward_body = {**body, "stream": False}

    response = await _http_client.post(
        upstream_url,
        json=forward_body,
        headers=_upstream_headers(),
    )
    response.raise_for_status()
    return response.json()


def _upstream_headers() -> dict[str, str]:
    """Build headers for upstream model requests."""
    headers = {"Content-Type": "application/json"}
    if _config and _config.upstream_api_key:
        headers["Authorization"] = f"Bearer {_config.upstream_api_key}"
    return headers


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _extract_response_text(response: dict) -> str:
    """Extract the text content from an OpenAI chat completion response."""
    try:
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "") or ""
    except (IndexError, AttributeError, TypeError):
        pass
    return ""


def _apply_redactions_to_response(response: dict, redacted_text: str) -> dict:
    """Replace the response text with the redacted version."""
    try:
        response = {**response}
        choices = list(response.get("choices", []))
        if choices:
            choice = {**choices[0]}
            message = {**choice.get("message", {})}
            message["content"] = redacted_text
            choice["message"] = message
            choices[0] = choice
            response["choices"] = choices
    except (IndexError, AttributeError, TypeError):
        pass
    return response


def _classify_termination_reason(reason: str) -> str:
    """Classify a termination reason string into a metric label."""
    reason_lower = reason.lower()
    if "toxic" in reason_lower:
        return "toxicity"
    if "leakage" in reason_lower or "leak" in reason_lower:
        return "leakage"
    if "cumulative" in reason_lower:
        return "cumulative"
    if "pii" in reason_lower:
        return "pii"
    return "other"


def _extract_sse_tokens(data_str: str) -> list[str]:
    """Extract tokens from an SSE data chunk (OpenAI format)."""
    try:
        data = json.loads(data_str)
        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                return content.split()
    except (json.JSONDecodeError, IndexError, AttributeError, TypeError):
        pass
    return []


def _block_response(request_id: str, message: str) -> JSONResponse:
    """Build a standard block response."""
    return JSONResponse(
        status_code=403,
        content={
            "error": {
                "message": message,
                "type": "security_block",
                "code": "aegis_blocked",
                "request_id": request_id,
            }
        },
    )


def _error_response(status_code: int, message: str, request_id: str) -> JSONResponse:
    """Build a standard error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "error",
                "code": f"aegis_{status_code}",
                "request_id": request_id,
            }
        },
    )


def _format_safety_sse(request_id: str, reasons: list[str]) -> str:
    """Format a safety termination SSE event."""
    data = {
        "id": f"aegis-safety-{request_id}",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "\n\n[AEGIS Security: Response terminated due to safety concern. "
                          f"Reasons: {'; '.join(reasons)}]"
            },
            "finish_reason": "content_filter",
        }],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"


def _format_error_sse(request_id: str, message: str) -> str:
    """Format an error SSE event."""
    data = {
        "id": f"aegis-error-{request_id}",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": f"\n\n[AEGIS Error: {message}]"},
            "finish_reason": "stop",
        }],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"
