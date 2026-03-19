# AEGIS — Claude Code Project Guide

**THIS FILE MUST NEVER BE DELETED. It is the institutional memory for this project. Update it in place when the codebase changes. Never remove or overwrite it.**

## What Is AEGIS

AEGIS (Adaptive Enterprise Guard for Intelligent Systems) is a commercial AI security platform that implements the human biological immune system as a unified, adaptive defense wrapper for enterprise AI systems. It sits between applications and AI model providers as an OpenAI-compatible reverse proxy, inspecting and protecting every interaction.

Applications connect by changing their base_url to point to AEGIS. Zero code changes required.

## Architecture: Eight-Layer Immune System

| Layer | Name | Immune Analog | Latency |
|-------|------|---------------|---------|
| L1 | Barrier | Skin / Mucous Membranes | <1ms |
| L2 | Innate Detection | Pattern Recognition Receptors / NK Cells | <5ms |
| L3 | Adaptive Analysis | T-Cells / B-Cells / Dendritic Cell Algorithm | 10-50ms async |
| L4 | Immune Memory | Memory B-Cells / Antibodies (FAISS Threat Vault) | 5-20ms |
| L5 | Output Validation | Complement System (5-stage cascade) | 10-100ms |
| L6 | Policy Engine | Regulatory T-Cells | <1ms |
| L7 | Self-Healing | Wound Healing / Tissue Repair | Event-driven |
| L8 | Federated Intel | Herd Immunity | Async batch (6h rounds) |

Dual-path processing: Every request runs through L2 (fast, synchronous, <5ms) and L3 (slow, asynchronous, 10-50ms) in parallel. Both produce threat scores merged by L6 Policy Engine.

Fail-closed: If any layer raises an unhandled exception or is None, the pipeline returns 503. Requests are never passed through unscanned.

## Directory Structure

aegis/
├── main.py                          # FastAPI entry point, all endpoints, layer wiring
├── config.py                        # AegisConfig (Pydantic), all thresholds and tunables
├── CLAUDE.md                        # THIS FILE — never delete
├── models/
│   ├── request_context.py           # RequestContext with identity metadata
│   ├── scan_result.py               # ScanResult, InnateScanReport, ThreatCategory enum
│   └── threat_indicator.py          # ThreatIndicator, IndicatorSource, LifecyclePhase enums
├── layers/
│   ├── barrier.py                   # L1: Rate limiting, auth, schema validation
│   ├── innate/
│   │   ├── __init__.py              # InnateDetectionLayer orchestrator
│   │   ├── regex_engine.py          # Scanner 1: 188 patterns, 8-step Unicode normalization (200+ homoglyphs)
│   │   ├── blocklist.py             # Scanner 2: Bloom filter with hash confirmation
│   │   ├── token_guard.py           # Scanner 4: Token counting and distribution analysis
│   │   ├── pii_regex.py             # Scanner 5: Fast PII regex (SSN, CC, email, phone)
│   │   ├── canary_verifier.py       # Scanner 6: HMAC-SHA256 canary token (NK cell analog)
│   │   ├── sliding_window.py        # Scanner 7: Sliding window for padding dilution defense
│   │   └── multilang_detector.py    # Scanner 8: Multi-language injection (10 languages, 54 patterns)
│   ├── adaptive/
│   │   ├── __init__.py              # AdaptiveAnalysisLayer with antibody callback
│   │   ├── injection_classifier.py  # DeBERTa-v3 prompt injection (ONNX, ~20ms)
│   │   ├── semantic_search.py       # Embedding + FAISS similarity search
│   │   ├── behavioral.py            # Baseline anomaly detection (PSI, KS tests)
│   │   ├── multi_turn.py            # Multi-turn: escalation trajectory, boundary testing, rapid-fire, topic drift
│   │   ├── distillation_defense.py  # Cross-session extraction detection (5 strategies)
│   │   ├── distillation_models.py   # Data models: InteractionRecord, DistillationSignal/Report, ReasoningScanResult
│   │   ├── margin_booster.py        # Confidence margin booster for fragile DeBERTa detections
│   │   ├── manipulation_detector.py # MTMD: 5-strategy autonomous jailbreak agent detection
│   │   └── source_profiler.py       # Source behavioral profiling: diversity, injection rate, automation
│   ├── adaptive_rate_limiter.py      # Adaptive per-source rate limiting (slowdown, cooling, hard stop)
│   ├── tool_proxy/
│   │   ├── __init__.py              # Tool Invocation Proxy (TIP) package exports
│   │   ├── proxy.py                 # TIP: intercept agent-to-tool calls, MCP proxy, invocation logging
│   │   ├── policy_engine.py         # TIPE: allowlist, rate limit, param validation (path/SSRF/injection)
│   │   ├── description_validator.py # TDIV: tool description injection scanning, quarantine set
│   │   ├── response_sanitizer.py    # TRS: tool output injection scanning, content delimiting, size enforcement
│   │   └── chain_detector.py        # TCAD: parasitic toolchain detection (dangerous sequences, volume, ordering)
│   ├── memory/
│   │   ├── threat_vault.py          # FAISS HNSW index, 3-phase lifecycle
│   │   ├── signatures.py           # Clonal selection generator + SignatureStore
│   │   └── jailbreak_taxonomy.py   # 15-category jailbreak attempt classification and trend tracking
│   ├── output/
│   │   ├── __init__.py              # OutputValidationLayer: 5-stage cascade orchestrator
│   │   ├── pii_redactor.py          # Stage 1: Presidio PII + 8 secret types (AWS, OpenAI, Anthropic, GitHub, etc.)
│   │   ├── toxicity.py              # Stage 2: Keyword + n-gram classifier, LlamaGuard stub, sensitivity levels
│   │   ├── hallucination.py         # Stage 3: N-gram source coverage for RAG responses
│   │   ├── leakage.py               # Stage 4: System prompt echo detection
│   │   ├── schema_validator.py      # Stage 5: JSON schema validation, injected field detection
│   │   ├── reasoning_sanitizer.py   # Stage 6: Reasoning trace sanitization (monitor/redact/summarize)
│   │   ├── coherence_analyzer.py    # Extension 1.2: Sliding-window coherence, pivot detection, CoT hijacking
│   │   ├── length_anomaly_detector.py # Extension 1.3: EMA baselines, per-(tenant,session), length anomaly
│   │   ├── alignment_validator.py   # Extension 1.4: Reasoning-output semantic alignment, misalignment detection
│   │   └── streaming.py             # StreamingInterceptor: hold buffer, PII scrub, cumulative threat
│   ├── policy/
│   │   ├── __init__.py              # L6: PolicyEngine, TenantPolicy, TLI (5 levels), score fusion
│   │   └── opa_engine.py            # OPAPolicyEngine: OPA HTTP backend with Python fallback
│   ├── healing.py                   # L7: Circuit breaker, session quarantine
│   ├── audit.py                     # JSONL audit logger with crash-safe flush
│   ├── supply_chain/
│   │   ├── __init__.py              # SupplyChainVerifier orchestrator (aggregate risk scoring)
│   │   ├── models.py                # StageResult, ModelVerificationReport (with risk_score)
│   │   ├── integrity.py             # Stage 1: SHA-256 hash verification (original)
│   │   ├── sigstore_verifier.py     # Stage 1 Enhanced: SHA-256 manifest + Sigstore signatures
│   │   ├── serialization.py         # Stage 2: Format risk scoring, pickle-in-ZIP, ONNX/SafeTensors validation
│   │   ├── dependency_audit.py      # Stage 3: CVE matching + SBOM generation
│   │   └── behavioral_probe.py      # Stage 4: 10 jailbreak probes, sleeper agent detection, divergence
│   ├── multimodal/
│   │   ├── __init__.py              # MultimodalPreprocessor: image, document, audio orchestrator
│   │   ├── image_scanner.py         # L2: format validation, OCR, metadata, steganalysis
│   │   ├── image_analyzer.py        # L3: sanitize-compare, adversarial heuristics
│   │   ├── ocr_engine.py            # Pillow heuristic + Tesseract OCR (3-tier fallback)
│   │   ├── image_sanitizer.py       # Re-encode to strip steganographic payloads
│   │   ├── steganalysis.py          # Chi-square LSB + RS analysis + alpha channel
│   │   ├── metadata_stripper.py     # EXIF/XMP/IPTC extraction
│   │   ├── document_scanner.py      # L2: document format validation, text extraction, hidden content
│   │   ├── document_analyzer.py     # L3: DeBERTa + semantic search on extracted text
│   │   ├── text_extractor.py        # Format-aware text extraction (PDF, HTML, Office, etc.)
│   │   ├── hidden_content_detector.py # Invisible CSS, comments, zero-width, macros, scripts
│   │   ├── format_validator.py      # Magic bytes, polyglot detection, size limits
│   │   ├── audio_transcriber.py     # Multi-backend audio-to-text (Whisper, speech_recognition, fallback)
│   │   ├── spectral_analyzer.py     # FFT spectral analysis: ultrasonic, infrasonic, entropy, bursts
│   │   ├── audio_sanitizer.py       # WaveGuard re-encoding: bandpass filter + WAV normalization
│   │   ├── audio_scanner.py         # L2: format/size/duration validation, spectral, transcribe→regex
│   │   ├── audio_analyzer.py        # L3: WaveGuard comparison, injection classifier, cross-modal
│   │   ├── cross_modal_engine.py    # Cross-modal correlation: laundering, inconsistency, escalation, volume, fragmentation, decoy
│   │   └── tool_use_scanner.py      # Tool use security: definition scanning, chain analysis, output scanning
│   └── agent_security/
│       ├── __init__.py              # AgentSecurityLayer orchestrator (MHC identity verification)
│       ├── identity.py              # AgentIdentityManager: JWT signing, trust mechanics, decay
│       ├── authorization.py         # AgentAuthorizationEngine: least-privilege, scope, escalation blocking
│       ├── message_validator.py     # AgentMessageValidator: injection scanning, quarantine
│       └── communication_monitor.py # IACM: inter-agent lateral movement detection, compromised agent flagging
├── services/
│   ├── __init__.py
│   ├── redis_client.py              # Async Redis singleton, health check, graceful degradation
│   ├── event_bus.py                 # EventBus: InMemoryEventBus / RedisEventBus (Redis Streams)
│   ├── db.py                        # Async SQLAlchemy + asyncpg singleton, health check, graceful degradation
│   ├── audit_logger.py              # PostgreSQL audit logger, fire-and-forget, in-memory buffer
│   ├── tenant_manager.py            # Multi-tenant config: cache, resolve, refresh, graceful degradation
│   ├── threat_intel.py              # STIX/TAXII ingestion, export with DP noise, dedup
│   ├── compliance/                  # ComplianceEngine: 6 frameworks × 10 capabilities, evidence, reports
│   ├── federated/                   # FederatedIntelligenceManager: DP noise, FedAvg, indicator sharing
│   └── deployment/                  # ConfigValidator, VaultBackupManager, DeepHealthMonitor
├── middleware/
│   ├── request_enrichment.py        # Build RequestContext
│   └── metrics.py                   # Prometheus metrics collection (30+ metrics)
├── data/
│   ├── patterns.json                # 188 regex patterns (MITRE ATLAS tagged)
│   ├── blocklist.txt                # Known malicious payloads
│   ├── seed_threats.json            # Initial threat vault embeddings
│   ├── known_cves.json              # CVE database for supply chain audit
│   ├── benign_prompts.json          # 55 benign prompts for clonal selection validation
│   ├── benchmark_benign.json        # 500 business prompts across 10 industries
│   ├── benchmark_attacks.json       # 110 labeled attacks across 10 categories
│   ├── stix_feeds/                  # STIX 2.1 indicator feeds (13 seed indicators)
│   └── opa_policies/                # OPA Rego policies (3-tier: global/tenant/adaptive)
├── tests/                           # 2628+ tests across 40+ test files
├── red_team/                        # Adversarial testing: 6 APT campaigns, multimodal APT, white-box
├── demo/                            # Interactive 6-scenario demo (requires Ollama)
├── docs/                            # Threat model (23 threats), deployment guide
├── requirements.txt                 # Core deps: FastAPI, transformers, FAISS, Presidio, pytesseract
├── requirements-docker.txt          # Docker deps: redis, asyncpg, sqlalchemy, alembic, pytesseract
├── Dockerfile                       # Multi-stage: builder (deps+models+tesseract) → runtime (slim)
├── docker-compose.yml               # Dev stack: AEGIS + Redis + PostgreSQL + OPA
├── docker-entrypoint.sh             # Waits for Redis/Postgres, runs alembic, prints startup info
└── db/
    └── init.sql                     # PostgreSQL schema (5 tables: tenants, audit_log, threat_indicators, signatures, circuit_breaker_events)

## Key Patterns

All scanners return ScanResult objects with: scanner_id, is_threat, confidence (0.0-1.0), threat_category (ThreatCategory enum), matched_patterns, sanitized_input, latency_ms.

Dual-path execution in main.py:
- innate_result = await _innate.scan(context) runs synchronous (<5ms)
- if innate_result.should_block: return block immediately
- adaptive_task = asyncio.create_task(_adaptive.analyze(context)) runs async in parallel with model call
- model_response = await forward_to_model(context)
- adaptive_result = await adaptive_task
- if adaptive_result.should_block: return block before delivering response

Antibody learning loop: When L3 **blocks** a novel attack that L2 missed (`is_novel and should_block`), the system embeds the attack in the vault, triggers clonal selection (generate regex candidates, affinity test against benign corpus, promote best to L2 regex engine). CRITICAL: antibody generation is gated on `should_block`, NOT just `adaptive_caught` — sub-threshold DeBERTa detections must NOT generate antibodies, as borderline/FP prompts would pollute the vault.

Adaptive blocking: MCAV fusion score (>= 0.9) OR any single analyzer with confidence >= 0.90 triggers block. The 0.90 threshold (not 0.85) avoids FPs from borderline DeBERTa scores on business prompts with injection-adjacent vocabulary.

Streaming handler: Adaptive task is awaited BEFORE returning StreamingResponse (not inside stream_generator). This ensures adaptive blocks prevent any response bytes from reaching the client.

Unicode normalization: All text passes through an 8-step pipeline before regex matching: (1) NFKC, (2) BIDI stripping, (3) zero-width stripping, (4) combining mark stripping, (5) homoglyph canonicalization (200+ mappings), (6) leetspeak normalization, (7) recursive base64 decode (max 3 iterations), (8) whitespace collapse. Implemented in regex_engine.normalize_text().

Threat vault lifecycle: Acute (0-30 days) → Persistent (3+ sources or confirmed) → Dormant (180 days unseen). Dormant reactivates to Acute on re-emergence.

Circuit breaker: Closed → Open (on threshold breach) → Half-Open (after cooldown) → Closed (if probes pass). Per-model-endpoint tracking.

Event bus: InMemoryEventBus in tests; RedisEventBus in production. Channels: `threat_detected`, `antibody_generated`, `circuit_breaker`, `policy_escalation`, `audit_event`, `tool_violation`.

Backing services (Redis, PostgreSQL): Both optional — AEGIS degrades gracefully to in-memory implementations. AEGIS must NEVER crash because a backing service is down.

## Endpoints

| Endpoint | Auth | Purpose |
|----------|------|---------|
| POST /v1/chat/completions | Yes | Main proxy — OpenAI-compatible |
| GET /health | No/Yes | Unauthenticated: basic. Authenticated: full layer status |
| GET /metrics | Yes | Prometheus metrics |
| GET /v1/models | No | List available models |
| GET /v1/audit/recent | Yes | Recent audit records |
| GET /v1/vault/stats | Yes | Threat vault statistics |
| POST /v1/supply-chain/verify | Yes | Run 4-stage model verification |
| POST /v1/threat-intel/ingest | Yes | Ingest STIX 2.1 threat bundle |
| GET /v1/threat-intel/export | Yes | Export indicators (DP noise) |
| POST /v1/agents/register | Yes | Register agent, return JWT |
| POST /v1/agents/{id}/authorize | Yes | Check agent tool authorization |
| POST /v1/agents/message/validate | Yes | Validate inter-agent message |
| GET /v1/compliance/matrix | Yes | Cross-framework coverage matrix |
| POST /v1/compliance/report | Yes | Generate compliance report |
| POST /v1/federated/round | Yes | Trigger federated learning round |
| POST /v1/admin/backup | Yes | Backup vault + signatures |
| POST /v1/admin/restore | Yes | Restore vault from backup |
| GET /v1/taxonomy/stats | Yes | Jailbreak taxonomy statistics |
| GET /v1/tool-proxy/stats | Yes | Tool proxy invocation and violation stats |
| POST /v1/tool-proxy/validate-description | Yes | TDIV: validate tool description for injection |
| GET /v1/agents/communication/stats | Yes | IACM: inter-agent communication monitor stats |
| GET /v1/admin/deep-health | Yes | Deep health check (6 components) |

## Running Tests

Full suite: `python3 -m pytest tests/ -x -q --tb=short -p no:warnings --ignore=tests/benchmark`
Benchmark: `python3 -m pytest tests/benchmark/ -q --tb=short -p no:warnings`
Single file: `python3 -m pytest tests/test_attack_battery.py -x -q --tb=short -p no:warnings`
Stress (live): `AEGIS_STRESS_FULL=1 python3 -m pytest tests/stress/test_stress.py -v --tb=short`
APT campaigns: `python3 -m red_team.run_red_team`

## Current Metrics (as of 2026-03-19)

- Tests: 2628 passing, 0 failed, 8 skipped (5 stress require AEGIS_STRESS_FULL=1, 2 Tesseract-dependent require tesseract-ocr binary, 1 API key-dependent)
- Benchmark (with DeBERTa): 110 attacks, 500 benign prompts
- TPR (full stack, DeBERTa loaded): 96.36% (106/110)
- TPR (innate L2 only): 95.45% (105/110)
- FPR: 0.00% (0/500) — with DeBERTa loaded
- Pattern library: 188 patterns + 54 multi-language patterns (+ dynamic patterns from clonal selection at runtime)
- CHIMERA campaign: 95% detection (19/20 in production Docker, 90% in CI without Tesseract)
- Threat model: 23 threats (T1-T23), pen test recommendations in docs/
- Live demo: All 6 scenarios working with Ollama (llama3.2:3b)

## Configuration

All configuration via environment variables prefixed AEGIS_ or via AegisConfig in config.py. Key variables:

- AEGIS_UPSTREAM_URL — upstream LLM provider URL
- AEGIS_UPSTREAM_API_KEY — upstream API key
- AEGIS_API_KEY — gateway API key required from clients
- AEGIS_CANARY_SECRET_KEY — HMAC secret for canary tokens (MUST be changed in production)
- AEGIS_BLOCK_THRESHOLD — confidence threshold for blocking (default 0.85)
- AEGIS_ALERT_THRESHOLD — confidence threshold for alerting (default 0.50)
- AEGIS_SUPPLY_CHAIN_MODEL_BASE — allowed base directory for model verification
- AEGIS_SKIP_MODEL_LOAD — skip DeBERTa/sentence-transformers loading (tests/CI). conftest.py sets automatically.
- AEGIS_MODEL — model name for upstream (e.g., llama3.2:3b for Ollama)
- AEGIS_POLICY_BACKEND — 'python' (default) or 'opa'
- AEGIS_POLICY_OPA_URL — OPA server URL (default: http://localhost:8181)
- AEGIS_DISTILLATION_DEFENSE_ENABLED — enable cross-session distillation defense (default: true)
- AEGIS_REASONING_TRACE_MODE — reasoning trace sanitization: monitor, redact, or summarize (default: monitor)
- AEGIS_TLI_AUTO_DECAY_ENABLED — automatic TLI de-escalation timers (default: true)
- AEGIS_MANIPULATION_DETECTION_ENABLED — enable MTMD autonomous jailbreak agent detection (default: true)
- AEGIS_MTMD_WINDOW_SIZE — MTMD sliding window size (default: 20)
- AEGIS_MTMD_BLOCK_THRESHOLD — MTMD blocking threshold (default: 0.7)
- AEGIS_MTMD_ALERT_THRESHOLD — MTMD alerting threshold (default: 0.4)
- AEGIS_PROFILER_ENABLED — enable source behavioral profiling (default: true)
- AEGIS_PROFILER_MAX_PROFILES — max source profiles (default: 10000)
- AEGIS_ADAPTIVE_RATE_LIMIT_ENABLED — enable adaptive per-source rate limiting (default: true)
- AEGIS_ADAPTIVE_RATE_LIMIT_HARD_STOP_THRESHOLD — manipulation attempts before hard stop (default: 5)
- AEGIS_ADAPTIVE_RATE_LIMIT_HARD_STOP_DURATION — hard stop duration in seconds (default: 300)
- AEGIS_JAILBREAK_TAXONOMY_ENABLED — enable jailbreak attempt taxonomy logging (default: true)
- AEGIS_JAILBREAK_TAXONOMY_MAX_ATTEMPTS — max stored attempts (default: 50000)
- AEGIS_TOOL_PROXY_ENABLED — enable Tool Invocation Proxy (default: true)
- AEGIS_TOOL_PROXY_DEFAULT_RPM — default per-tool rate limit (default: 60)
- AEGIS_TOOL_PROXY_MAX_PARAM_SIZE — max parameter size in bytes (default: 10000)
- AEGIS_TOOL_PROXY_MAX_INVOCATION_LOG — max invocation log entries (default: 10000)
- AEGIS_TOOL_PROXY_BLOCK_INTERNAL_URLS — block internal/private IP URLs (default: true)
- AEGIS_TOOL_PROXY_REQUIRE_HTTPS — require HTTPS for URL params (default: false)
- AEGIS_TDIV_ENABLED — enable Tool Description Integrity Validator (default: true)
- AEGIS_TDIV_MAX_DESCRIPTION_LENGTH — max tool description field length (default: 2000)
- AEGIS_TRS_ENABLED — enable Tool Response Sanitizer (default: true)
- AEGIS_TRS_DEFAULT_MAX_OUTPUT_SIZE — max tool output size in bytes (default: 102400)
- AEGIS_TRS_BLOCK_THRESHOLD — injection confidence for blocking tool output (default: 0.85)
- AEGIS_TCAD_ENABLED — enable Tool Chain Anomaly Detector (default: true)
- AEGIS_TCAD_WINDOW_SIZE — per-session tool chain window size (default: 20)
- AEGIS_TCAD_VOLUME_MULTIPLIER — volume anomaly threshold multiplier (default: 3.0)
- AEGIS_TCAD_BASELINE_SESSIONS — sessions before unusual ordering detection activates (default: 100)
- AEGIS_IACM_ENABLED — enable Inter-Agent Communication Monitor (default: true)
- AEGIS_IACM_TRUST_THRESHOLD — sender trust level for elevated scrutiny (default: 0.3)
- AEGIS_IACM_INJECTION_RATE_THRESHOLD — injection rate for compromised agent flagging (default: 0.5)
- AEGIS_COT_DEFENSE_ENABLED — enable chain-of-thought hijacking defense (default: true)
- AEGIS_COT_COHERENCE_WINDOW_SIZE — tokens per coherence window (default: 200)
- AEGIS_COT_COHERENCE_PIVOT_THRESHOLD — coherence drop below this is a pivot (default: 0.3)
- AEGIS_COT_LENGTH_ANOMALY_MULTIPLIER — baseline multiplier for length anomaly (default: 3.0)
- AEGIS_COT_LENGTH_ANOMALY_DEFAULT_BASELINE — default trace length baseline (default: 500)
- AEGIS_COT_ALIGNMENT_MISALIGN_THRESHOLD — alignment below this is MISALIGNED (default: 0.3)
- AEGIS_COT_DEFENSE_BLOCK_THRESHOLD — combined CoT score for blocking (default: 0.7)
- REDIS_URL — enables Redis-backed rate limiting and Redis Streams event bus
- DATABASE_URL — enables persistent audit logging, threat indicators, signatures

## Docker

Multi-stage build: builder installs deps + downloads DeBERTa/MiniLM models (~500MB), runtime copies venv + models + app code. Non-root `aegis` user (uid 1000). Tesseract OCR + 7 language packs (chi-sim, chi-tra, jpn, kor, ara, hin, rus) in both stages.

```bash
docker compose up -d              # Start AEGIS + Redis + PostgreSQL + OPA
docker compose logs -f aegis      # Follow AEGIS logs
docker compose down -v            # Stop and remove volumes
docker build -t aegis .           # Build image only
```

Key paths: `/app/aegis/` (code), `/opt/models/` (ML models), `/app/aegis/logs/` (audit), `/app/aegis/data/` (patterns).

## Technology Stack

| Component | Technology |
|-----------|-----------|
| API Gateway | FastAPI + uvicorn + httpx (ASGI async) |
| Streaming | sse-starlette (SSE for token-level inspection) |
| Prompt Injection ML | DeBERTa-v3-base (ONNX, ~20ms inference) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) |
| Vector Search | FAISS (HNSW index, sub-ms at 1M+ vectors) |
| PII Detection | Microsoft Presidio (NER + regex + checksum) |
| OCR | Tesseract + pytesseract (CJK/multilingual) with Pillow fallback |
| Policy Engine | OPA (Rego) with Python fallback |
| Event Bus | Redis Streams (prod) / asyncio.Queue (dev) |
| Database | PostgreSQL + pg_trgm (audit, signatures, tenants) |
| Federated Learning | Custom FedAvg + Google dp-accounting (ε=3) |
| Observability | Prometheus + 30+ metrics |

## Critical Rules

1. NEVER delete CLAUDE.md — this is the project's institutional memory
2. Fail-closed everywhere — if a security layer fails, block the request (503), never pass through
3. All tests must pass before any changes are considered complete — current baseline is 2628+
4. Benchmark thresholds: FPR < 1.0%, TPR >= 85%, no single industry FPR > 3%
5. Unicode normalize before regex — all text through normalize_text() before pattern matching
6. Auth required on sensitive endpoints — /metrics, /v1/audit/recent, /v1/vault/stats require valid Bearer token
7. Path validation on supply chain — model_path must resolve within supply_chain_model_base using Path.parents
8. Audit log flush — f.flush() after every JSONL write to prevent corruption on crash
9. No time.sleep() in tests — use timestamp backdating for expiry/cooldown simulation
10. Existing test files and test counts must not decrease — only add, never remove tests
11. CLAUDE.md must stay under 500 lines. When it grows past 500, split operational/narrative content to CLAUDE_OPS.md.

---
**For red team results and multimodal APT campaigns, see CLAUDE_EXTENDED.md. For bug fixes, hardening history, and operational changes, see CLAUDE_OPS.md.**
