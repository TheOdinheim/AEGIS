# AEGIS Operational History — Bug fixes, hardening, and change log

**For core architecture, see CLAUDE.md. For red team results, see CLAUDE_EXTENDED.md.**

---

## Bugs Fixed (2026-03-03 session)

1. **Hardcoded skip_model_load=True** (main.py): DeBERTa never loaded in any context. Fixed: reads `AEGIS_SKIP_MODEL_LOAD` env var, defaults to False. Created `tests/conftest.py` to set it for test runs.
2. **Streaming handler late adaptive check** (main.py _handle_streaming): Adaptive result was checked AFTER entire response stream sent to client. Fixed: await adaptive task BEFORE returning StreamingResponse.
3. **MCAV dilution** (layers/adaptive/__init__.py): Single DeBERTa PAMP (0.9999) diluted below block threshold by 3 SAFE signals from analyzers with no data. Fixed: `any_high_confidence` override — any analyzer with confidence >= 0.85 triggers block regardless of MCAV.
4. **Clonal selection patterns too specific** (layers/memory/signatures.py): N-gram patterns only matched exact attack text. Fixed: added semantic verb-target pair extraction (`_SEMANTIC_VERBS` x `_SEMANTIC_TARGETS`) with bidirectional regex matching.
5. **Affinity threshold unreachable** (layers/memory/signatures.py): `compute_affinity` max score with TPR=1.0/FPR=0 was 0.65, below 0.7 threshold. Large positive set (25 seed attacks) made TPR=1/26=0.038 for novel attack patterns. Fixed: two-stage affinity testing (source-only + breadth) with 1.15x antibody activation boost for source-matching patterns.
6. **Missing `import os`** (main.py): Caused NameError on `os.environ.get()` after env var fix.
7. **DeBERTa FP on business prompts** (layers/adaptive/__init__.py): `any_high_confidence` threshold was 0.85, but DeBERTa scores "override the previous estimate" at 0.874. Raised threshold to 0.90.
8. **Antibody cascade FP** (layers/adaptive/__init__.py): Antibody generation was gated on `is_novel and adaptive_caught` — sub-threshold DeBERTa detections (is_threat=True but below block threshold) generated antibodies from benign prompts, which clonal selection promoted to L2, cascading into 4 additional FPs. Fixed: gated on `is_novel and should_block` instead.
9. **Stale vault pollution**: Persisted vault files accumulated junk antibodies from repeated test/benchmark runs. Added `data/threat_vault_meta.*` to .gitignore.

## Bugs Fixed (2026-03-05 session)

10. **Health endpoint Redis/PG status check** (main.py lines 591, 600): `redis_health()` returns `{"status": "connected"}` but health check was using `.get("connected", False)` which always returned `False`, causing Redis/PG to always show as "degraded" even when connected. Fixed to `.get("status") == "connected"`.

## Production Hardening (2026-03-17)

Seven fixes applied before production deployment:

1. **Cross-modal text concatenation** (`cross_modal_engine.py` Check 5): Concatenates text extracted from all modalities (image OCR + document text + audio transcription), re-scans through L2 regex to catch fragmentation attacks where injection is split across modalities. Only triggers when 2+ modalities contribute text AND individual scans missed it.

2. **Timing side-channel fix** (`main.py`, `barrier.py`): Replaced `==` with `hmac.compare_digest()` for API key comparison in `_is_authenticated()` and barrier `process()`. Prevents byte-by-byte timing attacks.

3. **Per-tenant TLI** (`layers/policy/__init__.py`): TLI changed from single global value to `dict[str, ThreatLevel]` keyed by tenant_id. Global stored under `"__global__"`. Methods `get_threat_level(tenant_id)`, `set_threat_level(level, tenant_id)`, `escalate_threat_level(tenant_id)`, `de_escalate_threat_level(tenant_id)`. Backward-compatible `threat_level` property delegates to global.

4. **Alpha channel steganalysis** (`layers/multimodal/steganalysis.py`): `analyze_alpha_channel()` method on `Steganalyzer` — chi-square LSB test on alpha channel + ASCII decode of LSBs. `SteganalysisResult` extended with `alpha_channel_suspicious`, `alpha_channel_score`, `alpha_hidden_text` fields. Integrated into `analyze()`.

5. **Multi-language injection detection** (`layers/innate/multilang_detector.py`): Scanner 8 in L2 innate pipeline. 54 regex patterns across 10 languages (Spanish, French, German, Portuguese, Italian, Russian, Chinese, Japanese, Korean, Arabic). Confidence 0.90 — language-specific attacks are deliberate. Integrated into `InnateDetectionLayer.scan()`.

6. **Expanded OCR layout** (`layers/multimodal/ocr_engine.py`): `_preprocess_for_ocr()` pipeline: scale normalization (target ~1000px), contrast enhancement (histogram stretching for low-contrast images), rotation correction (edge-based skew detection, 1-15 degree correction). Falls back to original image on failure.

7. **TLI auto-decay** (`layers/policy/__init__.py`): Automatic de-escalation timers per threat level: RED→ORANGE 300s, ORANGE→YELLOW 180s, YELLOW→BLUE 120s, BLUE→GREEN 60s. Checked lazily on `get_threat_level()`. Configurable via `AEGIS_TLI_AUTO_DECAY_ENABLED` env var (default: true).

## CHIMERA Cross-Modal Hardening (2026-03-17)

CHIMERA campaign detection improved from 65% (13/20) to 90% (18/20).

**Root cause**: Document scanning, audio scanning, and cross-modal correlation engine were implemented but NOT wired into the main.py HTTP request pipeline. Only image scanning was active.

**Fixes applied**:
1. **Pipeline integration** (`main.py`): Wired document scanning (`_multimodal.scan_documents()`), document text extraction (`_multimodal.extract_document_text()`), audio scanning (`_multimodal.scan_audio()`), and cross-modal correlation (`_cross_modal.correlate()`) into the multimodal section of the request handler. All modality scan results and extracted text flow into cross-modal correlation.
2. **Cross-modal decoy detection** (`cross_modal_engine.py` Check 6): Detects when text says "ignore the image/audio/document" while that modality is present — a pattern used to direct attention away from adversarial payloads. Confidence 0.85. Uses `_DECOY_PATTERN` regex + `_DECOY_MODALITY_MAP` for canonical modality resolution.
3. **Authority impersonation patterns** (`patterns.json` AI-001, AI-002, AI-003): Detects fake AEGIS patch notes/updates/policy changes claiming to disable filtering or set thresholds to 0.
4. **Config reconnaissance patterns** (`patterns.json` CR-001, CR-002, CR-003): Detects probing for AEGIS config parameters, detection threshold values, and config zeroing attacks (setting thresholds to 0.0).
5. **Volume anomaly blocking** (`cross_modal_engine.py`): Confidence raised from 0.70 to 0.85 to reach blocking threshold.
6. **Document text extraction** (`layers/multimodal/__init__.py`): Added `extract_document_text()` method to MultimodalPreprocessor for extracting visible+hidden text from documents for cross-modal correlation.

**Accepted misses** (1/20 in CI, 1/20 in production):
- xm_016: Benign probe ("What models do you support?") — correctly not flagged
- xm_018: OCR-dependent CJK injection — requires Tesseract + CJK language packs. **BLOCKED in production Docker image** (Tesseract installed with chi-sim/chi-tra/jpn/kor/ara/hin/rus packs). Skipped in CI without Tesseract binary. Production detection rate: 19/20 = 95%.

**Tesseract OCR in Docker** (2026-03-17): Dockerfile updated to install `tesseract-ocr` + 7 language packs (chi-sim, chi-tra, jpn, kor, ara, hin, rus) in both builder and runtime stages. `pytesseract>=0.3.10` added to both `requirements.txt` and `requirements-docker.txt`. OCR engine (`layers/multimodal/ocr_engine.py`) already had Tesseract integration — it activates automatically when pytesseract is importable and tesseract binary is on PATH.

**Event loop fix** (`tests/conftest.py`): Added `pytest_runtest_setup` hook to ensure a fresh event loop exists before each test. Fixed 121 pre-existing failures caused by `asyncio.run()` closing the event loop for subsequent tests using `asyncio.get_event_loop().run_until_complete()`.

## Concurrency Safety

Concurrency fixes applied to prevent race conditions under async/threaded access:

1. **`tenant_manager.py`**: `evict_expired()` uses `pop(k, None)` instead of `del` to prevent KeyError during concurrent eviction.
2. **`healing.py`**: `get_breaker()` uses `setdefault()` pattern — creates breaker, then atomically inserts only if key is still absent. Prevents duplicate breakers.
3. **`barrier.py`**: `_get_tenant_limiter()` uses same `setdefault()` pattern for tenant rate limiters.
4. **`barrier.py`**: Redis rate limiter uses atomic Lua script instead of two-pipeline ZREMRANGEBYSCORE+ZCARD then ZADD. Eliminates TOCTOU race where concurrent requests could both pass the count check.
5. **`identity.py`**: `AgentIdentityManager` has `_registry_lock` (asyncio.Lock) for future concurrent registration protection.
6. **`threat_vault.py`**: `search()` takes snapshots of `_indicators` and `_embeddings` under `_lock`, then processes outside the lock. Prevents index/list mismatch during concurrent `add()`.
7. **`multi_turn.py`**: `_request_count` increment moved inside `with self._lock:` to prevent lost updates. Uses `threading.Lock` (not asyncio.Lock) because existing tests use it synchronously.

---

## Subsystem Reference

Detailed implementation docs for each subsystem. These describe internal behavior of existing code — read the code directly for the most current state.

### Multi-turn Sequence Analysis

`layers/adaptive/multi_turn.py`: Per-session sliding window (10 turns, 30min TTL) with four detection strategies: (1) **escalation trajectory** — monotonically increasing boundary scores over 3+ turns, (2) **boundary testing** — sub-threshold probing with boundary keywords at three intensity levels, (3) **rapid-fire after block** — 3+ requests within 30s after any layer blocks a request (automated retry/fuzzing detection), (4) **topic drift with threats** — high topic drift (>0.4) combined with 2+ turns having innate_max_confidence > 0.3. Multi-turn signals are **DANGER** (probabilistic), not PAMPs (definitive). Threat category: `ThreatCategory.MULTI_TURN_ESCALATION` ("AEGIS.MULTI_TURN"). SessionTurn records enriched with innate report data (innate_max_confidence, threat_categories, was_blocked) on the latest turn. `record_block()` called from main.py on every innate/adaptive/policy block to feed rapid-fire detection. Audit logs include multi-turn details (turn_count, escalation_score, pattern_type, triggered_strategies) in the adaptive JSONB column.

### Streaming Interception

`layers/output/streaming.py`: Production-grade `StreamingInterceptor` wraps upstream SSE streams with real-time validation:
- **Hold buffer** (32 tokens): Tokens are delayed before transmission, giving the PII detector time to scan. PII found in the hold buffer is redacted before the client sees it. PII detected after tokens already sent logs a warning (cannot be recalled).
- **Sliding window** (128 tokens, 50% overlap): At each window boundary, runs PII redaction, toxicity classification, and leakage detection. Each token is evaluated twice due to overlap.
- **Real-time PII scrubbing**: PII is transparently redacted in the stream — clients receive `[SSN]`, `[PERSON_NAME]`, etc. without knowing interception occurred. PII does NOT terminate the stream (redact and continue).
- **Toxicity/leakage termination**: Toxic content or system prompt echo immediately terminates the stream with a safety SSE event and publishes `threat_detected` to the event bus with `stream_interrupted=True`.
- **Cumulative threat tracking**: Rolling average of the last 5 window threat scores. If the average exceeds 0.6 for 3+ consecutive windows, terminates the stream even if no single window crossed the threshold. Catches "slow and low" drip attacks.
- **Stream termination SSE format**: `data: {"choices":[{"delta":{"content":"[AEGIS: Response interrupted...]"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n`
- **Metrics**: `aegis_stream_interruptions_total` (Counter, labels: reason [pii/toxicity/leakage/cumulative]), `aegis_stream_windows_evaluated_total` (Counter), `aegis_stream_tokens_processed_total` (Counter)

### Five-Stage Output Cascade

`layers/output/`: All five stages run sequentially; cascade escalation from earlier stages tightens thresholds for later stages.
- **Stage 1 — PII & Secrets Redaction** (`pii_redactor.py`): Presidio NER + regex fallback. 8 secret types: AWS access keys (`AKIA[0-9A-Z]{16}`), AWS secret keys (40-char base64 after known prefixes), OpenAI keys (`sk-*`), Anthropic keys (`sk-ant-*`), GitHub tokens (`ghp_/gho_/ghu_/ghs_/ghr_`), connection strings (`postgresql://`, `mysql://`, `mongodb://`, `redis://` with credentials), private key headers (`-----BEGIN (RSA|EC|OPENSSH)?PRIVATE KEY-----`), generic high-entropy secrets (`key=/token=/secret=` + 40+ chars). Each secret type has a distinct `entity_type` label for PII_REDACTIONS counter. Custom Presidio `PatternRecognizer` instances registered when Presidio is available; same patterns in `_SECRET_PATTERNS` list for regex fallback.
- **Stage 2 — Toxicity Classification** (`toxicity.py`): Keyword regex + weighted n-gram matching. N-gram rules require a trigger phrase AND a context word in the same text (e.g., "how to make" + "bomb" → violence, but "how to make" + "cake" → clean). Category-specific base thresholds: violence=0.55, self_harm=0.50, hate_speech/sexual=0.65, illegal=0.60, regulated_advice=0.75. Three sensitivity levels: HIGH (thresholds × 0.75), MEDIUM (× 1.0), LOW (× 1.25). `LlamaGuardClassifier` stub: async interface returning clean result with `classifier_type="llama_guard_not_loaded"`. Factory: `create_toxicity_classifier()` returns `LlamaGuardClassifier` when `AEGIS_LLAMA_GUARD_MODEL` env var is set.
- **Stage 3 — Hallucination Detection** (`hallucination.py`): N-gram source coverage for RAG responses. Detects RAG context via system message markers ("Context:", "Retrieved:", "Sources:") or `context.metadata["rag_sources"]`. Computes 4-gram overlap per sentence; sentences with zero overlap = unsupported claims. Flags as hallucination if `source_coverage < 0.3` AND `response > 100 chars`. Non-RAG responses return clean (`source_coverage=1.0`). **Production upgrade**: NLI model (`cross-encoder/nli-deberta-v3-base`) for entailment classification.
- **Stage 4 — Data Leakage Prevention** (`leakage.py`): N-gram overlap for system prompt echo, secret pattern detection, verbatim protected content check.
- **Stage 5 — Schema Compliance** (`schema_validator.py`): JSON parse attempt; if JSON, validates against expected schema (missing/extra fields, type mismatches). Detects injected fields with suspicious names (`system`, `prompt`, `instructions`, `role`, `override`, `exec`, etc.). Oversized payload detection: >10× expected field count. Non-JSON returns clean.
- **Cascade escalation**: PII/secret detection in Stage 1 sets `escalated=True` → Stage 2 toxicity threshold reduced by 20% (`× 0.80`), Stage 3 hallucination coverage threshold raised by 0.20. Elevated `scrutiny_level` (from policy engine) also triggers escalation.
- **Two validate methods**: `validate()` is synchronous (3 stages: PII, toxicity, leakage) — used by streaming chunks and backward-compatible code. `validate_full()` is async (all 5 stages) — used by main.py non-streaming path. `validate_chunk()` delegates to `validate()`.

### STIX/TAXII Threat Intelligence

`services/threat_intel.py`: `ThreatIntelManager` ingests STIX 2.1 bundles (indicator objects) into the Threat Vault. Each indicator's `x_aegis_pattern_text` is embedded via MiniLM (or deterministic hash fallback) and stored as a `ThreatIndicator` with `source=STIX_FEED`, `confirmed=False`. Deduplication by SHA-256 `payload_hash` — re-ingesting the same indicator increments `hit_count` instead of creating a duplicate. STIX export adds differential privacy noise to embeddings (Gaussian mechanism: `sigma = sensitivity / epsilon`, default ε=3.0, re-normalize to unit length). AEGIS extensions on STIX indicators: `x_aegis_pattern_text` (raw attack text), `x_aegis_embedding` (384-dim, DP-noised on export), `x_aegis_mitre_id` (ATLAS tactic), `x_aegis_affected_models`, `x_aegis_mitigation`. Category resolution: MITRE ID → `_MITRE_TO_CATEGORY` dict takes precedence; falls back to `_LABEL_TO_CATEGORY` from STIX labels. Seed feed (`data/stix_feeds/seed_threat_feed.json`, 13 indicators) auto-ingested on startup from all `*.json` files in `data/stix_feeds/`. API: POST `/v1/threat-intel/ingest` (ingest bundle), GET `/v1/threat-intel/export?since=<ISO>` (export with DP noise), GET `/v1/threat-intel/stats` (feed statistics). All endpoints require authentication.

### PostgreSQL Schema (db/init.sql)

Auto-applied on first PostgreSQL container startup. Extensions: `uuid-ossp`, `pg_trgm`.

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `tenants` | Multi-tenant config | name, api_key_hash, rate_limit_rpm, block/alert thresholds, allowed_models, policy_tier, custom_policy (JSONB) |
| `audit_log` | Every request/response cycle | request_id, tenant_id, session_id, prompt_hash (SHA-256, never raw), innate/adaptive/output results, final_action, latencies, circuit_state, threat_level |
| `threat_indicators` | Persistent vault metadata (L4) | mitre_tactic, confidence, prompt_hash, status (acute/persistent/dormant), hit_count, embedding_ref |
| `signatures` | Clonal selection output (L2/L4) | pattern_type (regex/yara/embedding), pattern_text, source_indicator_id FK, affinity_score, TPR/FPR, status (active/deprecated/candidate) |
| `circuit_breaker_events` | L7 state transitions | endpoint, previous_state, new_state, trigger_reason, failure_rate |

All tables have `created_at`/`updated_at` with auto-update triggers. Default `dev` tenant inserted on init. Indexes on timestamp, tenant, session, blocked requests, MITRE tactic, and pattern trigrams (`gin_trgm_ops`).

### Redis Integration

Redis is optional — AEGIS degrades gracefully to in-memory implementations when Redis is unavailable. AEGIS must NEVER crash because Redis is down.

**Initialization**: `services/redis_client.py` provides a singleton async connection pool. `init_redis()` called in lifespan; `get_redis()` returns the client or `None`. `redis_health()` returns connectivity status for `/health`.

**Rate Limiting** (`layers/barrier.py`): When a Redis client is available, `BarrierLayer` uses `RedisSlidingWindowRateLimiter` (Redis sorted sets: ZREMRANGEBYSCORE + ZCARD + ZADD). Key format: `aegis:ratelimit:{api_key_hash}` with TTL = window + 10s. On Redis failure mid-request, falls back to in-memory `SlidingWindowRateLimiter`.

**Event Bus** (`services/event_bus.py`): Cytokine signaling network between layers. Two implementations:
- `InMemoryEventBus`: asyncio callbacks, zero dependencies, used in tests and when Redis unavailable
- `RedisEventBus`: Redis Streams (XADD/XREAD with consumer groups), durable, ordered, multi-consumer

Factory: `create_event_bus(redis_client)` picks the right implementation.

### Event Bus Channels

| Channel | Publisher | Subscribers | Trigger |
|---------|-----------|-------------|---------|
| `threat_detected` | L3 Adaptive | L6 Policy (escalate TLI) | Novel attack blocked |
| `antibody_generated` | L3 Adaptive | (informational) | New antibody stored in vault |
| `circuit_breaker` | L7 Healing | L6 Policy (escalate TLI) | Breaker state transition |
| `policy_escalation` | L6 Policy | (informational) | TLI level change |
| `audit_event` | Cross-layer | (future: audit sink) | Any auditable action |

Subscriptions are wired in `main.py:_wire_event_bus_subscriptions()`. Event bus notifications are advisory — each layer maintains independent state.

### PostgreSQL Integration

PostgreSQL is optional — AEGIS degrades gracefully when PostgreSQL is unavailable. AEGIS must NEVER crash because PostgreSQL is down.

**Database Engine** (`services/db.py`): Async SQLAlchemy singleton with asyncpg driver. `init_db()` called in lifespan; `get_session_factory()` returns an `async_sessionmaker` or `None`. `db_health()` returns connectivity status for `/health`. Pool: size=10, max_overflow=5, pool_pre_ping=True. Handles `postgresql://` to `postgresql+asyncpg://` scheme conversion automatically.

**Audit Logger** (`services/audit_logger.py`): Fire-and-forget PostgreSQL audit writes via `asyncio.create_task()`. CRITICAL: prompt content is NEVER stored — only SHA-256 hash via `compute_prompt_hash()`. Features:
- `PgAuditRecord` dataclass mirrors `audit_log` table columns exactly
- `PgAuditLogger.log()` fires background task, buffers to `deque(maxlen=10000)` on DB failure
- Background flush loop drains buffer every 30s when DB recovers
- `log_circuit_breaker_event()` writes L7 state transitions to `circuit_breaker_events` table
- CRUD for `threat_indicators` and `signatures` tables (used by vault/signature dual-write)
- Module singleton: `get_pg_audit_logger()` / `reset_pg_audit_logger()`

**Dual-Write Pattern** (L4 Threat Vault + L2 Signatures): FAISS for sub-ms in-memory search, PostgreSQL for persistence. Operations complete in FAISS first (inside asyncio.Lock), then fire-and-forget PG write outside the lock. On startup, `threat_vault.load_from_postgres()` syncs lifecycle status and hit counts from PostgreSQL back into in-memory state.

- `threat_vault.py`: `add()`, `record_hit()`, `promote()`, `demote()`, `lifecycle_update()` all dual-write
- `signatures.py`: `add()`, `deprecate()`, `deprecation_check()` all dual-write
- Both accept `session_factory` parameter; when `None`, PG writes are silently skipped

**Audit Wiring** (`main.py`): `_fire_pg_audit()` helper builds `PgAuditRecord` from layer results and fires after every request — innate block, policy block, adaptive block, output block, and allow paths. Circuit breaker state transitions also logged via `_publish_circuit_breaker_event()`.

### Multi-Tenant Architecture

AEGIS supports per-tenant configuration via the `tenants` PostgreSQL table. Each tenant gets custom rate limits, detection thresholds, allowed model lists, and policy tiers. Multi-tenant is optional — without PostgreSQL or TenantManager, AEGIS falls back to environment-variable defaults.

**Tenant Manager** (`services/tenant_manager.py`): Loads tenant configs from PostgreSQL, caches in-memory keyed by `api_key_hash` (SHA-256). Features:
- `TenantConfig` dataclass: tenant_id, name, api_key_hash, rate_limit_rpm/burst, max_tokens_per_request, block/alert thresholds, allowed_models, policy_tier, custom_policy
- `load_all()`: Bulk-load all enabled tenants on startup
- `resolve_tenant(api_key_hash)`: Cache lookup (5-minute TTL) → DB lookup on miss → expired cache as fallback → None
- Background refresh every 60 seconds via `start_refresh()`
- Graceful degradation: if DB is down, cached tenants remain usable; if no cache, returns None and layers use env-var defaults

**Barrier Layer** (`layers/barrier.py`): Accepts optional `tenant_manager` parameter. When set: resolves tenant from API key hash, applies per-tenant rate limits, rejects unauthorized models (403), applies per-tenant `max_tokens_per_request`, binds `tenant_id` to RequestContext.

**Policy Engine** (`layers/policy/__init__.py`): Accepts optional `tenant_manager` parameter. Policy tier modifiers: `strict` (×0.85), `permissive` (×1.10), `standard` (unchanged), `custom` (JSONB `blocked_categories`).

**Configuration Precedence**: Manually registered TenantPolicy > TenantManager (from DB) > Global defaults (env vars)

### OPA Policy Engine

AEGIS supports Open Policy Agent (OPA) as an alternative policy backend. When `AEGIS_POLICY_BACKEND=opa`, the `OPAPolicyEngine` sends policy decisions to an OPA server via HTTP and falls back to the Python `PolicyEngine` on any failure (timeout, connection error, invalid response).

**Rego Policy** (`data/opa_policies/aegis.rego`): Three-tier evaluation mirroring the Python engine:
- **Tier 1 (Global)**: RED fail-closed (block all), hard block on MITRE ATLAS categories at ≥0.85 confidence
- **Tier 2 (Tenant)**: Custom blocked categories, threshold with policy_tier modifier
- **Tier 3 (Adaptive)**: TLI-adjusted fused score thresholds: GREEN=0.85, BLUE=0.765, YELLOW=0.6375, ORANGE=0.51

**Fallback**: On any OPA failure (50ms timeout, connection error, invalid response, missing httpx), the Python `PolicyEngine` handles the request. Fallback decisions tagged with `opa:FALLBACK`.

### Supply Chain Verification (Enhanced)

Four-stage model verification pipeline with Sigstore integration and aggregate risk scoring.

**Stage 1 — Cryptographic Integrity** (`integrity.py` + `sigstore_verifier.py`): SHA-256 hash manifest, optional Sigstore signature verification against Rekor transparency log.

**Stage 2 — Serialization Safety** (`serialization.py`): Format risk scoring (SafeTensors=0.0, ONNX=0.1, pickle/PyTorch=0.8), pickle-in-ZIP detection, structural validation.

**Stage 4 — Behavioral Probing** (`behavioral_probe.py`): 10 jailbreak resistance probes, 4 sleeper agent probes with conditioned/unconditioned comparison, Jaccard divergence scoring.

**Orchestrator** (`__init__.py`): Aggregate `risk_score` = weighted average (integrity=0.3, serialization=0.3, dependency=0.2, behavioral=0.2). Critical findings (risk ≥ 0.9) force `overall_status="fail"`.

### Multi-Agent Security (MHC Identity Verification)

**Agent Identity** (`layers/agent_security/identity.py`): In-memory registry with HMAC-SHA256 JWT signing. Trust constants: CLEAN_INTERACTION=+0.02, POLICY_VIOLATION=-0.1, CONFIRMED_ATTACK=-0.5, INJECTION_DETECTED=-0.2. Trust decay: 0.1/day after 24h inactive.

**Tool Authorization** (`layers/agent_security/authorization.py`): Trust requirements: code_exec≥0.7, mcp_server≥0.8, db_query≥0.6, file_read≥0.4, file_write≥0.6, web_search≥0.3, api_call≥0.5. Capability escalation always blocked.

**Message Validation** (`layers/agent_security/message_validator.py`): Delegates injection scanning to L2 innate layer. Per-agent quarantine sets.

**Pipeline Integration** (`main.py`): `X-AEGIS-Agent-ID` header → barrier verifies → trust bound to RequestContext. Scrutiny multiplier: trust < 0.3 → ×2.0, trust < 0.5 → ×1.5.

### Compliance Dashboard

Cross-framework regulatory compliance reporting. Maps AEGIS controls to six frameworks simultaneously (NIST AI RMF, ISO 42001, EU AI Act, CMMC 2.0, SOC 2, FedRAMP). 10 AEGIS capabilities × 6 frameworks = 68+ control mappings. Evidence collection from layer state, audit logs, config, and benchmark data. Report generation with per-framework compliance scores.

### Federated Threat Intelligence (L8)

Privacy-preserving federated learning network. Differential privacy (Gaussian mechanism, ε=3.0, δ=1e-5). Local trainer: logistic regression on TF-IDF character n-gram features. FedAvg aggregation with weight poisoning defense (L2 norm clipping). Indicator sharing with DP-noised embeddings — raw text never shared, only content hashes.

### Enterprise Hardening and Deployment

**Configuration Validator** (`services/deployment/config_validator.py`): 13 startup checks (errors=fatal, warnings=log). Called in lifespan BEFORE `_init_layers()`.

**Vault Backup/Restore** (`services/deployment/backup_restore.py`): JSON export/import of indicators + signatures. Embeddings NOT included (rebuilt on restore). Backup dir: `AEGIS_BACKUP_DIR` env var.

**Deep Health Monitor** (`services/deployment/health_monitor.py`): Six component checks (Redis, PostgreSQL, FAISS, ML Model, Disk Space, Circuit Breaker). Background monitor every 60s. Auto-recovery for Redis.

### Observability

30+ Prometheus metrics via `GET /metrics`. Key metrics: `aegis_requests_total`, `aegis_request_latency_seconds`, `aegis_threats_detected_total`, `aegis_blocks_total`, `aegis_threat_level`, `aegis_circuit_breaker_state`, `aegis_pii_redactions_total`, `aegis_vault_size`, `aegis_stream_interruptions_total`, `aegis_distillation_signals_total`. Health endpoint reports per-component status with overall derivation (any unhealthy → 503).

### Distillation Defense

Cross-session model extraction detection. 5 strategies: Query Diversity (0.25 weight), Boundary Mapping (0.30), Reasoning Coercion (0.20), Information Gain (0.10), Complexity Escalation (0.15). Action thresholds: <0.30 none, 0.30-0.60 monitor, 0.60-0.85 rate_limit, >=0.85 block (403).

**ReasoningTraceSanitizer** (`layers/output/reasoning_sanitizer.py`): Detects chain-of-thought, governance disclosure, decision process traces. Three modes: monitor, redact, summarize.

### Stress Test Infrastructure

Located in `tests/stress/`. **Concurrency tests** (18 tests): CI-safe, validates all concurrency fixes. **Full stress tests** (5 tests): Require `AEGIS_STRESS_FULL=1`, spawn real servers.

### Chaos Testing and Resilience

**ChaosRunner** (`tests/stress/chaos_runner.py`): Monkey-patches live objects to simulate failures with auto-restore. **Chaos regression tests** (44 tests): 11 test classes validating every degradation path.

### Resilience Guarantees

| Component Down | Behavior |
|---------------|----------|
| **Redis** | In-memory rate limiting + InMemoryEventBus |
| **PostgreSQL** | Audit buffers to deque (10k max), vault FAISS-only, tenant returns None |
| **FAISS index** | Brute-force numpy fallback for vector search |
| **All backing services** | Core L1-L7 pipeline operates in-memory only |

**Invariants**: AEGIS never returns HTTP 500 due to backing service failure. TLI RED blocks ALL requests (fail-closed). Audit buffer never exceeds 10,000 entries. Health reports "degraded" (not "unhealthy") when backing services are down but core layers are up.

### Degradation Behavior Matrix

| Component | Healthy Status | Degraded Behavior | Recovery |
|-----------|---------------|-------------------|----------|
| Redis | Redis rate limiter + Redis Streams event bus | In-memory SlidingWindowRateLimiter + InMemoryEventBus | New BarrierLayer with Redis client |
| PostgreSQL | Persistent audit, dual-write vault, tenant DB lookups | Deque buffer (10k), FAISS-only vault, cached/None tenants | Flush buffer on reconnect |
| FAISS | HNSW sub-ms ANN search | Brute-force numpy cosine similarity | Index rebuilt on next add() |
| Circuit Breaker | CLOSED, all requests to primary | OPEN → fallback routing, HALF_OPEN → probe requests | Auto-recovery via probe success |
| TLI | GREEN, standard thresholds | BLUE-ORANGE: progressively lower thresholds. RED: fail-closed block all | Auto-decay or manual de-escalation |

### Live Demo

`demo/live_demo.py` runs 6 scenarios against a live AEGIS instance + Ollama backend. Config in `demo/demo_config.env`. Key details:
- PYTHONPATH must point to parent of repo root (so `from aegis.config import ...` resolves)
- uvicorn target is `aegis.main:app` (not `main:app`)
- Startup timeout is 120s to allow DeBERTa model loading (~30s first run)
- 3-second delay between Scenario 4 and 5 allows async clonal selection to complete
- Scenarios: (1) direct injection → L2 regex, (2) encoded injection → L2 regex, (3) PII in response → L5 output, (4) paraphrased injection → L3 DeBERTa, (5) variant of S4 → L2 clonal selection pattern, (6) circuit breaker trip → L7 healing

### Docker Details

docker-entrypoint.sh: Waits up to 30s each for Redis and PostgreSQL TCP connectivity. Runs `alembic upgrade head` if alembic.ini exists. Prints startup info (upstream URL, model status, backing services). Parses REDIS_URL and DATABASE_URL environment variables to extract host:port (handles `postgresql+asyncpg://` scheme).

docker-compose.yml: Four services on bridge network. AEGIS gets 4G memory limit, `extra_hosts` for `host.docker.internal` (Ollama access). Redis uses 256MB maxmemory with LRU eviction, appendonly persistence. PostgreSQL auto-runs `db/init.sql` via `/docker-entrypoint-initdb.d/`. OPA (`openpolicyagent/opa:latest-static`) serves Rego policies on port 8181 (read-only volume mount from `data/opa_policies/`). All services have health checks.

requirements-docker.txt: Additional deps for containerized deployment: `redis[hiredis]`, `asyncpg`, `psycopg2-binary`, `sqlalchemy[asyncio]`, `alembic`, `pytesseract`. Install after requirements.txt.
