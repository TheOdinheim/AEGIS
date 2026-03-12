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
│   │   ├── regex_engine.py          # Scanner 1: 170+ patterns, 8-step Unicode normalization (200+ homoglyphs)
│   │   ├── blocklist.py             # Scanner 2: Bloom filter with hash confirmation
│   │   ├── token_guard.py           # Scanner 4: Token counting and distribution analysis
│   │   ├── pii_regex.py             # Scanner 5: Fast PII regex (SSN, CC, email, phone)
│   │   └── canary_verifier.py       # Scanner 6: HMAC-SHA256 canary token (NK cell analog)
│   ├── adaptive/
│   │   ├── __init__.py              # AdaptiveAnalysisLayer with antibody callback
│   │   ├── injection_classifier.py  # DeBERTa-v3 prompt injection (ONNX, ~20ms)
│   │   ├── semantic_search.py       # Embedding + FAISS similarity search
│   │   ├── behavioral.py            # Baseline anomaly detection (PSI, KS tests)
│   │   └── multi_turn.py            # Multi-turn: escalation trajectory, boundary testing, rapid-fire, topic drift
│   ├── memory/
│   │   ├── threat_vault.py          # FAISS HNSW index, 3-phase lifecycle
│   │   └── signatures.py            # Clonal selection generator + SignatureStore
│   ├── output/
│   │   ├── __init__.py              # OutputValidationLayer: 5-stage cascade orchestrator
│   │   ├── pii_redactor.py          # Stage 1: Presidio PII + 8 secret types (AWS, OpenAI, Anthropic, GitHub, etc.)
│   │   ├── toxicity.py              # Stage 2: Keyword + n-gram classifier, LlamaGuard stub, sensitivity levels
│   │   ├── hallucination.py         # Stage 3: N-gram source coverage for RAG responses
│   │   ├── leakage.py               # Stage 4: System prompt echo detection
│   │   ├── schema_validator.py      # Stage 5: JSON schema validation, injected field detection
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
│   └── agent_security/
│       ├── __init__.py              # AgentSecurityLayer orchestrator (MHC identity verification)
│       ├── identity.py              # AgentIdentityManager: JWT signing, trust mechanics, decay
│       ├── authorization.py         # AgentAuthorizationEngine: least-privilege, scope, escalation blocking
│       └── message_validator.py     # AgentMessageValidator: injection scanning, quarantine
├── services/
│   ├── __init__.py
│   ├── redis_client.py              # Async Redis singleton, health check, graceful degradation
│   ├── event_bus.py                 # EventBus: InMemoryEventBus / RedisEventBus (Redis Streams)
│   ├── db.py                        # Async SQLAlchemy + asyncpg singleton, health check, graceful degradation
│   ├── audit_logger.py              # PostgreSQL audit logger, fire-and-forget, in-memory buffer
│   ├── tenant_manager.py            # Multi-tenant config: cache, resolve, refresh, graceful degradation
│   ├── threat_intel.py              # STIX/TAXII ingestion, export with DP noise, dedup
│   ├── compliance/
│   │   ├── __init__.py              # ComplianceEngine orchestrator
│   │   ├── framework_mappings.py    # 6 frameworks × 10 capabilities cross-mapping (68+ controls)
│   │   ├── evidence_collector.py    # Automated evidence from AEGIS telemetry
│   │   └── report_generator.py      # Cross-framework compliance report generation
│   ├── federated/
│   │   ├── __init__.py              # FederatedIntelligenceManager orchestrator (L8 herd immunity)
│   │   ├── privacy.py               # DifferentialPrivacyEngine: Gaussian mechanism, budget tracking
│   │   ├── local_trainer.py         # LocalTrainer: logistic regression on TF-IDF features
│   │   ├── aggregation.py           # FederatedAggregator: FedAvg with L2 norm clipping
│   │   └── indicator_sharing.py     # IndicatorSharingService: DP-noised embedding exchange
│   └── deployment/
│       ├── __init__.py              # Package exports
│       ├── config_validator.py      # ConfigValidator: 13 startup checks (errors=fatal, warnings=log)
│       ├── backup_restore.py        # VaultBackupManager: JSON backup/restore of vault + signatures
│       └── health_monitor.py        # DeepHealthMonitor: 6-component deep checks + auto-recovery
├── middleware/
│   ├── request_enrichment.py        # Build RequestContext
│   └── metrics.py                   # Prometheus metrics collection
├── data/
│   ├── patterns.json                # ~110 regex patterns (MITRE ATLAS tagged)
│   ├── blocklist.txt                # Known malicious payloads
│   ├── seed_threats.json            # Initial threat vault embeddings
│   ├── known_cves.json              # CVE database for supply chain audit
│   ├── benign_prompts.json          # 55 benign prompts for clonal selection validation
│   ├── benchmark_benign.json        # 500 business prompts across 10 industries
│   ├── benchmark_attacks.json       # 110 labeled attacks across 10 categories
│   ├── benchmark_results.json       # Latest benchmark output
│   ├── stix_feeds/
│   │   └── seed_threat_feed.json    # 13 AI-specific STIX 2.1 indicators (seed data)
│   └── opa_policies/
│       ├── aegis.rego               # OPA Rego policy: 3-tier decision (global/tenant/adaptive)
│       └── aegis_test.rego          # OPA unit tests (opa test data/opa_policies/ -v)
├── demo/
│   ├── live_demo.py                 # Interactive 6-scenario demo (requires Ollama)
│   └── demo_config.env              # Demo environment configuration
├── tests/
│   ├── conftest.py                  # Sets AEGIS_SKIP_MODEL_LOAD=true for all tests
│   ├── test_innate.py
│   ├── test_adaptive.py
│   ├── test_output.py
│   ├── test_healing.py
│   ├── test_attack_battery.py       # 20-attack simulation + endpoint tests
│   ├── test_vault_lifecycle.py
│   ├── test_signatures.py
│   ├── test_supply_chain.py
│   ├── test_supply_chain_enhanced.py  # Sigstore, risk scoring, sleeper probes, pickle-in-ZIP (82 tests)
│   ├── test_agent_security.py         # Agent identity, authorization, message validation, endpoints (90 tests)
│   ├── test_compliance.py            # Compliance: mappings, evidence, reports, scoring, endpoints (76 tests)
│   ├── test_federated.py            # Federated: DP noise, FedAvg, poisoning defense, indicators, endpoints (106 tests)
│   ├── test_enterprise.py           # Enterprise: config validation, backup/restore, deep health, admin endpoints (81 tests)
│   ├── test_cleanup_items.py
│   ├── test_threat_model_fixes.py
│   ├── test_redis_integration.py    # Redis rate limiter, event bus, graceful degradation
│   ├── test_postgres_integration.py # PostgreSQL audit, dual-write, graceful degradation (62 tests)
│   ├── test_tenant.py               # Multi-tenant: resolution, rate limits, thresholds, policy tiers (58 tests)
│   ├── test_multi_turn.py            # Multi-turn: escalation, rapid-fire, DCA signals, innate integration (43 tests)
│   ├── test_metrics.py              # Observability: metric registration, /health, /metrics, instrumentation (106 tests)
│   ├── test_threat_intel.py         # STIX/TAXII: ingestion, dedup, export, DP noise, API endpoints (65 tests)
│   ├── test_output_enhanced.py      # Enhanced output: secrets, n-gram toxicity, hallucination, schema, cascade (76 tests)
│   ├── test_opa_policy.py           # OPA policy: input/response, fallback, health, config, Rego logic (56 tests)
│   ├── test_streaming_intercept.py  # Streaming: hold buffer, PII scrub, toxicity/leakage termination, cumulative threat (49 tests)
│   ├── test_red_team.py             # Red team framework: attack generator, evasion engine, learning, reports (57 tests)
│   ├── test_red_team_regression.py  # Red team regression: homoglyphs, combining marks, word SSN, base64 PII, padding (71 tests)
│   ├── test_normalization_hardened.py # Phase 4: BIDI, leetspeak, 200+ confusables, recursive B64, combined evasion (42 tests)
│   ├── test_pii_hardened.py          # Phase 4: code-context, URL-embedded, reversed, word-spelled PII + indirect injection (41 tests)
│   ├── test_infrastructure_security.py # Phase 5: infrastructure security (auth, rate limit, tenant, DoS, audit, federated, supply chain) (60 tests)
│   ├── test_adaptive_red_team.py      # Phase 6: adaptive meta-learner (data models, blind spots, predictor, co-evolution, fingerprints, hardening, pipeline) (63 tests)
│   ├── test_hardening_regression.py   # Phase 6: hardening regression (char mappings, regex patterns, PII patterns, convergence, plan integration) (32 tests)
│   ├── stress/
│   │   ├── __init__.py              # Stress test package
│   │   ├── mock_upstream.py         # FastAPI mock OpenAI API (configurable latency/errors/toxic/PII)
│   │   ├── load_generator.py        # Concurrent HTTP load generator with percentile metrics
│   │   ├── metrics_collector.py     # Prometheus /metrics scraper
│   │   ├── report.py               # Human-readable + JSON report generation
│   │   ├── stress_runner.py         # Subprocess lifecycle manager (mock upstream + AEGIS)
│   │   ├── test_stress.py           # Full stress tests (require AEGIS_STRESS_FULL=1, 5 tests)
│   │   ├── test_concurrency.py      # CI-safe concurrency tests (18 tests)
│   │   ├── chaos_runner.py          # ChaosRunner: monkey-patch component failures with auto-restore
│   │   └── test_chaos_regression.py # CI-safe chaos/resilience tests (44 tests)
│   └── benchmark/
│       ├── generate_corpus.py
│       ├── run_benchmark.py
│       └── test_benchmark.py
├── red_team/
│   ├── __init__.py                  # Attack, EvasionResult, EvasionReport, LearningReport data models
│   ├── attack_generator.py          # AdversarialAttackGenerator: 120+ attacks targeting L2/L3/L5/multi-layer
│   ├── evasion_engine.py            # EvasionEngine: runs attacks against live AEGIS, classifies results
│   ├── campaign_runner.py           # CampaignRunner: orchestrates generate → test → learn → report pipeline
│   ├── learning_validator.py        # LearningValidator: measures antibody generation and generalization
│   ├── report.py                    # RedTeamReport: JSON + markdown assessment reports
│   ├── apt_campaigns.py             # 6 APT campaigns: PHANTOM NEEDLE, SILENT SIPHON, SLOW BURN, HYDRA, GHOST PROTOCOL, CASCADING FAILURE
│   ├── run_red_team.py              # Entry point: python3 -m red_team.run_red_team
│   ├── extended_campaigns.py        # Phase 4: Campaigns 7-12 (SHAPESHIFTER, BABEL TOWER, THOUSAND CUTS, INSIDE JOB, MIRROR MIRROR, FULL SPECTRUM)
│   ├── infrastructure/              # Phase 5: Infrastructure security attack engine
│   │   ├── __init__.py              # AttackResult, InfrastructureAssessment dataclasses
│   │   ├── auth_attacks.py          # AuthAttacker: timing, format inference, 15 bypass techniques
│   │   ├── rate_limit_bypass.py     # RateLimitBypass: session rotation, header injection, quota exhaustion
│   │   ├── tenant_isolation.py      # TenantIsolationTester: cross-tenant leakage, privilege escalation, ID injection
│   │   ├── backing_service_attacks.py # BackingServiceAttacker: Redis/PostgreSQL/dependency trust analysis
│   │   ├── denial_of_service.py     # DenialOfServiceTester: circuit breaker manipulation, TLI, session flooding
│   │   ├── audit_integrity.py       # AuditIntegrityTester: log injection, completeness, evidence destruction
│   │   ├── federated_poisoning.py   # FederatedPoisoningAttacker: gradient/indicator poisoning, budget exhaustion
│   │   ├── supply_chain_self.py     # SelfSupplyChainTester: model tampering, pattern integrity, dependency audit
│   │   └── run_infrastructure_attacks.py # Orchestrator: runs all 8 modules, generates report
│   ├── adaptive/                    # Phase 6: Adaptive adversarial meta-learner
│   │   ├── __init__.py              # Data models: BlindSpotReport, PredictionReport, CoEvolutionReport, etc.
│   │   ├── blind_spot_detector.py   # BlindSpotDetector: systematic weakness analysis across Phases 1-5
│   │   ├── attack_predictor.py      # AttackPredictor: logistic regression on 12 features, evasion prediction
│   │   ├── co_evolution.py          # CoEvolutionEngine: 10-round attack/defense arms race simulation
│   │   ├── evasion_fingerprinter.py # EvasionFingerprinter: 12 root cause categories, bypass method ID
│   │   ├── hardening_generator.py   # HardeningGenerator: auto-generates detection rules from evasions
│   │   └── run_adaptive.py          # Orchestrator: 5-step pipeline, CLI, report generation
│   └── data/
│       ├── adversarial_corpus.json  # 120 pre-generated attacks (50 L2 + 30 L3 + 20 L5 + 20 multi-layer)
│       ├── campaign_results_initial.json  # Initial APT campaign results (before fixes)
│       ├── campaign_results_final.json    # Final APT campaign results (after fixes)
│       ├── detection_gaps.json            # 10 identified gaps (6 fixed, 4 open)
│       ├── red_team_assessment.md         # Full assessment report (Phase 2)
│       ├── FINAL_RED_TEAM_ASSESSMENT.md  # Final battle-tested assessment (Phase 4)
│       ├── final_metrics.json            # Machine-readable final metrics
│       ├── adaptive_assessment.json       # Phase 6 adaptive assessment results
│       ├── ADAPTIVE_RED_TEAM_ASSESSMENT.md # Phase 6 assessment report
│       ├── COMPREHENSIVE_RED_TEAM_REPORT.md # Full 6-phase comprehensive report
│       └── comprehensive_metrics.json     # Machine-readable comprehensive metrics (all phases)
├── docs/
│   ├── AEGIS_SELF_THREAT_MODEL.md   # 23 threats (T1-T23), pen test recommendations
│   └── DEPLOYMENT.md               # Production deployment guide, env vars, scaling, troubleshooting
├── requirements.txt
├── requirements-docker.txt          # Extra deps for Docker: redis, asyncpg, sqlalchemy, alembic
├── Dockerfile                       # Multi-stage: builder (deps+models) → runtime (slim)
├── docker-compose.yml               # Dev stack: AEGIS + Redis + PostgreSQL
├── docker-entrypoint.sh             # Waits for Redis/Postgres, runs alembic, prints startup info
├── .dockerignore                    # Excludes tests/, demo/, docs/, .git, .venv
├── .env.example                     # All configurable vars with defaults for Ollama
└── db/
    └── init.sql                     # PostgreSQL schema (5 tables, auto-mounted by compose)

## Key Patterns

All scanners return ScanResult objects with: scanner_id, is_threat, confidence (0.0-1.0), threat_category (ThreatCategory enum), matched_patterns, sanitized_input, latency_ms.

Dual-path execution in main.py:
- innate_result = await _innate.scan(context) runs synchronous (<5ms)
- if innate_result.should_block: return block immediately
- adaptive_task = asyncio.create_task(_adaptive.analyze(context)) runs async in parallel with model call
- model_response = await forward_to_model(context)
- adaptive_result = await adaptive_task
- if adaptive_result.should_block: return block before delivering response

Antibody learning loop: When L3 **blocks** a novel attack that L2 missed (`is_novel and should_block`), the system embeds the attack in the vault, triggers clonal selection (generate regex candidates, affinity test against benign corpus, promote best to L2 regex engine). Runs via async callback (`asyncio.create_task`, fire-and-forget). CRITICAL: antibody generation is gated on `should_block`, NOT just `adaptive_caught` — sub-threshold DeBERTa detections (is_threat=True but below 0.90) must NOT generate antibodies, as borderline/FP prompts would pollute the vault and cascade into L2 FPs via clonal selection. Clonal selection uses two-stage affinity testing: stage 1 scores against source attack only (must match), stage 2 scores against full positive set (breadth bonus). A 1.15x antibody activation boost ensures source-matching patterns reach the 0.7 affinity threshold. FPR=0 hard gate prevents false positives.

Adaptive blocking: MCAV fusion score (>= 0.9) OR any single analyzer with confidence >= 0.90 triggers block. The 0.90 threshold (not 0.85) avoids FPs from borderline DeBERTa scores on business prompts with injection-adjacent vocabulary like "override the previous estimate".

Multi-turn sequence analysis (`layers/adaptive/multi_turn.py`): Per-session sliding window (10 turns, 30min TTL) with four detection strategies: (1) **escalation trajectory** — monotonically increasing boundary scores over 3+ turns, (2) **boundary testing** — sub-threshold probing with boundary keywords at three intensity levels, (3) **rapid-fire after block** — 3+ requests within 30s after any layer blocks a request (automated retry/fuzzing detection), (4) **topic drift with threats** — high topic drift (>0.4) combined with 2+ turns having innate_max_confidence > 0.3. Multi-turn signals are **DANGER** (probabilistic), not PAMPs (definitive). Threat category: `ThreatCategory.MULTI_TURN_ESCALATION` ("AEGIS.MULTI_TURN"). SessionTurn records enriched with innate report data (innate_max_confidence, threat_categories, was_blocked) on the latest turn. `record_block()` called from main.py on every innate/adaptive/policy block to feed rapid-fire detection. Audit logs include multi-turn details (turn_count, escalation_score, pattern_type, triggered_strategies) in the adaptive JSONB column.

Streaming handler: Adaptive task is awaited BEFORE returning StreamingResponse (not inside stream_generator). This ensures adaptive blocks prevent any response bytes from reaching the client.

Streaming interception (`layers/output/streaming.py`): Production-grade `StreamingInterceptor` wraps upstream SSE streams with real-time validation:
- **Hold buffer** (32 tokens): Tokens are delayed before transmission, giving the PII detector time to scan. PII found in the hold buffer is redacted before the client sees it. PII detected after tokens already sent logs a warning (cannot be recalled).
- **Sliding window** (128 tokens, 50% overlap): At each window boundary, runs PII redaction, toxicity classification, and leakage detection. Each token is evaluated twice due to overlap.
- **Real-time PII scrubbing**: PII is transparently redacted in the stream — clients receive `[SSN]`, `[PERSON_NAME]`, etc. without knowing interception occurred. PII does NOT terminate the stream (redact and continue).
- **Toxicity/leakage termination**: Toxic content or system prompt echo immediately terminates the stream with a safety SSE event and publishes `threat_detected` to the event bus with `stream_interrupted=True`.
- **Cumulative threat tracking**: Rolling average of the last 5 window threat scores. If the average exceeds 0.6 for 3+ consecutive windows, terminates the stream even if no single window crossed the threshold. Catches "slow and low" drip attacks.
- **Stream termination SSE format**: `data: {"choices":[{"delta":{"content":"[AEGIS: Response interrupted...]"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n`
- **Metrics**: `aegis_stream_interruptions_total` (Counter, labels: reason [pii/toxicity/leakage/cumulative]), `aegis_stream_windows_evaluated_total` (Counter), `aegis_stream_tokens_processed_total` (Counter)

Five-stage output cascade (`layers/output/`): All five stages run sequentially; cascade escalation from earlier stages tightens thresholds for later stages.
- **Stage 1 — PII & Secrets Redaction** (`pii_redactor.py`): Presidio NER + regex fallback. 8 secret types: AWS access keys (`AKIA[0-9A-Z]{16}`), AWS secret keys (40-char base64 after known prefixes), OpenAI keys (`sk-*`), Anthropic keys (`sk-ant-*`), GitHub tokens (`ghp_/gho_/ghu_/ghs_/ghr_`), connection strings (`postgresql://`, `mysql://`, `mongodb://`, `redis://` with credentials), private key headers (`-----BEGIN (RSA|EC|OPENSSH)?PRIVATE KEY-----`), generic high-entropy secrets (`key=/token=/secret=` + 40+ chars). Each secret type has a distinct `entity_type` label for PII_REDACTIONS counter. Custom Presidio `PatternRecognizer` instances registered when Presidio is available; same patterns in `_SECRET_PATTERNS` list for regex fallback.
- **Stage 2 — Toxicity Classification** (`toxicity.py`): Keyword regex + weighted n-gram matching. N-gram rules require a trigger phrase AND a context word in the same text (e.g., "how to make" + "bomb" → violence, but "how to make" + "cake" → clean). Category-specific base thresholds: violence=0.55, self_harm=0.50, hate_speech/sexual=0.65, illegal=0.60, regulated_advice=0.75. Three sensitivity levels: HIGH (thresholds × 0.75), MEDIUM (× 1.0), LOW (× 1.25). `LlamaGuardClassifier` stub: async interface returning clean result with `classifier_type="llama_guard_not_loaded"`. Factory: `create_toxicity_classifier()` returns `LlamaGuardClassifier` when `AEGIS_LLAMA_GUARD_MODEL` env var is set.
- **Stage 3 — Hallucination Detection** (`hallucination.py`): N-gram source coverage for RAG responses. Detects RAG context via system message markers ("Context:", "Retrieved:", "Sources:") or `context.metadata["rag_sources"]`. Computes 4-gram overlap per sentence; sentences with zero overlap = unsupported claims. Flags as hallucination if `source_coverage < 0.3` AND `response > 100 chars`. Non-RAG responses return clean (`source_coverage=1.0`). **Production upgrade**: NLI model (`cross-encoder/nli-deberta-v3-base`) for entailment classification.
- **Stage 4 — Data Leakage Prevention** (`leakage.py`): N-gram overlap for system prompt echo, secret pattern detection, verbatim protected content check.
- **Stage 5 — Schema Compliance** (`schema_validator.py`): JSON parse attempt; if JSON, validates against expected schema (missing/extra fields, type mismatches). Detects injected fields with suspicious names (`system`, `prompt`, `instructions`, `role`, `override`, `exec`, etc.). Oversized payload detection: >10× expected field count. Non-JSON returns clean.
- **Cascade escalation**: PII/secret detection in Stage 1 sets `escalated=True` → Stage 2 toxicity threshold reduced by 20% (`× 0.80`), Stage 3 hallucination coverage threshold raised by 0.20. Elevated `scrutiny_level` (from policy engine) also triggers escalation.
- **Two validate methods**: `validate()` is synchronous (3 stages: PII, toxicity, leakage) — used by streaming chunks and backward-compatible code. `validate_full()` is async (all 5 stages) — used by main.py non-streaming path. `validate_chunk()` delegates to `validate()`.

STIX/TAXII Threat Intelligence (`services/threat_intel.py`): `ThreatIntelManager` ingests STIX 2.1 bundles (indicator objects) into the Threat Vault. Each indicator's `x_aegis_pattern_text` is embedded via MiniLM (or deterministic hash fallback) and stored as a `ThreatIndicator` with `source=STIX_FEED`, `confirmed=False`. Deduplication by SHA-256 `payload_hash` — re-ingesting the same indicator increments `hit_count` instead of creating a duplicate. STIX export adds differential privacy noise to embeddings (Gaussian mechanism: `sigma = sensitivity / epsilon`, default ε=3.0, re-normalize to unit length). AEGIS extensions on STIX indicators: `x_aegis_pattern_text` (raw attack text), `x_aegis_embedding` (384-dim, DP-noised on export), `x_aegis_mitre_id` (ATLAS tactic), `x_aegis_affected_models`, `x_aegis_mitigation`. Category resolution: MITRE ID → `_MITRE_TO_CATEGORY` dict takes precedence; falls back to `_LABEL_TO_CATEGORY` from STIX labels. Seed feed (`data/stix_feeds/seed_threat_feed.json`, 13 indicators) auto-ingested on startup from all `*.json` files in `data/stix_feeds/`. API: POST `/v1/threat-intel/ingest` (ingest bundle), GET `/v1/threat-intel/export?since=<ISO>` (export with DP noise), GET `/v1/threat-intel/stats` (feed statistics). All endpoints require authentication.

Threat vault lifecycle: Acute (0-30 days, full payload) → Persistent (3+ sources or confirmed, truncated payload) → Dormant (180 days unseen, excluded from active scanning). Dormant reactivates to Acute on re-emergence. Maintenance runs on startup and every 6 hours.

Circuit breaker: Closed → Open (on threshold breach) → Half-Open (after cooldown) → Closed (if probes pass). Per-model-endpoint tracking. State transitions publish to the "circuit_breaker" event bus channel.

Event bus (cytokine cascade): Layers communicate asynchronously via `services/event_bus.py`. InMemoryEventBus in tests; RedisEventBus (Redis Streams with consumer groups) in production. L3 publishes `threat_detected` when blocking a novel attack → L6 subscribes to escalate TLI. L7 publishes `circuit_breaker` on state transitions → L6 subscribes. Notifications are advisory, not authoritative — each layer validates against its own state.

Unicode normalization: All text passes through an 8-step pipeline before regex matching: (1) NFKC normalization, (2) BIDI stripping (9 chars), (3) zero-width stripping (22 chars), (4) combining mark stripping (Mn/Me), (5) homoglyph canonicalization (200+ mappings: Cyrillic, Greek, Armenian, Georgian, Cherokee, Fullwidth, Math Bold/Italic, Enclosed, Coptic, Tifinagh), (6) leetspeak normalization (context-aware, skips mixed-case), (7) recursive base64 decode (max 3 iterations), (8) whitespace collapse. Implemented in regex_engine.normalize_text().

Path traversal protection: Supply chain model_path validated using Path.parents comparison (not string startswith). Forbidden prefixes: /etc, /proc, /sys, /dev, /root, /var/run, /run.

## Endpoints

| Endpoint | Auth Required | Purpose |
|----------|--------------|---------|
| POST /v1/chat/completions | Yes (API key) | Main proxy — OpenAI-compatible |
| GET /health | No (minimal) / Yes (full) | Unauthenticated: {"status":"ok"}. Authenticated: full layer status |
| GET /metrics | Yes | Prometheus metrics |
| GET /v1/models | No | List available models |
| GET /v1/audit/recent | Yes | Recent audit records |
| GET /v1/vault/stats | Yes | Threat vault statistics |
| POST /v1/supply-chain/verify | Yes | Run 4-stage model verification |
| GET /v1/supply-chain/report/{id} | Yes | Retrieve verification report |
| POST /v1/threat-intel/ingest | Yes | Ingest STIX 2.1 threat bundle |
| GET /v1/threat-intel/export | Yes | Export indicators as STIX bundle (DP noise) |
| GET /v1/threat-intel/stats | Yes | Threat intel feed statistics |
| POST /v1/agents/register | Yes | Register agent, return JWT token |
| GET /v1/agents/{agent_id} | Yes | Agent status, trust, capabilities |
| POST /v1/agents/{agent_id}/authorize | Yes | Check agent tool authorization |
| POST /v1/agents/message/validate | Yes | Validate inter-agent message |
| GET /v1/agents/stats | Yes | Agent security statistics |
| GET /v1/compliance/frameworks | Yes | List frameworks with control counts |
| GET /v1/compliance/matrix | Yes | Cross-framework coverage matrix |
| POST /v1/compliance/report | Yes | Generate compliance report |
| GET /v1/compliance/report/{id} | Yes | Retrieve generated report |
| GET /v1/compliance/evidence/{cap} | Yes | Evidence for AEGIS capability |
| POST /v1/federated/round | Yes | Trigger federated learning round |
| GET /v1/federated/status | Yes | Federated intelligence status |
| POST /v1/federated/indicators/share | Yes | Share DP-noised threat indicator |
| POST /v1/federated/indicators/receive | Yes | Receive indicators from other instances |
| GET /v1/federated/privacy-budget | Yes | Privacy budget status |
| POST /v1/admin/backup | Yes | Backup vault + signatures to JSON |
| POST /v1/admin/restore | Yes | Restore vault from JSON backup |
| GET /v1/admin/deep-health | Yes | Deep health check (6 components) |
| GET /v1/admin/config-validation | Yes | Last config validation result |
| POST /v1/admin/config-reload | Yes | Re-run config validation |

## Running Tests

Full suite (excluding benchmark): python3 -m pytest tests/ -x -q --tb=short -p no:warnings --ignore=tests/benchmark
Benchmark only: python3 -m pytest tests/benchmark/ -q --tb=short -p no:warnings
Everything: python3 -m pytest tests/ -q --tb=short -p no:warnings
Single file: python3 -m pytest tests/test_attack_battery.py -x -q --tb=short -p no:warnings
Concurrency tests only: python3 -m pytest tests/stress/test_concurrency.py -v --tb=short
Chaos/resilience tests: python3 -m pytest tests/stress/test_chaos_regression.py -v --tb=short
Full stress tests (live servers): AEGIS_STRESS_FULL=1 python3 -m pytest tests/stress/test_stress.py -v --tb=short
Red team tests: python3 -m pytest tests/test_red_team.py -v --tb=short
Red team regression tests: python3 -m pytest tests/test_red_team_regression.py -v --tb=short
APT campaigns: python3 -m red_team.run_red_team (requires PYTHONPATH=/path/to/parent:/path/to/aegis)

## Current Metrics (as of 2026-03-10)

- Tests: 1872 passing, 0 failed, 5 skipped (stress tests require AEGIS_STRESS_FULL=1)
- Adaptive meta-learner: 95 tests (63 adaptive + 32 hardening regression) (Phase 6)
- Infrastructure security: 60 tests across 8 attack modules (Phase 5)
- Benchmark (with DeBERTa): 110 attacks, 500 benign prompts
- TPR (full stack, DeBERTa loaded): 96.36% (106/110)
- TPR (innate L2 only): 95.45% (105/110)
- FPR: 0.00% (0/500) — with DeBERTa loaded
- Pattern library: 170 patterns (+ dynamic patterns from clonal selection at runtime)
- Threat model: 23 threats (T1-T23), pen test recommendations in docs/
- Live demo: All 6 scenarios working with Ollama (llama3.2:3b), DeBERTa loads in ~30s

## Configuration

All configuration via environment variables prefixed AEGIS_ or via AegisConfig in config.py. Key variables:

- AEGIS_UPSTREAM_URL — upstream LLM provider URL
- AEGIS_UPSTREAM_API_KEY — upstream API key
- AEGIS_API_KEY — gateway API key required from clients
- AEGIS_CANARY_SECRET_KEY — HMAC secret for canary tokens (MUST be changed in production)
- AEGIS_BLOCK_THRESHOLD — confidence threshold for blocking (default 0.85)
- AEGIS_ALERT_THRESHOLD — confidence threshold for alerting (default 0.50)
- AEGIS_SUPPLY_CHAIN_MODEL_BASE — allowed base directory for model verification
- AEGIS_SKIP_MODEL_LOAD — set to "true" to skip DeBERTa/sentence-transformers loading (used in tests, CI). Default: False (models load for production/demo). tests/conftest.py sets this automatically.
- AEGIS_MODEL — model name for upstream (e.g., llama3.2:3b for Ollama)
- AEGIS_POLICY_BACKEND — policy engine backend: 'python' (default) or 'opa'
- AEGIS_POLICY_OPA_URL — OPA server URL (default: http://localhost:8181)
- REDIS_URL — Redis connection URL (redis://host:port/db). When set, enables Redis-backed rate limiting and Redis Streams event bus. When unset, falls back to in-memory implementations.
- DATABASE_URL — PostgreSQL connection URL (postgresql://user:pass@host:port/db). When set, enables persistent audit logging, threat indicator storage, and signature persistence. When unset, AEGIS runs without PostgreSQL (JSONL audit only, in-memory vault).

## Live Demo

`demo/live_demo.py` runs 6 scenarios against a live AEGIS instance + Ollama backend. Config in `demo/demo_config.env`. Key details:
- PYTHONPATH must point to parent of repo root (so `from aegis.config import ...` resolves)
- uvicorn target is `aegis.main:app` (not `main:app`)
- Startup timeout is 120s to allow DeBERTa model loading (~30s first run)
- 3-second delay between Scenario 4 and 5 allows async clonal selection to complete
- Scenarios: (1) direct injection → L2 regex, (2) encoded injection → L2 regex, (3) PII in response → L5 output, (4) paraphrased injection → L3 DeBERTa, (5) variant of S4 → L2 clonal selection pattern, (6) circuit breaker trip → L7 healing

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

## Docker

Multi-stage build: builder stage installs deps + downloads DeBERTa/MiniLM models (~500MB), runtime stage copies only venv + models + app code. Non-root `aegis` user (uid 1000). Models baked into image at `/opt/models` — no network download at startup.

```bash
docker compose up -d              # Start AEGIS + Redis + PostgreSQL
docker compose logs -f aegis      # Follow AEGIS logs
docker compose down -v            # Stop and remove volumes
docker build -t aegis .           # Build image only
```

Key container paths:
- `/app/aegis/` — application code (PYTHONPATH=/app)
- `/opt/models/` — HF/sentence-transformers model cache
- `/app/aegis/logs/` — audit logs (mounted volume)
- `/app/aegis/data/` — patterns, blocklist, seed threats

docker-entrypoint.sh: Waits up to 30s each for Redis and PostgreSQL TCP connectivity. Runs `alembic upgrade head` if alembic.ini exists. Prints startup info (upstream URL, model status, backing services). Parses REDIS_URL and DATABASE_URL environment variables to extract host:port (handles `postgresql+asyncpg://` scheme).

docker-compose.yml: Four services on bridge network. AEGIS gets 4G memory limit, `extra_hosts` for `host.docker.internal` (Ollama access). Redis uses 256MB maxmemory with LRU eviction, appendonly persistence. PostgreSQL auto-runs `db/init.sql` via `/docker-entrypoint-initdb.d/`. OPA (`openpolicyagent/opa:latest-static`) serves Rego policies on port 8181 (read-only volume mount from `data/opa_policies/`). All services have health checks.

requirements-docker.txt: Additional deps for containerized deployment: `redis[hiredis]`, `asyncpg`, `psycopg2-binary`, `sqlalchemy[asyncio]`, `alembic`. Install after requirements.txt.

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

## Redis Integration

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

## PostgreSQL Integration

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

## Multi-Tenant Architecture

AEGIS supports per-tenant configuration via the `tenants` PostgreSQL table. Each tenant gets custom rate limits, detection thresholds, allowed model lists, and policy tiers. Multi-tenant is optional — without PostgreSQL or TenantManager, AEGIS falls back to environment-variable defaults.

**Tenant Manager** (`services/tenant_manager.py`): Loads tenant configs from PostgreSQL, caches in-memory keyed by `api_key_hash` (SHA-256). Features:
- `TenantConfig` dataclass: tenant_id, name, api_key_hash, rate_limit_rpm/burst, max_tokens_per_request, block/alert thresholds, allowed_models, policy_tier, custom_policy
- `load_all()`: Bulk-load all enabled tenants on startup
- `resolve_tenant(api_key_hash)`: Cache lookup (5-minute TTL) → DB lookup on miss → expired cache as fallback → None
- Background refresh every 60 seconds via `start_refresh()`
- `stats` property for `/health` endpoint (cache_size, tenant_count, last_refresh, hit/miss counters)
- Graceful degradation: if DB is down, cached tenants remain usable; if no cache, returns None and layers use env-var defaults

**Barrier Layer** (`layers/barrier.py`): Accepts optional `tenant_manager` parameter. When set:
- Resolves tenant from API key hash after authentication
- Applies per-tenant rate limits (creates separate rate limiters per tenant)
- Rejects requests for models not in `tenant.allowed_models` (403)
- Applies per-tenant `max_tokens_per_request`
- Binds resolved `tenant_id` to RequestContext for downstream layers
- On tenant resolution failure, falls back to key-prefix tenant derivation

**Policy Engine** (`layers/policy/__init__.py`): Accepts optional `tenant_manager` parameter. Tier 2 now loads real per-tenant config:
- `_resolve_tenant_policy()`: Checks manually registered policies first, then TenantManager cache
- Policy tier modifiers applied to `block_threshold`:
  - `strict`: threshold * 0.85 (15% lower → more aggressive blocking)
  - `permissive`: threshold * 1.10 (10% higher → fewer blocks, capped at 1.0)
  - `standard`: threshold unchanged
  - `custom`: threshold unchanged, custom_policy JSONB evaluated
- Custom policy rules (from JSONB): `blocked_categories` (list of MITRE ATLAS tactic IDs to hard-block)

**Configuration Precedence**: Manually registered TenantPolicy > TenantManager (from DB) > Global defaults (env vars)

## OPA Policy Engine

AEGIS supports Open Policy Agent (OPA) as an alternative policy backend. When `AEGIS_POLICY_BACKEND=opa`, the `OPAPolicyEngine` sends policy decisions to an OPA server via HTTP and falls back to the Python `PolicyEngine` on any failure (timeout, connection error, invalid response).

**Configuration**:
- `AEGIS_POLICY_BACKEND`: `python` (default) or `opa`
- `AEGIS_POLICY_OPA_URL`: OPA server URL (default: `http://localhost:8181`)

**Rego Policy** (`data/opa_policies/aegis.rego`): Three-tier evaluation mirroring the Python engine:
- **Tier 1 (Global)**: RED fail-closed (block all), hard block on MITRE ATLAS categories (`AML.T0051`, `AML.T0054`, `AML.T0048`, `AML.T0051.001`, `AEGIS.PROMPT_INJECTION`, `AEGIS.JAILBREAK`, `AEGIS.SYSTEM_PROMPT_EXTRACTION`) at ≥0.85 confidence
- **Tier 2 (Tenant)**: Custom blocked categories, threshold with policy_tier modifier (strict ×0.85, permissive ×1.10, standard as-is)
- **Tier 3 (Adaptive)**: TLI-adjusted fused score thresholds: GREEN=0.85, BLUE=0.765, YELLOW=0.6375, ORANGE=0.51. Fused score = max(innate, adaptive) with corroboration boost when both ≥0.5

**OPA Input Document** (built by `build_opa_input()`): `input.request_id`, `input.model`, `input.threat_level` (GREEN/BLUE/YELLOW/ORANGE/RED), `input.innate.*`, `input.adaptive.*`, `input.tenant.*` (including `policy_tier` and `custom_policy`).

**OPA Response**: `{"action": "allow|block", "reason": "...", "triggered_by": "global|tenant|adaptive", "applied_policies": [...]}`. All policy IDs prefixed with `opa:` in the resulting `PolicyDecision`.

**Fallback**: On any OPA failure (50ms timeout, connection error, invalid response, missing httpx), the Python `PolicyEngine` handles the request. Fallback decisions tagged with `opa:FALLBACK`. Decision counts tracked via `_opa_count` and `_fallback_count` (exposed in `/health`).

**Docker Compose**: OPA service (`openpolicyagent/opa:latest-static`) on port 8181, policies mounted read-only from `data/opa_policies/`. OPA unit tests: `opa test data/opa_policies/ -v`.

## Supply Chain Verification (Enhanced)

Four-stage model verification pipeline with Sigstore integration and aggregate risk scoring.

**Stage 1 — Cryptographic Integrity** (`integrity.py` + `sigstore_verifier.py`):
- `SigstoreVerifier` class: walks model directory, computes SHA-256 hash for every file, generates manifest dict
- `IntegrityResult` dataclass: is_verified, hash_manifest, signature_valid, signer_identity, rekor_entry, warnings, files_hashed, files_skipped
- Optional Sigstore: if sigstore-python installed AND `.sigstore` bundle files present, verifies signatures against Rekor transparency log. Gracefully degrades to hash-only otherwise.
- Skips files >10GB with warning. Handles permission errors, symlink loops gracefully.
- `verify_against_manifest()`: compares computed hashes against expected manifest

**Stage 2 — Serialization Safety** (`serialization.py`):
- Format risk scoring: `FORMAT_RISK_SCORES` dict maps extensions to 0.0-1.0 risk. SafeTensors=0.0, ONNX=0.1, pickle/PyTorch=0.8, unknown=0.9.
- `get_format_risk(ext)` utility function for external consumers.
- Pickle-in-ZIP detection: `_detect_pickle_in_zip()` opens ZIP archives and checks for `.pkl` files or pickle magic bytes inside members. PyTorch `.pt` files are ZIP archives with pickle inside.
- ONNX structural validation: checks protobuf header (field 1, wire type 0 = 0x08). Informational warning on mismatch.
- SafeTensors header validation: checks 8-byte u64 LE length + JSON '{' start. Suspicious header length warning.
- Recommendations: "Convert to SafeTensors" when pickle-only, "Remove pickle files" when both formats present.
- `aggregate_risk` in details: max risk score across all files.
- Structural validation warnings are `severity: "info"` — do NOT affect scan status.

**Stage 4 — Behavioral Probing** (`behavioral_probe.py`):
- 10 jailbreak resistance probes (expanded from 5): ignore_instructions, DAN, roleplay, developer_mode, opposite_day, translation_bypass, hypothetical, grandma, system_override, academic.
- 4 sleeper agent probes: date-conditioned, deployment-env, version-trigger, code-trigger. Each has `prompt_conditioned` and `prompt_unconditioned`.
- Sleeper detection: both must refuse (pass). Conditioned complies + unconditioned refuses = `SLEEPER_INDICATOR` alert (hard fail). Both comply = general jailbreak issue.
- `compute_divergence(a, b)`: Jaccard distance between token sets (0.0=identical, 1.0=completely different). Included in sleeper results.
- `model_url` parameter: passed through to details for remote model probing (skipped when None).

**Orchestrator** (`__init__.py`):
- `SigstoreVerifier` instantiated in `SupplyChainVerifier.__init__()`.
- Aggregate `risk_score`: weighted average of per-stage risk scores (integrity=0.3, serialization=0.3, dependency=0.2, behavioral=0.2).
- `_stage_risk_score()` helper: maps status to base score (pass=0.0, warn=0.4, fail=0.8), then adjusts from stage-specific details (CRITICAL CVE→0.95, sleeper alert→0.95, hash mismatch→0.9).
- Skipped probes (all results "skipped") don't inflate behavioral risk score.
- Critical finding escalation: any stage with risk ≥ 0.9 added to `critical_findings` list. Aggregate risk ≥ 0.9 also. Critical findings override summary to "CRITICAL" and force `overall_status="fail"`.
- `ModelVerificationReport` enhanced: `risk_score`, `stage_risk_scores`, `critical_findings` fields.
- API response includes all new fields.

## Multi-Agent Security (MHC Identity Verification)

Biological analog: MHC molecules present antigens on cell surfaces. T-cells verify identity before interacting. Cells lacking proper MHC-I are destroyed.

**Agent Identity** (`layers/agent_security/identity.py`):
- `AgentIdentity` dataclass: agent_id, parent_agent_id, capabilities, scope, trust_level, lineage, created_at, expires_at, is_active, last_active, token
- `AgentIdentityManager`: in-memory registry with HMAC-SHA256 JWT signing
- `register_agent(parent_id, capabilities, scope, ttl_hours)`: creates identity, signs JWT, returns AgentIdentity with token
- `verify_agent(agent_id)`: checks active + not expired + applies trust decay, returns AgentIdentity or None
- `verify_token(token)`: verifies JWT signature + expiry, resolves to AgentIdentity
- `update_trust(agent_id, delta, reason)`: adjusts trust, clamped [0.0, 1.0], logged
- Trust constants: CLEAN_INTERACTION=+0.02, POLICY_VIOLATION=-0.1, CONFIRMED_ATTACK=-0.5, INJECTION_DETECTED=-0.2
- Trust decay: 0.1/day after 24h inactive (lazy evaluation on verify)
- Signing key: `AEGIS_AGENT_SIGNING_KEY` env var, explicit param, or auto-generated ephemeral key

**Tool Authorization** (`layers/agent_security/authorization.py`):
- `AgentAuthorizationEngine.authorize_tool(agent, tool_name, tool_params)`: escalation check → capability check → scope check → trust check
- Trust requirements: code_exec≥0.7, mcp_server≥0.8, db_query≥0.6, file_read≥0.4, file_write≥0.6, web_search≥0.3, api_call≥0.5. Unknown tools: 0.5.
- Scope enforcement: file path prefix matching, database allowlist, API endpoint prefix matching
- Capability escalation: 9 keywords (grant_capability, escalate, admin_access, etc.) detected in param keys and string values → always blocked
- Decision log: every authorization decision recorded for audit trail

**Message Validation** (`layers/agent_security/message_validator.py`):
- `AgentMessageValidator.validate_message(sender, receiver_id, message)`: sender active/expired → quarantine → injection scan → action capability
- Injection scanning: creates minimal RequestContext, delegates to L2 innate layer. On detection, reduces sender trust by TRUST_INJECTION_DETECTED (-0.2).
- Per-agent quarantine sets: `quarantine_sender(receiver_id, sender_id)`, `unquarantine_sender()`, `is_quarantined()`
- Without innate layer: logs warning, skips injection scan (does NOT block)

**Pipeline Integration** (`main.py`):
- `X-AEGIS-Agent-ID` header → barrier verifies agent identity → 403 if invalid
- Agent trust bound to `RequestContext.agent_trust_level` (default 1.0 for non-agent requests)
- Scrutiny multiplier: trust < 0.3 → ×2.0, trust < 0.5 → ×1.5, else ×1.0. Applied to output scrutiny_level.
- Trust reinforcement: +0.02 on successful (200) request, -0.1 on blocked (403) request
- AgentSecurityLayer initialized in `_init_layers` with innate_layer and event_bus

## Compliance Dashboard

Cross-framework regulatory compliance reporting. Maps AEGIS controls to six frameworks simultaneously, reducing compliance costs 40-60%.

**Six Frameworks** (`services/compliance/framework_mappings.py`):
- `ComplianceFramework` enum: NIST_AI_RMF, ISO_42001, EU_AI_ACT, CMMC_2, SOC_2, FEDRAMP
- 10 AEGIS capabilities × 6 frameworks = 68+ control mappings
- Each `FrameworkControl`: framework, control_id, control_name, description, aegis_capability, evidence_type
- Query functions: `get_controls_for_framework()`, `get_controls_for_capability()`, `get_coverage_matrix()`

**Ten AEGIS Capabilities**: real_time_threat_monitoring, automated_audit_trails, anomaly_drift_detection, policy_enforcement, incident_response, supply_chain_verification, data_protection_pii, human_oversight, risk_assessment, transparency_explainability

**Evidence Collection** (`services/compliance/evidence_collector.py`):
- Four evidence types: continuous_monitoring (metrics/layer state), audit_log (audit records), configuration (AEGIS config), test_result (benchmark data)
- `EvidenceCollector` accepts optional layer references (audit_logger, config, vault, policy, healing, supply_chain)
- Graceful degradation: missing sources noted as gaps in evidence, never crashes
- `collect_evidence(capability)` → Evidence with data dict, summary, and gaps list
- `collect_all()` → evidence for all 10 capabilities

**Report Generation** (`services/compliance/report_generator.py`):
- `ComplianceReportGenerator.generate_report(frameworks, time_range_hours)` → ComplianceReport
- ComplianceReport: report_id (UUID), generated_at, frameworks, overall_compliance_score (0-100), per_framework_scores, controls_met/total/with_evidence, gaps, evidence_summary
- ComplianceGap: framework, control_id, control_name, gap_type (no_evidence/partial_evidence/not_implemented), recommendation
- Score: controls_with_evidence / controls_total × 100 per framework
- Reports stored in-memory for retrieval by ID

**ComplianceEngine** (`services/compliance/__init__.py`): Orchestrator wrapping evidence collector and report generator. Initialized in `_init_layers` with layer references. Exposed via `/health` endpoint (frameworks_mapped, total_controls, reports_generated).

## Federated Threat Intelligence (L8 — Herd Immunity)

Privacy-preserving federated learning network. An attack seen by one AEGIS instance immunizes all instances. No raw data is ever shared — only model weight updates and DP-noised embeddings.

**Differential Privacy** (`services/federated/privacy.py`):
- Gaussian mechanism: σ = sensitivity × √(2 × ln(1.25/δ)) / ε
- Default: ε=3.0, δ=1e-5, total budget=100.0
- Budget tracking: each `add_noise()` call consumes ε from the budget
- Budget exhaustion → operations refused (RuntimeError)
- `add_noise_to_weights()` and `add_noise_to_embedding()` for specific use cases

**Local Trainer** (`services/federated/local_trainer.py`):
- Logistic regression on TF-IDF character n-gram features (3-5 grams)
- Hashing trick for bounded vocabulary (default 5000 features)
- `add_training_sample(text, is_threat)` → `train()` → `LocalModelUpdate`
- `LocalModelUpdate`: instance_id, round_number, weights dict, num_samples, metrics

**Federated Aggregation** (`services/federated/aggregation.py`):
- FedAvg: weighted average of model updates proportional to num_samples
- Weight poisoning defense: L2 norm > 10× median → clipped to median norm
- Minimum participant threshold (configurable, default 1)
- `aggregate(updates)` → `AggregationResult` with global weights

**Indicator Sharing** (`services/federated/indicator_sharing.py`):
- `prepare_indicator(text, embedding, ...)` → SharedIndicator with DP-noised embedding
- Raw text never included — only content hash (SHA-256) for deduplication
- `receive_indicator()` / `receive_batch()` with hash-based dedup

**FederatedIntelligenceManager** (`services/federated/__init__.py`): Orchestrator with round lifecycle. `run_federated_round()` trains local model, applies DP noise, aggregates with external updates, distributes global weights. Background scheduler runs rounds every 6 hours. Initialized in `_init_layers`, scheduler started/stopped in lifespan. Env vars: `AEGIS_INSTANCE_ID`, `AEGIS_DP_EPSILON`, `AEGIS_DP_DELTA`, `AEGIS_DP_BUDGET`.

## Enterprise Hardening and Deployment

Production-readiness infrastructure: startup validation, vault backup/restore, and deep health monitoring with auto-recovery.

**Configuration Validator** (`services/deployment/config_validator.py`):
- `ConfigValidator.validate(config)` → `ConfigValidationResult` with errors (fatal) and warnings (non-fatal)
- Called in `main.py` lifespan BEFORE `_init_layers()` — errors trigger `SystemExit(1)`
- 13 validation checks:
  1. Empty `AEGIS_API_KEY` → ERROR
  2. Weak API key ("changeme", "test", "example", <16 chars) → WARNING
  3. Default canary secret key → WARNING (production: ERROR)
  4. Default agent signing key → WARNING
  5. Alert threshold ≥ block threshold → ERROR
  6. Rate limit RPM ≤ 0 → ERROR
  7. Rate limit burst ≤ 0 → ERROR
  8. Max tokens per request ≤ 0 → ERROR
  9. Empty upstream URL → ERROR
  10. Production without SSL on DATABASE_URL → WARNING
  11. Production without REDIS_URL → WARNING
  12. DP epsilon ≤ 0 or > 50 → ERROR; ε > 10 → WARNING
  13. Invalid policy backend (not "python" or "opa") → ERROR

**Vault Backup/Restore** (`services/deployment/backup_restore.py`):
- `VaultBackupManager.backup_vault(vault, output_path, signature_store)` → `BackupResult`
- Exports indicators + signatures as portable JSON. Embeddings NOT included (rebuilt on restore with placeholder `[0.0]*384`).
- JSON structure: `{metadata: {timestamp, aegis_version, counts, vault_stats}, indicators: [...], signatures: [...]}`
- `restore_vault(vault, backup_path, signature_store)` → `RestoreResult`
- Validates JSON structure (metadata, indicators, signatures keys required), deduplicates by indicator_id
- Graceful failure: permission errors, malformed JSON, empty files, missing keys
- Backup dir configured via `AEGIS_BACKUP_DIR` env var (default `/tmp/aegis-backups`)

**Deep Health Monitor** (`services/deployment/health_monitor.py`):
- `DeepHealthMonitor.run_deep_health_check()` → `DeepHealthResult` with per-component status and recommendations
- Six component checks (Redis/PostgreSQL run concurrently, others sequential):
  1. **Redis**: PING + round-trip latency. >100ms → degraded. Disconnected → auto-recovery (re-init_redis)
  2. **PostgreSQL**: SELECT 1 + latency. >200ms → degraded
  3. **FAISS**: Index size vs indicator count. >10% mismatch → degraded
  4. **ML Model**: DeBERTa classifier + semantic model loaded check
  5. **Disk Space**: <1GB → degraded, <100MB → unhealthy (critical)
  6. **Circuit Breaker**: Open → unhealthy, half-open → degraded
- Overall status: any unhealthy → unhealthy, any degraded → degraded, else healthy
- Background monitor: `start_monitor()` runs checks every 60s, logs warnings on non-healthy status
- Enabled via `AEGIS_DEEP_HEALTH_ENABLED=true` env var
- Publishes health events to event bus on each check

**Admin API Endpoints** (all require authentication):
- `POST /v1/admin/backup`: Trigger vault backup. Optional `output_path` in body.
- `POST /v1/admin/restore`: Restore vault from backup file. Required `backup_path` in body.
- `GET /v1/admin/deep-health`: Run deep health check, returns per-component results.
- `GET /v1/admin/config-validation`: Returns last startup config validation result.
- `POST /v1/admin/config-reload`: Re-run config validation against current config.

**Deployment Guide**: `docs/DEPLOYMENT.md` — Prerequisites, Docker/local quick start, 12-item production checklist, complete env var reference (30+ variables), Kubernetes scaling guide, backup/restore procedures, 5 troubleshooting scenarios, Grafana dashboard panel recommendations.

## Observability

AEGIS exposes comprehensive Prometheus metrics via `GET /metrics` (authenticated) and structured health via `GET /health`.

**Metrics Catalog** (`middleware/metrics.py`):

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aegis_requests_total` | Counter | method, endpoint, status | Total requests processed |
| `aegis_request_latency_seconds` | Histogram | endpoint | End-to-end request latency |
| `aegis_active_connections` | Gauge | — | Currently active connections |
| `aegis_tenant_requests_total` | Counter | tenant_id, status | Per-tenant request counts (allowed/blocked/error) |
| `aegis_layer_latency_seconds` | Histogram | layer | Per-layer processing latency |
| `aegis_innate_scanner_latency_seconds` | Histogram | scanner | Per-scanner latency within L2 |
| `aegis_adaptive_analyzer_latency_seconds` | Histogram | analyzer | Per-analyzer latency within L3 |
| `aegis_upstream_latency_seconds` | Histogram | endpoint | Upstream model request latency |
| `aegis_threats_detected_total` | Counter | layer, category | Threats detected by layer+category |
| `aegis_policy_decisions_total` | Counter | action, tier | Policy engine decisions |
| `aegis_blocks_total` | Counter | layer, reason | Total blocked requests by layer+reason |
| `aegis_threat_level` | Gauge | — | Current TLI (1=GREEN..5=RED) |
| `aegis_circuit_breaker_state` | Gauge | endpoint | Breaker state (0=closed, 1=open, 2=half_open) |
| `aegis_circuit_breaker_trips_total` | Counter | endpoint | Breaker trip events |
| `aegis_pii_redactions_total` | Counter | entity_type | PII entities redacted |
| `aegis_toxicity_detections_total` | Counter | category | Toxic content detected |
| `aegis_output_cascade_stage_total` | Counter | stage, result | Output cascade stage activations (pii_redaction/toxicity/leakage × detected/clean) |
| `aegis_rate_limit_triggers_total` | Counter | tenant_id | Rate limit rejections per tenant |
| `aegis_vault_size` | Gauge | phase | Vault indicator count by lifecycle phase (acute/persistent/dormant) |
| `aegis_antibody_generations_total` | Counter | — | Antibody generation events |
| `aegis_quarantined_sessions` | Gauge | — | Currently quarantined sessions |
| `aegis_event_bus_events_total` | Counter | channel | Events published to event bus |
| `aegis_upstream_errors_total` | Counter | endpoint, error_type | Upstream model errors |
| `aegis_stream_interruptions_total` | Counter | reason | Stream interruptions (pii/toxicity/leakage/cumulative) |
| `aegis_stream_windows_evaluated_total` | Counter | — | Streaming validation windows evaluated |
| `aegis_stream_tokens_processed_total` | Counter | — | Tokens processed through streaming interceptor |
| `aegis_build_info` | Info | — | Build version metadata |

**Instrumentation Points**: Every layer call in `main.py` records timing via `LAYER_LATENCY`. Per-scanner and per-analyzer breakdowns use `INNATE_SCANNER_LATENCY` and `ADAPTIVE_ANALYZER_LATENCY` (converted from ms to seconds). Output cascade stages tracked individually. Upstream calls timed with `UPSTREAM_LATENCY`. Event bus publishes counted. Quarantine gauge updated on every adversarial event.

**`/health` Endpoint** (authenticated response schema):
```json
{
  "status": "healthy|degraded|unhealthy",
  "version": "0.1.0",
  "threat_level": "GREEN",
  "components": {
    "barrier": {"status": "healthy", "type": "security_layer"},
    "circuit_breaker": {"status": "healthy", "type": "resilience", "state": "closed", "consecutive_trips": 0},
    "redis": {"status": "degraded", "type": "backing_service", "details": {...}},
    "postgres": {"status": "degraded", "type": "backing_service", "details": {...}},
    "event_bus": {"status": "healthy", "type": "messaging", "backend": "memory"},
    "threat_vault": {"status": "healthy", "type": "memory", "total_indicators": 25}
  }
}
```
Overall status derived from worst component: any unhealthy security layer → unhealthy (503); degraded backing services → degraded (200); all healthy → healthy (200).

**`_record_request_metrics()`** (`main.py`): Centralized helper called at end of each request recording tenant-labeled request counts, threat level gauge, and quarantine gauge.

## Concurrency Safety

Concurrency fixes applied to prevent race conditions under async/threaded access:

1. **`tenant_manager.py`**: `evict_expired()` uses `pop(k, None)` instead of `del` to prevent KeyError during concurrent eviction.
2. **`healing.py`**: `get_breaker()` uses `setdefault()` pattern — creates breaker, then atomically inserts only if key is still absent. Prevents duplicate breakers.
3. **`barrier.py`**: `_get_tenant_limiter()` uses same `setdefault()` pattern for tenant rate limiters.
4. **`barrier.py`**: Redis rate limiter uses atomic Lua script instead of two-pipeline ZREMRANGEBYSCORE+ZCARD then ZADD. Eliminates TOCTOU race where concurrent requests could both pass the count check.
5. **`identity.py`**: `AgentIdentityManager` has `_registry_lock` (asyncio.Lock) for future concurrent registration protection.
6. **`threat_vault.py`**: `search()` takes snapshots of `_indicators` and `_embeddings` under `_lock`, then processes outside the lock. Prevents index/list mismatch during concurrent `add()`.
7. **`multi_turn.py`**: `_request_count` increment moved inside `with self._lock:` to prevent lost updates. Uses `threading.Lock` (not asyncio.Lock) because existing tests use it synchronously.

## Stress Test Infrastructure

Located in `tests/stress/`. Two categories:

**Concurrency tests** (`test_concurrency.py`, 18 tests): CI-safe, no live servers. Tests validate all concurrency fixes above using thread pools, asyncio.gather, and concurrent.futures. Always run in CI.

**Full stress tests** (`test_stress.py`, 5 tests): Require `AEGIS_STRESS_FULL=1`. Spawn real AEGIS + mock upstream as subprocesses. Test scenarios: baseline throughput, rate limit enforcement, circuit breaker trip, attack detection under load, mixed traffic.

Support modules: `mock_upstream.py` (configurable FastAPI mock LLM), `load_generator.py` (concurrent HTTP load with percentile metrics), `metrics_collector.py` (Prometheus scraper), `report.py` (JSON/human-readable reports), `stress_runner.py` (subprocess lifecycle).

## Chaos Testing and Resilience

**Chaos infrastructure** (`tests/stress/chaos_runner.py`): `ChaosRunner` class with context-manager methods that monkey-patch live AEGIS objects to simulate failures. Automatic state restoration on context exit (even on exceptions). Methods: `kill_redis_client()`, `kill_session_factory()`, `corrupt_faiss_index()`, `force_threat_level()`, `fill_audit_buffer()`, `slow_upstream()`, `upstream_errors()`.

**Chaos regression tests** (`tests/stress/test_chaos_regression.py`, 44 tests): CI-safe validation of every degradation path. Nine test classes:

| Class | Tests | Validates |
|-------|-------|-----------|
| TestRedisFailure | 3 | In-memory rate limiting fallback, event bus fallback, restore path |
| TestPostgresFailure | 4 | Audit buffering, vault FAISS-only mode, tenant resolution, buffer overflow |
| TestCircuitBreakerChaos | 3 | Full lifecycle, concurrent trips, recovery under load |
| TestTLIEscalation | 5 | RED fail-closed, recovery to GREEN, concurrent evaluate, step up/down |
| TestFAISSCorruption | 3 | Numpy fallback, recovery after corruption, concurrent search |
| TestAuditBuffer | 4 | Bounded buffer, concurrent write/read, chaos fill, empty flush |
| TestMemoryBounds | 3 | Session cleanup, vault proportional growth, agent expiry |
| TestEndToEndDegradation | 6 | No-Redis requests, no-PG requests, health status derivation, no-500 guarantees |
| TestPolicyEngineEdgeCases | 4 | None reports, high innate blocks, ORANGE threshold, GREEN moderate |
| TestChaosRunnerContextManager | 4 | State restore for FAISS, TLI, session_factory, exception safety |
| TestMultiComponentFailure | 3 | Redis+PG both down, FAISS+TLI elevated, all backing services down |

## Resilience Guarantees (Validated Under Test)

| Component Down | Behavior | Validated By |
|---------------|----------|--------------|
| **Redis** | In-memory rate limiting + InMemoryEventBus | TestRedisFailure (3 tests) |
| **PostgreSQL** | Audit buffers to deque (10k max), vault FAISS-only, tenant returns None | TestPostgresFailure (4 tests) |
| **FAISS index** | Brute-force numpy fallback for vector search | TestFAISSCorruption (3 tests) |
| **Redis + PostgreSQL** | Full pipeline works with zero backing services | TestMultiComponentFailure |
| **All backing services** | Core L1-L7 pipeline operates in-memory only | test_all_backing_services_down_pipeline_works |

**Invariants proven under test:**
1. AEGIS never returns HTTP 500 due to a backing service failure (4 tests)
2. TLI RED blocks ALL requests including benign (fail-closed proven)
3. TLI recovers cleanly from RED → GREEN (no stale state)
4. Circuit breaker completes full lifecycle under concurrent access
5. Audit buffer never exceeds 10,000 entries (deque maxlen enforced)
6. Session cleanup prevents unbounded memory growth
7. Health endpoint reports "degraded" (not "unhealthy") when backing services are down but core layers are up

## Degradation Behavior Matrix

| Component | Healthy Status | Degraded Behavior | Recovery |
|-----------|---------------|-------------------|----------|
| Redis | Redis rate limiter + Redis Streams event bus | In-memory SlidingWindowRateLimiter + InMemoryEventBus | New BarrierLayer with Redis client |
| PostgreSQL | Persistent audit, dual-write vault, tenant DB lookups | Deque buffer (10k), FAISS-only vault, cached/None tenants | Flush buffer on reconnect |
| FAISS | HNSW sub-ms ANN search | Brute-force numpy cosine similarity | Index rebuilt on next add() |
| Circuit Breaker | CLOSED, all requests to primary | OPEN → fallback routing, HALF_OPEN → probe requests | Auto-recovery via probe success |
| TLI | GREEN, standard thresholds | BLUE-ORANGE: progressively lower thresholds. RED: fail-closed block all | Manual de-escalation or auto-decay |

## Red Team Adversarial Framework

**Operational tool** (`red_team/`): Generates novel attacks designed to bypass AEGIS detection, runs them against the live pipeline, measures evasion rates, and validates immune system learning. NOT a CI test suite — it's used for continuous security validation against running instances.

### Attack Taxonomy (120 attacks in corpus)

| Target Layer | Technique | Count | Difficulty | Goal |
|---|---|---|---|---|
| **L2 Innate** | Unicode homoglyph (extended) | 10 | 3 | Bypass normalize_text() canonicalization |
| **L2 Innate** | Zero-width char insertion | 10 | 3 | Break regex token matching |
| **L2 Innate** | Encoding stacking (ROT13+B64) | 10 | 3 | Evade single-layer decoding |
| **L2 Innate** | Semantic restructuring | 10 | 4 | Avoid trigger keywords entirely |
| **L2 Innate** | Mixed Latin/Cyrillic script | 10 | 4 | Half-and-half script confusion |
| **L3 Adaptive** | Adversarial token insertion | 5 | 4 | Dilute DeBERTa attention |
| **L3 Adaptive** | Confidence boundary probing | 5 | 5 | Find exact DeBERTa threshold |
| **L3 Adaptive** | Context dilution | 5 | 4 | Bury injection in 800+ word doc |
| **L3 Adaptive** | Persona framing | 5 | 3 | Authority claim wrapping |
| **L3 Adaptive** | Hypothetical wrapping | 5 | 3 | Fictional framing |
| **L3 Adaptive** | Academic framing | 5 | 3 | Research claim wrapping |
| **L5 Output** | Partial PII (words) | 4 | 4 | SSN as spoken words |
| **L5 Output** | Encoded PII (B64/hex) | 4 | 4 | PII in encoded form |
| **L5 Output** | Steganographic PII | 4 | 5 | Acrostics, code comments |
| **L5 Output** | Structured PII | 4 | 3 | PII in JSON keys, table headers |
| **L5 Output** | Split PII | 4 | 4 | SSN spread across sentences |
| **Multi-layer** | L2-pass-L3-catch | 5 | 4 | No keywords, semantic intent only |
| **Multi-layer** | L3-pass-L5-catch | 5 | 4 | Benign prompt eliciting PII output |
| **Multi-layer** | Multi-turn escalation | 10 | 5 | 5-message benign-to-inject sequence |

### Campaign Pipeline

1. **Generate/Load** — `AdversarialAttackGenerator.generate_full_corpus()` or load from `adversarial_corpus.json`
2. **Initial Evasion Test** — `EvasionEngine.run_evasion_test()` sends each attack through AEGIS, classifies blocked (403) vs evaded (200)
3. **Learning Validation** — `LearningValidator.validate_learning()` checks antibody generation, sends technique variants, measures generalization rate
4. **Report** — `RedTeamReport.generate_full_report()` produces JSON + markdown with executive summary, per-layer/technique breakdown, recommendations

### Running Red Team

```bash
# Generate corpus
python3 -c "from red_team.attack_generator import AdversarialAttackGenerator; g = AdversarialAttackGenerator(); g.save_corpus(g.generate_full_corpus(), 'red_team/data/adversarial_corpus.json')"

# Run campaign (requires live AEGIS)
python3 -c "
import asyncio
from red_team.campaign_runner import CampaignRunner, CampaignConfig
config = CampaignConfig(name='phase1', aegis_url='http://localhost:8000', api_key='YOUR_KEY')
asyncio.run(CampaignRunner(config).run())
"

# Run red team tests (CI-safe, no live AEGIS needed)
python3 -m pytest tests/test_red_team.py -v
```

### Learning Validation Methodology

- **Antibody count**: Compare threat vault size before/after attack campaign
- **Generalization rate**: For each blocked attack, generate a technique variant (same method, different wording) and re-test. Variant also blocked = generalized. Variant evaded = only learned exact pattern.
- **Detection gaps**: Evaded attacks are logged with full metadata for manual analysis and code fixes
- **Report grades**: STRONG (<=5% evasion), ADEQUATE (<=15%), NEEDS_IMPROVEMENT (<=30%), CRITICAL (>30%)

## APT Campaign Simulations (Red Team Phase 2)

Six APT campaigns test AEGIS detection across all layers. Run in-process via TestClient (no live server). Each campaign reinitializes layers for clean state.

**Campaigns** (`red_team/apt_campaigns.py`):

| Campaign | Attacks | Technique | Tests |
|---|---|---|---|
| PHANTOM NEEDLE | 18 | 5-stage injection: recon → probing → regex evasion → ML evasion → extraction | System prompt extraction defense |
| SILENT SIPHON | 13 | L5 output evasion: word-spelled PII, base64, steganographic | PII detection coverage |
| SLOW BURN | 20 | 20-message multi-turn progressive jailbreak | Multi-turn escalation detection |
| HYDRA | 200 | 6 mutation strategies × top 20 attacks (cyrillic, extended_homoglyph, reorder, synonym, padding, encoding_mix) | Adversarial fuzzing resilience |
| GHOST PROTOCOL | 7 | Agent registration → capability escalation → injection | Multi-agent security |
| CASCADING FAILURE | 12 | Attacks during Redis kill + recovery | Graceful degradation under attack |

**Results** (DeBERTa/Presidio offline):
- PHANTOM NEEDLE: 77.8% evasion, attacker FAILED (extraction blocked)
- SILENT SIPHON: 100% evasion, attacker SUCCESS (regex PII insufficient)
- SLOW BURN: 50% evasion, attacker FAILED (jailbreak blocked at stage 3)
- HYDRA: 83% evasion, attacker SUCCESS (mutations evade regex)
- GHOST PROTOCOL: 71.4% evasion, attacker SUCCESS
- CASCADING FAILURE: 41.7% evasion, detection maintained during degradation

**Production Code Fixes Applied**:
1. **Extended homoglyphs** (`regex_engine.py`): Added 21 mappings — Armenian (6), Georgian (2), Cherokee (1), extended Cyrillic (5), Greek (2), Roman numerals (3), Latin extended (2). Total: 56+ homoglyph mappings.
2. **Combining mark stripping** (`regex_engine.py`): Strip Mn (nonspacing) and Me (enclosing) unicode categories after zero-width stripping.
3. **Zero-width expansion** (`regex_engine.py`): Expanded from 12 to 22 invisible characters (Hangul fillers, Khmer vowels, Braille blank, Mongolian separator, line/paragraph separators).
4. **Word-spelled SSN** (`pii_redactor.py`): New regex matching 9 consecutive number words.
5. **Base64 PII context** (`pii_redactor.py`): Detect base64 strings near PII keywords (SSN, password, etc.).
6. **Padding-resistant scoring** (`behavioral.py`): Per-hit minimum score (0.15) ensures keywords buried in filler still register.

**Open Gaps** (require production dependencies):
- Steganographic PII — needs Presidio NER
- Synonym/reorder mutations — needs DeBERTa (L3 Adaptive's designed role)
- Encoding mixing — needs recursive base64/ROT13 decoding
- Agent capability escalation — needs tighter pipeline integration

**Running campaigns**: `python3 -m red_team.run_red_team` (all) or `--campaign "PHANTOM NEEDLE"` (specific)

**Regression tests**: `tests/test_red_team_regression.py` — 71 tests validating each fix individually (homoglyphs, combining marks, zero-width, word SSN, base64 PII, padding scoring).

**Assessment report**: `red_team/data/red_team_assessment.md`

## Adversarial ML Attack Engine (Red Team Phase 3)

Algorithm-driven black-box attacks targeting AEGIS detection boundaries. All algorithms accept a `QueryFn = Callable[[str], tuple[int, str, float]]` callable abstracting transport. All tests are CI-safe with mocked query functions.

**Directory**: `red_team/adversarial_ml/`

```
red_team/adversarial_ml/
├── __init__.py              # QueryFn type alias, dataclasses (AdversarialResult, EvolutionResult, TimingAnalysis, BoundaryMap, RealWorldAttack, AdversarialAssessment)
├── attack_corpus_loader.py  # 200 hardcoded real-world attacks (6 categories), no network
├── black_box_attack.py      # BlackBoxAttacker: TextFooler (synonym), CharSwap (confusable), Paraphrase (50 templates)
├── mutation_engine.py       # EvolutionaryMutationEngine: tournament selection, crossover, 7 mutation operators
├── timing_oracle.py         # TimingOracle: Welch's t-test (no scipy), timing side-channel detection
├── boundary_mapper.py       # DecisionBoundaryMapper: L2 regex, L3 DeBERTa, L5 PII boundary probing
└── run_adversarial_ml.py    # Entry point, query_fn factories, 7-step assessment pipeline
```

**Algorithms**:
- **TextFooler**: Word importance ranking (delete-and-test), synonym substitution (500+ entries)
- **CharSwap**: 229 confusable character mappings (Math Alphanumeric U+1D400, Enclosed U+24B6, Coptic, Tifinagh — beyond normalize_text coverage), zero-width + combining mark insertions
- **Paraphrase**: 50 templates across 8 categories (active→passive, imperative→interrogative, direct→indirect, nominalization, double negation, euphemism, code switching, metaphorical)
- **Evolutionary**: Population-based (50 pop, 30 gen), tournament selection (k=3), sentence crossover, 7 mutation operators (synonym_replace, char_perturb, reorder, inject_benign, paraphrase_clause, encode_segment, language_switch)
- **Timing Oracle**: Welch's t-test (from scratch), p<0.05 = timing leak detected
- **Boundary Mapper**: L2 binary search on word count, L3 word interpolation binary search, L5 PII format variant testing

**Attack Corpus** (200 attacks, `red_team/data/real_world_corpus.json`):

| Category | Count | Examples |
|---|---|---|
| owasp | 50 | OWASP LLM Top 10 (prompt injection, output handling, DoS, supply chain, disclosure, plugins, excessive agency, overreliance, theft) |
| indirect_injection | 30 | HTML comments, markdown images, JSON fields, code comments, email chains, YAML, XML, zero-width chars |
| multi_language | 30 | FR, DE, ES, CN, AR, RU, KO, JP, HI, mixed-language attacks |
| jailbreak | 30 | DAN, STAN, Developer Mode, Grandma, opposite day, roleplay, AI conversation sim, sandbox framing |
| encoding | 30 | Base64, hex, ROT13, URL-encoding, leetspeak, Morse, NATO phonetic, pig latin, reverse, binary, HTML entities |
| logic_manipulation | 30 | Conditional logic, recursive instructions, meta-paradox, Socratic method, appeals to authority, game theory |

**Running**: `python3 -m red_team.adversarial_ml.run_adversarial_ml [--url URL] [--api-key KEY] [--output PATH]`

**Tests**: `tests/test_adversarial_ml.py` — 42 tests (9 test classes, all mocked, CI-safe)

## Extended Campaigns & Hardening (Red Team Phase 4)

Phase 4 combines all attack methods into sustained campaigns, fixes every detection gap, and produces the final battle-tested assessment.

**Extended campaigns** (`red_team/extended_campaigns.py`): 6 new campaigns (7-12):

| # | Campaign | Technique | Attacks |
|---|----------|-----------|---------|
| 7 | SHAPESHIFTER | Evolutionary evasion — 5 mutation generations, burst of best | ~170 |
| 8 | BABEL TOWER | Multi-language (10 languages) + mixed-language + transliterated | ~45 |
| 9 | THOUSAND CUTS | Hidden in traffic: 400 benign + 75 borderline + 25 attacks | 500 |
| 10 | INSIDE JOB | Document injection: HTML comments, markdown, JSON, YAML, BIDI, acrostic | 10 |
| 11 | MIRROR MIRROR | L5 PII evasion: 20 input formats + 10 output formats | 30 |
| 12 | FULL SPECTRUM | Everything combined: 700 requests with 5 checkpoint measurements | 700 |

**Normalization hardening** (`layers/innate/regex_engine.py`):
- `normalize_text()` expanded from 3 steps to 8 steps
- BIDI stripping (9 chars), leetspeak normalization (8 leet chars, context-aware), recursive base64 decode (max 3 iterations)
- Homoglyph map expanded: 56 → 200+ (Fullwidth, Math Bold/Italic, Enclosed, Coptic, Tifinagh)
- Combining mark stripping (Mn/Me categories)
- Leetspeak skips mixed-case tokens (protects base64 strings like `aWdub3Jl`)

**Detection hardening**:
- L2: 8 new indirect injection patterns (II-001 through II-008): HTML comments, markdown comments, JSON metadata, YAML config, code comments, third-person injection, footnotes, safety filter disable
- L5: 8 new PII patterns: code-context SSN/email/phone, URL-embedded SSN/email, reversed SSN, word-spelled CC (13-19 digits), word-spelled phone (10 digits)

**Regression tests**:
- `tests/test_normalization_hardened.py` — 42 tests: confusable map coverage, BIDI stripping, zero-width, homoglyphs, leetspeak, recursive base64, combined evasion
- `tests/test_pii_hardened.py` — 41 tests: code-context PII, URL-embedded PII, word-spelled CC/phone, reversed SSN, indirect injection patterns, existing PII unchanged

**Assessment reports**:
- `red_team/data/FINAL_RED_TEAM_ASSESSMENT.md` — Full assessment with OWASP mapping, residual risks, recommendations
- `red_team/data/final_metrics.json` — Machine-readable metrics

**OWASP LLM Top 10 Coverage**: All 10 risks mapped. STRONG coverage on prompt injection (L2+L3+L4), output handling (L5), DoS (L1+L2+L7), supply chain, and sensitive info disclosure. MODERATE on training data poisoning, plugin design, excessive agency, overreliance, model theft.

**Residual risks**: Paraphrased injection (needs L3 DeBERTa), steganographic PII (needs Presidio NER), non-English attacks (needs language detection), timing side-channel (needs constant-time padding).

## Infrastructure Security Assessment (Red Team Phase 5)

Phase 5 attacks AEGIS itself — not the AI interactions it protects, but the security product's own infrastructure, authentication, isolation, and resilience mechanisms.

**Directory**: `red_team/infrastructure/` — 8 attack modules + orchestrator

**Attack Modules and Findings**:

| Module | Class | Attacks | Key Findings |
|---|---|---|---|
| `auth_attacks.py` | AuthAttacker | Timing attack, format inference, 15 bypass techniques | Non-constant-time key comparison (`==` in main.py, `not in` set in barrier.py), no RBAC |
| `rate_limit_bypass.py` | RateLimitBypass | Session rotation, header injection, quota exhaustion | Rate limiting correctly tied to API key (not headers) |
| `tenant_isolation.py` | TenantIsolationTester | Cross-tenant leakage, privilege escalation, ID injection | No cross-tenant data leakage on tested endpoints |
| `backing_service_attacks.py` | BackingServiceAttacker | Redis/PostgreSQL analysis, dependency trust | Dev config without auth, wildcard dependency versions |
| `denial_of_service.py` | DenialOfServiceTester | Circuit breaker manipulation, TLI manipulation, session flooding, oversized requests | Global TLI (not per-tenant) — single attacker can DOS all; unbounded `_session_strikes` dict |
| `audit_integrity.py` | AuditIntegrityTester | Log injection, completeness, evidence destruction | Prompt hash collision via pipe separator; DELETE/PUT correctly rejected |
| `federated_poisoning.py` | FederatedPoisoningAttacker | Gradient poisoning, indicator poisoning, privacy budget exhaustion | Single-participant clipping bypass (min_participants=1); self-reported num_samples amplification; no indicator verification; reset_budget() unrestricted |
| `supply_chain_self.py` | SelfSupplyChainTester | Model tampering, pattern integrity, dependency audit | No file hash manifest; patterns.json writable; wildcard versions in requirements.txt |

**Security Vulnerabilities Found** (severity breakdown):

| Severity | Count | Examples |
|---|---|---|
| **Critical** | 0 | — |
| **High** | 3 | Timing side-channel on key validation, single-participant gradient poisoning, self-reported num_samples |
| **Medium** | 4 | No file integrity manifest, indicator poisoning without verification, global TLI, privacy budget race condition |
| **Low** | 3 | Unbounded session tracking, no lock file, dev Redis without auth |

**Recommendations** (priority order):
1. Use `hmac.compare_digest()` for API key comparison (constant-time)
2. Set `min_participants >= 3` for federated aggregation
3. Validate `num_samples` against actual training data size
4. Make TLI per-tenant to prevent cross-tenant DoS
5. Add `threading.Lock` to `DifferentialPrivacyEngine._spent_epsilon`
6. Create SHA-256 manifest for `data/*.json` files, verify on startup
7. Add indicator quality verification before vault integration
8. Pin all dependencies to exact versions, add lock file

**Tests**: `tests/test_infrastructure_security.py` — 60 tests across 10 test classes (auth, rate limiting, tenant isolation, DoS, audit, federated, backing services, supply chain, orchestrator, data models). All CI-safe, complete in ~30s.

**Running**:
```bash
# Infrastructure security tests
python3 -m pytest tests/test_infrastructure_security.py -v --tb=short

# Full orchestrator (all attack modules)
python3 -m red_team.infrastructure.run_infrastructure_attacks [--output results.json]
```

## Adaptive Meta-Learner (Red Team Phase 6)

Phase 6 builds an AI that attacks AEGIS's blind spots, learns from each attempt, generates increasingly sophisticated evasion techniques, then uses discoveries to harden detection.

**Directory**: `red_team/adaptive/` — 7 files

**5-Step Pipeline** (`run_adaptive.py`):
1. **Blind Spot Detection** — Analyzes all Phases 1-5 results for systematic weaknesses, coverage gaps, confidence boundaries, temporal degradation
2. **Attack Prediction** — 12-feature logistic regression trained on attack outcomes. Features: length, keyword density, encoding presence, non-ASCII ratio, homoglyph presence, zero-width presence, sentence count, question ratio, capitalization, special chars, repetition, multi-language. Generates predicted evasions via 4 mutation strategies.
3. **Co-Evolution** — 10-round arms race. Each round: generate attacks (seed → simple mutation → predictor-guided → advanced + novelty bonus), test against query_fn, retrain predictor, track evasion rate/FPR/vault growth. Determines winner (attacker/defender/stalemate) and convergence round.
4. **Evasion Fingerprinting** — Classifies each successful evasion into 12 root causes: NORMALIZATION_GAP, CLASSIFIER_BLIND_SPOT, CONTEXT_DILUTION, LANGUAGE_EVASION, FORMAT_EVASION, ENCODING_EVASION, SEMANTIC_RESTRUCTURING, PII_FORMAT_EVASION, LAYER_GAP, INFRASTRUCTURE_WEAKNESS, TIMING_EXPLOIT, TRUST_EXPLOITATION. Merges duplicates, ranks by frequency.
5. **Hardening Generation** — Produces concrete detection rules: 119 new character mappings (5 Unicode blocks), 6 new regex patterns (multi-language, context dilution, persona, meta-instruction, chain-of-thought), 3 new PII patterns (Morse, NATO digits, delimited digits), vault seed entries, configuration changes.

**Key Design Decisions**:
- All components accept `QueryFn = Callable[[str], tuple[int, str, float]]` for transport abstraction
- Co-evolution uses seeded Random for reproducibility
- Predictor uses logistic regression (no external ML deps beyond numpy)
- Fingerprinter is rule-based using payload characteristics
- All tests are CI-safe with mock query functions

**Co-Evolution Results** (mock query_fn, 10 rounds):
- Attacker wins: mutations learn to avoid blocked keywords
- Confirms L3 DeBERTa is essential — keyword-based L2 alone insufficient against adaptive attacker
- Convergence at round 6

**Data Models** (`red_team/adaptive/__init__.py`):
- `EvasionRootCause` enum (12 values)
- `BlindSpot`, `BlindSpotReport` — systematic weakness analysis
- `PredictionReport` — classifier accuracy, feature importances, predicted evasions
- `CoEvolutionRound`, `CoEvolutionReport` — per-round and aggregate arms race results
- `EvasionFingerprint`, `FingerprintReport` — root cause classification
- `HardeningRule`, `HardeningPlan` — concrete detection improvements
- `AdaptiveAssessment` — complete assessment with grade (STRONG/ADEQUATE/NEEDS_IMPROVEMENT/CRITICAL)

**Reports Generated**:
- `red_team/data/adaptive_assessment.json` — Full assessment results
- `red_team/data/ADAPTIVE_RED_TEAM_ASSESSMENT.md` — Phase 6 markdown report
- `red_team/data/COMPREHENSIVE_RED_TEAM_REPORT.md` — All 6 phases combined
- `red_team/data/comprehensive_metrics.json` — Machine-readable comprehensive metrics

**Tests**: `tests/test_adaptive_red_team.py` (63 tests) + `tests/test_hardening_regression.py` (32 tests) = 95 new tests. All CI-safe.

**Running**:
```bash
# Adaptive red team tests
python3 -m pytest tests/test_adaptive_red_team.py tests/test_hardening_regression.py -v --tb=short

# Run full adaptive assessment
python3 -m red_team.adaptive.run_adaptive [--rounds 10] [--attacks-per-round 20] [--output DIR]
```

## Bugs Fixed (2026-03-05 session)

10. **Health endpoint Redis/PG status check** (main.py lines 591, 600): `redis_health()` returns `{"status": "connected"}` but health check was using `.get("connected", False)` which always returned `False`, causing Redis/PG to always show as "degraded" even when connected. Fixed to `.get("status") == "connected"`.

## Critical Rules

1. NEVER delete CLAUDE.md — this is the project's institutional memory
2. Fail-closed everywhere — if a security layer fails, block the request (503), never pass through
3. All tests must pass before any changes are considered complete — current baseline is 1872+
4. Benchmark thresholds: FPR < 1.0%, TPR >= 85%, no single industry FPR > 3%
5. Unicode normalize before regex — all text through normalize_text() before pattern matching
6. Auth required on sensitive endpoints — /metrics, /v1/audit/recent, /v1/vault/stats require valid Bearer token
7. Path validation on supply chain — model_path must resolve within supply_chain_model_base using Path.parents
8. Audit log flush — f.flush() after every JSONL write to prevent corruption on crash
9. No time.sleep() in tests — use timestamp backdating for expiry/cooldown simulation
10. Existing test files and test counts must not decrease — only add, never remove tests
