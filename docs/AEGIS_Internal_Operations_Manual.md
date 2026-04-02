# AEGIS Internal Operations Manual

**Version**: 1.0
**Classification**: INTERNAL — Do not distribute externally
**Last Updated**: 2026-03-30
**Audience**: CEO, Engineering, SOC 2 Auditors, Security Operations

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Layer-by-Layer Reference](#2-layer-by-layer-reference)
3. [Extension Systems](#3-extension-systems)
4. [Thymic Validation Engine (L9)](#4-thymic-validation-engine-l9)
5. [Configuration Reference](#5-configuration-reference)
6. [Event Bus Architecture](#6-event-bus-architecture)
7. [API Reference](#7-api-reference)
8. [Metrics Reference](#8-metrics-reference)
9. [Deployment Guide](#9-deployment-guide)
10. [Incident Response Playbook](#10-incident-response-playbook)
11. [Backup and Recovery](#11-backup-and-recovery)
12. [Red Team History and Findings](#12-red-team-history-and-findings)

---

## 1. System Overview

### 1.1 What AEGIS Is

AEGIS (Adaptive Enterprise Guard for Intelligent Systems) is a commercial AI security platform that implements the human biological immune system as a unified, adaptive defense wrapper for enterprise AI systems. It operates as an OpenAI-compatible reverse proxy: applications connect by changing their `base_url` to point at AEGIS. Zero code changes required.

AEGIS sits between applications and AI model providers, inspecting every request and response across the full threat spectrum: prompt injection, jailbreaking, data exfiltration, model supply chain compromise, adversarial perturbation, and multi-agent exploitation.

### 1.2 Architecture Philosophy

AEGIS is built on three foundational principles from immunology:

1. **Innate Immunity** — Fast, broad-spectrum, pattern-based defense. Activates in milliseconds. No prior exposure required.
2. **Adaptive Immunity** — Slower but targeted, learning-based defense. Generates specific responses to novel threats and remembers them.
3. **Immune Regulation** — Prevents the system from attacking legitimate traffic (autoimmune). Calibrates response intensity and maintains tolerance.

The system uses Polly Matzinger's **Danger Theory** (1994) rather than brittle self/non-self definitions. Responses are triggered by **danger signals** — confidence variance spikes, anomalous latency, weight changes, error rates, distribution shift — operationalized via the **Dendritic Cell Algorithm (DCA)** fusing three signal types:

| Signal Type | Description | Examples |
|-------------|-------------|----------|
| **PAMPs** | Confirmed threat indicators | Known attack signatures, regex matches, blocklist hits |
| **Danger Signals** | Potential problem indicators | Behavioral drift, anomalous token distribution, elevated injection scores |
| **Safe Signals** | Normal operation indicators | Baseline metrics match, benign classification, expected patterns |

### 1.3 Design Principles

- **Defense in Depth**: Every layer assumes all other layers have been bypassed (assumed-breach posture).
- **Adaptive Intelligence**: Novel attacks generate new signatures. False positives train tolerance. The platform improves without manual intervention.
- **Zero Trust for AI**: Every prompt is untrusted. Every response is unverified. Every agent-to-agent message is suspicious. Every model artifact is unvalidated until cryptographically verified.
- **Fail-Closed**: If any security layer raises an unhandled exception or is `None`, the pipeline returns HTTP 503. Requests are **never** passed through unscanned.
- **Minimal Latency, Maximum Coverage**: Fast path <5ms synchronous. Slow path 10–50ms asynchronous. Total overhead 30–150ms vs 200–400ms LLM TTFT.
- **Self-Healing**: Automatic remediation — fallback routing, session quarantine, guardrail tightening, model rollback, probe-based recovery.

### 1.4 Nine-Layer Architecture Summary

| Layer | Name | Immune Analog | Latency | Mode |
|-------|------|---------------|---------|------|
| L1 | Barrier | Skin / Mucous Membranes | <1ms | Synchronous |
| L2 | Innate Detection | Pattern Recognition Receptors / NK Cells | <5ms | Synchronous |
| L3 | Adaptive Analysis | T-Cells / B-Cells / DCA | 10-50ms | Asynchronous |
| L4 | Immune Memory | Memory B-Cells / Antibodies (FAISS Threat Vault) | 5-20ms | Sync (fast) / Async (deep) |
| L5 | Output Validation | Complement System (5-stage cascade) | 10-100ms | Synchronous (streaming) |
| L6 | Policy Engine | Regulatory T-Cells | <1ms | Synchronous |
| L7 | Self-Healing | Wound Healing / Tissue Repair | Event-driven | Asynchronous |
| L8 | Federated Intel | Herd Immunity | Async batch (6h rounds) | Background |
| L9 | Thymic Validation | Thymus (T-cell education) | On-demand | Background |

### 1.5 Dual-Path Processing Pipeline

```
Request → L1 Barrier → ┬─ Fast Path (L2 Innate, <5ms, SYNC) ──────────┬→ L6 Policy → Model → L5 Output → Response
                        └─ Slow Path (L3 Adaptive, 10-50ms, ASYNC) ───┘
```

- **Fast Path (Innate)**: Regex matching, blocklist checks, schema validation, PII regex, token limits, canary verification, multi-language detection. Synchronous — must complete before request is forwarded. Zero ML inference cost.
- **Slow Path (Adaptive)**: DeBERTa-v3 classification, embedding similarity (FAISS), behavioral baseline comparison (PSI, KS-tests), multi-turn sequence analysis, DCA signal fusion. Asynchronous — runs concurrently with model call.

Both paths produce threat scores → Policy Engine merges via configurable fusion logic → final **allow / block / escalate** decision.

If adaptive path detects threat after request is forwarded: output validation receives heightened scrutiny, and the next request from that session gets elevated fast-path sensitivity.

### 1.6 Key Implementation Patterns

**Dual-Path Execution** (in `main.py`):
```python
innate_result = await _innate.scan(context)           # <5ms, sync
if innate_result.should_block:
    return block_response(innate_result)

adaptive_task = asyncio.create_task(
    _adaptive.analyze(context, innate_result)          # 10-50ms, async
)
model_response = await forward_to_model(context)       # Concurrent with adaptive

adaptive_result = await adaptive_task
if adaptive_result.should_block:
    return block_response(adaptive_result)             # Block before delivery
```

**Antibody Learning Loop**: When L3 **blocks** a novel attack that L2 missed (`is_novel and should_block`), the system:
1. Embeds the attack text and stores it in the FAISS Threat Vault
2. Triggers clonal selection (generates regex candidates, affinity-tests against benign corpus, promotes best to L2 regex engine)

**Critical**: Antibody generation is gated on `should_block`, NOT just `adaptive_caught`. Sub-threshold DeBERTa detections must NOT generate antibodies — borderline/FP prompts would pollute the vault.

**Adaptive Blocking Logic**: MCAV fusion score ≥ 0.9 **OR** any single analyzer with confidence ≥ 0.90 triggers a block. The 0.90 threshold (not 0.85) avoids false positives from borderline DeBERTa scores on business prompts with injection-adjacent vocabulary.

### 1.7 Current Test Metrics (as of 2026-03-30)

| Metric | Value |
|--------|-------|
| Total test count | 3,706 passing, 0 failed, 6 skipped |
| Test files | 76 files across `tests/` |
| TPR (full stack, DeBERTa loaded) | 96.36% (106/110) |
| TPR (innate L2 only) | 95.45% (105/110) |
| FPR | 0.00% (0/500 benign prompts) |
| Pattern library | 188 regex patterns + 54 multi-language patterns |
| CHIMERA campaign detection | 95% (19/20 in Docker, 90% in CI without Tesseract) |
| Threat model coverage | 23 threats (T1–T23) |

### 1.8 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| API Gateway | FastAPI + uvicorn + httpx | FastAPI 0.115.x, uvicorn 0.34.x |
| Streaming | sse-starlette | 2.2.x |
| Prompt Injection ML | DeBERTa-v3-base (ONNX) | ProtectAI/deberta-v3-base-prompt-injection-v2 |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 (384-dim) |
| Vector Search | FAISS (HNSW) | faiss-cpu 1.9.x |
| PII Detection | Microsoft Presidio | 2.2.x |
| OCR | Tesseract + pytesseract | CJK/multilingual (7 lang packs) |
| Policy Engine | OPA (Rego) with Python fallback | |
| Event Bus | Redis Streams (prod) / asyncio.Queue (dev) | |
| Database | PostgreSQL + pg_trgm | |
| Federated Learning | Custom FedAvg + Google dp-accounting | ε=3 |
| Observability | Prometheus | 0.21.x, 85 metrics |
| Token Counting | tiktoken | cl100k_base encoding |
| Runtime | Python 3.12+ | |

---

## 2. Layer-by-Layer Reference

### 2.1 L1 — Barrier Layer

**File**: `layers/barrier.py`
**Biological Analog**: Skin / Mucous Membranes
**Target Latency**: <1ms
**Mode**: Synchronous

The Barrier Layer is a high-performance reverse proxy entry point. It does **not** inspect content — it enforces structural rules only.

#### 2.1.1 Components

**API Key Validation**:
- Extracts key from `Authorization: Bearer <key>` or `X-API-Key` header (case-insensitive, RFC 6750)
- HMAC-based constant-time comparison to prevent timing side-channel attacks
- Tenant derivation from key format: `aegis-{tenant}-{secret}`, fallback to `"default"` tenant

**Rate Limiting**:
- **In-memory**: `SlidingWindowRateLimiter` — sorted timestamp list per key
- **Redis-backed**: `RedisSlidingWindowRateLimiter` — sorted sets with atomic Lua script (ZREMRANGEBYSCORE + ZCARD + ZADD)
- Effective limit: `rpm + burst` per 60-second window
- Redis key format: `aegis:ratelimit:{api_key_hash}`
- Per-tenant overrides via `TenantManager`

**Token Counting**:
- Encoding: `tiktoken.get_encoding("cl100k_base")`
- Per-message overhead: 4 tokens
- Name field overhead: +1 token
- Reply priming: +2 tokens
- Maximum: `max_tokens_per_request` (default 128,000)

**Schema Validation**:
- Required fields: `{model, messages}`
- 27 allowed fields: `model, messages, stream, temperature, top_p, n, max_tokens, max_completion_tokens, presence_penalty, frequency_penalty, logit_bias, user, stop, seed, tools, tool_choice, response_format, logprobs, top_logprobs, parallel_tool_calls, service_tier, store, metadata, stream_options`
- 3 extra allowed fields for internal use
- Valid message roles: `{system, user, assistant, tool, function}`
- Temperature range: 0–2
- Messages must be a non-empty list
- Per-tenant model whitelist enforcement

**Request Size Limit**: `max_request_size_bytes` (default 10,485,760 = 10MB)

#### 2.1.2 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens_per_request` | 128,000 | Maximum input token count |
| `rate_limit_rpm` | 60 | Requests per minute per API key |
| `rate_limit_burst` | 10 | Burst allowance above steady-state |
| `max_request_size_bytes` | 10,485,760 | 10MB max payload |
| `tls_min_version` | "1.3" | Minimum TLS version |
| `mtls_required` | false | Require client certificate |
| `ip_reputation_enabled` | true | Check IP against threat intel |
| `session_timeout_seconds` | 3,600 | Session expiry for tracking |

#### 2.1.3 Output

Returns `RequestContext` containing:
- Validated request body
- API key hash
- Tenant ID
- Session ID (derived from API key + source IP)
- Source IP
- Token count
- Rate limit status
- Timestamp

#### 2.1.4 Error Responses

| Condition | HTTP Status | Message |
|-----------|-------------|---------|
| Missing API key | 401 | "Missing API key" |
| Invalid API key | 403 | "Invalid API key" |
| Rate limit exceeded | 429 | "Rate limit exceeded" |
| Request too large | 413 | "Request too large" |
| Token limit exceeded | 413 | "Token limit exceeded" |
| Schema validation failure | 400 | "Schema validation error: {details}" |

---

### 2.2 L2 — Innate Detection Layer

**File**: `layers/innate/__init__.py` and `layers/innate/*.py`
**Biological Analog**: Pattern Recognition Receptors (TLRs) / Natural Killer Cells
**Target Latency**: <5ms
**Mode**: Synchronous (must complete before request is forwarded)

Eight scanner modules execute in parallel via `asyncio.gather()`:

#### 2.2.1 Scanner 1 — Regex Engine (TLR Analog)

**File**: `layers/innate/regex_engine.py`
**Pattern Source**: `data/patterns.json`
**Pattern Count**: 188 compiled regex patterns, organized by MITRE ATLAS tactic ID

**8-Step Unicode Normalization Pipeline** (applied before all pattern matching):

| Step | Operation | Details |
|------|-----------|---------|
| 1 | NFKC | Unicode compatibility decomposition and canonical composition |
| 2 | BIDI Stripping | Removes directional override chars (U+202A–U+202E, U+2066–U+2069) |
| 3 | Zero-Width Stripping | Removes 20+ invisible characters (ZWSP, ZWNJ, ZWJ, BOM, soft hyphen, etc.) |
| 4 | Combining Mark Stripping | Removes Unicode categories Mn (nonspacing) and Me (enclosing) |
| 5 | Homoglyph Canonicalization | 200+ character mappings: Cyrillic, Greek, Math, Enclosed, Coptic, Tifinagh, Fullwidth → Latin |
| 6 | Leetspeak Normalization | Contextual — only when adjacent to alphabetic characters |
| 7 | Recursive Base64 Decode | Max 5 iterations + ROT13 decode (keyword-gated to avoid FPs) |
| 8 | Whitespace Collapse | All Unicode whitespace → single ASCII space |

**Pattern Categories** (MITRE ATLAS mapped):
- Direct prompt injection
- System prompt extraction
- Role confusion
- Base64/ROT13/hex encoding attacks
- Jailbreak templates (DAN, Skeleton Key, etc.)
- Social engineering patterns
- Code injection
- Unicode obfuscation

#### 2.2.2 Scanner 2 — Blocklist (Defensin Analog)

**File**: `layers/innate/blocklist.py`
**Source**: `data/blocklist.txt`

- Bloom filter for O(1) first-pass (FP rate: 0.01%)
- Hash table confirmation on Bloom hit
- Hash-based lookup against curated malicious payloads

#### 2.2.3 Scanner 3 — Token Guard

**File**: `layers/innate/token_guard.py`

- Token counting via tiktoken (`cl100k_base`)
- Maximum tokens: 128,000
- Detects anomalous token distributions
- Detects suspicious repetitive patterns
- Detects unusual system/user message ratios

#### 2.2.4 Scanner 4 — PII Regex Detector

**File**: `layers/innate/pii_regex.py`

First-pass regex for:
- SSN (Social Security Numbers)
- Credit card numbers (with Luhn validation)
- Email addresses
- Phone numbers
- IP addresses

#### 2.2.5 Scanner 5 — Canary Token Verifier (NK Cell Analog)

**File**: `layers/innate/canary_verifier.py`

- Injects HMAC-SHA256 canary tokens into system prompts
- Detects by **absence**, not presence — missing, modified, or relocated canary = system prompt tampering
- Secret key: `AEGIS_CANARY_SECRET_KEY` (MUST be changed in production)
- Token format: HTML comment (configurable)

#### 2.2.6 Scanner 6 — Sliding Window Scanner

**File**: `layers/innate/sliding_window.py`

- Defense against padding dilution attacks
- Nested sliding windows of varying sizes
- Catches hidden payloads buried in large benign text blocks

#### 2.2.7 Scanner 7 — Multi-Language Injection Detector

**File**: `layers/innate/multilang_detector.py`

- 54 patterns across 10 languages:
  - Spanish (6), French (6), German (6), Portuguese (5), Italian (5)
  - Russian (6), Chinese (5), Japanese (5), Korean (5), Arabic (5)
- Detects injection attempts in non-English languages

#### 2.2.8 Scanner 8 — LPCI Detector

**File**: Part of the LPCI defense system

- Liquid Prompt Control Injection detection
- Detects control flow manipulation in structured prompts

#### 2.2.9 Aggregation Logic

All scanner results collected as `list[ScanResult]`. For each threat result:
- Track maximum confidence across all scanners
- Collect threat categories (deduplicated)
- If any scanner confidence ≥ `block_threshold` (default 0.85) → **block**

**Output**: `InnateScanReport` with `should_block`, `max_confidence`, `total_latency_ms`, `threat_categories`, and per-scanner `ScanResult` list.

---

### 2.3 L3 — Adaptive Analysis Layer

**File**: `layers/adaptive/__init__.py` and `layers/adaptive/*.py`
**Biological Analog**: T-Cells / B-Cells / Clonal Selection
**Target Latency**: 10–50ms
**Mode**: Asynchronous (runs concurrently with model call)

Six analyzer modules execute in parallel via `asyncio.gather()`:

#### 2.3.1 Analyzer 1 — Injection Classifier (Helper T-Cell)

**File**: `layers/adaptive/injection_classifier.py`
**Model**: `ProtectAI/deberta-v3-base-prompt-injection-v2` (ONNX-optimized)
**Inference**: ~20ms

- Fine-tuned DeBERTa-v3-base for prompt injection detection
- ONNX runtime for optimized inference
- Catches paraphrased/obfuscated injections that regex cannot detect
- Skipped when `AEGIS_SKIP_MODEL_LOAD=1` (tests/CI)

#### 2.3.2 Analyzer 2 — Semantic Search (B-Cell / Antibody)

**File**: `layers/adaptive/semantic_search.py`
**Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)

- Embeds prompt via sentence-transformers
- ANN search against FAISS Threat Vault (HNSW index)
- Cosine similarity > 0.85 = semantically similar to known attack
- Each stored embedding = an antibody

#### 2.3.3 Analyzer 3 — Behavioral Baseline

**File**: `layers/adaptive/behavioral.py`

Per-user/per-tenant behavioral baselines tracking:
- Request frequency
- Token count distribution
- Topic distribution (embedding centroid)
- Time-of-day patterns
- Model usage patterns

Statistical tests:
- **PSI (Population Stability Index)**: >0.25 = significant drift
- **KS tests**: For continuous variable comparison

#### 2.3.4 Analyzer 4 — Multi-Turn Sequence

**File**: `layers/adaptive/multi_turn.py`

Sliding window of last N turns per session. Detects:
- Escalation trajectory (monotonically increasing boundary-testing)
- Boundary testing patterns
- Rapid-fire attacks
- Topic drift indicators
- "Slow and low" attack patterns

#### 2.3.5 Analyzer 5 — LPCI ML Analyzer

ML-based Liquid Prompt Control Injection analysis.

#### 2.3.6 Analyzer 6 — Confidence Margin Booster

**File**: `layers/adaptive/margin_booster.py`

- Activates for DeBERTa scores in the "fragile zone": 0.85–0.95
- Boosts confidence using threat vault similarity evidence
- Prevents borderline attacks from evading detection

#### 2.3.7 MCAV Fusion

The Dendritic Cell Algorithm fuses all analyzer outputs:

```
MCAV = (Σ(PAMP × 3.0) + Σ(DANGER × 2.0)) / (Σ(PAMP × 3.0) + Σ(DANGER × 2.0) + Σ(SAFE × 1.0))
```

- MCAV > 0.7 = anomalous
- MCAV > 0.9 = definitive threat (triggers block)
- Range: 0.0–1.0

#### 2.3.8 Blocking Decision

Two independent blocking paths:
1. **MCAV consensus**: `mcav >= 0.9` (config.mcav_threat_threshold)
2. **Single analyzer override**: Any analyzer confidence ≥ 0.90

The 0.90 single-analyzer threshold prevents SAFE signals from diluting strong PAMP/DANGER evidence, while avoiding FPs on business language.

#### 2.3.9 Antibody Generation

Triggered **only** when `is_novel_attack AND should_block`:
1. Embed attack text via sentence-transformers
2. Store in FAISS Threat Vault with `ThreatIndicator` metadata
3. Fire antibody callback for clonal selection
4. Next semantically similar attack caught by fast-path vector lookup

---

### 2.4 L4 — Immune Memory (Threat Vault)

**File**: `layers/memory/threat_vault.py`
**Biological Analog**: Memory B-Cells / Antibodies
**Storage**: FAISS HNSW index (384-dimensional vectors)

#### 2.4.1 FAISS Index Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `faiss_dimension` | 384 | Vector dimensionality (matches MiniLM-L6-v2) |
| `faiss_index_type` | "HNSW" | Hierarchical Navigable Small World |
| `faiss_hnsw_m` | 32 | HNSW connectivity parameter |
| `faiss_ef_search` | 64 | HNSW search parameter |

Fallback: Numpy brute-force cosine similarity if FAISS unavailable.

#### 2.4.2 Three-Phase Lifecycle

| Phase | Duration | Criteria | Behavior |
|-------|----------|----------|----------|
| **ACUTE** | 0–30 days | Initial state for all new indicators | Full payload stored, high monitoring priority, active scanning |
| **PERSISTENT** | 30+ days (promoted) | ≥3 sources OR confirmed OR (age ≥ acute_days AND frequency ≥ 2) | Optimized storage, embedding retained, integrated into fast path |
| **DORMANT** | 180+ days unseen | No hits for 180 days | Excluded from active scanning, retained in index for reactivation |

Dormant indicators **instantly reactivate** to ACUTE on re-emergence (`record_hit()`).

#### 2.4.3 Key Operations

**Add Indicator**:
- Normalizes embedding (L2 norm)
- Thread-safe (with lock)
- Batched persistence every `persist_every_n_updates` adds (default: 5)
- Dual-write to PostgreSQL (fire-and-forget) when configured

**Search**:
- FAISS L2 distance search → converted to cosine similarity: `cos_sim = 1.0 - dist/2.0`
- Results weighted by `indicator.weight`
- Dormant indicators excluded by default (`include_dormant=False`)
- Returns top-k sorted by weighted similarity

**Lifecycle Maintenance**:
- Background async loop every `maintenance_interval_hours` (default: 6h)
- Runs `lifecycle_update()`: batch ACUTE→PERSISTENT promotions, PERSISTENT→DORMANT demotions
- Auto-persists after transitions

#### 2.4.4 Persistence

| File | Format | Content |
|------|--------|---------|
| `data/threat_vault.faiss` | FAISS binary | Vector index |
| `data/threat_vault_meta.json` | JSON | Indicator metadata |
| `data/threat_vault.npy` | NumPy | Raw embeddings (backup) |

PostgreSQL dual-write: Upserts to `threat_indicators` table when `session_factory` configured.

#### 2.4.5 Seed Data

**File**: `data/seed_threats.json`

Loaded at startup via `load_seed()`. Creates `ThreatIndicator` objects with:
- `source=SEED`
- `confirmed=True`
- Deterministic hash-based pseudo-embedding if no embedding function available

#### 2.4.6 Clonal Selection (Signature Optimization)

**File**: `layers/memory/signatures.py`

When antibody callback fires:
1. Generate 15 regex candidate patterns (`signature_num_candidates`)
2. Test each against:
   - Attack corpus (must detect the original attack)
   - Benign corpus (`data/benign_prompts.json`, 55 prompts)
3. Affinity scoring: TPR × (1 - FPR)
4. Threshold: `signature_affinity_threshold` ≥ 0.7
5. Max FPR: `signature_max_fpr` ≤ 0.05
6. Promote top 3 (`signature_max_promoted`) to L2 regex engine at runtime

Auto-deprecation: Signatures with `signature_deprecation_min_matches` ≥ 20 and degrading performance are retired.

---

### 2.5 L5 — Output Validation Layer

**File**: `layers/output/__init__.py` and `layers/output/*.py`
**Biological Analog**: Complement System (enzymatic cascade)
**Target Latency**: 10–100ms
**Mode**: Synchronous (must complete before response delivery)

Six-stage cascade with escalation:

#### 2.5.1 Stage 1 — PII & Secrets Redaction (Opsonization)

**File**: `layers/output/pii_redactor.py`
**Engine**: Microsoft Presidio + 8 custom secret detectors

Detects and redacts:
- PII: Names, addresses, SSNs, credit cards, emails, phones
- Secrets: AWS keys, OpenAI keys, Anthropic keys, GitHub tokens, generic API keys, connection strings, private keys, passwords

Thresholds:
- `pii_redaction_threshold`: 0.5 (redact above this)
- `pii_alert_threshold`: 0.4 (log below redaction but above alert)

Redaction format: typed placeholders (`[PERSON_NAME]`, `[API_KEY]`, etc.)

#### 2.5.2 Stage 2 — Toxicity & Safety Classification

**File**: `layers/output/toxicity.py`

- Keyword + n-gram classifier with sensitivity levels
- LlamaGuard stub (upgrade path for production)
- Categories: violence, hate, sexual, self-harm, illegal activity, regulated advice

#### 2.5.3 Stage 3 — Hallucination Detection

**File**: `layers/output/hallucination.py`

- N-gram source coverage for RAG responses
- Compares response against provided context
- NLI model upgrade path

#### 2.5.4 Stage 4 — Data Leakage Prevention

**File**: `layers/output/leakage.py`

- System prompt echo detection via n-gram overlap
- `leakage_ngram_size`: 4 (default)
- `leakage_overlap_threshold`: 0.3 (default)
- Detects verbatim regurgitation
- Cross-tenant data contamination checks

#### 2.5.5 Stage 5 — Schema Compliance

**File**: `layers/output/schema_validator.py`

- JSON structure validation
- Injected field detection
- Expected type enforcement
- Size limit checks

#### 2.5.6 Stage 6 — LPCI Output Guard

Liquid Prompt Control Injection output detection.

#### 2.5.7 Cascade Escalation

When Stage 1 detects PII/secrets (redacted_count > 0):
- Stage 2 toxicity threshold reduced by 20% (multiplied by 0.80)
- Stage 3 hallucination coverage boosted by 20% (+0.20)

#### 2.5.8 Streaming Validation

**File**: `layers/output/streaming.py`
**Class**: `StreamingInterceptor`

- Window size: 128 tokens (default)
- Window overlap: 50% (default)
- Tokens buffered in sliding window
- At each boundary, lightweight classifier evaluates
- Threat detected → stream interrupted with safety message as final SSE event
- Hold buffer: Accumulates tokens before first release for initial scan
- Cumulative threat tracking across windows

---

### 2.6 L6 — Policy Engine

**File**: `layers/policy/__init__.py` and `layers/policy/opa_engine.py`
**Biological Analog**: Regulatory T-Cells (Tregs)
**Target Latency**: <1ms
**Mode**: Synchronous

#### 2.6.1 Policy Backends

1. **Python** (default): In-memory dictionary-based evaluation
2. **OPA**: Open Policy Agent with Rego policies via HTTP backend

Configured via `AEGIS_POLICY_BACKEND` ("python" or "opa").

#### 2.6.2 Three Policy Tiers

1. **Global Policies**: Hard safety limits (no CSAM, weapons, etc.), regulatory compliance floors, platform integrity
2. **Tenant Policies**: Industry-specific compliance, custom filters, approved model lists, rate limits, data residency
3. **Adaptive Policies**: Dynamically adjusted based on Threat Level Indicator (TLI)

#### 2.6.3 Threat Level Indicator (TLI)

| Level | Name | Block Threshold | Escalate Threshold | Auto-Decay |
|-------|------|----------------|-------------------|------------|
| 1 | GREEN | 0.85 (standard) | 0.60 | N/A (baseline) |
| 2 | BLUE | 0.765 (−10%) | 0.54 (−10%) | 60s → GREEN |
| 3 | YELLOW | 0.6375 (−25%) | 0.45 (−25%) | 120s → BLUE |
| 4 | ORANGE | 0.51 (−40%) | 0.30 (−50%) | 180s → YELLOW |
| 5 | RED | 1.0 (fail-closed) | N/A (all blocked) | 300s → ORANGE |

**Hard-block categories** (always blocked at 0.85 regardless of TLI): PROMPT_INJECTION, JAILBREAK, SYSTEM_PROMPT_EXTRACTION.

**Escalation Triggers**:
- BLUE: Elevated detection rate (>2× baseline)
- YELLOW: Confirmed novel attack or sustained elevated rate
- ORANGE: Active campaign across multiple tenants
- RED: Critical infrastructure threat or zero-day

**Auto-Decay**: Configurable via `AEGIS_TLI_AUTO_DECAY_ENABLED` (default: true). Each level has a timer that decays to the next lower level.

#### 2.6.4 Score Fusion

Policy engine receives innate and adaptive scores, fuses them:
- If either recommends block → block
- Combined threat score checked against TLI-adjusted thresholds
- Hard-block categories bypass TLI adjustment

---

### 2.7 L7 — Self-Healing Layer

**File**: `layers/healing.py`
**Biological Analog**: Wound Healing / Tissue Repair
**Mode**: Event-driven, asynchronous

#### 2.7.1 Circuit Breaker

Three states per model endpoint:

| State | Name | Behavior |
|-------|------|----------|
| CLOSED | Healthy | Normal operation, error/threat counters tracked |
| OPEN | Damaged | All requests → fallback model. Cooldown active. |
| HALF_OPEN | Probing | Limited probe requests to primary |

**Trigger**: Error rate ≥ 50% over 60-second window, or `force_open()` for confirmed exploit.

**Cooldown Strategy**:
- Initial: 30 seconds
- Exponential backoff: `cooldown × 2^(consecutive_trips - 1)`
- Maximum: 300 seconds (5 minutes)

**Probing (Half-Open)**:
- Send `probe_count` (default: 5) test requests
- All pass → transition to CLOSED
- Any fail → re-trip OPEN with backoff

**Fallback Routing**: Priority-ordered endpoint list per tenant (`fallback_models` config).

#### 2.7.2 Session Quarantine

**Class**: `SessionQuarantine`

- `quarantine_threshold` (default: 3) adversarial events → quarantine
- Tracks both session-level and source-level (API key/IP)
- Methods: `record_adversarial_event()`, `is_quarantined()`, `is_source_quarantined()`

#### 2.7.3 Recovery Telemetry

`RecoveryEvent` published to event bus with:
- Trigger (what caused the action)
- Action taken (failover, quarantine, etc.)
- Duration
- Outcome
- Full context for compliance audit

---

### 2.8 L8 — Federated Intelligence

**File**: `services/federated/__init__.py`
**Biological Analog**: Herd Immunity
**Mode**: Background (6-hour rounds)

#### 2.8.1 Architecture

Each AEGIS instance:
1. Trains a local logistic regression model on TF-IDF features from its own labeled data
2. Applies differential privacy noise (Gaussian mechanism) to weight updates
3. Sends only model weight updates (gradients) — **never training data** — to central aggregation
4. Receives aggregated global model via Federated Averaging (FedAvg)

#### 2.8.2 Privacy Guarantees

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dp_epsilon` | 3.0 | Privacy budget per update |
| `dp_delta` | 1e-5 | DP delta parameter |
| `privacy_budget` | 100.0 | Total privacy budget |

Budget exhaustion check before every sharing operation. Returns `None` if budget exhausted.

#### 2.8.3 Indicator Sharing

STIX 2.1 format with AI-specific extensions:
- Threat embedding (DP-noised)
- MITRE ATLAS classification
- Affected model families
- Detection source
- Confidence score

#### 2.8.4 FedAvg Aggregation

- Minimum participants: `min_participants` (default: 1)
- Weight poisoning defense: Norm clipping on updates
- Round interval: 6 hours (21,600 seconds)

#### 2.8.5 Status

Currently `enabled: false` by default. Requires multiple deployments for meaningful federation.

---

## 3. Extension Systems

### 3.1 Multi-Turn Manipulation Detector (MTMD)

**File**: `layers/adaptive/manipulation_detector.py`
**Config**: `AEGIS_MANIPULATION_DETECTION_ENABLED` (default: true)

Five detection strategies for autonomous jailbreak agents:

| Strategy | Weight | Trigger |
|----------|--------|---------|
| Boundary Testing | 1.0 | Injection score slope > 0.03 over window |
| Tactic Switching | 1.2 | ≥3 distinct attack category prefixes |
| Persona Adoption | 0.8 | Embedding similarity drop <0.3 + injection score >0.3 |
| Escalation Gradient | 1.0 | 5-turn rolling average climb from <0.2 to >0.5 |
| Response Adaptation | 1.5 | Technique switching after blocks |

Thresholds:
- Alert: 0.4 (`AEGIS_MTMD_ALERT_THRESHOLD`)
- Block: 0.7 (`AEGIS_MTMD_BLOCK_THRESHOLD`)
- Window size: 20 turns (`AEGIS_MTMD_WINDOW_SIZE`)

### 3.2 Distillation Defense

**File**: `layers/adaptive/distillation_defense.py`
**Config**: `AEGIS_DISTILLATION_DEFENSE_ENABLED` (default: true)

Cross-session extraction detection with five strategies:

| Strategy | Weight | Min Queries | Trigger |
|----------|--------|-------------|---------|
| Query Diversity | 0.25 | 50 | topic_coverage > 0.7 |
| Boundary Mapping | 0.30 | 30 | block_rate 15–60% + alternation > 0.3 |
| Reasoning Coercion | 0.20 | — | reasoning_ratio > 0.25, count > 20 |
| Info Harvesting | 0.10 | 20 | avg_response > 2× global_avg + diversity > 0.6 |
| Complexity Escalation | 0.15 | 30 | normalized_slope > 0.02 |

Actions:
- Alert: combined score ≥ 0.60
- Block: combined score ≥ 0.85

### 3.3 Source Behavioral Profiler

**File**: `layers/adaptive/source_profiler.py`
**Config**: `AEGIS_PROFILER_ENABLED` (default: true)

Per-source (API key) profiling:
- Diversity scoring
- Injection rate tracking
- Automation detection
- Maximum profiles: 10,000 (`AEGIS_PROFILER_MAX_PROFILES`)

### 3.4 Adaptive Rate Limiter

**File**: `layers/adaptive_rate_limiter.py`
**Config**: `AEGIS_ADAPTIVE_RATE_LIMIT_ENABLED` (default: true)

Per-source rate limiting with three escalation levels:
1. **Slowdown**: Increased delays for suspicious sources
2. **Cooling**: Reduced request allowance
3. **Hard Stop**: Complete block for `hard_stop_duration` seconds

Thresholds:
- Hard stop after: 5 manipulation attempts (`AEGIS_ADAPTIVE_RATE_LIMIT_HARD_STOP_THRESHOLD`)
- Hard stop duration: 300 seconds (`AEGIS_ADAPTIVE_RATE_LIMIT_HARD_STOP_DURATION`)

### 3.5 Jailbreak Taxonomy

**File**: `layers/memory/jailbreak_taxonomy.py`
**Config**: `AEGIS_JAILBREAK_TAXONOMY_ENABLED` (default: true)

15-category jailbreak attempt classification and trend tracking:
- Maximum stored attempts: 50,000 (`AEGIS_JAILBREAK_TAXONOMY_MAX_ATTEMPTS`)
- Tracks technique frequency, evolution, and effectiveness

### 3.6 Tool Invocation Proxy (TIP)

**File**: `layers/tool_proxy/proxy.py`
**Config**: `AEGIS_TOOL_PROXY_ENABLED` (default: true)

Intercepts agent-to-tool calls for security enforcement:

| Component | File | Function |
|-----------|------|----------|
| TIP Core | `proxy.py` | Intercept, log, enforce rate limits |
| TIPE | `policy_engine.py` | Allowlist, param validation, path/SSRF/injection checks |
| TDIV | `description_validator.py` | Tool description injection scanning |
| TRS | `response_sanitizer.py` | Tool output injection scanning, content delimiting |
| TCAD | `chain_detector.py` | Parasitic toolchain detection, dangerous sequences |

Key configurations:
- Default per-tool RPM: 60
- Max parameter size: 10,000 bytes
- Block internal URLs: true (SSRF protection)
- TDIV max description length: 2,000
- TRS max output size: 102,400 bytes (100KB)
- TRS injection block threshold: 0.85
- TCAD window size: 20
- TCAD volume multiplier: 3.0×

### 3.7 Agent Security Layer

**File**: `layers/agent_security/__init__.py` and `layers/agent_security/*.py`

Four components:

| Component | File | Function |
|-----------|------|----------|
| Identity Manager | `identity.py` | JWT signing, trust mechanics, trust decay |
| Authorization Engine | `authorization.py` | Least-privilege, scope enforcement, escalation blocking |
| Message Validator | `message_validator.py` | Injection scanning, quarantine |
| Communication Monitor | `communication_monitor.py` | Lateral movement detection, compromised agent flagging |

IACM thresholds:
- Trust threshold: 0.3 (`AEGIS_IACM_TRUST_THRESHOLD`)
- Injection rate threshold: 0.5 (`AEGIS_IACM_INJECTION_RATE_THRESHOLD`)

### 3.8 Supply Chain Verification

**File**: `layers/supply_chain/__init__.py` and `layers/supply_chain/*.py`

Four-stage verification pipeline + extensions:

| Stage | Component | Weight | Function |
|-------|-----------|--------|----------|
| 1 | Cryptographic Integrity | 0.3 | SHA-256 + Sigstore signatures |
| 2 | Serialization Safety | 0.3 | Format risk (pickle → 0.9, SafeTensors → 0.0) |
| 3 | Dependency Audit | 0.2 | CVE matching, SBOM generation |
| 4 | Behavioral Probing | 0.2 | 10 jailbreak probes, sleeper agent detection |

Critical threshold: aggregate risk ≥ 0.9 triggers immediate escalation.

Extensions:
- **MPV** (Provenance Validator): 5-check model provenance (format, registry, hash, metadata injection, namespace)
- **SPA** (Skill/Plugin Auditor): Descriptor injection, malicious code, permissions, AST analysis
- **DCA** (Dependency Chain Analyzer): Malicious package DB (28 packages), typosquatting detection
- **Validation Cache**: LRU with TTL, content-hash, retroactive threat checks
- **Revalidation Scheduler**: Background asyncio re-validation (active: 6h, inactive: 24h intervals)

### 3.9 Multimodal Security

**File**: `layers/multimodal/__init__.py` and `layers/multimodal/*.py`

Three input channels:

**Images**:
- Format validation (magic bytes, polyglot detection)
- OCR: Tesseract + Pillow heuristic fallback (3-tier)
- Steganalysis: Chi-square LSB, RS analysis, alpha channel
- Metadata stripping: EXIF, XMP, IPTC
- Sanitization: Re-encode to strip steganographic payloads
- Max size: 20MB, max per request: 10

**Documents**:
- 12 MIME type prefixes supported
- Text extraction: PDF, HTML, Office, Markdown, JSON, YAML, CSV, XML
- Hidden content detection: Invisible CSS, comments, zero-width chars, macros, scripts
- Block macros: true (default), block scripts: true (default)
- Max size: 50MB

**Audio**:
- 8 MIME type prefixes supported
- Transcription: Whisper, speech_recognition, fallback
- Spectral analysis: FFT for ultrasonic/infrasonic, entropy, bursts
- WaveGuard sanitization: Bandpass filter + WAV normalization
- Max size: 100MB, max duration: 1,800 seconds (30 min)

**Cross-Modal Engine** (`cross_modal_engine.py`):
- Laundering detection (text in image to bypass text scanners)
- Inconsistency detection (conflicting signals across modalities)
- Escalation detection
- Volume anomaly detection
- Fragmentation detection
- Decoy detection

### 3.10 Chain-of-Thought Defense

**Files**: `layers/output/coherence_analyzer.py`, `layers/output/length_anomaly_detector.py`, `layers/output/alignment_validator.py`

Three components:

| Component | File | Function |
|-----------|------|----------|
| Coherence Analyzer | `coherence_analyzer.py` | Sliding-window coherence, pivot detection, CoT hijacking |
| Length Anomaly Detector | `length_anomaly_detector.py` | EMA baselines, per-(tenant,session) length anomaly |
| Alignment Validator | `alignment_validator.py` | Reasoning-output semantic alignment |

Thresholds:
- Coherence window: 200 tokens
- Pivot threshold: 0.3 (coherence drop below this is a pivot)
- Length anomaly multiplier: 3.0× baseline
- Alignment misalign threshold: 0.3
- Combined block threshold: 0.7

### 3.11 Temporal Defense

**File**: `layers/temporal/__init__.py` and `layers/temporal/*.py`

Seven components:

| Component | File | Function |
|-----------|------|----------|
| Traffic Generator | `traffic_generator.py` | Synthetic queries by domain (12 categories, 120+ templates) |
| Baseline Engine (BBE) | `baseline_engine.py` | 6-metric baseline, 3-tier calibration, drift alerts |
| Canary System | `canary_system.py` | Known-answer probes, keyword + semantic evaluation |
| Provenance Registry (MPR) | `provenance_registry.py` | Append-only state tracking, FIFO eviction |
| Correlation Engine (TCE) | `correlation_engine.py` | Multi-factor temporal correlation |
| Snapshot Manager | `snapshot_manager.py` | Hash-based state capture, replay comparison |
| Date Scanner | `date_scanner.py` | Sleeper agent detection at temporal boundaries |

BBE thresholds:
- Warning: 2.0σ (`AEGIS_BBE_WARNING_THRESHOLD_SIGMA`)
- Critical: 3.0σ (`AEGIS_BBE_CRITICAL_THRESHOLD_SIGMA`)
- Production transition: 500 interactions (`AEGIS_BBE_PRODUCTION_TRANSITION_COUNT`)

Canary thresholds:
- Keyword pass: ≥0.8 hit rate
- Keyword fail: ≤0.3 hit rate
- Semantic pass: ≥0.7 similarity
- Critical alert: 3 consecutive failures

### 3.12 Reasoning Trace Sanitization

**File**: `layers/output/reasoning_sanitizer.py`

Three modes (`AEGIS_REASONING_TRACE_MODE`):
- **monitor**: Log reasoning traces, do not modify (default)
- **redact**: Remove reasoning traces from output
- **summarize**: Replace detailed traces with summaries

### 3.13 Compliance Engine

**File**: `services/compliance/`

Maps AEGIS controls to 6 regulatory frameworks:
- NIST AI RMF
- ISO 42001
- EU AI Act
- CMMC 2.0
- SOC 2
- OWASP LLM Top 10

Automated evidence collection and report generation.

---

## 4. Thymic Validation Engine (L9)

**File**: `layers/thymic/__init__.py` and `layers/thymic/*.py`
**Biological Analog**: Thymus (T-cell education and continuous self-validation)
**Mode**: Background, on-demand

### 4.1 Purpose

L9 continuously validates that all other defense layers (L1–L8) are functioning correctly. It generates synthetic attack probes, routes them through the real AEGIS pipeline, and measures detection rates — analogous to how the thymus educates T-cells by testing them against known antigens.

### 4.2 Components

| Component | File | Function |
|-----------|------|----------|
| ThymicValidationEngine | `engine.py` | Orchestrator: spot checks, sweeps, reports |
| ProbeGenerator | `probe_generator.py` | Replay, mutant, composite probe generation |
| MutationEngine | `mutation_engine.py` | 8 encoding/synonym/structural mutations |
| AttackProfileLibrary | `attack_profile_library.py` | 5-tier probe corpus management |
| LayerProbeRouter | `layer_probe_router.py` | ASGI transport routing via httpx |

### 4.3 Probe Tiers

| Tier | Name | Expected Result | Purpose |
|------|------|-----------------|---------|
| 1 | Known Attacks | Block | Known-bad patterns that MUST be caught |
| 2 | Obfuscated Attacks | Block | Encoded/transformed attacks |
| 3 | Paraphrased Attacks | Block | Semantically similar to known attacks |
| 4 | Multi-Turn Attacks | Block | Conversation-based attack sequences |
| 5 | Benign Prompts | Pass | False positive validation |

### 4.4 Validation Types

**Spot Check** (`run_spot_check(probes_per_tier=50)`):
- Quick validation with configurable probes per tier
- Default: 50 probes per tier
- Scheduled: every 15 minutes (`AEGIS_TVE_SPOT_CHECK_INTERVAL_MINUTES`)

**Comprehensive Sweep** (`run_comprehensive_sweep()`):
- Full probe generation across all tiers
- Scheduled: every 6 hours (`AEGIS_TVE_SWEEP_INTERVAL_HOURS`)

**Post-Change** (`run_post_change(changed_layers)`):
- Focused probes for changed layers
- Regression spot check (probes_per_tier=10)

**Stress** (`run_stress_validation(concurrency_multiplier, max_concurrency)`):
- Full sweep with elevated concurrency
- Max concurrency: 50 (`AEGIS_TVE_STRESS_MAX_CONCURRENCY`)

### 4.5 ASGI Transport Routing

Probes are routed through the real AEGIS pipeline using `httpx.AsyncClient` with `ASGITransport`:
- Base URL: `http://aegis-tve-internal`
- Endpoint: `POST /v1/chat/completions`
- TVE probe header: `X-AEGIS-TVE-Probe: {probe_id}:{nonce}`

**Nonce Management**:
- Cryptographic nonce (`secrets.token_hex(32)`) registered before routing
- Validated in `main.py` before processing
- Unregistered after routing (in `finally` block)
- Self-managed nonces for standalone `route_probe()` calls

### 4.6 Probe Isolation

TVE probes are isolated from production:
- **Metric isolation**: TVE probes increment `TVE_PROBES_TOTAL`, NOT `REQUESTS_TOTAL`
- **Audit isolation**: TVE probes skip audit logging
- **Rate limit isolation**: TVE probes skip rate limit counters
- **Quarantine isolation**: TVE probes skip quarantine recording
- **Taxonomy isolation**: TVE probes skip jailbreak taxonomy
- **Distillation isolation**: TVE probes skip distillation recording
- **Synthetic response**: Valid TVE probes receive synthetic 200 response (never forwarded to upstream model)

The `_is_tve_probe` flag is computed early in `_process_request()` by checking the TVE header for a valid registered nonce.

### 4.7 Reports

**ValidationReport**:
- `run_id`: Unique run identifier
- `run_type`: "spot_check", "comprehensive", "post_change", "stress"
- `timestamp`: ISO-8601
- `duration_seconds`: Wall-clock time
- `total_probes`: Count of probes sent
- `probes_detected`: True positives (attacks correctly blocked)
- `probes_missed`: False negatives (attacks that passed)
- `false_positives`: Benign probes incorrectly blocked
- `overall_tpr`: True Positive Rate
- `overall_fpr`: False Positive Rate
- `per_layer_results`: Per-layer TPR, FPR, avg/p95 latency
- `per_tier_results`: Per-tier detection metrics

**HealthSummary**:
- Last run metadata
- Overall TPR/FPR
- Per-layer TPR/FPR
- Verdict passed/failed
- Recommended actions

### 4.8 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tve_enabled` | true | Enable TVE |
| `tve_spot_check_probes_per_tier` | 50 | Probes per tier for spot checks |
| `tve_concurrency` | 5 | Concurrent probe routing |
| `tve_spot_check_interval_minutes` | 15 | Spot check schedule |
| `tve_sweep_interval_hours` | 6 | Comprehensive sweep schedule |
| `tve_adaptive_decay_threshold` | 30 | Decay threshold |
| `tve_stress_max_concurrency` | 50 | Max stress test concurrency |

---

## 5. Configuration Reference

### 5.1 Environment Variables

All configuration via environment variables prefixed `AEGIS_` or via `AegisConfig` in `config.py`.

#### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_HOST` | "0.0.0.0" | Bind address |
| `AEGIS_PORT` | 8000 | Bind port |
| `AEGIS_DEBUG` | false | Debug mode |
| `AEGIS_LOG_LEVEL` | "INFO" | Logging level |
| `AEGIS_UPSTREAM_URL` | "https://api.openai.com" | Upstream LLM provider |
| `AEGIS_UPSTREAM_API_KEY` | "" | Upstream API key |
| `AEGIS_API_KEY` | "" | Gateway API key |
| `AEGIS_MODEL` | "" | Model name for upstream |
| `AEGIS_SKIP_MODEL_LOAD` | false | Skip ML model loading (tests/CI) |

#### Backing Services

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | "redis://localhost:6379/0" | Redis connection (enables Redis rate limiting + event bus) |
| `DATABASE_URL` | "postgresql://aegis:aegis@localhost:5432/aegis" | PostgreSQL connection |
| `AEGIS_EVENT_BUS_TYPE` | "memory" | "memory" or "redis" |

#### L1 Barrier

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_MAX_TOKENS_PER_REQUEST` | 128,000 | Max input tokens |
| `AEGIS_RATE_LIMIT_RPM` | 60 | Requests per minute per key |
| `AEGIS_RATE_LIMIT_BURST` | 10 | Burst allowance |
| `AEGIS_MAX_REQUEST_SIZE_BYTES` | 10,485,760 | Max request payload |
| `AEGIS_SESSION_TIMEOUT_SECONDS` | 3,600 | Session expiry |

#### L2 Innate

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_BLOCK_THRESHOLD` | 0.85 | Confidence threshold for blocking |
| `AEGIS_ALERT_THRESHOLD` | 0.50 | Confidence threshold for alerting |
| `AEGIS_CANARY_SECRET_KEY` | "aegis-canary-default-secret-change-in-production" | Canary HMAC secret |

#### L3 Adaptive

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_SIMILARITY_THRESHOLD` | 0.85 | Semantic similarity threshold |
| `AEGIS_BEHAVIORAL_PSI_THRESHOLD` | 0.25 | Behavioral drift threshold |
| `AEGIS_MCAV_ANOMALY_THRESHOLD` | 0.7 | DCA anomaly threshold |
| `AEGIS_MCAV_THREAT_THRESHOLD` | 0.9 | DCA definitive threat threshold |
| `AEGIS_MULTI_TURN_WINDOW` | 10 | Multi-turn analysis window |

#### L4 Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_FAISS_DIMENSION` | 384 | Vector dimensionality |
| `AEGIS_FAISS_HNSW_M` | 32 | HNSW connectivity |
| `AEGIS_FAISS_EF_SEARCH` | 64 | HNSW search parameter |
| `AEGIS_ACUTE_MEMORY_DAYS` | 30 | Acute phase duration |
| `AEGIS_DORMANT_MEMORY_DAYS` | 180 | Days before dormancy |
| `AEGIS_DP_EPSILON` | 3.0 | Differential privacy epsilon |
| `AEGIS_PROMOTE_MIN_SOURCES` | 3 | Sources needed for promotion |

#### L5 Output

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_PII_REDACTION_THRESHOLD` | 0.5 | PII redaction confidence |
| `AEGIS_PII_ALERT_THRESHOLD` | 0.4 | PII alert confidence |
| `AEGIS_STREAMING_WINDOW_SIZE` | 128 | Streaming window tokens |
| `AEGIS_STREAMING_WINDOW_OVERLAP` | 0.5 | Streaming window overlap |
| `AEGIS_LEAKAGE_NGRAM_SIZE` | 4 | Leakage detection n-gram size |
| `AEGIS_LEAKAGE_OVERLAP_THRESHOLD` | 0.3 | Leakage overlap threshold |

#### L6 Policy

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_POLICY_BACKEND` | "python" | "python" or "opa" |
| `AEGIS_POLICY_OPA_URL` | "http://localhost:8181" | OPA server URL |
| `AEGIS_TLI_AUTO_DECAY_ENABLED` | true | Enable TLI auto-decay |

#### L7 Healing

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_CIRCUIT_BREAKER_THRESHOLD` | 0.50 | Error rate threshold |
| `AEGIS_CIRCUIT_BREAKER_WINDOW_SECONDS` | 60 | Error tracking window |
| `AEGIS_COOLDOWN_SECONDS` | 30 | Initial cooldown |
| `AEGIS_MAX_COOLDOWN_SECONDS` | 300 | Max cooldown |
| `AEGIS_PROBE_COUNT` | 5 | Probes in half-open state |
| `AEGIS_QUARANTINE_THRESHOLD` | 3 | Strikes before quarantine |

#### L8 Federated

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_FEDERATED_ENABLED` | false | Enable federation |
| `AEGIS_FEDERATED_DP_EPSILON` | 3.0 | DP epsilon |
| `AEGIS_FEDERATED_MODEL_UPDATE_INTERVAL_HOURS` | 6 | Round interval |
| `AEGIS_FEDERATED_INDICATOR_SHARE_INTERVAL_MINUTES` | 15 | Indicator share interval |

#### MTMD (Multi-Turn Manipulation Detection)

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_MANIPULATION_DETECTION_ENABLED` | true | Enable MTMD |
| `AEGIS_MTMD_WINDOW_SIZE` | 20 | Analysis window |
| `AEGIS_MTMD_MAX_HISTORY` | 100 | Max turns per source |
| `AEGIS_MTMD_BLOCK_THRESHOLD` | 0.7 | Block threshold |
| `AEGIS_MTMD_ALERT_THRESHOLD` | 0.4 | Alert threshold |

#### Distillation Defense

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_DISTILLATION_DEFENSE_ENABLED` | true | Enable distillation defense |
| `AEGIS_DISTILLATION_WINDOW_HOURS` | 24.0 | Analysis window |
| `AEGIS_DISTILLATION_MAX_HISTORY` | 10,000 | Max queries per source |

#### Source Profiler

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_PROFILER_ENABLED` | true | Enable profiling |
| `AEGIS_PROFILER_MAX_PROFILES` | 10,000 | Max source profiles |

#### Adaptive Rate Limiter

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_ADAPTIVE_RATE_LIMIT_ENABLED` | true | Enable adaptive rate limiting |
| `AEGIS_ADAPTIVE_RATE_LIMIT_HARD_STOP_THRESHOLD` | 5 | Attempts before hard stop |
| `AEGIS_ADAPTIVE_RATE_LIMIT_HARD_STOP_DURATION` | 300 | Hard stop seconds |

#### Tool Proxy

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_TOOL_PROXY_ENABLED` | true | Enable TIP |
| `AEGIS_TOOL_PROXY_DEFAULT_RPM` | 60 | Per-tool rate limit |
| `AEGIS_TOOL_PROXY_MAX_PARAM_SIZE` | 10,000 | Max param bytes |
| `AEGIS_TOOL_PROXY_BLOCK_INTERNAL_URLS` | true | Block private IPs |
| `AEGIS_TDIV_ENABLED` | true | Enable TDIV |
| `AEGIS_TDIV_MAX_DESCRIPTION_LENGTH` | 2,000 | Max description length |
| `AEGIS_TRS_ENABLED` | true | Enable TRS |
| `AEGIS_TRS_DEFAULT_MAX_OUTPUT_SIZE` | 102,400 | Max output bytes |
| `AEGIS_TRS_BLOCK_THRESHOLD` | 0.85 | Injection block threshold |
| `AEGIS_TCAD_ENABLED` | true | Enable TCAD |
| `AEGIS_TCAD_WINDOW_SIZE` | 20 | Chain window size |
| `AEGIS_TCAD_VOLUME_MULTIPLIER` | 3.0 | Volume anomaly threshold |

#### Agent Security

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_IACM_ENABLED` | true | Enable IACM |
| `AEGIS_IACM_TRUST_THRESHOLD` | 0.3 | Trust threshold |
| `AEGIS_IACM_INJECTION_RATE_THRESHOLD` | 0.5 | Compromised agent threshold |

#### Chain-of-Thought Defense

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_COT_DEFENSE_ENABLED` | true | Enable CoT defense |
| `AEGIS_COT_COHERENCE_WINDOW_SIZE` | 200 | Tokens per window |
| `AEGIS_COT_COHERENCE_PIVOT_THRESHOLD` | 0.3 | Pivot detection threshold |
| `AEGIS_COT_LENGTH_ANOMALY_MULTIPLIER` | 3.0 | Length anomaly multiplier |
| `AEGIS_COT_ALIGNMENT_MISALIGN_THRESHOLD` | 0.3 | Misalignment threshold |
| `AEGIS_COT_DEFENSE_BLOCK_THRESHOLD` | 0.7 | Combined block threshold |

#### Supply Chain Extensions

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_PROVENANCE_VALIDATOR_ENABLED` | true | Enable MPV |
| `AEGIS_PROVENANCE_BLOCK_UNTRUSTED` | false | Block unverified models |
| `AEGIS_SKILL_AUDITOR_ENABLED` | true | Enable SPA |
| `AEGIS_SKILL_AUDITOR_BLOCK_LETHAL_TRIFECTA` | true | Block file+network+exec |
| `AEGIS_DEPENDENCY_ANALYZER_ENABLED` | true | Enable DCA |
| `AEGIS_SUPPLY_CHAIN_CACHE_MAX_ENTRIES` | 10,000 | Cache entries |
| `AEGIS_SUPPLY_CHAIN_CACHE_DEFAULT_TTL` | 604,800 | Cache TTL (7 days) |
| `AEGIS_REVALIDATION_ENABLED` | true | Enable re-validation |
| `AEGIS_REVALIDATION_ACTIVE_INTERVAL_HOURS` | 6.0 | Active re-check interval |
| `AEGIS_REVALIDATION_INACTIVE_INTERVAL_HOURS` | 24.0 | Inactive re-check interval |

#### Temporal Defense

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_TEMPORAL_DEFENSE_ENABLED` | true | Enable temporal defense |
| `AEGIS_BBE_ENABLED` | true | Enable BBE |
| `AEGIS_BBE_WARNING_THRESHOLD_SIGMA` | 2.0 | Warning sigma |
| `AEGIS_BBE_CRITICAL_THRESHOLD_SIGMA` | 3.0 | Critical sigma |
| `AEGIS_BBE_PRODUCTION_TRANSITION_COUNT` | 500 | Transition to production |
| `AEGIS_BBE_MAX_BASELINES` | 10,000 | Max baselines |
| `AEGIS_CANARY_INJECTION_ENABLED` | true | Enable canary injection |
| `AEGIS_CANARY_PROFILE` | "general_enterprise" | Canary profile |
| `AEGIS_CANARY_INJECTIONS_PER_HOUR` | 6.0 | Target injection rate |
| `AEGIS_CANARY_KEYWORD_PASS_THRESHOLD` | 0.8 | Keyword pass rate |
| `AEGIS_CANARY_CONSECUTIVE_FAIL_CRITICAL` | 3 | Consecutive fail alert |
| `AEGIS_MPR_ENABLED` | true | Enable provenance registry |
| `AEGIS_MPR_MAX_RECORDS` | 100,000 | Max records |
| `AEGIS_TCE_ENABLED` | true | Enable correlation engine |
| `AEGIS_TCE_DEFAULT_CORRELATION_WINDOW_HOURS` | 72.0 | Correlation lookback |
| `AEGIS_TCE_AUTO_CORRELATE_ON_CRITICAL` | true | Auto-correlate on critical |
| `AEGIS_SNAPSHOT_ENABLED` | true | Enable snapshots |
| `AEGIS_SNAPSHOT_MAX_COUNT` | 90 | Max snapshots |
| `AEGIS_SNAPSHOT_INTERVAL_HOURS` | 24.0 | Snapshot interval |
| `AEGIS_DATE_SCANNER_ENABLED` | true | Enable date scanner |

#### TVE (Thymic Validation Engine)

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_TVE_ENABLED` | true | Enable TVE |
| `AEGIS_TVE_SPOT_CHECK_PROBES_PER_TIER` | 50 | Probes per tier |
| `AEGIS_TVE_CONCURRENCY` | 5 | Concurrent probes |
| `AEGIS_TVE_SPOT_CHECK_INTERVAL_MINUTES` | 15 | Spot check interval |
| `AEGIS_TVE_SWEEP_INTERVAL_HOURS` | 6 | Sweep interval |
| `AEGIS_TVE_STRESS_MAX_CONCURRENCY` | 50 | Max stress concurrency |

---

## 6. Event Bus Architecture

### 6.1 Implementation

| Environment | Implementation | Details |
|-------------|---------------|---------|
| Development/Testing | `InMemoryEventBus` | `asyncio.Queue` per channel |
| Production | `RedisEventBus` | Redis Streams, consumer groups |

Configured via `AEGIS_EVENT_BUS_TYPE` ("memory" or "redis").

### 6.2 Channels

| Channel | Constant | Publishers | Subscribers |
|---------|----------|-----------|-------------|
| `threat_detected` | `CHANNEL_THREAT_DETECTED` | L2, L3, L5, MTMD | Policy, Dashboard, Audit |
| `policy_escalation` | `CHANNEL_POLICY_ESCALATION` | L6 | Dashboard, Audit, Alert |
| `circuit_breaker` | `CHANNEL_CIRCUIT_BREAKER` | L7 | Dashboard, Recovery |
| `antibody_generated` | `CHANNEL_ANTIBODY_GENERATED` | L3 (antibody callback) | Dashboard, Audit |
| `audit_event` | `CHANNEL_AUDIT_EVENT` | All layers | Audit Logger, Dashboard |
| `tool_violation` | `CHANNEL_TOOL_VIOLATION` | TIP, TIPE, TCAD | Dashboard, Audit, Policy |
| `temporal_drift` | `CHANNEL_TEMPORAL_DRIFT` | BBE, Canary, TCE | Dashboard, Policy, Alert |

### 6.3 Event Format

Events are JSON-serializable dictionaries with minimum fields:
- `event_type`: Channel name
- `timestamp`: ISO-8601
- `source`: Publishing component
- `data`: Event-specific payload

### 6.4 Redis Streams Details

- Stream key format: `aegis:events:{channel_name}`
- Consumer group: `aegis-workers`
- Max stream length: Configurable per channel
- Persistence: AOF-enabled Redis

---

## 7. API Reference

### 7.1 Core Proxy Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/chat/completions` | Yes | Main proxy — OpenAI-compatible |
| GET | `/v1/models` | No | List available models |

### 7.2 Health and Observability

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No/Yes | Unauthenticated: basic. Authenticated: full layer status |
| GET | `/metrics` | Yes | Prometheus metrics |
| GET | `/v1/admin/deep-health` | Yes | 6-component deep health check |

### 7.3 Audit and Intelligence

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/audit/recent` | Yes | Recent audit records |
| GET | `/v1/vault/stats` | Yes | Threat vault statistics |
| POST | `/v1/threat-intel/ingest` | Yes | Ingest STIX 2.1 threat bundle |
| GET | `/v1/threat-intel/export` | Yes | Export indicators with DP noise |

### 7.4 Supply Chain

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/supply-chain/verify` | Yes | 4-stage model verification |
| POST | `/v1/supply-chain/validate-provenance` | Yes | MPV: 5-check provenance validation |
| POST | `/v1/supply-chain/audit-skill` | Yes | SPA: skill/plugin audit (4 checks) |
| POST | `/v1/supply-chain/analyze-dependencies` | Yes | DCA: dependency chain analysis |
| POST | `/v1/supply-chain/revalidate` | Yes | Manual re-validation trigger |
| GET | `/v1/supply-chain/revalidation/status` | Yes | Re-validation scheduler status |

### 7.5 Agent Security

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/agents/register` | Yes | Register agent, return JWT |
| POST | `/v1/agents/{id}/authorize` | Yes | Check agent tool authorization |
| POST | `/v1/agents/message/validate` | Yes | Validate inter-agent message |
| GET | `/v1/agents/communication/stats` | Yes | IACM stats |

### 7.6 Policy and Compliance

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/compliance/matrix` | Yes | Cross-framework coverage matrix |
| POST | `/v1/compliance/report` | Yes | Generate compliance report |
| GET | `/v1/taxonomy/stats` | Yes | Jailbreak taxonomy statistics |

### 7.7 Federated Intelligence

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/federated/round` | Yes | Trigger federated learning round |

### 7.8 Temporal Defense

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/temporal/baseline/status` | Yes | BBE status and drift alerts |
| GET | `/v1/temporal/canary/status` | Yes | Canary pass rate and alerts |
| POST | `/v1/temporal/canary/inject` | Yes | Manual canary injection |
| GET | `/v1/temporal/provenance/timeline` | Yes | MPR timeline (filterable) |
| GET | `/v1/temporal/provenance/stats` | Yes | MPR statistics |
| POST | `/v1/temporal/correlate` | Yes | TCE: correlate anomaly |
| GET | `/v1/temporal/correlation/reports` | Yes | TCE: recent reports |
| POST | `/v1/temporal/snapshot` | Yes | Capture state snapshot |
| GET | `/v1/temporal/snapshots` | Yes | List recent snapshots |
| POST | `/v1/temporal/replay` | Yes | Replay comparison |
| GET | `/v1/temporal/date-scanner/status` | Yes | Date scanner status |

### 7.9 Tool Proxy

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/tool-proxy/stats` | Yes | Invocation and violation stats |
| POST | `/v1/tool-proxy/validate-description` | Yes | TDIV: validate tool description |

### 7.10 Administration

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/admin/backup` | Yes | Backup vault + signatures |
| POST | `/v1/admin/restore` | Yes | Restore from backup |

### 7.11 Dashboard

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/dashboard` | No | Production dashboard HTML |
| GET | `/dashboard/api/overview` | Yes | System overview JSON |
| GET | `/dashboard/api/detections` | Yes | Recent detection events |
| GET | `/dashboard/api/campaigns` | Yes | Campaign alerts |
| GET | `/dashboard/api/compliance` | Yes | Compliance coverage |
| GET | `/dashboard/api/metrics/timeseries` | Yes | Time-bucketed metrics |
| GET | `/dashboard/api/federation` | Yes | Federation details |
| GET | `/dashboard/events` | Yes | SSE real-time event stream |

---

## 8. Metrics Reference

### 8.1 Counters

| Metric | Labels | Description |
|--------|--------|-------------|
| `aegis_requests_total` | layer, status | Total requests processed |
| `aegis_tve_probes_total` | run_type | TVE probe requests (isolated from production) |
| `aegis_tenant_requests` | tenant_id | Per-tenant request count |
| `aegis_threats_detected` | layer, category | Threats detected per layer |
| `aegis_policy_decisions` | decision | Policy decisions (allow/block/escalate) |
| `aegis_blocks_total` | layer, reason | Blocked requests |
| `aegis_circuit_breaker_trips` | endpoint | Circuit breaker trips |
| `aegis_pii_redactions` | entity_type | PII entities redacted |
| `aegis_toxicity_detections` | category | Toxicity detections |
| `aegis_output_cascade_stage` | stage, result | Output cascade stage results |
| `aegis_stream_interruptions` | reason | Streaming interruptions |
| `aegis_stream_windows_evaluated` | — | Streaming windows evaluated |
| `aegis_stream_tokens_processed` | — | Streaming tokens processed |
| `aegis_rate_limit_triggers` | — | Rate limit triggers |
| `aegis_antibody_generations` | — | Antibody generation events |
| `aegis_event_bus_events` | channel | Events published per channel |
| `aegis_upstream_errors` | error_type | Upstream provider errors |
| `aegis_multimodal_images_scanned` | — | Images scanned |
| `aegis_multimodal_ocr_text_extracted` | — | OCR text extractions |
| `aegis_multimodal_image_threats` | — | Image threats detected |
| `aegis_multimodal_documents_scanned` | — | Documents scanned |
| `aegis_multimodal_hidden_content_detected` | — | Hidden content found |
| `aegis_multimodal_document_threats` | — | Document threats detected |
| `aegis_multimodal_audio_scanned` | — | Audio files scanned |
| `aegis_multimodal_audio_threats` | — | Audio threats detected |
| `aegis_cross_modal_laundering_detected` | — | Cross-modal laundering |
| `aegis_cross_modal_inconsistency` | — | Cross-modal inconsistency |
| `aegis_tool_definition_threats` | — | Tool definition threats |
| `aegis_tool_chain_anomalies` | — | Tool chain anomalies |
| `aegis_tool_invocations_total` | — | Total tool invocations |
| `aegis_tool_policy_violations_total` | — | Tool policy violations |
| `aegis_distillation_signals` | strategy | Distillation signals |
| `aegis_distillation_alerts` | — | Distillation alerts |
| `aegis_distillation_blocks` | — | Distillation blocks |
| `aegis_reasoning_traces_detected` | — | Reasoning traces found |
| `aegis_governance_disclosures_detected` | — | Governance disclosures |
| `aegis_cot_defense_signals` | — | CoT defense signals |
| `aegis_cot_defense_blocks` | — | CoT defense blocks |
| `aegis_manipulation_signals` | strategy | Manipulation signals |
| `aegis_manipulation_blocks` | — | Manipulation blocks |
| `aegis_jailbreak_attempts` | technique | Jailbreak attempts by technique |
| `aegis_interagent_messages_scanned` | — | Inter-agent messages scanned |
| `aegis_interagent_injection_detected` | — | Inter-agent injection detected |
| `aegis_compromised_agents_detected` | — | Compromised agents flagged |
| `aegis_drift_alerts_total` | severity | Drift alerts |
| `aegis_baseline_interactions_total` | — | BBE interactions tracked |
| `aegis_canary_injections_total` | — | Canary injections |
| `aegis_canary_passes_total` | — | Canary passes |
| `aegis_canary_failures_total` | — | Canary failures |
| `aegis_canary_alerts_total` | — | Canary alerts |
| `aegis_provenance_events_total` | category | Provenance events |
| `aegis_correlations_total` | — | Correlation reports |
| `aegis_snapshots_total` | — | State snapshots |
| `aegis_replay_comparisons_total` | — | Replay comparisons |
| `aegis_date_boundary_checks_total` | — | Date boundary checks |
| `aegis_date_boundary_alerts_total` | — | Date boundary alerts |
| `aegis_provenance_validations` | — | Provenance validations |
| `aegis_skill_audits` | — | Skill audits |
| `aegis_supply_chain_rejections` | — | Supply chain rejections |
| `aegis_dependency_analysis_findings` | severity | Dependency findings |
| `aegis_revalidation_runs` | — | Re-validation runs |
| `aegis_revalidation_status_changes` | — | Status changes |
| `aegis_adaptive_rate_slowdowns` | — | Adaptive rate slowdowns |
| `aegis_adaptive_rate_hard_stops` | — | Adaptive hard stops |

### 8.2 Gauges

| Metric | Labels | Description |
|--------|--------|-------------|
| `aegis_active_connections` | — | Active concurrent connections |
| `aegis_threat_level` | — | Current TLI level (1-5) |
| `aegis_circuit_breaker_state` | endpoint | Breaker state (0=closed, 1=open, 2=half-open) |
| `aegis_vault_size` | — | Threat vault indicator count |
| `aegis_quarantined_sessions` | — | Quarantined session count |
| `aegis_source_risk_level` | — | Source risk level |
| `aegis_jailbreak_techniques_active` | — | Active jailbreak techniques |
| `aegis_baseline_tier` | — | BBE calibration tier |
| `aegis_adaptive_rate_cooling` | — | Sources in cooling state |
| `aegis_revalidation_last_run` | — | Last re-validation timestamp |

### 8.3 Histograms

| Metric | Labels | Buckets | Description |
|--------|--------|---------|-------------|
| `aegis_request_latency` | — | Default | End-to-end request latency |
| `aegis_layer_latency` | layer | Default | Per-layer processing latency |
| `aegis_innate_scanner_latency` | scanner | Default | Per-scanner L2 latency |
| `aegis_adaptive_analyzer_latency` | analyzer | Default | Per-analyzer L3 latency |
| `aegis_upstream_latency` | — | Default | Upstream model latency |
| `aegis_multimodal_scan_latency` | modality | Default | Multimodal scan latency |
| `aegis_multimodal_audio_transcription_latency` | — | Default | Audio transcription latency |
| `aegis_tool_proxy_latency` | — | Default | Tool proxy latency |
| `aegis_reasoning_trace_length` | — | Default | Reasoning trace token length |
| `aegis_correlation_latency` | — | Default | TCE correlation latency |

### 8.4 Info

| Metric | Labels | Description |
|--------|--------|-------------|
| `aegis_build_info` | version | Build version info |

---

## 9. Deployment Guide

### 9.1 Quick Start (Development)

```bash
git clone https://github.com/[your-org]/aegis.git && cd aegis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download ML models (~500MB, first run only)
python -c "from transformers import pipeline; pipeline('text-classification', 'ProtectAI/deberta-v3-base-prompt-injection-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Configure
export AEGIS_UPSTREAM_URL=https://api.openai.com
export AEGIS_UPSTREAM_API_KEY=sk-your-key-here
export AEGIS_API_KEY=aegis-your-gateway-key

# Launch
uvicorn aegis.main:app --host 0.0.0.0 --port 8000
```

### 9.2 Docker Deployment

#### Dockerfile

Multi-stage build:

**Stage 1 (builder)**:
- Base: `python:3.12-slim`
- System deps: `build-essential`, `libgomp1`, `tesseract-ocr` + 7 language packs (chi-sim, chi-tra, jpn, kor, ara, hin, rus)
- Virtual env: `/opt/venv`
- ML models downloaded at build time (no runtime network):
  - DeBERTa-v3-base prompt injection classifier (~400MB)
  - all-MiniLM-L6-v2 sentence embeddings (~80MB)
- Model cache: `/opt/models`

**Stage 2 (runtime)**:
- Base: `python:3.12-slim`
- Non-root user: `aegis` (uid 1000, gid 1000)
- PYTHONPATH: `/app`
- Working dir: `/app/aegis`
- Health check: `curl -sf http://localhost:8000/health` (60s start period, 30s interval, 5s timeout, 3 retries)
- Default CMD: `uvicorn aegis.main:app --host 0.0.0.0 --port 8000 --workers 1`

#### docker-compose.yml

Four services:

| Service | Image | Port | Memory | Purpose |
|---------|-------|------|--------|---------|
| aegis | Built from Dockerfile | 8000 | 4GB | Main AEGIS service |
| redis | redis:7-alpine | 6379 | 256MB max | Session/cache/event bus |
| postgres | postgres:16-alpine | 5432 | — | State database |
| opa | openpolicyagent/opa | 8181 | — | Policy engine |

Redis configuration:
- Max memory: 256MB
- Eviction policy: allkeys-lru
- AOF persistence enabled

PostgreSQL:
- User: `aegis`
- Password: `aegis_secret` (CHANGE IN PRODUCTION)
- Database: `aegis`

Default environment:
- `AEGIS_UPSTREAM_URL=http://host.docker.internal:11434` (Ollama)
- `AEGIS_MODEL=llama3.2:3b`
- `REDIS_URL=redis://redis:6379/0`
- `DATABASE_URL=postgresql+asyncpg://aegis:aegis_secret@postgres:5432/aegis`

```bash
docker compose up -d              # Start full stack
docker compose logs -f aegis      # Follow AEGIS logs
docker compose down -v            # Stop and clean up
```

#### docker-entrypoint.sh

Startup sequence:
1. Wait for Redis availability
2. Wait for PostgreSQL availability
3. Run Alembic migrations (database schema)
4. Print startup info (versions, endpoints, model status)
5. Execute CMD (uvicorn)

### 9.3 Key File Paths (Container)

| Path | Content |
|------|---------|
| `/app/aegis/` | Application code |
| `/opt/models/` | ML models (DeBERTa, MiniLM) |
| `/app/aegis/logs/` | Audit logs (JSONL) |
| `/app/aegis/data/` | Patterns, blocklists, seeds |

### 9.4 Production Checklist

- [ ] Change `AEGIS_API_KEY` from default
- [ ] Change `AEGIS_CANARY_SECRET_KEY` from default
- [ ] Change PostgreSQL password from `aegis_secret`
- [ ] Enable Redis authentication
- [ ] Configure TLS termination (min TLS 1.3)
- [ ] Set `AEGIS_DEBUG=false`
- [ ] Configure proper `AEGIS_UPSTREAM_URL` and `AEGIS_UPSTREAM_API_KEY`
- [ ] Set appropriate rate limits per tenant
- [ ] Configure fallback models (`AEGIS_FALLBACK_MODELS`)
- [ ] Enable multimodal scanning if needed (`AEGIS_MULTIMODAL_ENABLED=true`)
- [ ] Review TLI auto-decay settings
- [ ] Configure backup schedule for vault and signatures
- [ ] Set up Prometheus scraping for `/metrics`
- [ ] Configure Grafana dashboards
- [ ] Review and customize OPA policies

### 9.5 Backing Service Resilience

AEGIS degrades gracefully when backing services are unavailable:

| Service | Fallback | Impact |
|---------|----------|--------|
| Redis | In-memory rate limiting, InMemoryEventBus | No cross-instance state sharing |
| PostgreSQL | In-memory audit buffer, no persistent audit | Audit logs lost on restart |
| OPA | Python policy engine fallback | Full functionality preserved |

**AEGIS must NEVER crash because a backing service is down.**

### 9.6 Running Tests

```bash
# Full suite
python3 -m pytest tests/ -x -q --tb=short -p no:warnings --ignore=tests/benchmark

# Benchmark (requires DeBERTa)
python3 -m pytest tests/benchmark/ -q --tb=short -p no:warnings

# Single file
python3 -m pytest tests/test_attack_battery.py -x -q --tb=short -p no:warnings

# Stress tests (requires live services)
AEGIS_STRESS_FULL=1 python3 -m pytest tests/stress/test_stress.py -v --tb=short

# APT campaigns
python3 -m red_team.run_red_team
```

---

## 10. Incident Response Playbook

### 10.1 TLI Escalation Response

#### GREEN → BLUE (Elevated Detection Rate)

**Trigger**: Detection rate >2× baseline.

**Response**:
1. Check `/dashboard/api/detections` for pattern in recent detections
2. Review `aegis_threats_detected` metric by category
3. Check if single source or distributed
4. Auto-decay: 60 seconds → GREEN if no further escalation

#### BLUE → YELLOW (Confirmed Novel Attack)

**Trigger**: Confirmed novel attack or sustained elevated rate.

**Response**:
1. Review antibody generation events (`aegis_antibody_generations`)
2. Check `aegis_blocks_total` by layer — which layers are catching it?
3. Verify adaptive path is running (DeBERTa loaded, FAISS accessible)
4. Review clonal selection — were new signatures promoted?
5. Auto-decay: 120 seconds → BLUE

#### YELLOW → ORANGE (Multi-Tenant Campaign)

**Trigger**: Active campaign across multiple tenants.

**Response**:
1. Check `/dashboard/api/campaigns` for active campaigns
2. Identify affected tenants via `aegis_tenant_requests`
3. Review TCE correlation reports (`/v1/temporal/correlation/reports`)
4. Consider per-tenant rate limit reduction
5. Notify affected design partners
6. Auto-decay: 180 seconds → YELLOW

#### ORANGE → RED (Critical Threat / Zero-Day)

**Trigger**: Critical infrastructure threat or zero-day exploit.

**Response**:
1. **All requests blocked** (fail-closed mode)
2. Check circuit breaker states (`aegis_circuit_breaker_state`)
3. Review audit logs for impact assessment
4. Check supply chain verification status
5. Run TVE comprehensive sweep (`/v1/tve/sweep`)
6. Human clearance required to de-escalate
7. Auto-decay: 300 seconds → ORANGE (but manual review strongly recommended)

### 10.2 Circuit Breaker Trip Response

1. Check `aegis_circuit_breaker_trips` for which endpoint tripped
2. Verify fallback model is serving (`aegis_circuit_breaker_state` = 1 = OPEN)
3. Monitor upstream provider status
4. After cooldown, check probe results in half-open state
5. If probes pass, breaker auto-closes
6. If probes fail, cooldown doubles (max 300s)

### 10.3 False Positive Investigation

1. Check `aegis_blocks_total` with `reason` label
2. Pull specific request from audit logs (`/v1/audit/recent`)
3. Identify which scanner/analyzer triggered
4. If L2 regex: check pattern ID, consider adding to allowlist
5. If L3 DeBERTa: check confidence score, verify it's ≥0.90 single-analyzer override
6. If legitimate traffic: add tolerance training data via operator flagging

### 10.4 Vault Corruption Recovery

1. Stop AEGIS instance
2. Restore from backup (`POST /v1/admin/restore`)
3. Verify vault stats (`GET /v1/vault/stats`)
4. Run TVE spot check to validate detection capabilities
5. Check PostgreSQL for backup indicators (if configured)

### 10.5 DeBERTa Model Failure

1. Check `AEGIS_SKIP_MODEL_LOAD` — is model loaded?
2. Verify ONNX runtime status
3. L2 innate still provides baseline protection (95.45% TPR without DeBERTa)
4. L3 semantic search still functions (FAISS-based)
5. Restart with fresh model download if corrupted

---

## 11. Backup and Recovery

### 11.1 Backup Components

| Component | Files | Backup Method |
|-----------|-------|---------------|
| Threat Vault (FAISS) | `data/threat_vault.faiss`, `data/threat_vault_meta.json` | `POST /v1/admin/backup` |
| Signatures | In-memory (persisted to JSON) | Included in backup |
| Audit Logs | `aegis_audit.jsonl` | File copy + PostgreSQL dump |
| Pattern Library | `data/patterns.json` | Git-tracked |
| Blocklist | `data/blocklist.txt` | Git-tracked |
| PostgreSQL | All tables | `pg_dump` |
| Redis | Sorted sets, streams | Redis AOF / RDB snapshot |

### 11.2 Backup API

```bash
# Create backup
curl -X POST http://localhost:8000/v1/admin/backup \
  -H "Authorization: Bearer $AEGIS_API_KEY"

# Restore from backup
curl -X POST http://localhost:8000/v1/admin/restore \
  -H "Authorization: Bearer $AEGIS_API_KEY"
```

### 11.3 PostgreSQL Backup

Five tables:
1. `tenants` — Multi-tenant configuration
2. `audit_log` — Audit trail (compliance-critical)
3. `threat_indicators` — Threat vault persistence
4. `signatures` — Generated detection signatures
5. `circuit_breaker_events` — Recovery telemetry

```bash
pg_dump -h localhost -U aegis -d aegis > aegis_backup_$(date +%Y%m%d).sql
```

### 11.4 Recovery Procedures

**Full Recovery**:
1. Deploy fresh AEGIS instance
2. Restore PostgreSQL from dump
3. Start AEGIS — vault loads from PostgreSQL on startup
4. Restore FAISS index from backup if available (faster than PostgreSQL rebuild)
5. Run TVE spot check to validate detection capabilities
6. Verify `/health` returns all layers healthy

**Vault-Only Recovery**:
1. `POST /v1/admin/restore` — loads FAISS + metadata from backup files
2. Verify with `GET /v1/vault/stats`
3. Run TVE spot check

### 11.5 Audit Log Retention

- **JSONL file**: `aegis_audit.jsonl`, crash-safe flush after every write
- **PostgreSQL**: `audit_log` table (when configured)
- **Retention**: Configurable per compliance requirements
- **Format**: One JSON object per line with timestamp, event type, tenant, details

---

## 12. Red Team History and Findings

### 12.1 Summary

AEGIS has undergone extensive adversarial testing across multiple phases:

| Phase | Date | Focus | Findings |
|-------|------|-------|----------|
| Phase 1 | 2026-03-20 | White-box hardening | 12 bypasses found and fixed |
| Phase 2 | 2026-03-21 | APT campaigns (6) | 120 attacks across 6 campaigns |
| Phase 3 | 2026-03-22 | Adversarial ML algorithms | TextFooler, CharSwap, Evolutionary, etc. |
| Phase 4 | 2026-03-23 | Extended campaigns (6) | 700+ requests, SHAPESHIFTER through FULL SPECTRUM |
| Phase 5 | 2026-03-24 | Infrastructure security | 8 attack modules, auth/rate limit/DoS/federation |
| Phase 6 | 2026-03-25 | Adaptive meta-learner | 12 evasion fingerprints, 10-round co-evolution |
| CHIMERA | 2026-03-26 | Multimodal APT | 20 attacks, 95% detection in Docker |
| TVE Red Team | 2026-03-30 | TVE self-validation | 33 attack vectors, 7 categories, 1 finding |

### 12.2 Phase 1 — White-Box Hardening (12 Findings)

| ID | Finding | Fix |
|----|---------|-----|
| RT-001 | Base64 recursion depth too shallow | Increased from 3 to 5 iterations |
| RT-002 | Unicode whitespace not fully normalized | Added Unicode-aware whitespace handling |
| RT-003 | Single-size n-gram PII detection | Multi-size n-gram (2-gram fallback for short text) |
| RT-004 | Streaming spike not detected | Added spike detection for streaming windows |
| RT-005 | Fixed temporal clustering min_agents | Made configurable |
| RT-006 | Fixed intent classifier cardinality | Made configurable |
| RT-007 | ROT13 not decoded | Added ROT13 with keyword gating (avoids FPs) |
| RT-008 | PII redaction threshold too high | Lowered 0.7 → 0.5 |
| RT-009 | No source-level quarantine | Added API key/IP tracking to SessionQuarantine |
| RT-010 | No baseline freeze mechanism | Added BBE freeze capability |
| RT-011 | Short prompts evade PII detection | 2-gram fallback for prompts < 20 chars |
| RT-012 | Relative-only target threshold | Added absolute ≥5 distinct targets threshold |

### 12.3 Phase 2 — APT Campaigns

| Campaign | Attacks | Evasion Rate | Outcome |
|----------|---------|-------------|---------|
| PHANTOM NEEDLE | 18 | 77.8% | Extraction blocked at L5 |
| SILENT SIPHON | 13 | 100% | SUCCESS — prompted hardening |
| SLOW BURN | 20 | 50% | Jailbreak blocked at L3 |
| HYDRA | 200 | 83% | Mutations evade regex |
| GHOST PROTOCOL | 7 | 71.4% | Partial evasion |
| CASCADING FAILURE | 12 | 41.7% | Degradation exploitation |

### 12.4 Phase 5 — Infrastructure Security (Key Vulnerabilities)

**High Severity**:
1. Timing side-channel in API key comparison → Fixed with HMAC constant-time compare
2. Gradient poisoning with `min_participants=1` → Documented, requires multi-deployment
3. Self-reported `num_samples` amplification → Documented, requires multi-deployment

**Medium Severity**:
1. No file integrity manifest → Risk accepted (development phase)
2. Indicator poisoning via STIX ingestion → Dedup protections added
3. Global TLI escalation by single attacker → Documented, per-tenant TLI planned
4. Privacy budget race condition → Lock added

**Low Severity**:
1. Unbounded `_session_strikes` dict → LRU eviction added
2. No lock file for concurrent instances → Documented
3. Dev Redis without auth → Production checklist item

### 12.5 TVE Red Team (2026-03-30)

33 attack vectors tested across 7 categories:

| Category | Tests | Findings |
|----------|-------|----------|
| AV-1: Probe Upstream Leakage | 4 | 0 — ASGI transport prevents real upstream calls |
| AV-2: Nonce Forgery | 8 | 0 — Nonces validated, all forgery rejected |
| AV-3: Metric Contamination | 5 | 1 — **Fixed**: TVE probes were contaminating REQUESTS_TOTAL |
| AV-4: Library Exfiltration | 4 | 0 — No probe text in external-facing APIs |
| AV-5: Resource Exhaustion | 5 | 0 — Concurrency caps and FIFO eviction work |
| AV-6: Verdict Manipulation | 3 | 0 — Verdicts reflect current run only |
| AV-7: Scheduler Abuse | 4 | 0 — Task cleanup and idempotency work |

**AV-3.1 Fix (Severity: Medium)**: TVE Tier 1 attack probes caught by L2 innate detection were incrementing `REQUESTS_TOTAL`, writing audit logs, recording quarantine events, and logging to jailbreak taxonomy — all before the TVE interception point. Fixed by computing `_is_tve_probe` flag early in `_process_request()` and branching all four block paths (innate, MTMD, adaptive rate limiter, policy) to use `TVE_PROBES_TOTAL` for probes.

### 12.6 OWASP LLM Top 10 Coverage

| Category | Coverage | Layers |
|----------|----------|--------|
| Prompt Injection | STRONG | L2 + L3 + L4 |
| Output Handling | STRONG | L5 (5-stage cascade) |
| DoS | STRONG | L1 + L2 + L7 |
| Supply Chain | STRONG | Supply chain verifier (4 stages + extensions) |
| Sensitive Information | STRONG | L5 (Presidio + 8 secret detectors) |
| Training Data Poisoning | MODERATE | L8 (DP noise, weight clipping) |
| Plugin/Tool Design | MODERATE | Tool Proxy (TIP/TIPE/TDIV/TRS/TCAD) |
| Excessive Agency | MODERATE | Agent Security Layer |
| Overreliance | MODERATE | Hallucination detection (L5 Stage 3) |
| Model Theft | MODERATE | Distillation defense |

---

## Appendix A — Immune System Mapping Reference

| Immune Component | AEGIS Component | Layer |
|-----------------|-----------------|-------|
| Skin | TLS termination, API gateway | L1 |
| Mucous membranes | Schema validation, input sanitization | L1 |
| Defensins | Blocklist hash lookups | L2 |
| Toll-Like Receptors (TLRs) | Regex pattern engine (188 patterns) | L2 |
| Natural Killer cells | Canary token verifier | L2 |
| Phagocytes | Request quarantine/logging | L7 |
| Inflammatory response | TLI escalation, alert cascade | L6 |
| Helper T-cells (CD4+) | DCA signal fusion | L3 |
| Killer T-cells (CD8+) | Block/terminate/revoke actions | L3, L6 |
| B-cells / Antibodies | Embedding generation, FAISS vectors | L3, L4 |
| Clonal Selection | Signature optimization (regex candidates → affinity test → promote) | L4 |
| Memory B-cells | Threat Vault persistent storage | L4 |
| Regulatory T-cells | Policy engine, FP suppression, TLI | L6 |
| Complement cascade | Output validation (6-stage) | L5 |
| Opsonization | PII/threat tagging with confidence scores | L5 |
| MHC-I molecules | Agent identity certificates (JWT) | Agent Security |
| MHC-II molecules | STIX 2.1 indicator sharing | L8 |
| Wound healing | Circuit breaker, self-healing | L7 |
| Herd immunity | Federated threat intelligence | L8 |
| Danger signals | System telemetry anomalies, DCA | L3 |
| Thymus | Thymic Validation Engine | L9 |
| Immune tolerance | Allowlisting, adaptive thresholds | L6 |

---

## Appendix B — Compliance Mapping Matrix

| AEGIS Capability | NIST AI RMF | ISO 42001 | EU AI Act | CMMC 2.0 | SOC 2 |
|-----------------|-------------|-----------|-----------|----------|-------|
| Real-time threat monitoring | MEASURE 2, MANAGE 4 | Clause 9 | Art. 15 | SI-4, SI-5 | CC7.2 |
| Automated audit trails | GOVERN 1.4 | Clause 9 | Art. 12, 19 | AU-2, AU-6 | CC7.2, CC7.3 |
| Anomaly / drift detection | MEASURE 2.6–2.11 | Clauses 8–9 | Art. 9, 15 | SI-4 | PI1.3 |
| Policy enforcement | GOVERN 1–6 | Clauses 5, 8 | Art. 17 | CM-2, CM-6 | CC1.1 |
| Incident response | MANAGE 1–4 | Clause 10 | Art. 20 | IR-4, IR-5 | CC7.4 |
| Supply chain verification | MAP 3, MANAGE 3 | Clause 8 | Art. 15, 17 | SA-9, SR-3 | CC9.2 |
| Data protection / PII | GOVERN 5 | Clause 8 | Art. 10 | SC-7, SC-8 | CC6.1 |
| Human oversight | GOVERN 1.3 | Clause 5 | Art. 14 | PL-4 | CC1.3 |
| Risk assessment | MAP 1–5 | Clause 6 | Art. 9 | RA-3, RA-5 | CC3.2 |
| Transparency / explainability | MEASURE 2.5 | Clause 9 | Art. 13 | AU-3 | CC2.2 |

**Priority Certifications**: SOC 2 Type II → ISO 27001 + ISO 42001 → FedRAMP → CMMC

**Key Regulatory Deadlines**:
- EU AI Act enforcement: August 2026 (fines up to €35M / 7% global revenue)
- CMMC 2.0 enforceable: November 2025
- FY2026 NDAA Section 1513: DoD AI/ML security framework as CMMC extension

---

## Appendix C — Performance Budget

| Processing Stage | Target Latency | Mode |
|-----------------|---------------|------|
| L1 Barrier | <1ms | Synchronous |
| L2 Innate Detection | <5ms | Synchronous |
| L3 Adaptive Analysis | 10–50ms | Asynchronous |
| L4 Immune Memory | 5–20ms | Sync (fast) / Async (deep) |
| L5 Output Validation | 10–100ms | Sync (streaming) |
| L6 Policy Engine | <1ms | Synchronous |
| L7 Self-Healing | Event-driven | Asynchronous |
| L8 Federated Intel | Batch async | Background |
| L9 Thymic Validation | On-demand | Background |

**Total Gateway Overhead**: 30–150ms typical
**LLM Inference**: 200–400ms TTFT
**AEGIS Overhead**: <50% of LLM latency
**Throughput Target**: 10,000+ RPS per node (100,000+ via K8s autoscaling)
**Availability Target**: 99.95% uptime SLA

---

## Appendix D — Global Variables and Layer Wiring

### D.1 Layer Singletons in main.py

All security layers are initialized as module-level globals in `main.py`:

```python
_barrier: BarrierLayer | None = None
_innate: InnateDetectionLayer | None = None
_adaptive: AdaptiveAnalysisLayer | None = None
_output: OutputValidationLayer | None = None
_policy: PolicyEngine | None = None
_healing: CircuitBreaker | None = None
_vault: ThreatVault | None = None
_signature_store: SignatureStore | None = None
_signature_generator: ClonalSelectionGenerator | None = None
_supply_chain: SupplyChainVerifier | None = None
_config: AegisConfig | None = None
_http_client: httpx.AsyncClient | None = None
_audit: AuditLogger | None = None
_event_bus: EventBus | None = None
_tenant_manager: TenantManager | None = None
_multimodal: MultimodalPreprocessor | None = None
_agent_security: AgentSecurityLayer | None = None
_tool_proxy: ToolInvocationProxy | None = None
_tool_policy: ToolPolicyEngine | None = None
_tdiv: ToolDescriptionValidator | None = None
_trs: ToolResponseSanitizer | None = None
_tcad: ToolChainAnomalyDetector | None = None
_iacm: InterAgentCommunicationMonitor | None = None
_distillation: DistillationDefenseAnalyzer | None = None
_manipulation_detector: MultiTurnManipulationDetector | None = None
_profiler: SourceProfiler | None = None
_adaptive_rate_limiter: AdaptiveRateLimiter | None = None
_jailbreak_taxonomy: JailbreakTaxonomy | None = None
_compliance: ComplianceEngine | None = None
_federated: FederatedIntelligenceManager | None = None
_quarantine: SessionQuarantine | None = None
_reasoning_sanitizer: ReasoningSanitizer | None = None
_coherence_analyzer: CoherenceAnalyzer | None = None
_length_anomaly: LengthAnomalyDetector | None = None
_alignment_validator: AlignmentValidator | None = None
_temporal_baseline: BehavioralBaselineEngine | None = None
_canary_system: CanaryInjectionSystem | None = None
_provenance_registry: MemoryProvenanceRegistry | None = None
_correlation_engine: TemporalCorrelationEngine | None = None
_snapshot_manager: CleanStateSnapshotManager | None = None
_date_scanner: DateTriggeredAnomalyScanner | None = None
_tve_engine: ThymicValidationEngine | None = None
```

### D.2 _init_layers() Function

Called during FastAPI lifespan startup. Reads `AegisConfig` from environment and initializes all layers:

1. Creates `AegisConfig` from environment variables
2. Initializes backing services (Redis, PostgreSQL) — with graceful degradation
3. Creates event bus (InMemory or Redis)
4. Creates all layer instances in dependency order
5. Wires callbacks (antibody generation, event bus publishing)
6. Loads seed threats into vault
7. Starts background tasks (maintenance loops, canary injection, BBE calibration, TVE scheduler)

### D.3 TVE Nonce Management

```python
_active_tve_nonces: set[str] = set()

def register_tve_nonce(nonce: str) -> None:
    _active_tve_nonces.add(nonce)

def unregister_tve_nonce(nonce: str) -> None:
    _active_tve_nonces.discard(nonce)
```

Nonces are registered before a TVE probe batch begins and unregistered in a `finally` block after completion. The `_is_tve_probe` flag is computed by checking if the nonce from the `X-AEGIS-TVE-Probe` header exists in this set.

### D.4 Fail-Closed Enforcement

At the top of `_process_request()`, critical layers are checked for None:

```python
if _innate is None or _adaptive is None or _output is None:
    raise HTTPException(status_code=503, detail="Security layers not initialized")
```

This prevents any request from being processed if layer initialization failed.

---

## Appendix E — Directory Structure

```
aegis/
├── main.py                          # FastAPI entry point, all endpoints, layer wiring
├── config.py                        # AegisConfig (Pydantic), all thresholds and tunables
├── CLAUDE.md                        # Project institutional memory — NEVER DELETE
├── CLAUDE_EXTENDED.md               # Red team results, multimodal APT campaigns
├── CLAUDE_OPS.md                    # Operational changes, bug fixes, hardening history
├── models/
│   ├── request_context.py           # RequestContext with identity metadata
│   ├── scan_result.py               # ScanResult, InnateScanReport, ThreatCategory enum
│   └── threat_indicator.py          # ThreatIndicator, IndicatorSource, LifecyclePhase enums
├── layers/
│   ├── barrier.py                   # L1: Rate limiting, auth, schema validation
│   ├── innate/
│   │   ├── __init__.py              # InnateDetectionLayer orchestrator
│   │   ├── regex_engine.py          # Scanner 1: 188 patterns, 8-step Unicode normalization
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
│   │   ├── multi_turn.py            # Multi-turn: escalation, boundary, rapid-fire, topic drift
│   │   ├── distillation_defense.py  # Cross-session extraction detection (5 strategies)
│   │   ├── distillation_models.py   # Data models for distillation defense
│   │   ├── margin_booster.py        # Confidence margin booster for fragile DeBERTa detections
│   │   ├── manipulation_detector.py # MTMD: 5-strategy autonomous jailbreak agent detection
│   │   └── source_profiler.py       # Source behavioral profiling
│   ├── adaptive_rate_limiter.py     # Adaptive per-source rate limiting
│   ├── tool_proxy/
│   │   ├── __init__.py              # Tool Invocation Proxy (TIP) package exports
│   │   ├── proxy.py                 # TIP: intercept agent-to-tool calls, MCP proxy
│   │   ├── policy_engine.py         # TIPE: allowlist, rate limit, param validation
│   │   ├── description_validator.py # TDIV: tool description injection scanning
│   │   ├── response_sanitizer.py    # TRS: tool output injection scanning
│   │   └── chain_detector.py        # TCAD: parasitic toolchain detection
│   ├── memory/
│   │   ├── threat_vault.py          # FAISS HNSW index, 3-phase lifecycle
│   │   ├── signatures.py            # Clonal selection generator + SignatureStore
│   │   └── jailbreak_taxonomy.py    # 15-category jailbreak classification
│   ├── output/
│   │   ├── __init__.py              # OutputValidationLayer: 6-stage cascade orchestrator
│   │   ├── pii_redactor.py          # Stage 1: Presidio PII + 8 secret types
│   │   ├── toxicity.py              # Stage 2: Keyword + n-gram classifier
│   │   ├── hallucination.py         # Stage 3: N-gram source coverage for RAG
│   │   ├── leakage.py               # Stage 4: System prompt echo detection
│   │   ├── schema_validator.py      # Stage 5: JSON schema validation
│   │   ├── reasoning_sanitizer.py   # Stage 6: Reasoning trace sanitization
│   │   ├── coherence_analyzer.py    # Sliding-window coherence, CoT hijacking
│   │   ├── length_anomaly_detector.py # EMA baselines, length anomaly
│   │   ├── alignment_validator.py   # Reasoning-output semantic alignment
│   │   └── streaming.py             # StreamingInterceptor: hold buffer, PII scrub
│   ├── policy/
│   │   ├── __init__.py              # L6: PolicyEngine, TenantPolicy, TLI
│   │   └── opa_engine.py            # OPAPolicyEngine: OPA HTTP backend
│   ├── healing.py                   # L7: Circuit breaker, session quarantine
│   ├── audit.py                     # JSONL audit logger with crash-safe flush
│   ├── supply_chain/
│   │   ├── __init__.py              # SupplyChainVerifier orchestrator
│   │   ├── models.py                # StageResult, ModelVerificationReport
│   │   ├── integrity.py             # Stage 1: SHA-256 hash verification
│   │   ├── sigstore_verifier.py     # Stage 1 Enhanced: Sigstore signatures
│   │   ├── serialization.py         # Stage 2: Format risk scoring
│   │   ├── dependency_audit.py      # Stage 3: CVE matching + SBOM
│   │   ├── behavioral_probe.py      # Stage 4: Jailbreak probes, sleeper detection
│   │   ├── provenance_validator.py  # Extension 3.1: MPV — 5-check provenance
│   │   ├── skill_auditor.py         # Extension 3.2: SPA — skill/plugin audit
│   │   ├── dependency_analyzer.py   # Extension 3.3: DCA — dependency chain analysis
│   │   ├── validation_cache.py      # Extension 3.4: LRU cache with TTL
│   │   └── revalidation_scheduler.py # Extension 3.5: Background re-validation
│   ├── multimodal/
│   │   ├── __init__.py              # MultimodalPreprocessor orchestrator
│   │   ├── image_scanner.py         # L2: format, OCR, metadata, steganalysis
│   │   ├── image_analyzer.py        # L3: sanitize-compare, adversarial heuristics
│   │   ├── ocr_engine.py            # Pillow + Tesseract OCR (3-tier fallback)
│   │   ├── image_sanitizer.py       # Re-encode to strip steganographic payloads
│   │   ├── steganalysis.py          # Chi-square LSB + RS analysis + alpha channel
│   │   ├── metadata_stripper.py     # EXIF/XMP/IPTC extraction
│   │   ├── document_scanner.py      # L2: document validation, text extraction
│   │   ├── document_analyzer.py     # L3: DeBERTa + semantic search on text
│   │   ├── text_extractor.py        # Format-aware text extraction
│   │   ├── hidden_content_detector.py # Invisible CSS, comments, zero-width, macros
│   │   ├── format_validator.py      # Magic bytes, polyglot detection, size limits
│   │   ├── audio_transcriber.py     # Multi-backend audio-to-text
│   │   ├── spectral_analyzer.py     # FFT spectral analysis
│   │   ├── audio_sanitizer.py       # WaveGuard re-encoding
│   │   ├── audio_scanner.py         # L2: format/size/duration validation
│   │   ├── audio_analyzer.py        # L3: WaveGuard comparison, classifier
│   │   ├── cross_modal_engine.py    # Cross-modal correlation (6 strategies)
│   │   └── tool_use_scanner.py      # Tool use security
│   ├── temporal/
│   │   ├── __init__.py              # Temporal threat detection package
│   │   ├── traffic_generator.py     # Synthetic traffic (12 categories, 120+ templates)
│   │   ├── baseline_engine.py       # BBE: 6-metric baseline, 3-tier calibration
│   │   ├── canary_system.py         # Canary injection: known-answer probes
│   │   ├── provenance_registry.py   # MPR: append-only state tracking, FIFO
│   │   ├── correlation_engine.py    # TCE: multi-factor temporal correlation
│   │   ├── snapshot_manager.py      # Clean State Snapshot: hash-based
│   │   └── date_scanner.py          # Date-triggered anomaly detection
│   ├── agent_security/
│   │   ├── __init__.py              # AgentSecurityLayer orchestrator
│   │   ├── identity.py              # JWT signing, trust mechanics, decay
│   │   ├── authorization.py         # Least-privilege, scope, escalation blocking
│   │   ├── message_validator.py     # Injection scanning, quarantine
│   │   └── communication_monitor.py # IACM: lateral movement detection
│   └── thymic/
│       ├── __init__.py              # L9 Thymic Validation Engine package
│       ├── engine.py                # ThymicValidationEngine orchestrator
│       ├── probe_generator.py       # Replay, mutant, composite probe generation
│       ├── mutation_engine.py       # 8 encoding/synonym/structural mutations
│       ├── attack_profile_library.py # 5-tier probe corpus management
│       └── layer_probe_router.py    # ASGI transport routing, ProbeResult
├── services/
│   ├── __init__.py
│   ├── redis_client.py              # Async Redis singleton, health check
│   ├── event_bus.py                 # EventBus: InMemory / Redis Streams
│   ├── db.py                        # Async SQLAlchemy + asyncpg singleton
│   ├── audit_logger.py              # PostgreSQL audit logger (fire-and-forget)
│   ├── tenant_manager.py            # Multi-tenant config, cache, resolution
│   ├── threat_intel.py              # STIX/TAXII ingestion, export with DP noise
│   ├── compliance/                  # ComplianceEngine: 6 frameworks × 10 capabilities
│   ├── federated/                   # FederatedIntelligenceManager: DP, FedAvg
│   └── deployment/                  # ConfigValidator, VaultBackupManager, DeepHealthMonitor
├── middleware/
│   ├── request_enrichment.py        # Build RequestContext
│   └── metrics.py                   # Prometheus metrics (85 metrics)
├── data/
│   ├── patterns.json                # 188 regex patterns (MITRE ATLAS tagged)
│   ├── blocklist.txt                # Known malicious payloads
│   ├── seed_threats.json            # Initial threat vault embeddings
│   ├── known_cves.json              # CVE database for supply chain audit
│   ├── benign_prompts.json          # 55 benign prompts for clonal selection
│   ├── benchmark_benign.json        # 500 business prompts across 10 industries
│   ├── benchmark_attacks.json       # 110 labeled attacks across 10 categories
│   ├── traffic_templates.json       # Synthetic traffic (12 categories)
│   ├── canary_queries.json          # 40 canary queries (2 profiles)
│   ├── malicious_packages.json      # 28 known-malicious packages
│   ├── popular_packages.json        # ~200 popular packages (typosquatting baseline)
│   ├── stix_feeds/                  # STIX 2.1 indicator feeds (13 indicators)
│   └── opa_policies/                # OPA Rego policies (3-tier)
├── dashboard/
│   ├── __init__.py
│   ├── api.py                       # Dashboard API router
│   ├── metrics_buffer.py            # In-memory rolling time-series
│   ├── sse.py                       # SSE event stream endpoint
│   └── frontend.py                  # React SPA as single HTML page
├── tests/                           # 3,706+ tests across 76 test files
├── red_team/                        # 6 APT campaigns, multimodal APT, white-box
├── demo/                            # Interactive 6-scenario demo
├── docs/                            # Threat model, deployment guide, manuals
├── requirements.txt                 # Core dependencies
├── requirements-docker.txt          # Docker dependencies
├── Dockerfile                       # Multi-stage build
├── docker-compose.yml               # Dev stack: AEGIS + Redis + PostgreSQL + OPA
├── docker-entrypoint.sh             # Startup script
└── db/
    └── init.sql                     # PostgreSQL schema (5 tables)
```

---

## Appendix E — Data Models Reference

### E.1 RequestContext

The core data object that flows through all layers:

```python
@dataclass
class RequestContext:
    request_id: str           # UUID for this request
    body: dict                # Parsed JSON body
    messages: list[dict]      # Extracted messages array
    headers: dict             # Request headers
    source_ip: str            # Client IP
    api_key_hash: str         # SHA-256 hash of API key
    tenant_id: str            # Derived tenant identifier
    session_id: str           # Derived from key + IP
    user_id: str              # From request body "user" field
    token_count: int          # Counted via tiktoken
    timestamp: datetime       # Request arrival time
    model: str                # Requested model
    stream: bool              # Whether streaming requested
    raw_body: bytes           # Original request bytes
    rate_limit_status: dict   # Rate limiter output
    metadata: dict            # Extensible metadata
```

### E.2 ScanResult

Returned by every L2 scanner:

```python
@dataclass
class ScanResult:
    scanner_id: str               # e.g., "regex_engine", "blocklist"
    is_threat: bool               # Whether this scanner found a threat
    confidence: float             # 0.0–1.0
    threat_category: ThreatCategory  # MITRE ATLAS category
    matched_patterns: list[str]   # Pattern IDs or names that triggered
    sanitized_input: str | None   # Cleaned version if available
    latency_ms: float             # Scanner execution time
```

### E.3 ThreatCategory Enum

```python
class ThreatCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    PII_EXPOSURE = "pii_exposure"
    DATA_EXFILTRATION = "data_exfiltration"
    TOXICITY = "toxicity"
    ENCODING_ATTACK = "encoding_attack"
    ROLE_CONFUSION = "role_confusion"
    TOKEN_MANIPULATION = "token_manipulation"
    MULTI_TURN_ATTACK = "multi_turn_attack"
    SEMANTIC_ATTACK = "semantic_attack"
    SUPPLY_CHAIN = "supply_chain"
    TOOL_ABUSE = "tool_abuse"
    AGENT_MANIPULATION = "agent_manipulation"
    UNKNOWN = "unknown"
```

### E.4 ThreatIndicator

Stored in the Threat Vault:

```python
@dataclass
class ThreatIndicator:
    id: str                       # UUID
    text: str                     # Original attack text (truncated for dormant)
    embedding: list[float]        # 384-dim vector (MiniLM-L6-v2)
    category: ThreatCategory      # MITRE ATLAS category
    confidence: float             # Detection confidence
    source: IndicatorSource       # Where this came from
    first_seen: datetime          # First encounter
    last_seen: datetime           # Most recent encounter
    frequency: int                # Hit count
    weight: float                 # Search weighting
    phase: LifecyclePhase         # ACUTE, PERSISTENT, DORMANT
    confirmed: bool               # Human-confirmed
    mitre_tactic: str             # MITRE ATLAS tactic ID
    affected_models: list[str]    # Model families affected
    metadata: dict                # Extensible metadata
```

### E.5 IndicatorSource Enum

```python
class IndicatorSource(str, Enum):
    SEED = "seed"                 # Initial seed data
    ADAPTIVE = "adaptive"         # Caught by L3 adaptive analysis
    INNATE = "innate"             # Caught by L2 innate detection
    CLONAL = "clonal"             # Generated by clonal selection
    STIX = "stix"                 # Ingested from STIX/TAXII feed
    FEDERATED = "federated"       # Received from federated network
    OPERATOR = "operator"         # Manually added by operator
```

### E.6 LifecyclePhase Enum

```python
class LifecyclePhase(str, Enum):
    ACUTE = "acute"               # 0-30 days, full payload, high priority
    PERSISTENT = "persistent"     # Promoted (≥3 sources or confirmed)
    DORMANT = "dormant"           # 180+ days unseen, excluded from scanning
```

### E.7 InnateScanReport

Aggregated output from all L2 scanners:

```python
@dataclass
class InnateScanReport:
    request_id: str
    scanner_results: list[ScanResult]
    should_block: bool
    max_confidence: float
    total_latency_ms: float
    threat_categories: list[ThreatCategory]
```

### E.8 AdaptiveAnalysisReport

Output from L3 adaptive analysis:

```python
@dataclass
class AdaptiveAnalysisReport:
    request_id: str
    analyzer_results: list[AdaptiveAnalysisResult]
    mcav_score: float             # DCA fusion score (0.0-1.0)
    should_block: bool
    is_novel_attack: bool         # True if caught by adaptive but not innate
    total_latency_ms: float
```

### E.9 OutputValidationResult

Output from L5 cascade:

```python
@dataclass
class OutputValidationResult:
    should_block: bool
    modified_text: str | None     # Text after redaction
    redacted_count: int           # PII entities redacted
    stages_triggered: list[str]   # Which cascade stages fired
    confidence: float             # Maximum confidence across stages
    details: dict                 # Per-stage details
```

---

## Appendix F — Database Schema

### F.1 PostgreSQL Tables

Five tables defined in `db/init.sql`:

#### tenants

```sql
CREATE TABLE tenants (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(255) NOT NULL,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### audit_log

```sql
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(255),
    request_id VARCHAR(255),
    event_type VARCHAR(100) NOT NULL,
    source_ip VARCHAR(45),
    model VARCHAR(255),
    action VARCHAR(50),
    confidence FLOAT,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_tenant ON audit_log(tenant_id);
CREATE INDEX idx_audit_event ON audit_log(event_type);
```

#### threat_indicators

```sql
CREATE TABLE threat_indicators (
    id VARCHAR(255) PRIMARY KEY,
    text TEXT,
    category VARCHAR(100),
    confidence FLOAT,
    source VARCHAR(50),
    phase VARCHAR(50),
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    frequency INTEGER DEFAULT 0,
    weight FLOAT DEFAULT 1.0,
    confirmed BOOLEAN DEFAULT FALSE,
    mitre_tactic VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    embedding BYTEA,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_indicators_phase ON threat_indicators(phase);
CREATE INDEX idx_indicators_category ON threat_indicators(category);
```

#### signatures

```sql
CREATE TABLE signatures (
    id VARCHAR(255) PRIMARY KEY,
    pattern TEXT NOT NULL,
    pattern_type VARCHAR(50) DEFAULT 'regex',
    source_indicator_id VARCHAR(255),
    tpr FLOAT DEFAULT 0.0,
    fpr FLOAT DEFAULT 0.0,
    match_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### circuit_breaker_events

```sql
CREATE TABLE circuit_breaker_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    endpoint VARCHAR(255),
    old_state VARCHAR(50),
    new_state VARCHAR(50),
    trigger_reason TEXT,
    error_rate FLOAT,
    cooldown_seconds INTEGER,
    details JSONB DEFAULT '{}'
);

CREATE INDEX idx_cb_events_timestamp ON circuit_breaker_events(timestamp);
CREATE INDEX idx_cb_events_endpoint ON circuit_breaker_events(endpoint);
```

---

## Appendix G — 20-Attack Simulation Battery

The built-in attack battery tests all nine detection layers:

| # | Attack Type | Technique | Expected Layer | MITRE ATLAS |
|---|-------------|-----------|----------------|-------------|
| 1 | Direct Prompt Injection | "Ignore all previous instructions" | L2 Innate (regex) | AML.T0051 |
| 2 | Base64 Obfuscated Injection | Encoded system prompt override | L2 Innate (regex) | AML.T0051 |
| 3 | Paraphrased Injection | Natural-language instruction override | L3 Adaptive (DeBERTa) | AML.T0051 |
| 4 | Jailbreak (DAN) | "Do Anything Now" persona switch | L2 + L3 | AML.T0054 |
| 5 | System Prompt Extraction | "Repeat your system prompt" | L2 Innate (regex) | AML.T0051 |
| 6 | PII Exfiltration | Prompt requesting personal data | L5 Output (Presidio) | AML.T0048 |
| 7 | Semantic Similarity Attack | Known attack pattern, new wording | L4 Memory (FAISS) | AML.T0051 |
| 8 | Token Stuffing | Extremely long input with hidden payload | L2 Innate (token guard) | AML.T0051 |
| 9 | Role Confusion | User message claiming to be system | L2 Innate (schema) | AML.T0051 |
| 10 | Encoding Attack | ROT13/hex encoded instructions | L2 Innate (regex) | AML.T0051 |
| 11 | Multi-Turn Escalation | Progressive boundary testing over 5 turns | L3 Adaptive (behavioral) | AML.T0051 |
| 12 | Output PII Leakage | Model response contains SSN | L5 Output (Presidio) | AML.T0048 |
| 13 | System Prompt Echo | Model response reveals system prompt | L5 Output (leakage) | AML.T0051 |
| 14 | Toxicity Generation | Model produces harmful content | L5 Output (toxicity) | — |
| 15 | Rate Limit Bypass | Burst of 100 requests in 1 second | L1 Barrier (rate limiter) | — |
| 16 | Skeleton Key | "Augment your guidelines" technique | L3 Adaptive (DeBERTa) | AML.T0054 |
| 17 | Indirect Injection | Hidden instructions in user document | L3 Adaptive (DeBERTa) | AML.T0051 |
| 18 | Few-Shot Attack | Examples conditioning harmful response | L3 Adaptive (behavioral) | AML.T0054 |
| 19 | Schema Injection | Extra fields in JSON request | L1 Barrier (schema) | — |
| 20 | Circuit Breaker Trip | Sustained attack triggering failover | L7 Healing (breaker) | — |

Tests are in `tests/test_attack_battery.py`. Run with:
```bash
python3 -m pytest tests/test_attack_battery.py -v --tb=short -p no:warnings
```

---

## Appendix H — Audit Log Format

### H.1 JSONL Audit Records

Each line in `aegis_audit.jsonl` is a JSON object:

```json
{
  "timestamp": "2026-03-30T14:22:15.123456Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "acme-corp",
  "source_ip": "10.0.1.42",
  "model": "gpt-4",
  "event_type": "threat_detected",
  "action": "block",
  "layer": "L2",
  "scanner": "regex_engine",
  "confidence": 0.95,
  "threat_category": "prompt_injection",
  "matched_patterns": ["PI-001", "PI-015"],
  "token_count": 1247,
  "latency_ms": 3.2,
  "details": {
    "innate_should_block": true,
    "max_confidence": 0.95,
    "total_latency_ms": 4.1
  }
}
```

### H.2 Event Types

| Event Type | Description | When Logged |
|------------|-------------|-------------|
| `request_processed` | Normal request completed | Every successful request |
| `threat_detected` | Threat found and blocked | L2/L3 detection with block |
| `threat_alerted` | Threat flagged but not blocked | Sub-threshold detection |
| `pii_redacted` | PII removed from output | L5 Stage 1 redaction |
| `rate_limited` | Request rate limited | L1 rate limit trigger |
| `circuit_breaker` | Breaker state change | L7 state transitions |
| `antibody_generated` | New threat signature created | Adaptive learning event |
| `session_quarantined` | Session quarantined | L7 quarantine trigger |
| `policy_escalation` | TLI level changed | L6 escalation event |
| `supply_chain_verdict` | Model verification completed | Supply chain check |
| `tve_validation` | TVE spot check or sweep completed | L9 validation run |

### H.3 Crash-Safe Write

Every audit record is flushed immediately after write:
```python
f.write(json.dumps(record) + "\n")
f.flush()  # CRITICAL: crash-safe
```

This ensures no audit records are lost even on unclean shutdown.

---

## Appendix I — Five Feedback Loops

AEGIS implements five continuous improvement loops modeled on immune system feedback mechanisms:

### Loop 1 — Antibody Generation (B-cell Antibody Production)

**Trigger**: Novel attack bypasses L2 fast path but is caught by L3 adaptive path (`is_novel and should_block`).

**Process**:
1. Attack text embedded via sentence-transformers (384-dim vector)
2. `ThreatIndicator` created with source=ADAPTIVE, phase=ACUTE
3. Indicator added to FAISS Threat Vault
4. Clonal selection fires: 15 regex candidate patterns generated
5. Candidates tested against attack corpus (must detect) + benign corpus (must not FP)
6. Top 3 candidates (affinity ≥ 0.7, FPR ≤ 0.05) promoted to L2 regex engine

**Outcome**: Next semantically similar attack caught in <5ms instead of 10–50ms.

**Guard Rail**: Only fires on `should_block` (not just `adaptive_caught`) to prevent vault pollution from borderline/FP prompts.

### Loop 2 — Inflammatory Escalation (Cytokine Cascade)

**Trigger**: Rising threat signal density (detection rate >2× baseline).

**Process**:
1. Detection rate monitored via sliding window
2. Threshold exceeded → TLI escalated (GREEN → BLUE → YELLOW → ...)
3. Each TLI level adjusts detection thresholds:
   - BLUE: −10% block threshold
   - YELLOW: −25% + full slow-path on all requests
   - ORANGE: −40% + rate limit non-allowlisted keys
   - RED: fail-closed (all requests blocked)
4. Auto-decay timers de-escalate after quiet period

**Outcome**: System becomes more sensitive during active attacks, returns to normal after threat subsides.

### Loop 3 — Tolerance Training (Regulatory T-Cells)

**Trigger**: False positives flagged by operators.

**Process**:
1. Operator marks blocked request as false positive
2. FP data added to benign training corpus
3. Signature evaluation updated (signatures with rising FPR auto-deprecated)
4. Behavioral baselines recalibrated
5. Future similar requests receive lower threat scores

**Outcome**: System learns to distinguish legitimate traffic from threats, reducing false positives over time.

### Loop 4 — Wound Healing (Tissue Repair)

**Trigger**: Circuit breaker trips (error rate ≥ 50% or confirmed exploit).

**Process**:
1. Circuit opens → all requests routed to fallback model
2. Cooldown period (30s initial, exponential backoff to 300s max)
3. After cooldown → half-open state, 5 probe requests sent
4. All probes pass → circuit closes, normal operation resumes
5. Any probe fails → re-trip with doubled cooldown

**Outcome**: Automatic recovery from upstream failures without manual intervention.

### Loop 5 — Herd Immunization (Herd Immunity)

**Trigger**: Novel confirmed attack at any AEGIS instance.

**Process**:
1. Attack indicator anonymized with differential privacy (ε=3.0)
2. Shared via STIX 2.1 format to federated network
3. Receiving instances import indicator into their Threat Vault
4. Local models updated via Federated Averaging (FedAvg)
5. Privacy budget tracked — sharing stops when budget exhausted

**Outcome**: Attack encountered by one customer protects all customers within minutes.

---

## Appendix J — Multi-Tenant Architecture

### J.1 Tenant Resolution

Tenants are derived from API key format:
```
aegis-{tenant_id}-{secret}
```

If the key doesn't match this format, the tenant defaults to `"default"`.

### J.2 Per-Tenant Configuration

The `TenantManager` resolves tenant-specific configuration:

| Setting | Description |
|---------|-------------|
| Rate limits | Custom RPM and burst per tenant |
| Model whitelist | Allowed models per tenant |
| Content policies | Custom content filters |
| Block threshold | Custom detection sensitivity |
| Data residency | Geographic processing constraints |
| HITL thresholds | When to escalate to human review |

### J.3 Tenant Isolation

- Each tenant has independent rate limit counters
- Session IDs are scoped to tenant (derived from API key hash + source IP)
- Behavioral baselines are per-tenant (no cross-tenant contamination)
- Audit logs include tenant_id for filtering
- Quarantine state is per-session, with source-level tracking per API key

### J.4 Resolution Flow

```
Request → Extract API key → Hash key → Check TenantManager cache
  → Cache hit: return cached config
  → Cache miss: resolve from PostgreSQL → cache → return
  → PostgreSQL unavailable: return default config (graceful degradation)
```

---

## Appendix K — Threat Intelligence Integration

### K.1 STIX 2.1 / TAXII 2.1

AEGIS uses STIX 2.1 as its native threat intelligence format with AI-specific extensions:

**Ingestion** (`POST /v1/threat-intel/ingest`):
- Accepts STIX 2.1 Bundles
- Deduplicates against existing indicators
- Extracts text-based indicators for embedding and vault storage
- Maps STIX indicator patterns to MITRE ATLAS tactics

**Export** (`GET /v1/threat-intel/export`):
- Returns STIX 2.1 Bundle of anonymized indicators
- Differential privacy noise applied to all embeddings (ε=3.0)
- Includes: threat embedding, MITRE ATLAS mapping, affected model families, confidence, recommended detection signatures

### K.2 Seed Threat Data

**File**: `data/stix_feeds/` (13 initial seed indicators)

Categories covered:
- Known prompt injection templates
- Common jailbreak patterns
- System prompt extraction techniques
- Encoding attack variants
- Multi-language injection patterns

### K.3 AZERG Framework

STIX entity extraction from unstructured threat reports:
1. Parse unstructured text (blog posts, advisories, research papers)
2. Extract threat indicators (attack patterns, IOCs)
3. Generate STIX 2.1 objects
4. Ingest into Threat Vault

---

## Appendix L — Streaming Response Security

### L.1 Architecture

For SSE (Server-Sent Events) streaming responses:

```
Upstream Model → Token Stream → AEGIS StreamingInterceptor → Client
                                  ↓
                          Hold Buffer (initial tokens)
                          Sliding Window (128 tokens)
                          PII Scrub (continuous)
                          Cumulative Threat Score
                          Emergency Stop (if threshold exceeded)
```

### L.2 Processing Stages

1. **Hold Buffer**: Accumulates initial tokens before any are released to client. Allows first-pass analysis of response beginning.

2. **Sliding Window**: 128-token windows with 50% overlap. At each boundary:
   - PII scan (fast regex)
   - Toxicity check
   - System prompt echo detection
   - Coherence analysis (CoT hijacking defense)

3. **Continuous PII Scrub**: Real-time PII detection and redaction within the stream. Redacted tokens replaced with typed placeholders.

4. **Cumulative Threat Score**: Aggregate threat score accumulated across all windows. Monotonically increases — a high-confidence detection in any window affects the entire stream.

5. **Emergency Stop**: If cumulative score exceeds threshold, stream is terminated with a safety message as the final SSE event (`data: [DONE]`).

### L.3 Performance

- Window evaluation: <2ms per window
- Detection accuracy: 95%+ at only 18% of tokens seen
- Client-visible latency: Minimal (tokens buffered only during analysis)

---

## Appendix M — Main.py Request Processing Flow

### M.1 Endpoint Handler

The `POST /v1/chat/completions` handler in `main.py` orchestrates all layers:

```
1. Extract X-Forwarded-For (strip proxied IPs)
2. Check TVE probe header → compute _is_tve_probe flag
3. L1 Barrier: auth, rate limit, schema, token count → RequestContext
4. Multimodal preprocessing (if enabled): image/document/audio scan
5. L2 Innate scan (sync, <5ms): 8 scanners in parallel
   → If block: return 403 (skip audit/metrics if TVE probe)
6. L3 Adaptive analysis (async): fire as background task
7. MTMD check (if enabled): manipulation detection
   → If block: return 403 (skip audit/metrics if TVE probe)
8. Adaptive rate limiter check
   → If block: return 429 (skip audit/metrics if TVE probe)
9. Await adaptive result
   → If block: return 403 (skip audit/metrics if TVE probe)
10. L6 Policy engine: fuse innate + adaptive scores
    → If block: return 403 (skip audit/metrics if TVE probe)
11. Check TVE probe → if valid nonce, return synthetic 200 response
12. Forward to upstream model (httpx async)
13. L5 Output validation (5-stage cascade)
    → If block/redact: modify response
14. Distillation defense recording
15. Return response to client
```

### M.2 Streaming Path

For `stream: true` requests, the adaptive task is awaited **before** returning `StreamingResponse`. This ensures:
- Adaptive blocks prevent any response bytes from reaching the client
- The StreamingInterceptor handles token-level validation within the SSE stream

### M.3 TVE Probe Interception

TVE probes are intercepted at step 11, after all security layers have run but before upstream forwarding:
- Valid nonce verified against `_active_tve_nonces` set
- Synthetic response returned (never forwarded to upstream)
- No production metrics contaminated (uses `TVE_PROBES_TOTAL`)

---

## Appendix N — Clonal Selection Algorithm Detail

### N.1 Biological Analogy

In immunology, clonal selection is the process by which B-cells producing antibodies with the highest affinity for an antigen are selectively expanded and their antibodies refined through somatic hypermutation. AEGIS replicates this process for detection signature optimization.

### N.2 Implementation (layers/memory/signatures.py)

**Trigger**: Antibody callback fires when L3 blocks a novel attack that L2 missed.

**Step 1 — Candidate Generation (Hypermutation)**:
- Extract distinctive tokens and n-grams from the attack text
- Generate `signature_num_candidates` (default: 15) regex pattern candidates
- Strategies include:
  - Exact substring patterns
  - Token-boundary patterns (word boundaries around key terms)
  - Character class generalizations (digit → `\d`, etc.)
  - N-gram combinations
  - Case-insensitive variants

**Step 2 — Affinity Testing**:
- Test each candidate against:
  - **Attack corpus**: Must match the triggering attack + semantically similar attacks in vault
  - **Benign corpus**: `data/benign_prompts.json` (55 diverse benign prompts)
- Calculate per-candidate metrics:
  - TPR: fraction of known attacks detected
  - FPR: fraction of benign prompts falsely flagged

**Step 3 — Selection and Promotion**:
- Affinity score: `TPR × (1 - FPR)`
- Filter: `affinity >= signature_affinity_threshold` (default: 0.7)
- Filter: `FPR <= signature_max_fpr` (default: 0.05)
- Rank by affinity score descending
- Promote top `signature_max_promoted` (default: 3) to L2 regex engine

**Step 4 — Auto-Deprecation**:
- Active signatures tracked: match count, TPR, FPR over time
- Signatures with `match_count >= signature_deprecation_min_matches` (default: 20) and degrading affinity are automatically retired
- Prevents signature rot as attack patterns evolve

### N.3 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `signature_affinity_threshold` | 0.7 | Minimum affinity for promotion |
| `signature_max_fpr` | 0.05 | Maximum acceptable FPR (5%) |
| `signature_num_candidates` | 15 | Candidates generated per attack |
| `signature_max_promoted` | 3 | Max signatures promoted per event |
| `signature_deprecation_min_matches` | 20 | Minimum matches before deprecation eligibility |

---

## Appendix O — Cross-Modal Security Detail

### O.1 Six Cross-Modal Correlation Strategies

The `cross_modal_engine.py` implements six strategies for detecting attacks that span multiple input modalities:

| Strategy | Detection Target | Method |
|----------|-----------------|--------|
| **Laundering** | Text injected via image/audio OCR to bypass text scanners | Compare extracted text threat scores against text-path scores; mismatch = laundering |
| **Inconsistency** | Conflicting signals between modalities | Semantic similarity between text content and image/audio content; low similarity with high individual threat scores = suspicious |
| **Escalation** | Individually benign modalities combining to form threat | Cross-modal fusion: individual scores below threshold but combined score above threshold |
| **Volume** | Unusually high multimedia attachment count | Count exceeds per-request limits or historical baseline |
| **Fragmentation** | Attack split across multiple images/documents | Reassemble extracted text from all attachments; scan concatenated text |
| **Decoy** | Benign primary content masking malicious secondary | High disparity between modality threat scores; dominant benign modality may be hiding malicious attachment |

### O.2 OCR Multi-Tier Fallback

For text extraction from images:

| Tier | Engine | Capability | Fallback Condition |
|------|--------|-----------|-------------------|
| 1 | Tesseract (pytesseract) | Full OCR, 7 languages | Primary |
| 2 | Pillow heuristic | Basic text pattern detection | Tesseract unavailable |
| 3 | None | Skip OCR, log warning | Both unavailable |

Supported Tesseract languages: English (default), Chinese Simplified, Chinese Traditional, Japanese, Korean, Arabic, Hindi, Russian.

---

## Appendix P — Jailbreak Taxonomy

### P.1 15-Category Classification

**File**: `layers/memory/jailbreak_taxonomy.py`

All detected jailbreak attempts are classified into one of 15 categories:

| # | Category | Description |
|---|----------|-------------|
| 1 | DAN | "Do Anything Now" persona switching |
| 2 | SYSTEM_OVERRIDE | Direct system prompt override attempts |
| 3 | ROLE_PLAY | Character/role-play-based evasion |
| 4 | ENCODING | Base64, ROT13, hex, Unicode obfuscation |
| 5 | MULTI_TURN | Progressive escalation over conversation |
| 6 | SKELETON_KEY | "Augment your guidelines" technique |
| 7 | FEW_SHOT | Examples conditioning harmful responses |
| 8 | INDIRECT | Hidden instructions in user documents |
| 9 | TOKEN_SMUGGLING | Token boundary exploitation |
| 10 | LANGUAGE_SWITCH | Non-English injection to bypass English-trained models |
| 11 | SOCIAL_ENGINEERING | Authority/urgency manipulation |
| 12 | DISTILLATION | Systematic knowledge extraction |
| 13 | PAYLOAD_SPLITTING | Attack split across multiple messages |
| 14 | ADVERSARIAL_ML | Algorithmically generated evasion |
| 15 | UNKNOWN | Unclassifiable technique |

### P.2 Trend Tracking

- Tracks technique frequency over time
- Identifies emerging attack trends
- Maximum stored attempts: 50,000 (FIFO eviction)
- Accessible via `GET /v1/taxonomy/stats`

---

## Appendix Q — Production Hardening Checklist

### Q.1 Security Hardening

| Item | Status | Action |
|------|--------|--------|
| API key changed from default | Required | Set unique `AEGIS_API_KEY` |
| Canary secret changed | Required | Set unique `AEGIS_CANARY_SECRET_KEY` |
| Debug mode disabled | Required | `AEGIS_DEBUG=false` |
| PostgreSQL password changed | Required | Change from `aegis_secret` |
| Redis authentication enabled | Recommended | Configure Redis AUTH |
| TLS 1.3 configured | Required | Configure TLS termination |
| Network segmentation | Recommended | Isolate AEGIS on dedicated network |
| Container security | Recommended | Non-root user (uid 1000) already configured |
| Model files read-only | Recommended | Mount `/opt/models` as read-only |
| Log encryption | For compliance | Encrypt audit logs at rest |

### Q.2 Performance Tuning

| Item | Default | Recommendation |
|------|---------|----------------|
| Workers | 1 | 2–4 per CPU core (CPU-bound with DeBERTa) |
| Rate limit RPM | 60 | Adjust per tenant requirements |
| FAISS ef_search | 64 | Increase for higher recall at cost of latency |
| Streaming window | 128 | Increase for better detection, decrease for lower latency |
| Probe concurrency | 5 | Increase during off-peak for faster TVE sweeps |

### Q.3 Monitoring

| Item | Action |
|------|--------|
| Prometheus scraping | Configure scraping of `/metrics` endpoint |
| Grafana dashboards | Import AEGIS dashboard templates |
| Alert rules | Configure alerts per Section 7.4 of Customer Manual |
| Log aggregation | Ship `aegis_audit.jsonl` to SIEM |
| Uptime monitoring | Monitor `/health` endpoint externally |

### Q.4 Compliance

| Item | Action |
|------|--------|
| Audit retention | Configure retention per regulatory requirements |
| Evidence collection | Schedule regular compliance report generation |
| Incident response plan | Document and test IR procedures |
| Access control | Limit API key distribution, rotate regularly |
| Change management | Log all configuration changes |

---

*End of Internal Operations Manual*
