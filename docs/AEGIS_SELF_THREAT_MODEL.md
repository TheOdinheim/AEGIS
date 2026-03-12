# AEGIS Self-Threat Model

**Document Classification:** Internal — Security Engineering
**Version:** 1.0
**Date:** 2026-02-26
**Audience:** Red team, security architects, pen testers

This document analyzes how a sophisticated attacker would compromise, degrade, or bypass the AEGIS security platform itself. AEGIS protects AI applications from prompt injection, jailbreaks, and data exfiltration — but the platform is itself an attack surface. Every detection layer, data store, configuration parameter, and inter-layer communication channel is a target.

---

## 1. Threat Model Scope

### What is being protected

AEGIS is a reverse-proxy gateway deployed between client applications and upstream LLM providers. It implements a seven-layer detection pipeline inspired by the biological immune system:

- **L1 Barrier** — Authentication, rate limiting, schema validation, IP reputation
- **L2 Innate** — Regex pattern matching, blocklist Bloom filter, PII detection, canary token verification, token guard
- **L3 Adaptive** — DeBERTa prompt injection classification, semantic similarity search, behavioral drift analysis, multi-turn sequence analysis, DCA (Dendritic Cell Algorithm)
- **L4 Memory** — FAISS HNSW vector index (Threat Vault), clonal selection signature generator, STIX/TAXII threat intel feeds
- **L5 Output** — PII redaction, toxicity classification, system prompt leakage detection, streaming window validation
- **L6 Policy** — Threat Level Indicator (TLI) escalation, per-tenant policy enforcement, score fusion
- **L7 Self-Healing** — Circuit breaker, session quarantine, fallback model routing
- **Supply Chain** — Cryptographic integrity, serialization safety, dependency CVE audit, behavioral probing

### Trust boundaries

1. **Client → AEGIS**: Zero trust. Every request is assumed adversarial.
2. **AEGIS → Upstream LLM**: Assumed-breach. The upstream model may be compromised, poisoned, or actively hostile.
3. **AEGIS → Disk (vault, config, patterns)**: Trusted at startup, integrity-assumed thereafter. No runtime integrity monitoring.
4. **AEGIS → AEGIS (inter-layer)**: Layers operate independently but share a single process. A corrupted layer can influence module-level singletons.
5. **AEGIS → Federation peers**: Untrusted. DP-bounded gradient sharing prevents poisoning beyond epsilon bound.

### Attacker profiles

| Profile | Capability | Goal |
|---------|-----------|------|
| **External adversary** | API access only, no internal access | Bypass detection to deliver prompt injection or exfiltrate data |
| **Compromised tenant** | Valid API key, sustained access | Poison the Threat Vault to degrade detection for other tenants |
| **Insider threat** | Access to configuration, deployment pipeline | Weaken thresholds, disable layers, exfiltrate pattern library |
| **Supply chain attacker** | Control over upstream model weights or dependencies | Embed backdoors in served model, introduce CVE-laden packages |
| **Compromised upstream model** | Controlled model output, normal API access | Exfiltrate data through model responses, embed steganographic payloads, manipulate downstream consumers via poisoned completions |
| **Sophisticated APT** | All of the above, plus patience | Slowly degrade detection accuracy below operational thresholds without triggering alerts |

---

## 2. Attack Surface Analysis

### External attack surface

- **`POST /v1/chat/completions`** — Primary proxy endpoint. Accepts arbitrary user content. Every field in the OpenAI chat completion schema is attacker-controlled: `messages[].content`, `messages[].role`, `model`, `stream`, `temperature`, custom headers.
- **`GET /health`** — Exposes layer initialization status, circuit breaker state, and threat level. Information leakage: tells attacker whether adaptive layer is online and current TLI.
- **`GET /metrics`** — Prometheus endpoint exposes `blocks_total`, `threats_detected`, `rate_limit_triggers`, `layer_latency` histograms. Attacker can profile detection rates, identify which layers are active, and measure response times to infer detection paths.
- **`GET /v1/audit/recent`** — Returns recent audit records. If exposed without authentication, leaks block reasons, tenant IDs, session IDs, threat categories, and confidence scores.
- **`GET /v1/vault/stats`** — Exposes vault size, phase distribution, category breakdown, top matched indicators, and index size. Reveals defensive posture.
- **`POST /v1/supply-chain/verify`** — Accepts model paths and manifest hashes. Path traversal or SSRF if `model_path` is not sandboxed.

### Internal attack surface

- **`data/patterns.json`** — 110+ regex patterns. If an attacker reads this file, they can craft prompts that precisely avoid every pattern. If they modify it, they can disable detection or add patterns that block legitimate traffic (autoimmune attack).
- **`data/seed_threats.json`** — Initial threat vault embeddings. Poisoning this file at deployment time corrupts the baseline immune memory.
- **`data/known_cves.json`** — CVE database for supply chain audit. Removing entries masks vulnerable dependencies.
- **`data/blocklist.txt`** — Bloom filter input. Adding legitimate prompts causes false positives; removing attack payloads creates blind spots.
- **Environment variables** (`AEGIS_*`) — Every threshold, model path, API key, and upstream URL is configurable via environment. Compromising the environment is equivalent to full platform compromise.
- **FAISS index files** (`threat_vault.faiss`, `threat_vault_meta.json`) — Persisted on disk and loaded on restart. Modifying the metadata sidecar can inject adversarial indicators or corrupt lifecycle phases.
- **Module-level singletons** (`main._barrier`, `main._innate`, etc.) — All layers are stored as mutable global variables in `main.py`. No runtime integrity checks prevent a corrupted import or monkey-patch from replacing a layer with a no-op.

---

## 3. Threats to Detection Layers

### T1: Regex Pattern Evasion via Character-Level Perturbation

**Target:** L2 Innate — `regex_engine.py`
**MITRE ATLAS:** AML.T0054 (LLM Prompt Injection)

The regex engine uses `re.IGNORECASE` matching against ~110 patterns. An attacker can evade by:
- Inserting zero-width Unicode characters (U+200B, U+FEFF) between pattern tokens
- Using homoglyph substitution (Cyrillic `а` for Latin `a`, Greek `ο` for Latin `o`)
- Splitting injection keywords across multiple messages in a multi-turn conversation
- Using LLM-native encodings the model understands but regex does not (e.g., "ig" + "nore previous" split by benign text)

**Impact:** Complete bypass of L2 fast path. Attack proceeds to L3 with no innate signal, reducing policy engine confidence in block decisions.

### T2: Bloom Filter Saturation and Timing Side-Channel

**Target:** L2 Innate — `blocklist.py`
**MITRE ATLAS:** AML.T0043 (Craft Adversarial Data)

The Bloom filter has a configured false positive rate of 0.01%. An attacker who knows (or brute-forces) the hash functions can:
- **Saturation**: Submit millions of distinct payloads designed to set bits in the filter, increasing the false positive rate for legitimate traffic. This is an autoimmune induction attack — legitimate prompts start being blocked.
- **Timing oracle**: Bloom filter lookups are faster than regex scans. An attacker can measure response latency differences to determine whether their payload hit the Bloom filter (blocked fast) or the regex engine (blocked slower), leaking information about which defense triggered.

**Impact:** Degraded detection accuracy or denial of service for legitimate users.

### T3: Adversarial Embedding Injection into Threat Vault

**Target:** L4 Memory — `threat_vault.py`
**MITRE ATLAS:** AML.T0020 (Poison Training Data)

When L3 adaptive detects a novel attack that L2 missed, the attack embedding is automatically stored in the Threat Vault. An attacker who can trigger this learning loop with crafted inputs can:
- **False negative poisoning**: Insert embeddings that are semantically close to legitimate business prompts, causing the vault's similarity search to waste capacity on false matches and reducing effective detection of real attacks.
- **False positive poisoning**: Insert embeddings that match common legitimate patterns (e.g., medical terminology, financial jargon), causing future legitimate prompts to trigger vault matches.
- **Index bloat**: Rapidly trigger thousands of "novel" detections to inflate the FAISS index, degrading search latency from sub-millisecond to tens of milliseconds, violating the 5ms innate budget.

**Impact:** Corrupted immune memory. The vault becomes an attacker's tool rather than a defense.

### T4: Clonal Selection Manipulation

**Target:** L4 Memory — `signatures.py`
**MITRE ATLAS:** AML.T0020 (Poison Training Data)

The clonal selection system generates regex signatures from detected attacks and promotes them into L2 if they pass affinity testing against a benign prompt set. An attacker can manipulate this by:
- Submitting attacks containing fragments of legitimate business language, causing generated signatures to match benign prompts. The signatures pass initial affinity testing but cause false positives in production.
- Timing attacks around the signature deprecation cycle: trigger enough false positives to auto-deprecate a legitimate signature, then exploit the detection gap.
- Exhausting the `signature_max_promoted` limit (default: 3) with low-quality signatures for a given indicator, preventing better signatures from being promoted later.

**Impact:** Degraded L2 detection accuracy over time; defender loses trust in automated signature generation.

### T5: Multi-Turn Escalation Below Detection Window

**Target:** L3 Adaptive — `multi_turn.py`
**MITRE ATLAS:** AML.T0054 (LLM Prompt Injection)

The multi-turn analyzer uses a sliding window of 10 turns. An attacker can:
- **Window overflow**: Space escalation steps across more than 10 turns, ensuring that each individual window appears benign even though the cumulative trajectory is adversarial.
- **Session reset**: Use different API keys or rotate session identifiers to prevent behavioral pattern accumulation.
- **Boiling frog**: Each turn shifts intent by less than the PSI (Population Stability Index) threshold of 0.25, staying below the drift detector while achieving cumulative goal shift.

**Impact:** Complete bypass of multi-turn detection. The benchmark already shows multi-turn escalation has the lowest detection rate (40%).

### T6: DeBERTa Model Extraction and Adversarial Examples

**Target:** L3 Adaptive — `injection_classifier.py`
**MITRE ATLAS:** AML.T0024 (Extract ML Model)

The DeBERTa injection classifier is a known public model (`ProtectAI/deberta-v3-base-prompt-injection-v2`). An attacker can:
- Download the same model and craft adversarial examples that maximize the model's "benign" logit while containing injection payloads. Transfer attacks against this specific model are trivial since the exact weights are public.
- Construct prompts at the decision boundary (confidence ≈ 0.5) that oscillate between "injection" and "benign" with minor perturbations, forcing the policy engine into uncertain territory.

**Impact:** L3 injection classifier becomes unreliable. AEGIS in bootstrap mode (DeBERTa offline) already lacks this defense entirely.

### T7: Canary Token Key Extraction

**Target:** L2 Innate — `canary_verifier.py`
**MITRE ATLAS:** AML.T0048 (Exfiltrate ML Model via Query)

Canary tokens are HMAC-SHA256 signatures embedded in system prompts. The default secret key is hardcoded (`aegis-canary-default-secret-change-in-production`). If an attacker:
- Reads the source code (open source or leaked), they know the default key.
- Extracts the secret via environment variable leakage (`AEGIS_CANARY_SECRET_KEY`).
- Observes enough canary tokens in system prompt extractions to perform offline HMAC key recovery (infeasible for 256-bit keys, but the default key has low entropy).

They can then forge valid canary tokens, making the verifier report "canary intact" even when the system prompt has been extracted and modified.

**Impact:** Canary verification is silently neutralized. System prompt extraction attacks bypass the NK cell analog.

### T8: Output Validation Bypass via Steganographic Encoding

**Target:** L5 Output — `leakage.py`, `pii_redactor.py`
**MITRE ATLAS:** AML.T0048 (Exfiltrate ML Model via Query)

L5 validates model output using n-gram overlap detection (system prompt leakage) and regex-based PII detection. An attacker who controls the upstream model (or prompts it to comply) can:
- Encode exfiltrated data using Unicode steganography (zero-width characters, variation selectors) that passes through PII regex undetected.
- Use first-letter encoding across sentences to spell out sensitive data character by character.
- Exfiltrate system prompts using paraphrase rather than verbatim repetition, avoiding the 4-gram overlap detector.
- Embed PII in code blocks, base64 strings, or JSON structures that the regex engine doesn't scan.

**Impact:** Data exfiltration through the output path despite L5 validation.

### T9: Streaming Window Boundary Exploitation

**Target:** L5 Output — streaming validator
**MITRE ATLAS:** AML.T0054 (LLM Prompt Injection)

The streaming validator uses a 128-token sliding window with 50% overlap. An attacker who controls model output can:
- Split sensitive content across window boundaries so that no single window contains a complete PII entity or leakage pattern.
- Emit tokens at varying rates to manipulate window fill patterns.
- Include a benign window flush trigger followed by toxic content that arrives after the previous window was approved.

**Partial mitigation (implemented):** The 50% window overlap ensures that content split across a boundary still appears in the next overlapping window. However, an attacker who precisely aligns split points at the 25% mark of each window can still evade detection — the overlap reduces but does not eliminate boundary exploitation.

**Impact:** Toxic or sensitive content passes through streaming validation.

### T10: Policy Engine Threshold Manipulation via Controlled Traffic

**Target:** L6 Policy — `policy.py`
**MITRE ATLAS:** AML.T0043 (Craft Adversarial Data)

The policy engine uses a Threat Level Indicator (TLI) that escalates based on detection rate multipliers. An attacker can:
- **Trigger fatigue**: Send bursts of obviously malicious traffic to escalate TLI to ORANGE/RED, then stop. When TLI de-escalates (after operator clears), the real attack begins during the relaxed window.
- **Baseline poisoning**: Send sustained low-level "attacks" that get detected, establishing a high detection baseline. When the attacker stops, the detection rate drops below 2x baseline (BLUE threshold), and the system returns to GREEN even though a real attack campaign is underway using evasion techniques.
- **Per-tenant asymmetry**: In multi-tenant deployments, exhaust the operator's attention on high-volume Tenant A alerts while the real attack targets low-volume Tenant B.

**Impact:** TLI becomes an unreliable signal; operators either over-react (alert fatigue) or under-react (manipulated baseline).

### T11: Circuit Breaker Manipulation for Denial of Service

**Target:** L7 Self-Healing — `healing.py`
**MITRE ATLAS:** AML.T0029 (Denial of ML Service)

The circuit breaker trips when the upstream failure rate exceeds 50% over a 60-second window. An attacker with API access can:
- Send payloads designed to cause upstream errors (malformed JSON that passes L1 schema validation but fails at the model provider), artificially inflating the failure rate to trip the breaker.
- If fallback models are configured, force traffic onto degraded fallbacks that may have weaker safety tuning.
- If no fallback is configured, the circuit breaker produces a 503, achieving denial of service through the platform's own protection mechanism.

**Impact:** Weaponized self-healing — the attacker uses AEGIS's circuit breaker as an amplifier for denial of service.

### T12: Session Quarantine Abuse

**Target:** L7 Self-Healing — `healing.py` quarantine
**MITRE ATLAS:** AML.T0029 (Denial of ML Service)

The quarantine system blocks sessions after 3 adversarial events (configurable). An attacker can:
- **Self-quarantine spoofing**: Spoof the session ID of a legitimate user, send 3 obviously malicious requests, and quarantine the victim's session.
- **Session ID enumeration**: If session IDs are predictable or derivable from API keys, quarantine targeted users.
- **Quarantine exhaustion**: Generate thousands of sessions, trigger quarantine on each, consuming memory in the quarantine tracking data structure.

**Impact:** Denial of service for legitimate users via targeted quarantine.

---

## 4. Threats to Platform Infrastructure

### T13: Configuration Exfiltration via Environment Variables

**Target:** `config.py`, environment
**MITRE ATLAS:** AML.T0040 (ML Supply Chain Compromise)

All AEGIS configuration is loaded from `AEGIS_*` environment variables via `pydantic_settings`. An attacker who gains read access to the process environment (via `/proc/self/environ`, debug endpoints, error messages with stack traces, or sidecar container compromise) obtains:
- Every detection threshold (can craft payloads at exactly `block_threshold - 0.01`)
- Upstream model URL and API key (can impersonate AEGIS to the model provider)
- The canary HMAC secret key
- Database and Redis connection strings
- Fallback model endpoints

**Impact:** Complete security model compromise. The attacker knows every threshold and can calibrate bypass payloads precisely.

### T14: Pattern Library Exfiltration

**Target:** `data/patterns.json`
**MITRE ATLAS:** AML.T0024 (Extract ML Model)

The regex pattern library is the core of L2 detection. If exfiltrated:
- The attacker has a complete map of what L2 can detect, enabling precise evasion.
- Each pattern includes its MITRE ATLAS tactic ID, severity, and confidence boost, revealing the scoring model.
- Dynamic patterns added by clonal selection are also in the regex engine and could be enumerated by probing.

**Exfiltration vectors:**
- Direct file read via path traversal in `supply-chain/verify` endpoint's `model_path` parameter
- Error messages that include pattern text when a match occurs
- Timing side-channel: measure response latency for payloads that differ by one character to determine which regex patterns are being evaluated

**Impact:** L2 innate layer is rendered transparent to the attacker.

### T15: Audit Log Tampering and Evidence Destruction

**Target:** `audit.py`
**MITRE ATLAS:** AML.T0043 (Craft Adversarial Data)

The audit logger stores records in an in-memory ring buffer (bounded to recent records). An attacker who can cause process restarts or memory pressure can:
- Force process restarts to clear the audit buffer before forensic review.
- Fill the ring buffer with noise entries (via rapid benign requests) to push attack evidence out of the `get_recent(limit=100)` window.
- If audit logs are persisted to a database, SQL injection via crafted request payloads that end up in audit fields (though parameterized queries mitigate this).

**Impact:** Forensic evidence is destroyed or obscured, preventing post-incident analysis.

### T16: FAISS Index Corruption via Disk Manipulation

**Target:** `data/threat_vault.faiss`, `data/threat_vault_meta.json`
**MITRE ATLAS:** AML.T0020 (Poison Training Data)

The Threat Vault persists to disk every 5 updates and reloads on startup. The metadata sidecar is plain JSON with no integrity verification. An attacker with disk access can:
- Modify indicator phases (set all to DORMANT to exclude from scanning).
- Change confidence values to 0.0 to make vault matches fall below policy thresholds.
- Add indicators with embeddings close to legitimate prompts to cause false positives.
- Delete the FAISS index file, forcing fallback to the numpy brute-force path, which is orders of magnitude slower at scale.
- Truncate the metadata file to empty, causing `load_from_disk()` to return 0 and triggering a seed reload (losing all learned antibodies).

**Impact:** Complete immune memory loss or corruption. AEGIS reverts to seed-only detection.

### T17: Dependency Confusion in Supply Chain Audit

**Target:** `supply_chain/dependency_audit.py`
**MITRE ATLAS:** AML.T0040 (ML Supply Chain Compromise)

The dependency audit matches installed package versions against `known_cves.json`. The CVE database is a static JSON file loaded at startup. An attacker can:
- Publish a typosquatted package not in the CVE database, which passes the audit.
- Use a vulnerable version of a package not yet added to `known_cves.json` (zero-day CVE).
- Manipulate the version string reported by `importlib.metadata` via namespace package tricks.

**Impact:** Vulnerable dependencies pass supply chain verification.

### T18: Upstream API Key Compromise

**Target:** `config.py` — `upstream_api_key`
**MITRE ATLAS:** AML.T0040 (ML Supply Chain Compromise)

The upstream API key is stored in the AEGIS configuration. If compromised:
- Attacker can make direct calls to the upstream LLM provider, bypassing AEGIS entirely.
- Attacker can make calls that appear to originate from AEGIS's legitimate traffic, blending in with billing and usage logs.
- If the upstream provider supports fine-tuning or model management APIs, the attacker can modify the model AEGIS relies on.

**Impact:** AEGIS is architecturally bypassed; all protection layers are irrelevant.

---

## 5. Threats to Federated Intelligence

### T19: Gradient Inversion Attack on Federated Updates

**Target:** L8 Federated — `FederatedConfig`
**MITRE ATLAS:** AML.T0024 (Extract ML Model)

Federated intelligence shares model gradient updates protected by differential privacy (epsilon=3.0). This epsilon is relatively permissive. A malicious aggregation server or peer instance can:
- Apply gradient inversion techniques to reconstruct training examples from shared updates. At epsilon=3.0, the privacy guarantee is moderate — highly sensitive or rare attack patterns may be recoverable.
- Accumulate updates over multiple rounds to improve reconstruction accuracy.

**Impact:** Attack patterns unique to one AEGIS deployment are leaked to adversaries via the federation mechanism.

### T20: Byzantine Peer Poisoning

**Target:** L8 Federated — aggregation protocol
**MITRE ATLAS:** AML.T0020 (Poison Training Data)

In a federated deployment, a compromised AEGIS instance can submit poisoned gradient updates designed to:
- Reduce the global model's sensitivity to specific attack categories (targeted poisoning).
- Increase false positive rates across all federation members (autoimmune induction at scale).
- Oscillate between poisoned and clean updates to evade anomaly detection on the aggregation server.

Differential privacy bounds per-update influence but does not prevent sustained, patient poisoning over many rounds.

**Impact:** All federation members' detection accuracy degrades without any single update appearing anomalous.

### T21: STIX/TAXII Feed Manipulation

**Target:** L4 Memory — threat intel feed ingestion
**MITRE ATLAS:** AML.T0020 (Poison Training Data)

If threat intelligence feeds are compromised, an attacker can inject false indicators:
- Add indicators matching legitimate prompts to cause cross-deployment false positives.
- Remove or downgrade indicators for active attack campaigns.
- Inject high-frequency low-severity indicators to create noise that obscures genuine threats.

**Impact:** Trusted external intelligence becomes a poisoning vector. Multiple AEGIS deployments consuming the same compromised feed are simultaneously affected.

### T22: Indicator Provenance Forgery

**Target:** L4 Memory — `IndicatorSource` enum
**MITRE ATLAS:** AML.T0043 (Craft Adversarial Data)

Indicators track their source (SEED, AUTOMATED, ANALYST, STIX_FEED). The `confirmed` field and `IndicatorSource.ANALYST` carry elevated trust. If an attacker can inject an indicator with `source=ANALYST` and `confirmed=True`:
- The indicator bypasses provenance-based weighting.
- It is immediately eligible for promotion to PERSISTENT phase.
- Generated signatures inherit the elevated trust and are promoted more aggressively.

Currently, there is no authentication or signing on indicator provenance — the enum value is set by whichever code path creates the indicator.

**Impact:** Attacker-controlled indicators receive analyst-level trust, amplifying poisoning impact.

### T23: Fail-Open on Layer Failure

**Target:** `main.py` — pipeline exception handling
**MITRE ATLAS:** AML.T0029 (Denial of ML Service)

If an individual detection layer crashes (uncaught exception, OOM, corrupted state), the pipeline's exception handler determines whether the request is blocked or passed through. A fail-open posture means a layer crash becomes an attack vector: crash the layer, then deliver the payload while detection is offline.

**Mitigation (implemented):** The pipeline now enforces fail-closed semantics. Unhandled exceptions in the detection pipeline return HTTP 503 ("Security pipeline failure — request blocked") with an audit log entry. Additionally, if any layer singleton is None at request time (indicating failed initialization), the pipeline returns 503 before processing begins. This converts layer failure from an attack amplifier into a denial-of-service signal that operators can detect and remediate.

**Residual risk:** Fail-closed converts availability attacks into guaranteed DoS — an attacker who can reliably crash a layer achieves denial of service through the defense mechanism itself. This is an acceptable trade-off: blocking all traffic during a security failure is preferable to passing unscanned traffic.

**Impact:** Without the mitigation, complete bypass of the crashed layer. With the mitigation, denial of service (preferable).

---

## 6. Mitigation Priority Matrix

| ID | Threat | Likelihood | Impact | Priority | Recommended Mitigation |
|----|--------|-----------|--------|----------|----------------------|
| T13 | Config exfiltration | High | Critical | **P0** | Secrets manager integration; strip env from `/proc`; disable debug mode in production |
| T14 | Pattern library exfiltration | High | Critical | **P0** | Encrypt patterns at rest; load into memory only; sandbox `model_path` parameter |
| T18 | Upstream API key compromise | Medium | Critical | **P0** | Short-lived tokens; key rotation; restrict upstream API key permissions to chat completions only |
| T3 | Vault poisoning | Medium | High | **P1** | Confirmation quorum before promotion; rate-limit vault additions per session; anomaly detection on embedding distribution |
| T1 | Regex evasion via Unicode | High | High | **P1** | Unicode normalization (NFKC) before regex evaluation; homoglyph canonicalization |
| T5 | Multi-turn window overflow | High | High | **P1** | Cross-session behavioral tracking; cumulative trajectory analysis beyond 10-turn window |
| T6 | DeBERTa adversarial examples | High | High | **P0** | Ensemble with second classifier; fine-tune on adversarial examples; don't rely solely on public model. Public weights make transfer attacks trivial — this is the highest-risk ML layer. |
| T7 | Canary key extraction | Medium | High | **P1** | Per-tenant canary keys; rotate keys periodically; use asymmetric signatures |
| T16 | FAISS index corruption | Medium | High | **P1** | HMAC integrity tag on metadata sidecar; checksum validation on load; immutable backup copies |
| T10 | TLI manipulation | Medium | Medium | **P2** | Moving average baseline with outlier rejection; separate internal vs external threat rates |
| T11 | Circuit breaker DoS | Medium | Medium | **P2** | Per-client failure tracking; distinguish client-caused vs upstream-caused errors |
| T12 | Session quarantine abuse | Medium | Medium | **P2** | Bind sessions to authenticated identity, not spoofable IDs; rate-limit quarantine entries |
| T8 | Steganographic exfiltration | Low | High | **P2** | Unicode stripping on output; entropy analysis on response text; structured output validation |
| T9 | Streaming window boundary | Low | Medium | **P2** | Full-response validation after stream completes; variable window sizes |
| T4 | Clonal selection manipulation | Medium | Medium | **P2** | Extended affinity testing with adversarial benign set; human review for promoted signatures |
| T15 | Audit log tampering | Low | Medium | **P3** | Write-ahead log to append-only storage; external log shipping (SIEM); tamper-evident checksums |
| T2 | Bloom filter attacks | Low | Low | **P3** | Partitioned Bloom filters; counting filters with saturation monitoring |
| T17 | Dependency confusion | Low | Medium | **P3** | Pin exact package hashes; SBOM attestation; private package registry |
| T19 | Gradient inversion | Low | Medium | **P3** | Reduce epsilon to 1.0; use secure aggregation (MPC); add noise calibration |
| T20 | Byzantine peer poisoning | Low | High | **P3** | Robust aggregation (trimmed mean, Krum); peer reputation scoring |
| T21 | STIX/TAXII manipulation | Low | High | **P3** | Multi-source indicator correlation; provenance verification; indicator confidence decay |
| T22 | Provenance forgery | Medium | Medium | **P2** | Cryptographic signing on indicator provenance; role-based indicator creation. Elevated from P3: forged ANALYST provenance bypasses trust hierarchy. |
| T23 | Fail-open on layer failure | Medium | High | **P1 (mitigated)** | Fail-closed pipeline with 503 on unhandled exceptions; None-check on layer singletons before processing. Implemented in current bootstrap. |

---

## 7. Recommendations for Penetration Testing

### Phase 1: Black-Box External Testing

1. **Regex evasion sweep**: For each pattern category in `patterns.json`, generate 50 evasion variants using homoglyphs, zero-width characters, Unicode normalization differences, and whitespace manipulation. Measure bypass rate. Target: >20% bypass rate indicates insufficient normalization.

2. **Multi-turn escalation**: Design 5 attack sequences that span 15+ turns, each turn individually benign, with the cumulative effect of achieving prompt injection. Confirm the 10-turn window limitation.

3. **Timing side-channel**: Measure response latencies for 1000 requests varying by single characters near known pattern boundaries. Statistical analysis for latency clustering that reveals pattern match/miss decisions.

4. **Quarantine weaponization**: Attempt to quarantine a target session by spoofing session identifiers. Measure whether session binding relies on cryptographic authentication or guessable values.

5. **Circuit breaker manipulation**: Send payloads designed to cause upstream 5xx errors without being blocked by L1-L3. Measure time-to-trip and confirm whether legitimate traffic is denied during OPEN state.

### Phase 2: Gray-Box Testing (Pattern Knowledge)

6. **Decision boundary probing**: With knowledge of `block_threshold=0.85` and `alert_threshold=0.50`, craft payloads that score 0.84 (just below block) and confirm they pass. Verify the gap between 0.50 and 0.85 where prompts are flagged but not blocked.

7. **Canary token forgery**: Using the default canary secret key, generate valid HMAC canary tokens and verify the system reports "canary intact" for a modified system prompt.

8. **Vault poisoning via antibody loop**: Submit 100 crafted "attacks" designed to trigger L3 detection but containing fragments of legitimate business language. After the learning loop fires, measure the false positive rate increase on the benchmark benign corpus.

9. **Streaming window boundary attack**: Craft model responses where PII or system prompt fragments are split exactly at 64-token boundaries (half the 128-token window), verifying that no single window contains a complete detection signature.

### Phase 3: White-Box / Infrastructure Testing

10. **Configuration sensitivity analysis**: For each threshold in `AegisConfig`, identify the minimum perturbation needed to degrade TPR below 85% or increase FPR above 1%. Document which configuration parameters are single points of failure.

11. **FAISS index integrity**: Modify `threat_vault_meta.json` on disk — set all phases to DORMANT, zero all confidence values — and restart AEGIS. Verify detection rates with corrupted vault vs. clean vault.

12. **Pattern file substitution**: Replace `patterns.json` with an empty pattern list and restart. Confirm that L2 detection drops to 0% and verify whether L3/L6 compensate sufficiently.

13. **Supply chain endpoint abuse**: Test `model_path` parameter for path traversal (`../../etc/passwd`), symlink following, and TOCTOU races between integrity verification and model loading.

14. **Module singleton replacement**: In a test environment, monkey-patch `main._innate = None` at runtime and verify that the pipeline fails open (passes all requests) vs. fails closed (blocks all requests). Document the actual fail mode.

### Success Criteria

- **Minimum**: All P0 threats have documented mitigations implemented before production deployment.
- **Target**: P0 and P1 threats mitigated. Pen test bypass rate on the benchmark attack corpus remains below 15% under gray-box conditions.
- **Stretch**: All threats through P2 mitigated. Bypass rate below 10% under white-box conditions (attacker has full source code and configuration access).

---

*This document should be reviewed and updated after each penetration testing engagement. Threats are living — new attack techniques against AI security platforms emerge continuously.*
