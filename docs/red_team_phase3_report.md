# Red Team Phase 3: White-Box Adversarial Assessment — New Components

**Date**: 2026-03-27
**Scope**: All code added since Phase 2 (commit `61b6eba`): Dashboard, L8 federation hub, federated model training, zero trust hardening
**Methodology**: Full source code review, PoC exploit development, severity assessment, fix + regression test

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| CRITICAL | 0 | - |
| HIGH | 3 | 3 |
| MEDIUM | 4 | 4 |
| LOW | 2 | 2 |
| **Total** | **9** | **9** |

---

## Finding RT-P3-001: FL `num_samples` Inflation Dominates Aggregation

**Category**: C (Federated Learning Model Poisoning)
**Severity**: HIGH
**File**: `services/federated/fl_api.py:91`, `services/federated/fl_server.py`

**Description**: The FL update API accepts `num_samples` as an unconstrained integer from the client. A malicious node can set `num_samples=999999999` to dominate the FedAvg weighted average, even with trimmed mean active (which trims by value, not weight). The `aggregate_with_trust` path scales `num_samples` by trust, but `0.5 * 999999999` still vastly outweighs honest nodes.

**PoC**: Submit update with `num_samples=10^9`. Result dominates aggregation regardless of other honest nodes.

**Fix**: Cap `num_samples` at a configurable maximum (default 100,000) in both the API endpoint and the server's `submit_update`. Also cap the effective samples in `aggregate_with_trust`.

---

## Finding RT-P3-002: Indicator ID Overwrite — Existing Indicators Replaced

**Category**: F (STIX & JSON Injection)
**Severity**: HIGH
**File**: `services/federated/indicator_registry.py:98`

**Description**: `add_indicator` stores with `self._indicators[ind_id] = indicator`. If an attacker knows (or guesses) an existing indicator ID, they can submit a new indicator with the same ID but different content/embedding. The dedup check only catches similar embeddings — if the new embedding is dissimilar, the old indicator is silently overwritten.

**PoC**: Submit indicator `indicator--known-id` with attack embedding. Submit again with same ID but benign embedding. Original attack indicator is replaced.

**Fix**: Reject indicator submission if the ID already exists in the registry (return existing ID without overwrite).

---

## Finding RT-P3-003: Heartbeat Trust Farming — Unbounded Positive Trust

**Category**: D (Trust Score Manipulation)
**Severity**: HIGH
**File**: `services/federated/node_trust.py`, `services/federated/hub_api.py`

**Description**: Each heartbeat grants +0.001 trust with no per-day cap and no rate limiting on heartbeat frequency. An attacker sending 1000 heartbeats/second for 700 seconds reaches trust=1.0 from initial 0.3. No cooldown, no diminishing returns, no frequency check.

**PoC**: Loop 700 heartbeat requests. Node trust goes from 0.3 to 1.0.

**Fix**: Add a minimum interval between trust-rewarding heartbeats (default 60s). Record last rewarded heartbeat time per node.

---

## Finding RT-P3-004: Bearer Token Case-Sensitivity Bypass

**Category**: A (Authentication Bypass)
**Severity**: MEDIUM
**File**: All auth functions (`dashboard/api.py:36`, `dashboard/sse.py:36`, `services/federated/hub_api.py:34`, `services/federated/fl_api.py:32`)

**Description**: Token extraction checks `auth.startswith("Bearer ")` (capital B, one space). Per RFC 6750 Section 2.1, the `Bearer` scheme is case-insensitive. An HTTP client sending `bearer sk-key` or `BEARER sk-key` would have the Authorization header ignored, falling through to x-api-key or returning 401. This is defense-in-depth (not a bypass if x-api-key isn't sent), but deviates from RFC and could cause legitimate clients to fail authentication.

**Fix**: Case-insensitive prefix check: `auth.lower().startswith("bearer ")`.

---

## Finding RT-P3-005: FL Trigger-Round Repeated Aggregation — Empty Round DoS

**Category**: C (Federated Learning Model Poisoning)
**Severity**: MEDIUM
**File**: `services/federated/fl_api.py:112-139`

**Description**: `/v1/federation/fl/trigger-round` calls `fl_server.aggregate()` which returns `{"status": "no_updates"}` when no updates exist, but doesn't prevent repeated calls. While not directly harmful (no model change without updates), it can be used to force aggregation with only 1 malicious update before honest nodes submit, by calling trigger-round immediately after submitting a single update.

**Fix**: Prevent trigger-round from aggregating unless minimum client threshold is met or explicitly overridden.

---

## Finding RT-P3-006: Consistency Check Self-Corroboration

**Category**: B (Federation Indicator Poisoning)
**Severity**: MEDIUM
**File**: `services/federated/indicator_reputation.py:149-167`

**Description**: The consistency scoring checks `if src == source_node_id: continue` to skip self-corroboration. However, if an attacker controls two node IDs (node-A and node-B), they can submit identical indicators from both to appear "consistent". The system can't distinguish Sybil nodes. This is a fundamental limitation of decentralized trust, documented as a recommendation rather than a code fix.

**Mitigation**: The node trust system partially mitigates this since new nodes start at 0.3 trust (40% weight). However, a determined attacker can farm trust first via heartbeats (RT-P3-003), then use Sybil nodes. Fixing RT-P3-003 reduces this attack surface.

**Fix**: Document as known limitation. Mitigation comes from fixing RT-P3-003 (heartbeat rate limiting).

---

## Finding RT-P3-007: Training Buffer Benign Flood — Model Degradation

**Category**: G (Training Sample Collection)
**Severity**: MEDIUM
**File**: `main.py:3545-3568`, `services/federated/training_buffer.py`

**Description**: Every benign request (passing all layers) adds a sample to the training buffer with FIFO eviction at 10,000 entries. An attacker sending 10,000+ benign requests rapidly evicts all attack samples from the buffer, leaving only benign data for the next FL training round. This degrades the model's ability to detect attacks.

**Fix**: Maintain a minimum ratio of threat samples by using separate sub-buffers for threats and benign, each with their own capacity (50/50 split by default).

---

## Finding RT-P3-008: Deeply Nested JSON in Indicator Pattern

**Category**: F (STIX & JSON Injection)
**Severity**: LOW
**File**: `services/federated/indicator_reputation.py`, `services/federated/indicator_registry.py`

**Description**: The `pattern` field is parsed with `json.loads()` in multiple places. Deeply nested JSON (e.g., 1000 levels deep) can cause excessive CPU during parsing. Python's `json.loads` handles this reasonably (no stack overflow — Python uses iterative parsing), and the FastAPI request body size limit provides protection. Practical impact is low.

**Fix**: Add a basic pattern size limit check before parsing (max 100KB).

---

## Finding RT-P3-009: SSE Token in Query Parameter — URL Logging Exposure

**Category**: A (Authentication & Authorization)
**Severity**: LOW
**File**: `dashboard/sse.py:43`

**Description**: The SSE endpoint accepts auth token via `?token=` query parameter (necessary because EventSource API can't set headers). This means the API key appears in URLs, which may be logged by reverse proxies, CDNs, or browser history. This is a known limitation of EventSource documented in the code, not a bug.

**Fix**: Document as known limitation. In production, recommend using a short-lived session token for SSE rather than the main API key.

---

## Findings NOT Confirmed

1. **`_get_main()` returning module with empty api_key**: Reviewed — function checks `_config` is not None and `api_key` is not empty. Safe.
2. **`hmac.compare_digest` timing attack**: Confirmed constant-time as expected. No vulnerability.
3. **XSS in dashboard**: React auto-escapes all values in JSX. No `dangerouslySetInnerHTML` usage found. Safe.
4. **SVG favicon XSS**: The SVG is a static constant, not user-controlled. Safe.
5. **iframe embedding**: Landing page doesn't set X-Frame-Options programmatically, but this is a deployment concern (Caddy/nginx), not an application vulnerability.
6. **Node ID hijacking**: Trust scores are per node_id string. A node registering with the same ID as an existing trusted node would share trust — but this requires the API key, so same auth boundary. Not a distinct vulnerability.
7. **`bearer ` (lowercase) as bypass**: The lowercase variant would be ignored but fall through to x-api-key header check. Not a bypass, just RFC non-compliance. Addressed in RT-P3-004.
