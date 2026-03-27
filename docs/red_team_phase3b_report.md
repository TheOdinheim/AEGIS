# Red Team Phase 3B: Black-Box Assessment Report

**Date**: 2026-03-27
**Methodology**: Black-box testing — no source code access, only HTTP API interaction
**Scope**: All externally reachable AEGIS endpoints
**Tester**: Automated black-box assessment harness

---

## Executive Summary

Black-box assessment of AEGIS uncovered 7 vulnerabilities (2 HIGH, 5 MEDIUM) across API documentation exposure, authentication handling, path traversal, and information leakage. All findings have been remediated.

**Security posture validated**: All prompt injection vectors (classic, base64, role confusion, Skeleton Key, multi-turn, indirect, Unicode) were correctly blocked. Authentication rejects invalid tokens, rate limiting escalates TLI appropriately, and standard endpoints resist path traversal.

---

## Findings

### RT-P3B-001 — OpenAPI Documentation Exposure (HIGH)

**Endpoint**: `GET /docs`, `GET /redoc`, `GET /openapi.json`
**Impact**: Full API schema exposed without authentication, revealing all endpoint paths, parameters, request/response models, and internal data structures. Provides complete attack surface map to adversaries.
**Fix**: Disabled FastAPI auto-generated docs: `docs_url=None, redoc_url=None, openapi_url=None` in app constructor.
**File**: `main.py`

### RT-P3B-002 — Case-Sensitive Bearer Token Parsing (MEDIUM)

**Endpoint**: All authenticated endpoints
**Impact**: RFC 6750 specifies case-insensitive "Bearer" scheme. Sending `bearer` or `BEARER` prefix bypassed authentication checks in both `_is_authenticated()` and `BarrierLayer._extract_api_key()`.
**Fix**: `auth.lower().startswith("bearer ")` in both locations.
**Files**: `main.py`, `layers/barrier.py`

### RT-P3B-003 — Path Traversal in /v1/admin/restore (HIGH)

**Endpoint**: `POST /v1/admin/restore`
**Impact**: `backup_path` parameter accepted arbitrary filesystem paths. An authenticated attacker could read arbitrary files (e.g., `/etc/passwd`, `/etc/shadow`) by crafting a malicious backup path.
**Fix**: Resolved path must be within `/tmp/aegis-backups/` directory. Traversal attempts return 400.
**File**: `main.py`

### RT-P3B-004 — Config Validation Warning Leakage (MEDIUM)

**Endpoint**: `GET /v1/admin/config-validation`
**Impact**: Warning messages exposed sensitive configuration details (weak API keys, default credentials, missing production settings). Information useful for tailoring attacks.
**Fix**: Warnings redacted to opaque labels (`warning_1`, `warning_2`, ...). Only count is visible.
**File**: `main.py`

### RT-P3B-005 — Vault Stats Payload Exposure (MEDIUM)

**Endpoint**: `GET /v1/vault/stats`
**Impact**: `payload_summary` field in top indicators exposed actual attack text (first 100 chars). Leaks attack patterns and potentially sensitive content from intercepted prompts.
**Fix**: Replaced `payload_summary` with `payload_length` (integer only).
**File**: `layers/memory/threat_vault.py`

### RT-P3B-006 — STIX Indicator Stored XSS (MEDIUM)

**Endpoint**: `POST /v1/threat-intel/ingest`
**Impact**: STIX indicator `name` and `description` fields accepted raw HTML. If rendered in dashboard or exported, could execute stored XSS attacks.
**Fix**: Strip all HTML tags via `re.sub(r"<[^>]+>", "", field)` on ingest.
**File**: `services/threat_intel.py`

### RT-P3B-007 — Backup Path Injection and Filesystem Exposure (MEDIUM)

**Endpoint**: `POST /v1/admin/backup`
**Impact**: `output_path` parameter allowed arbitrary filesystem write locations. Response included full filesystem path, revealing directory structure.
**Fix**: Ignore user-supplied `output_path`; always write to designated backup directory. Response shows filename only, not full path.
**File**: `main.py`

---

## Confirmed Working (Not Findings)

| Test | Result |
|------|--------|
| Classic prompt injection ("ignore all previous instructions") | Blocked (403) |
| Base64-encoded injection | Blocked (403) |
| Role confusion (user claiming system role) | Blocked (403) |
| Skeleton Key ("augment your guidelines") | Blocked (403) |
| Multi-turn escalation (5-message sequence) | Blocked (403) |
| Indirect injection (hidden in document) | Blocked (403) |
| Unicode obfuscated injection | Blocked (403) |
| Empty/null auth tokens | Rejected (401) |
| SQL injection in auth header | Rejected (401) |
| Null byte in auth token | Rejected (401) |
| Oversized auth token (10KB) | Rejected (401) |
| Rate limit burst (100 rapid requests) | Rate limited, TLI escalated to RED |
| Path traversal on standard endpoints | Rejected |
| Admin endpoints without auth | Rejected (401) |

---

## Remediation Summary

| ID | Severity | Status | Regression Test |
|----|----------|--------|-----------------|
| RT-P3B-001 | HIGH | Fixed | `test_openapi_docs_disabled` |
| RT-P3B-002 | MEDIUM | Fixed | `test_bearer_case_insensitive_*` |
| RT-P3B-003 | HIGH | Fixed | `test_restore_path_traversal_*` |
| RT-P3B-004 | MEDIUM | Fixed | `test_config_validation_warnings_redacted` |
| RT-P3B-005 | MEDIUM | Fixed | `test_vault_stats_no_payload_text` |
| RT-P3B-006 | MEDIUM | Fixed | `test_stix_html_sanitization` |
| RT-P3B-007 | MEDIUM | Fixed | `test_backup_ignores_custom_path` |
