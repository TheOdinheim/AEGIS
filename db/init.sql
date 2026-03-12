-- AEGIS PostgreSQL Schema
-- Applied automatically on first container startup via
-- /docker-entrypoint-initdb.d/init.sql
--
-- ASSUMED-BREACH POSTURE: Raw prompts are NEVER stored. Only SHA-256
-- hashes are persisted. All PII fields are redacted before insertion.

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- -------------------------------------------------------------------------
-- Helper: auto-update updated_at on row modification
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- -------------------------------------------------------------------------
-- tenants — multi-tenant configuration
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL UNIQUE,
    api_key_hash    TEXT,                         -- bcrypt hash of tenant API key
    rate_limit_rpm  INTEGER NOT NULL DEFAULT 60,
    rate_limit_burst INTEGER NOT NULL DEFAULT 10,
    block_threshold REAL NOT NULL DEFAULT 0.85,
    alert_threshold REAL NOT NULL DEFAULT 0.50,
    allowed_models  TEXT[] DEFAULT '{}',
    policy_tier     TEXT NOT NULL DEFAULT 'standard'
                    CHECK (policy_tier IN ('standard', 'strict', 'permissive', 'custom')),
    custom_policy   JSONB DEFAULT '{}',
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Default development tenant
INSERT INTO tenants (name, api_key_hash, policy_tier)
VALUES ('dev', NULL, 'standard')
ON CONFLICT (name) DO NOTHING;

-- -------------------------------------------------------------------------
-- audit_log — every request/response cycle
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    request_id          UUID NOT NULL DEFAULT uuid_generate_v4(),
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id           UUID REFERENCES tenants(tenant_id),
    user_id             TEXT,
    session_id          TEXT,
    source_ip           INET,
    model               TEXT,
    prompt_hash         TEXT NOT NULL,              -- SHA-256, never raw prompt
    token_count         INTEGER,

    -- L2 innate results
    innate_is_threat    BOOLEAN,
    innate_confidence   REAL,
    innate_scanners     JSONB DEFAULT '{}',         -- per-scanner summary

    -- L3 adaptive results
    adaptive_is_threat  BOOLEAN,
    adaptive_confidence REAL,
    adaptive_mcav       REAL,
    adaptive_analyzers  JSONB DEFAULT '{}',         -- per-analyzer summary

    -- L5 output validation
    output_pii_found    BOOLEAN DEFAULT FALSE,
    output_toxicity     REAL,
    output_leakage      BOOLEAN DEFAULT FALSE,
    output_validation   JSONB DEFAULT '{}',

    -- Decision
    final_action        TEXT NOT NULL DEFAULT 'allow'
                        CHECK (final_action IN ('allow', 'block', 'alert', 'quarantine')),
    block_reason        TEXT,
    threat_level        INTEGER DEFAULT 1
                        CHECK (threat_level BETWEEN 1 AND 5),

    -- Performance
    latency_innate_ms   REAL,
    latency_adaptive_ms REAL,
    latency_output_ms   REAL,
    latency_total_ms    REAL,

    -- System state
    circuit_state       TEXT DEFAULT 'closed'
                        CHECK (circuit_state IN ('closed', 'open', 'half_open'))
);

-- Indexes for common query patterns
CREATE INDEX idx_audit_timestamp ON audit_log (timestamp DESC);
CREATE INDEX idx_audit_tenant ON audit_log (tenant_id, timestamp DESC);
CREATE INDEX idx_audit_session ON audit_log (session_id, timestamp DESC);
CREATE INDEX idx_audit_blocked ON audit_log (final_action, timestamp DESC)
    WHERE final_action != 'allow';
CREATE INDEX idx_audit_request_id ON audit_log (request_id);

-- -------------------------------------------------------------------------
-- threat_indicators — persistent vault metadata (L4)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS threat_indicators (
    indicator_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mitre_tactic    TEXT,
    threat_category TEXT,
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    source          TEXT NOT NULL DEFAULT 'adaptive_detection',
    prompt_hash     TEXT NOT NULL,                  -- SHA-256, never raw
    embedding_ref   TEXT,                           -- reference to FAISS index position
    status          TEXT NOT NULL DEFAULT 'acute'
                    CHECK (status IN ('acute', 'persistent', 'dormant')),
    hit_count       INTEGER NOT NULL DEFAULT 1,
    first_seen      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_count    INTEGER NOT NULL DEFAULT 1,     -- distinct sources reporting
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_threat_status ON threat_indicators (status);
CREATE INDEX idx_threat_tactic ON threat_indicators (mitre_tactic);
CREATE INDEX idx_threat_last_seen ON threat_indicators (last_seen DESC);
CREATE INDEX idx_threat_prompt_hash ON threat_indicators USING hash (prompt_hash);

CREATE TRIGGER threat_indicators_updated_at
    BEFORE UPDATE ON threat_indicators
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- -------------------------------------------------------------------------
-- signatures — clonal selection output (L2/L4)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signatures (
    signature_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_type        TEXT NOT NULL DEFAULT 'regex'
                        CHECK (pattern_type IN ('regex', 'yara', 'embedding')),
    pattern_text        TEXT NOT NULL,
    source_indicator_id UUID REFERENCES threat_indicators(indicator_id)
                        ON DELETE SET NULL,
    affinity_score      REAL CHECK (affinity_score BETWEEN 0.0 AND 1.0),
    tpr                 REAL CHECK (tpr BETWEEN 0.0 AND 1.0),
    fpr                 REAL CHECK (fpr BETWEEN 0.0 AND 1.0),
    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'deprecated', 'candidate')),
    match_count         INTEGER NOT NULL DEFAULT 0,
    last_matched        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sig_status ON signatures (status);
CREATE INDEX idx_sig_source ON signatures (source_indicator_id);
CREATE INDEX idx_sig_pattern_trgm ON signatures USING gin (pattern_text gin_trgm_ops);

CREATE TRIGGER signatures_updated_at
    BEFORE UPDATE ON signatures
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- -------------------------------------------------------------------------
-- circuit_breaker_events — L7 state transitions
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    endpoint        TEXT NOT NULL,
    previous_state  TEXT NOT NULL
                    CHECK (previous_state IN ('closed', 'open', 'half_open')),
    new_state       TEXT NOT NULL
                    CHECK (new_state IN ('closed', 'open', 'half_open')),
    trigger_reason  TEXT,
    failure_rate    REAL,
    window_seconds  INTEGER,
    cooldown_seconds INTEGER
);

CREATE INDEX idx_cb_endpoint ON circuit_breaker_events (endpoint, timestamp DESC);
CREATE INDEX idx_cb_timestamp ON circuit_breaker_events (timestamp DESC);
