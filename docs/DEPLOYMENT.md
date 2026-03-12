# AEGIS Deployment Guide

Production deployment guide for AEGIS — Adaptive Enterprise Guard for Intelligent Systems.

## Prerequisites

- **Python 3.12+** (or Docker)
- **Docker + Docker Compose** (recommended for production)
- **4GB+ RAM** (DeBERTa v3 + MiniLM models require ~2GB)
- **Ollama** (for local models) or **OpenAI/Anthropic API key** (for cloud models)
- **Redis** (production rate limiting and event bus)
- **PostgreSQL** (production audit logging and persistence)

## Quick Start

### Docker Compose (Recommended)

```bash
git clone https://github.com/your-org/aegis.git && cd aegis
cp .env.example .env
# Edit .env with your API keys

docker compose up -d
# Verify
curl http://localhost:8000/health
```

### Local Development

```bash
git clone https://github.com/your-org/aegis.git && cd aegis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download ML models (~500MB, first run only)
python -c "from transformers import pipeline; pipeline('text-classification', 'ProtectAI/deberta-v3-base-prompt-injection-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Configure
export AEGIS_UPSTREAM_URL=https://api.openai.com
export AEGIS_UPSTREAM_API_KEY=sk-your-key-here
export AEGIS_API_KEY=$(openssl rand -hex 32)

# Launch
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Production Checklist

Complete all items before deploying to production:

1. **Change AEGIS_API_KEY** — Generate a cryptographically random key:
   ```bash
   export AEGIS_API_KEY=$(openssl rand -hex 32)
   ```

2. **Change AEGIS_CANARY_SECRET_KEY** — Used for system prompt integrity verification:
   ```bash
   export AEGIS_CANARY_SECRET_KEY=$(openssl rand -hex 32)
   ```

3. **Set AEGIS_AGENT_SIGNING_KEY** — For persistent agent JWT tokens across restarts:
   ```bash
   export AEGIS_AGENT_SIGNING_KEY=$(openssl rand -hex 32)
   ```

4. **Configure upstream model** — Set your AI provider:
   ```bash
   export AEGIS_UPSTREAM_URL=https://api.openai.com
   export AEGIS_UPSTREAM_API_KEY=sk-your-key
   ```

5. **Enable PostgreSQL** — With SSL for audit logging and persistence:
   ```bash
   export DATABASE_URL=postgresql://user:pass@host:5432/aegis?sslmode=require
   ```

6. **Enable Redis** — For distributed rate limiting and event bus:
   ```bash
   export REDIS_URL=redis://host:6379/0
   ```

7. **Set production environment**:
   ```bash
   export AEGIS_ENV=production
   ```

8. **Configure per-tenant rate limits** — Insert tenant configurations into the `tenants` PostgreSQL table.

9. **Set up monitoring** — Configure Prometheus to scrape `/metrics` (authenticated):
   ```yaml
   # prometheus.yml
   scrape_configs:
     - job_name: aegis
       bearer_token: "your-aegis-api-key"
       static_configs:
         - targets: ["aegis:8000"]
   ```

10. **Configure vault backup** — Schedule daily backups:
    ```bash
    # crontab
    0 2 * * * curl -X POST http://localhost:8000/v1/admin/backup \
      -H "Authorization: Bearer $AEGIS_API_KEY" \
      -H "Content-Type: application/json"
    ```

11. **Review detection thresholds** — Adjust for your use case:
    ```bash
    export AEGIS_INNATE_BLOCK_THRESHOLD=0.85    # Lower = more aggressive
    export AEGIS_INNATE_ALERT_THRESHOLD=0.50    # Lower = more alerts
    ```

12. **Enable deep health monitoring**:
    ```bash
    export AEGIS_DEEP_HEALTH_ENABLED=true
    ```

## Environment Variable Reference

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `AEGIS_API_KEY` | (empty) | **Yes** | Gateway API key for client authentication |
| `AEGIS_UPSTREAM_URL` | `https://api.openai.com` | **Yes** | Upstream AI model provider URL |
| `AEGIS_UPSTREAM_API_KEY` | (empty) | **Yes** | API key for upstream provider |
| `AEGIS_ENV` | (empty) | No | Set to `production` for production checks |
| `AEGIS_HOST` | `0.0.0.0` | No | Bind address |
| `AEGIS_PORT` | `8000` | No | Bind port |
| `AEGIS_LOG_LEVEL` | `INFO` | No | Logging level |
| `AEGIS_MODEL` | (empty) | No | Model name for upstream (e.g., `gpt-4o`) |
| `AEGIS_CANARY_SECRET_KEY` | (default) | **Prod** | HMAC secret for canary tokens |
| `AEGIS_AGENT_SIGNING_KEY` | (auto) | **Prod** | JWT signing key for agent identity |
| `AEGIS_SKIP_MODEL_LOAD` | `false` | No | Skip DeBERTa/MiniLM loading (tests, CI) |
| `AEGIS_INNATE_BLOCK_THRESHOLD` | `0.85` | No | Confidence threshold for blocking |
| `AEGIS_INNATE_ALERT_THRESHOLD` | `0.50` | No | Confidence threshold for alerting |
| `AEGIS_BARRIER_RATE_LIMIT_RPM` | `60` | No | Requests per minute per API key |
| `AEGIS_BARRIER_RATE_LIMIT_BURST` | `10` | No | Burst allowance above rate limit |
| `AEGIS_BARRIER_MAX_TOKENS_PER_REQUEST` | `128000` | No | Maximum input tokens |
| `AEGIS_POLICY_BACKEND` | `python` | No | Policy engine: `python` or `opa` |
| `AEGIS_POLICY_OPA_URL` | `http://localhost:8181` | No | OPA server URL |
| `AEGIS_INSTANCE_ID` | `local` | No | Instance ID for federated learning |
| `AEGIS_DP_EPSILON` | `3.0` | No | Differential privacy epsilon |
| `AEGIS_DP_DELTA` | `1e-5` | No | Differential privacy delta |
| `AEGIS_DP_BUDGET` | `100.0` | No | Total privacy budget |
| `AEGIS_DEEP_HEALTH_ENABLED` | `false` | No | Enable background health monitoring |
| `AEGIS_BACKUP_DIR` | `/tmp/aegis-backups` | No | Vault backup directory |
| `AEGIS_SUPPLY_CHAIN_MODEL_BASE` | `/tmp/aegis-models` | No | Allowed model directory |
| `REDIS_URL` | (empty) | **Prod** | Redis connection URL |
| `DATABASE_URL` | (empty) | **Prod** | PostgreSQL connection URL |

## Scaling Guide

### Horizontal Scaling with Kubernetes

AEGIS is designed for horizontal scaling via Kubernetes:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: aegis
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: aegis
          image: aegis:latest
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          env:
            - name: REDIS_URL
              value: redis://redis:6379/0
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: aegis-secrets
                  key: database-url
```

**Key considerations:**
- Use **StatefulSet** for vault persistence (FAISS index per pod)
- **Redis** for shared rate limits across pods (required for horizontal scaling)
- **PostgreSQL** for shared audit logs, tenant config, and threat indicator persistence
- Resource requests: **2 CPU, 4GB RAM** per pod minimum
- DeBERTa model loads into memory per pod (~1.5GB)

### Performance Targets

| Metric | Target |
|--------|--------|
| Total gateway overhead | 30-150ms |
| L2 innate fast path | <5ms |
| L3 adaptive slow path | 10-50ms async |
| Throughput per node | 10,000+ RPS |
| Availability | 99.95% |

## Backup and Restore

### Backup

```bash
# Manual backup
curl -X POST http://localhost:8000/v1/admin/backup \
  -H "Authorization: Bearer $AEGIS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"output_path": "/backups/vault_backup.json"}'

# Recommended: daily cron job at 2 AM
0 2 * * * curl -X POST http://localhost:8000/v1/admin/backup \
  -H "Authorization: Bearer $AEGIS_API_KEY"
```

### Restore

```bash
curl -X POST http://localhost:8000/v1/admin/restore \
  -H "Authorization: Bearer $AEGIS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"backup_path": "/backups/vault_backup.json"}'
```

**Note:** Embeddings are NOT stored in backups. On restore, indicators are loaded with placeholder embeddings. Re-embedding requires the embedding model.

## Troubleshooting

### 1. Model Load Failure

**Symptom:** Startup hangs or crashes during model download.

**Fix:**
```bash
# Pre-download models
python -c "from transformers import pipeline; pipeline('text-classification', 'ProtectAI/deberta-v3-base-prompt-injection-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Or skip models for testing
export AEGIS_SKIP_MODEL_LOAD=true
```

### 2. High False Positive Rate

**Symptom:** Legitimate requests blocked.

**Fix:**
- Increase `AEGIS_INNATE_BLOCK_THRESHOLD` (e.g., 0.90)
- Review `/v1/audit/recent` for blocked requests
- Add tenant-specific thresholds for business-critical applications
- Use tolerance training: flag false positives in audit logs

### 3. Redis Connection Issues

**Symptom:** Rate limiting not working across instances.

**Fix:**
```bash
# Verify Redis connectivity
redis-cli -u $REDIS_URL ping

# Check deep health
curl http://localhost:8000/v1/admin/deep-health \
  -H "Authorization: Bearer $AEGIS_API_KEY"
```

### 4. PostgreSQL Connection Issues

**Symptom:** Audit logs not persisting.

**Fix:**
```bash
# Verify PostgreSQL connectivity
psql $DATABASE_URL -c "SELECT 1"

# Check schema
psql $DATABASE_URL -f db/init.sql
```

### 5. Circuit Breaker Stuck Open

**Symptom:** All requests returning 503, upstream model unreachable.

**Fix:**
```bash
# Check breaker state
curl http://localhost:8000/v1/admin/deep-health \
  -H "Authorization: Bearer $AEGIS_API_KEY" | jq '.component_results.circuit_breaker'

# Verify upstream is reachable
curl $AEGIS_UPSTREAM_URL/v1/models \
  -H "Authorization: Bearer $AEGIS_UPSTREAM_API_KEY"
```

The circuit breaker will automatically probe and recover when the upstream model becomes available again (30s cooldown, exponential backoff up to 5 minutes).

## Monitoring

### Recommended Grafana Dashboard Panels

| Panel | Metric | Description |
|-------|--------|-------------|
| Request Rate | `rate(aegis_requests_total[5m])` | Requests per second |
| Latency P50/P95/P99 | `histogram_quantile(0.95, aegis_request_latency_seconds)` | Request latency percentiles |
| Threat Detection Rate | `rate(aegis_threats_detected_total[5m])` | Threats detected per second |
| Block Rate | `rate(aegis_blocks_total[5m])` | Requests blocked per second |
| TLI Level | `aegis_threat_level` | Current Threat Level Indicator |
| Circuit Breaker State | `aegis_circuit_breaker_state` | 0=closed, 1=open, 2=half_open |
| Vault Size | `aegis_vault_size` | Threat indicators by lifecycle phase |
| PII Redactions | `rate(aegis_pii_redactions_total[5m])` | PII detections per second |
| Stream Interruptions | `rate(aegis_stream_interruptions_total[5m])` | Streams terminated for safety |
| Privacy Budget | Custom via `/v1/federated/privacy-budget` | DP budget remaining |
