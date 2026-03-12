#!/usr/bin/env bash
# AEGIS Docker Entrypoint
#
# Waits for Redis and PostgreSQL to be ready before starting uvicorn.
# Runs Alembic migrations if alembic.ini exists.
# Timeout after 30 seconds per service to fail fast on misconfiguration.

set -euo pipefail

# --------------------------------------------------------------------------
# Wait for a TCP service to be reachable
# Usage: wait_for <host> <port> <service_name>
# --------------------------------------------------------------------------
wait_for() {
    local host="$1"
    local port="$2"
    local name="$3"
    local timeout=30
    local elapsed=0

    echo "Waiting for ${name} at ${host}:${port}..."
    while ! python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('${host}', ${port}))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "ERROR: ${name} at ${host}:${port} not reachable after ${timeout}s"
            exit 1
        fi
        sleep 1
    done
    echo "${name} is ready."
}

# --------------------------------------------------------------------------
# Parse connection URLs and wait for backing services
# --------------------------------------------------------------------------

# Redis — extract host:port from REDIS_URL (redis://host:port/db)
if [ -n "${REDIS_URL:-}" ]; then
    redis_host=$(echo "$REDIS_URL" | sed -E 's|redis://([^:]+):([0-9]+).*|\1|')
    redis_port=$(echo "$REDIS_URL" | sed -E 's|redis://([^:]+):([0-9]+).*|\2|')
    if [ -n "$redis_host" ] && [ -n "$redis_port" ]; then
        wait_for "$redis_host" "$redis_port" "Redis"
    fi
fi

# PostgreSQL — extract host:port from DATABASE_URL
# Handles both postgresql:// and postgresql+asyncpg:// schemes
if [ -n "${DATABASE_URL:-}" ]; then
    pg_host=$(echo "$DATABASE_URL" | sed -E 's|postgresql(\+asyncpg)?://[^@]+@([^:]+):([0-9]+).*|\2|')
    pg_port=$(echo "$DATABASE_URL" | sed -E 's|postgresql(\+asyncpg)?://[^@]+@([^:]+):([0-9]+).*|\3|')
    if [ -n "$pg_host" ] && [ -n "$pg_port" ]; then
        wait_for "$pg_host" "$pg_port" "PostgreSQL"
    fi
fi

# --------------------------------------------------------------------------
# Run Alembic migrations (if configured)
# --------------------------------------------------------------------------
if [ -f "alembic.ini" ]; then
    echo "Running Alembic migrations..."
    alembic upgrade head || echo "WARNING: Alembic migrations failed (continuing startup)"
fi

# --------------------------------------------------------------------------
# Startup info
# --------------------------------------------------------------------------
echo "============================================"
echo "  AEGIS — Adaptive Enterprise Guard"
echo "============================================"
echo "  Upstream URL:   ${AEGIS_UPSTREAM_URL:-not set}"
echo "  ML Models:      ${AEGIS_SKIP_MODEL_LOAD:-false} (skip_model_load)"
echo "  Redis:          ${REDIS_URL:-not configured}"
echo "  PostgreSQL:     $(echo "${DATABASE_URL:-not configured}" | sed -E 's|://([^:]+):[^@]+@|://\1:***@|')"
echo "============================================"

echo "Starting AEGIS..."
exec "$@"
