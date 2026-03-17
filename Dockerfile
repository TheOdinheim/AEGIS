# AEGIS Dockerfile — Multi-stage production build
#
# ASSUMED-BREACH POSTURE: The container image is a trust boundary. We use a
# minimal base image, run as non-root, and never embed secrets in the image.
# DeBERTa and MiniLM models are downloaded at build time for deterministic
# startup — no network access required at runtime. In production, this image
# runs behind gVisor (medium threat) or Firecracker (high threat) for
# additional kernel-level isolation.
#
# Build: docker build -t aegis .
# Run:   docker run -p 8000:8000 --env-file .env aegis

# =========================================================================
# Stage 1: Build — install dependencies and download ML models
# =========================================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies for building C extensions (numpy, faiss, onnxruntime)
# and Tesseract OCR with CJK + multilingual language packs for image text extraction
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
        tesseract-ocr-jpn \
        tesseract-ocr-kor \
        tesseract-ocr-ara \
        tesseract-ocr-hin \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies into a virtual environment for clean copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-docker.txt ./
RUN pip install -r requirements.txt -r requirements-docker.txt

# Download ML models at build time so they're baked into the image.
# Models are cached under /opt/models and will be available at runtime
# without any network access.
#
# DeBERTa-v3-base prompt injection classifier (~400MB)
# all-MiniLM-L6-v2 sentence embeddings for threat vault (~80MB)
#
# No try/except — if models fail to download, the build MUST fail.
# An image without models is a security gap, not graceful degradation.
RUN mkdir -p /opt/models

ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProtectAI/deberta-v3-base-prompt-injection-v2'); AutoModelForSequenceClassification.from_pretrained('ProtectAI/deberta-v3-base-prompt-injection-v2')"

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN ls -la /opt/models && du -sh /opt/models

# =========================================================================
# Stage 2: Runtime — minimal image with only what's needed
# =========================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime-only system dependencies (no build-essential)
# Tesseract OCR + language packs must be in runtime image (binary executed at request time)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
        tesseract-ocr-jpn \
        tesseract-ocr-kor \
        tesseract-ocr-ara \
        tesseract-ocr-hin \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 aegis && \
    useradd --uid 1000 --gid aegis --shell /bin/bash --create-home aegis

# Copy virtual environment with all installed packages from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy baked-in ML models from builder
COPY --from=builder /opt/models /opt/models
ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers

# PYTHONPATH: the application code lives at /app/aegis/ and imports use
# "from aegis.config import ..." so /app must be on the path.
ENV PYTHONPATH=/app

WORKDIR /app/aegis

# Copy application code (respects .dockerignore)
COPY . .

# Copy entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create directories for runtime state and set ownership
RUN mkdir -p /app/aegis/logs /app/aegis/data && \
    chown -R aegis:aegis /app /opt/models

USER aegis

EXPOSE 8000

# Health check against the unauthenticated health endpoint
# 60s start period allows DeBERTa to load (~30s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "aegis.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
