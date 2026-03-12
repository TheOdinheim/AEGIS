"""
Prometheus Metrics Middleware — Collects per-request and per-layer metrics.

Tracks: request count, latency histograms per layer, threat detection rates,
circuit breaker state changes, rate limit triggers, active connections,
per-tenant request counts, threat level, vault size, scanner/analyzer
breakdown, output cascade stages, session quarantine, event bus activity,
and upstream model latency/errors.

Exposed on GET /metrics in Prometheus exposition format.

ASSUMED-BREACH POSTURE: Metrics are a side channel. An attacker observing
metrics could infer detection thresholds, traffic patterns, and which
attacks succeed. In production, the /metrics endpoint is served on a
separate internal port (not exposed through the public gateway) and
requires authentication. Metric labels never include PII, prompt content,
or API keys. Histogram buckets are designed to not leak exact threshold
values.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

# ---------------------------------------------------------------------------
# Request-level metrics
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "aegis_requests_total",
    "Total requests processed by AEGIS",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "aegis_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ACTIVE_CONNECTIONS = Gauge(
    "aegis_active_connections",
    "Currently active connections",
)

# ---------------------------------------------------------------------------
# Per-tenant metrics
# ---------------------------------------------------------------------------

TENANT_REQUESTS = Counter(
    "aegis_tenant_requests_total",
    "Requests per tenant",
    ["tenant_id", "status"],
)

# ---------------------------------------------------------------------------
# Per-layer latency
# ---------------------------------------------------------------------------

LAYER_LATENCY = Histogram(
    "aegis_layer_latency_seconds",
    "Per-layer processing latency",
    ["layer"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5),
)

INNATE_SCANNER_LATENCY = Histogram(
    "aegis_innate_scanner_latency_seconds",
    "Per-scanner latency within L2 innate detection",
    ["scanner"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025),
)

ADAPTIVE_ANALYZER_LATENCY = Histogram(
    "aegis_adaptive_analyzer_latency_seconds",
    "Per-analyzer latency within L3 adaptive analysis",
    ["analyzer"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

UPSTREAM_LATENCY = Histogram(
    "aegis_upstream_latency_seconds",
    "Upstream model request latency",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ---------------------------------------------------------------------------
# Threat detection metrics
# ---------------------------------------------------------------------------

THREATS_DETECTED = Counter(
    "aegis_threats_detected_total",
    "Threats detected by layer",
    ["layer", "category"],
)

POLICY_DECISIONS = Counter(
    "aegis_policy_decisions_total",
    "Policy engine decisions",
    ["action", "tier"],
)

BLOCKS_TOTAL = Counter(
    "aegis_blocks_total",
    "Total blocked requests",
    ["layer", "reason"],
)

THREAT_LEVEL = Gauge(
    "aegis_threat_level",
    "Current Threat Level Indicator (1=GREEN, 2=BLUE, 3=YELLOW, 4=ORANGE, 5=RED)",
)

# ---------------------------------------------------------------------------
# Circuit breaker metrics
# ---------------------------------------------------------------------------

CIRCUIT_BREAKER_STATE = Gauge(
    "aegis_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["endpoint"],
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "aegis_circuit_breaker_trips_total",
    "Circuit breaker trip events",
    ["endpoint"],
)

# ---------------------------------------------------------------------------
# Output validation metrics
# ---------------------------------------------------------------------------

PII_REDACTIONS = Counter(
    "aegis_pii_redactions_total",
    "PII entities redacted from output",
    ["entity_type"],
)

TOXICITY_DETECTIONS = Counter(
    "aegis_toxicity_detections_total",
    "Toxic content detected in output",
    ["category"],
)

OUTPUT_CASCADE_STAGE = Counter(
    "aegis_output_cascade_stage_total",
    "Output validation cascade stage activations",
    ["stage", "result"],
)

# ---------------------------------------------------------------------------
# Streaming interception metrics
# ---------------------------------------------------------------------------

STREAM_INTERRUPTIONS = Counter(
    "aegis_stream_interruptions_total",
    "Stream interruptions by reason",
    ["reason"],
)

STREAM_WINDOWS_EVALUATED = Counter(
    "aegis_stream_windows_evaluated_total",
    "Total streaming validation windows evaluated",
)

STREAM_TOKENS_PROCESSED = Counter(
    "aegis_stream_tokens_processed_total",
    "Total tokens processed through streaming interceptor",
)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT_TRIGGERS = Counter(
    "aegis_rate_limit_triggers_total",
    "Rate limit rejections",
    ["tenant_id"],
)

# ---------------------------------------------------------------------------
# Immune memory (L4) metrics
# ---------------------------------------------------------------------------

VAULT_SIZE = Gauge(
    "aegis_vault_size",
    "Number of threat indicators in the vault",
    ["phase"],
)

ANTIBODY_GENERATIONS = Counter(
    "aegis_antibody_generations_total",
    "Antibody (vault entry) generation events",
)

# ---------------------------------------------------------------------------
# Session quarantine (L7)
# ---------------------------------------------------------------------------

QUARANTINED_SESSIONS = Gauge(
    "aegis_quarantined_sessions",
    "Number of currently quarantined sessions",
)

# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------

EVENT_BUS_EVENTS = Counter(
    "aegis_event_bus_events_total",
    "Events published to the event bus",
    ["channel"],
)

# ---------------------------------------------------------------------------
# Upstream model
# ---------------------------------------------------------------------------

UPSTREAM_ERRORS = Counter(
    "aegis_upstream_errors_total",
    "Upstream model request errors",
    ["endpoint", "error_type"],
)

# ---------------------------------------------------------------------------
# Multimodal metrics
# ---------------------------------------------------------------------------

MULTIMODAL_IMAGES_SCANNED = Counter(
    "aegis_multimodal_images_scanned_total",
    "Total images scanned by multimodal preprocessor",
)

MULTIMODAL_OCR_TEXT_EXTRACTED = Counter(
    "aegis_multimodal_ocr_text_extracted_total",
    "Total images with OCR text extracted",
)

MULTIMODAL_IMAGE_THREATS = Counter(
    "aegis_multimodal_image_threats_detected_total",
    "Image-specific threats detected",
    ["threat_type"],
)

MULTIMODAL_SCAN_LATENCY = Histogram(
    "aegis_multimodal_image_scan_latency_seconds",
    "Multimodal image scan latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ---------------------------------------------------------------------------
# Multimodal document metrics
# ---------------------------------------------------------------------------

MULTIMODAL_DOCUMENTS_SCANNED = Counter(
    "aegis_multimodal_documents_scanned_total",
    "Total documents scanned by multimodal document scanner",
)

MULTIMODAL_HIDDEN_CONTENT_DETECTED = Counter(
    "aegis_multimodal_hidden_content_detected_total",
    "Documents with hidden content (injection, metadata, comments) detected",
    ["surface"],
)

MULTIMODAL_DOCUMENT_THREATS = Counter(
    "aegis_multimodal_document_threats_detected_total",
    "Document-specific threats detected",
    ["threat_type"],
)

# ---------------------------------------------------------------------------
# Multimodal audio metrics
# ---------------------------------------------------------------------------

MULTIMODAL_AUDIO_SCANNED = Counter(
    "aegis_multimodal_audio_scanned_total",
    "Total audio files scanned by multimodal audio scanner",
)

MULTIMODAL_AUDIO_THREATS = Counter(
    "aegis_multimodal_audio_threats_detected_total",
    "Audio-specific threats detected",
    ["threat_type"],
)

MULTIMODAL_AUDIO_TRANSCRIPTION_LATENCY = Histogram(
    "aegis_multimodal_audio_transcription_latency_seconds",
    "Audio transcription latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# ---------------------------------------------------------------------------
# Cross-modal correlation metrics
# ---------------------------------------------------------------------------

CROSS_MODAL_LAUNDERING_DETECTED = Counter(
    "aegis_cross_modal_laundering_detected_total",
    "Cross-modal laundering detections (injection in media, clean text)",
)

CROSS_MODAL_INCONSISTENCY = Counter(
    "aegis_cross_modal_inconsistency_total",
    "Cross-modal semantic inconsistency detections",
)

# ---------------------------------------------------------------------------
# Tool use security metrics
# ---------------------------------------------------------------------------

TOOL_DEFINITION_THREATS = Counter(
    "aegis_tool_definition_threats_total",
    "Tool definition injection threats detected",
)

TOOL_CHAIN_ANOMALIES = Counter(
    "aegis_tool_chain_anomalies_total",
    "Suspicious tool chain sequences detected",
)

# ---------------------------------------------------------------------------
# Build info
# ---------------------------------------------------------------------------

BUILD_INFO = Info(
    "aegis_build",
    "AEGIS build information",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def track_latency(histogram: Histogram, **labels: str) -> Generator[None, None, None]:
    """Context manager to track latency in a histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        histogram.labels(**labels).observe(elapsed)


def get_metrics_text() -> bytes:
    """Generate Prometheus metrics in exposition format."""
    return generate_latest()
