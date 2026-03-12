"""
AEGIS Middleware — FastAPI middleware for request enrichment and observability.

ASSUMED-BREACH POSTURE: Middleware runs before any defense layer. It assumes
the raw HTTP request is adversarial: headers may be forged, content-length
may lie, and authentication tokens may be stolen. Middleware builds the
RequestContext from independently verified sources only. Prometheus metrics
are collected on a separate internal port to prevent metric exfiltration.
"""
