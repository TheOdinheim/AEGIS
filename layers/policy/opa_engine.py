"""
OPA Policy Engine — Open Policy Agent backend for L6 Policy Engine.

Sends policy evaluation requests to an OPA server via HTTP. Builds an
input document from innate/adaptive reports, tenant config, and current
threat level. Parses the OPA response back to a PolicyDecision.

Graceful degradation: if OPA is unreachable, times out, or returns an
unparseable response, falls back to the Python PolicyEngine with a
warning log. The Python engine is always loaded as the fallback.

ASSUMED-BREACH POSTURE: The OPA server is a critical trust boundary. In
production, OPA policies must be version-controlled, code-reviewed, and
loaded from a trusted bundle server. The OPA container should run with
minimal privileges (read-only filesystem, no network egress except the
policy bundle server). A compromised OPA could allow all requests or
block all legitimate traffic — the fallback to the Python engine provides
a second opinion but is not a complete mitigation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aegis.config import PolicyConfig, ThreatLevel
from aegis.layers.policy import PolicyEngine, TenantPolicy
from aegis.models.adaptive_result import AdaptiveAnalysisReport
from aegis.models.policy_decision import PolicyAction, PolicyDecision, PolicyTier
from aegis.models.scan_result import InnateScanReport

logger = logging.getLogger(__name__)

# OPA response timeout — policy evaluation must be fast
_OPA_TIMEOUT_SECONDS = 0.050  # 50ms

# Map OPA action strings to PolicyAction enum
_ACTION_MAP = {
    "allow": PolicyAction.ALLOW,
    "block": PolicyAction.BLOCK,
    "escalate": PolicyAction.ESCALATE,
    "allow_degraded": PolicyAction.ALLOW_DEGRADED,
    "quarantine": PolicyAction.QUARANTINE,
}


def build_opa_input(
    request_id: str,
    innate_report: InnateScanReport | None,
    adaptive_report: AdaptiveAnalysisReport | None,
    tenant_id: str,
    model: str,
    threat_level: ThreatLevel,
    tenant_policy: TenantPolicy | None = None,
) -> dict[str, Any]:
    """Build the OPA input document from layer reports and config.

    This document is sent as POST body to OPA's /v1/data/aegis/policy/decision
    endpoint. The Rego policy reads values from input.* to render a decision.
    """
    innate_data: dict[str, Any] = {
        "max_confidence": 0.0,
        "threat_categories": [],
        "should_block": False,
    }
    if innate_report:
        innate_data = {
            "max_confidence": innate_report.max_confidence,
            "threat_categories": [c.value for c in innate_report.threat_categories],
            "should_block": innate_report.should_block,
        }

    adaptive_data: dict[str, Any] = {
        "mcav_score": 0.0,
        "is_novel_attack": False,
        "should_block": False,
        "analyzers": {},
    }
    if adaptive_report:
        analyzers = {}
        for ar in (adaptive_report.analyzer_results or []):
            analyzers[ar.analyzer_id] = {
                "confidence": ar.confidence,
                "is_threat": ar.is_threat,
            }
        adaptive_data = {
            "mcav_score": adaptive_report.mcav_score,
            "is_novel_attack": adaptive_report.is_novel_attack,
            "should_block": adaptive_report.should_block,
            "analyzers": analyzers,
        }

    tenant_data: dict[str, Any] = {
        "tenant_id": tenant_id,
        "block_threshold": 0.85,
        "alert_threshold": 0.50,
        "policy_tier": "standard",
        "custom_policy": {},
    }
    if tenant_policy:
        tenant_data = {
            "tenant_id": tenant_id,
            "block_threshold": tenant_policy.custom_block_threshold or 0.85,
            "alert_threshold": tenant_policy.custom_alert_threshold or 0.50,
            "policy_tier": tenant_policy.policy_tier,
            "custom_policy": tenant_policy.custom_policy or {},
        }

    return {
        "request_id": request_id,
        "model": model,
        "threat_level": threat_level.name,
        "innate": innate_data,
        "adaptive": adaptive_data,
        "tenant": tenant_data,
    }


def parse_opa_response(
    opa_result: dict[str, Any],
    request_id: str,
    innate_max: float,
    adaptive_mcav: float,
    threat_level: ThreatLevel,
) -> PolicyDecision:
    """Parse OPA response into a PolicyDecision.

    Expected OPA result format:
    {
        "action": "allow|block|escalate",
        "reason": "Human-readable reason",
        "triggered_by": "global|tenant|adaptive",
        "applied_policies": ["POLICY-ID-1", "POLICY-ID-2"]
    }
    """
    action_str = opa_result.get("action", "allow")
    action = _ACTION_MAP.get(action_str, PolicyAction.ALLOW)

    triggered_str = opa_result.get("triggered_by", "adaptive")
    try:
        triggered_by = PolicyTier(triggered_str)
    except ValueError:
        triggered_by = PolicyTier.ADAPTIVE

    reason = opa_result.get("reason", "OPA policy decision")
    applied = opa_result.get("applied_policies", [])
    # Tag all OPA decisions for audit trail
    applied = [f"opa:{p}" for p in applied]

    fused_score = max(innate_max, adaptive_mcav)

    return PolicyDecision(
        request_id=request_id,
        action=action,
        triggered_by=triggered_by,
        threat_level=threat_level,
        innate_max_confidence=innate_max,
        adaptive_mcav=adaptive_mcav,
        fused_score=fused_score,
        reasons=[reason] if reason else [],
        applied_policies=applied,
    )


class OPAPolicyEngine:
    """OPA-backed policy engine with Python fallback.

    Sends policy evaluation requests to an OPA server. If OPA is
    unreachable, times out, or returns an unparseable response,
    gracefully falls back to the Python PolicyEngine.

    The Python engine is always loaded as the fallback and is also
    used for local operations like threat level management and
    tenant policy registration.
    """

    def __init__(
        self,
        opa_url: str = "http://localhost:8181",
        fallback: PolicyEngine | None = None,
        config: PolicyConfig | None = None,
        tenant_manager: Any | None = None,
    ):
        self._opa_url = opa_url.rstrip("/")
        self._decision_url = f"{self._opa_url}/v1/data/aegis/policy/decision"
        self._health_url = f"{self._opa_url}/health"
        self._fallback = fallback or PolicyEngine(
            config=config, tenant_manager=tenant_manager,
        )
        self._client = None  # Lazy httpx.AsyncClient
        self._opa_available = False
        self._fallback_count = 0
        self._opa_count = 0

    @property
    def fallback(self) -> PolicyEngine:
        """The Python PolicyEngine used as fallback."""
        return self._fallback

    @property
    def threat_level(self) -> ThreatLevel:
        return self._fallback.threat_level

    @threat_level.setter
    def threat_level(self, level: ThreatLevel) -> None:
        self._fallback.threat_level = level

    @property
    def tenant_manager(self) -> Any | None:
        return self._fallback.tenant_manager

    @tenant_manager.setter
    def tenant_manager(self, value: Any | None) -> None:
        self._fallback.tenant_manager = value

    def register_tenant_policy(self, policy: TenantPolicy) -> None:
        """Delegate to fallback for tenant policy registration."""
        self._fallback.register_tenant_policy(policy)

    def escalate_threat_level(self) -> ThreatLevel:
        return self._fallback.escalate_threat_level()

    def de_escalate_threat_level(self) -> ThreatLevel:
        return self._fallback.de_escalate_threat_level()

    async def _get_client(self):
        """Lazy-init httpx.AsyncClient."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(timeout=_OPA_TIMEOUT_SECONDS)
            except ImportError:
                logger.warning("httpx not available for OPA engine")
                return None
        return self._client

    async def evaluate(
        self,
        request_id: str,
        innate_report: InnateScanReport | None = None,
        adaptive_report: AdaptiveAnalysisReport | None = None,
        tenant_id: str = "default",
        model: str = "",
    ) -> PolicyDecision:
        """Evaluate policy via OPA with Python fallback.

        Attempts to call OPA. On any failure (timeout, connection error,
        invalid response), falls back to the Python PolicyEngine.
        """
        # Build tenant policy for OPA input
        tenant_policy = self._fallback._resolve_tenant_policy(tenant_id)

        opa_input = build_opa_input(
            request_id=request_id,
            innate_report=innate_report,
            adaptive_report=adaptive_report,
            tenant_id=tenant_id,
            model=model,
            threat_level=self._fallback.get_threat_level(tenant_id),
            tenant_policy=tenant_policy,
        )

        innate_max = opa_input["innate"]["max_confidence"]
        adaptive_mcav = opa_input["adaptive"]["mcav_score"]

        client = await self._get_client()
        if client is None:
            return self._do_fallback(
                "httpx unavailable", request_id, innate_report,
                adaptive_report, tenant_id, model,
            )

        try:
            resp = await client.post(
                self._decision_url,
                json={"input": opa_input},
            )
            resp.raise_for_status()
            body = resp.json()

            # OPA wraps result under "result" key
            result = body.get("result")
            if not isinstance(result, dict) or "action" not in result:
                return self._do_fallback(
                    f"invalid OPA response: {body!r}", request_id,
                    innate_report, adaptive_report, tenant_id, model,
                )

            self._opa_available = True
            self._opa_count += 1

            return parse_opa_response(
                result, request_id, innate_max, adaptive_mcav,
                self._fallback.get_threat_level(tenant_id),
            )

        except Exception as exc:
            return self._do_fallback(
                str(exc), request_id, innate_report,
                adaptive_report, tenant_id, model,
            )

    def _do_fallback(
        self,
        reason: str,
        request_id: str,
        innate_report: InnateScanReport | None,
        adaptive_report: AdaptiveAnalysisReport | None,
        tenant_id: str,
        model: str,
    ) -> PolicyDecision:
        """Fall back to Python PolicyEngine with warning."""
        logger.warning("OPA fallback (reason: %s) — using Python policy engine", reason)
        self._opa_available = False
        self._fallback_count += 1
        decision = self._fallback.evaluate(
            request_id=request_id,
            innate_report=innate_report,
            adaptive_report=adaptive_report,
            tenant_id=tenant_id,
            model=model,
        )
        # Tag the decision as a fallback
        decision.applied_policies.append("opa:FALLBACK")
        return decision

    async def health(self) -> dict[str, Any]:
        """Check OPA server health."""
        client = await self._get_client()
        if client is None:
            return {
                "status": "unhealthy",
                "opa_url": self._opa_url,
                "error": "httpx unavailable",
            }
        try:
            resp = await client.get(self._health_url)
            return {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "opa_url": self._opa_url,
                "status_code": resp.status_code,
                "opa_decisions": self._opa_count,
                "fallback_decisions": self._fallback_count,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "opa_url": self._opa_url,
                "error": str(exc),
                "opa_decisions": self._opa_count,
                "fallback_decisions": self._fallback_count,
            }

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None
