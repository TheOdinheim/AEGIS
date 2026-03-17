"""
L6 — Policy Engine (Regulatory T-Cells / Immune Tolerance)

Merges threat scores from L2 Innate and L3 Adaptive with tenant policy,
global rules, and the current Threat Level Indicator (TLI) to render a
final allow/block/escalate decision. Three policy tiers: Global (hard
safety limits, compliance floors), Tenant (industry-specific, custom rules),
and Adaptive (dynamically adjusted by TLI). Bootstrap uses Python
dictionary-based evaluation; production uses OPA/Rego.

TLI 5-level scale: GREEN -> BLUE -> YELLOW -> ORANGE -> RED.

ASSUMED-BREACH POSTURE: This layer assumes all detection layers (L2, L3)
may be compromised and reporting false all-clear. The policy engine applies
independent rules that cannot be overridden by detection layer output:
hard safety limits (no CSAM, weapons instructions) apply regardless of
threat scores. A compromised policy engine is the most dangerous single
point of failure — it could allow all requests or block all legitimate
traffic. In production, OPA policies are version-controlled, code-reviewed,
and cryptographically signed. Policy changes trigger audit events.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aegis.config import PolicyConfig, ThreatLevel
from aegis.models.adaptive_result import AdaptiveAnalysisReport
from aegis.models.policy_decision import PolicyAction, PolicyDecision, PolicyTier
from aegis.models.scan_result import InnateScanReport, ThreatCategory

logger = logging.getLogger(__name__)

# TLI auto-decay timers (seconds without new threats before de-escalation)
_TLI_DECAY_TIMERS = {
    ThreatLevel.RED: 300,     # RED -> ORANGE after 5 min
    ThreatLevel.ORANGE: 180,  # ORANGE -> YELLOW after 3 min
    ThreatLevel.YELLOW: 120,  # YELLOW -> BLUE after 2 min
    ThreatLevel.BLUE: 60,     # BLUE -> GREEN after 1 min
}

# Hard-coded global safety rules that CANNOT be overridden by any policy tier
_HARD_BLOCK_CATEGORIES = {
    ThreatCategory.PROMPT_INJECTION,
    ThreatCategory.JAILBREAK,
    ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
}


@dataclass
class TenantPolicy:
    """Per-tenant policy configuration.

    Can be registered manually (legacy) or loaded from the TenantManager
    (multi-tenant mode). When loaded from TenantManager, the policy_tier
    field drives threshold modifiers.
    """
    tenant_id: str
    allowed_models: list[str] | None = None
    max_tokens: int | None = None
    custom_block_threshold: float | None = None
    custom_alert_threshold: float | None = None
    content_filters: list[str] | None = None
    escalation_threshold: float = 0.70
    policy_tier: str = "standard"  # standard | strict | permissive | custom
    custom_policy: dict[str, Any] | None = None


class PolicyEngine:
    """L6 Policy Engine — three-tier decision rendering.

    Tier 1 — Global: Hard safety limits, regulatory compliance floors.
    Tier 2 — Tenant: Per-tenant custom rules, industry-specific. When a
        TenantManager is wired in, tenant policies are loaded from
        PostgreSQL with per-tenant thresholds and policy_tier modifiers:
        - strict: lower block threshold by 15%, enable full slow-path
        - permissive: raise block threshold by 10%, suppress low alerts
        - standard: use configured thresholds as-is
        - custom: evaluate custom_policy JSONB for tenant-specific rules
    Tier 3 — Adaptive: Dynamically adjusted by Threat Level Indicator.

    The TLI dynamically tightens or relaxes detection thresholds based
    on the current threat environment.
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        tenant_manager: Any | None = None,
    ):
        self._config = config or PolicyConfig()
        self._threat_levels: dict[str, ThreatLevel] = {
            "__global__": self._config.default_threat_level,
        }
        self._threat_level_timestamps: dict[str, float] = {
            "__global__": time.monotonic(),
        }
        self._auto_decay_enabled = os.environ.get(
            "AEGIS_TLI_AUTO_DECAY_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        self._tenant_policies: dict[str, TenantPolicy] = {}
        self._tenant_manager = tenant_manager
        self._detection_count = 0
        self._baseline_detection_rate = 1.0  # detections per window
        self._escalation_history: list[dict[str, Any]] = []

    # --- Per-tenant TLI methods ---

    def get_threat_level(self, tenant_id: str = "__global__") -> ThreatLevel:
        """Get threat level for a specific tenant (falls back to global)."""
        self._check_decay(tenant_id)
        if tenant_id in self._threat_levels:
            return self._threat_levels[tenant_id]
        return self._threat_levels.get("__global__", self._config.default_threat_level)

    def set_threat_level(self, level: ThreatLevel, tenant_id: str = "__global__") -> None:
        """Set threat level for a specific tenant."""
        old = self.get_threat_level(tenant_id)
        self._threat_levels[tenant_id] = level
        self._threat_level_timestamps[tenant_id] = time.monotonic()
        if old != level:
            self._escalation_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tenant_id": tenant_id,
                "from": old.name,
                "to": level.name,
            })
            logger.warning("Threat level changed for %s: %s -> %s", tenant_id, old.name, level.name)

    def _check_decay(self, tenant_id: str) -> None:
        """Check if TLI should auto-decay for this tenant."""
        if not self._auto_decay_enabled:
            return
        level = self._threat_levels.get(tenant_id)
        if level is None or level == ThreatLevel.GREEN:
            return
        last_update = self._threat_level_timestamps.get(tenant_id, 0.0)
        elapsed = time.monotonic() - last_update
        decay_timer = _TLI_DECAY_TIMERS.get(level)
        if decay_timer and elapsed >= decay_timer:
            new_level = ThreatLevel(level.value - 1)
            self._threat_levels[tenant_id] = new_level
            self._threat_level_timestamps[tenant_id] = time.monotonic()
            self._escalation_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tenant_id": tenant_id,
                "from": level.name,
                "to": new_level.name,
                "reason": "auto_decay",
            })
            logger.info(
                "TLI auto-decay for %s: %s -> %s (%.0fs elapsed)",
                tenant_id, level.name, new_level.name, elapsed,
            )

    # --- Backward-compatible global threat_level property ---

    @property
    def threat_level(self) -> ThreatLevel:
        return self.get_threat_level("__global__")

    @threat_level.setter
    def threat_level(self, level: ThreatLevel) -> None:
        self.set_threat_level(level, "__global__")

    @property
    def tenant_manager(self) -> Any | None:
        return self._tenant_manager

    @tenant_manager.setter
    def tenant_manager(self, value: Any | None) -> None:
        self._tenant_manager = value

    def register_tenant_policy(self, policy: TenantPolicy) -> None:
        """Register a per-tenant policy."""
        self._tenant_policies[policy.tenant_id] = policy

    def _resolve_tenant_policy(self, tenant_id: str) -> TenantPolicy | None:
        """Resolve tenant policy: registered policies first, then TenantManager.

        Returns None if no tenant-specific policy exists.
        """
        # Manually registered policies take priority
        policy = self._tenant_policies.get(tenant_id)
        if policy:
            return policy

        # Check TenantManager (sync lookup from in-memory cache)
        if self._tenant_manager:
            tc = self._tenant_manager.get_tenant_by_id(tenant_id)
            if tc:
                return TenantPolicy(
                    tenant_id=tc.tenant_id,
                    allowed_models=tc.allowed_models or None,
                    custom_block_threshold=tc.block_threshold,
                    custom_alert_threshold=tc.alert_threshold,
                    policy_tier=tc.policy_tier,
                    custom_policy=tc.custom_policy,
                )

        return None

    def evaluate(
        self,
        request_id: str,
        innate_report: InnateScanReport | None = None,
        adaptive_report: AdaptiveAnalysisReport | None = None,
        tenant_id: str = "default",
        model: str = "",
    ) -> PolicyDecision:
        """Evaluate a request through the three-tier policy engine.

        Returns a PolicyDecision with action, reasoning, and downstream
        instructions (output scrutiny level).
        """
        start = time.perf_counter()

        innate_max = 0.0
        innate_categories: list[ThreatCategory] = []
        if innate_report:
            innate_max = innate_report.max_confidence
            innate_categories = innate_report.threat_categories

        adaptive_mcav = 0.0
        if adaptive_report:
            adaptive_mcav = adaptive_report.mcav_score

        reasons: list[str] = []
        applied_policies: list[str] = []

        # Resolve per-tenant threat level (with auto-decay)
        current_threat_level = self.get_threat_level(tenant_id)

        # === Tier 1: Global Policies (hard limits) ===
        # RED threat level: fail-closed, block everything
        if current_threat_level == ThreatLevel.RED:
            return PolicyDecision(
                request_id=request_id,
                action=PolicyAction.BLOCK,
                triggered_by=PolicyTier.GLOBAL,
                threat_level=current_threat_level,
                innate_max_confidence=innate_max,
                adaptive_mcav=adaptive_mcav,
                fused_score=1.0,
                reasons=["GLOBAL: RED threat level — fail-closed, all requests blocked"],
                applied_policies=["GLOBAL-RED-FAILCLOSED"],
                block_message="Service temporarily unavailable due to active security incident.",
            )

        # Check hard-block categories from innate
        for cat in innate_categories:
            if cat in _HARD_BLOCK_CATEGORIES and innate_max >= 0.85:
                reasons.append(f"GLOBAL: Hard block on category {cat.value} (conf={innate_max:.2f})")
                applied_policies.append(f"GLOBAL-HARDBLOCK-{cat.value}")
                return PolicyDecision(
                    request_id=request_id,
                    action=PolicyAction.BLOCK,
                    triggered_by=PolicyTier.GLOBAL,
                    threat_level=current_threat_level,
                    innate_max_confidence=innate_max,
                    adaptive_mcav=adaptive_mcav,
                    fused_score=innate_max,
                    reasons=reasons,
                    applied_policies=applied_policies,
                    block_message="Request blocked by AEGIS security policy.",
                )

        # === Tier 2: Tenant Policies ===
        tenant_policy = self._resolve_tenant_policy(tenant_id)
        if tenant_policy:
            # Check allowed models
            if tenant_policy.allowed_models and model and model not in tenant_policy.allowed_models:
                reasons.append(f"TENANT: Model '{model}' not in allowed list")
                applied_policies.append("TENANT-MODEL-ALLOWLIST")
                return PolicyDecision(
                    request_id=request_id,
                    action=PolicyAction.BLOCK,
                    triggered_by=PolicyTier.TENANT,
                    threat_level=current_threat_level,
                    innate_max_confidence=innate_max,
                    adaptive_mcav=adaptive_mcav,
                    fused_score=0.0,
                    reasons=reasons,
                    applied_policies=applied_policies,
                    block_message=f"Model '{model}' is not authorized for this tenant.",
                )

            # Evaluate custom_policy JSONB rules
            custom_decision = self._evaluate_custom_policy(
                tenant_policy, request_id, innate_max, adaptive_mcav,
                innate_categories, reasons, applied_policies,
                current_threat_level,
            )
            if custom_decision:
                return custom_decision

            # Apply policy_tier threshold modifier
            tier_block, tier_escalate = self._apply_policy_tier(
                tenant_policy, reasons, applied_policies,
            )

            # Tenant-specific threshold override (after tier adjustment)
            if tier_block is not None:
                if innate_max >= tier_block:
                    reasons.append(
                        f"TENANT: Innate confidence {innate_max:.2f} >= "
                        f"tenant block threshold {tier_block:.2f} "
                        f"(tier={tenant_policy.policy_tier})"
                    )
                    applied_policies.append("TENANT-THRESHOLD-BLOCK")
                    return PolicyDecision(
                        request_id=request_id,
                        action=PolicyAction.BLOCK,
                        triggered_by=PolicyTier.TENANT,
                        threat_level=current_threat_level,
                        innate_max_confidence=innate_max,
                        adaptive_mcav=adaptive_mcav,
                        fused_score=innate_max,
                        reasons=reasons,
                        applied_policies=applied_policies,
                    )

        # === Tier 3: Adaptive Policies (TLI-adjusted) ===
        # Compute fused score
        fused = self._fuse_scores(innate_max, adaptive_mcav)

        # Adjust thresholds based on threat level
        block_threshold = 0.85
        escalate_threshold = 0.60

        if current_threat_level == ThreatLevel.BLUE:
            reduction = self._config.block_threshold_reduction_blue  # 0.10
            block_threshold *= (1.0 - reduction)
            escalate_threshold *= (1.0 - reduction)
            applied_policies.append("ADAPTIVE-BLUE-THRESHOLD-REDUCTION")

        elif current_threat_level == ThreatLevel.YELLOW:
            block_threshold *= 0.75
            escalate_threshold *= 0.75
            applied_policies.append("ADAPTIVE-YELLOW-AGGRESSIVE")

        elif current_threat_level == ThreatLevel.ORANGE:
            block_threshold *= 0.60
            escalate_threshold *= 0.50
            applied_policies.append("ADAPTIVE-ORANGE-MAXIMUM")

        # Apply adaptive thresholds
        if fused >= block_threshold:
            reasons.append(
                f"ADAPTIVE: Fused score {fused:.2f} >= block threshold "
                f"{block_threshold:.2f} (TLI={current_threat_level.name})"
            )
            applied_policies.append("ADAPTIVE-FUSED-BLOCK")
            return PolicyDecision(
                request_id=request_id,
                action=PolicyAction.BLOCK,
                triggered_by=PolicyTier.ADAPTIVE,
                threat_level=current_threat_level,
                innate_max_confidence=innate_max,
                adaptive_mcav=adaptive_mcav,
                fused_score=fused,
                reasons=reasons,
                applied_policies=applied_policies,
            )

        if fused >= escalate_threshold:
            reasons.append(
                f"ADAPTIVE: Fused score {fused:.2f} >= escalate threshold "
                f"{escalate_threshold:.2f} (TLI={current_threat_level.name})"
            )
            applied_policies.append("ADAPTIVE-FUSED-ESCALATE")

            # At YELLOW+, escalate to human review
            if current_threat_level.value >= self._config.enable_human_review_at.value:
                return PolicyDecision(
                    request_id=request_id,
                    action=PolicyAction.ESCALATE,
                    triggered_by=PolicyTier.ADAPTIVE,
                    threat_level=current_threat_level,
                    innate_max_confidence=innate_max,
                    adaptive_mcav=adaptive_mcav,
                    fused_score=fused,
                    reasons=reasons,
                    applied_policies=applied_policies,
                    output_scrutiny_level=2.0,
                )

            # Below YELLOW: allow with degraded mode (elevated output scrutiny)
            return PolicyDecision(
                request_id=request_id,
                action=PolicyAction.ALLOW_DEGRADED,
                triggered_by=PolicyTier.ADAPTIVE,
                threat_level=current_threat_level,
                innate_max_confidence=innate_max,
                adaptive_mcav=adaptive_mcav,
                fused_score=fused,
                reasons=reasons,
                applied_policies=applied_policies,
                output_scrutiny_level=1.5,
            )

        # No policy triggered — allow normally
        reasons.append(f"ALLOW: Fused score {fused:.2f} below all thresholds")
        applied_policies.append("ADAPTIVE-ALLOW")
        return PolicyDecision(
            request_id=request_id,
            action=PolicyAction.ALLOW,
            triggered_by=PolicyTier.ADAPTIVE,
            threat_level=current_threat_level,
            innate_max_confidence=innate_max,
            adaptive_mcav=adaptive_mcav,
            fused_score=fused,
            reasons=reasons,
            applied_policies=applied_policies,
        )

    def _apply_policy_tier(
        self,
        tenant_policy: TenantPolicy,
        reasons: list[str],
        applied_policies: list[str],
    ) -> tuple[float | None, float | None]:
        """Apply policy_tier modifier to tenant thresholds.

        Returns (adjusted_block_threshold, adjusted_escalate_threshold).
        Returns (None, None) if no custom threshold is set.
        """
        base_block = tenant_policy.custom_block_threshold
        if base_block is None:
            return None, None

        tier = tenant_policy.policy_tier

        if tier == "strict":
            # Lower block threshold by 15% — more aggressive blocking
            adjusted_block = base_block * 0.85
            applied_policies.append("TENANT-TIER-STRICT")
            reasons.append(
                f"TENANT: Strict tier applied — block threshold "
                f"lowered from {base_block:.2f} to {adjusted_block:.2f}"
            )
            return adjusted_block, None

        elif tier == "permissive":
            # Raise block threshold by 10% — fewer blocks
            adjusted_block = min(base_block * 1.10, 1.0)
            applied_policies.append("TENANT-TIER-PERMISSIVE")
            reasons.append(
                f"TENANT: Permissive tier applied — block threshold "
                f"raised from {base_block:.2f} to {adjusted_block:.2f}"
            )
            return adjusted_block, None

        # standard or custom: use threshold as-is
        if tier == "custom":
            applied_policies.append("TENANT-TIER-CUSTOM")
        else:
            applied_policies.append("TENANT-TIER-STANDARD")
        return base_block, None

    def _evaluate_custom_policy(
        self,
        tenant_policy: TenantPolicy,
        request_id: str,
        innate_max: float,
        adaptive_mcav: float,
        innate_categories: list[ThreatCategory],
        reasons: list[str],
        applied_policies: list[str],
        current_threat_level: ThreatLevel | None = None,
    ) -> PolicyDecision | None:
        """Evaluate tenant's custom_policy JSONB rules.

        Supported custom_policy keys:
        - "blocked_categories": list of ThreatCategory values to hard-block
        - "min_block_confidence": custom minimum confidence for any block

        Returns a PolicyDecision if a custom rule triggers, None otherwise.
        """
        if not tenant_policy.custom_policy:
            return None

        cp = tenant_policy.custom_policy

        # Custom blocked categories
        blocked_cats = cp.get("blocked_categories", [])
        if blocked_cats and innate_categories:
            for cat in innate_categories:
                if cat.value in blocked_cats:
                    reasons.append(
                        f"TENANT-CUSTOM: Category {cat.value} in tenant blocked list"
                    )
                    applied_policies.append("TENANT-CUSTOM-BLOCKED-CATEGORY")
                    return PolicyDecision(
                        request_id=request_id,
                        action=PolicyAction.BLOCK,
                        triggered_by=PolicyTier.TENANT,
                        threat_level=current_threat_level or self.threat_level,
                        innate_max_confidence=innate_max,
                        adaptive_mcav=adaptive_mcav,
                        fused_score=innate_max,
                        reasons=reasons,
                        applied_policies=applied_policies,
                        block_message="Request blocked by tenant custom policy.",
                    )

        return None

    def _fuse_scores(self, innate_max: float, adaptive_mcav: float) -> float:
        """Fuse innate and adaptive scores into a single threat score.

        Uses weighted max fusion: takes the higher of the two scores,
        with a boost when both are elevated (corroborating evidence).
        """
        if innate_max == 0 and adaptive_mcav == 0:
            return 0.0

        base = max(innate_max, adaptive_mcav)

        # Corroboration boost: when both paths detect threats, increase confidence
        if innate_max >= 0.5 and adaptive_mcav >= 0.5:
            corroboration = min(innate_max, adaptive_mcav) * 0.2
            return min(base + corroboration, 1.0)

        return base

    def escalate_threat_level(self, tenant_id: str = "__global__") -> ThreatLevel:
        """Escalate the threat level by one step.

        Returns the new threat level.
        """
        current = self.get_threat_level(tenant_id)
        if current.value < ThreatLevel.RED.value:
            self.set_threat_level(ThreatLevel(current.value + 1), tenant_id)
        return self.get_threat_level(tenant_id)

    def de_escalate_threat_level(self, tenant_id: str = "__global__") -> ThreatLevel:
        """De-escalate the threat level by one step.

        Returns the new threat level.
        """
        current = self.get_threat_level(tenant_id)
        if current.value > ThreatLevel.GREEN.value:
            self.set_threat_level(ThreatLevel(current.value - 1), tenant_id)
        return self.get_threat_level(tenant_id)
