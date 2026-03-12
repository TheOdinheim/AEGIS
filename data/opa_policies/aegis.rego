# AEGIS Policy Engine — Rego rules for OPA
#
# Three-tier policy evaluation mirroring the Python PolicyEngine:
#   Tier 1 (Global): Hard safety blocks, RED fail-closed
#   Tier 2 (Tenant): Per-tenant threshold evaluation with policy_tier modifiers
#   Tier 3 (Adaptive): TLI-based threshold adjustment
#
# Input document (from OPAPolicyEngine.build_opa_input):
#   input.request_id            — string
#   input.model                 — string
#   input.threat_level          — "GREEN"|"BLUE"|"YELLOW"|"ORANGE"|"RED"
#   input.innate.max_confidence — float 0-1
#   input.innate.threat_categories — list of MITRE ATLAS tactic IDs
#   input.innate.should_block   — bool
#   input.adaptive.mcav_score   — float 0-1
#   input.adaptive.should_block — bool
#   input.tenant.tenant_id      — string
#   input.tenant.block_threshold — float 0-1
#   input.tenant.alert_threshold — float 0-1
#   input.tenant.policy_tier    — "standard"|"strict"|"permissive"|"custom"
#   input.tenant.custom_policy  — object

package aegis.policy

import rego.v1

# --- Default decision: allow ---
default decision := {
	"action": "allow",
	"reason": "No policy triggered",
	"triggered_by": "adaptive",
	"applied_policies": [],
}

# --- Tier 1: Global — RED fail-closed ---
decision := result if {
	input.threat_level == "RED"
	result := {
		"action": "block",
		"reason": "GLOBAL: RED threat level — fail-closed, all requests blocked",
		"triggered_by": "global",
		"applied_policies": ["GLOBAL-RED-FAILCLOSED"],
	}
}

# --- Tier 1: Global — Hard block categories ---
# Prompt injection at high confidence
decision := result if {
	input.threat_level != "RED"
	input.innate.max_confidence >= 0.85
	some cat in input.innate.threat_categories
	cat in {"AML.T0051", "AML.T0054", "AML.T0048", "AML.T0051.001", "AEGIS.PROMPT_INJECTION", "AEGIS.JAILBREAK", "AEGIS.SYSTEM_PROMPT_EXTRACTION"}
	result := {
		"action": "block",
		"reason": sprintf("GLOBAL: Hard block on category %s (conf=%.2f)", [cat, input.innate.max_confidence]),
		"triggered_by": "global",
		"applied_policies": [sprintf("GLOBAL-HARDBLOCK-%s", [cat])],
	}
}

# --- Tier 2: Tenant — custom blocked categories ---
decision := result if {
	input.threat_level != "RED"
	count(input.tenant.custom_policy.blocked_categories) > 0
	some cat in input.innate.threat_categories
	some blocked in input.tenant.custom_policy.blocked_categories
	cat == blocked
	result := {
		"action": "block",
		"reason": sprintf("TENANT-CUSTOM: Category %s in tenant blocked list", [cat]),
		"triggered_by": "tenant",
		"applied_policies": ["TENANT-CUSTOM-BLOCKED-CATEGORY"],
	}
}

# --- Tier 2: Tenant — threshold with policy_tier modifier ---
# Strict tier: threshold * 0.85
decision := result if {
	input.threat_level != "RED"
	not _global_hard_block
	not _tenant_custom_block
	input.tenant.policy_tier == "strict"
	effective := input.tenant.block_threshold * 0.85
	input.innate.max_confidence >= effective
	result := {
		"action": "block",
		"reason": sprintf("TENANT: Innate confidence %.2f >= strict threshold %.2f", [input.innate.max_confidence, effective]),
		"triggered_by": "tenant",
		"applied_policies": ["TENANT-TIER-STRICT", "TENANT-THRESHOLD-BLOCK"],
	}
}

# Permissive tier: threshold * 1.10 (capped at 1.0)
decision := result if {
	input.threat_level != "RED"
	not _global_hard_block
	not _tenant_custom_block
	input.tenant.policy_tier == "permissive"
	raw := input.tenant.block_threshold * 1.10
	effective := min([raw, 1.0])
	input.innate.max_confidence >= effective
	result := {
		"action": "block",
		"reason": sprintf("TENANT: Innate confidence %.2f >= permissive threshold %.2f", [input.innate.max_confidence, effective]),
		"triggered_by": "tenant",
		"applied_policies": ["TENANT-TIER-PERMISSIVE", "TENANT-THRESHOLD-BLOCK"],
	}
}

# Standard/custom tier: threshold as-is
decision := result if {
	input.threat_level != "RED"
	not _global_hard_block
	not _tenant_custom_block
	input.tenant.policy_tier in {"standard", "custom"}
	input.innate.max_confidence >= input.tenant.block_threshold
	result := {
		"action": "block",
		"reason": sprintf("TENANT: Innate confidence %.2f >= threshold %.2f", [input.innate.max_confidence, input.tenant.block_threshold]),
		"triggered_by": "tenant",
		"applied_policies": ["TENANT-TIER-STANDARD", "TENANT-THRESHOLD-BLOCK"],
	}
}

# --- Tier 3: Adaptive — TLI-adjusted fused score ---
decision := result if {
	input.threat_level != "RED"
	not _global_hard_block
	not _tenant_custom_block
	not _tenant_threshold_block
	fused := _fused_score
	threshold := _adaptive_block_threshold
	fused >= threshold
	result := {
		"action": "block",
		"reason": sprintf("ADAPTIVE: Fused score %.2f >= block threshold %.2f (TLI=%s)", [fused, threshold, input.threat_level]),
		"triggered_by": "adaptive",
		"applied_policies": [sprintf("ADAPTIVE-%s-FUSED-BLOCK", [input.threat_level])],
	}
}

# --- Helper rules ---

# Fused score: max of innate and adaptive, with corroboration boost
_fused_score := score if {
	innate := input.innate.max_confidence
	adaptive := input.adaptive.mcav_score
	base := max([innate, adaptive])
	innate >= 0.5
	adaptive >= 0.5
	corr := min([innate, adaptive]) * 0.2
	raw := base + corr
	score := min([raw, 1.0])
} else := score if {
	score := max([input.innate.max_confidence, input.adaptive.mcav_score])
}

# Adaptive block threshold adjusted by TLI
_adaptive_block_threshold := 0.85 if {
	input.threat_level == "GREEN"
}

_adaptive_block_threshold := threshold if {
	input.threat_level == "BLUE"
	threshold := 0.85 * 0.90
}

_adaptive_block_threshold := threshold if {
	input.threat_level == "YELLOW"
	threshold := 0.85 * 0.75
}

_adaptive_block_threshold := threshold if {
	input.threat_level == "ORANGE"
	threshold := 0.85 * 0.60
}

# Guard rules for mutual exclusion
_global_hard_block if {
	input.innate.max_confidence >= 0.85
	some cat in input.innate.threat_categories
	cat in {"AML.T0051", "AML.T0054", "AML.T0048", "AML.T0051.001", "AEGIS.PROMPT_INJECTION", "AEGIS.JAILBREAK", "AEGIS.SYSTEM_PROMPT_EXTRACTION"}
}

_tenant_custom_block if {
	count(input.tenant.custom_policy.blocked_categories) > 0
	some cat in input.innate.threat_categories
	some blocked in input.tenant.custom_policy.blocked_categories
	cat == blocked
}

_tenant_threshold_block if {
	input.tenant.policy_tier == "strict"
	input.innate.max_confidence >= input.tenant.block_threshold * 0.85
}

_tenant_threshold_block if {
	input.tenant.policy_tier == "permissive"
	raw := input.tenant.block_threshold * 1.10
	effective := min([raw, 1.0])
	input.innate.max_confidence >= effective
}

_tenant_threshold_block if {
	input.tenant.policy_tier in {"standard", "custom"}
	input.innate.max_confidence >= input.tenant.block_threshold
}
