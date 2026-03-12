# OPA unit tests for AEGIS policy rules.
#
# Run with: opa test data/opa_policies/ -v
#
# Tests mirror the Python PolicyEngine behavior to ensure Rego rules
# produce identical decisions.

package aegis.policy_test

import rego.v1

import data.aegis.policy

# --- Helper: base input document ---
_base_input := {
	"request_id": "test-123",
	"model": "gpt-4",
	"threat_level": "GREEN",
	"innate": {
		"max_confidence": 0.0,
		"threat_categories": [],
		"should_block": false,
	},
	"adaptive": {
		"mcav_score": 0.0,
		"is_novel_attack": false,
		"should_block": false,
		"analyzers": {},
	},
	"tenant": {
		"tenant_id": "test-tenant",
		"block_threshold": 0.85,
		"alert_threshold": 0.50,
		"policy_tier": "standard",
		"custom_policy": {},
	},
}

# --- Test: Default allow on benign request ---
test_default_allow if {
	result := policy.decision with input as _base_input
	result.action == "allow"
	result.reason == "No policy triggered"
}

# --- Test: RED fail-closed ---
test_red_fail_closed if {
	inp := object.union(_base_input, {"threat_level": "RED"})
	result := policy.decision with input as inp
	result.action == "block"
	result.triggered_by == "global"
	"GLOBAL-RED-FAILCLOSED" in result.applied_policies
}

# --- Test: Global hard block on prompt injection ---
test_global_hard_block_prompt_injection if {
	inp := object.union(_base_input, {
		"innate": {
			"max_confidence": 0.90,
			"threat_categories": ["AEGIS.PROMPT_INJECTION"],
			"should_block": true,
		},
	})
	result := policy.decision with input as inp
	result.action == "block"
	result.triggered_by == "global"
}

# --- Test: Global hard block on jailbreak ---
test_global_hard_block_jailbreak if {
	inp := object.union(_base_input, {
		"innate": {
			"max_confidence": 0.92,
			"threat_categories": ["AEGIS.JAILBREAK"],
			"should_block": true,
		},
	})
	result := policy.decision with input as inp
	result.action == "block"
}

# --- Test: High confidence but non-blocked category → allow ---
test_high_confidence_non_blocked_category if {
	inp := object.union(_base_input, {
		"innate": {
			"max_confidence": 0.90,
			"threat_categories": ["AEGIS.PII_EXFILTRATION"],
			"should_block": false,
		},
	})
	result := policy.decision with input as inp
	result.action == "allow"
}

# --- Test: Strict tier lowers threshold ---
test_strict_tier if {
	inp := object.union(_base_input, {
		"innate": {
			"max_confidence": 0.73,
			"threat_categories": [],
			"should_block": false,
		},
		"tenant": {
			"tenant_id": "strict-tenant",
			"block_threshold": 0.85,
			"alert_threshold": 0.50,
			"policy_tier": "strict",
			"custom_policy": {},
		},
	})
	# Effective threshold = 0.85 * 0.85 = 0.7225. Confidence 0.73 >= 0.7225.
	result := policy.decision with input as inp
	result.action == "block"
	result.triggered_by == "tenant"
}

# --- Test: Permissive tier raises threshold ---
test_permissive_tier if {
	inp := object.union(_base_input, {
		"innate": {
			"max_confidence": 0.90,
			"threat_categories": [],
			"should_block": false,
		},
		"tenant": {
			"tenant_id": "permissive-tenant",
			"block_threshold": 0.85,
			"alert_threshold": 0.50,
			"policy_tier": "permissive",
			"custom_policy": {},
		},
	})
	# Effective threshold = min(0.85 * 1.10, 1.0) = 0.935. 0.90 < 0.935.
	result := policy.decision with input as inp
	result.action == "allow"
}

# --- Test: Custom blocked categories ---
test_custom_blocked_categories if {
	inp := object.union(_base_input, {
		"innate": {
			"max_confidence": 0.50,
			"threat_categories": ["AML.T0051"],
			"should_block": false,
		},
		"tenant": {
			"tenant_id": "custom-tenant",
			"block_threshold": 0.85,
			"alert_threshold": 0.50,
			"policy_tier": "standard",
			"custom_policy": {"blocked_categories": ["AML.T0051"]},
		},
	})
	result := policy.decision with input as inp
	result.action == "block"
	result.triggered_by == "tenant"
}

# --- Test: TLI BLUE reduces adaptive threshold ---
test_blue_threshold_reduction if {
	inp := object.union(_base_input, {
		"threat_level": "BLUE",
		"innate": {
			"max_confidence": 0.77,
			"threat_categories": [],
			"should_block": false,
		},
		"adaptive": {
			"mcav_score": 0.0,
			"is_novel_attack": false,
			"should_block": false,
			"analyzers": {},
		},
	})
	# BLUE threshold = 0.85 * 0.90 = 0.765. Score 0.77 >= 0.765.
	result := policy.decision with input as inp
	result.action == "block"
	result.triggered_by == "adaptive"
}

# --- Test: TLI ORANGE aggressive threshold ---
test_orange_threshold if {
	inp := object.union(_base_input, {
		"threat_level": "ORANGE",
		"innate": {
			"max_confidence": 0.52,
			"threat_categories": [],
			"should_block": false,
		},
	})
	# ORANGE threshold = 0.85 * 0.60 = 0.51. Score 0.52 >= 0.51.
	result := policy.decision with input as inp
	result.action == "block"
}
