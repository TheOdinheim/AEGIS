"""
STIX 2.1 Indicator Generator — Federated Threat Intelligence.

Generates STIX 2.1 Indicator and Relationship objects from AEGIS detections.
Uses plain Python dicts — no stix2 library dependency.

Each indicator encodes:
- DP-noised threat embedding (never raw prompt text)
- MITRE ATLAS tactic mapping
- Detection metadata (layer, confidence, category)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# MITRE ATLAS tactic IDs to human-readable names
_ATLAS_TACTICS: dict[str, str] = {
    "AML.T0051": "LLM Prompt Injection",
    "AML.T0054": "LLM Jailbreak",
    "AML.T0043": "Adversarial Input",
    "AML.T0040": "ML Model Inference API Access",
    "AML.T0042": "Model Evasion",
    "AML.T0048": "Data Poisoning",
    "AML.T0044": "Full ML Model Access",
    "AML.T0047": "ML Supply Chain Compromise",
    "AML.T0025": "Model Exfiltration",
}


def generate_indicator(
    *,
    noised_embedding: np.ndarray,
    mitre_tactic: str = "",
    detection_layer: str = "",
    confidence: float = 0.0,
    attack_category: str = "",
    description: str = "",
) -> dict[str, Any]:
    """Generate a STIX 2.1 Indicator object.

    Args:
        noised_embedding: DP-noised embedding vector (never raw).
        mitre_tactic: MITRE ATLAS tactic ID (e.g. "AML.T0051").
        detection_layer: Which AEGIS layer caught this (e.g. "L3").
        confidence: Detection confidence 0.0-1.0.
        attack_category: Category from the AEGIS taxonomy.
        description: Brief description of the attack technique.

    Returns:
        STIX 2.1 Indicator as a plain dict.
    """
    indicator_id = f"indicator--{uuid.uuid4()}"
    now = datetime.now(timezone.utc).isoformat()

    tactic_name = _ATLAS_TACTICS.get(mitre_tactic, mitre_tactic or "unknown")
    name = f"AEGIS Detection — {tactic_name}"
    if attack_category:
        name = f"AEGIS Detection — {attack_category}"

    pattern = json.dumps({
        "threat_embedding": noised_embedding.tolist(),
        "mitre_tactic": mitre_tactic,
        "detection_layer": detection_layer,
        "confidence": round(confidence, 4),
        "attack_category": attack_category,
    })

    labels = ["malicious-activity"]
    if mitre_tactic:
        labels.append(mitre_tactic)

    indicator: dict[str, Any] = {
        "type": "indicator",
        "spec_version": "2.1",
        "id": indicator_id,
        "created": now,
        "modified": now,
        "name": name,
        "description": description or f"Novel attack detected by AEGIS {detection_layer}",
        "pattern_type": "aegis",
        "pattern": pattern,
        "valid_from": now,
        "labels": labels,
        "confidence": min(100, max(0, int(confidence * 100))),
    }

    return indicator


def generate_relationship(
    *,
    indicator_id: str,
    mitre_tactic: str,
) -> dict[str, Any]:
    """Generate a STIX 2.1 Relationship linking indicator to MITRE ATLAS.

    Args:
        indicator_id: The STIX indicator ID.
        mitre_tactic: MITRE ATLAS tactic ID.

    Returns:
        STIX 2.1 Relationship as a plain dict.
    """
    now = datetime.now(timezone.utc).isoformat()
    attack_pattern_id = f"attack-pattern--{uuid.uuid5(uuid.NAMESPACE_URL, mitre_tactic)}"

    return {
        "type": "relationship",
        "spec_version": "2.1",
        "id": f"relationship--{uuid.uuid4()}",
        "created": now,
        "modified": now,
        "relationship_type": "indicates",
        "source_ref": indicator_id,
        "target_ref": attack_pattern_id,
    }


def generate_stix_bundle(
    *,
    noised_embedding: np.ndarray,
    mitre_tactic: str = "",
    detection_layer: str = "",
    confidence: float = 0.0,
    attack_category: str = "",
    description: str = "",
) -> dict[str, Any]:
    """Generate a complete STIX 2.1 bundle with indicator and relationship.

    Returns:
        STIX 2.1 Bundle dict with indicator and relationship objects.
    """
    indicator = generate_indicator(
        noised_embedding=noised_embedding,
        mitre_tactic=mitre_tactic,
        detection_layer=detection_layer,
        confidence=confidence,
        attack_category=attack_category,
        description=description,
    )

    objects: list[dict[str, Any]] = [indicator]

    if mitre_tactic:
        rel = generate_relationship(
            indicator_id=indicator["id"],
            mitre_tactic=mitre_tactic,
        )
        objects.append(rel)

    return {
        "type": "bundle",
        "id": f"bundle--{uuid.uuid4()}",
        "objects": objects,
    }
