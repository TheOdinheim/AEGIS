"""
Stage 3: Dependency Audit — CVE matching and SBOM generation.

Biological Analog: Screening incoming material for known pathogens (CVE database)
and maintaining a registry of all biological components (SBOM).

Scans model dependencies against a local CVE database (data/known_cves.json) and
generates a Software Bill of Materials (SBOM) in a simplified CycloneDX-inspired
format.

ASSUMED-BREACH POSTURE: Dependencies are untrusted. A model may declare benign
dependencies but actually load malicious transitive dependencies at runtime.
This stage checks declared dependencies only — runtime behavior is Stage 4's job.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from aegis.layers.supply_chain.models import StageResult

logger = logging.getLogger(__name__)


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '1.2.3' into a comparable tuple."""
    parts = re.findall(r"\d+", version_str)
    return tuple(int(p) for p in parts)


def _version_less_than(version: str, threshold: str) -> bool:
    """Check if version < threshold using tuple comparison."""
    try:
        return _parse_version(version) < _parse_version(threshold)
    except (ValueError, IndexError):
        return False


def load_cve_database(cve_file: Path) -> list[dict[str, Any]]:
    """Load CVE database from JSON file."""
    if not cve_file.exists():
        logger.warning("CVE database not found at %s", cve_file)
        return []
    try:
        with open(cve_file) as f:
            data = json.load(f)
        return data.get("vulnerabilities", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load CVE database: %s", e)
        return []


def audit_dependencies(
    dependencies: dict[str, str],
    cve_database: list[dict[str, Any]],
) -> StageResult:
    """Run Stage 3: Dependency Audit.

    Args:
        dependencies: Mapping of package name → version string.
        cve_database: List of CVE entries from known_cves.json.

    Returns:
        StageResult with pass/fail/warn and vulnerability details.
    """
    start = time.perf_counter()

    details: dict[str, Any] = {
        "packages_scanned": len(dependencies),
        "vulnerabilities_found": [],
        "severity_counts": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "sbom": _generate_sbom(dependencies),
    }

    if not dependencies:
        return StageResult(
            stage="dependency_audit",
            status="pass",
            details={**details, "note": "No dependencies declared"},
            duration_ms=_elapsed(start),
        )

    for cve in cve_database:
        pkg_name = cve.get("package", "")
        if pkg_name not in dependencies:
            continue

        installed_version = dependencies[pkg_name]
        affected_versions = cve.get("affected_versions", "")

        # Parse "<X.Y.Z" constraint
        match = re.match(r"<\s*([\d.]+)", affected_versions)
        if not match:
            continue

        threshold = match.group(1)
        if _version_less_than(installed_version, threshold):
            severity = cve.get("severity", "UNKNOWN")
            details["vulnerabilities_found"].append({
                "cve_id": cve.get("cve_id"),
                "package": pkg_name,
                "installed_version": installed_version,
                "affected_versions": affected_versions,
                "fixed_version": cve.get("fixed_version"),
                "severity": severity,
                "description": cve.get("description"),
            })
            if severity in details["severity_counts"]:
                details["severity_counts"][severity] += 1

    # Determine status based on severity
    if details["severity_counts"]["CRITICAL"] > 0:
        status = "fail"
    elif details["severity_counts"]["HIGH"] > 0:
        status = "fail"
    elif details["severity_counts"]["MEDIUM"] > 0:
        status = "warn"
    elif details["vulnerabilities_found"]:
        status = "warn"
    else:
        status = "pass"

    return StageResult(
        stage="dependency_audit",
        status=status,
        details=details,
        duration_ms=_elapsed(start),
    )


def _generate_sbom(dependencies: dict[str, str]) -> dict[str, Any]:
    """Generate a simplified CycloneDX-inspired SBOM."""
    components = []
    for name, version in sorted(dependencies.items()):
        components.append({
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name}@{version}",
        })

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": components,
        "total_components": len(components),
    }


def _elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
