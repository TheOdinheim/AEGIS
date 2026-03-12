"""
Self supply-chain attack module — tests AEGIS's own integrity verification:
model file tampering detection, pattern file integrity, and dependency auditing.

These tests exercise supply chain components directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from red_team.infrastructure import AttackResult


class SelfSupplyChainTester:
    """Test AEGIS's own supply chain integrity.

    Verifies that AEGIS can detect tampering with its own critical files:
    - Pattern definitions (data/patterns.json)
    - Blocklist (data/blocklist.txt)
    - Seed threats (data/seed_threats.json)
    - Configuration files
    """

    def __init__(self, project_root: str | None = None):
        if project_root:
            self._root = Path(project_root)
        else:
            # Try to find project root from this file's location
            self._root = Path(__file__).resolve().parent.parent.parent

    def model_tampering_detection(self) -> AttackResult:
        """Test if AEGIS detects tampering with its own model/data files.

        Creates tampered copies of critical files and checks if the supply
        chain verification module would catch the modifications.
        """
        findings: list[str] = []

        critical_files = [
            "data/patterns.json",
            "data/blocklist.txt",
            "data/seed_threats.json",
        ]

        # Compute baseline hashes
        baseline_hashes: dict[str, str] = {}
        for rel_path in critical_files:
            full_path = self._root / rel_path
            if full_path.exists():
                content = full_path.read_bytes()
                baseline_hashes[rel_path] = hashlib.sha256(content).hexdigest()

        if not baseline_hashes:
            return AttackResult(
                attack_name="model_tampering_detection",
                category="supply_chain_self",
                vulnerable=False,
                severity="info",
                description="No critical data files found to test",
                recommendation="Ensure data files are present",
            )

        # Test: Create tampered versions and verify hash mismatch detection
        tampered_detected = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            for rel_path, original_hash in baseline_hashes.items():
                full_path = self._root / rel_path
                content = full_path.read_bytes()

                # Tamper: inject a backdoor pattern
                if rel_path.endswith(".json"):
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            data.append({"id": "BACKDOOR", "pattern": ".*"})
                        elif isinstance(data, dict):
                            data["_backdoor"] = True
                        tampered = json.dumps(data).encode()
                    except json.JSONDecodeError:
                        tampered = content + b"\n# BACKDOOR INJECTED"
                else:
                    tampered = content + b"\n# BACKDOOR INJECTED"

                tampered_hash = hashlib.sha256(tampered).hexdigest()

                if tampered_hash != original_hash:
                    tampered_detected += 1

        # Check: Does AEGIS have any integrity verification for its own files?
        # Look for hash manifest or integrity check
        manifest_exists = (self._root / "data" / "file_hashes.json").exists()
        integrity_module_exists = (
            self._root / "layers" / "supply_chain" / "integrity.py"
        ).exists()

        if not manifest_exists:
            findings.append(
                "No file hash manifest (data/file_hashes.json) exists — "
                "AEGIS has no baseline to detect tampering of its own data files"
            )

        # Check if patterns.json has any checksum embedded
        patterns_path = self._root / "data" / "patterns.json"
        if patterns_path.exists():
            patterns_content = patterns_path.read_text()
            if "checksum" not in patterns_content.lower() and "hash" not in patterns_content.lower():
                findings.append(
                    "patterns.json has no embedded checksum — "
                    "a modified patterns file would be loaded without verification"
                )

        return AttackResult(
            attack_name="model_tampering_detection",
            category="supply_chain_self",
            vulnerable=len(findings) > 0,
            severity="medium" if findings else "info",
            description=(
                f"Tested integrity verification for {len(baseline_hashes)} "
                f"critical files. Supply chain module exists: {integrity_module_exists}."
            ),
            evidence=findings,
            recommendation=(
                "Create a SHA-256 manifest of all critical data files, "
                "verify on startup, alert on mismatch"
            ),
            details={
                "files_checked": list(baseline_hashes.keys()),
                "manifest_exists": manifest_exists,
                "integrity_module_exists": integrity_module_exists,
            },
        )

    def pattern_file_integrity(self) -> AttackResult:
        """Test pattern file integrity and injection resistance.

        Checks whether patterns.json can be corrupted or injected with
        malicious patterns that would cause false negatives.
        """
        findings: list[str] = []

        patterns_path = self._root / "data" / "patterns.json"
        if not patterns_path.exists():
            return AttackResult(
                attack_name="pattern_file_integrity",
                category="supply_chain_self",
                vulnerable=False,
                severity="info",
                description="patterns.json not found",
            )

        content = patterns_path.read_text()
        data = json.loads(content)
        # patterns.json may be {"patterns": [...]} or a flat list
        patterns = data.get("patterns", data) if isinstance(data, dict) else data

        # Check 1: Are there any overly broad patterns that would match everything?
        overly_broad = []
        for p in patterns:
            pattern_text = p.get("pattern", "")
            if pattern_text in (".*", ".+", "^.*$", "^.+$", "\\s*\\S*"):
                overly_broad.append(p.get("id", "unknown"))

        if overly_broad:
            findings.append(
                f"Overly broad patterns found that match all text: {overly_broad}"
            )

        # Check 2: Duplicate pattern IDs (could mask injected patterns)
        ids = [p.get("id", "") for p in patterns]
        duplicates = [pid for pid in set(ids) if ids.count(pid) > 1]
        if duplicates:
            findings.append(f"Duplicate pattern IDs: {duplicates}")

        # Check 3: Patterns without MITRE ATLAS mapping
        unmapped = [
            p.get("id", "unknown") for p in patterns
            if not p.get("mitre_tactic")
        ]
        if len(unmapped) > len(patterns) * 0.2:
            findings.append(
                f"{len(unmapped)}/{len(patterns)} patterns lack MITRE ATLAS mapping"
            )

        # Check 4: File permissions (should be read-only in production)
        if os.access(str(patterns_path), os.W_OK):
            findings.append(
                "patterns.json is writable — an attacker with file system "
                "access could modify detection patterns"
            )

        # Check 5: Regex ReDoS patterns (catastrophic backtracking)
        import re
        redos_suspects = []
        for p in patterns:
            pattern_text = p.get("pattern", "")
            # Check for nested quantifiers
            if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern_text):
                redos_suspects.append(p.get("id", "unknown"))

        if redos_suspects:
            findings.append(
                f"Potential ReDoS patterns (nested quantifiers): {redos_suspects[:5]}"
            )

        return AttackResult(
            attack_name="pattern_file_integrity",
            category="supply_chain_self",
            vulnerable=len(findings) > 0,
            severity="medium" if findings else "info",
            description=f"Analyzed {len(patterns)} patterns for integrity issues",
            evidence=findings,
            recommendation=(
                "Make patterns.json read-only in production, add checksum "
                "verification, audit for ReDoS patterns, ensure all patterns "
                "have MITRE ATLAS mapping"
            ),
            details={"total_patterns": len(patterns)},
        )

    def dependency_audit(self) -> AttackResult:
        """Audit AEGIS's own Python dependencies for known vulnerabilities.

        Checks requirements.txt for pinned versions and known risky packages.
        """
        findings: list[str] = []

        req_path = self._root / "requirements.txt"
        if not req_path.exists():
            return AttackResult(
                attack_name="dependency_audit_self",
                category="supply_chain_self",
                vulnerable=False,
                severity="info",
                description="requirements.txt not found",
            )

        content = req_path.read_text()
        lines = [l.strip() for l in content.splitlines()
                 if l.strip() and not l.strip().startswith("#")]

        # Check 1: Unpinned dependencies
        unpinned = []
        for line in lines:
            if "==" not in line and ">=" not in line and "<=" not in line:
                unpinned.append(line)

        if unpinned:
            findings.append(
                f"Unpinned dependencies (supply chain risk): {unpinned[:5]}"
            )

        # Check 2: Dependencies using pickle/exec (known risky)
        risky_packages = {
            "joblib": "Uses pickle for serialization",
            "dill": "Extended pickle — arbitrary code execution",
            "cloudpickle": "Remote code execution via pickle",
        }
        for pkg, reason in risky_packages.items():
            for line in lines:
                if pkg in line.lower():
                    findings.append(f"Risky dependency: {pkg} — {reason}")

        # Check 3: Version wildcard usage
        wildcard_deps = [l for l in lines if ".*" in l]
        if wildcard_deps:
            findings.append(
                f"Wildcard versions (non-deterministic builds): "
                f"{[d.split('==')[0].split('>=')[0].strip() for d in wildcard_deps[:5]]}"
            )

        # Check 4: No lock file
        lock_exists = (
            (self._root / "requirements.lock").exists()
            or (self._root / "poetry.lock").exists()
            or (self._root / "Pipfile.lock").exists()
        )
        if not lock_exists:
            findings.append(
                "No dependency lock file found — builds are not reproducible"
            )

        # Check 5: Check for requirements-docker.txt
        docker_req_path = self._root / "requirements-docker.txt"
        if docker_req_path.exists():
            docker_content = docker_req_path.read_text()
            docker_lines = [l.strip() for l in docker_content.splitlines()
                           if l.strip() and not l.strip().startswith("#")]
            unpinned_docker = [l for l in docker_lines
                              if "==" not in l and ">=" not in l and "<=" not in l]
            if unpinned_docker:
                findings.append(
                    f"Unpinned Docker deps: {unpinned_docker[:3]}"
                )

        return AttackResult(
            attack_name="dependency_audit_self",
            category="supply_chain_self",
            vulnerable=len(findings) > 0,
            severity="low" if findings else "info",
            description=f"Audited {len(lines)} dependencies",
            evidence=findings,
            recommendation=(
                "Pin all dependencies to exact versions, add a lock file, "
                "run safety/pip-audit in CI, avoid pickle-based packages"
            ),
            details={"total_dependencies": len(lines)},
        )

    def run_all(self) -> list[AttackResult]:
        """Run all self supply-chain tests."""
        return [
            self.model_tampering_detection(),
            self.pattern_file_integrity(),
            self.dependency_audit(),
        ]
