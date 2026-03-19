"""
Dependency Chain Analyzer (DCA) — Extension 3.3

Maps the full dependency graph of any component and validates every
dependency against known vulnerability and malicious package databases.

Four checks per dependency:
    1. Known Malicious Package Detection — exact match against curated list
    2. Typosquatting Detection — Levenshtein distance ≤ 2 to popular packages
    3. Slopsquatting Detection — LLM-hallucinated package name patterns
    4. Freshness/Risk Profile — metadata-based risk flags

Supports requirements.txt, package.json, and flat (name, version, ecosystem)
tuple input formats.

ASSUMED-BREACH POSTURE: Package registries are untrusted. Attackers
register typosquatted names, compromise maintainer accounts, and exploit
LLM hallucinations to inject malicious dependencies. Every dependency is
suspect until validated.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegis.layers.supply_chain.provenance_validator import _levenshtein_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slopsquatting heuristic patterns (LLM-hallucinated names)
# ---------------------------------------------------------------------------

_SLOPSQUAT_PATTERNS = [
    re.compile(r"^[a-z]+-[a-z]+-[a-z]+(-[a-z]+)?$"),        # verb-noun-adj combos
    re.compile(r"^(python|py|node|js)-[a-z]+-[a-z]+$"),      # lang-prefix combos
    re.compile(r"^(flask|django|express|react)-[a-z]+-[a-z]+(-[a-z]+)?$"),  # framework-utility
    re.compile(r"^[a-z]+-advanced-[a-z]+$"),                  # overly descriptive
    re.compile(r"^[a-z]+-super-[a-z]+$"),                     # overly descriptive
    re.compile(r"^easy-[a-z]+(-[a-z]+)?$"),                   # easy-{thing}
    re.compile(r"^simple-[a-z]+(-[a-z]+)?$"),                 # simple-{thing}
    re.compile(r"^fast-[a-z]+(-[a-z]+)?$"),                   # fast-{thing}
    re.compile(r"^auto-[a-z]+(-[a-z]+)?$"),                   # auto-{thing}
]

# Requirements.txt line parser
_REQ_LINE_RE = re.compile(
    r"^([A-Za-z0-9][-A-Za-z0-9_.]*)"    # package name
    r"(\[[^\]]*\])?"                      # optional extras
    r"(.*)$"                              # version specifier
)

_VERSION_SPEC_RE = re.compile(r"[><=!~]+\s*[\d][\d.]*")


@dataclass
class DependencyInfo:
    name: str
    version: str | None = None
    ecosystem: str = "pypi"  # "pypi", "npm", "other"
    extras: list[str] = field(default_factory=list)
    source_file: str | None = None


@dataclass
class DependencyFinding:
    package_name: str
    finding_type: str  # "known_malicious", "typosquatting", "slopsquatting", "high_risk_profile"
    severity: str  # "critical", "high", "medium", "low"
    details: str
    similar_to: str | None = None


@dataclass
class DependencyAnalysisReport:
    total_dependencies: int
    findings: list[DependencyFinding] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    packages_checked: list[str] = field(default_factory=list)
    overall_risk: str = "clean"  # "clean", "low", "medium", "high", "critical"
    analysis_latency_ms: float = 0.0


class DependencyChainAnalyzer:
    """Validates dependency chains against malicious/typosquat/slopsquat databases.

    Loads known-malicious and popular package lists from JSON seed files.
    """

    def __init__(
        self,
        *,
        malicious_file: Path | None = None,
        popular_file: Path | None = None,
    ) -> None:
        self._malicious: dict[str, set[str]] = {"pypi": set(), "npm": set()}
        self._popular: dict[str, set[str]] = {"pypi": set(), "npm": set()}
        self._popular_lists: dict[str, list[str]] = {"pypi": [], "npm": []}

        if malicious_file and malicious_file.exists():
            self._load_malicious(malicious_file)
        if popular_file and popular_file.exists():
            self._load_popular(popular_file)

    def _load_malicious(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            for eco in ("pypi", "npm"):
                self._malicious[eco] = {p.lower() for p in data.get(eco, [])}
            logger.info(
                "Loaded malicious packages: %d PyPI, %d npm",
                len(self._malicious["pypi"]), len(self._malicious["npm"]),
            )
        except Exception as e:
            logger.warning("Failed to load malicious packages from %s: %s", path, e)

    def _load_popular(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            for eco in ("pypi", "npm"):
                pkgs = data.get(eco, [])
                self._popular[eco] = {p.lower() for p in pkgs}
                self._popular_lists[eco] = [p.lower() for p in pkgs]
            logger.info(
                "Loaded popular packages: %d PyPI, %d npm",
                len(self._popular["pypi"]), len(self._popular["npm"]),
            )
        except Exception as e:
            logger.warning("Failed to load popular packages from %s: %s", path, e)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_requirements_txt(self, content: str, source_file: str | None = None) -> list[DependencyInfo]:
        deps: list[DependencyInfo] = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Remove inline comments
            if " #" in line:
                line = line[:line.index(" #")].strip()

            m = _REQ_LINE_RE.match(line)
            if not m:
                continue

            name = m.group(1)
            extras_str = m.group(2)
            version_str = m.group(3).strip() if m.group(3) else None

            extras: list[str] = []
            if extras_str:
                extras = [e.strip() for e in extras_str.strip("[]").split(",") if e.strip()]

            # Extract version number from specifier
            version = None
            if version_str:
                vm = _VERSION_SPEC_RE.search(version_str)
                if vm:
                    version = re.sub(r"^[><=!~]+\s*", "", vm.group())

            deps.append(DependencyInfo(
                name=name, version=version, ecosystem="pypi",
                extras=extras, source_file=source_file,
            ))
        return deps

    def parse_package_json(self, content: str, source_file: str | None = None) -> list[DependencyInfo]:
        deps: list[DependencyInfo] = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return deps

        for section in ("dependencies", "devDependencies"):
            for name, version_spec in data.get(section, {}).items():
                version = re.sub(r"^[\^~>=<]+", "", str(version_spec)).strip()
                deps.append(DependencyInfo(
                    name=name, version=version or None, ecosystem="npm",
                    source_file=source_file,
                ))
        return deps

    def parse_dependency_list(
        self, dep_list: list[tuple[str, str | None, str]],
    ) -> list[DependencyInfo]:
        return [
            DependencyInfo(name=name, version=version, ecosystem=eco)
            for name, version, eco in dep_list
        ]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        dependencies: list[DependencyInfo],
        *,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> DependencyAnalysisReport:
        """Run all 4 checks on dependencies.

        Args:
            dependencies: Parsed dependency list.
            metadata: Optional per-package metadata keyed by lowercase name.
                      Keys: publish_date_days_ago (int), download_count (int),
                      has_repository (bool), maintainer_package_count (int).
        """
        start = time.perf_counter()
        metadata = metadata or {}
        findings: list[DependencyFinding] = []
        packages_checked: list[str] = []

        for dep in dependencies:
            name_lower = dep.name.lower()
            packages_checked.append(dep.name)
            eco = dep.ecosystem if dep.ecosystem in ("pypi", "npm") else "pypi"

            # Check 1: Known malicious
            if name_lower in self._malicious.get(eco, set()):
                findings.append(DependencyFinding(
                    package_name=dep.name,
                    finding_type="known_malicious",
                    severity="critical",
                    details=f"Package '{dep.name}' is in the known-malicious list for {eco}",
                ))
                continue  # no need for further checks on known malware

            # Check 2: Typosquatting
            typosquat = self._check_typosquatting(name_lower, eco)
            if typosquat:
                findings.append(typosquat)

            # Check 3: Slopsquatting
            slopsquat = self._check_slopsquatting(name_lower, eco)
            if slopsquat:
                findings.append(slopsquat)

            # Check 4: Risk profile
            pkg_meta = metadata.get(name_lower, {})
            risk_finding = self._check_risk_profile(dep.name, pkg_meta)
            if risk_finding:
                findings.append(risk_finding)

        critical = sum(1 for f in findings if f.severity == "critical")
        high = sum(1 for f in findings if f.severity == "high")
        medium = sum(1 for f in findings if f.severity == "medium")

        if critical > 0:
            overall = "critical"
        elif high > 0:
            overall = "high"
        elif medium > 0:
            overall = "medium"
        elif findings:
            overall = "low"
        else:
            overall = "clean"

        elapsed = (time.perf_counter() - start) * 1000

        return DependencyAnalysisReport(
            total_dependencies=len(dependencies),
            findings=findings,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            packages_checked=packages_checked,
            overall_risk=overall,
            analysis_latency_ms=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Check 2: Typosquatting
    # ------------------------------------------------------------------

    def _check_typosquatting(self, name: str, ecosystem: str) -> DependencyFinding | None:
        popular = self._popular_lists.get(ecosystem, [])
        if not popular:
            return None

        # Exact match with popular package = legitimate
        if name in self._popular.get(ecosystem, set()):
            return None

        for pop_name in popular:
            dist = _levenshtein_distance(name, pop_name)
            if 1 <= dist <= 2:
                return DependencyFinding(
                    package_name=name,
                    finding_type="typosquatting",
                    severity="high",
                    details=f"Package '{name}' is {dist} edit(s) from popular package '{pop_name}'",
                    similar_to=pop_name,
                )
        return None

    # ------------------------------------------------------------------
    # Check 3: Slopsquatting
    # ------------------------------------------------------------------

    def _check_slopsquatting(self, name: str, ecosystem: str) -> DependencyFinding | None:
        # Skip if it's a known popular package
        if name in self._popular.get(ecosystem, set()):
            return None

        for pattern in _SLOPSQUAT_PATTERNS:
            if pattern.match(name):
                return DependencyFinding(
                    package_name=name,
                    finding_type="slopsquatting",
                    severity="medium",
                    details=f"Package '{name}' matches LLM-hallucinated naming pattern",
                )
        return None

    # ------------------------------------------------------------------
    # Check 4: Risk profile
    # ------------------------------------------------------------------

    def _check_risk_profile(
        self, name: str, meta: dict[str, Any],
    ) -> DependencyFinding | None:
        if not meta:
            return None

        risk_factors: list[str] = []

        days_ago = meta.get("publish_date_days_ago")
        if days_ago is not None and days_ago < 30:
            risk_factors.append(f"published {days_ago} days ago")

        downloads = meta.get("download_count")
        if downloads is not None and downloads < 100:
            risk_factors.append(f"only {downloads} downloads")

        if meta.get("has_repository") is False:
            risk_factors.append("no repository URL")

        maintainer_count = meta.get("maintainer_package_count")
        if maintainer_count is not None and maintainer_count <= 1:
            risk_factors.append("single-package maintainer")

        if not risk_factors:
            return None

        severity = "high" if len(risk_factors) >= 3 else "medium" if len(risk_factors) >= 2 else "low"

        return DependencyFinding(
            package_name=name,
            finding_type="high_risk_profile",
            severity=severity,
            details=f"Risk factors: {', '.join(risk_factors)}",
        )
