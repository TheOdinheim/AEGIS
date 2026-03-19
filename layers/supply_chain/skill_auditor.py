"""
Skill/Plugin Auditor (SPA) — Extension 3.2

Audits third-party skills, plugins, and extensions before they are loaded
into an agent framework. Combines static analysis (descriptor injection,
malicious code patterns, permission scope) with dynamic AST analysis.

Three static checks:
    1. Descriptor Injection — scan name/description/metadata via L2 regex + TDIV
    2. Malicious Code Patterns — regex scan for exfiltration, C2, evasion
    3. Permission Scope — flag excessive permissions ("Lethal Trifecta")

One dynamic check:
    4. AST Analysis — Python AST inspection for dangerous imports, exec/eval,
       subprocess, network calls, obfuscation (compile+exec)

ASSUMED-BREACH POSTURE: Skills/plugins are untrusted code from third-party
registries. A compromised plugin marketplace can inject backdoors,
exfiltration routines, or prompt injection payloads into any descriptor
or code body. SPA blocks loading before execution.
"""

from __future__ import annotations

import ast
import enum
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from aegis.layers.innate.regex_engine import normalize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Malicious code pattern regexes (static analysis)
# ---------------------------------------------------------------------------

_EXFIL_PATTERNS = [
    re.compile(r"\brequests\.(get|post|put|delete|patch)\s*\(", re.IGNORECASE),
    re.compile(r"\burllib\.(request|parse)", re.IGNORECASE),
    re.compile(r"\bhttpx\.(get|post|put|delete|patch|AsyncClient|Client)", re.IGNORECASE),
    re.compile(r"\baiohttp\.ClientSession", re.IGNORECASE),
    re.compile(r"\bsocket\.(?:connect|send|recv)", re.IGNORECASE),
]

_C2_PATTERNS = [
    re.compile(r"\bsubprocess\.(run|Popen|call|check_output|check_call)\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.(?:system|popen|exec[lv]?[pe]?)\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\beval\s*\(", re.IGNORECASE),
]

_EVASION_PATTERNS = [
    re.compile(r"\bbase64\.(b64decode|decodebytes)\s*\(", re.IGNORECASE),
    re.compile(r"\bcodecs\.decode\s*\(.*['\"]rot", re.IGNORECASE),
    re.compile(r"\bcompile\s*\(.*exec", re.IGNORECASE),
    re.compile(r"\bgetattr\s*\([^,]+,\s*['\"]__", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Dangerous AST constructs
# ---------------------------------------------------------------------------

_DANGEROUS_IMPORTS = frozenset({
    "subprocess", "os", "sys", "shutil", "socket", "ctypes",
    "importlib", "pickle", "shelve", "marshal",
})

_DANGEROUS_FUNCS = frozenset({
    "exec", "eval", "compile", "__import__",
})

# ---------------------------------------------------------------------------
# Permission categories for scope analysis
# ---------------------------------------------------------------------------

_FILE_PERMISSIONS = frozenset({
    "file_read", "file_write", "file_delete", "filesystem",
    "read_file", "write_file", "disk",
})

_NETWORK_PERMISSIONS = frozenset({
    "network", "http", "https", "internet", "api_call",
    "web_request", "fetch", "socket",
})

_EXECUTION_PERMISSIONS = frozenset({
    "execute", "exec", "run", "shell", "subprocess",
    "code_execution", "system", "command",
})


class AuditVerdict(str, enum.Enum):
    APPROVED = "approved"
    FLAGGED = "flagged"
    REJECTED = "rejected"


@dataclass
class AuditCheckResult:
    check_name: str
    passed: bool
    severity: str = "info"  # info, warning, critical
    findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillAuditReport:
    skill_name: str
    verdict: AuditVerdict
    checks: list[AuditCheckResult] = field(default_factory=list)
    risk_score: float = 0.0
    lethal_trifecta: bool = False
    audit_latency_ms: float = 0.0
    blocked: bool = False


class SkillPluginAuditor:
    """Audits third-party skills/plugins before loading.

    Reuses L2 regex engine for descriptor injection scanning and TDIV
    for structured tool description validation.
    """

    def __init__(
        self,
        *,
        regex_engine: Any | None = None,
        tdiv: Any | None = None,
        block_lethal_trifecta: bool = True,
    ) -> None:
        self._regex_engine = regex_engine
        self._tdiv = tdiv
        self._block_lethal_trifecta = block_lethal_trifecta

    async def audit(
        self,
        skill_name: str,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        code: str | None = None,
        permissions: list[str] | None = None,
    ) -> SkillAuditReport:
        """Run all audit checks on a skill/plugin.

        Args:
            skill_name: Name of the skill or plugin.
            description: Skill description text.
            metadata: Additional metadata dict (author, version, etc.).
            code: Python source code body (for static + AST analysis).
            permissions: Declared permission list (e.g., ["file_read", "network"]).
        """
        start = time.perf_counter()
        checks: list[AuditCheckResult] = []

        # Check 1: Descriptor Injection
        c1 = await self._check_descriptor_injection(skill_name, description, metadata)
        checks.append(c1)

        # Check 2: Malicious Code Patterns
        c2 = self._check_malicious_patterns(code)
        checks.append(c2)

        # Check 3: Permission Scope
        c3 = self._check_permission_scope(permissions or [])
        checks.append(c3)
        lethal_trifecta = bool(c3.findings and any(
            f.get("lethal_trifecta") for f in c3.findings
        ))

        # Check 4: AST Analysis
        c4 = self._check_ast(code)
        checks.append(c4)

        # Compute verdict
        # When block_lethal_trifecta is False, exclude permission_scope
        # critical findings from causing REJECTED verdict
        critical_checks = [
            c for c in checks
            if c.severity == "critical"
            and not (c.check_name == "permission_scope" and not self._block_lethal_trifecta)
        ]
        warning_checks = [c for c in checks if c.severity == "warning" and not c.passed]

        if critical_checks:
            verdict = AuditVerdict.REJECTED
        elif lethal_trifecta and self._block_lethal_trifecta:
            verdict = AuditVerdict.REJECTED
        elif warning_checks:
            verdict = AuditVerdict.FLAGGED
        elif any(not c.passed for c in checks):
            verdict = AuditVerdict.FLAGGED
        else:
            verdict = AuditVerdict.APPROVED

        # Risk score
        risk = 0.0
        if critical_checks:
            risk = max(risk, 0.9)
        if lethal_trifecta:
            risk = max(risk, 0.85)
        if warning_checks:
            risk = max(risk, 0.5)
        for c in checks:
            if not c.passed and c.severity == "info":
                risk = max(risk, 0.3)

        blocked = verdict == AuditVerdict.REJECTED or (
            self._block_lethal_trifecta and lethal_trifecta
        )

        elapsed = (time.perf_counter() - start) * 1000

        report = SkillAuditReport(
            skill_name=skill_name,
            verdict=verdict,
            checks=checks,
            risk_score=round(risk, 4),
            lethal_trifecta=lethal_trifecta,
            audit_latency_ms=round(elapsed, 2),
            blocked=blocked,
        )

        logger.info(
            "Skill audit for %s: %s (risk=%.3f, trifecta=%s, %.1fms)",
            skill_name, verdict.value, risk, lethal_trifecta, elapsed,
        )

        return report

    # ------------------------------------------------------------------
    # Check 1: Descriptor Injection
    # ------------------------------------------------------------------

    async def _check_descriptor_injection(
        self,
        skill_name: str,
        description: str,
        metadata: dict[str, Any] | None,
    ) -> AuditCheckResult:
        findings: list[dict[str, Any]] = []

        # Scan description text via L2 regex engine
        texts_to_scan = [("description", description)]
        if metadata:
            for key, val in metadata.items():
                if isinstance(val, str) and len(val) > 5:
                    texts_to_scan.append((f"metadata.{key}", val))

        for source, text in texts_to_scan:
            normalized = normalize_text(text)
            if self._regex_engine is not None:
                try:
                    scan_result = await self._regex_engine.scan(normalized)
                    if scan_result.is_threat:
                        findings.append({
                            "source": source,
                            "patterns": list(scan_result.matched_patterns),
                            "confidence": scan_result.confidence,
                        })
                except Exception:
                    logger.debug("SPA regex scan failed for %s", source, exc_info=True)

        # Also use TDIV if available
        if self._tdiv is not None and description:
            try:
                tdiv_scan = await self._tdiv.validate(
                    skill_name, {}, description=description,
                )
                if tdiv_scan.injection_detected:
                    findings.append({
                        "source": "tdiv",
                        "verdict": tdiv_scan.verdict.value,
                        "patterns": tdiv_scan.pattern_matches,
                    })
            except Exception as e:
                logger.debug("TDIV scan failed for %s: %s", skill_name, e)

        severity = "critical" if findings else "info"
        return AuditCheckResult(
            check_name="descriptor_injection",
            passed=len(findings) == 0,
            severity=severity,
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Check 2: Malicious Code Patterns (static regex)
    # ------------------------------------------------------------------

    def _check_malicious_patterns(self, code: str | None) -> AuditCheckResult:
        if not code:
            return AuditCheckResult(
                check_name="malicious_code_patterns",
                passed=True,
                severity="info",
                findings=[{"note": "no code provided"}],
            )

        findings: list[dict[str, Any]] = []

        for pattern in _EXFIL_PATTERNS:
            matches = pattern.findall(code)
            if matches:
                findings.append({
                    "category": "exfiltration",
                    "pattern": pattern.pattern[:60],
                    "match_count": len(matches),
                })

        for pattern in _C2_PATTERNS:
            matches = pattern.findall(code)
            if matches:
                findings.append({
                    "category": "command_and_control",
                    "pattern": pattern.pattern[:60],
                    "match_count": len(matches),
                })

        for pattern in _EVASION_PATTERNS:
            matches = pattern.findall(code)
            if matches:
                findings.append({
                    "category": "evasion",
                    "pattern": pattern.pattern[:60],
                    "match_count": len(matches),
                })

        has_critical = any(
            f["category"] in ("command_and_control", "evasion") for f in findings
        )
        severity = "critical" if has_critical else ("warning" if findings else "info")

        return AuditCheckResult(
            check_name="malicious_code_patterns",
            passed=len(findings) == 0,
            severity=severity,
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Check 3: Permission Scope
    # ------------------------------------------------------------------

    def _check_permission_scope(self, permissions: list[str]) -> AuditCheckResult:
        if not permissions:
            return AuditCheckResult(
                check_name="permission_scope",
                passed=True,
                severity="info",
                findings=[{"note": "no permissions declared"}],
            )

        perm_lower = {p.lower() for p in permissions}

        has_file = bool(perm_lower & _FILE_PERMISSIONS)
        has_network = bool(perm_lower & _NETWORK_PERMISSIONS)
        has_exec = bool(perm_lower & _EXECUTION_PERMISSIONS)

        lethal_trifecta = has_file and has_network and has_exec

        findings: list[dict[str, Any]] = []
        if lethal_trifecta:
            findings.append({
                "lethal_trifecta": True,
                "file_perms": sorted(perm_lower & _FILE_PERMISSIONS),
                "network_perms": sorted(perm_lower & _NETWORK_PERMISSIONS),
                "exec_perms": sorted(perm_lower & _EXECUTION_PERMISSIONS),
            })
            severity = "critical"
        elif has_exec:
            findings.append({
                "lethal_trifecta": False,
                "note": "execution permission without full trifecta",
                "exec_perms": sorted(perm_lower & _EXECUTION_PERMISSIONS),
            })
            severity = "warning"
        elif has_file and has_network:
            findings.append({
                "lethal_trifecta": False,
                "note": "file + network without execution",
                "file_perms": sorted(perm_lower & _FILE_PERMISSIONS),
                "network_perms": sorted(perm_lower & _NETWORK_PERMISSIONS),
            })
            severity = "warning"
        else:
            severity = "info"

        passed = not lethal_trifecta and severity != "critical"
        return AuditCheckResult(
            check_name="permission_scope",
            passed=passed,
            severity=severity,
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Check 4: AST Analysis (Python source)
    # ------------------------------------------------------------------

    def _check_ast(self, code: str | None) -> AuditCheckResult:
        if not code:
            return AuditCheckResult(
                check_name="ast_analysis",
                passed=True,
                severity="info",
                findings=[{"note": "no code provided"}],
            )

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return AuditCheckResult(
                check_name="ast_analysis",
                passed=False,
                severity="warning",
                findings=[{"error": f"syntax error: {e}"}],
            )

        findings: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _DANGEROUS_IMPORTS:
                        findings.append({
                            "type": "dangerous_import",
                            "module": alias.name,
                            "line": getattr(node, "lineno", None),
                        })
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _DANGEROUS_IMPORTS:
                    findings.append({
                        "type": "dangerous_import",
                        "module": node.module,
                        "line": getattr(node, "lineno", None),
                    })
            # Check dangerous function calls
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in _DANGEROUS_FUNCS:
                    findings.append({
                        "type": "dangerous_call",
                        "function": func_name,
                        "line": getattr(node, "lineno", None),
                    })

        has_dangerous_call = any(f["type"] == "dangerous_call" for f in findings)
        severity = "critical" if has_dangerous_call else ("warning" if findings else "info")

        return AuditCheckResult(
            check_name="ast_analysis",
            passed=len(findings) == 0,
            severity=severity,
            findings=findings,
        )
