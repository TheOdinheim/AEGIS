"""
Multi-Language Injection Detector — Pattern-based injection detection in 10 languages.

Extends L2 innate detection to catch prompt injections in non-English languages.
Attackers assume AI security tools are English-only; this module closes that gap.

Languages covered: Spanish, French, German, Portuguese, Italian, Russian,
Chinese (Mandarin), Japanese, Korean, Arabic.

Uses regex patterns targeting common injection templates translated and
adapted for each language's grammar and vocabulary.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from aegis.models.scan_result import ScanResult, ThreatCategory

logger = logging.getLogger(__name__)


@dataclass
class MultiLangResult:
    """Result from multi-language injection detection."""
    is_threat: bool = False
    confidence: float = 0.0
    language: str = ""
    matched_patterns: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Patterns by language
# Each entry: (pattern_id, compiled_regex, description)
# ---------------------------------------------------------------------------

def _compile_patterns(patterns: list[tuple[str, str, str]]) -> list[tuple[str, re.Pattern, str]]:
    """Compile regex patterns with case-insensitive and unicode flags."""
    compiled = []
    for pid, regex, desc in patterns:
        try:
            compiled.append((pid, re.compile(regex, re.IGNORECASE | re.UNICODE), desc))
        except re.error as e:
            logger.warning("Failed to compile multi-lang pattern %s: %s", pid, e)
    return compiled


_SPANISH_PATTERNS = _compile_patterns([
    ("ML-ES-001", r"ignora\s+(todas?\s+)?(las?\s+)?instrucciones?\s+(anteriores?|previas?)", "Ignore previous instructions"),
    ("ML-ES-002", r"olvida\s+(todas?\s+)?(las?\s+)?reglas?", "Forget the rules"),
    ("ML-ES-003", r"muestra(me)?\s+(el|tu)\s+prompt\s+del?\s+sistema", "Show system prompt"),
    ("ML-ES-004", r"act[uú]a\s+como\s+(un\s+)?(ai|ia)\s+sin\s+restricciones", "Act as unrestricted AI"),
    ("ML-ES-005", r"desactiva\s+(tus?\s+)?(filtros?|restricciones?|seguridad)", "Disable filters/safety"),
    ("ML-ES-006", r"revela\s+(tus?\s+)?instrucciones?\s+(iniciales?|del?\s+sistema)", "Reveal initial instructions"),
])

_FRENCH_PATTERNS = _compile_patterns([
    ("ML-FR-001", r"ignore[sz]?\s+(toutes?\s+)?(les?\s+)?instructions?\s+(pr[eé]c[eé]dentes?|ant[eé]rieures?)", "Ignore previous instructions"),
    ("ML-FR-002", r"oublie[sz]?\s+(toutes?\s+)?(les?\s+)?r[eè]gles?", "Forget the rules"),
    ("ML-FR-003", r"montre[sz]?[\s-]*(moi\s+)?(le|ton|votre)\s+prompt\s+(syst[eè]me|du?\s+syst[eè]me)", "Show system prompt"),
    ("ML-FR-004", r"agis?\s+comme\s+(une?\s+)?(ia|ai)\s+sans\s+restrictions?", "Act as unrestricted AI"),
    ("ML-FR-005", r"d[eé]sactive[sz]?\s+(tes?|vos?|les?)\s+(filtres?|restrictions?|s[eé]curit[eé])", "Disable filters/safety"),
    ("ML-FR-006", r"r[eé]v[eè]le[sz]?\s+(tes?|vos?)?\s*instructions?\s+(initiales?|syst[eè]me)", "Reveal initial instructions"),
])

_GERMAN_PATTERNS = _compile_patterns([
    ("ML-DE-001", r"ignorier(e|en)?\s+(alle\s+)?(vorherigen?\s+)?anweisungen", "Ignore previous instructions"),
    ("ML-DE-002", r"vergiss?\s+(alle\s+)?regeln", "Forget the rules"),
    ("ML-DE-003", r"zeig(e|en)?\s+(mir\s+)?(den|deinen?|ihren?)\s+system[\s-]?prompt", "Show system prompt"),
    ("ML-DE-004", r"agier(e|en)?\s+als\s+(eine?\s+)?uneingeschr[aä]nkte\s+(ki|ai)", "Act as unrestricted AI"),
    ("ML-DE-005", r"deaktivier(e|en)?\s+(deine?|ihre?)?\s*(filter|einschr[aä]nkungen|sicherheit)", "Disable filters/safety"),
    ("ML-DE-006", r"offenbar(e|en)?\s+(deine?|ihre?)?\s*system[\s-]?anweisungen", "Reveal system instructions"),
])

_PORTUGUESE_PATTERNS = _compile_patterns([
    ("ML-PT-001", r"ignor(e|ar)\s+(todas?\s+)?(as?\s+)?instru[cç][oõ]es?\s+(anteriores?|pr[eé]vias?)", "Ignore previous instructions"),
    ("ML-PT-002", r"esque[cç]a\s+(todas?\s+)?(as?\s+)?regras?", "Forget the rules"),
    ("ML-PT-003", r"mostre[\s-]*(me\s+)?(o|seu|sua)\s+prompt\s+(do?\s+)?sistema", "Show system prompt"),
    ("ML-PT-004", r"aja\s+como\s+(um(a)?\s+)?(ia|ai)\s+sem\s+restri[cç][oõ]es", "Act as unrestricted AI"),
    ("ML-PT-005", r"desativ(e|ar)\s+(seus?|suas?)?\s*(filtros?|restri[cç][oõ]es|seguran[cç]a)", "Disable filters/safety"),
])

_ITALIAN_PATTERNS = _compile_patterns([
    ("ML-IT-001", r"ignora\s+(tutte?\s+)?(le?\s+)?istruzioni\s+(precedenti|anteriori)", "Ignore previous instructions"),
    ("ML-IT-002", r"dimentica\s+(tutte?\s+)?(le?\s+)?regole?", "Forget the rules"),
    ("ML-IT-003", r"mostra(mi)?\s+(il|tuo)\s+prompt\s+(di|del)\s+sistema", "Show system prompt"),
    ("ML-IT-004", r"agisci\s+come\s+(un[a']?\s+)?(ia|ai)\s+senza\s+restrizioni", "Act as unrestricted AI"),
    ("ML-IT-005", r"disattiva\s+(i\s+)?(tuoi?\s+)?(filtri|restrizioni|sicurezza)", "Disable filters/safety"),
])

_RUSSIAN_PATTERNS = _compile_patterns([
    ("ML-RU-001", r"игнорируй(те)?\s+(все\s+)?(предыдущие\s+)?инструкции", "Ignore previous instructions"),
    ("ML-RU-002", r"забудь(те)?\s+(все\s+)?правила", "Forget the rules"),
    ("ML-RU-003", r"покажи(те)?\s+(мне\s+)?(свой|системный)\s+(промпт|запрос)", "Show system prompt"),
    ("ML-RU-004", r"действуй(те)?\s+как\s+(неограниченный|свободный)\s+(ии|ИИ|ai)", "Act as unrestricted AI"),
    ("ML-RU-005", r"отключи(те)?\s+(свои\s+)?(фильтры|ограничения|безопасность)", "Disable filters/safety"),
    ("ML-RU-006", r"раскрой(те)?\s+(свои\s+)?(начальные\s+)?инструкции", "Reveal initial instructions"),
])

_CHINESE_PATTERNS = _compile_patterns([
    ("ML-ZH-001", r"忽略.{0,5}(之前|以前|先前).{0,5}(指令|指示|说明)", "Ignore previous instructions"),
    ("ML-ZH-002", r"忘记.{0,5}(所有|全部).{0,5}(规则|规定)", "Forget the rules"),
    ("ML-ZH-003", r"(显示|展示|告诉我).{0,5}(系统|初始).{0,5}(提示|指令|prompt)", "Show system prompt"),
    ("ML-ZH-004", r"(扮演|充当|作为).{0,5}(不受限|无限制).{0,5}(AI|人工智能)", "Act as unrestricted AI"),
    ("ML-ZH-005", r"(关闭|禁用|取消).{0,5}(过滤|限制|安全)", "Disable filters/safety"),
])

_JAPANESE_PATTERNS = _compile_patterns([
    ("ML-JA-001", r"(以前|前|これまで)の(指示|命令|指令)(を|は).{0,5}(無視|忘れ)", "Ignore previous instructions"),
    ("ML-JA-002", r"(全て|すべて)の(ルール|規則)(を|は).{0,5}(忘れ|無視)", "Forget the rules"),
    ("ML-JA-003", r"システム.{0,5}(プロンプト|指示|命令)(を|は).{0,5}(見せ|教え|表示)", "Show system prompt"),
    ("ML-JA-004", r"制限(なし|のない|を?(解除|外し)).{0,5}(AI|ＡＩ)", "Act as unrestricted AI"),
    ("ML-JA-005", r"(フィルター|制限|安全).{0,5}(を|は).{0,5}(無効|解除|オフ)", "Disable filters/safety"),
])

_KOREAN_PATTERNS = _compile_patterns([
    ("ML-KO-001", r"(이전|앞|기존).{0,5}(지시|명령|지침).{0,5}(무시|잊어)", "Ignore previous instructions"),
    ("ML-KO-002", r"(모든|전부).{0,5}(규칙|규정).{0,5}(잊어|무시)", "Forget the rules"),
    ("ML-KO-003", r"시스템.{0,5}(프롬프트|지시|명령).{0,5}(보여|알려|표시)", "Show system prompt"),
    ("ML-KO-004", r"제한.{0,5}(없는|없이|해제).{0,5}(AI|인공지능)", "Act as unrestricted AI"),
    ("ML-KO-005", r"(필터|제한|보안).{0,5}(비활성|해제|끄)", "Disable filters/safety"),
])

_ARABIC_PATTERNS = _compile_patterns([
    ("ML-AR-001", r"تجاهل.{0,10}(التعليمات|الأوامر)\s*(السابقة|القديمة)", "Ignore previous instructions"),
    ("ML-AR-002", r"انس[ى]?\s*(كل|جميع)\s*(القواعد|القوانين)", "Forget the rules"),
    ("ML-AR-003", r"(أظهر|اعرض|أرني).{0,10}(موجه|تعليمات)\s*(النظام)", "Show system prompt"),
    ("ML-AR-004", r"(تصرف|اعمل)\s*(ك|مثل).{0,10}(ذكاء\s*اصطناعي|AI)\s*(بدون|بلا)\s*(قيود)", "Act as unrestricted AI"),
    ("ML-AR-005", r"(عطل|أوقف|ألغ).{0,10}(الفلاتر|القيود|الأمان)", "Disable filters/safety"),
])


# All language pattern sets with metadata
_ALL_LANGUAGES: list[tuple[str, str, list[tuple[str, re.Pattern, str]]]] = [
    ("es", "Spanish", _SPANISH_PATTERNS),
    ("fr", "French", _FRENCH_PATTERNS),
    ("de", "German", _GERMAN_PATTERNS),
    ("pt", "Portuguese", _PORTUGUESE_PATTERNS),
    ("it", "Italian", _ITALIAN_PATTERNS),
    ("ru", "Russian", _RUSSIAN_PATTERNS),
    ("zh", "Chinese", _CHINESE_PATTERNS),
    ("ja", "Japanese", _JAPANESE_PATTERNS),
    ("ko", "Korean", _KOREAN_PATTERNS),
    ("ar", "Arabic", _ARABIC_PATTERNS),
]


class MultiLangDetector:
    """Multi-language injection detector for L2 innate scanning.

    Scans input text against injection patterns in 10 languages.
    Returns threat if any pattern matches with confidence 0.90
    (language-specific attacks are deliberate).
    """

    def __init__(self, confidence: float = 0.90):
        self._confidence = confidence
        self._total_patterns = sum(len(patterns) for _, _, patterns in _ALL_LANGUAGES)

    @property
    def total_patterns(self) -> int:
        return self._total_patterns

    @property
    def languages(self) -> list[str]:
        return [name for _, name, _ in _ALL_LANGUAGES]

    async def scan(self, text: str) -> ScanResult:
        """Scan text for injection patterns in all supported languages.

        Returns a ScanResult. If any pattern matches, is_threat=True.
        """
        start = time.perf_counter()
        matched: list[str] = []
        detected_language = ""

        for lang_code, lang_name, patterns in _ALL_LANGUAGES:
            for pid, regex, desc in patterns:
                if regex.search(text):
                    matched.append(f"{pid} ({lang_name}: {desc})")
                    if not detected_language:
                        detected_language = lang_name

        elapsed = (time.perf_counter() - start) * 1000

        if matched:
            return ScanResult(
                scanner_id="multilang_detector",
                is_threat=True,
                confidence=self._confidence,
                threat_category=ThreatCategory.PROMPT_INJECTION,
                matched_patterns=matched,
                latency_ms=elapsed,
            )

        return ScanResult(
            scanner_id="multilang_detector",
            is_threat=False,
            confidence=0.0,
            latency_ms=elapsed,
        )


__all__ = ["MultiLangDetector", "MultiLangResult"]
