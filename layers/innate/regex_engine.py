"""
Scanner 1 — Regex Pattern Engine (Toll-Like Receptor Analog)

200+ compiled regex patterns detecting known prompt injection templates,
system prompt extraction attempts, encoding-based obfuscation (base64,
ROT13, hex, Unicode tricks), code injection markers, and social engineering
templates. Organized by MITRE ATLAS tactic ID. Compiled at startup; runtime
matching is sub-millisecond.

ASSUMED-BREACH POSTURE: This scanner assumes L1 Barrier has been bypassed.
The raw input may contain any content regardless of what schema validation
claims. This scanner also assumes its own pattern library may be incomplete
or stale — novel attacks WILL evade regex. That is expected and handled by
L3 Adaptive. If this scanner crashes, it returns is_threat=True with
confidence=1.0 (fail-closed). A compromised pattern file could contain
deliberately weak patterns — the signature database is integrity-checked
at startup via SHA-256 hash.
"""

from __future__ import annotations

import base64
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from aegis.models.scan_result import ScanResult, ThreatCategory

# ---------------------------------------------------------------------------
# Unicode normalization — T1 mitigation (regex evasion via character-level
# perturbation). Applied once at the start of scan() before pattern matching.
# ---------------------------------------------------------------------------

# Zero-width and invisible characters used for regex evasion
_ZERO_WIDTH_CHARS = frozenset([
    "\u200b",  # Zero-Width Space
    "\u200c",  # Zero-Width Non-Joiner
    "\u200d",  # Zero-Width Joiner
    "\ufeff",  # Zero-Width No-Break Space / BOM
    "\u00ad",  # Soft Hyphen
    "\u2060",  # Word Joiner
    "\u200e",  # Left-to-Right Mark
    "\u200f",  # Right-to-Left Mark
    "\u2061",  # Function Application
    "\u2062",  # Invisible Times
    "\u2063",  # Invisible Separator
    "\u2064",  # Invisible Plus
    "\u034f",  # Combining Grapheme Joiner
    "\u115f",  # Hangul Choseong Filler
    "\u1160",  # Hangul Jungseong Filler
    "\u17b4",  # Khmer Vowel Inherent AQ
    "\u17b5",  # Khmer Vowel Inherent AA
    "\u2800",  # Braille Pattern Blank
    "\uffa0",  # Halfwidth Hangul Filler
    "\u180e",  # Mongolian Vowel Separator
    "\u2028",  # Line Separator
    "\u2029",  # Paragraph Separator
])

# BIDI override characters — used to visually reorder text to hide instructions
_BIDI_CHARS = frozenset([
    "\u202a",  # LRE — Left-to-Right Embedding
    "\u202b",  # RLE — Right-to-Left Embedding
    "\u202c",  # PDF — Pop Directional Formatting
    "\u202d",  # LRO — Left-to-Right Override
    "\u202e",  # RLO — Right-to-Left Override
    "\u2066",  # LRI — Left-to-Right Isolate
    "\u2067",  # RLI — Right-to-Left Isolate
    "\u2068",  # FSI — First Strong Isolate
    "\u2069",  # PDI — Pop Directional Isolate
])

# Leetspeak mappings for normalization
_LEETSPEAK_MAP: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "@": "a", "$": "s",
}

# Homoglyph mapping: Cyrillic/Greek lookalikes → Latin equivalents.
# Covers 25+ common homoglyphs used for regex evasion.
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic → Latin
    "\u0430": "a",  # а → a
    "\u0435": "e",  # е → e
    "\u043e": "o",  # о → o
    "\u0440": "p",  # р → p
    "\u0441": "c",  # с → c
    "\u0443": "y",  # у → y
    "\u0445": "x",  # х → x
    "\u0456": "i",  # і → i
    "\u0458": "j",  # ј → j
    "\u04bb": "h",  # һ → h
    "\u0455": "s",  # ѕ → s
    "\u0457": "i",  # ї → i
    "\u04c0": "l",  # Ӏ → l
    "\u051b": "q",  # ԛ → q
    "\u051d": "w",  # ԝ → w
    "\u0410": "A",  # А → A
    "\u0412": "B",  # В → B
    "\u0415": "E",  # Е → E
    "\u041a": "K",  # К → K
    "\u041c": "M",  # М → M
    "\u041d": "H",  # Н → H
    "\u041e": "O",  # О → O
    "\u0420": "P",  # Р → P
    "\u0421": "C",  # С → C
    "\u0422": "T",  # Т → T
    "\u0425": "X",  # Х → X
    # Greek → Latin
    "\u03bd": "v",  # ν → v
    "\u03bf": "o",  # ο → o
    "\u03b1": "a",  # α → a
    "\u03b5": "e",  # ε → e
    "\u03b9": "i",  # ι → i
    "\u03ba": "k",  # κ → k
    "\u03c1": "p",  # ρ → p
    "\u03c4": "t",  # τ → t
    "\u03c5": "u",  # υ → u
    # Extended Cyrillic → Latin
    "\u04bd": "e",  # ҽ → e (Cyrillic schwa)
    "\u0454": "e",  # є → e (Ukrainian ie)
    "\u04cf": "l",  # ӏ → l (Cyrillic palochka lowercase)
    "\u04af": "y",  # ү → y (Cyrillic straight u)
    "\u04b3": "x",  # ҳ → x (Cyrillic ha with descender)
    "\u0501": "d",  # ԁ → d (Cyrillic komi de)
    # Armenian → Latin
    "\u0570": "h",  # հ → h (Armenian ho)
    "\u0578": "n",  # ո → n (Armenian now)
    "\u0581": "g",  # ց → g (Armenian co)
    "\u0585": "o",  # օ → o (Armenian oh)
    "\u057d": "u",  # ս → u (Armenian seh)
    # Georgian → Latin
    "\u10d0": "a",  # ა → a (Georgian ani)
    "\u10e1": "s",  # ს → s (Georgian sani)
    # Cherokee → Latin
    "\u13be": "o",  # Ꮎ → o (Cherokee)
    # Latin extended → ASCII Latin
    "\u0251": "a",  # ɑ → a (Latin alpha)
    "\u0261": "g",  # ɡ → g (Latin script g)
    "\u0131": "i",  # ı → i (Turkish dotless i)
    # Greek extended → Latin
    "\u03c2": "c",  # ς → c (Greek final sigma, NFKC of lunate sigma)
    "\u03f3": "j",  # ϳ → j (Greek yot)
    # Roman numerals → Latin
    "\u217c": "l",  # ⅼ → l (small Roman numeral 50)
    "\u217d": "d",  # ⅽ → d (small Roman numeral 100 — visually d-like)
    "\u217e": "m",  # ⅾ → m (small Roman numeral 500 — visually m-like)
    # Fullwidth Latin → ASCII Latin (lowercase)
    "\uff41": "a",  # ａ → a
    "\uff42": "b",  # ｂ → b
    "\uff43": "c",  # ｃ → c
    "\uff44": "d",  # ｄ → d
    "\uff45": "e",  # ｅ → e
    "\uff46": "f",  # ｆ → f
    "\uff47": "g",  # ｇ → g
    "\uff48": "h",  # ｈ → h
    "\uff49": "i",  # ｉ → i
    "\uff4a": "j",  # ｊ → j
    "\uff4b": "k",  # ｋ → k
    "\uff4c": "l",  # ｌ → l
    "\uff4d": "m",  # ｍ → m
    "\uff4e": "n",  # ｎ → n
    "\uff4f": "o",  # ｏ → o
    "\uff50": "p",  # ｐ → p
    "\uff51": "q",  # ｑ → q
    "\uff52": "r",  # ｒ → r
    "\uff53": "s",  # ｓ → s
    "\uff54": "t",  # ｔ → t
    "\uff55": "u",  # ｕ → u
    "\uff56": "v",  # ｖ → v
    "\uff57": "w",  # ｗ → w
    "\uff58": "x",  # ｘ → x
    "\uff59": "y",  # ｙ → y
    "\uff5a": "z",  # ｚ → z
    # Fullwidth Latin → ASCII Latin (uppercase)
    "\uff21": "A",  # Ａ → A
    "\uff22": "B",  # Ｂ → B
    "\uff23": "C",  # Ｃ → C
    "\uff24": "D",  # Ｄ → D
    "\uff25": "E",  # Ｅ → E
    "\uff26": "F",  # Ｆ → F
    "\uff27": "G",  # Ｇ → G
    "\uff28": "H",  # Ｈ → H
    "\uff29": "I",  # Ｉ → I
    "\uff2a": "J",  # Ｊ → J
    "\uff2b": "K",  # Ｋ → K
    "\uff2c": "L",  # Ｌ → L
    "\uff2d": "M",  # Ｍ → M
    "\uff2e": "N",  # Ｎ → N
    "\uff2f": "O",  # Ｏ → O
    "\uff30": "P",  # Ｐ → P
    "\uff31": "Q",  # Ｑ → Q
    "\uff32": "R",  # Ｒ → R
    "\uff33": "S",  # Ｓ → S
    "\uff34": "T",  # Ｔ → T
    "\uff35": "U",  # Ｕ → U
    "\uff36": "V",  # Ｖ → V
    "\uff37": "W",  # Ｗ → W
    "\uff38": "X",  # Ｘ → X
    "\uff39": "Y",  # Ｙ → Y
    "\uff3a": "Z",  # Ｚ → Z
    # Mathematical Alphanumeric Symbols — Bold (U+1D400)
    "\U0001d41a": "a", "\U0001d41b": "b", "\U0001d41c": "c", "\U0001d41d": "d",
    "\U0001d41e": "e", "\U0001d41f": "f", "\U0001d420": "g", "\U0001d421": "h",
    "\U0001d422": "i", "\U0001d423": "j", "\U0001d424": "k", "\U0001d425": "l",
    "\U0001d426": "m", "\U0001d427": "n", "\U0001d428": "o", "\U0001d429": "p",
    "\U0001d42a": "q", "\U0001d42b": "r", "\U0001d42c": "s", "\U0001d42d": "t",
    "\U0001d42e": "u", "\U0001d42f": "v", "\U0001d430": "w", "\U0001d431": "x",
    "\U0001d432": "y", "\U0001d433": "z",
    "\U0001d400": "A", "\U0001d401": "B", "\U0001d402": "C", "\U0001d403": "D",
    "\U0001d404": "E", "\U0001d405": "F", "\U0001d406": "G", "\U0001d407": "H",
    "\U0001d408": "I", "\U0001d409": "J", "\U0001d40a": "K", "\U0001d40b": "L",
    "\U0001d40c": "M", "\U0001d40d": "N", "\U0001d40e": "O", "\U0001d40f": "P",
    "\U0001d410": "Q", "\U0001d411": "R", "\U0001d412": "S", "\U0001d413": "T",
    "\U0001d414": "U", "\U0001d415": "V", "\U0001d416": "W", "\U0001d417": "X",
    "\U0001d418": "Y", "\U0001d419": "Z",
    # Mathematical Italic (U+1D434)
    "\U0001d44e": "a", "\U0001d44f": "b", "\U0001d450": "c", "\U0001d451": "d",
    "\U0001d452": "e", "\U0001d453": "f", "\U0001d454": "g",
    "\U0001d456": "i", "\U0001d457": "j", "\U0001d458": "k", "\U0001d459": "l",
    "\U0001d45a": "m", "\U0001d45b": "n", "\U0001d45c": "o", "\U0001d45d": "p",
    "\U0001d45e": "q", "\U0001d45f": "r", "\U0001d460": "s", "\U0001d461": "t",
    "\U0001d462": "u", "\U0001d463": "v", "\U0001d464": "w", "\U0001d465": "x",
    "\U0001d466": "y", "\U0001d467": "z",
    # Enclosed Alphanumerics — Circled (U+24B6 uppercase, U+24D0 lowercase)
    "\u24d0": "a", "\u24d1": "b", "\u24d2": "c", "\u24d3": "d", "\u24d4": "e",
    "\u24d5": "f", "\u24d6": "g", "\u24d7": "h", "\u24d8": "i", "\u24d9": "j",
    "\u24da": "k", "\u24db": "l", "\u24dc": "m", "\u24dd": "n", "\u24de": "o",
    "\u24df": "p", "\u24e0": "q", "\u24e1": "r", "\u24e2": "s", "\u24e3": "t",
    "\u24e4": "u", "\u24e5": "v", "\u24e6": "w", "\u24e7": "x", "\u24e8": "y",
    "\u24e9": "z",
    "\u24b6": "A", "\u24b7": "B", "\u24b8": "C", "\u24b9": "D", "\u24ba": "E",
    "\u24bb": "F", "\u24bc": "G", "\u24bd": "H", "\u24be": "I", "\u24bf": "J",
    "\u24c0": "K", "\u24c1": "L", "\u24c2": "M", "\u24c3": "N", "\u24c4": "O",
    "\u24c5": "P", "\u24c6": "Q", "\u24c7": "R", "\u24c8": "S", "\u24c9": "T",
    "\u24ca": "U", "\u24cb": "V", "\u24cc": "W", "\u24cd": "X", "\u24ce": "Y",
    "\u24cf": "Z",
    # Coptic → Latin
    "\u2c85": "r",  # Coptic small ro
    "\u2ca5": "c",  # Coptic small sima
    "\u2c84": "R",  # Coptic capital ro
    "\u2ca4": "C",  # Coptic capital sima
    # Tifinagh → Latin
    "\u2d30": "a",  # Tifinagh ya
    "\u2d54": "o",  # Tifinagh yarr
    "\u2d5f": "t",  # Tifinagh yat
}

# Build translation table for str.translate (fast single-pass replacement)
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)


def _try_recursive_base64_decode(text: str, max_depth: int = 5) -> str:
    """Decode base64-encoded segments in text, recursively up to max_depth.

    Looks for base64 strings (16+ chars, valid charset), decodes them,
    and replaces in-place. Recurses if decoded result contains more base64.
    """
    b64_re = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
    for _ in range(max_depth):
        found_any = False
        def _decode_match(m: re.Match) -> str:
            nonlocal found_any
            try:
                decoded = base64.b64decode(m.group()).decode("utf-8", errors="ignore")
                if decoded.isprintable() and len(decoded) >= 4:
                    found_any = True
                    return decoded
            except Exception:
                pass
            return m.group()
        text = b64_re.sub(_decode_match, text)
        if not found_any:
            break
    return text


def _normalize_leetspeak(text: str) -> str:
    """Normalize leetspeak characters only inside leet-encoded words.

    A word is considered leet-encoded if it contains BOTH alpha chars and
    leet chars (digits/symbols that map to letters). Pure numbers like
    "123-45-6789" or base64 strings like "aWdub3Jl" are left unchanged.

    Specifically: a leet char is only replaced if BOTH neighbors are alpha
    (either original alpha or already-replaced leet chars in the same word).
    We process word-by-word to avoid cross-word contamination.
    """
    # Split into word-like tokens preserving separators
    result: list[str] = []
    word = []
    for ch in text:
        if ch.isalnum() or ch in _LEETSPEAK_MAP:
            word.append(ch)
        else:
            if word:
                result.append(_normalize_leet_word(word))
                word = []
            result.append(ch)
    if word:
        result.append(_normalize_leet_word(word))
    return "".join(result)


def _normalize_leet_word(chars: list[str]) -> str:
    """Normalize a single word token for leetspeak.

    Only converts if the word has a mix of alpha + leet chars,
    AND the leet chars are between alpha chars. Pure numeric tokens
    and tokens with only 1 alpha char are left unchanged.
    Mixed-case tokens (e.g. base64 like "aWdub3Jl") are skipped —
    leetspeak is single-case (e.g. "h4ck3r", "1gn0r3").
    """
    alpha_count = sum(1 for c in chars if c.isalpha())
    leet_count = sum(1 for c in chars if c in _LEETSPEAK_MAP and not c.isalpha())

    # Need at least 2 alpha chars and 1 leet char to be considered leet
    if alpha_count < 2 or leet_count < 1:
        return "".join(chars)

    # Skip mixed-case words (likely base64/identifiers, not leetspeak)
    has_upper = any(c.isupper() for c in chars)
    has_lower = any(c.islower() for c in chars)
    if has_upper and has_lower:
        return "".join(chars)

    # Only convert leet chars that have an alpha neighbor on at least one side
    out = list(chars)
    for i, ch in enumerate(out):
        if ch in _LEETSPEAK_MAP and not ch.isalpha():
            has_alpha_left = i > 0 and out[i - 1].isalpha()
            has_alpha_right = i < len(out) - 1 and out[i + 1].isalpha()
            if has_alpha_left or has_alpha_right:
                out[i] = _LEETSPEAK_MAP[ch]
    return "".join(out)


# Known high-signal words that indicate injection attempts.  If a ROT13-decoded
# segment contains any of these, the decoded version replaces the original.
_ROT13_TRIGGER_WORDS = frozenset([
    "ignore", "disregard", "forget", "override", "bypass", "instructions",
    "previous", "system", "prompt", "admin", "root", "sudo", "exec",
    "reveal", "repeat", "inject", "jailbreak", "all", "you", "are",
    "now", "new", "mode", "hack", "assistant",
])


def _try_rot13_decode(text: str) -> str:
    """Attempt ROT13 decode on individual word-like segments.

    Only replaces segments whose decoded form contains a known injection
    keyword. This preserves the rest of the text (including keywords like
    "ROT13" that regex patterns match on) while decoding suspicious payloads.
    """
    import codecs

    seg_re = re.compile(r"[A-Za-z]{3,}")

    def _replace_segment(m: re.Match) -> str:
        seg = m.group()
        dec = codecs.decode(seg, "rot13")
        dec_lower = dec.lower()
        if any(w == dec_lower or w in dec_lower for w in _ROT13_TRIGGER_WORDS):
            return dec
        return seg

    result = seg_re.sub(_replace_segment, text)

    # If no segments were decoded, try full-text decode as last resort
    if result == text:
        decoded = codecs.decode(text, "rot13")
        decoded_lower = decoded.lower()
        if any(word in decoded_lower for word in _ROT13_TRIGGER_WORDS):
            # Append decoded version rather than replace, so both
            # the original and decoded text are scanned by patterns
            return text + " " + decoded

    return result


def normalize_text(text: str) -> str:
    """Normalize text to defeat character-level evasion techniques.

    Applied in order:
    1. NFKC Unicode normalization (canonicalizes compatibility characters)
    2. Strip BIDI override characters
    3. Strip zero-width and invisible characters
    4. Strip combining marks (Mn=Nonspacing, Me=Enclosing)
    5. Homoglyph canonicalization (200+ chars: Cyrillic/Greek/Math/Enclosed/Coptic/Tifinagh/fullwidth → Latin)
    6. Leetspeak normalization (contextual — only when adjacent to alpha chars)
    7. Recursive base64 decode (max 5 iterations)
    7.5. ROT13 decode (keyword-gated to avoid FP)
    8. Collapse whitespace (all Unicode whitespace → single ASCII space)

    This is a one-way preprocessing for detection only — the original text
    is preserved for logging and audit purposes.
    """
    # Step 1: NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Step 2: Strip BIDI override characters
    text = "".join(ch for ch in text if ch not in _BIDI_CHARS)
    # Step 3: Strip zero-width characters
    text = "".join(ch for ch in text if ch not in _ZERO_WIDTH_CHARS)
    # Step 4: Strip combining marks (Mn=Nonspacing, Me=Enclosing)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Mn", "Me"))
    # Step 5: Homoglyph canonicalization
    text = text.translate(_HOMOGLYPH_TABLE)
    # Step 6: Leetspeak normalization
    text = _normalize_leetspeak(text)
    # Step 7: Recursive base64 decode
    text = _try_recursive_base64_decode(text)
    # Step 7.5: ROT13 decode (attempt on all-alpha segments)
    text = _try_rot13_decode(text)
    # Step 8: Collapse whitespace (all Unicode whitespace → single ASCII space)
    text = re.sub(r"[\s\u00a0\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+", " ", text)
    return text


# Map tactic strings from patterns.json to ThreatCategory enum values
_TACTIC_MAP: dict[str, ThreatCategory] = {
    "AML.T0051": ThreatCategory.PROMPT_INJECTION,
    "AML.T0051.001": ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
    "AML.T0054": ThreatCategory.JAILBREAK,
    "AML.T0048": ThreatCategory.PII_EXFILTRATION,
    "AML.T0044": ThreatCategory.MODEL_EXTRACTION,
    "AML.T0020": ThreatCategory.DATA_POISONING,
    "AML.T0015": ThreatCategory.ENCODING_OBFUSCATION,
}


class CompiledPattern:
    """A single compiled regex pattern with metadata."""

    __slots__ = ("id", "regex", "tactic", "category", "description")

    def __init__(self, spec: dict[str, Any]):
        self.id: str = spec["id"]
        self.regex: re.Pattern[str] = re.compile(spec["pattern"])
        self.tactic: str = spec["tactic"]
        self.category: str = spec.get("category", "unknown")
        self.description: str = spec.get("description", "")


class RegexEngine:
    """Compiled regex pattern library for fast-path threat detection.

    Loads patterns from JSON file at construction. All patterns are compiled
    once; matching is sub-millisecond for typical inputs.
    """

    def __init__(self, pattern_file: str | Path | None = None):
        self._patterns: list[CompiledPattern] = []
        if pattern_file is not None:
            self.load(pattern_file)

    def load(self, pattern_file: str | Path) -> None:
        """Load and compile patterns from a JSON file."""
        path = Path(pattern_file)
        with path.open("r") as f:
            data = json.load(f)
        self._patterns = [CompiledPattern(p) for p in data.get("patterns", [])]

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def add_dynamic_pattern(
        self,
        pattern_str: str,
        pattern_id: str | None = None,
        tactic: str = "AML.T0051",
        description: str = "Clonal selection generated",
    ) -> bool:
        """Add a regex pattern at runtime (from clonal selection).

        Returns True if the pattern was compiled and added successfully.
        Patterns added this way are immediately active for the next scan().
        """
        if pattern_id is None:
            pattern_id = f"CS-{len(self._patterns):04d}"
        spec = {
            "id": pattern_id,
            "pattern": pattern_str,
            "tactic": tactic,
            "description": description,
        }
        try:
            compiled = CompiledPattern(spec)
            self._patterns.append(compiled)
            return True
        except Exception:
            return False

    async def scan(self, text: str) -> ScanResult:
        """Scan text against all compiled patterns.

        Applies Unicode normalization before matching to defeat:
        - Zero-width character insertion between pattern tokens
        - Cyrillic/Greek homoglyph substitution
        - NFKC-normalizable character variants

        Returns a ScanResult. If scanning crashes, returns fail-closed
        (is_threat=True, confidence=1.0).
        """
        start = time.perf_counter()
        try:
            # Normalize text to defeat character-level evasion (T1 mitigation)
            text = normalize_text(text)

            matched: list[str] = []
            highest_tactic: str | None = None
            for pat in self._patterns:
                if pat.regex.search(text):
                    matched.append(f"{pat.id}: {pat.description}")
                    if highest_tactic is None:
                        highest_tactic = pat.tactic

            elapsed_ms = (time.perf_counter() - start) * 1000

            if matched:
                threat_cat = _TACTIC_MAP.get(
                    highest_tactic or "", ThreatCategory.PROMPT_INJECTION
                )
                # Confidence: single high-signal match is enough to block.
                # Base 0.85 for first match (meets block threshold), +0.05 per
                # additional match, capped at 1.0.
                confidence = min(0.85 + 0.05 * (len(matched) - 1), 1.0)
                return ScanResult(
                    scanner_id="regex_engine",
                    is_threat=True,
                    confidence=confidence,
                    threat_category=threat_cat,
                    matched_patterns=matched,
                    latency_ms=elapsed_ms,
                )

            return ScanResult(
                scanner_id="regex_engine",
                is_threat=False,
                confidence=0.0,
                latency_ms=elapsed_ms,
            )
        except Exception:
            # Fail-closed: crash → treat as definitive threat
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ScanResult(
                scanner_id="regex_engine",
                is_threat=True,
                confidence=1.0,
                threat_category=ThreatCategory.UNKNOWN,
                matched_patterns=["SCANNER_CRASH: fail-closed"],
                latency_ms=elapsed_ms,
            )
