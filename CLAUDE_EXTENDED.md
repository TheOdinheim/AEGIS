# AEGIS Extended Documentation — Red Team, Multimodal, Battle Testing

**This file extends CLAUDE.md. Read CLAUDE.md first for core architecture.**

---

## Red Team Adversarial Framework

**Operational tool** (`red_team/`): Generates novel attacks designed to bypass AEGIS detection, runs them against the live pipeline, measures evasion rates, and validates immune system learning. NOT a CI test suite — it's used for continuous security validation against running instances.

### Attack Taxonomy (120 attacks in corpus)

| Target Layer | Technique | Count | Difficulty | Goal |
|---|---|---|---|---|
| **L2 Innate** | Unicode homoglyph (extended) | 10 | 3 | Bypass normalize_text() canonicalization |
| **L2 Innate** | Zero-width char insertion | 10 | 3 | Break regex token matching |
| **L2 Innate** | Encoding stacking (ROT13+B64) | 10 | 3 | Evade single-layer decoding |
| **L2 Innate** | Semantic restructuring | 10 | 4 | Avoid trigger keywords entirely |
| **L2 Innate** | Mixed Latin/Cyrillic script | 10 | 4 | Half-and-half script confusion |
| **L3 Adaptive** | Adversarial token insertion | 5 | 4 | Dilute DeBERTa attention |
| **L3 Adaptive** | Confidence boundary probing | 5 | 5 | Find exact DeBERTa threshold |
| **L3 Adaptive** | Context dilution | 5 | 4 | Bury injection in 800+ word doc |
| **L3 Adaptive** | Persona framing | 5 | 3 | Authority claim wrapping |
| **L3 Adaptive** | Hypothetical wrapping | 5 | 3 | Fictional framing |
| **L3 Adaptive** | Academic framing | 5 | 3 | Research claim wrapping |
| **L5 Output** | Partial PII (words) | 4 | 4 | SSN as spoken words |
| **L5 Output** | Encoded PII (B64/hex) | 4 | 4 | PII in encoded form |
| **L5 Output** | Steganographic PII | 4 | 5 | Acrostics, code comments |
| **L5 Output** | Structured PII | 4 | 3 | PII in JSON keys, table headers |
| **L5 Output** | Split PII | 4 | 4 | SSN spread across sentences |
| **Multi-layer** | L2-pass-L3-catch | 5 | 4 | No keywords, semantic intent only |
| **Multi-layer** | L3-pass-L5-catch | 5 | 4 | Benign prompt eliciting PII output |
| **Multi-layer** | Multi-turn escalation | 10 | 5 | 5-message benign-to-inject sequence |

### Campaign Pipeline

1. **Generate/Load** — `AdversarialAttackGenerator.generate_full_corpus()` or load from `adversarial_corpus.json`
2. **Initial Evasion Test** — `EvasionEngine.run_evasion_test()` sends each attack through AEGIS, classifies blocked (403) vs evaded (200)
3. **Learning Validation** — `LearningValidator.validate_learning()` checks antibody generation, sends technique variants, measures generalization rate
4. **Report** — `RedTeamReport.generate_full_report()` produces JSON + markdown with executive summary, per-layer/technique breakdown, recommendations

### Running Red Team

```bash
# Generate corpus
python3 -c "from red_team.attack_generator import AdversarialAttackGenerator; g = AdversarialAttackGenerator(); g.save_corpus(g.generate_full_corpus(), 'red_team/data/adversarial_corpus.json')"

# Run campaign (requires live AEGIS)
python3 -c "
import asyncio
from red_team.campaign_runner import CampaignRunner, CampaignConfig
config = CampaignConfig(name='phase1', aegis_url='http://localhost:8000', api_key='YOUR_KEY')
asyncio.run(CampaignRunner(config).run())
"

# Run red team tests (CI-safe, no live AEGIS needed)
python3 -m pytest tests/test_red_team.py -v
```

### Learning Validation Methodology

- **Antibody count**: Compare threat vault size before/after attack campaign
- **Generalization rate**: For each blocked attack, generate a technique variant (same method, different wording) and re-test. Variant also blocked = generalized. Variant evaded = only learned exact pattern.
- **Detection gaps**: Evaded attacks are logged with full metadata for manual analysis and code fixes
- **Report grades**: STRONG (<=5% evasion), ADEQUATE (<=15%), NEEDS_IMPROVEMENT (<=30%), CRITICAL (>30%)

## APT Campaign Simulations (Red Team Phase 2)

Six APT campaigns test AEGIS detection across all layers. Run in-process via TestClient (no live server). Each campaign reinitializes layers for clean state.

**Campaigns** (`red_team/apt_campaigns.py`):

| Campaign | Attacks | Technique | Tests |
|---|---|---|---|
| PHANTOM NEEDLE | 18 | 5-stage injection: recon → probing → regex evasion → ML evasion → extraction | System prompt extraction defense |
| SILENT SIPHON | 13 | L5 output evasion: word-spelled PII, base64, steganographic | PII detection coverage |
| SLOW BURN | 20 | 20-message multi-turn progressive jailbreak | Multi-turn escalation detection |
| HYDRA | 200 | 6 mutation strategies × top 20 attacks (cyrillic, extended_homoglyph, reorder, synonym, padding, encoding_mix) | Adversarial fuzzing resilience |
| GHOST PROTOCOL | 7 | Agent registration → capability escalation → injection | Multi-agent security |
| CASCADING FAILURE | 12 | Attacks during Redis kill + recovery | Graceful degradation under attack |

**Results** (DeBERTa/Presidio offline):
- PHANTOM NEEDLE: 77.8% evasion, attacker FAILED (extraction blocked)
- SILENT SIPHON: 100% evasion, attacker SUCCESS (regex PII insufficient)
- SLOW BURN: 50% evasion, attacker FAILED (jailbreak blocked at stage 3)
- HYDRA: 83% evasion, attacker SUCCESS (mutations evade regex)
- GHOST PROTOCOL: 71.4% evasion, attacker SUCCESS
- CASCADING FAILURE: 41.7% evasion, detection maintained during degradation

**Production Code Fixes Applied**:
1. **Extended homoglyphs** (`regex_engine.py`): Added 21 mappings — Armenian (6), Georgian (2), Cherokee (1), extended Cyrillic (5), Greek (2), Roman numerals (3), Latin extended (2). Total: 56+ homoglyph mappings.
2. **Combining mark stripping** (`regex_engine.py`): Strip Mn (nonspacing) and Me (enclosing) unicode categories after zero-width stripping.
3. **Zero-width expansion** (`regex_engine.py`): Expanded from 12 to 22 invisible characters (Hangul fillers, Khmer vowels, Braille blank, Mongolian separator, line/paragraph separators).
4. **Word-spelled SSN** (`pii_redactor.py`): New regex matching 9 consecutive number words.
5. **Base64 PII context** (`pii_redactor.py`): Detect base64 strings near PII keywords (SSN, password, etc.).
6. **Padding-resistant scoring** (`behavioral.py`): Per-hit minimum score (0.15) ensures keywords buried in filler still register.

**Open Gaps** (require production dependencies):
- Steganographic PII — needs Presidio NER
- Synonym/reorder mutations — needs DeBERTa (L3 Adaptive's designed role)
- Encoding mixing — needs recursive base64/ROT13 decoding
- Agent capability escalation — needs tighter pipeline integration

**Running campaigns**: `python3 -m red_team.run_red_team` (all) or `--campaign "PHANTOM NEEDLE"` (specific)

**Regression tests**: `tests/test_red_team_regression.py` — 71 tests validating each fix individually (homoglyphs, combining marks, zero-width, word SSN, base64 PII, padding scoring).

**Assessment report**: `red_team/data/red_team_assessment.md`

## Adversarial ML Attack Engine (Red Team Phase 3)

Algorithm-driven black-box attacks targeting AEGIS detection boundaries. All algorithms accept a `QueryFn = Callable[[str], tuple[int, str, float]]` callable abstracting transport. All tests are CI-safe with mocked query functions.

**Directory**: `red_team/adversarial_ml/`

```
red_team/adversarial_ml/
├── __init__.py              # QueryFn type alias, dataclasses (AdversarialResult, EvolutionResult, TimingAnalysis, BoundaryMap, RealWorldAttack, AdversarialAssessment)
├── attack_corpus_loader.py  # 200 hardcoded real-world attacks (6 categories), no network
├── black_box_attack.py      # BlackBoxAttacker: TextFooler (synonym), CharSwap (confusable), Paraphrase (50 templates)
├── mutation_engine.py       # EvolutionaryMutationEngine: tournament selection, crossover, 7 mutation operators
├── timing_oracle.py         # TimingOracle: Welch's t-test (no scipy), timing side-channel detection
├── boundary_mapper.py       # DecisionBoundaryMapper: L2 regex, L3 DeBERTa, L5 PII boundary probing
└── run_adversarial_ml.py    # Entry point, query_fn factories, 7-step assessment pipeline
```

**Algorithms**:
- **TextFooler**: Word importance ranking (delete-and-test), synonym substitution (500+ entries)
- **CharSwap**: 229 confusable character mappings (Math Alphanumeric U+1D400, Enclosed U+24B6, Coptic, Tifinagh — beyond normalize_text coverage), zero-width + combining mark insertions
- **Paraphrase**: 50 templates across 8 categories (active→passive, imperative→interrogative, direct→indirect, nominalization, double negation, euphemism, code switching, metaphorical)
- **Evolutionary**: Population-based (50 pop, 30 gen), tournament selection (k=3), sentence crossover, 7 mutation operators (synonym_replace, char_perturb, reorder, inject_benign, paraphrase_clause, encode_segment, language_switch)
- **Timing Oracle**: Welch's t-test (from scratch), p<0.05 = timing leak detected
- **Boundary Mapper**: L2 binary search on word count, L3 word interpolation binary search, L5 PII format variant testing

**Attack Corpus** (200 attacks, `red_team/data/real_world_corpus.json`):

| Category | Count | Examples |
|---|---|---|
| owasp | 50 | OWASP LLM Top 10 (prompt injection, output handling, DoS, supply chain, disclosure, plugins, excessive agency, overreliance, theft) |
| indirect_injection | 30 | HTML comments, markdown images, JSON fields, code comments, email chains, YAML, XML, zero-width chars |
| multi_language | 30 | FR, DE, ES, CN, AR, RU, KO, JP, HI, mixed-language attacks |
| jailbreak | 30 | DAN, STAN, Developer Mode, Grandma, opposite day, roleplay, AI conversation sim, sandbox framing |
| encoding | 30 | Base64, hex, ROT13, URL-encoding, leetspeak, Morse, NATO phonetic, pig latin, reverse, binary, HTML entities |
| logic_manipulation | 30 | Conditional logic, recursive instructions, meta-paradox, Socratic method, appeals to authority, game theory |

**Running**: `python3 -m red_team.adversarial_ml.run_adversarial_ml [--url URL] [--api-key KEY] [--output PATH]`

**Tests**: `tests/test_adversarial_ml.py` — 42 tests (9 test classes, all mocked, CI-safe)

## Extended Campaigns & Hardening (Red Team Phase 4)

Phase 4 combines all attack methods into sustained campaigns, fixes every detection gap, and produces the final battle-tested assessment.

**Extended campaigns** (`red_team/extended_campaigns.py`): 6 new campaigns (7-12):

| # | Campaign | Technique | Attacks |
|---|----------|-----------|---------|
| 7 | SHAPESHIFTER | Evolutionary evasion — 5 mutation generations, burst of best | ~170 |
| 8 | BABEL TOWER | Multi-language (10 languages) + mixed-language + transliterated | ~45 |
| 9 | THOUSAND CUTS | Hidden in traffic: 400 benign + 75 borderline + 25 attacks | 500 |
| 10 | INSIDE JOB | Document injection: HTML comments, markdown, JSON, YAML, BIDI, acrostic | 10 |
| 11 | MIRROR MIRROR | L5 PII evasion: 20 input formats + 10 output formats | 30 |
| 12 | FULL SPECTRUM | Everything combined: 700 requests with 5 checkpoint measurements | 700 |

**Normalization hardening** (`layers/innate/regex_engine.py`):
- `normalize_text()` expanded from 3 steps to 8 steps
- BIDI stripping (9 chars), leetspeak normalization (8 leet chars, context-aware), recursive base64 decode (max 3 iterations)
- Homoglyph map expanded: 56 → 200+ (Fullwidth, Math Bold/Italic, Enclosed, Coptic, Tifinagh)
- Combining mark stripping (Mn/Me categories)
- Leetspeak skips mixed-case tokens (protects base64 strings like `aWdub3Jl`)

**Detection hardening**:
- L2: 8 new indirect injection patterns (II-001 through II-008): HTML comments, markdown comments, JSON metadata, YAML config, code comments, third-person injection, footnotes, safety filter disable
- L5: 8 new PII patterns: code-context SSN/email/phone, URL-embedded SSN/email, reversed SSN, word-spelled CC (13-19 digits), word-spelled phone (10 digits)

**Regression tests**:
- `tests/test_normalization_hardened.py` — 42 tests: confusable map coverage, BIDI stripping, zero-width, homoglyphs, leetspeak, recursive base64, combined evasion
- `tests/test_pii_hardened.py` — 41 tests: code-context PII, URL-embedded PII, word-spelled CC/phone, reversed SSN, indirect injection patterns, existing PII unchanged

**Assessment reports**:
- `red_team/data/FINAL_RED_TEAM_ASSESSMENT.md` — Full assessment with OWASP mapping, residual risks, recommendations
- `red_team/data/final_metrics.json` — Machine-readable metrics

**OWASP LLM Top 10 Coverage**: All 10 risks mapped. STRONG coverage on prompt injection (L2+L3+L4), output handling (L5), DoS (L1+L2+L7), supply chain, and sensitive info disclosure. MODERATE on training data poisoning, plugin design, excessive agency, overreliance, model theft.

**Residual risks**: Paraphrased injection (needs L3 DeBERTa), steganographic PII (needs Presidio NER), non-English attacks (needs language detection), timing side-channel (needs constant-time padding).

## Infrastructure Security Assessment (Red Team Phase 5)

Phase 5 attacks AEGIS itself — not the AI interactions it protects, but the security product's own infrastructure, authentication, isolation, and resilience mechanisms.

**Directory**: `red_team/infrastructure/` — 8 attack modules + orchestrator

**Attack Modules and Findings**:

| Module | Class | Attacks | Key Findings |
|---|---|---|---|
| `auth_attacks.py` | AuthAttacker | Timing attack, format inference, 15 bypass techniques | Non-constant-time key comparison (`==` in main.py, `not in` set in barrier.py), no RBAC |
| `rate_limit_bypass.py` | RateLimitBypass | Session rotation, header injection, quota exhaustion | Rate limiting correctly tied to API key (not headers) |
| `tenant_isolation.py` | TenantIsolationTester | Cross-tenant leakage, privilege escalation, ID injection | No cross-tenant data leakage on tested endpoints |
| `backing_service_attacks.py` | BackingServiceAttacker | Redis/PostgreSQL analysis, dependency trust | Dev config without auth, wildcard dependency versions |
| `denial_of_service.py` | DenialOfServiceTester | Circuit breaker manipulation, TLI manipulation, session flooding, oversized requests | Global TLI (not per-tenant) — single attacker can DOS all; unbounded `_session_strikes` dict |
| `audit_integrity.py` | AuditIntegrityTester | Log injection, completeness, evidence destruction | Prompt hash collision via pipe separator; DELETE/PUT correctly rejected |
| `federated_poisoning.py` | FederatedPoisoningAttacker | Gradient poisoning, indicator poisoning, privacy budget exhaustion | Single-participant clipping bypass (min_participants=1); self-reported num_samples amplification; no indicator verification; reset_budget() unrestricted |
| `supply_chain_self.py` | SelfSupplyChainTester | Model tampering, pattern integrity, dependency audit | No file hash manifest; patterns.json writable; wildcard versions in requirements.txt |

**Security Vulnerabilities Found** (severity breakdown):

| Severity | Count | Examples |
|---|---|---|
| **Critical** | 0 | — |
| **High** | 3 | Timing side-channel on key validation, single-participant gradient poisoning, self-reported num_samples |
| **Medium** | 4 | No file integrity manifest, indicator poisoning without verification, global TLI, privacy budget race condition |
| **Low** | 3 | Unbounded session tracking, no lock file, dev Redis without auth |

**Recommendations** (priority order):
1. Use `hmac.compare_digest()` for API key comparison (constant-time)
2. Set `min_participants >= 3` for federated aggregation
3. Validate `num_samples` against actual training data size
4. Make TLI per-tenant to prevent cross-tenant DoS
5. Add `threading.Lock` to `DifferentialPrivacyEngine._spent_epsilon`
6. Create SHA-256 manifest for `data/*.json` files, verify on startup
7. Add indicator quality verification before vault integration
8. Pin all dependencies to exact versions, add lock file

**Tests**: `tests/test_infrastructure_security.py` — 60 tests across 10 test classes (auth, rate limiting, tenant isolation, DoS, audit, federated, backing services, supply chain, orchestrator, data models). All CI-safe, complete in ~30s.

**Running**:
```bash
# Infrastructure security tests
python3 -m pytest tests/test_infrastructure_security.py -v --tb=short

# Full orchestrator (all attack modules)
python3 -m red_team.infrastructure.run_infrastructure_attacks [--output results.json]
```

## Adaptive Meta-Learner (Red Team Phase 6)

Phase 6 builds an AI that attacks AEGIS's blind spots, learns from each attempt, generates increasingly sophisticated evasion techniques, then uses discoveries to harden detection.

**Directory**: `red_team/adaptive/` — 7 files

**5-Step Pipeline** (`run_adaptive.py`):
1. **Blind Spot Detection** — Analyzes all Phases 1-5 results for systematic weaknesses, coverage gaps, confidence boundaries, temporal degradation
2. **Attack Prediction** — 12-feature logistic regression trained on attack outcomes. Features: length, keyword density, encoding presence, non-ASCII ratio, homoglyph presence, zero-width presence, sentence count, question ratio, capitalization, special chars, repetition, multi-language. Generates predicted evasions via 4 mutation strategies.
3. **Co-Evolution** — 10-round arms race. Each round: generate attacks (seed → simple mutation → predictor-guided → advanced + novelty bonus), test against query_fn, retrain predictor, track evasion rate/FPR/vault growth. Determines winner (attacker/defender/stalemate) and convergence round.
4. **Evasion Fingerprinting** — Classifies each successful evasion into 12 root causes: NORMALIZATION_GAP, CLASSIFIER_BLIND_SPOT, CONTEXT_DILUTION, LANGUAGE_EVASION, FORMAT_EVASION, ENCODING_EVASION, SEMANTIC_RESTRUCTURING, PII_FORMAT_EVASION, LAYER_GAP, INFRASTRUCTURE_WEAKNESS, TIMING_EXPLOIT, TRUST_EXPLOITATION. Merges duplicates, ranks by frequency.
5. **Hardening Generation** — Produces concrete detection rules: 119 new character mappings (5 Unicode blocks), 6 new regex patterns (multi-language, context dilution, persona, meta-instruction, chain-of-thought), 3 new PII patterns (Morse, NATO digits, delimited digits), vault seed entries, configuration changes.

**Key Design Decisions**:
- All components accept `QueryFn = Callable[[str], tuple[int, str, float]]` for transport abstraction
- Co-evolution uses seeded Random for reproducibility
- Predictor uses logistic regression (no external ML deps beyond numpy)
- Fingerprinter is rule-based using payload characteristics
- All tests are CI-safe with mock query functions

**Co-Evolution Results** (mock query_fn, 10 rounds):
- Attacker wins: mutations learn to avoid blocked keywords
- Confirms L3 DeBERTa is essential — keyword-based L2 alone insufficient against adaptive attacker
- Convergence at round 6

**Data Models** (`red_team/adaptive/__init__.py`):
- `EvasionRootCause` enum (12 values)
- `BlindSpot`, `BlindSpotReport` — systematic weakness analysis
- `PredictionReport` — classifier accuracy, feature importances, predicted evasions
- `CoEvolutionRound`, `CoEvolutionReport` — per-round and aggregate arms race results
- `EvasionFingerprint`, `FingerprintReport` — root cause classification
- `HardeningRule`, `HardeningPlan` — concrete detection improvements
- `AdaptiveAssessment` — complete assessment with grade (STRONG/ADEQUATE/NEEDS_IMPROVEMENT/CRITICAL)

**Reports Generated**:
- `red_team/data/adaptive_assessment.json` — Full assessment results
- `red_team/data/ADAPTIVE_RED_TEAM_ASSESSMENT.md` — Phase 6 markdown report
- `red_team/data/COMPREHENSIVE_RED_TEAM_REPORT.md` — All 6 phases combined
- `red_team/data/comprehensive_metrics.json` — Machine-readable comprehensive metrics

**Tests**: `tests/test_adaptive_red_team.py` (63 tests) + `tests/test_hardening_regression.py` (32 tests) = 95 new tests. All CI-safe.

**Running**:
```bash
# Adaptive red team tests
python3 -m pytest tests/test_adaptive_red_team.py tests/test_hardening_regression.py -v --tb=short

# Run full adaptive assessment
python3 -m red_team.adaptive.run_adaptive [--rounds 10] [--attacks-per-round 20] [--output DIR]
```

## Multimodal Security

### Phase 1 — Image Security (Complete)

Image scanning via `layers/multimodal/`: OCR text extraction (Pillow heuristic + tesseract fallback), EXIF/XMP metadata stripping, LSB steganalysis (chi-square + RS analysis), perceptual hashing, image sanitization (re-encode to strip steganographic payloads), adversarial perturbation heuristics. Orchestrated by `ImageScanner` (L2 fast path, <10ms) and `ImageAnalyzer` (L3 slow path, sanitize-and-compare pixel difference).

### Phase 2 — Document Security (Complete)

Document scanning: `DocumentTextExtractor` (PDF, HTML, Markdown, JSON, CSV, XML, YAML, Office formats via format-aware parsing), `HiddenContentDetector` (invisible CSS, HTML comments, zero-width chars, metadata injection, PDF JavaScript, VBA macros), `FormatValidator` (magic bytes, polyglot detection, size limits). `DocumentScanner` (L2) and `DocumentAnalyzer` (L3 with DeBERTa + semantic search).

### Phase 3 — Audio Security (Complete)

Voice-enabled AI models (GPT-4o voice, Gemini, Qwen2.5-Omni) process audio alongside text. Research shows adversarial audio perturbations achieve 80-100% attack success rates against commercial ASR systems. WhisperInject demonstrates benign-sounding audio that covertly induces harmful text generation with 86%+ success.

**Defense principle**: Transcribe audio to text → feed through existing L2/L3 text detection. Additionally, spectral analysis detects adversarial perturbations, and audio sanitization via re-encoding destroys them.

**Modules** (`layers/multimodal/`):

| Module | Class | Function |
|--------|-------|----------|
| `audio_transcriber.py` | `AudioTranscriber` | Multi-backend transcription: Whisper → speech_recognition → unavailable. Format detection by magic bytes (WAV, MP3, FLAC, OGG, WebM). WAV duration parsing from header. |
| `spectral_analyzer.py` | `SpectralAnalyzer` | FFT-based spectral analysis. Detects: ultrasonic (>20kHz, inaudible but model-processable), infrasonic (<20Hz, hidden data), spectral entropy anomalies, spectral discontinuities (adversarial bursts via windowed centroid analysis). |
| `audio_sanitizer.py` | `AudioSanitizer` | WaveGuard-style re-encoding: decode → low-pass filter at 16kHz (removes ultrasonic) → high-pass at 80Hz (removes infrasonic) → re-encode as 16kHz 16-bit mono WAV. Destroys encoding-specific adversarial artifacts. |
| `audio_scanner.py` | `AudioScanner` | L2 fast path (<10ms excl. transcription): format validation, size check (<100MB), duration check (<30min), spectral analysis, transcribe → regex engine scan. |
| `audio_analyzer.py` | `AudioAnalyzer` | L3 slow path (50-200ms): deep transcription → DeBERTa injection classifier, WaveGuard comparison (sanitize, re-transcribe, word-level edit distance divergence > 0.1 = adversarial), full spectral analysis, cross-modal consistency check. |

**WaveGuard Comparison**: Sanitize audio by re-encoding → transcribe both original and sanitized → compute word-level edit distance / total words. If divergence > 0.1 → flag as potential adversarial audio. Clean audio produces identical transcriptions; adversarial perturbations are destroyed by re-encoding, changing the transcription.

**Audio extraction from messages** (`MultimodalPreprocessor._extract_audio`): Handles three formats:
- `{"type": "input_audio", "input_audio": {"data": "base64...", "format": "wav"}}` — OpenAI format
- `{"type": "image_url", "image_url": {"url": "data:audio/wav;base64,..."}}` — data URI
- `{"type": "file", "file": {"data": "base64...", "mime_type": "audio/wav"}}` — file attachment

**Configuration** (env vars):
- `AEGIS_MULTIMODAL_AUDIO_SCANNING_ENABLED` (default: true)
- `AEGIS_MULTIMODAL_AUDIO_MAX_SIZE_MB` (default: 100)
- `AEGIS_MULTIMODAL_AUDIO_MAX_DURATION_SECONDS` (default: 1800)

**Metrics**:
- `aegis_multimodal_audio_scanned_total` (Counter)
- `aegis_multimodal_audio_threats_detected_total` (Counter, label: threat_type)
- `aegis_multimodal_audio_transcription_latency_seconds` (Histogram)

**Tests**: `tests/test_multimodal_audio.py` — 55 tests covering transcription, spectral analysis, sanitization, scanner integration, WaveGuard comparison, preprocessor integration, config, and metrics.

### Phase 4 — Cross-Modal Correlation & Tool Use Security (Complete)

Cross-modal attacks exploit the shared semantic space between modalities. An image containing injection text combined with a benign text prompt creates a compound attack that neither image-only nor text-only scanners catch individually. Tool use attacks exploit function calling as a distinct jailbreak vector.

**Cross-Modal Correlation Engine** (`layers/multimodal/cross_modal_engine.py`):

Four correlation checks run after all per-modality scans complete:

| Check | Description | Confidence |
|-------|-------------|------------|
| **Modality laundering** | Text prompt is clean but media (image/document/audio) contains injection. Amplification factor applied to media confidence. | Base × 1.5 (configurable) |
| **Semantic inconsistency** | Keyword overlap between text prompt and extracted media text is below threshold (default 0.1), combined with media threats. | 0.6–0.85 |
| **Progressive escalation** | Session starts text-only, then introduces media with rising threat scores. Detected via per-session modality history (10 turns). | 0.5–0.80 |
| **Volume anomaly** | Excessive attachments: >5 images, >3 documents, or >2 audio files per request. | 0.70 |

`CrossModalCorrelationEngine.correlate()` takes text scan result, per-modality scan results, extracted text, session ID, and media counts. Returns `CrossModalReport` with `should_block`, `max_confidence`, `is_threat` properties.

**Tool Use Scanner** (`layers/multimodal/tool_use_scanner.py`):

Three scanning capabilities:

| Capability | Function | Detection |
|-----------|----------|-----------|
| **Definition scanning** | Scans function names, descriptions, and parameter descriptions against 10 compiled injection patterns | Confidence ≥ 0.90 |
| **Chain analysis** | Tracks per-session tool call sequences. Flags suspicious chains: search/retrieval → code execution, file read → any dangerous tool | Confidence 0.75 |
| **Output scanning** | Scans tool-role messages for injection patterns (search results containing prompt injection) | Confidence ≥ 0.85 |

`ToolUseScanner` supports optional `regex_engine` for additional pattern matching. Per-session chain history kept to 20 calls. Session cleanup via `clear_session()`.

**Configuration** (env vars):
- `AEGIS_MULTIMODAL_CROSS_MODAL_ENABLED` (default: true)
- `AEGIS_MULTIMODAL_TOOL_DEFINITION_SCANNING` (default: true)
- `AEGIS_MULTIMODAL_TOOL_OUTPUT_SCANNING` (default: true)

**Metrics**:
- `aegis_cross_modal_laundering_detected_total` (Counter)
- `aegis_cross_modal_inconsistency_total` (Counter)
- `aegis_tool_definition_threats_total` (Counter)
- `aegis_tool_chain_anomalies_total` (Counter)

**Tests**: `tests/test_cross_modal.py` — 46 tests (laundering 5, inconsistency 4, escalation 3, volume 3, report 5, definition scanning 7, chain analysis 6, output scanning 5, report 3, keywords 4). `tests/test_multimodal_integration.py` — 12 end-to-end integration tests via TestClient (image/audio/tool use pipeline, cross-modal standalone, config, metrics).

### Multimodal Priority Matrix

| Attack Vector | Detection Layer | Module | Confidence |
|---|---|---|---|
| Text in image (OCR injection) | L2 Innate | ImageScanner → regex_engine | 0.85+ |
| Steganographic payload | L2 Innate | Steganalyzer | 0.70+ |
| Adversarial image perturbation | L3 Adaptive | ImageAnalyzer (sanitize-compare) | 0.75+ |
| Hidden text in document (CSS/comments) | L2 Innate | HiddenContentDetector | 0.85+ |
| Document macro/script execution | L2 Innate | DocumentScanner | 0.90+ |
| Polyglot file attack | L2 Innate | FormatValidator | 0.85+ |
| Adversarial audio perturbation | L3 Adaptive | AudioAnalyzer (WaveGuard) | 0.75+ |
| Ultrasonic/infrasonic injection | L2 Innate | SpectralAnalyzer | 0.80+ |
| Cross-modal laundering | Post-scan | CrossModalCorrelationEngine | 0.85+ (amplified) |
| Semantic inconsistency | Post-scan | CrossModalCorrelationEngine | 0.60–0.85 |
| Tool definition injection | Pre-scan | ToolUseScanner | 0.90+ |
| Tool chain attack | Pre-scan | ToolUseScanner | 0.75 |
| Tool output injection | Pre-scan | ToolUseScanner | 0.85+ |
