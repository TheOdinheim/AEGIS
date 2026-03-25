"""Serves the AEGIS product landing page at /.

Static marketing page for investors and design partners.
No authentication required. Pure inline HTML/CSS/JS.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["landing"])

_LANDING_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AEGIS — The AI Immune System</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0e17;--bg2:#111827;--bg3:#1f2937;--border:#374151;
  --text:#e5e7eb;--text2:#9ca3af;--accent:#3b82f6;--green:#10b981;
  --red:#ef4444;--amber:#f59e0b;--purple:#8b5cf6;--cyan:#06b6d4;
}
html{scroll-behavior:smooth}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}

/* Layout */
.container{max-width:1100px;margin:0 auto;padding:0 24px}
section{padding:80px 0}
.section-label{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:var(--cyan);margin-bottom:12px;font-weight:600}
.section-title{font-size:32px;font-weight:800;margin-bottom:16px;letter-spacing:-0.5px}
.section-subtitle{color:var(--text2);font-size:16px;max-width:640px;margin-bottom:40px}

/* Nav */
nav{position:fixed;top:0;left:0;right:0;z-index:50;background:rgba(10,14,23,0.85);backdrop-filter:blur(12px);border-bottom:1px solid var(--border)}
.nav-inner{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;max-width:1100px;margin:0 auto}
.nav-logo{font-size:20px;font-weight:800;letter-spacing:2px;background:linear-gradient(135deg,var(--accent),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-links{display:flex;gap:24px;align-items:center}
.nav-links a{color:var(--text2);font-size:13px;font-weight:500;text-decoration:none;transition:color .2s}
.nav-links a:hover{color:var(--text)}
.nav-cta{padding:8px 20px;border-radius:8px;background:var(--accent);color:#fff!important;font-weight:600;font-size:13px;transition:background .2s}
.nav-cta:hover{background:#2563eb;text-decoration:none}

/* Hero */
.hero{padding:160px 0 100px;text-align:center}
.hero-logo{font-size:64px;font-weight:900;letter-spacing:6px;background:linear-gradient(135deg,var(--accent),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
.hero-tagline{font-size:24px;font-weight:300;color:var(--cyan);margin-bottom:4px;letter-spacing:1px}
.hero-sub{font-size:15px;color:var(--text2);margin-bottom:32px;font-weight:500;letter-spacing:0.5px}
.hero-desc{font-size:17px;color:var(--text);max-width:680px;margin:0 auto 40px;line-height:1.7}
.hero-buttons{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
.btn-primary{padding:14px 32px;border-radius:10px;background:var(--accent);color:#fff;font-weight:700;font-size:15px;border:none;cursor:pointer;transition:background .2s;text-decoration:none}
.btn-primary:hover{background:#2563eb;text-decoration:none}
.btn-secondary{padding:14px 32px;border-radius:10px;background:transparent;color:var(--text);font-weight:700;font-size:15px;border:1px solid var(--border);cursor:pointer;transition:all .2s;text-decoration:none}
.btn-secondary:hover{border-color:var(--accent);color:var(--accent);text-decoration:none}

/* Stat cards */
.stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}
.stat-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:28px 24px;text-align:center}
.stat-card .stat-num{font-size:22px;font-weight:800;margin-bottom:8px;color:var(--red)}
.stat-card .stat-text{font-size:14px;color:var(--text2);line-height:1.5}

/* Layers */
.layers-stack{display:flex;flex-direction:column;gap:8px;max-width:720px;margin:0 auto}
.layer-row{display:flex;align-items:center;padding:16px 20px;background:var(--bg2);border:1px solid var(--border);border-radius:10px;transition:border-color .3s,box-shadow .3s}
.layer-row:hover{border-color:var(--accent);box-shadow:0 0 20px rgba(59,130,246,0.1)}
.layer-id{font-weight:800;color:var(--accent);width:36px;font-size:14px;flex-shrink:0}
.layer-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);margin-right:14px;flex-shrink:0;animation:pulse-glow 2s ease-in-out infinite}
.layer-info{flex:1;min-width:0}
.layer-tech{font-size:14px;font-weight:600}
.layer-bio{font-size:12px;color:var(--text2);margin-top:2px}
@keyframes pulse-glow{
  0%,100%{box-shadow:0 0 6px var(--green);opacity:1}
  50%{box-shadow:0 0 14px var(--green);opacity:0.8}
}
.layers-note{text-align:center;color:var(--text2);font-size:14px;margin-top:28px;font-style:italic}

/* Validation metrics */
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.metric-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:24px;text-align:center}
.metric-value{font-size:32px;font-weight:800;background:linear-gradient(135deg,var(--green),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.metric-label{font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px}
.metric-detail{font-size:12px;color:var(--text2)}

/* Compliance */
.frameworks-row{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}
.fw-badge{padding:10px 20px;border-radius:8px;background:var(--bg2);border:1px solid var(--border);font-size:13px;font-weight:600;color:var(--text)}
.compliance-note{color:var(--text2);font-size:15px}

/* Steps */
.steps-row{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}
.step-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:28px 24px}
.step-num{font-size:36px;font-weight:900;background:linear-gradient(135deg,var(--accent),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px}
.step-title{font-size:15px;font-weight:700;margin-bottom:8px}
.step-desc{font-size:13px;color:var(--text2);line-height:1.5}

/* Footer */
footer{border-top:1px solid var(--border);padding:40px 0;text-align:center}
.footer-brand{font-size:14px;font-weight:700;margin-bottom:8px}
.footer-contact{font-size:13px;color:var(--text2);margin-bottom:4px}
.footer-links{display:flex;gap:20px;justify-content:center;margin-top:16px}
.footer-links a{font-size:13px;color:var(--text2)}

/* Fade-in animation */
.fade-in{opacity:0;transform:translateY(20px);transition:opacity 0.6s ease,transform 0.6s ease}
.fade-in.visible{opacity:1;transform:translateY(0)}

/* Responsive */
@media(max-width:768px){
  .hero{padding:120px 0 60px}
  .hero-logo{font-size:40px}
  .hero-tagline{font-size:18px}
  .hero-desc{font-size:15px}
  .stats-row{grid-template-columns:1fr}
  .metrics-grid{grid-template-columns:repeat(2,1fr)}
  .steps-row{grid-template-columns:1fr}
  .section-title{font-size:24px}
  .nav-links a:not(.nav-cta){display:none}
}
</style>
</head>
<body>

<nav>
  <div class="nav-inner">
    <span class="nav-logo">AEGIS</span>
    <div class="nav-links">
      <a href="#architecture">Architecture</a>
      <a href="#validation">Validation</a>
      <a href="#compliance">Compliance</a>
      <a href="/dashboard" class="nav-cta">Live Dashboard</a>
    </div>
  </div>
</nav>

<section class="hero">
  <div class="container">
    <div class="hero-logo">AEGIS</div>
    <div class="hero-tagline">The AI Immune System</div>
    <div class="hero-sub">Adaptive Enterprise Guard for Intelligent Systems</div>
    <p class="hero-desc">
      The first AI security platform built on the architecture of the human biological
      immune system. Seven layers of adaptive defense that learn, evolve, and heal &mdash;
      protecting every AI interaction in real time.
    </p>
    <div class="hero-buttons">
      <a href="/dashboard" class="btn-primary">View Live Dashboard &rarr;</a>
      <a href="mailto:odinheimllc@gmail.com?subject=AEGIS%20Technical%20Paper%20Request" class="btn-secondary">Read the Technical Paper &rarr;</a>
    </div>
  </div>
</section>

<section id="problem">
  <div class="container">
    <div class="section-label fade-in">The Problem</div>
    <h2 class="section-title fade-in">AI Systems Are Under Industrial-Scale Attack</h2>
    <div class="stats-row fade-in">
      <div class="stat-card">
        <div class="stat-num">82.4%</div>
        <div class="stat-text">of LLMs compromisable through inter-agent communication</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">$1.5B+</div>
        <div class="stat-text">in AI security acquisitions in 12 months &mdash; and still no unified defense</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">Aug 2026</div>
        <div class="stat-text">EU AI Act enforcement &mdash; &euro;35M penalties for non-compliance</div>
      </div>
    </div>
  </div>
</section>

<section id="architecture">
  <div class="container">
    <div class="section-label fade-in">The Architecture</div>
    <h2 class="section-title fade-in">Seven Layers. One Immune System.</h2>
    <div class="section-subtitle fade-in">Modeled on the human biological immune system. Each layer operates independently under assumed-breach posture &mdash; no layer trusts another layer's verdict.</div>
    <div class="layers-stack fade-in">
      <div class="layer-row">
        <span class="layer-id">L1</span>
        <span class="layer-dot"></span>
        <div class="layer-info">
          <div class="layer-tech">Barrier Layer</div>
          <div class="layer-bio">Skin &amp; Mucous Membranes &mdash; TLS, auth, rate limiting, schema validation</div>
        </div>
      </div>
      <div class="layer-row">
        <span class="layer-id">L2</span>
        <span class="layer-dot"></span>
        <div class="layer-info">
          <div class="layer-tech">Innate Detection</div>
          <div class="layer-bio">Pattern Recognition Receptors &mdash; 188+ regex patterns, blocklist, token guard (&lt;5ms)</div>
        </div>
      </div>
      <div class="layer-row">
        <span class="layer-id">L3</span>
        <span class="layer-dot"></span>
        <div class="layer-info">
          <div class="layer-tech">Adaptive Analysis</div>
          <div class="layer-bio">T-Cells &amp; B-Cells &mdash; DeBERTa classifier, semantic search, behavioral baselines</div>
        </div>
      </div>
      <div class="layer-row">
        <span class="layer-id">L4</span>
        <span class="layer-dot"></span>
        <div class="layer-info">
          <div class="layer-tech">Immune Memory</div>
          <div class="layer-bio">Memory B-Cells &amp; Antibodies &mdash; FAISS threat vault, clonal selection, 3-phase lifecycle</div>
        </div>
      </div>
      <div class="layer-row">
        <span class="layer-id">L5</span>
        <span class="layer-dot"></span>
        <div class="layer-info">
          <div class="layer-tech">Output Validation</div>
          <div class="layer-bio">Complement Cascade &mdash; PII redaction, toxicity, hallucination, leakage (5-stage)</div>
        </div>
      </div>
      <div class="layer-row">
        <span class="layer-id">L6</span>
        <span class="layer-dot"></span>
        <div class="layer-info">
          <div class="layer-tech">Policy Engine</div>
          <div class="layer-bio">Regulatory T-Cells &mdash; OPA/Rego, 5-level TLI, adaptive thresholds</div>
        </div>
      </div>
      <div class="layer-row">
        <span class="layer-id">L7</span>
        <span class="layer-dot"></span>
        <div class="layer-info">
          <div class="layer-tech">Self-Healing</div>
          <div class="layer-bio">Wound Healing &amp; Tissue Repair &mdash; Circuit breakers, fallback routing, session quarantine</div>
        </div>
      </div>
    </div>
    <p class="layers-note fade-in">Every layer communicates through an event bus. Detection at one layer automatically strengthens all others.</p>
  </div>
</section>

<section id="validation" style="background:var(--bg2)">
  <div class="container">
    <div class="section-label fade-in">Validation</div>
    <h2 class="section-title fade-in">Built and Validated. Not Pitched.</h2>
    <div class="section-subtitle fade-in">Every claim is backed by automated tests, adversarial red teaming, and reproducible benchmarks.</div>
    <div class="metrics-grid fade-in">
      <div class="metric-card">
        <div class="metric-value">3,139</div>
        <div class="metric-label">Tests</div>
        <div class="metric-detail">Zero failures. Zero skips.</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">96.36%</div>
        <div class="metric-label">True Positive Rate</div>
        <div class="metric-detail">Full-stack benchmark</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">0%</div>
        <div class="metric-label">False Positive Rate</div>
        <div class="metric-detail">500 benign business prompts</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">Red Team Phases</div>
        <div class="metric-detail">15 findings, all fixed</div>
      </div>
    </div>
  </div>
</section>

<section id="compliance">
  <div class="container">
    <div class="section-label fade-in">Compliance</div>
    <h2 class="section-title fade-in">One Platform. Six Frameworks.</h2>
    <div class="frameworks-row fade-in">
      <span class="fw-badge">NIST AI RMF</span>
      <span class="fw-badge">ISO 42001</span>
      <span class="fw-badge">EU AI Act</span>
      <span class="fw-badge">CMMC 2.0</span>
      <span class="fw-badge">SOC 2</span>
      <span class="fw-badge">FedRAMP</span>
    </div>
    <p class="compliance-note fade-in">71 controls mapped. 100% architectural coverage. Evidence collection automated via comprehensive audit logging.</p>
  </div>
</section>

<section id="deploy" style="background:var(--bg2)">
  <div class="container">
    <div class="section-label fade-in">Deploy</div>
    <h2 class="section-title fade-in">Deploy in Minutes. No Code Changes.</h2>
    <div class="section-subtitle fade-in">AEGIS is an OpenAI-compatible reverse proxy. Your applications connect by changing one URL.</div>
    <div class="steps-row fade-in">
      <div class="step-card">
        <div class="step-num">1</div>
        <div class="step-title">Point your base_url to AEGIS</div>
        <div class="step-desc">Replace your AI provider URL with the AEGIS gateway. Zero code changes in your application.</div>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <div class="step-title">Seven layers defend every request</div>
        <div class="step-desc">Every prompt and response flows through adaptive innate and learned defenses in real time. 30&ndash;150ms total overhead.</div>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <div class="step-title">Monitor from the live dashboard</div>
        <div class="step-desc">Real-time threat level, detection feed, circuit breaker status, and compliance coverage &mdash; all in one view.</div>
      </div>
    </div>
  </div>
</section>

<footer>
  <div class="container">
    <div class="footer-brand">AEGIS by Odin LLC</div>
    <div class="footer-contact"><a href="mailto:odinheimllc@gmail.com">odinheimllc@gmail.com</a></div>
    <div class="footer-contact">odinheim.io</div>
    <div class="footer-links">
      <a href="/dashboard">Dashboard</a>
      <a href="#architecture">Architecture</a>
      <a href="#compliance">Compliance</a>
    </div>
  </div>
</footer>

<script>
// IntersectionObserver fade-in
(function(){
  var els=document.querySelectorAll('.fade-in');
  if(!('IntersectionObserver' in window)){
    els.forEach(function(e){e.classList.add('visible')});
    return;
  }
  var obs=new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(entry.isIntersecting){
        entry.target.classList.add('visible');
        obs.unobserve(entry.target);
      }
    });
  },{threshold:0.15});
  els.forEach(function(e){obs.observe(e)});
})();
</script>
</body>
</html>"""


@router.get("/", include_in_schema=False)
async def landing_page(request: Request) -> HTMLResponse:
    """Serve the AEGIS product landing page."""
    return HTMLResponse(content=_LANDING_HTML)
