"""
AEGIS Benchmark Runner — measures detection rate and false positive rate.

Loads the benign and attack corpora, runs every prompt through the full
AEGIS pipeline via FastAPI TestClient, and produces a detection rate matrix.

Run directly:
    python -m tests.benchmark.run_benchmark

Key metrics produced:
    - False Positive Rate (FPR): % of benign prompts incorrectly blocked
    - True Positive Rate (TPR full stack): % of attacks blocked with all layers
    - True Positive Rate (innate only): % blocked by L2 regex alone
    - Per-industry FPR breakdown
    - Per-attack-category detection rate
    - Average latency per layer
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import patch

from fastapi.testclient import TestClient

from aegis.config import AegisConfig, BarrierConfig, HealingConfig
from aegis.layers.audit import reset_audit_logger
from aegis.main import _init_layers, app
import aegis.main as main_module

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
API_KEY = "aegis-benchmark-key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_upstream_response(content: str = "Hello, how can I help you?") -> dict:
    return {
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _patch_upstream():
    async def mock_forward(body, upstream_url):
        return _mock_upstream_response()
    return patch("aegis.main._forward_to_upstream", side_effect=mock_forward)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _chat_body(content: str) -> dict[str, Any]:
    return {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": content}],
    }


def _setup_aegis() -> TestClient:
    """Initialize AEGIS layers and return a test client."""
    config = AegisConfig(
        api_key=API_KEY,
        upstream_url="https://mock-upstream.test",
        upstream_api_key="test-key",
        barrier=BarrierConfig(
            rate_limit_rpm=9999,
            rate_limit_burst=9999,
            ip_reputation_enabled=False,
        ),
        healing=HealingConfig(quarantine_threshold=9999),
    )

    @asynccontextmanager
    async def test_lifespan(a) -> AsyncGenerator[None, None]:
        yield

    app.router.lifespan_context = test_lifespan
    _init_layers(config)
    return TestClient(app, raise_server_exceptions=False)


def _teardown_aegis() -> None:
    """Reset all AEGIS module-level state."""
    main_module._barrier = None
    main_module._innate = None
    main_module._adaptive = None
    main_module._output = None
    main_module._policy = None
    main_module._healing = None
    main_module._vault = None
    main_module._signature_store = None
    main_module._signature_generator = None
    main_module._supply_chain = None
    main_module._config = None
    main_module._http_client = None
    main_module._audit = None
    reset_audit_logger()


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def _run_prompt(client: TestClient, prompt: str) -> dict[str, Any]:
    """Run a single prompt through the pipeline and capture results."""
    start = time.perf_counter()
    with _patch_upstream():
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body(prompt),
            headers=_headers(),
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    blocked = resp.status_code != 200
    result: dict[str, Any] = {
        "status_code": resp.status_code,
        "blocked": blocked,
        "latency_ms": round(elapsed_ms, 2),
    }

    # Try to extract detection info from response
    if blocked:
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", "")
            if "innate" in msg.lower():
                result["detected_by"] = "L2"
            elif "adaptive" in msg.lower():
                result["detected_by"] = "L3"
            elif "output" in msg.lower():
                result["detected_by"] = "L5"
            elif "policy" in msg.lower():
                result["detected_by"] = "L6"
            elif "barrier" in msg.lower() or resp.status_code in (400, 401, 429):
                result["detected_by"] = "L1"
            else:
                result["detected_by"] = "unknown"
        except Exception:
            result["detected_by"] = "unknown"
    else:
        result["detected_by"] = None

    return result


def _check_innate_only(prompt: str) -> bool:
    """Check if L2 innate layer alone would block this prompt.

    Uses the innate layer's scan directly, bypassing the full pipeline.
    """
    import asyncio
    from aegis.models.request_context import RequestContext

    innate = main_module._innate
    if not innate:
        return False

    context = RequestContext(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        raw_body=prompt.encode(),
    )

    try:
        loop = asyncio.new_event_loop()
        report = loop.run_until_complete(innate.scan(context))
        loop.close()
        return report.should_block
    except Exception:
        return False


def run_benchmark() -> dict[str, Any]:
    """Run the complete benchmark and return results.

    Returns a dict containing all metrics, per-prompt results, and summary.
    """
    # Load corpora
    benign_path = DATA_DIR / "benchmark_benign.json"
    attacks_path = DATA_DIR / "benchmark_attacks.json"

    if not benign_path.exists() or not attacks_path.exists():
        from tests.benchmark.generate_corpus import generate
        generate()

    with open(benign_path) as f:
        benign_data = json.load(f)
    with open(attacks_path) as f:
        attacks_data = json.load(f)

    benign_prompts = benign_data["prompts"]
    attack_prompts = attacks_data["prompts"]

    # Initialize AEGIS
    client = _setup_aegis()

    try:
        # --- Run benign prompts ---
        benign_results = []
        for entry in benign_prompts:
            result = _run_prompt(client, entry["prompt"])
            result["id"] = entry["id"]
            result["industry"] = entry["industry"]
            benign_results.append(result)

        # --- Run attack prompts (full stack) ---
        attack_results = []
        for entry in attack_prompts:
            result = _run_prompt(client, entry["prompt"])
            result["id"] = entry["id"]
            result["category"] = entry["category"]
            result["expected_layer"] = entry["expected_layer"]
            result["atlas_tactic"] = entry["atlas_tactic"]
            attack_results.append(result)

        # --- Innate-only pass on attacks ---
        innate_only_blocked = 0
        for entry in attack_prompts:
            if _check_innate_only(entry["prompt"]):
                innate_only_blocked += 1

    finally:
        _teardown_aegis()

    # --- Compute metrics ---
    total_benign = len(benign_results)
    total_attacks = len(attack_results)

    benign_blocked = sum(1 for r in benign_results if r["blocked"])
    attacks_blocked = sum(1 for r in attack_results if r["blocked"])

    fpr = benign_blocked / total_benign if total_benign else 0.0
    tpr_full = attacks_blocked / total_attacks if total_attacks else 0.0
    tpr_innate = innate_only_blocked / total_attacks if total_attacks else 0.0

    # TPR no-ML is same as full stack in bootstrap mode (DeBERTa offline)
    tpr_no_ml = tpr_full

    # Per-industry FPR
    industry_fpr: dict[str, dict[str, Any]] = {}
    industry_groups: dict[str, list] = defaultdict(list)
    for r in benign_results:
        industry_groups[r["industry"]].append(r)
    for industry, results in sorted(industry_groups.items()):
        blocked = sum(1 for r in results if r["blocked"])
        industry_fpr[industry] = {
            "total": len(results),
            "blocked": blocked,
            "fpr_pct": round(blocked / len(results) * 100, 2) if results else 0.0,
            "blocked_ids": [r["id"] for r in results if r["blocked"]],
        }

    # Per-category detection rate
    category_detection: dict[str, dict[str, Any]] = {}
    cat_groups: dict[str, list] = defaultdict(list)
    for r in attack_results:
        cat_groups[r["category"]].append(r)
    for category, results in sorted(cat_groups.items()):
        blocked = sum(1 for r in results if r["blocked"])
        category_detection[category] = {
            "total": len(results),
            "blocked": blocked,
            "detection_rate_pct": round(blocked / len(results) * 100, 2) if results else 0.0,
            "missed_ids": [r["id"] for r in results if not r["blocked"]],
        }

    # Average latency
    benign_latencies = [r["latency_ms"] for r in benign_results]
    attack_latencies = [r["latency_ms"] for r in attack_results]
    avg_benign_latency = sum(benign_latencies) / len(benign_latencies) if benign_latencies else 0
    avg_attack_latency = sum(attack_latencies) / len(attack_latencies) if attack_latencies else 0

    results = {
        "summary": {
            "total_benign": total_benign,
            "total_attacks": total_attacks,
            "benign_blocked": benign_blocked,
            "attacks_blocked": attacks_blocked,
            "innate_only_blocked": innate_only_blocked,
            "false_positive_rate_pct": round(fpr * 100, 2),
            "true_positive_rate_full_stack_pct": round(tpr_full * 100, 2),
            "true_positive_rate_innate_only_pct": round(tpr_innate * 100, 2),
            "true_positive_rate_no_ml_pct": round(tpr_no_ml * 100, 2),
            "avg_benign_latency_ms": round(avg_benign_latency, 2),
            "avg_attack_latency_ms": round(avg_attack_latency, 2),
        },
        "per_industry_fpr": industry_fpr,
        "per_category_detection": category_detection,
        "benign_results": benign_results,
        "attack_results": attack_results,
    }

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary table to stdout."""
    s = results["summary"]

    print("\n" + "=" * 72)
    print("  AEGIS BENCHMARK RESULTS")
    print("=" * 72)

    print(f"\n{'Metric':<45} {'Value':>10}")
    print("-" * 57)
    print(f"{'Total benign prompts':<45} {s['total_benign']:>10}")
    print(f"{'Total attack prompts':<45} {s['total_attacks']:>10}")
    print(f"{'Benign blocked (false positives)':<45} {s['benign_blocked']:>10}")
    print(f"{'Attacks blocked (true positives)':<45} {s['attacks_blocked']:>10}")
    print()
    print(f"{'FALSE POSITIVE RATE':<45} {s['false_positive_rate_pct']:>9.2f}%")
    print(f"{'TRUE POSITIVE RATE (full stack)':<45} {s['true_positive_rate_full_stack_pct']:>9.2f}%")
    print(f"{'TRUE POSITIVE RATE (innate L2 only)':<45} {s['true_positive_rate_innate_only_pct']:>9.2f}%")
    print(f"{'TRUE POSITIVE RATE (no ML / bootstrap)':<45} {s['true_positive_rate_no_ml_pct']:>9.2f}%")
    print()
    print(f"{'Avg latency per benign prompt (ms)':<45} {s['avg_benign_latency_ms']:>9.2f}")
    print(f"{'Avg latency per attack prompt (ms)':<45} {s['avg_attack_latency_ms']:>9.2f}")

    print(f"\n{'--- Per-Industry False Positive Rate ---':^57}")
    print(f"{'Industry':<25} {'Total':>7} {'Blocked':>9} {'FPR %':>9}")
    print("-" * 57)
    for industry, data in sorted(results["per_industry_fpr"].items()):
        print(f"{industry:<25} {data['total']:>7} {data['blocked']:>9} {data['fpr_pct']:>8.2f}%")

    print(f"\n{'--- Per-Category Detection Rate ---':^57}")
    print(f"{'Category':<30} {'Total':>7} {'Blocked':>9} {'Det %':>9}")
    print("-" * 57)
    for category, data in sorted(results["per_category_detection"].items()):
        print(f"{category:<30} {data['total']:>7} {data['blocked']:>9} {data['detection_rate_pct']:>8.2f}%")

    # Missed attacks summary
    missed = [r for r in results["attack_results"] if not r["blocked"]]
    if missed:
        print(f"\n{'--- Missed Attacks ({len(missed)} total) ---':^57}")
        for r in missed:
            print(f"  {r['id']:<12} [{r['category']}]")

    print("\n" + "=" * 72)


def save_results(results: dict[str, Any]) -> Path:
    """Save results to data/benchmark_results.json."""
    output_path = DATA_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return output_path


def main() -> dict[str, Any]:
    """Run full benchmark, print summary, save results."""
    # Ensure corpora exist
    benign_path = DATA_DIR / "benchmark_benign.json"
    attacks_path = DATA_DIR / "benchmark_attacks.json"
    if not benign_path.exists() or not attacks_path.exists():
        from tests.benchmark.generate_corpus import generate
        generate()

    results = run_benchmark()
    print_summary(results)
    save_results(results)
    return results


if __name__ == "__main__":
    main()
