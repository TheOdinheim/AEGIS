"""
Stress test report generation.

Produces human-readable and JSON reports from load test results.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from tests.stress.load_generator import LoadTestResult


def generate_report(result: LoadTestResult, scenario_name: str) -> dict:
    """Generate a structured report dict from load test results."""
    return {
        "scenario": scenario_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_requests": result.total_requests,
            "successful": result.successful,
            "blocked": result.blocked,
            "errors": result.errors,
            "success_rate": round(
                result.successful / max(result.total_requests, 1) * 100, 2
            ),
        },
        "latency": {
            "p50_ms": round(result.p50_ms, 2),
            "p95_ms": round(result.p95_ms, 2),
            "p99_ms": round(result.p99_ms, 2),
            "max_ms": round(result.max_ms, 2),
        },
        "throughput": {
            "rps_achieved": round(result.rps_achieved, 2),
            "duration_seconds": round(result.duration_seconds, 2),
        },
    }


def print_report(report: dict) -> None:
    """Print a human-readable report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  AEGIS Stress Test: {report['scenario']}")
    print(f"{'=' * 60}")

    s = report["summary"]
    print(f"  Total: {s['total_requests']}  |  OK: {s['successful']}  |"
          f"  Blocked: {s['blocked']}  |  Errors: {s['errors']}")
    print(f"  Success rate: {s['success_rate']}%")

    lat = report["latency"]
    print(f"\n  Latency (ms):  P50={lat['p50_ms']}  P95={lat['p95_ms']}"
          f"  P99={lat['p99_ms']}  Max={lat['max_ms']}")

    tp = report["throughput"]
    print(f"  Throughput: {tp['rps_achieved']} RPS over {tp['duration_seconds']}s")
    print(f"{'=' * 60}\n")


def save_report(report: dict, output_dir: str = "/tmp/aegis-stress-reports") -> str:
    """Save report as JSON file. Returns the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"stress_{report['scenario']}_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    return filepath
