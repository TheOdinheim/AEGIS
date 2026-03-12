"""
AEGIS stress test runner — manages subprocess lifecycle for mock upstream and AEGIS.

Spawns uvicorn for both the mock upstream (port 9999) and AEGIS (port 8765),
waits for health endpoints, and provides clean shutdown.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time

import httpx


class AegisStressRunner:
    """Manages mock upstream + AEGIS server processes for stress testing."""

    def __init__(
        self,
        *,
        mock_port: int = 9999,
        aegis_port: int = 8765,
        api_key: str = "stress-test-key",
        mock_latency_ms: float = 50.0,
        mock_error_rate: float = 0.0,
        mock_toxic_rate: float = 0.0,
        mock_pii_rate: float = 0.0,
    ):
        self._mock_port = mock_port
        self._aegis_port = aegis_port
        self._api_key = api_key
        self._mock_latency_ms = mock_latency_ms
        self._mock_error_rate = mock_error_rate
        self._mock_toxic_rate = mock_toxic_rate
        self._mock_pii_rate = mock_pii_rate
        self._mock_proc: subprocess.Popen | None = None
        self._aegis_proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._aegis_port}"

    @property
    def api_key(self) -> str:
        return self._api_key

    async def start(self, timeout: float = 30.0) -> None:
        """Start mock upstream and AEGIS servers, wait for health."""
        # Find the aegis package root
        aegis_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        mock_env = os.environ.copy()
        mock_env["MOCK_LATENCY_MS"] = str(self._mock_latency_ms)
        mock_env["MOCK_ERROR_RATE"] = str(self._mock_error_rate)
        mock_env["MOCK_TOXIC_RATE"] = str(self._mock_toxic_rate)
        mock_env["MOCK_PII_RATE"] = str(self._mock_pii_rate)

        # Start mock upstream
        self._mock_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "tests.stress.mock_upstream:app",
                "--host", "127.0.0.1",
                "--port", str(self._mock_port),
                "--log-level", "warning",
            ],
            cwd=aegis_root,
            env=mock_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for mock health
        await self._wait_for_health(
            f"http://127.0.0.1:{self._mock_port}/health",
            timeout=timeout,
        )

        # Start AEGIS
        aegis_env = os.environ.copy()
        aegis_env["AEGIS_UPSTREAM_URL"] = f"http://127.0.0.1:{self._mock_port}"
        aegis_env["AEGIS_API_KEY"] = self._api_key
        aegis_env["AEGIS_UPSTREAM_API_KEY"] = "mock-key"
        aegis_env["AEGIS_SKIP_MODEL_LOAD"] = "true"

        self._aegis_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "aegis.main:app",
                "--host", "127.0.0.1",
                "--port", str(self._aegis_port),
                "--log-level", "warning",
            ],
            cwd=aegis_root,
            env=aegis_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for AEGIS health
        await self._wait_for_health(
            f"http://127.0.0.1:{self._aegis_port}/health",
            timeout=timeout,
        )

    async def stop(self) -> None:
        """Terminate both servers gracefully."""
        for proc in (self._aegis_proc, self._mock_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

    async def _wait_for_health(self, url: str, timeout: float = 30.0) -> None:
        """Poll health endpoint until it responds 200."""
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.2)
        raise TimeoutError(f"Health check failed for {url} after {timeout}s")
