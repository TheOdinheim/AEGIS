"""Global test configuration.

Sets AEGIS_SKIP_MODEL_LOAD=true so that DeBERTa and sentence-transformers
are not downloaded/loaded during test runs. Tests that need the ML models
can override this per-test.
"""

import asyncio
import os

os.environ.setdefault("AEGIS_SKIP_MODEL_LOAD", "true")


def pytest_runtest_setup(item):
    """Ensure a fresh event loop exists before each test.

    Some tests use asyncio.run() which closes the event loop. Subsequent tests
    using asyncio.get_event_loop().run_until_complete() then fail with
    'There is no current event loop'. This hook creates a new loop if needed.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        try:
            loop = asyncio._get_running_loop()  # noqa: SLF001
        except AttributeError:
            loop = None
    # If no running loop, ensure get_event_loop() will work
    if loop is None:
        try:
            existing = asyncio.get_event_loop_policy().get_event_loop()
            if existing.is_closed():
                raise RuntimeError
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
