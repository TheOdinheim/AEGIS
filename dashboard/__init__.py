"""AEGIS Production Dashboard — API, SSE, and React frontend."""

from __future__ import annotations

import sys
from types import ModuleType


def _get_main() -> ModuleType:
    """Return the running main module regardless of how uvicorn loaded it.

    In Docker, uvicorn runs ``aegis.main:app`` so the module is registered
    as ``aegis.main`` in sys.modules.  Locally (and in tests), it's ``main``.

    Some test files do ``import aegis.main as m`` which creates a *second*
    module object under a different sys.modules key.  The lifespan (which
    sets _config, _barrier, etc.) only runs on *one* of them.  We prefer
    whichever copy actually has _config populated; if neither does yet we
    fall back to ``main`` (the name used by the test harness).
    """
    candidates = (
        sys.modules.get("main"),
        sys.modules.get("aegis.main"),
    )
    # Prefer whichever has _config set (lifespan ran on it)
    for c in candidates:
        if c is not None and getattr(c, "_config", None) is not None:
            return c
    # Neither has _config yet — return whichever exists (prefer 'main')
    for c in candidates:
        if c is not None:
            return c
    import main as m  # type: ignore[no-redef]
    return m
