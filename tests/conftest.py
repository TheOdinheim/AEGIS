"""Global test configuration.

Sets AEGIS_SKIP_MODEL_LOAD=true so that DeBERTa and sentence-transformers
are not downloaded/loaded during test runs. Tests that need the ML models
can override this per-test.
"""

import os

os.environ.setdefault("AEGIS_SKIP_MODEL_LOAD", "true")
