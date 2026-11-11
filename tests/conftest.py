import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAISS_SRC = ROOT / "faiss-src"


@pytest.fixture(scope="session")
def official_source():
    """The pinned Faiss checkout. Skipped when absent, unless EMV_REQUIRE_OFFICIAL=1 (CI)."""
    if not (FAISS_SRC / "contrib" / "e_means" / "__init__.py").is_file():
        msg = "faiss-src missing: run `bash scripts/setup_faiss.sh`"
        if os.environ.get("EMV_REQUIRE_OFFICIAL") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    return FAISS_SRC
