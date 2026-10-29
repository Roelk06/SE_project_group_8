import sys
from pathlib import Path

import pytest

# lets the test files (including the ones in sub folders) do "from helpers import ..."
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_collection_modifyitems(items):
    """Mark every test as unit or integration based on the folder it is in."""
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
