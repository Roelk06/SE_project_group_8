import sys
from pathlib import Path

# lets the test files (in any sub folder) do "from helpers import ..."
sys.path.insert(0, str(Path(__file__).resolve().parent))
