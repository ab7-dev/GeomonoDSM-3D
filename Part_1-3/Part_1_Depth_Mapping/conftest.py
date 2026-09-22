"""pytest configuration for the Depth Mapping Part 1 project.

This file ensures the project root is on sys.path so that
``depth_mapping`` (the package directory inside this folder) is importable
regardless of how pytest is invoked (from the project root, the SIH workspace
root, or via ``python -m pytest``).
"""

import sys
from pathlib import Path

# Project root = directory containing this conftest.py
# Inserting it at position 0 guarantees our depth_mapping/ package is found
# before any system-level package of the same name.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
