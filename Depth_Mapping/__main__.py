"""Entry point for ``python -m Depth_Mapping_Part_1`` (run from parent directory).

Preferred usage is from inside the project root:

    cd D:\\Collaborate_Projects\\SIH\\Depth_Mapping_Part_1
    python -m depth_mapping --image Input/test_1.jpeg

This file enables the alternative form when the SIH workspace root is the
working directory:

    cd D:\\Collaborate_Projects\\SIH
    python -m Depth_Mapping_Part_1 --image ...

Both forms produce identical results.
"""

import sys
from pathlib import Path

# Ensure the project root is importable so depth_mapping/ is found.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from depth_mapping.depth import main  # noqa: E402

main()
