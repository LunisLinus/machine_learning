from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from carvana.training.engine import main


if __name__ == "__main__":
    main()
