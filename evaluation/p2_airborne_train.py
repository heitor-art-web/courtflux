from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_airborne_train_two_stage import *  # noqa: F401,F403,E402
from evaluation.p2_airborne_train_two_stage import main  # noqa: E402


if __name__ == "__main__":
    main()
