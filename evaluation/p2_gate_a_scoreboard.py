from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_gate_a_scoreboard_v3 import *  # noqa: F401,F403,E402
from evaluation.p2_gate_a_scoreboard_v3 import (  # noqa: E402
    _aggregate,
    _gate_checks,
    _match_intervals,
    _predicted_intervals,
    main,
)


if __name__ == "__main__":
    main()
