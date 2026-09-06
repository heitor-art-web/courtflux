from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Backward-compatible public surface: existing imports/tests can keep using this module.
from evaluation.ball_recovery_error_mining_base import *  # noqa: F401,F403,E402
from evaluation.ball_recovery_error_mining_extended import main  # noqa: E402


if __name__ == "__main__":
    main()
