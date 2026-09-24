from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_gate_d_real_video_probe import (
    SUPPORTED_CLIPS,
    VALIDATION_CLIPS,
    _clip_defaults,
    _clip_status,
)


def main() -> None:
    assert "v6" in SUPPORTED_CLIPS
    assert VALIDATION_CLIPS == ("v6",)
    assert "v7" not in SUPPORTED_CLIPS

    defaults = _clip_defaults("v6")
    assert defaults["video"] == "https://www.youtube.com/watch?v=VLZugJeVOSU"
    assert defaults["start"] == 720.0
    assert defaults["end"] == 908.0
    assert defaults["match_config"] == "evaluation/match_configs/unknown.json"
    assert _clip_status("v6") == "validation"
    assert _clip_status("v2") == "challenge_non_blocking"
    assert _clip_status("v3") == "development"

    print("p2_gate_d_v6_probe_test: PASS")


if __name__ == "__main__":
    main()
