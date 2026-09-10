from __future__ import annotations

import pandas as pd

from jersey_quality_gate_analysis import (
    _best_for_target,
    _candidate_keep,
    _combined_candidates,
    _float_frame,
    _metrics,
    _single_candidates,
)


def main() -> int:
    frame = pd.DataFrame(
        [
            # Readable crops: generally sharper/larger, with one difficult readable case.
            {"label_kind": "number", "torso_sharpness": 120, "torso_width": 80, "torso_height": 100},
            {"label_kind": "number", "torso_sharpness": 100, "torso_width": 75, "torso_height": 95},
            {"label_kind": "number", "torso_sharpness": 85, "torso_width": 70, "torso_height": 90},
            {"label_kind": "number", "torso_sharpness": 70, "torso_width": 65, "torso_height": 85},
            {"label_kind": "number", "torso_sharpness": 40, "torso_width": 60, "torso_height": 80},
            # Unreadable crops: mostly blur/small size.
            {"label_kind": "unreadable", "torso_sharpness": 35, "torso_width": 55, "torso_height": 70},
            {"label_kind": "unreadable", "torso_sharpness": 25, "torso_width": 50, "torso_height": 65},
            {"label_kind": "unreadable", "torso_sharpness": 18, "torso_width": 45, "torso_height": 60},
            {"label_kind": "unreadable", "torso_sharpness": 12, "torso_width": 40, "torso_height": 55},
            {"label_kind": "unreadable", "torso_sharpness": 8, "torso_width": 35, "torso_height": 50},
        ]
    )
    frame["player_width"] = frame["torso_width"]
    frame["player_height"] = frame["torso_height"]
    frame["player_conf"] = 0.5
    frame["team_score"] = 0.5
    frame["team_margin"] = 0.2
    frame = _float_frame(frame)
    frame["is_readable"] = frame["label_kind"] == "number"

    baseline = _metrics(frame, pd.Series([True] * len(frame)))
    assert baseline["readable_recall"] == 1.0
    assert baseline["player_ocr_call_reduction"] == 0.0

    sharpness = _single_candidates(frame, "torso_sharpness", "sharpness")
    winner_100 = _best_for_target(sharpness, 1.0)
    assert winner_100 is not None
    assert winner_100["calibration"]["readable_recall"] == 1.0
    assert winner_100["calibration"]["unreadable_rejection_rate"] > 0.0

    winner_80 = _best_for_target(sharpness, 0.8)
    assert winner_80 is not None
    assert winner_80["calibration"]["readable_recall"] >= 0.8
    assert winner_80["calibration"]["unreadable_rejection_rate"] >= winner_100["calibration"]["unreadable_rejection_rate"]

    combined = _combined_candidates(frame)
    winner_combined = _best_for_target(combined, 0.8)
    assert winner_combined is not None
    keep = _candidate_keep(frame, winner_combined)
    metrics = _metrics(frame, keep)
    assert metrics["readable_recall"] >= 0.8
    assert metrics["player_ocr_call_reduction"] > 0.0

    print("jersey quality gate analysis smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
