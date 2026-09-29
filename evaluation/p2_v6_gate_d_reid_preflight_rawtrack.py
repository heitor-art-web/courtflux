from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_v6_gate_d_reid_preflight as base


def _raw_track_prediction(row: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    """Return raw-track geometry without depending on Gate-D identity output.

    team_predictions is produced upstream of the game-player resolver and remains available
    during gameplay_limited, where game_player_predictions may intentionally be empty.
    The formal human mapping is to a raw track, so this is the correct crop source for the
    ReID preflight. game_player_predictions is retained only as a geometry fallback.
    """

    for source_name in ("team_predictions", "game_player_predictions"):
        predictions = row.get(source_name) or []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            try:
                candidate_track = int(prediction.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if candidate_track != int(track_id):
                continue
            box = prediction.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            result = dict(prediction)
            result["reid_crop_source"] = source_name
            return result
    return None


# Patch only the crop lookup used by the diagnostic preflight. The embedding/scoring logic
# remains exactly the pinned implementation in p2_v6_gate_d_reid_preflight.py.
base._prediction_for_track = _raw_track_prediction


if __name__ == "__main__":
    base.main()
