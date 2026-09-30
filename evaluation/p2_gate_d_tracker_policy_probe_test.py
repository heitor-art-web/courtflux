from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.p2_gate_d_tracker_policy_probe import (
    DEFAULT_GEOMETRY,
    _anchors_by_id,
    _derive_probe_pairs,
    _load_json,
    _match_anchor,
)


def main() -> None:
    geometry = _load_json(DEFAULT_GEOMETRY)
    anchors = _anchors_by_id(geometry)
    pairs = _derive_probe_pairs(anchors)

    handoffs = {
        (row["left"], row["right"], row["baseline_track_id"])
        for row in pairs["handoffs"]
    }
    assert handoffs == {
        ("V6-O04", "V6-O06", 61),
        ("V6-O16", "V6-O17", 360),
        ("V6-O17", "V6-O18", 360),
    }, handoffs

    controls = {
        (row["left"], row["right"], row["baseline_track_id"])
        for row in pairs["stable_controls"]
    }
    assert ("V6-O05", "V6-O07", 45) in controls, controls
    # O16/O18 are both A23 on raw track 360, but A7 occupies that same raw track
    # between them. It is therefore not a clean tracker-continuity control.
    assert ("V6-O16", "V6-O18", 360) not in controls, controls

    match = _match_anchor(
        [100.0, 100.0, 200.0, 300.0],
        [
            {"track_id": 1, "box": [0.0, 0.0, 50.0, 50.0], "confidence": 0.9},
            {"track_id": 2, "box": [105.0, 105.0, 198.0, 295.0], "confidence": 0.8},
        ],
        min_iou=0.15,
        max_center_distance_norm=0.10,
    )
    assert match is not None
    assert match["track_id"] == 2
    assert match["iou_to_frozen_anchor"] > 0.8

    print("p2_gate_d_tracker_policy_probe_test: PASS")


if __name__ == "__main__":
    main()
