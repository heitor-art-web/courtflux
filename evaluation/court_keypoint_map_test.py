from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.court_geometry import load_keypoint_map

MAP_PATH = (
    ROOT
    / "services"
    / "vision"
    / "court_keypoint_maps"
    / "roboflow_basketball_court_v19_fiba.json"
)

EXPECTED_LABELS = [
    "01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "13",
    "14", "15", "16", "17", "19", "21", "23", "25", "26", "27", "28",
    "29", "30", "31", "32", "33", "34", "35", "37", "38", "40", "41",
]


def main() -> None:
    payload = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    source = payload["source"]
    assert source["keypoint_labels"] == EXPECTED_LABELS
    assert source["geometry_source_commit"] == "415751384e34f1cb7089a00909e0d7932fe73a75"
    assert source["status"] == "derived_from_open_upstream_pr"

    points = payload["points"]
    assert len(points) == 33
    assert [point["index"] for point in points] == list(range(33))
    assert [point["label"] for point in points] == EXPECTED_LABELS

    mapping = load_keypoint_map(MAP_PATH)
    assert len(mapping) == 33
    assert mapping[0] == (0.0, 0.0)
    assert mapping[5] == (0.0, 15.0)
    assert mapping[16] == (14.0, 7.5)
    assert mapping[27] == (28.0, 0.0)
    assert mapping[32] == (28.0, 15.0)

    # FIBA paint/free-throw and rim/arc anchors from the upstream geometry.
    assert mapping[2] == (0.0, 5.05)
    assert mapping[9] == (5.8, 5.05)
    assert mapping[13] == (8.32, 7.5)
    assert mapping[19] == (19.68, 7.5)
    assert mapping[23] == (22.2, 9.95)
    assert mapping[26] == (26.43, 7.5)

    print("court_keypoint_map_test: PASS")


if __name__ == "__main__":
    main()
