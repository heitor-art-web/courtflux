from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.p2_gate_d_detection_association_probe import (
    AssocPolicy,
    DetectionAssociator,
)


def det(team: str | None, x: float = 100.0) -> dict:
    return {
        "box": [x, 100.0, x + 50.0, 200.0],
        "confidence": 0.9,
        "team": team,
        "embedding": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    }


def main() -> None:
    # Hungarian must choose the globally minimal 1:1 assignment.
    cost = np.asarray([[0.20, 0.10, 9.0, 9.0], [0.11, 0.40, 9.0, 9.0]], dtype=np.float64)
    assignment = DetectionAssociator._hungarian(cost)
    assert assignment[:2] == [1, 0], assignment

    # Frozen team evidence is a hard cannot-link only after a short accepted history.
    team_policy = AssocPolicy("motion_team_test", velocity=True, team_gate=True, torso=False)
    tracker = DetectionAssociator(team_policy)
    first_id = None
    for _ in range(3):
        rows = tracker.update([det("team_a")])
        assert len(rows) == 1
        first_id = rows[0]["track_id"]
    assert first_id is not None
    assert rows[0]["locked_team"] == "team_a", rows

    switched = tracker.update([det("team_b")])
    assert len(switched) == 1
    assert switched[0]["track_id"] != first_id, switched

    # Without team gating, identical geometry remains the same anonymous track.
    motion_policy = AssocPolicy("motion_test", velocity=True, team_gate=False, torso=False)
    tracker2 = DetectionAssociator(motion_policy)
    for _ in range(3):
        rows2 = tracker2.update([det("team_a")])
    second_id = rows2[0]["track_id"]
    continued = tracker2.update([det("team_b")])
    assert continued[0]["track_id"] == second_id, continued

    print("p2_gate_d_detection_association_probe_test: PASS")


if __name__ == "__main__":
    main()
