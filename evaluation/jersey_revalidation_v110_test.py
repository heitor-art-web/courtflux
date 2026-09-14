from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.jersey_revalidation_v110 import (
    FROZEN_POLICY,
    _evaluate,
    _fragmentation_summary,
    _greedy_match,
    _iou,
    _track_groups,
)


def _row(
    *,
    track_id: int,
    label_kind: str,
    jersey_gt: str,
    pred: int,
    uncertainty: float = 0.2,
    split: str = "train",
    original_group: str = "g1",
) -> dict:
    probs = np.full(100, 1e-6, dtype=float)
    probs[pred] = 0.9
    probs[(pred + 1) % 100] = 0.1
    probs /= probs.sum()
    return {
        "new_track_id": track_id,
        "label_kind": label_kind,
        "jersey_gt": jersey_gt,
        "pred_number": pred,
        "pred_score": float(probs[pred]),
        "uncertainty": uncertainty,
        "absent_argmax": False,
        "_number_probs": probs,
        "split": split,
        "original_group": original_group,
    }


def main() -> int:
    assert FROZEN_POLICY["aggregation"] == "best_uncertainty"
    assert FROZEN_POLICY["min_frames"] == 2

    assert abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    detections = [
        {"box": (0, 0, 10, 10), "track_id": 11},
        {"box": (20, 20, 30, 30), "track_id": 22},
    ]
    matches, unmatched_gt, unmatched_det = _greedy_match(
        [(1, 1, 9, 9), (21, 21, 29, 29)], detections, 0.3
    )
    assert len(matches) == 2
    assert not unmatched_gt
    assert not unmatched_det

    records = [
        _row(track_id=1, label_kind="number", jersey_gt="7", pred=7, original_group="g1"),
        _row(track_id=1, label_kind="number", jersey_gt="7", pred=7, original_group="g1"),
        _row(track_id=2, label_kind="number", jersey_gt="12", pred=12, original_group="g2"),
        _row(track_id=3, label_kind="unreadable", jersey_gt="unreadable", pred=4, original_group="g3"),
        _row(track_id=3, label_kind="unreadable", jersey_gt="unreadable", pred=4, original_group="g3"),
        _row(track_id=4, label_kind="number", jersey_gt="8", pred=8, original_group="g4"),
        _row(track_id=4, label_kind="number", jersey_gt="9", pred=9, original_group="g5"),
    ]
    groups, conflicts = _track_groups(records)
    assert len(conflicts) == 1
    assert conflicts[0]["new_track_id"] == 4

    metrics = _evaluate(groups, "all")
    assert metrics["numeric_tracks"] == 2
    assert metrics["accepted_numeric_tracks"] == 1
    assert metrics["correct"] == 1
    assert metrics["insufficient_policy_frames"] == 1
    assert metrics["unreadable_tracks"] == 1
    assert metrics["accepted_unreadable_tracks"] == 1
    assert abs(metrics["track_coverage"] - 0.5) < 1e-9
    assert abs(metrics["decision_precision"] - 0.5) < 1e-9

    gt = pd.DataFrame(
        [
            {"original_group": "g1"},
            {"original_group": "g1"},
            {"original_group": "g2"},
        ]
    )
    mapped = [
        {"original_group": "g1", "new_track_id": 1},
        {"original_group": "g1", "new_track_id": 5},
        {"original_group": "g2", "new_track_id": 2},
    ]
    fragmentation = _fragmentation_summary(gt, mapped)
    assert fragmentation["original_groups_total"] == 2
    assert fragmentation["groups_mapping_to_multiple_new_tracks"] == 1
    assert fragmentation["groups_mapping_to_one_new_track"] == 1

    print("jersey_revalidation_v110_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
