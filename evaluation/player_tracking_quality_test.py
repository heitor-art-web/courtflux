from __future__ import annotations

from evaluation.player_tracking_quality import evaluate_frames


def _frame(frame_id, players, timing=10.0):
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 30.0,
        "valid_gameplay": True,
        "team_predictions": players,
        "timing_ms": {"track": timing},
    }


def test_tracking_metrics_and_gt_boundaries():
    frames = [
        _frame(
            1,
            [
                {"track_id": 7, "box": [0, 0, 10, 20], "court_xy": [1.0, 1.0], "team": "A"},
                {"track_id": 8, "box": [20, 0, 30, 20], "court_xy": [5.0, 1.0], "team": "B"},
            ],
        ),
        _frame(
            2,
            [
                {"track_id": 7, "box": [1, 0, 11, 20], "court_xy": [1.2, 1.0], "team": "A"},
                {"track_id": -1, "box": [40, 0, 50, 20], "court_xy": None, "team": None},
            ],
        ),
        _frame(
            4,
            [
                {"track_id": 7, "box": [3, 0, 13, 20], "court_xy": [1.6, 1.0], "team": "B"},
                {"track_id": 9, "box": [60, 0, 70, 20], "court_xy": [8.0, 1.0], "team": "A"},
            ],
        ),
    ]

    report = evaluate_frames(frames, short_track_frames=1)
    assert report["frames"]["gameplay_frames"] == 3
    assert report["tracking"]["unique_track_ids"] == 3
    assert report["tracking"]["untracked_observations"] == 1
    assert report["tracking"]["same_id_interruption_events"] == 1
    assert report["team_stability"]["team_change_events"] == 1
    assert report["court_continuity"]["tracks_with_multiple_court_samples"] == 1
    assert report["ground_truth_status"]["true_id_switches"] == "requires_player_identity_ground_truth"
    assert report["ground_truth_status"]["true_track_fragmentation"] == "requires_player_identity_ground_truth"


def test_duplicate_box_proxy_is_not_called_true_fragmentation():
    frames = [
        _frame(
            1,
            [
                {"track_id": 1, "box": [0, 0, 100, 100], "team": "A"},
                {"track_id": 2, "box": [2, 2, 98, 98], "team": "A"},
            ],
        )
    ]
    report = evaluate_frames(frames)
    assert report["tracking"]["high_iou_distinct_track_pairs"] == 1
    assert report["ground_truth_status"]["reported_duplicate_boxes"] == "high_iou_distinct_track_proxy_only"


if __name__ == "__main__":
    test_tracking_metrics_and_gt_boundaries()
    test_duplicate_box_proxy_is_not_called_true_fragmentation()
    print("player_tracking_quality_test: PASS")
