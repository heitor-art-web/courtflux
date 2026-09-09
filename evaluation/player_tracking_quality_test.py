from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.player_tracking_quality import evaluate_frames


def _player(track_id, box, *, team=None, court_xy=None, court_xy_valid=False, frame_height=100):
    return {
        "track_id": track_id,
        "box": box,
        "team": team,
        "court_xy": court_xy,
        "court_xy_valid": court_xy_valid,
        "frame_height": frame_height,
    }


def _frame(frame_id, players, timing=10.0, accepted=None):
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 30.0,
        "valid_gameplay": True,
        "accepted_player_candidates": len(players) if accepted is None else accepted,
        "team_predictions": players,
        "timing_ms": {"track": timing},
    }


def test_tracking_metrics_and_gt_boundaries():
    frames = [
        _frame(
            1,
            [
                _player(7, [0, 0, 10, 20], team="A", court_xy=[1.0, 1.0], court_xy_valid=True),
                _player(8, [20, 0, 30, 20], team="B", court_xy=[5.0, 1.0], court_xy_valid=True),
            ],
        ),
        _frame(
            2,
            [
                _player(7, [1, 0, 11, 20], team="A", court_xy=[1.2, 1.0], court_xy_valid=True),
                _player(-1, [40, 0, 50, 20]),
            ],
        ),
        _frame(
            4,
            [
                _player(7, [3, 0, 13, 20], team="B", court_xy=[1.6, 1.0], court_xy_valid=True),
                _player(9, [60, 0, 70, 20], team="A", court_xy=[8.0, 1.0], court_xy_valid=True),
            ],
        ),
    ]

    report = evaluate_frames(frames, short_track_frames=1)
    assert report["schema_version"] == 2
    assert report["frames"]["gameplay_frames"] == 3
    assert report["frames"]["analytics_eligible_frames"] == 3
    assert report["tracking"]["unique_track_ids"] == 3
    assert report["tracking"]["untracked_observations"] == 1
    assert report["tracking"]["same_id_interruption_events"] == 1
    assert report["detector_tracker_bridge"]["detected_player_observations"] == 6
    assert report["detector_tracker_bridge"]["tracked_player_observations"] == 5
    assert report["team_stability"]["team_change_events"] == 1
    assert report["court_continuity"]["tracks_with_multiple_valid_court_samples"] == 1
    assert report["court_continuity"]["nonconsecutive_valid_pairs_skipped"] == 1
    assert report["ground_truth_status"]["true_id_switches"] == "requires_player_identity_ground_truth"
    assert report["ground_truth_status"]["true_track_fragmentation"] == "requires_player_identity_ground_truth"


def test_duplicate_box_proxy_is_not_called_true_fragmentation():
    frames = [
        _frame(
            1,
            [
                _player(1, [0, 0, 100, 100], team="A", frame_height=200),
                _player(2, [2, 2, 98, 98], team="A", frame_height=200),
            ],
        )
    ]
    report = evaluate_frames(frames)
    assert report["tracking"]["high_iou_distinct_track_pairs"] == 1
    assert report["ground_truth_status"]["reported_duplicate_boxes"] == "high_iou_distinct_track_proxy_only"


def test_closeup_frames_are_not_analytics_eligible():
    closeup = _frame(
        1,
        [
            _player(1, [0, 0, 100, 70], frame_height=100),
            _player(2, [110, 0, 210, 80], frame_height=100),
        ],
    )
    wide = _frame(
        2,
        [
            _player(1, [0, 0, 50, 25], frame_height=100),
            _player(2, [60, 0, 110, 30], frame_height=100),
            _player(3, [120, 0, 170, 28], frame_height=100),
        ],
    )
    report = evaluate_frames([closeup, wide])
    assert report["frames"]["gameplay_frames"] == 2
    assert report["frames"]["analytics_eligible_frames"] == 1
    assert report["frames"]["analytics_eligible"]["players_per_frame_mean"] == 3.0


def test_all_untracked_frame_and_invalid_court_are_separated():
    frames = [
        _frame(
            1,
            [
                _player(-1, [0, 0, 20, 20], court_xy=[1000.0, 700.0], court_xy_valid=False),
                _player(-1, [30, 0, 50, 20], court_xy=[-800.0, 80.0], court_xy_valid=False),
            ],
            accepted=2,
        )
    ]
    report = evaluate_frames(frames)
    bridge = report["detector_tracker_bridge"]
    assert bridge["frames_all_detections_untracked"] == 1
    assert bridge["tracker_assignment_rate_pct"] == 0.0
    assert report["court_continuity"]["tracks_with_multiple_valid_court_samples"] == 0
    assert report["court_continuity"]["court_step_m_per_consecutive_frame_median"] is None


if __name__ == "__main__":
    test_tracking_metrics_and_gt_boundaries()
    test_duplicate_box_proxy_is_not_called_true_fragmentation()
    test_closeup_frames_are_not_analytics_eligible()
    test_all_untracked_frame_and_invalid_court_are_separated()
    print("player_tracking_quality_test: PASS")
