from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_recovery_error_mining_reporting import (
    _counterfactual_label,
    aggregate_replay_recovery_events,
    attach_replay_events_to_episodes,
)
from services.vision.ball_tracker import TemporalBallTracker


def test_counterfactual_labels() -> None:
    visible = {"visible": True}
    invisible = {"visible": False}
    off_correct = {"selected_correct": True}
    off_wrong = {"selected_correct": False}
    on_correct = {"selected_correct": True}
    on_wrong = {"selected_correct": False}

    assert _counterfactual_label(None, off_correct, on_wrong) == "unlabeled_event"
    assert _counterfactual_label(invisible, off_wrong, on_wrong) == "ball_not_visible"
    assert _counterfactual_label(visible, off_wrong, on_correct) == "recovery_win"
    assert _counterfactual_label(visible, off_correct, on_wrong) == "recovery_loss"
    assert _counterfactual_label(visible, off_correct, on_correct) == "neutral_both_correct"
    assert _counterfactual_label(visible, off_wrong, on_wrong) == "neutral_both_wrong"


def test_episode_trigger_uses_current_replay_recovery() -> None:
    episodes = [
        {
            "episode_id": "E001",
            "start_frame": 100,
            "end_frame": 120,
            "trigger_recovery_event_id": "R009",
            "trigger_recovery_frame": 112,
            "trigger_lag_frames": -12,
            "recovery_event_count": 2,
            "recovery_event_ids": ["R009", "R010"],
            "recovery_event_frames": [112, 118],
            "trigger_features": {"selected_confidence": 0.2},
        }
    ]
    replay_events = [
        {
            "event_id": "RR001",
            "frame_id": 101,
            "selected_confidence": 0.8,
            "selected_gate_ratio": 2.0,
            "selected_size_similarity": 0.9,
            "pre_state": {
                "track_state": "REACQUIRING",
                "last_confidence": 0.5,
                "challenger_streak": 3,
                "challenger_confidence": 0.8,
            },
            "candidate_pool": {"candidate_count": 2},
            "episode_id": None,
        }
    ]

    attach_replay_events_to_episodes(episodes, replay_events)
    episode = episodes[0]
    assert episode["production_trigger_recovery_event_id"] == "R009"
    assert episode["production_trigger_recovery_frame"] == 112
    assert episode["trigger_recovery_event_id"] == "RR001"
    assert episode["trigger_recovery_frame"] == 101
    assert episode["recovery_event_count"] == 1
    assert episode["replay_recovery_event_frames"] == [101]
    assert episode["trigger_features"]["selected_size_similarity"] == 0.9
    assert replay_events[0]["episode_id"] == "E001"


def test_replay_summary_is_separate_from_production() -> None:
    events = [
        {
            "event_id": "RR001",
            "frame_id": 10,
            "truth_available": True,
            "truth_visible": True,
            "selected_correct": True,
            "counterfactual_label": "recovery_win",
            "selected_confidence": 0.8,
            "selected_gate_ratio": 3.0,
            "selected_size_similarity": 0.9,
        },
        {
            "event_id": "RR002",
            "frame_id": 20,
            "truth_available": True,
            "truth_visible": True,
            "selected_correct": False,
            "counterfactual_label": "recovery_loss",
            "selected_confidence": 0.4,
            "selected_gate_ratio": 2.0,
            "selected_size_similarity": 0.3,
        },
        {
            "event_id": "RR003",
            "frame_id": 30,
            "truth_available": False,
            "truth_visible": None,
            "selected_correct": None,
            "counterfactual_label": "unlabeled_event",
            "selected_confidence": 0.6,
            "selected_gate_ratio": 4.0,
            "selected_size_similarity": 0.7,
        },
    ]
    summary = aggregate_replay_recovery_events(events, 5, 10, "state_aware_v3d")
    assert summary["policy"] == "state_aware_v3d"
    assert summary["events_total_replay"] == 3
    assert summary["events_with_sparse_truth"] == 2
    assert summary["events_without_sparse_truth"] == 1
    assert summary["correct_on_visible_labeled_events"] == 1
    assert summary["wrong_on_visible_labeled_events"] == 1
    assert summary["precision_pct_on_visible_labeled_events"] == 50.0
    assert summary["counterfactual_labels"]["recovery_win"] == 1
    assert summary["counterfactual_labels"]["recovery_loss"] == 1


def test_current_tracker_policy_is_v3d() -> None:
    tracker = TemporalBallTracker(reacquire_enabled=True)
    assert tracker.reacquire_policy == "state_aware_v3d"
    assert tracker.reacquire_min_unsupported_frames == 2
    assert tracker.reacquire_confirm_frames == 2
    assert tracker.reacquire_promotion_frames == 3
    assert tracker.reacquire_promotion_min_confidence == 0.85
    assert tracker.reacquire_promotion_min_confidence == tracker.high_confidence_gate_threshold


def main() -> None:
    tests = [
        test_counterfactual_labels,
        test_episode_trigger_uses_current_replay_recovery,
        test_replay_summary_is_separate_from_production,
        test_current_tracker_policy_is_v3d,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] {len(tests)} replay recovery reporting tests")


if __name__ == "__main__":
    main()
