from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_recovery_error_mining_extended import (
    aggregate_episodes,
    build_divergence_episodes,
    compare_replays,
    event_window_overlap_summary,
    parse_horizons,
)


def record(
    frame_id: int,
    *,
    source: str = "detector_associated",
    selected_index: int | None = 0,
    center: list[float] | None = None,
    correct: bool = True,
    reacquired: bool = False,
    last_center: list[float] | None = None,
    velocity: list[float] | None = None,
) -> dict[str, object]:
    if center is None:
        center = [100.0 + frame_id, 100.0]
    if last_center is None:
        last_center = [100.0 + max(frame_id - 1, 0), 100.0]
    if velocity is None:
        velocity = [1.0, 0.0]
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 25.0,
        "truth_available": True,
        "truth_visible": True,
        "source": source,
        "track_state": "TRACKED",
        "selected_index": selected_index,
        "selected_correct": correct,
        "trajectory_present": source != "missing",
        "reacquired": reacquired,
        "candidate_count": 1,
        "true_candidate_available": True,
        "output_center": center if source != "missing" else None,
        "pre_state": {
            "track_state": "TRACKED",
            "last_center": last_center,
            "last_frame_id": max(frame_id - 1, 0),
            "last_confidence": 0.7,
            "last_area": 400.0,
            "velocity": velocity,
            "challenger_streak": 0,
            "challenger_confidence": None,
        },
    }


def metrics(correct: int, conditioned: int, direct_not_visible: int) -> dict[str, object]:
    return {
        "association": {
            "correct_when_true_candidate_available": conditioned,
            "direct_not_visible": direct_not_visible,
        },
        "trajectory": {"correct_visible": correct},
    }


def test_parse_horizons() -> None:
    assert parse_horizons("10, 1,5,5") == [1, 5, 10]


def test_divergence_episode_attributes_downstream_harm() -> None:
    off = {frame_id: record(frame_id) for frame_id in range(8)}
    on = {frame_id: record(frame_id) for frame_id in range(8)}

    # Recovery is instantaneously correct but resets the trajectory onto a state that
    # causes two later visible GT samples to be wrong.
    on[2] = record(
        2,
        source="detector_reacquired",
        reacquired=True,
        correct=True,
        center=[102.0, 100.0],
    )
    on[3] = record(
        3,
        correct=False,
        center=[300.0, 300.0],
        last_center=[102.0, 100.0],
        velocity=[0.0, 0.0],
    )
    on[4] = record(
        4,
        correct=False,
        center=[305.0, 300.0],
        last_center=[300.0, 300.0],
        velocity=[5.0, 0.0],
    )
    # Frame 5 has converged output but divergent pre-state; frames 6-7 are fully converged.
    on[5] = record(
        5,
        correct=True,
        center=[105.0, 100.0],
        last_center=[305.0, 300.0],
        velocity=[0.0, 0.0],
    )

    truth = {
        frame_id: {
            "frame_id": frame_id,
            "visible": True,
            "point": [100.0 + frame_id, 100.0],
        }
        for frame_id in range(8)
    }
    comparisons = compare_replays(off, on, truth, center_tolerance_px=1.0)

    event = {
        "event_id": "R001",
        "frame_id": 2,
        "selected_confidence": 0.70,
        "selected_gate_ratio": 2.0,
        "selected_size_similarity": 0.8,
        "selected_minus_best_true_confidence": 0.0,
        "candidate_count": 1,
        "production_challenger_streak": 2,
        "production_challenger_confidence": 0.70,
        "v2_pre_state": {"track_state": "DEGRADED", "last_confidence": 0.6},
        "previous_production": {
            "source": "predicted_hold",
            "track_state": "DEGRADED",
        },
        "candidate_pool": {
            "selected_margin_to_next_confidence": None,
            "selected_confidence_x_size_similarity": 0.56,
        },
    }
    episodes, frame_to_episode = build_divergence_episodes(
        comparisons,
        [event],
        off,
        on,
        truth,
        convergence_frames=2,
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["trigger_recovery_frame"] == 2
    assert episode["effect"] == "harmful"
    assert episode["ground_truth_effect"]["net_v2_correct_delta"] == -2
    assert frame_to_episode[2] == episode["episode_id"]
    assert event["episode_id"] == episode["episode_id"]


def test_episode_attribution_matches_global_delta() -> None:
    episodes = [
        {
            "episode_id": "E001",
            "effect": "harmful",
            "duration_frames": 5,
            "materially_divergent_frames": 4,
            "ground_truth_effect": {
                "net_v2_correct_delta": -2,
                "conditioned_correct_delta": -2,
                "direct_not_visible_delta": 1,
            },
            "trigger_recovery_frame": 10,
            "trigger_features": {
                "selected_confidence": 0.4,
                "selected_gate_ratio": 2.2,
                "selected_size_similarity": 0.5,
                "selected_minus_best_true_confidence": -0.1,
                "pre_last_confidence": 0.7,
                "candidate_pool": {
                    "selected_margin_to_next_confidence": -0.05,
                    "selected_confidence_x_size_similarity": 0.2,
                },
            },
        }
    ]
    summary = aggregate_episodes(
        episodes,
        metrics(correct=10, conditioned=10, direct_not_visible=2),
        metrics(correct=8, conditioned=8, direct_not_visible=3),
        total_production_frames=100,
    )
    attribution = summary["attribution"]
    assert attribution["visible_correct_residual"] == 0
    assert attribution["conditioned_correct_residual"] == 0
    assert attribution["direct_not_visible_residual"] == 0


def test_event_window_overlap_is_explicit() -> None:
    events = [{"frame_id": 10}, {"frame_id": 15}, {"frame_id": 40}]
    summary = event_window_overlap_summary(events, before=5, after=10)
    assert summary["overlapping_pairs"] == 1
    assert summary["events_in_any_overlap"] == 2
    assert summary["max_window_multiplicity"] == 2


def main() -> None:
    tests = [
        test_parse_horizons,
        test_divergence_episode_attributes_downstream_harm,
        test_episode_attribution_matches_global_delta,
        test_event_window_overlap_is_explicit,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] {len(tests)} extended recovery-mining tests")


if __name__ == "__main__":
    main()
