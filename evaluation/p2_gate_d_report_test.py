from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_gate_d_report import build_gate_d_report


def _state(frame_id, rows):
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 10.0,
        "game_player_predictions": rows,
    }


def test_report_builds_temporal_segments_and_entities() -> None:
    states = [
        _state(
            1,
            [
                {
                    "track_id": 1,
                    "episode_index": 0,
                    "game_player_id": "game_player_001",
                    "appearance_available": True,
                    "association_reason": "new_entity_no_compatible_history",
                    "episode_open_reason": "initial_track_observation",
                }
            ],
        ),
        _state(
            2,
            [
                {
                    "track_id": 1,
                    "episode_index": 0,
                    "game_player_id": "game_player_001",
                    "appearance_available": True,
                    "association_reason": "new_entity_no_compatible_history",
                    "episode_open_reason": "initial_track_observation",
                }
            ],
        ),
        _state(
            3,
            [
                {
                    "track_id": 1,
                    "episode_index": 1,
                    "game_player_id": "game_player_002",
                    "appearance_available": True,
                    "association_reason": "new_entity_strong_visual_separation",
                    "episode_open_reason": "appearance_conflict_raw",
                    "episode_open_appearance_similarity": 0.42,
                    "episode_open_split_similarity": 0.61,
                    "episode_open_candidate_similarity": 0.94,
                    "association_best_score": 0.51,
                    "association_candidate_count": 1,
                }
            ],
        ),
    ]
    report = build_gate_d_report(states, clip="v4")
    assert report["version"] == "courtflux_gate_d_report_v4"
    assert report["tracks"] == 1
    assert report["episodes"] == 2
    assert report["resolved_episodes"] == 2
    assert report["unknown_episodes"] == 0
    assert report["resolved_episode_rate"] == 1.0
    assert report["game_player_entities"] == 2
    assert len(report["segments"]) == 2
    assert report["segments"][0]["frame_start"] == 1
    assert report["segments"][0]["frame_end"] == 2
    assert report["segments"][1]["episode_index"] == 1
    assert report["segments"][1]["episode_open_reason"] == "appearance_conflict_raw"
    assert report["segments"][1]["episode_open_appearance_similarity"] == 0.42
    assert report["segments"][1]["episode_open_split_similarity"] == 0.61
    assert report["segments"][1]["episode_open_candidate_similarity"] == 0.94
    assert report["association_reason_counts"]["new_entity_no_compatible_history"] == 2
    assert report["episode_association_reason_counts"]["new_entity_no_compatible_history"] == 1
    assert report["episode_open_reason_counts"]["appearance_conflict_raw"] == 1


def test_episode_counts_use_final_resolution_once() -> None:
    states = [
        _state(
            1,
            [
                {
                    "track_id": 4,
                    "episode_index": 0,
                    "game_player_id": None,
                    "association_reason": "unknown_weak_best_match",
                    "episode_open_reason": "initial_track_observation",
                }
            ],
        ),
        _state(
            2,
            [
                {
                    "track_id": 4,
                    "episode_index": 0,
                    "game_player_id": "game_player_003",
                    "association_reason": "matched_existing_entity",
                    "episode_open_reason": "initial_track_observation",
                }
            ],
        ),
    ]
    report = build_gate_d_report(states)
    assert report["episodes"] == 1
    assert report["resolved_episodes"] == 1
    assert report["episode_open_reason_counts"]["initial_track_observation"] == 1
    assert report["episode_association_reason_counts"]["matched_existing_entity"] == 1
    assert len(report["segments"]) == 2


def test_report_summarizes_global_visual_shift_once_per_frame() -> None:
    states = [
        _state(
            23,
            [
                {
                    "track_id": 1,
                    "episode_index": 0,
                    "game_player_id": "game_player_001",
                    "global_visual_shift": True,
                    "global_visual_shift_comparable_tracks": 8,
                    "global_visual_shift_conflicting_tracks": 4,
                    "global_visual_shift_conflict_ratio": 0.5,
                },
                {
                    "track_id": 2,
                    "episode_index": 0,
                    "game_player_id": "game_player_002",
                    "global_visual_shift": True,
                    "global_visual_shift_comparable_tracks": 8,
                    "global_visual_shift_conflicting_tracks": 4,
                    "global_visual_shift_conflict_ratio": 0.5,
                },
            ],
        )
    ]
    report = build_gate_d_report(states)
    assert report["global_visual_shift_frames"] == 1
    assert report["global_visual_shifts"] == [
        {
            "frame_id": 23,
            "timestamp_sec": 2.3,
            "comparable_tracks": 8,
            "conflicting_tracks": 4,
            "conflict_ratio": 0.5,
        }
    ]
    assert report["segments"][0]["global_visual_shift_observations"] == 1


def test_report_detects_simultaneous_identity_conflict() -> None:
    states = [
        _state(
            5,
            [
                {"track_id": 1, "episode_index": 0, "game_player_id": "game_player_001"},
                {"track_id": 2, "episode_index": 0, "game_player_id": "game_player_001"},
            ],
        )
    ]
    report = build_gate_d_report(states)
    assert report["invariants"]["simultaneous_same_game_player_conflicts"] == 1


def test_unknown_is_reported_without_creating_entity() -> None:
    states = [
        _state(
            1,
            [
                {
                    "track_id": 3,
                    "episode_index": 0,
                    "game_player_id": None,
                    "association_reason": "unknown_ambiguous_match",
                    "association_best_score": 0.91,
                    "association_second_score": 0.89,
                    "association_candidate_count": 2,
                }
            ],
        )
    ]
    report = build_gate_d_report(states)
    assert report["prediction_observations"] == 1
    assert report["unknown_observations"] == 1
    assert report["episodes"] == 1
    assert report["unknown_episodes"] == 1
    assert report["game_player_entities"] == 0
    assert report["association_reason_counts"]["unknown_ambiguous_match"] == 1
    assert report["episode_association_reason_counts"]["unknown_ambiguous_match"] == 1
    assert report["segments"][0]["association_candidate_count_max"] == 2


def main() -> None:
    test_report_builds_temporal_segments_and_entities()
    test_episode_counts_use_final_resolution_once()
    test_report_summarizes_global_visual_shift_once_per_frame()
    test_report_detects_simultaneous_identity_conflict()
    test_unknown_is_reported_without_creating_entity()
    print("p2_gate_d_report_test: PASS")


if __name__ == "__main__":
    main()
