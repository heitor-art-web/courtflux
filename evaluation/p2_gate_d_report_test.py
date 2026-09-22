from __future__ import annotations

from evaluation.p2_gate_d_report import build_gate_d_report


def _state(frame_id, rows):
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 10.0,
        "game_player_predictions": rows,
    }


def test_report_builds_temporal_segments_and_entities() -> None:
    states = [
        _state(1, [{"track_id": 1, "episode_index": 0, "game_player_id": "game_player_001", "appearance_available": True}]),
        _state(2, [{"track_id": 1, "episode_index": 0, "game_player_id": "game_player_001", "appearance_available": True}]),
        _state(3, [{"track_id": 1, "episode_index": 1, "game_player_id": "game_player_002", "appearance_available": True}]),
    ]
    report = build_gate_d_report(states, clip="v4")
    assert report["tracks"] == 1
    assert report["game_player_entities"] == 2
    assert len(report["segments"]) == 2
    assert report["segments"][0]["frame_start"] == 1
    assert report["segments"][0]["frame_end"] == 2
    assert report["segments"][1]["episode_index"] == 1


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
    states = [_state(1, [{"track_id": 3, "episode_index": 0, "game_player_id": None}])]
    report = build_gate_d_report(states)
    assert report["prediction_observations"] == 1
    assert report["unknown_observations"] == 1
    assert report["game_player_entities"] == 0


def main() -> None:
    test_report_builds_temporal_segments_and_entities()
    test_report_detects_simultaneous_identity_conflict()
    test_unknown_is_reported_without_creating_entity()
    print("p2_gate_d_report_test: PASS")


if __name__ == "__main__":
    main()
