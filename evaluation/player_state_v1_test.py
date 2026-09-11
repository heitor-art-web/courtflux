from __future__ import annotations

from services.vision.player_state_v1 import PlayerStateEstimatorV1


def main() -> None:
    estimator = PlayerStateEstimatorV1(memory_frames=30)

    first = estimator.update(
        [
            {
                "track_id": 7,
                "class_name": "player",
                "box": [10, 20, 30, 60],
                "confidence": 0.9,
                "court_position_m": [4.0, 3.0],
            },
            {
                "track_id": 91,
                "class_name": "referee",
                "box": [100, 20, 120, 60],
                "confidence": 0.95,
            },
            {
                "class_name": "player",
                "box": [200, 20, 220, 60],
                "confidence": 0.8,
            },
            {
                "track_id": -1,
                "class_name": "player",
                "box": [230, 20, 250, 60],
                "confidence": 0.85,
            },
        ],
        frame_id=10,
        timestamp_sec=1.0,
    )
    assert len(first) == 1
    p0 = first[0]
    assert p0.track_id == 7
    assert p0.center_px == (20.0, 40.0)
    assert p0.velocity_px_per_frame == (0.0, 0.0)
    assert p0.court_position_m == (4.0, 3.0)
    assert p0.speed_mps is None
    assert abs(p0.uncertainty - 0.1) < 1e-9
    assert p0.visible is True
    assert p0.provenance["player_referee_separation"] is True
    assert p0.provenance["identity_resolution"] == "deferred"

    diagnostics = estimator.last_diagnostics
    assert diagnostics["player_states"] == 1
    assert diagnostics["referee_excluded"] == 1
    assert diagnostics["player_missing_track_id"] == 2

    second = estimator.update(
        [
            {
                "track_id": 7,
                "class_name": "player",
                "bbox": [20, 20, 40, 60],
                "confidence": 0.8,
                "court_position_m": [6.0, 3.0],
                "source": "botsort_tracked_player",
            }
        ],
        frame_id=11,
        timestamp_sec=2.0,
    )
    assert len(second) == 1
    p1 = second[0]
    assert p1.center_px == (30.0, 40.0)
    assert p1.velocity_px_per_frame == (10.0, 0.0)
    assert abs((p1.speed_mps or 0.0) - 2.0) < 1e-9
    assert p1.source == "botsort_tracked_player"
    assert p1.position() == (6.0, 3.0)

    payload = p1.as_dict()
    assert "has_ball" not in payload
    assert "pressure" not in payload
    assert "team" not in payload
    assert "shirt_number" not in payload

    metadata = estimator.metadata()
    assert metadata["requires_upstream_track_id"] is True
    assert metadata["negative_track_ids_are_untracked"] is True
    assert metadata["tracking_association_owned_upstream"] is True
    assert metadata["referees_are_player_states"] is False
    assert metadata["identity_resolution"] == "deferred_to_identity_consolidation"
    assert metadata["p2_event_semantics"] is False

    print("player_state_v1_test: PASS")


if __name__ == "__main__":
    main()
