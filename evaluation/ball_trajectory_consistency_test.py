from __future__ import annotations

from services.vision.ball_hybrid_v1 import TrajectoryAwareBallTrackerV1
from services.vision.ball_trajectory_consistency import BallTrajectoryConsistencyV1


FRAME = (1000, 1000, 3)


def ball(x, y, conf=0.5, size=12):
    half = size // 2
    return {
        "box": [x - half, y - half, x + half, y + half],
        "center": (float(x), float(y)),
        "confidence": float(conf),
    }


def main() -> int:
    trajectory = BallTrajectoryConsistencyV1(
        max_distance_norm=0.12,
        max_speed_residual_norm=0.08,
        max_acceleration_norm=0.08,
    )
    tracker = TrajectoryAwareBallTrackerV1(
        trajectory=trajectory,
        max_prediction_gap=2,
        velocity_alpha=1.0,
    )

    seed = tracker.update([ball(100, 100, 0.55)], frame_id=0, frame_shape=FRAME)
    assert seed.source == "detector_seed"

    second = tracker.update([ball(120, 100, 0.55)], frame_id=1, frame_shape=FRAME)
    assert second.source == "detector_trajectory_consistent"

    # The smooth candidate should beat the more confident but trajectory-inconsistent one.
    decision = tracker.update(
        [ball(140, 100, 0.35), ball(200, 170, 0.95)],
        frame_id=2,
        frame_shape=FRAME,
    )
    assert decision.source == "detector_trajectory_consistent"
    assert decision.selected_index == 0
    assert decision.ball is not None
    assert decision.ball["center"] == (140.0, 100.0)
    assert 0.0 <= decision.ball["trajectory_score"] <= 1.0
    assert "trajectory_consistency" in decision.ball

    # A candidate that violates the trajectory gates is not promoted to a direct detection.
    gap = tracker.update([ball(900, 900, 0.99)], frame_id=3, frame_shape=FRAME)
    assert gap.source == "predicted_short_gap"
    assert gap.ball is not None and gap.ball["interpolated"]

    gap2 = tracker.update([ball(900, 900, 0.99)], frame_id=4, frame_shape=FRAME)
    assert gap2.source == "predicted_short_gap"

    missing = tracker.update([ball(900, 900, 0.99)], frame_id=5, frame_shape=FRAME)
    assert missing.source == "missing"
    assert missing.ball is None
    assert missing.age_since_detection == 3
    assert tracker.last_center is None
    assert tracker.last_frame_id is None

    # A smooth candidate after the short-gap budget must also hit an explicit missing
    # boundary rather than being accepted against stale trajectory state.
    expiry_tracker = TrajectoryAwareBallTrackerV1(
        trajectory=trajectory,
        max_prediction_gap=2,
        velocity_alpha=1.0,
    )
    expiry_tracker.update([ball(100, 100, 0.55)], frame_id=0, frame_shape=FRAME)
    expiry_tracker.update([ball(120, 100, 0.55)], frame_id=1, frame_shape=FRAME)
    expiry_tracker.update([], frame_id=2, frame_shape=FRAME)
    expiry_tracker.update([], frame_id=3, frame_shape=FRAME)
    expired = expiry_tracker.update([ball(180, 100, 0.90)], frame_id=4, frame_shape=FRAME)
    assert expired.source == "missing"
    assert expired.ball is None
    assert expired.age_since_detection == 3
    assert expiry_tracker.last_center is None
    assert expiry_tracker.last_frame_id is None

    reseed = expiry_tracker.update([ball(200, 100, 0.90)], frame_id=5, frame_shape=FRAME)
    assert reseed.source == "detector_seed"
    assert reseed.velocity_px_per_frame == (0.0, 0.0)

    # With no detector observation after expiry, the result is missing and state is cleared.
    missing_tracker = TrajectoryAwareBallTrackerV1(
        trajectory=trajectory,
        max_prediction_gap=2,
        velocity_alpha=1.0,
    )
    missing_tracker.update([ball(100, 100, 0.55)], frame_id=0, frame_shape=FRAME)
    missing_tracker.update([ball(120, 100, 0.55)], frame_id=1, frame_shape=FRAME)
    missing_tracker.update([], frame_id=2, frame_shape=FRAME)
    missing_tracker.update([], frame_id=3, frame_shape=FRAME)
    expired_missing = missing_tracker.update([], frame_id=4, frame_shape=FRAME)
    assert expired_missing.source == "missing"
    assert expired_missing.ball is None
    assert expired_missing.age_since_detection == 3
    assert missing_tracker.last_center is None
    assert missing_tracker.last_frame_id is None

    metadata = tracker.metadata()
    assert metadata["trajectory_consistency"] is True
    assert metadata["reacquisition"] is False
    assert metadata["trajectory"]["owns_track_state"] is False

    print("ball_trajectory_consistency_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
