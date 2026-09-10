from __future__ import annotations

import numpy as np

from services.vision.ball_temporal_v1 import TemporalBallTrackerV1


FRAME = (1000, 1000, 3)


def ball(x, y, conf=0.5, size=12):
    half = size // 2
    return {
        "box": [x - half, y - half, x + half, y + half],
        "center": (float(x), float(y)),
        "confidence": float(conf),
    }


def main() -> int:
    tracker = TemporalBallTrackerV1(max_prediction_gap=2, velocity_alpha=1.0)

    d0 = tracker.update([ball(100, 100, 0.6)], frame_id=0, frame_shape=FRAME)
    assert d0.source == "detector_seed"
    assert d0.ball and not d0.ball["interpolated"]

    d1 = tracker.update([ball(120, 100, 0.6)], frame_id=1, frame_shape=FRAME)
    assert d1.source == "detector_temporal_association"
    assert d1.velocity_px_per_frame == (20.0, 0.0)

    # A high-confidence far outlier must not steal an active temporal track.
    d2 = tracker.update(
        [ball(900, 900, 0.99), ball(140, 100, 0.35)],
        frame_id=2,
        frame_shape=FRAME,
    )
    assert d2.selected_index == 1
    assert d2.ball and d2.ball["center"] == (140.0, 100.0)

    direct_center = tracker.last_center.copy()
    direct_frame_id = tracker.last_frame_id
    direct_velocity = tracker.velocity.copy()

    # Direct observations are the only state updates. Short predictions bridge a tiny gap.
    d3 = tracker.update([], frame_id=3, frame_shape=FRAME)
    assert d3.source == "predicted_short_gap"
    assert d3.age_since_detection == 1
    assert d3.ball and d3.ball["interpolated"]
    assert tracker.last_frame_id == direct_frame_id
    assert np.array_equal(tracker.last_center, direct_center)
    assert np.array_equal(tracker.velocity, direct_velocity)

    d4 = tracker.update([], frame_id=4, frame_shape=FRAME)
    assert d4.source == "predicted_short_gap"
    assert d4.age_since_detection == 2
    assert tracker.last_frame_id == direct_frame_id
    assert np.array_equal(tracker.last_center, direct_center)
    assert np.array_equal(tracker.velocity, direct_velocity)

    d5 = tracker.update([], frame_id=5, frame_shape=FRAME)
    assert d5.source == "missing"
    assert d5.ball is None
    assert d5.age_since_detection == 3
    assert tracker.last_center is None
    assert tracker.last_frame_id is None
    assert np.array_equal(tracker.velocity, np.zeros(2, dtype=np.float64))

    # Crossing the prediction budget is always an explicit missing boundary, even if a
    # detector candidate is present. Only a later frame may seed a fresh track.
    expiry_tracker = TemporalBallTrackerV1(max_prediction_gap=2, velocity_alpha=1.0)
    expiry_tracker.update([ball(100, 100, 0.6)], frame_id=0, frame_shape=FRAME)
    expiry_tracker.update([ball(120, 100, 0.6)], frame_id=1, frame_shape=FRAME)
    expiry_tracker.update([], frame_id=2, frame_shape=FRAME)
    expiry_tracker.update([], frame_id=3, frame_shape=FRAME)
    expired = expiry_tracker.update([ball(180, 100, 0.9)], frame_id=4, frame_shape=FRAME)
    assert expired.source == "missing"
    assert expired.ball is None
    assert expired.age_since_detection == 3
    assert expiry_tracker.last_center is None
    assert expiry_tracker.last_frame_id is None

    reseed = expiry_tracker.update([ball(200, 100, 0.9)], frame_id=5, frame_shape=FRAME)
    assert reseed.source == "detector_seed"
    assert reseed.age_since_detection == 0
    assert reseed.velocity_px_per_frame == (0.0, 0.0)

    meta = tracker.metadata()
    assert meta["prediction_updates_state"] is False
    assert meta["reacquisition"] is False
    assert meta["trajectory_consistency"] is False

    print("ball_temporal_v1_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
