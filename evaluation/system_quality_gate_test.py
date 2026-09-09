from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_track_coverage_eval import evaluate_pair
from services.vision.ball_tracker import TemporalBallTracker
from services.vision.court_geometry import (
    estimate_homography,
    project_image_point_to_court,
)


def test_homography_exact() -> None:
    src = [[100, 100], [100, 700], [1100, 100], [1100, 700], [600, 400]]
    dst = [[0, 0], [0, 15], [28, 0], [28, 15], [14, 7.5]]
    estimate = estimate_homography(src, dst, max_rmse_px=1.0)
    assert estimate.valid
    assert estimate.inlier_count >= 4
    assert estimate.reprojection_rmse_px is not None
    assert estimate.reprojection_rmse_px <= 1.0
    projected = project_image_point_to_court((600, 400), estimate.matrix)
    assert projected is not None
    assert np.allclose(projected, (14.0, 7.5), atol=0.05)


def test_homography_ransac_rejects_pixel_outlier() -> None:
    court = np.asarray(
        [
            [0.0, 0.0],
            [28.0, 0.0],
            [28.0, 15.0],
            [0.0, 15.0],
            [14.0, 7.5],
            [5.8, 5.05],
            [22.2, 9.95],
        ],
        dtype=np.float32,
    )
    image_corners = np.asarray(
        [[140.0, 100.0], [1100.0, 140.0], [1000.0, 680.0], [180.0, 700.0]],
        dtype=np.float32,
    )
    court_to_image = cv2.getPerspectiveTransform(court[:4], image_corners)
    image = cv2.perspectiveTransform(court.reshape(-1, 1, 2), court_to_image).reshape(-1, 2)

    # One deliberately catastrophic keypoint. A 3 px RANSAC threshold must reject it
    # in image space rather than interpreting the threshold as court metres.
    image[-1] = [20.0, 20.0]
    estimate = estimate_homography(
        image.tolist(),
        court.tolist(),
        ransac_reprojection_px=3.0,
        max_rmse_px=3.0,
    )
    assert estimate.valid
    assert 4 <= estimate.inlier_count < len(court)
    assert estimate.reprojection_rmse_px is not None
    assert estimate.reprojection_rmse_px <= 3.0

    projected = project_image_point_to_court(image[4], estimate.matrix)
    assert projected is not None
    assert np.allclose(projected, court[4], atol=0.05)


def test_homography_rejects_degenerate_geometry() -> None:
    estimate = estimate_homography(
        [[0, 0], [10, 0], [20, 0], [30, 0]],
        [[0, 0], [5, 0], [10, 0], [15, 0]],
        ransac_reprojection_px=3.0,
        max_rmse_px=3.0,
    )
    assert not estimate.valid


def ball_candidate(x: float, y: float, confidence: float) -> dict:
    return {
        "center": (x, y),
        "box": [int(x - 5), int(y - 5), int(x + 5), int(y + 5)],
        "confidence": confidence,
        "interpolated": False,
    }


def test_temporal_association_prefers_plausible_candidate() -> None:
    tracker = TemporalBallTracker(
        max_hold_frames=3,
        reset_after_frames=12,
        reacquire_enabled=False,
    )
    shape = (720, 1280, 3)

    first = tracker.select([ball_candidate(100, 100, 0.8)], 0, shape)
    assert first.source == "detector_associated"

    # A very high-confidence false candidate must not beat an in-gate lower-confidence
    # candidate. This is the core P0.2 safety property.
    decision = tracker.select(
        [
            ball_candidate(105, 100, 0.4),
            ball_candidate(700, 500, 0.99),
        ],
        1,
        shape,
    )
    assert decision.source == "detector_associated"
    assert decision.ball is not None
    assert np.allclose(decision.ball["center"], (105, 100), atol=0.01)
    assert decision.rejected_candidates >= 1


def test_predicted_hold_is_short_and_non_recursive() -> None:
    tracker = TemporalBallTracker(
        max_hold_frames=3,
        reset_after_frames=12,
        reacquire_enabled=False,
    )
    shape = (720, 1280, 3)
    tracker.select([ball_candidate(100, 100, 0.8)], 0, shape)
    tracker.select([ball_candidate(110, 100, 0.8)], 1, shape)

    direct_frame_id = tracker.last_frame_id
    hold1 = tracker.select([], 2, shape)
    hold2 = tracker.select([], 3, shape)
    assert hold1.source == "predicted_hold"
    assert hold2.source == "predicted_hold"
    assert hold1.hold_age_frames == 1
    assert hold2.hold_age_frames == 2

    # Predicted observations must never become new direct anchors; otherwise drift
    # recursively extends itself and P0.3 stops being a bounded interpolation/hold.
    assert tracker.last_frame_id == direct_frame_id == 1

    expired = tracker.select([], 5, shape)
    assert expired.source == "missing"
    assert expired.ball is None


def test_reacquisition_is_off_in_frozen_baseline() -> None:
    tracker = TemporalBallTracker(
        max_hold_frames=2,
        reset_after_frames=12,
        reacquire_enabled=False,
    )
    shape = (720, 1280, 3)
    tracker.select([ball_candidate(100, 100, 0.8)], 0, shape)

    # An implausible alternative outside the normal gate cannot be promoted while the
    # frozen baseline keeps hard reacquisition disabled.
    decision = tracker.select([ball_candidate(800, 500, 0.99)], 3, shape)
    assert decision.source == "missing"
    assert decision.ball is None
    assert decision.reacquired is False


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_track_coverage_bridge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = root / "frames.jsonl"
        gt = root / "annotations.jsonl"
        production_rows = []
        truth_rows = []
        for frame_id in range(5):
            timestamp = frame_id / 30.0
            truth_rows.append(
                {
                    "timestamp_sec": timestamp,
                    "visible": True,
                    "point": [100 + 10 * frame_id, 100],
                }
            )
            candidates = []
            if frame_id in {0, 4}:
                candidates = [
                    {
                        "center": [100 + 10 * frame_id, 100],
                        "box": [
                            95 + 10 * frame_id,
                            95,
                            105 + 10 * frame_id,
                            105,
                        ],
                        "confidence": 0.9,
                    }
                ]
            production_rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp,
                    "valid_gameplay": True,
                    "ball_association": {"candidates": candidates},
                }
            )
        write_jsonl(frames, production_rows)
        write_jsonl(gt, truth_rows)
        result = evaluate_pair(
            frames,
            gt,
            radius_px=40.0,
            gap_limits=[2, 3],
            timestamp_tolerance_sec=0.001,
        )
        assert result["coverage"]["direct"]["correct_frames"] == 2
        assert result["coverage"]["direct_plus_linear_gap_2"]["correct_frames"] == 2
        assert result["coverage"]["direct_plus_linear_gap_3"]["correct_frames"] == 5
        assert result["coverage"]["direct_plus_linear_gap_3"]["bridged_frames"] == 3


def main() -> None:
    test_homography_exact()
    test_homography_ransac_rejects_pixel_outlier()
    test_homography_rejects_degenerate_geometry()
    test_temporal_association_prefers_plausible_candidate()
    test_predicted_hold_is_short_and_non_recursive()
    test_reacquisition_is_off_in_frozen_baseline()
    test_track_coverage_bridge()
    print("[PASS] system quality gate")


if __name__ == "__main__":
    main()
