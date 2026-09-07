from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import (
    align_production_truth,
    candidate_matches_truth,
    crop_box_around_point,
    nearest_distance_bucket,
)


def test_candidate_matches_point_truth_uses_official_radius() -> None:
    truth = {"visible": True, "point": [100.0, 100.0], "bbox": None}
    inside = {"center": [130.0, 120.0], "box": [120, 110, 140, 130], "confidence": 0.8}
    outside = {"center": [141.0, 100.0], "box": [131, 90, 151, 110], "confidence": 0.8}
    assert candidate_matches_truth(inside, truth, 40.0, 0.25)[0] is True
    assert candidate_matches_truth(outside, truth, 40.0, 0.25)[0] is False


def test_alignment_uses_timestamp_not_frame_id() -> None:
    production = [
        {"frame_id": 0, "timestamp_sec": 10.0, "valid_gameplay": True},
        {"frame_id": 1, "timestamp_sec": 10.04, "valid_gameplay": True},
    ]
    truth = [
        {"frame_id": 250, "timestamp_sec": 10.0, "visible": True, "point": [1, 1]},
        {"frame_id": 251, "timestamp_sec": 10.04, "visible": False, "point": None},
    ]
    aligned, unmatched = align_production_truth(production, truth, 0.001)
    assert unmatched == 0
    assert len(aligned) == 2
    assert aligned[0][1]["frame_id"] == 250
    assert aligned[1][1]["frame_id"] == 251


def test_nearest_distance_buckets() -> None:
    assert nearest_distance_bucket(None, 0, 40.0) == "no_candidate"
    assert nearest_distance_bucket(55.0, 1, 40.0) == "gt_radius_to_80px"
    assert nearest_distance_bucket(120.0, 1, 40.0) == "80_to_160px"
    assert nearest_distance_bucket(300.0, 1, 40.0) == "over_160px"


def test_crop_box_preserves_requested_size_near_edges() -> None:
    assert crop_box_around_point(1920, 1080, [10, 10], 640) == (0, 0, 640, 640)
    assert crop_box_around_point(1920, 1080, [1910, 1070], 640) == (1280, 440, 1920, 1080)


def main() -> None:
    test_candidate_matches_point_truth_uses_official_radius()
    test_alignment_uses_timestamp_not_frame_id()
    test_nearest_distance_buckets()
    test_crop_box_preserves_requested_size_near_edges()
    print("[PASS] P0.1 production error mining helpers")


if __name__ == "__main__":
    main()
