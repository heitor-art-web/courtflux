from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_track_coverage_eval import evaluate_pair
from services.vision.court_geometry import estimate_homography, project_image_point_to_court


def test_homography() -> None:
    src = [[100, 100], [100, 700], [1100, 100], [1100, 700], [600, 400]]
    dst = [[0, 0], [0, 15], [28, 0], [28, 15], [14, 7.5]]
    estimate = estimate_homography(src, dst, max_rmse_px=1.0)
    assert estimate.valid
    assert estimate.inlier_count >= 4
    projected = project_image_point_to_court((600, 400), estimate.matrix)
    assert projected is not None
    assert np.allclose(projected, (14.0, 7.5), atol=0.05)


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
            truth_rows.append({"timestamp_sec": timestamp, "visible": True, "point": [100 + 10 * frame_id, 100]})
            candidates = []
            if frame_id in {0, 4}:
                candidates = [{"center": [100 + 10 * frame_id, 100], "box": [95 + 10 * frame_id, 95, 105 + 10 * frame_id, 105], "confidence": 0.9}]
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
    test_homography()
    test_track_coverage_bridge()
    print("[PASS] system quality gate")


if __name__ == "__main__":
    main()
