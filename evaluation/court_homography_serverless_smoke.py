from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.camera_tracker import CameraTracker
from services.vision.court_geometry import project_image_point_to_court

DEFAULT_MAP = (
    ROOT
    / "services"
    / "vision"
    / "court_keypoint_maps"
    / "roboflow_basketball_court_v19_fiba.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one existing-video frame through the Roboflow v19 court keypoint model "
            "and report real homography diagnostics. Does not run ball inference."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--model", default="basketball-court-detection-2/19")
    parser.add_argument("--map", default=str(DEFAULT_MAP))
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--keypoint-confidence", type=float, default=0.50)
    parser.add_argument("--max-rmse-px", type=float, default=12.0)
    parser.add_argument("--api-url", default="https://serverless.roboflow.com")
    parser.add_argument(
        "--output",
        default="evaluation/court_homography_serverless_smoke.json",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.is_absolute():
        video_path = (ROOT / video_path).resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    if not (
        os.getenv("ROBOFLOW_PRIVATE_API_KEY", "").strip()
        or os.getenv("ROBOFLOW_API_KEY", "").strip()
    ):
        raise SystemExit(
            "No Roboflow API key found. Set ROBOFLOW_PRIVATE_API_KEY in the environment/.env."
        )

    os.environ["BASKETBALL_COURT_INFERENCE_BACKEND"] = "serverless"
    os.environ["BASKETBALL_ROBOFLOW_API_URL"] = str(args.api_url)

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(float(args.time), 0.0) * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Could not read {video_path} at {args.time:.3f}s")

    tracker = CameraTracker(
        first_frame=frame,
        model_id=str(args.model),
        confidence=float(args.confidence),
        anchor_confidence=float(args.keypoint_confidence),
        keypoint_map_path=str(Path(args.map).expanduser()),
        max_reprojection_rmse_px=float(args.max_rmse_px),
    )
    diagnostics = dict(tracker.last_calibration_diagnostics)
    H = tracker.initial_H
    homography_is_identity = bool(H is None or np.allclose(H, np.eye(3)))

    height, width = frame.shape[:2]
    image_points = {
        "frame_center": (width / 2.0, height / 2.0),
        "bottom_center": (width / 2.0, float(height - 1)),
    }
    projected = {
        name: (
            None
            if H is None or homography_is_identity
            else project_image_point_to_court(point, H)
        )
        for name, point in image_points.items()
    }

    payload = {
        "video": str(video_path),
        "time_sec": float(args.time),
        "model_id": str(args.model),
        "keypoint_map": str(Path(args.map).expanduser()),
        "backend": "serverless",
        "diagnostics": diagnostics,
        "homography_is_identity": homography_is_identity,
        "sample_projected_court_xy": {
            key: None if value is None else [round(value[0], 4), round(value[1], 4)]
            for key, value in projected.items()
        },
    }

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("COURT HOMOGRAPHY SERVERLESS SMOKE")
    print("=" * 72)
    print(f"model:       {args.model}")
    print(f"source:      {diagnostics.get('source')}")
    print(f"backend:     {diagnostics.get('keypoint_backend')}")
    print(f"anchors:     {diagnostics.get('anchor_count')}")
    print(f"inliers:     {diagnostics.get('inlier_count')}")
    print(f"RMSE px:     {diagnostics.get('reprojection_rmse_px')}")
    print(f"valid:       {diagnostics.get('valid')}")
    if diagnostics.get("error"):
        print(f"error:       {diagnostics.get('error')}")
    print(f"identity H:  {homography_is_identity}")
    print(f"summary:     {output}")

    if not bool(diagnostics.get("valid")) or homography_is_identity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
