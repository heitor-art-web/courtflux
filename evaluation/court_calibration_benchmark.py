from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.gameplay_filter import GameplayFilter
from services.vision.court_geometry import estimate_homography, load_keypoint_map

load_dotenv(ROOT / ".env", override=False)


def _extract_keypoints(result: Any) -> list[Any]:
    if isinstance(result, dict):
        predictions = result.get("predictions") or []
        if predictions:
            first = predictions[0]
            if isinstance(first, dict):
                return list(first.get("keypoints") or [])
        return list(result.get("keypoints") or [])
    predictions = getattr(result, "predictions", None) or []
    if predictions:
        return list(getattr(predictions[0], "keypoints", []) or [])
    return list(getattr(result, "keypoints", []) or [])


def _kp_value(kp: Any, name: str, default: Any = None) -> Any:
    if isinstance(kp, dict):
        return kp.get(name, default)
    return getattr(kp, name, default)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    alpha = pos - lo
    value = float(ordered[lo]) * (1.0 - alpha) + float(ordered[hi]) * alpha
    return round(value, 6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark direct basketball-court keypoint homography via Roboflow hosted inference."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--stride", type=int, default=5, help="Evaluate every Nth gameplay frame")
    parser.add_argument("--model-id", default="basketball-court-detection-2/14")
    parser.add_argument("--keypoint-map", default="")
    parser.add_argument("--anchor-conf", type=float, default=0.5)
    parser.add_argument("--max-rmse-px", type=float, default=12.0)
    parser.add_argument("--output", default="evaluation/court_calibration_benchmark")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required for hosted court calibration benchmark")

    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RuntimeError(
            "Install optional dependencies first: python3 -m pip install -r requirements-roboflow.txt"
        ) from exc

    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)
    keypoint_map = load_keypoint_map(args.keypoint_map or None)
    cap = cv2.VideoCapture(str(Path(args.video)))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = max(0, int(round(args.start * fps)))
    end_frame = max(start_frame, int(round(args.end * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    gameplay_filter = GameplayFilter(
        min_court_ratio=0.15,
        min_motion_ratio=0.003,
        max_grace_frames=15,
        processing_scale=0.5,
    )

    sampled = 0
    valid = 0
    gameplay_seen = 0
    anchors: list[float] = []
    inliers: list[float] = []
    rmses: list[float] = []
    rows: list[dict[str, Any]] = []
    frame_id = start_frame
    stride = max(1, int(args.stride))

    while frame_id <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if gameplay_filter.is_gameplay(frame):
            gameplay_seen += 1
            if (gameplay_seen - 1) % stride == 0:
                result = client.infer(frame, model_id=args.model_id)
                keypoints = _extract_keypoints(result)
                src: list[list[float]] = []
                dst: list[tuple[float, float]] = []
                for fallback_idx, kp in enumerate(keypoints):
                    x = _kp_value(kp, "x")
                    y = _kp_value(kp, "y")
                    conf = float(_kp_value(kp, "confidence", 1.0) or 0.0)
                    idx = _kp_value(kp, "id", _kp_value(kp, "class_id", fallback_idx))
                    try:
                        idx = int(idx)
                    except (TypeError, ValueError):
                        idx = fallback_idx
                    if x is not None and y is not None and conf >= args.anchor_conf and idx in keypoint_map:
                        src.append([float(x), float(y)])
                        dst.append(keypoint_map[idx])

                estimate = estimate_homography(src, dst, max_rmse_px=float(args.max_rmse_px))
                diag = estimate.diagnostics()
                sampled += 1
                if estimate.valid:
                    valid += 1
                anchors.append(float(diag.get("anchor_count", 0) or 0))
                inliers.append(float(diag.get("inlier_count", 0) or 0))
                rmse = diag.get("reprojection_rmse_px")
                if rmse is not None:
                    rmses.append(float(rmse))
                rows.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_sec": frame_id / max(fps, 1e-6),
                        **diag,
                    }
                )
        frame_id += 1

    cap.release()
    summary = {
        "scope": "direct hosted court-keypoint homography benchmark",
        "video": args.video,
        "start_sec": args.start,
        "end_sec": args.end,
        "fps": fps,
        "model_id": args.model_id,
        "keypoint_map_size": len(keypoint_map),
        "anchor_confidence": args.anchor_conf,
        "max_reprojection_rmse_px": args.max_rmse_px,
        "gameplay_frames_seen": gameplay_seen,
        "sample_stride_gameplay_frames": stride,
        "sampled_frames": sampled,
        "valid_homography_frames": valid,
        "valid_homography_rate_pct": round(100.0 * valid / max(sampled, 1), 2),
        "anchors": {
            "median": _percentile(anchors, 0.5),
            "p10": _percentile(anchors, 0.1),
            "p90": _percentile(anchors, 0.9),
        },
        "inliers": {
            "median": _percentile(inliers, 0.5),
            "p10": _percentile(inliers, 0.1),
            "p90": _percentile(inliers, 0.9),
        },
        "reprojection_rmse_px": {
            "count": len(rmses),
            "median": _percentile(rmses, 0.5),
            "p90": _percentile(rmses, 0.9),
            "max": None if not rmses else round(max(rmses), 6),
        },
        "note": "Direct sampled calibration only. This does not measure optical-flow propagation or ball tracking quality.",
    }

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"[COURT CALIBRATION BENCHMARK] {output / 'summary.json'}")


if __name__ == "__main__":
    main()
