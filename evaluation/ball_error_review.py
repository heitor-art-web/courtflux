from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual CourtFlux P0.1 false-negative / false-positive review frames."
    )
    parser.add_argument("--eval-dir", required=True, help="Directory containing per_frame.json")
    parser.add_argument("--video", required=True, help="Local source video path")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--point-margin-px", type=float, default=4.0)
    parser.add_argument(
        "--max-fp-frames",
        type=int,
        default=80,
        help="Save the highest-confidence false-positive frames; use 0 for all.",
    )
    return parser.parse_args()


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_distance(box: list[float], point: list[float]) -> float:
    cx, cy = box_center(box)
    return math.hypot(cx - float(point[0]), cy - float(point[1]))


def point_matches_box(box: list[float], point: list[float], base_margin_px: float) -> bool:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    margin = max(float(base_margin_px), 0.25 * max(width, height))
    x, y = float(point[0]), float(point[1])
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def choose_match(
    predictions: list[dict[str, Any]],
    gt_box: list[float] | None,
    gt_point: list[float] | None,
    point_margin_px: float,
) -> int | None:
    if not predictions:
        return None
    if gt_box is not None:
        scored = [(iou_xyxy(prediction["box"], gt_box), idx) for idx, prediction in enumerate(predictions)]
        score, idx = max(scored, key=lambda item: item[0])
        return idx if score >= 0.25 else None
    if gt_point is not None:
        matches = [
            (point_distance(prediction["box"], gt_point), idx)
            for idx, prediction in enumerate(predictions)
            if point_matches_box(prediction["box"], gt_point, point_margin_px)
        ]
        if not matches:
            return None
        return min(matches, key=lambda item: item[0])[1]
    return None


def draw_box(frame, box: list[float], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y = max(20, y1 - 7)
    cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def draw_header(frame, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)


def read_frame(capture: cv2.VideoCapture, frame_id: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id}")
    return frame


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    per_frame_path = eval_dir / "per_frame.json"
    if not per_frame_path.exists():
        raise FileNotFoundError(per_frame_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    rows = json.loads(per_frame_path.read_text(encoding="utf-8"))
    threshold = float(args.threshold)
    point_margin_px = max(0.0, float(args.point_margin_px))

    false_negatives: list[dict[str, Any]] = []
    fp_frames: list[dict[str, Any]] = []
    false_positive_detections = 0

    for row in rows:
        predictions = [
            prediction
            for prediction in row.get("predictions", [])
            if float(prediction.get("confidence", 0.0)) >= threshold
        ]
        visible = bool(row.get("visible"))
        gt_box = row.get("bbox")
        gt_point = row.get("point")
        match_index = choose_match(predictions, gt_box, gt_point, point_margin_px) if visible else None

        if visible and match_index is None:
            false_negatives.append({**row, "filtered_predictions": predictions})

        fp_predictions = [
            prediction for idx, prediction in enumerate(predictions) if idx != match_index
        ]
        if fp_predictions:
            false_positive_detections += len(fp_predictions)
            fp_frames.append(
                {
                    **row,
                    "filtered_predictions": predictions,
                    "fp_predictions": fp_predictions,
                    "matched_prediction": predictions[match_index] if match_index is not None else None,
                    "max_fp_confidence": max(float(p["confidence"]) for p in fp_predictions),
                }
            )

    fp_frames.sort(key=lambda row: float(row["max_fp_confidence"]), reverse=True)
    max_fp_frames = int(args.max_fp_frames)
    selected_fp_frames = fp_frames if max_fp_frames <= 0 else fp_frames[:max_fp_frames]

    review_dir = eval_dir / "review"
    fn_dir = review_dir / "false_negatives"
    fp_dir = review_dir / "false_positives"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    manifest_rows: list[dict[str, Any]] = []

    for row in false_negatives:
        frame_id = int(row["frame_id"])
        frame = read_frame(capture, frame_id)
        point = row.get("point")
        if isinstance(point, list) and len(point) == 2:
            cv2.drawMarker(
                frame,
                (int(round(point[0])), int(round(point[1]))),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                28,
                3,
            )
        box = row.get("bbox")
        if isinstance(box, list) and len(box) == 4:
            draw_box(frame, box, "GT BALL", (0, 255, 255))
        for prediction in row.get("filtered_predictions", []):
            draw_box(frame, prediction["box"], f"OTHER {float(prediction['confidence']):.2f}", (0, 0, 255))
        draw_header(
            frame,
            f"FALSE NEGATIVE | frame {frame_id} | t={float(row.get('timestamp_sec', 0.0)):.2f}s | conf>={threshold:.2f}",
        )
        output_path = fn_dir / f"fn_frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        manifest_rows.append(
            {
                "type": "false_negative",
                "frame_id": frame_id,
                "timestamp_sec": row.get("timestamp_sec"),
                "visible": True,
                "fp_count": len(row.get("filtered_predictions", [])),
                "max_fp_confidence": max(
                    (float(p["confidence"]) for p in row.get("filtered_predictions", [])), default=None
                ),
                "image": str(output_path.relative_to(eval_dir)),
            }
        )

    for row in selected_fp_frames:
        frame_id = int(row["frame_id"])
        frame = read_frame(capture, frame_id)
        point = row.get("point")
        if isinstance(point, list) and len(point) == 2:
            cv2.drawMarker(
                frame,
                (int(round(point[0])), int(round(point[1]))),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )
        matched = row.get("matched_prediction")
        if isinstance(matched, dict):
            draw_box(frame, matched["box"], f"TRUE {float(matched['confidence']):.2f}", (0, 200, 0))
        for prediction in row.get("fp_predictions", []):
            draw_box(frame, prediction["box"], f"FP {float(prediction['confidence']):.2f}", (0, 0, 255))
        draw_header(
            frame,
            f"FALSE POSITIVE | frame {frame_id} | t={float(row.get('timestamp_sec', 0.0)):.2f}s | {len(row['fp_predictions'])} FP",
        )
        output_path = fp_dir / f"fp_frame_{frame_id:06d}_max_{float(row['max_fp_confidence']):.3f}.jpg"
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        manifest_rows.append(
            {
                "type": "false_positive",
                "frame_id": frame_id,
                "timestamp_sec": row.get("timestamp_sec"),
                "visible": bool(row.get("visible")),
                "fp_count": len(row.get("fp_predictions", [])),
                "max_fp_confidence": row.get("max_fp_confidence"),
                "image": str(output_path.relative_to(eval_dir)),
            }
        )

    capture.release()

    summary = {
        "threshold": threshold,
        "point_margin_px": point_margin_px,
        "frames_evaluated": len(rows),
        "false_negative_frames": len(false_negatives),
        "false_positive_detections": false_positive_detections,
        "false_positive_frames": len(fp_frames),
        "false_positive_frames_exported": len(selected_fp_frames),
        "review_dir": str(review_dir),
        "instructions": [
            "Review false_negatives first and identify why the real ball was missed.",
            "Then review the highest-confidence false_positives and identify what the detector confuses with the ball.",
            "Use those repeated failure modes to build the next P0.1 training set.",
        ],
    }
    (review_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (review_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "type",
                "frame_id",
                "timestamp_sec",
                "visible",
                "fp_count",
                "max_fp_confidence",
                "image",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(json.dumps(summary, indent=2))
    print(f"\nReview false negatives: {fn_dir}")
    print(f"Review false positives: {fp_dir}")


if __name__ == "__main__":
    main()
