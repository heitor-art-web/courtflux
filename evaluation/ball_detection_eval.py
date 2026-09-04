from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CourtFlux P0.1 direct ball detector evaluation against frame-level ground truth."
    )
    parser.add_argument("--video", required=True, help="Local video path")
    parser.add_argument("--annotations", required=True, help="JSONL annotations created by prepare_ball_ground_truth.py")
    parser.add_argument("--model", default="weights/Model_v0.1.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--capture-conf",
        type=float,
        default=0.01,
        help="Low YOLO confidence used to capture candidates for the confidence sweep.",
    )
    parser.add_argument(
        "--eval-conf",
        type=float,
        default=0.10,
        help="Primary operating threshold reported in detail.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument(
        "--point-margin-px",
        type=float,
        default=4.0,
        help="Extra pixel margin around a predicted box when matching one-click point truth.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to evaluation/ball_eval/<timestamp>.",
    )
    return parser.parse_args()


def is_ball_class(name: str) -> bool:
    value = name.lower()
    return "ball" in value or "basketball" in value or "sports ball" in value


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


def point_center_distance_px(pred: list[float], point: list[float]) -> float:
    px, py = box_center(pred)
    return math.hypot(px - float(point[0]), py - float(point[1]))


def point_matches_box(pred: list[float], point: list[float], base_margin_px: float) -> bool:
    x1, y1, x2, y2 = pred
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    margin = max(float(base_margin_px), 0.25 * max(width, height))
    x, y = float(point[0]), float(point[1])
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def center_error_gt_diag(pred: list[float], gt: list[float]) -> float:
    px, py = box_center(pred)
    gx = (gt[0] + gt[2]) / 2.0
    gy = (gt[1] + gt[3]) / 2.0
    gt_diag = math.hypot(gt[2] - gt[0], gt[3] - gt[1])
    return math.hypot(px - gx, py - gy) / max(gt_diag, 1.0)


def size_bucket(gt_box: list[float] | None, frame_width: int, frame_height: int) -> str:
    if not gt_box:
        return "unspecified"
    area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
    scale = math.sqrt(area) / max(math.hypot(frame_width, frame_height), 1.0)
    if scale < 0.006:
        return "tiny"
    if scale < 0.010:
        return "small"
    if scale < 0.016:
        return "medium"
    return "large"


def load_annotations(path: Path) -> dict[int, dict[str, Any]]:
    annotations: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("visible"), bool):
                continue
            frame_id = int(row["frame_id"])
            if row["visible"]:
                bbox = row.get("bbox")
                point = row.get("point")
                has_bbox = isinstance(bbox, list) and len(bbox) == 4
                has_point = isinstance(point, list) and len(point) == 2
                if not has_bbox and not has_point:
                    raise ValueError(
                        f"Line {line_number}: visible=true requires bbox=[x1,y1,x2,y2] or point=[x,y]"
                    )
                row["bbox"] = [float(value) for value in bbox] if has_bbox else None
                row["point"] = [float(value) for value in point] if has_point else None
            else:
                row["bbox"] = None
                row["point"] = None
            annotations[frame_id] = row
    if not annotations:
        raise ValueError(
            "No completed annotations found. Use ball_point_annotator.py to click the ball or mark it not visible."
        )
    return annotations


def class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def best_prediction_for_truth(
    predictions: list[dict[str, Any]],
    gt_box: list[float] | None,
    gt_point: list[float] | None,
) -> tuple[int | None, float | None]:
    if not predictions:
        return None, None
    if gt_box is not None:
        index, score = max(
            ((idx, iou_xyxy(prediction["box"], gt_box)) for idx, prediction in enumerate(predictions)),
            key=lambda item: item[1],
        )
        return index, score
    if gt_point is not None:
        index, distance = min(
            ((idx, point_center_distance_px(prediction["box"], gt_point)) for idx, prediction in enumerate(predictions)),
            key=lambda item: item[1],
        )
        return index, distance
    return None, None


def prediction_matches_truth(
    prediction: dict[str, Any],
    gt_box: list[float] | None,
    gt_point: list[float] | None,
    iou_threshold: float,
    point_margin_px: float,
) -> bool:
    if gt_box is not None:
        return iou_xyxy(prediction["box"], gt_box) >= iou_threshold
    if gt_point is not None:
        return point_matches_box(prediction["box"], gt_point, point_margin_px)
    return False


def evaluate_at_threshold(
    rows: list[dict[str, Any]],
    threshold: float,
    iou_threshold: float,
    point_margin_px: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    visible_frames = 0
    candidate_frames = 0
    matched_visible_frames = 0
    localization_errors: list[float] = []
    point_errors_px: list[float] = []
    matched_ious: list[float] = []
    point_truth_frames = 0
    bbox_truth_frames = 0

    for row in rows:
        predictions = [p for p in row["predictions"] if p["confidence"] >= threshold]
        if predictions:
            candidate_frames += 1
        visible = bool(row["visible"])
        gt_box = row.get("bbox")
        gt_point = row.get("point")

        if not visible:
            fp += len(predictions)
            continue

        visible_frames += 1
        if gt_box is not None:
            bbox_truth_frames += 1
        elif gt_point is not None:
            point_truth_frames += 1

        if not predictions:
            fn += 1
            continue

        best_index, _ = best_prediction_for_truth(predictions, gt_box, gt_point)
        if best_index is None:
            fn += 1
            fp += len(predictions)
            continue
        best_prediction = predictions[best_index]

        if prediction_matches_truth(best_prediction, gt_box, gt_point, iou_threshold, point_margin_px):
            tp += 1
            fp += max(0, len(predictions) - 1)
            matched_visible_frames += 1
            if gt_box is not None:
                best_iou = iou_xyxy(best_prediction["box"], gt_box)
                matched_ious.append(best_iou)
                localization_errors.append(center_error_gt_diag(best_prediction["box"], gt_box))
            elif gt_point is not None:
                point_errors_px.append(point_center_distance_px(best_prediction["box"], gt_point))
        else:
            fp += len(predictions)
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_frames = len(rows)

    return {
        "threshold": round(float(threshold), 4),
        "bbox_iou_threshold": iou_threshold,
        "point_margin_px": round(float(point_margin_px), 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "annotated_frames": total_frames,
        "visible_frames": visible_frames,
        "point_truth_frames": point_truth_frames,
        "bbox_truth_frames": bbox_truth_frames,
        "candidate_frame_rate": round(100.0 * candidate_frames / max(total_frames, 1), 3),
        "visible_frame_detection_rate": round(100.0 * matched_visible_frames / max(visible_frames, 1), 3),
        "fp_per_annotated_frame": round(fp / max(total_frames, 1), 6),
        "matched_iou_mean": round(sum(matched_ious) / len(matched_ious), 6) if matched_ious else None,
        "center_error_gt_diag_mean": (
            round(sum(localization_errors) / len(localization_errors), 6) if localization_errors else None
        ),
        "point_center_error_px_mean": (
            round(sum(point_errors_px) / len(point_errors_px), 3) if point_errors_px else None
        ),
    }


def group_breakdown(
    rows: list[dict[str, Any]], threshold: float, iou_threshold: float, point_margin_px: float
) -> dict[str, Any]:
    dimensions = ["size_bucket", "occlusion", "court_region", "camera", "event_context"]
    output: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = row.get(dimension)
            if key is None or str(key).strip() == "":
                key = "unspecified"
            groups[str(key)].append(row)
        output[dimension] = {
            key: evaluate_at_threshold(group_rows, threshold, iou_threshold, point_margin_px)
            for key, group_rows in sorted(groups.items())
        }
    return output


def confidence_calibration(
    rows: list[dict[str, Any]], iou_threshold: float, point_margin_px: float
) -> list[dict[str, Any]]:
    labeled_predictions: list[tuple[float, bool]] = []
    for row in rows:
        predictions = sorted(row["predictions"], key=lambda p: p["confidence"], reverse=True)
        visible = bool(row["visible"])
        gt_box = row.get("bbox")
        gt_point = row.get("point")
        matched_index: int | None = None
        if visible and predictions:
            best_index, _ = best_prediction_for_truth(predictions, gt_box, gt_point)
            if best_index is not None and prediction_matches_truth(
                predictions[best_index], gt_box, gt_point, iou_threshold, point_margin_px
            ):
                matched_index = best_index
        for index, prediction in enumerate(predictions):
            labeled_predictions.append((float(prediction["confidence"]), index == matched_index))

    bins: list[dict[str, Any]] = []
    for lower_int in range(0, 10):
        lower = lower_int / 10.0
        upper = (lower_int + 1) / 10.0
        values = [
            item
            for item in labeled_predictions
            if lower <= item[0] < upper or (upper == 1.0 and lower <= item[0] <= 1.0)
        ]
        if not values:
            continue
        positives = sum(1 for _, matched in values if matched)
        bins.append(
            {
                "conf_min": lower,
                "conf_max": upper,
                "predictions": len(values),
                "empirical_precision": round(positives / len(values), 6),
                "mean_confidence": round(sum(conf for conf, _ in values) / len(values), 6),
            }
        )
    return bins


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    annotation_path = Path(args.annotations).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    annotations = load_annotations(annotation_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("evaluation") / "ball_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    model = YOLO(str(model_path))
    rows: list[dict[str, Any]] = []

    for index, frame_id in enumerate(sorted(annotations), start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")

        results = model.predict(
            frame,
            conf=max(0.001, float(args.capture_conf)),
            iou=float(args.nms_iou),
            imgsz=int(args.imgsz),
            max_det=int(args.max_det),
            device=args.device,
            verbose=False,
        )

        predictions: list[dict[str, Any]] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = class_name(model.names, cls_id)
                if not is_ball_class(name):
                    continue
                confidence = float(box.conf[0])
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                predictions.append(
                    {
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": round(confidence, 8),
                        "box": [round(value, 3) for value in xyxy],
                    }
                )

        annotation = annotations[frame_id]
        row = {
            "frame_id": frame_id,
            "timestamp_sec": round(frame_id / fps, 6) if fps > 0 else annotation.get("timestamp_sec"),
            "visible": bool(annotation["visible"]),
            "bbox": annotation.get("bbox"),
            "point": annotation.get("point"),
            "occlusion": annotation.get("occlusion"),
            "court_region": annotation.get("court_region"),
            "camera": annotation.get("camera"),
            "event_context": annotation.get("event_context"),
            "size_bucket": size_bucket(annotation.get("bbox"), frame_width, frame_height),
            "predictions": predictions,
        }
        rows.append(row)
        print(f"[{index}/{len(annotations)}] frame={frame_id} ball_candidates={len(predictions)}", flush=True)

    capture.release()

    eval_conf = max(float(args.capture_conf), float(args.eval_conf))
    point_margin_px = max(0.0, float(args.point_margin_px))
    sweep_thresholds = sorted(
        {
            round(float(args.capture_conf), 3),
            0.02,
            0.03,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        }
    )
    sweep_thresholds = [threshold for threshold in sweep_thresholds if threshold >= float(args.capture_conf)]

    primary_iou25 = evaluate_at_threshold(rows, eval_conf, 0.25, point_margin_px)
    primary_iou50 = evaluate_at_threshold(rows, eval_conf, 0.50, point_margin_px)
    sweep_iou25 = [
        evaluate_at_threshold(rows, threshold, 0.25, point_margin_px) for threshold in sweep_thresholds
    ]
    sweep_iou50 = [
        evaluate_at_threshold(rows, threshold, 0.50, point_margin_px) for threshold in sweep_thresholds
    ]

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "CourtFlux P0.1 direct ball detector evaluation; no tracker or temporal association",
        "video": str(video_path),
        "annotations": str(annotation_path),
        "model": str(model_path),
        "imgsz": int(args.imgsz),
        "device": args.device,
        "capture_conf": float(args.capture_conf),
        "eval_conf": eval_conf,
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "point_margin_px": point_margin_px,
        "annotated_frames": len(rows),
        "visible_ball_frames": sum(1 for row in rows if row["visible"]),
        "annotation_modes": {
            "point": sum(1 for row in rows if row["visible"] and row.get("point") is not None and row.get("bbox") is None),
            "bbox": sum(1 for row in rows if row["visible"] and row.get("bbox") is not None),
            "not_visible": sum(1 for row in rows if not row["visible"]),
        },
        "primary": {
            "iou_0_25_or_point": primary_iou25,
            "iou_0_50_or_point": primary_iou50,
        },
        "threshold_sweep": {
            "iou_0_25_or_point": sweep_iou25,
            "iou_0_50_or_point": sweep_iou50,
        },
        "breakdown_at_eval_conf_iou_0_25_or_point": group_breakdown(
            rows, eval_conf, 0.25, point_margin_px
        ),
        "confidence_calibration_iou_0_25_or_point": confidence_calibration(
            rows, 0.25, point_margin_px
        ),
        "notes": [
            "Point truth is the default fast workflow: a prediction matches when the clicked ball point lies inside the predicted box plus a small margin.",
            "BBox truth remains supported; bbox rows use IoU@0.25 / IoU@0.50.",
            "Point annotations deliberately prioritize fast presence/recall verification over exact ball-size localization.",
            "Do not use candidate-frame coverage as a substitute for ground-truth precision/recall.",
        ],
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "per_frame.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "timestamp_sec",
                "visible",
                "truth_mode",
                "point",
                "size_bucket",
                "occlusion",
                "court_region",
                "camera",
                "event_context",
                "candidate_count",
                "max_confidence",
                "best_iou",
                "best_point_distance_px",
            ],
        )
        writer.writeheader()
        for row in rows:
            gt_box = row.get("bbox")
            gt_point = row.get("point")
            predictions = row["predictions"]
            best_iou = max((iou_xyxy(p["box"], gt_box) for p in predictions), default=0.0) if gt_box else None
            best_point_distance = (
                min((point_center_distance_px(p["box"], gt_point) for p in predictions), default=None)
                if gt_point
                else None
            )
            truth_mode = "bbox" if gt_box is not None else "point" if gt_point is not None else "not_visible"
            writer.writerow(
                {
                    "frame_id": row["frame_id"],
                    "timestamp_sec": row["timestamp_sec"],
                    "visible": row["visible"],
                    "truth_mode": truth_mode,
                    "point": json.dumps(gt_point) if gt_point is not None else None,
                    "size_bucket": row["size_bucket"],
                    "occlusion": row.get("occlusion"),
                    "court_region": row.get("court_region"),
                    "camera": row.get("camera"),
                    "event_context": row.get("event_context"),
                    "candidate_count": len(predictions),
                    "max_confidence": max((p["confidence"] for p in predictions), default=None),
                    "best_iou": round(best_iou, 6) if best_iou is not None else None,
                    "best_point_distance_px": (
                        round(best_point_distance, 3) if best_point_distance is not None else None
                    ),
                }
            )

    print("\nP0.1 direct detector evaluation")
    print(json.dumps(summary["primary"], indent=2))
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
