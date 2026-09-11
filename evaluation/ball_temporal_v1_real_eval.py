from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from services.vision.ball_temporal_v1 import TemporalBallTrackerV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TemporalBallTrackerV1 against the direct detector on real annotated video."
    )
    parser.add_argument("--video", required=True, help="Local video path")
    parser.add_argument(
        "--annotations",
        required=True,
        help="JSONL frame-level ball GT produced by prepare_ball_ground_truth.py / ball_point_annotator.py",
    )
    parser.add_argument("--model", default="weights/model_v1.0.0.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.03,
        help="Detector threshold supplied to both the direct baseline and temporal tracker.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--point-margin-px", type=float, default=4.0)
    parser.add_argument("--max-prediction-gap", type=int, default=4)
    parser.add_argument("--association-gate-norm", type=float, default=0.12)
    parser.add_argument("--gate-growth-per-frame", type=float, default=0.02)
    parser.add_argument("--max-gate-norm", type=float, default=0.24)
    parser.add_argument("--velocity-alpha", type=float, default=0.45)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory; defaults to evaluation/ball_temporal_v1_real/<timestamp>.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
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
                row["bbox"] = [float(v) for v in bbox] if has_bbox else None
                row["point"] = [float(v) for v in point] if has_point else None
            else:
                row["bbox"] = None
                row["point"] = None
            rows[frame_id] = row
    if not rows:
        raise ValueError("No completed ball annotations found")
    return rows


def class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def is_ball_class(name: str) -> bool:
    value = name.lower()
    return "ball" in value or "basketball" in value or "sports ball" in value


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def point_matches_box(box: list[float], point: list[float], base_margin_px: float) -> bool:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    margin = max(float(base_margin_px), 0.25 * max(width, height))
    x, y = float(point[0]), float(point[1])
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def matches_truth(prediction: dict[str, Any] | None, annotation: dict[str, Any], point_margin_px: float) -> bool:
    if prediction is None or not annotation["visible"]:
        return False
    box = [float(v) for v in prediction["box"]]
    if annotation.get("bbox") is not None:
        return iou_xyxy(box, annotation["bbox"]) >= 0.25
    if annotation.get("point") is not None:
        return point_matches_box(box, annotation["point"], point_margin_px)
    return False


def point_error(prediction: dict[str, Any] | None, annotation: dict[str, Any]) -> float | None:
    if prediction is None or annotation.get("point") is None:
        return None
    cx, cy = box_center([float(v) for v in prediction["box"]])
    px, py = annotation["point"]
    return math.hypot(cx - float(px), cy - float(py))


def extract_ball_candidates(model: YOLO, frame: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    results = model.predict(
        frame,
        conf=max(0.001, float(args.ball_conf)),
        iou=float(args.nms_iou),
        imgsz=int(args.imgsz),
        max_det=int(args.max_det),
        device=args.device,
        verbose=False,
    )
    candidates: list[dict[str, Any]] = []
    if not results or results[0].boxes is None:
        return candidates
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = class_name(model.names, cls_id)
        if not is_ball_class(name):
            continue
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        candidates.append(
            {
                "class_id": cls_id,
                "class_name": name,
                "confidence": float(box.conf[0]),
                "box": xyxy,
                "center": box_center(xyxy),
            }
        )
    return candidates


def summarize(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    annotated = len(records)
    visible = sum(1 for row in records if row["visible"])
    invisible = annotated - visible
    output_frames = sum(1 for row in records if row[key] is not None)
    correct_visible = sum(1 for row in records if row["visible"] and row[f"{key}_correct"])
    false_positive_invisible = sum(1 for row in records if not row["visible"] and row[key] is not None)
    wrong_visible = sum(1 for row in records if row["visible"] and row[key] is not None and not row[f"{key}_correct"])
    precision_den = correct_visible + wrong_visible + false_positive_invisible
    errors = [row[f"{key}_point_error_px"] for row in records if row[f"{key}_point_error_px"] is not None and row[f"{key}_correct"]]
    return {
        "annotated_frames": annotated,
        "visible_frames": visible,
        "invisible_frames": invisible,
        "output_frames": output_frames,
        "output_coverage": output_frames / annotated if annotated else 0.0,
        "correct_visible_frames": correct_visible,
        "visible_recall": correct_visible / visible if visible else 0.0,
        "wrong_visible_outputs": wrong_visible,
        "false_positive_invisible_frames": false_positive_invisible,
        "decision_precision": correct_visible / precision_den if precision_den else None,
        "mean_point_error_px_on_correct": sum(errors) / len(errors) if errors else None,
    }


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not annotations_path.exists():
        raise FileNotFoundError(annotations_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    annotations = load_annotations(annotations_path)
    first_frame, last_frame = min(annotations), max(annotations)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("evaluation") / "ball_temporal_v1_real" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    model = YOLO(str(model_path))
    tracker = TemporalBallTrackerV1(
        max_prediction_gap=args.max_prediction_gap,
        association_gate_norm=args.association_gate_norm,
        gate_growth_per_frame=args.gate_growth_per_frame,
        max_gate_norm=args.max_gate_norm,
        velocity_alpha=args.velocity_alpha,
    )

    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    processed = 0
    total_frames = last_frame - first_frame + 1

    for frame_id in range(first_frame, last_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")
        candidates = extract_ball_candidates(model, frame, args)
        direct = max(candidates, key=lambda item: item["confidence"]) if candidates else None
        decision = tracker.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
        temporal = decision.ball
        source_counts[decision.source] += 1
        processed += 1

        if frame_id in annotations:
            annotation = annotations[frame_id]
            direct_correct = matches_truth(direct, annotation, args.point_margin_px)
            temporal_correct = matches_truth(temporal, annotation, args.point_margin_px)
            records.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": frame_id / fps if fps > 0 else annotation.get("timestamp_sec"),
                    "visible": bool(annotation["visible"]),
                    "candidate_count": len(candidates),
                    "direct": direct,
                    "direct_correct": direct_correct,
                    "direct_point_error_px": point_error(direct, annotation),
                    "temporal": temporal,
                    "temporal_source": decision.source,
                    "temporal_correct": temporal_correct,
                    "temporal_point_error_px": point_error(temporal, annotation),
                    "age_since_detection": decision.age_since_detection,
                    "distance_norm": decision.distance_norm,
                    "velocity_px_per_frame": decision.velocity_px_per_frame,
                }
            )

        if processed == 1 or processed % 100 == 0 or frame_id == last_frame:
            print(
                f"[{processed}/{total_frames}] {100.0 * processed / max(total_frames, 1):.1f}% "
                f"frame={frame_id} candidates={len(candidates)} source={decision.source}",
                flush=True,
            )

    capture.release()

    direct_summary = summarize(records, "direct")
    temporal_summary = summarize(records, "temporal")
    bridged_annotated = [row for row in records if row["temporal_source"] == "predicted_short_gap"]
    bridged_visible = [row for row in bridged_annotated if row["visible"]]
    bridged_correct = sum(1 for row in bridged_visible if row["temporal_correct"])
    recovered = sum(
        1
        for row in records
        if row["visible"] and not row["direct_correct"] and row["temporal_correct"]
    )
    regressions = sum(
        1
        for row in records
        if row["visible"] and row["direct_correct"] and not row["temporal_correct"]
    )
    tracker_overrides = sum(
        1
        for row in records
        if row["direct"] is not None
        and row["temporal"] is not None
        and not row["temporal"].get("interpolated", False)
        and row["direct"]["box"] != row["temporal"]["box"]
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 TemporalBallTrackerV1 real-video comparison against highest-confidence direct ball detection",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "model": str(model_path),
        "device": args.device,
        "imgsz": args.imgsz,
        "ball_conf": args.ball_conf,
        "frame_range": [first_frame, last_frame],
        "processed_frames": processed,
        "annotated_frames": len(records),
        "tracker_metadata": tracker.metadata(),
        "source_counts_all_processed_frames": dict(source_counts),
        "direct_baseline": direct_summary,
        "temporal_tracker": temporal_summary,
        "delta": {
            "visible_recall": temporal_summary["visible_recall"] - direct_summary["visible_recall"],
            "output_coverage": temporal_summary["output_coverage"] - direct_summary["output_coverage"],
            "recovered_visible_frames": recovered,
            "regressed_visible_frames": regressions,
            "tracker_direct_candidate_overrides_on_annotated_frames": tracker_overrides,
        },
        "short_gap_predictions_on_annotated_frames": {
            "count": len(bridged_annotated),
            "visible_count": len(bridged_visible),
            "correct_visible_count": bridged_correct,
            "visible_precision": bridged_correct / len(bridged_visible) if bridged_visible else None,
        },
        "notes": [
            "All frames between the first and last GT annotations are processed so temporal state is continuous.",
            "Metrics are computed only on annotated frames.",
            "Direct baseline selects the highest-confidence ball candidate at the same ball_conf supplied to the tracker.",
            "Point truth uses the same box-plus-margin contract as ball_detection_eval.py; bbox truth uses IoU >= 0.25.",
            "A short-gap prediction is useful only if it improves visible recall without unacceptable wrong/false outputs.",
            "This runner evaluates TemporalBallTrackerV1 only; trajectory-consistency scoring belongs to PR #39.",
        ],
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "annotated_frames.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
