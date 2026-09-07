from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CourtFlux P0.1 false-negative / false-positive diagnostics. "
            "Supports the legacy per_frame.json workflow and the production "
            "frames.jsonl + one-click annotations workflow."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--eval-dir",
        help="Legacy directory containing per_frame.json from ball_detection_eval.py.",
    )
    source.add_argument(
        "--frames",
        help="Production frames.jsonl to compare directly against --annotations.",
    )
    source.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help=(
            "Production run id. Repeat for batch analysis. Paths default to "
            "evaluation/runs/<id>/frames.jsonl and "
            "evaluation/ball_ground_truth/<id>/annotations.jsonl."
        ),
    )
    parser.add_argument(
        "--annotations",
        help="Ground-truth annotations.jsonl for --frames mode.",
    )
    parser.add_argument(
        "--video",
        help=(
            "Local source video. Required for legacy image export. Optional in production "
            "mode when it can be discovered from ground-truth metadata or run summary."
        ),
    )
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Production batch defaults to evaluation/p0_1_error_review; "
            "single production run defaults beside frames.jsonl; legacy defaults to <eval-dir>/review."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--gt-radius-px",
        type=float,
        default=40.0,
        help="Official point-truth center radius for production P0.1 matching.",
    )
    parser.add_argument("--bbox-iou", type=float, default=0.25)
    parser.add_argument(
        "--timestamp-tolerance-sec",
        type=float,
        default=0.001,
        help="Maximum timestamp delta used to align production rows with GT.",
    )
    parser.add_argument(
        "--point-margin-px",
        type=float,
        default=4.0,
        help="Legacy point-in-box margin; ignored by production 40px center matching.",
    )
    parser.add_argument(
        "--max-fn-frames",
        type=int,
        default=200,
        help="Maximum false-negative review images per run; 0 exports all. Metrics always use all frames.",
    )
    parser.add_argument(
        "--max-fp-frames",
        type=int,
        default=80,
        help="Maximum false-positive review images per run; 0 exports all. Metrics always use all frames.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Compute manifests and summaries without exporting review JPEGs.",
    )
    parser.add_argument(
        "--oracle-crop",
        action="store_true",
        help=(
            "Run an optional diagnostic crop centered on GT for each false negative. "
            "This is an oracle experiment only, never a production mode."
        ),
    )
    parser.add_argument("--oracle-crop-size", type=int, default=640)
    parser.add_argument("--model", default="weights/Model_v0.1.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def point_distance(box: list[float], point: list[float]) -> float:
    cx, cy = box_center(box)
    return math.hypot(cx - float(point[0]), cy - float(point[1]))


def center_distance(center: list[float] | tuple[float, float], point: list[float]) -> float:
    return math.hypot(float(center[0]) - float(point[0]), float(center[1]) - float(point[1]))


def candidate_center(candidate: dict[str, Any]) -> tuple[float, float] | None:
    center = candidate.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    box = candidate.get("box")
    if isinstance(box, list) and len(box) == 4:
        return box_center(box)
    return None


def point_matches_box(box: list[float], point: list[float], base_margin_px: float) -> bool:
    x1, y1, x2, y2 = [float(value) for value in box]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    margin = max(float(base_margin_px), 0.25 * max(width, height))
    x, y = float(point[0]), float(point[1])
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def choose_legacy_match(
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


def candidate_matches_truth(
    candidate: dict[str, Any],
    truth: dict[str, Any],
    gt_radius_px: float,
    bbox_iou: float,
) -> tuple[bool, float | None]:
    if not bool(truth.get("visible")):
        return False, None
    point = truth.get("point")
    center = candidate_center(candidate)
    if isinstance(point, list) and len(point) == 2 and center is not None:
        distance = center_distance(center, point)
        return distance <= gt_radius_px, distance
    gt_box = truth.get("bbox")
    box = candidate.get("box")
    if isinstance(gt_box, list) and len(gt_box) == 4 and isinstance(box, list) and len(box) == 4:
        score = iou_xyxy(box, gt_box)
        return score >= bbox_iou, None
    return False, None


def nearest_candidate_distance(
    candidates: list[dict[str, Any]],
    truth: dict[str, Any],
) -> float | None:
    point = truth.get("point")
    if not isinstance(point, list) or len(point) != 2:
        return None
    distances = [
        center_distance(center, point)
        for candidate in candidates
        if (center := candidate_center(candidate)) is not None
    ]
    return min(distances) if distances else None


def nearest_distance_bucket(distance: float | None, candidate_count: int, gt_radius_px: float) -> str:
    if candidate_count <= 0 or distance is None:
        return "no_candidate"
    if distance <= gt_radius_px:
        return "within_gt_radius"
    if distance <= 80.0:
        return "gt_radius_to_80px"
    if distance <= 160.0:
        return "80_to_160px"
    return "over_160px"


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


def read_frame_at_timestamp(capture: cv2.VideoCapture, timestamp_sec: float):
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps > 0:
        frame_id = max(0, int(round(float(timestamp_sec) * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_sec)) * 1000.0)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read video at t={timestamp_sec:.6f}s")
    return frame


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": len(values),
        "mean": round(float(statistics.fmean(values)), 6),
        "median": round(float(statistics.median(values)), 6),
        "p10": round(float(percentile(values, 0.10) or 0.0), 6),
        "p90": round(float(percentile(values, 0.90) or 0.0), 6),
    }


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * float(numerator) / max(float(denominator), 1.0), 2)


def align_production_truth(
    production_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    tolerance_sec: float,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], int]:
    labelled_truth = [row for row in truth_rows if isinstance(row.get("visible"), bool)]
    labelled_truth.sort(key=lambda row: float(row.get("timestamp_sec", 0.0)))
    truth_times = [float(row.get("timestamp_sec", 0.0)) for row in labelled_truth]
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    unmatched = 0

    for production in production_rows:
        if not bool(production.get("valid_gameplay")):
            continue
        timestamp = float(production.get("timestamp_sec", 0.0))
        index = bisect.bisect_left(truth_times, timestamp)
        options = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(labelled_truth)]
        if not options:
            unmatched += 1
            continue
        truth_index = min(options, key=lambda candidate: abs(truth_times[candidate] - timestamp))
        delta = abs(truth_times[truth_index] - timestamp)
        if delta > tolerance_sec:
            unmatched += 1
            continue
        matches.append((production, labelled_truth[truth_index], delta))
    return matches, unmatched


def resolve_video_path(
    explicit_video: str | None,
    gt_dir: Path | None,
    run_dir: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit_video:
        candidates.append(Path(explicit_video).expanduser())
    if gt_dir is not None:
        metadata_path = gt_dir / "metadata.json"
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                value = payload.get("video")
                if value:
                    candidates.append(Path(str(value)).expanduser())
            except (OSError, json.JSONDecodeError):
                pass
    if run_dir is not None:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                value = (payload.get("extra") or {}).get("video_path")
                if value:
                    candidates.append(Path(str(value)).expanduser())
            except (OSError, json.JSONDecodeError):
                pass
    cwd = Path.cwd()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else cwd / candidate
        if resolved.exists():
            return resolved.resolve()
    return None


def crop_box_around_point(
    width: int,
    height: int,
    point: list[float],
    crop_size: int,
) -> tuple[int, int, int, int]:
    crop_w = min(max(32, int(crop_size)), max(width, 1))
    crop_h = min(max(32, int(crop_size)), max(height, 1))
    cx, cy = float(point[0]), float(point[1])
    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    x1 = max(0, min(x1, max(width - crop_w, 0)))
    y1 = max(0, min(y1, max(height - crop_h, 0)))
    return x1, y1, x1 + crop_w, y1 + crop_h


def is_ball_class(name: str) -> bool:
    value = name.lower()
    return "ball" in value or "basketball" in value or "sports ball" in value


def ball_class_ids(names: Any) -> list[int]:
    if isinstance(names, dict):
        ids = [int(cls_id) for cls_id, name in names.items() if is_ball_class(str(name))]
    elif isinstance(names, (list, tuple)):
        ids = [index for index, name in enumerate(names) if is_ball_class(str(name))]
    else:
        ids = []
    return ids or [0]


def run_oracle_crop(
    model: Any,
    frame: Any,
    truth: dict[str, Any],
    *,
    crop_size: int,
    threshold: float,
    gt_radius_px: float,
    bbox_iou: float,
    imgsz: int,
    device: str,
    nms_iou: float,
    max_det: int,
) -> dict[str, Any]:
    point = truth.get("point")
    if not isinstance(point, list) or len(point) != 2:
        return {"eligible": False, "recovered": False, "candidate_count": 0}
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = crop_box_around_point(width, height, point, crop_size)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"eligible": False, "recovered": False, "candidate_count": 0}
    ids = ball_class_ids(getattr(model, "names", None))
    results = model.predict(
        crop,
        conf=float(threshold),
        imgsz=int(imgsz),
        device=device,
        iou=float(nms_iou),
        max_det=int(max_det),
        classes=ids,
        verbose=False,
    )
    candidates: list[dict[str, Any]] = []
    if results:
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                global_box = [
                    float(xyxy[0]) + x1,
                    float(xyxy[1]) + y1,
                    float(xyxy[2]) + x1,
                    float(xyxy[3]) + y1,
                ]
                center = box_center(global_box)
                candidates.append(
                    {
                        "box": global_box,
                        "center": [center[0], center[1]],
                        "confidence": confidence,
                    }
                )
    matches = [candidate_matches_truth(candidate, truth, gt_radius_px, bbox_iou)[0] for candidate in candidates]
    nearest = nearest_candidate_distance(candidates, truth)
    return {
        "eligible": True,
        "recovered": any(matches),
        "candidate_count": len(candidates),
        "nearest_distance_px": None if nearest is None else round(float(nearest), 4),
        "max_confidence": (
            round(max(float(candidate["confidence"]) for candidate in candidates), 6)
            if candidates
            else None
        ),
        "crop_box": [x1, y1, x2, y2],
    }


def select_review_rows(rows: list[dict[str, Any]], limit: int, *, false_negative: bool) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    if false_negative:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("nearest_distance_bucket")), []).append(row)
        selected: list[dict[str, Any]] = []
        for bucket_rows in grouped.values():
            selected.append(bucket_rows[len(bucket_rows) // 2])
        remaining = max(limit - len(selected), 0)
        if remaining:
            ordered = sorted(rows, key=lambda row: float(row.get("timestamp_sec", 0.0)))
            step = max(len(ordered) / remaining, 1.0)
            for index in range(remaining):
                selected.append(ordered[min(int(index * step), len(ordered) - 1)])
        dedup = {(str(row.get("run_id")), int(row.get("frame_id", -1))): row for row in selected}
        return list(dedup.values())[:limit]
    ordered = sorted(rows, key=lambda row: float(row.get("max_fp_confidence") or 0.0), reverse=True)
    return ordered[:limit]


def analyze_production_case(
    *,
    run_id: str,
    frames_path: Path,
    annotations_path: Path,
    output_dir: Path,
    video_path: Path | None,
    gt_radius_px: float,
    bbox_iou: float,
    timestamp_tolerance_sec: float,
    threshold: float,
    max_fn_frames: int,
    max_fp_frames: int,
    export_images: bool,
    oracle_model: Any | None,
    oracle_crop_size: int,
    imgsz: int,
    device: str,
    nms_iou: float,
    max_det: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    production_rows = load_jsonl(frames_path)
    truth_rows = load_jsonl(annotations_path)
    aligned, unmatched_gameplay = align_production_truth(
        production_rows,
        truth_rows,
        tolerance_sec=max(0.0, float(timestamp_tolerance_sec)),
    )
    if not aligned:
        raise ValueError(f"No gameplay rows aligned with GT for {run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "review"
    fn_dir = review_dir / "false_negatives"
    fp_dir = review_dir / "false_positives"
    if export_images:
        fn_dir.mkdir(parents=True, exist_ok=True)
        fp_dir.mkdir(parents=True, exist_ok=True)
    capture = None
    if export_images or oracle_model is not None:
        if video_path is None:
            raise FileNotFoundError(
                f"Video is required for image export/oracle crop in run {run_id}; "
                "pass --video for a single run or keep metadata.json/summary.json beside the run."
            )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")

    visible_frames = 0
    not_visible_frames = 0
    true_candidate_available = 0
    candidate_frames = 0
    false_candidate_total = 0
    false_candidate_not_visible = 0
    not_visible_candidate_frames = 0
    true_candidate_confidences: list[float] = []
    false_candidate_confidences: list[float] = []
    false_negatives: list[dict[str, Any]] = []
    false_positive_frames: list[dict[str, Any]] = []
    nearest_buckets: Counter[str] = Counter()
    nearest_fn_distances: list[float] = []
    manifest_rows: list[dict[str, Any]] = []
    oracle_eligible = 0
    oracle_recovered = 0

    for production, truth, timestamp_delta in aligned:
        association = production.get("ball_association") or {}
        candidates = [
            dict(candidate)
            for candidate in (association.get("candidates") or [])
            if isinstance(candidate, dict)
        ]
        timestamp = float(production.get("timestamp_sec", 0.0))
        visible = bool(truth.get("visible"))
        if candidates:
            candidate_frames += 1
        matches: list[int] = []
        for index, candidate in enumerate(candidates):
            correct, _ = candidate_matches_truth(candidate, truth, gt_radius_px, bbox_iou)
            if correct:
                matches.append(index)
        if visible:
            visible_frames += 1
            if matches:
                true_candidate_available += 1
                for index in matches:
                    confidence = candidates[index].get("confidence")
                    if confidence is not None:
                        true_candidate_confidences.append(float(confidence))
        else:
            not_visible_frames += 1
            if candidates:
                not_visible_candidate_frames += 1
        false_indices = [index for index in range(len(candidates)) if index not in matches]
        false_candidate_total += len(false_indices)
        if not visible:
            false_candidate_not_visible += len(false_indices)
        for index in false_indices:
            confidence = candidates[index].get("confidence")
            if confidence is not None:
                false_candidate_confidences.append(float(confidence))

        if visible and not matches:
            nearest = nearest_candidate_distance(candidates, truth)
            bucket = nearest_distance_bucket(nearest, len(candidates), gt_radius_px)
            nearest_buckets[bucket] += 1
            if nearest is not None:
                nearest_fn_distances.append(float(nearest))
            row = {
                "type": "false_negative",
                "run_id": run_id,
                "frame_id": int(production.get("frame_id", -1)),
                "timestamp_sec": timestamp,
                "timestamp_delta_sec": round(float(timestamp_delta), 8),
                "visible": True,
                "candidate_count": len(candidates),
                "false_candidate_count": len(false_indices),
                "nearest_candidate_distance_px": None if nearest is None else round(float(nearest), 4),
                "nearest_distance_bucket": bucket,
                "max_candidate_confidence": (
                    round(max(float(candidate.get("confidence", 0.0)) for candidate in candidates), 6)
                    if candidates
                    else None
                ),
                "gt_point": truth.get("point"),
                "gt_bbox": truth.get("bbox"),
                "candidates": candidates,
            }
            if oracle_model is not None and capture is not None:
                frame = read_frame_at_timestamp(capture, timestamp)
                oracle = run_oracle_crop(
                    oracle_model,
                    frame,
                    truth,
                    crop_size=oracle_crop_size,
                    threshold=threshold,
                    gt_radius_px=gt_radius_px,
                    bbox_iou=bbox_iou,
                    imgsz=imgsz,
                    device=device,
                    nms_iou=nms_iou,
                    max_det=max_det,
                )
                row["oracle_crop"] = oracle
                if oracle.get("eligible"):
                    oracle_eligible += 1
                    if oracle.get("recovered"):
                        oracle_recovered += 1
            false_negatives.append(row)

        if false_indices:
            fp_candidates = [candidates[index] for index in false_indices]
            false_positive_frames.append(
                {
                    "type": "false_positive",
                    "run_id": run_id,
                    "frame_id": int(production.get("frame_id", -1)),
                    "timestamp_sec": timestamp,
                    "visible": visible,
                    "candidate_count": len(candidates),
                    "fp_count": len(fp_candidates),
                    "max_fp_confidence": max(float(candidate.get("confidence", 0.0)) for candidate in fp_candidates),
                    "gt_point": truth.get("point"),
                    "gt_bbox": truth.get("bbox"),
                    "true_candidate_indices": matches,
                    "candidates": candidates,
                    "fp_candidates": fp_candidates,
                }
            )

    selected_fn_rows = select_review_rows(false_negatives, max_fn_frames, false_negative=True)
    selected_fp_rows = select_review_rows(false_positive_frames, max_fp_frames, false_negative=False)

    if export_images and capture is not None:
        for row in selected_fn_rows:
            frame = read_frame_at_timestamp(capture, float(row["timestamp_sec"]))
            point = row.get("gt_point")
            if isinstance(point, list) and len(point) == 2:
                cv2.drawMarker(
                    frame,
                    (int(round(point[0])), int(round(point[1]))),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    28,
                    3,
                )
            gt_box = row.get("gt_bbox")
            if isinstance(gt_box, list) and len(gt_box) == 4:
                draw_box(frame, gt_box, "GT BALL", (0, 255, 255))
            for candidate in row.get("candidates", []):
                box = candidate.get("box")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                center = candidate_center(candidate)
                distance = None
                if center is not None and isinstance(point, list) and len(point) == 2:
                    distance = center_distance(center, point)
                label = f"CAND {float(candidate.get('confidence', 0.0)):.2f}"
                if distance is not None:
                    label += f" d={distance:.0f}px"
                draw_box(frame, box, label, (0, 0, 255))
            draw_header(
                frame,
                f"P0.1 FN @ {gt_radius_px:.0f}px | {run_id} | frame {row['frame_id']} | t={float(row['timestamp_sec']):.2f}s | {row['nearest_distance_bucket']}",
            )
            output_path = fn_dir / f"fn_frame_{int(row['frame_id']):06d}_t{float(row['timestamp_sec']):09.3f}.jpg"
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            row["image"] = str(output_path.relative_to(output_dir))

        for row in selected_fp_rows:
            frame = read_frame_at_timestamp(capture, float(row["timestamp_sec"]))
            point = row.get("gt_point")
            if isinstance(point, list) and len(point) == 2:
                cv2.drawMarker(
                    frame,
                    (int(round(point[0])), int(round(point[1]))),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    24,
                    2,
                )
            for candidate in row.get("fp_candidates", []):
                box = candidate.get("box")
                if isinstance(box, list) and len(box) == 4:
                    draw_box(frame, box, f"FP {float(candidate.get('confidence', 0.0)):.2f}", (0, 0, 255))
            draw_header(
                frame,
                f"P0.1 FP | {run_id} | frame {row['frame_id']} | t={float(row['timestamp_sec']):.2f}s | {row['fp_count']} FP",
            )
            output_path = fp_dir / f"fp_frame_{int(row['frame_id']):06d}_max_{float(row['max_fp_confidence']):.3f}.jpg"
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            row["image"] = str(output_path.relative_to(output_dir))

    if capture is not None:
        capture.release()

    manifest_rows.extend(false_negatives)
    manifest_rows.extend(false_positive_frames)
    gameplay_frames = len(aligned)
    summary = {
        "scope": "P0.1 production candidate error mining against one-click GT",
        "run_id": run_id,
        "frames_file": str(frames_path),
        "annotations_file": str(annotations_path),
        "video_file": None if video_path is None else str(video_path),
        "threshold": float(threshold),
        "gt_radius_px": float(gt_radius_px),
        "bbox_iou": float(bbox_iou),
        "timestamp_tolerance_sec": float(timestamp_tolerance_sec),
        "gameplay_frames_aligned": gameplay_frames,
        "gameplay_frames_unmatched": unmatched_gameplay,
        "visible_frames": visible_frames,
        "not_visible_frames": not_visible_frames,
        "candidate_oracle": {
            "true_candidate_available_visible": true_candidate_available,
            "visible_candidate_recall_pct": pct(true_candidate_available, visible_frames),
            "false_negative_frames": len(false_negatives),
        },
        "candidate_noise": {
            "candidate_frames": candidate_frames,
            "candidate_frame_rate_pct": pct(candidate_frames, gameplay_frames),
            "false_candidates_total": false_candidate_total,
            "false_candidates_per_gameplay_frame": round(false_candidate_total / max(gameplay_frames, 1), 6),
            "not_visible_candidate_frames": not_visible_candidate_frames,
            "not_visible_candidate_frame_rate_pct": pct(not_visible_candidate_frames, not_visible_frames),
            "false_candidates_not_visible": false_candidate_not_visible,
        },
        "confidence": {
            "true_candidates": distribution(true_candidate_confidences),
            "false_candidates": distribution(false_candidate_confidences),
        },
        "false_negative_structure": {
            "nearest_distance_buckets": dict(nearest_buckets),
            "nearest_candidate_distance_px": distribution(nearest_fn_distances),
        },
        "review_export": {
            "false_negative_images": len(selected_fn_rows) if export_images else 0,
            "false_positive_images": len(selected_fp_rows) if export_images else 0,
        },
        "oracle_crop": {
            "enabled": oracle_model is not None,
            "crop_size": int(oracle_crop_size),
            "eligible_false_negatives": oracle_eligible,
            "recovered_false_negatives": oracle_recovered,
            "recovery_pct": pct(oracle_recovered, oracle_eligible) if oracle_eligible else None,
            "warning": "Oracle crop uses ground-truth position and is diagnostic only; it is not a production result.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "type",
        "run_id",
        "frame_id",
        "timestamp_sec",
        "visible",
        "candidate_count",
        "false_candidate_count",
        "fp_count",
        "max_candidate_confidence",
        "max_fp_confidence",
        "nearest_candidate_distance_px",
        "nearest_distance_bucket",
        "oracle_crop_recovered",
        "image",
    ]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in manifest_rows:
            oracle = row.get("oracle_crop") if isinstance(row.get("oracle_crop"), dict) else {}
            writer.writerow(
                {
                    "type": row.get("type"),
                    "run_id": row.get("run_id"),
                    "frame_id": row.get("frame_id"),
                    "timestamp_sec": row.get("timestamp_sec"),
                    "visible": row.get("visible"),
                    "candidate_count": row.get("candidate_count"),
                    "false_candidate_count": row.get("false_candidate_count"),
                    "fp_count": row.get("fp_count"),
                    "max_candidate_confidence": row.get("max_candidate_confidence"),
                    "max_fp_confidence": row.get("max_fp_confidence"),
                    "nearest_candidate_distance_px": row.get("nearest_candidate_distance_px"),
                    "nearest_distance_bucket": row.get("nearest_distance_bucket"),
                    "oracle_crop_recovered": oracle.get("recovered"),
                    "image": row.get("image"),
                }
            )
    return summary, manifest_rows


def aggregate_production_summaries(
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    visible = sum(int(summary["visible_frames"]) for summary in summaries)
    not_visible = sum(int(summary["not_visible_frames"]) for summary in summaries)
    gameplay = sum(int(summary["gameplay_frames_aligned"]) for summary in summaries)
    available = sum(int(summary["candidate_oracle"]["true_candidate_available_visible"]) for summary in summaries)
    false_candidates = sum(int(summary["candidate_noise"]["false_candidates_total"]) for summary in summaries)
    invisible_candidate_frames = sum(int(summary["candidate_noise"]["not_visible_candidate_frames"]) for summary in summaries)
    false_negatives = sum(int(summary["candidate_oracle"]["false_negative_frames"]) for summary in summaries)
    oracle_eligible = sum(int(summary["oracle_crop"]["eligible_false_negatives"]) for summary in summaries)
    oracle_recovered = sum(int(summary["oracle_crop"]["recovered_false_negatives"]) for summary in summaries)
    buckets: Counter[str] = Counter()
    for summary in summaries:
        buckets.update(summary["false_negative_structure"]["nearest_distance_buckets"])
    macro_recall = statistics.fmean(
        float(summary["candidate_oracle"]["visible_candidate_recall_pct"]) for summary in summaries
    )
    macro_invisible = statistics.fmean(
        float(summary["candidate_noise"]["not_visible_candidate_frame_rate_pct"]) for summary in summaries
    )
    aggregate = {
        "scope": "P0.1 multi-run production candidate error mining",
        "runs": [summary["run_id"] for summary in summaries],
        "run_count": len(summaries),
        "gameplay_frames_aligned": gameplay,
        "visible_frames": visible,
        "not_visible_frames": not_visible,
        "candidate_oracle": {
            "true_candidate_available_visible": available,
            "false_negative_frames": false_negatives,
            "micro_visible_candidate_recall_pct": pct(available, visible),
            "macro_visible_candidate_recall_pct": round(float(macro_recall), 2),
            "worst_video_visible_candidate_recall_pct": min(
                float(summary["candidate_oracle"]["visible_candidate_recall_pct"]) for summary in summaries
            ),
        },
        "candidate_noise": {
            "false_candidates_total": false_candidates,
            "micro_false_candidates_per_gameplay_frame": round(false_candidates / max(gameplay, 1), 6),
            "not_visible_candidate_frames": invisible_candidate_frames,
            "micro_not_visible_candidate_frame_rate_pct": pct(invisible_candidate_frames, not_visible),
            "macro_not_visible_candidate_frame_rate_pct": round(float(macro_invisible), 2),
        },
        "false_negative_structure": {
            "nearest_distance_buckets": dict(buckets),
        },
        "oracle_crop": {
            "enabled": any(bool(summary["oracle_crop"]["enabled"]) for summary in summaries),
            "eligible_false_negatives": oracle_eligible,
            "recovered_false_negatives": oracle_recovered,
            "recovery_pct": pct(oracle_recovered, oracle_eligible) if oracle_eligible else None,
            "warning": "Oracle crop uses ground-truth position and is diagnostic only; it is not a production result.",
        },
        "per_run": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def run_legacy_review(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    per_frame_path = eval_dir / "per_frame.json"
    if not per_frame_path.exists():
        raise FileNotFoundError(per_frame_path)
    if not args.video:
        raise ValueError("--video is required with --eval-dir")
    video_path = Path(args.video).expanduser().resolve()
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
        match_index = choose_legacy_match(predictions, gt_box, gt_point, point_margin_px) if visible else None
        if visible and match_index is None:
            false_negatives.append({**row, "filtered_predictions": predictions})
        fp_predictions = [prediction for idx, prediction in enumerate(predictions) if idx != match_index]
        if fp_predictions:
            false_positive_detections += len(fp_predictions)
            fp_frames.append(
                {
                    **row,
                    "filtered_predictions": predictions,
                    "fp_predictions": fp_predictions,
                    "matched_prediction": predictions[match_index] if match_index is not None else None,
                    "max_fp_confidence": max(float(prediction["confidence"]) for prediction in fp_predictions),
                }
            )
    fp_frames.sort(key=lambda row: float(row["max_fp_confidence"]), reverse=True)
    selected_fn_frames = false_negatives if args.max_fn_frames <= 0 else false_negatives[: args.max_fn_frames]
    selected_fp_frames = fp_frames if args.max_fp_frames <= 0 else fp_frames[: args.max_fp_frames]
    review_dir = Path(args.output).expanduser().resolve() if args.output else eval_dir / "review"
    fn_dir = review_dir / "false_negatives"
    fp_dir = review_dir / "false_positives"
    if not args.no_images:
        fn_dir.mkdir(parents=True, exist_ok=True)
        fp_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        for row in selected_fn_frames:
            frame_id = int(row["frame_id"])
            frame = read_frame(capture, frame_id)
            point = row.get("point")
            if isinstance(point, list) and len(point) == 2:
                cv2.drawMarker(frame, (int(round(point[0])), int(round(point[1]))), (0, 255, 255), cv2.MARKER_CROSS, 28, 3)
            box = row.get("bbox")
            if isinstance(box, list) and len(box) == 4:
                draw_box(frame, box, "GT BALL", (0, 255, 255))
            for prediction in row.get("filtered_predictions", []):
                draw_box(frame, prediction["box"], f"OTHER {float(prediction['confidence']):.2f}", (0, 0, 255))
            draw_header(frame, f"FALSE NEGATIVE | frame {frame_id} | t={float(row.get('timestamp_sec', 0.0)):.2f}s | conf>={threshold:.2f}")
            cv2.imwrite(str(fn_dir / f"fn_frame_{frame_id:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        for row in selected_fp_frames:
            frame_id = int(row["frame_id"])
            frame = read_frame(capture, frame_id)
            for prediction in row.get("fp_predictions", []):
                draw_box(frame, prediction["box"], f"FP {float(prediction['confidence']):.2f}", (0, 0, 255))
            draw_header(frame, f"FALSE POSITIVE | frame {frame_id} | t={float(row.get('timestamp_sec', 0.0)):.2f}s | {len(row['fp_predictions'])} FP")
            cv2.imwrite(str(fp_dir / f"fp_frame_{frame_id:06d}_max_{float(row['max_fp_confidence']):.3f}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        capture.release()
    review_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "legacy_per_frame",
        "threshold": threshold,
        "point_margin_px": point_margin_px,
        "frames_evaluated": len(rows),
        "false_negative_frames": len(false_negatives),
        "false_positive_detections": false_positive_detections,
        "false_positive_frames": len(fp_frames),
        "false_negative_frames_exported": 0 if args.no_images else len(selected_fn_frames),
        "false_positive_frames_exported": 0 if args.no_images else len(selected_fp_frames),
        "review_dir": str(review_dir),
    }
    (review_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def load_oracle_model(args: argparse.Namespace) -> Any | None:
    if not args.oracle_crop:
        return None
    from ultralytics import YOLO
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    print(f"Loading oracle-crop model: {model_path}")
    return YOLO(str(model_path))


def main() -> None:
    args = parse_args()
    if args.eval_dir:
        if args.oracle_crop:
            raise ValueError("--oracle-crop is supported only with production frames/annotations mode")
        run_legacy_review(args)
        return
    oracle_model = load_oracle_model(args)
    summaries: list[dict[str, Any]] = []
    all_manifest_rows: list[dict[str, Any]] = []

    if args.frames:
        if not args.annotations:
            raise ValueError("--annotations is required with --frames")
        frames_path = Path(args.frames).expanduser().resolve()
        annotations_path = Path(args.annotations).expanduser().resolve()
        if not frames_path.exists():
            raise FileNotFoundError(frames_path)
        if not annotations_path.exists():
            raise FileNotFoundError(annotations_path)
        run_id = frames_path.parent.name
        output_dir = Path(args.output).expanduser().resolve() if args.output else frames_path.parent / "p0_1_error_review"
        video_path = resolve_video_path(args.video, annotations_path.parent, frames_path.parent)
        summary, _ = analyze_production_case(
            run_id=run_id,
            frames_path=frames_path,
            annotations_path=annotations_path,
            output_dir=output_dir,
            video_path=video_path,
            gt_radius_px=max(0.0, float(args.gt_radius_px)),
            bbox_iou=max(0.0, float(args.bbox_iou)),
            timestamp_tolerance_sec=max(0.0, float(args.timestamp_tolerance_sec)),
            threshold=float(args.threshold),
            max_fn_frames=int(args.max_fn_frames),
            max_fp_frames=int(args.max_fp_frames),
            export_images=not args.no_images,
            oracle_model=oracle_model,
            oracle_crop_size=int(args.oracle_crop_size),
            imgsz=int(args.imgsz),
            device=str(args.device),
            nms_iou=float(args.nms_iou),
            max_det=int(args.max_det),
        )
        print(json.dumps(summary, indent=2))
        print(f"[P0.1 REVIEW] {output_dir}")
        return

    run_ids = list(args.run_ids or [])
    if not run_ids:
        raise ValueError("At least one --run-id is required")
    run_root = Path(args.run_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    batch_output = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / "evaluation" / "p0_1_error_review").resolve()
    batch_output.mkdir(parents=True, exist_ok=True)
    for run_id in run_ids:
        run_dir = run_root / run_id
        gt_dir = gt_root / run_id
        frames_path = run_dir / "frames.jsonl"
        annotations_path = gt_dir / "annotations.jsonl"
        if not frames_path.exists():
            raise FileNotFoundError(frames_path)
        if not annotations_path.exists():
            raise FileNotFoundError(annotations_path)
        video_path = resolve_video_path(None, gt_dir, run_dir)
        summary, rows = analyze_production_case(
            run_id=run_id,
            frames_path=frames_path,
            annotations_path=annotations_path,
            output_dir=batch_output / run_id,
            video_path=video_path,
            gt_radius_px=max(0.0, float(args.gt_radius_px)),
            bbox_iou=max(0.0, float(args.bbox_iou)),
            timestamp_tolerance_sec=max(0.0, float(args.timestamp_tolerance_sec)),
            threshold=float(args.threshold),
            max_fn_frames=int(args.max_fn_frames),
            max_fp_frames=int(args.max_fp_frames),
            export_images=not args.no_images,
            oracle_model=oracle_model,
            oracle_crop_size=int(args.oracle_crop_size),
            imgsz=int(args.imgsz),
            device=str(args.device),
            nms_iou=float(args.nms_iou),
            max_det=int(args.max_det),
        )
        summaries.append(summary)
        all_manifest_rows.extend(rows)
        print(
            f"[{run_id}] recall={summary['candidate_oracle']['visible_candidate_recall_pct']:.2f}% "
            f"FN={summary['candidate_oracle']['false_negative_frames']} "
            f"false/frame={summary['candidate_noise']['false_candidates_per_gameplay_frame']:.3f}"
        )
    aggregate = aggregate_production_summaries(summaries, batch_output)
    with (batch_output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(aggregate, indent=2))
    print(f"[P0.1 BATCH REVIEW] {batch_output}")


if __name__ == "__main__":
    main()
