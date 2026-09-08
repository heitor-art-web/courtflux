from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2

from services.vision.ball_candidate_backend import create_ball_candidate_backend, normalize_label


def center_xywh(box: list[float]) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in box]
    return x + w / 2.0, y + h / 2.0


def iou_xywh_xyxy(gt: list[float], pred: list[float]) -> float:
    gx, gy, gw, gh = [float(v) for v in gt]
    px1, py1, px2, py2 = [float(v) for v in pred]
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(gx, px1), max(gy, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = gw * gh + max(0.0, px2 - px1) * max(0.0, py2 - py1) - inter
    return inter / union if union > 0 else 0.0


def is_ball_label(name: str) -> bool:
    return normalize_label(name) in {"ball", "basketball", "sports ball"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a CourtFlux-compatible ball backend on an external COCO detection dataset."
    )
    parser.add_argument("--images", required=True, help="Directory containing COCO images")
    parser.add_argument("--annotations", required=True, help="COCO _annotations.coco.json")
    parser.add_argument("--backend", choices=["yolo", "huggingface", "roboflow"], required=True)
    parser.add_argument("--yolo-model", default="weights/Model_v0.1.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--center-radius-px", type=float, default=40.0)
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="evaluation/external_ball_benchmark")
    args = parser.parse_args()

    annotations_path = Path(args.annotations).resolve()
    images_dir = Path(args.images).resolve()
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    ball_category_ids = {category_id for category_id, name in categories.items() if is_ball_label(name)}
    if not ball_category_ids:
        raise RuntimeError(f"No exact ball category found. categories={categories}")

    images = {int(item["id"]): item for item in payload.get("images", [])}
    gt_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        if int(annotation.get("category_id", -1)) in ball_category_ids:
            gt_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    if args.backend == "huggingface" and not os.getenv("BASKETBALL_HF_BALL_LABELS"):
        os.environ["BASKETBALL_HF_BALL_LABELS"] = "ball,sports ball,basketball"
    if args.backend == "roboflow" and not os.getenv("BASKETBALL_ROBOFLOW_BALL_LABELS"):
        os.environ["BASKETBALL_ROBOFLOW_BALL_LABELS"] = "ball"

    backend = create_ball_candidate_backend(
        backend_name=args.backend,
        yolo_model_path=args.yolo_model if args.backend == "yolo" else None,
        device=args.device,
        ball_predicate=lambda name: is_ball_label(name),
    )

    image_ids = sorted(gt_by_image)
    if args.limit > 0:
        image_ids = image_ids[: args.limit]

    gt_total = 0
    gt_matched_center = 0
    gt_matched_iou = 0
    frames_with_ball = 0
    frames_with_center_match = 0
    false_candidates = 0
    candidate_total = 0
    rows: list[dict[str, Any]] = []

    for image_id in image_ids:
        info = images.get(image_id)
        if not info:
            continue
        frame = cv2.imread(str(images_dir / info["file_name"]))
        if frame is None:
            raise FileNotFoundError(images_dir / info["file_name"])
        candidates = backend.predict(
            frame,
            conf=args.conf,
            iou=0.5,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=getattr(backend, "device", args.device),
        )
        gts = gt_by_image[image_id]
        gt_total += len(gts)
        frames_with_ball += 1
        candidate_total += len(candidates)
        center_matches_for_frame = 0
        matched_candidate_indices: set[int] = set()

        for gt in gts:
            gt_center = center_xywh(gt["bbox"])
            center_hits: list[tuple[float, int]] = []
            iou_hits: list[tuple[float, int]] = []
            for idx, candidate in enumerate(candidates):
                center = candidate.get("center")
                box = candidate.get("box")
                if center is not None:
                    distance = math.hypot(float(center[0]) - gt_center[0], float(center[1]) - gt_center[1])
                    if distance <= args.center_radius_px:
                        center_hits.append((distance, idx))
                if isinstance(box, list) and len(box) == 4:
                    score = iou_xywh_xyxy(gt["bbox"], box)
                    if score >= args.iou:
                        iou_hits.append((score, idx))
            if center_hits:
                _, idx = min(center_hits, key=lambda item: item[0])
                gt_matched_center += 1
                center_matches_for_frame += 1
                matched_candidate_indices.add(idx)
            if iou_hits:
                gt_matched_iou += 1
                matched_candidate_indices.add(max(iou_hits, key=lambda item: item[0])[1])

        if center_matches_for_frame > 0:
            frames_with_center_match += 1
        false_candidates += max(0, len(candidates) - len(matched_candidate_indices))
        rows.append({
            "image_id": image_id,
            "file_name": info["file_name"],
            "gt_ball_instances": len(gts),
            "candidate_count": len(candidates),
            "center_matches": center_matches_for_frame,
        })

    summary = {
        "scope": "external image-level ball detector benchmark",
        "dataset": str(annotations_path),
        "backend": backend.metadata() if hasattr(backend, "metadata") else {"backend": args.backend},
        "threshold": args.conf,
        "center_radius_px": args.center_radius_px,
        "iou_threshold": args.iou,
        "images_with_ball": frames_with_ball,
        "gt_ball_instances": gt_total,
        "candidate_total": candidate_total,
        "gt_instance_center_recall_pct": round(100.0 * gt_matched_center / max(gt_total, 1), 2),
        "gt_instance_iou_recall_pct": round(100.0 * gt_matched_iou / max(gt_total, 1), 2),
        "ball_frame_center_recall_pct": round(100.0 * frames_with_center_match / max(frames_with_ball, 1), 2),
        "false_candidates": false_candidates,
        "false_candidates_per_image": round(false_candidates / max(frames_with_ball, 1), 6),
        "metric_warning": (
            "This is an external image-detection benchmark, not temporal tracking coverage. "
            "Do not compare its recall directly to CourtFlux V1-V5 without accounting for dataset/domain differences."
        ),
    }

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[EXTERNAL BALL BENCHMARK] {output / 'summary.json'}")


if __name__ == "__main__":
    main()
