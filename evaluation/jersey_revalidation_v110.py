from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.jersey_uncertainty_ocr_benchmark import (
    UncertaintyJNRModel,
    _load_checkpoint,
    _preprocess_rgb,
    _resolve_device,
    _sync_device,
)
from tools.jersey_uncertainty_track_policy_benchmark import _group_prediction


P1_CLASSES = ("basketball", "player", "referee", "backboard", "rim")

# Frozen from the accepted #37 selective track policy. Revalidation must not retune it.
FROZEN_POLICY: dict[str, Any] = {
    "require_not_absent": False,
    "uncertainty_max": 0.9995630085468292,
    "pred_score_min": None,
    "aggregation": "best_uncertainty",
    "min_frames": 2,
    "agg_margin_min": 0.0004642619751393791,
    "vote_fraction_min": 0.0,
}

HISTORICAL_SELECTED_POLICY = {
    "detector": "weights/model_v1.0.0.pt",
    "tracker": "botsort.yaml",
    "player_confidence": 0.10,
    "all_numeric_tracks": 39,
    "accepted_numeric_tracks": 15,
    "correct_tracks": 14,
    "track_coverage": 15 / 39,
    "decision_precision": 14 / 15,
    "correct_recall": 14 / 39,
    "unreadable_false_accept_rate": 0.0,
    "semantics": "Historical #37 selected-policy result; selective track precision, not global jersey accuracy.",
}


def _class_name(model: YOLO, cls_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        value = names.get(cls_id, cls_id)
    else:
        value = names[cls_id] if 0 <= cls_id < len(names) else cls_id
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _canonical_class(name: str) -> str | None:
    value = str(name or "").strip().lower().replace("_", " ").replace("-", " ")
    if "basketball" in value or value in {"ball", "sports ball"}:
        return "basketball"
    if value == "player" or value == "basketball player" or value == "person":
        return "player"
    if "referee" in value or value == "ref":
        return "referee"
    if "backboard" in value:
        return "backboard"
    if value == "rim" or "basket rim" in value:
        return "rim"
    return None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _greedy_match(
    gt_boxes: list[tuple[float, float, float, float]],
    detections: list[dict[str, Any]],
    min_iou: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, gt_box in enumerate(gt_boxes):
        for di, detection in enumerate(detections):
            score = _iou(gt_box, tuple(float(v) for v in detection["box"]))
            if score >= min_iou:
                candidates.append((score, gi, di))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, gi, di in candidates:
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi)
        used_det.add(di)
        matches.append((gi, di, float(score)))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    unmatched_det = [i for i in range(len(detections)) if i not in used_det]
    return matches, unmatched_gt, unmatched_det


def _safe_crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _torso_crop(player_crop: np.ndarray) -> np.ndarray | None:
    h, w = player_crop.shape[:2]
    y1, y2 = int(h * 0.15), max(int(h * 0.65), int(h * 0.15) + 1)
    x1, x2 = int(w * 0.10), max(int(w * 0.90), int(w * 0.10) + 1)
    torso = player_crop[y1:y2, x1:x2]
    return torso if torso.size else None


def _load_gt(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "sample_id",
        "video",
        "frame_id",
        "track_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "subject_type",
        "jersey_gt",
        "label_kind",
        "dataset_role",
        "split",
        "identity_group",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ground truth missing required columns: {missing}")

    frame = frame[
        (frame["dataset_role"] == "ocr")
        & (frame["subject_type"] == "player")
        & frame["label_kind"].isin(["number", "unreadable"])
    ].copy()
    if frame.empty:
        raise ValueError("No OCR-eligible player rows in ground truth")

    for col in ("frame_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["frame_id"] = frame["frame_id"].astype(int)
    frame["jersey_gt_int"] = pd.to_numeric(
        frame["jersey_gt"].where(frame["label_kind"] == "number", ""), errors="coerce"
    )
    frame["original_group"] = frame.apply(
        lambda row: (
            f"{row['video']}|identity:{str(row['identity_group']).strip()}"
            if str(row["identity_group"]).strip()
            else f"{row['video']}|track:{row['track_id']}"
        ),
        axis=1,
    )
    return frame.reset_index(drop=True)


def _extract_detections(
    result: Any,
    model: YOLO,
    player_confidence: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    players: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    if result is None or result.boxes is None:
        return players, counts

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name_raw = _class_name(model, cls_id)
        canonical = _canonical_class(cls_name_raw)
        if canonical is not None:
            counts[canonical] += 1
        if canonical != "player":
            continue
        confidence = float(box.conf[0])
        if confidence < player_confidence or box.id is None:
            continue
        track_id = int(box.id[0])
        if track_id < 0:
            continue
        players.append(
            {
                "track_id": track_id,
                "confidence": confidence,
                "box": tuple(float(v) for v in box.xyxy[0].tolist()),
            }
        )
    return players, counts


def _infer_jerseys(
    records: list[dict[str, Any]],
    model_path: Path,
    device_name: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = _resolve_device(device_name)
    model = UncertaintyJNRModel()
    checkpoint = _load_checkpoint(model, model_path)
    model = model.to(device).eval()

    output: list[dict[str, Any]] = []
    for start in range(0, len(records), max(1, batch_size)):
        chunk = records[start : start + max(1, batch_size)]
        tensors = []
        valid_rows = []
        for row in chunk:
            crop = row.pop("_torso_crop")
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(_preprocess_rgb(rgb, mode="torso_direct"))
            valid_rows.append(row)
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        _sync_device(device)
        with torch.inference_mode():
            result = model(batch)
        _sync_device(device)

        probs = result["number_probs"].detach().cpu().numpy()
        all_logits = result["all_logits"].detach().cpu().numpy()
        uncertainty = result["uncertainty"].detach().cpu().numpy()
        for index, source in enumerate(valid_rows):
            p = probs[index].astype(float)
            order = np.argsort(p)[::-1]
            pred = int(order[0])
            max_number_logit = float(np.max(all_logits[index, :100]))
            absent_logit = float(all_logits[index, 100])
            item = dict(source)
            item.update(
                {
                    "pred_number": pred,
                    "pred_score": float(p[pred]),
                    "top2_score": float(p[order[1]]),
                    "number_margin": float(p[order[0]] - p[order[1]]),
                    "uncertainty": float(uncertainty[index]),
                    "absent_argmax": bool(int(np.argmax(all_logits[index])) == 100),
                    "absent_margin": float(absent_logit - max_number_logit),
                    "_number_probs": p,
                }
            )
            output.append(item)
    return output, {"device": device.type, "checkpoint_coverage": checkpoint["coverage"]}


def _track_groups(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_track: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_track[int(row["new_track_id"])].append(row)

    groups: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for track_id, rows in sorted(by_track.items()):
        numeric_labels = sorted(
            {
                int(row["jersey_gt"])
                for row in rows
                if row["label_kind"] == "number" and str(row["jersey_gt"]).isdigit()
            }
        )
        original_groups = sorted({str(row["original_group"]) for row in rows})
        splits = sorted({str(row["split"]) for row in rows if str(row["split"])})
        if len(numeric_labels) > 1:
            conflicts.append(
                {
                    "new_track_id": track_id,
                    "numeric_labels": numeric_labels,
                    "observations": len(rows),
                    "original_groups": original_groups,
                    "splits": splits,
                }
            )
            continue
        groups.append(
            {
                "new_track_id": track_id,
                "kind": "numeric" if numeric_labels else "unreadable",
                "gt": numeric_labels[0] if numeric_labels else None,
                "split": splits[0] if len(splits) == 1 else "mixed",
                "splits": splits,
                "original_groups": original_groups,
                "frame": pd.DataFrame(rows),
            }
        )
    return groups, conflicts


def _evaluate(groups: list[dict[str, Any]], split: str = "all") -> dict[str, Any]:
    subset = [
        group
        for group in groups
        if split == "all" or group["split"] == split
    ]
    numeric = [group for group in subset if group["kind"] == "numeric"]
    unreadable = [group for group in subset if group["kind"] == "unreadable"]

    accepted_groups = 0
    accepted_numeric = 0
    accepted_unreadable = 0
    correct = 0
    insufficient_policy_frames = 0
    decisions: list[dict[str, Any]] = []

    for group in subset:
        frame = group["frame"]
        eligible = frame
        if FROZEN_POLICY["uncertainty_max"] is not None:
            eligible = eligible[
                eligible["uncertainty"].astype(float) <= float(FROZEN_POLICY["uncertainty_max"])
            ]
        if len(eligible) < int(FROZEN_POLICY["min_frames"]):
            insufficient_policy_frames += 1
        result = _group_prediction(group, FROZEN_POLICY)
        accepted = bool(result.get("accepted"))
        decision = {
            "new_track_id": int(group["new_track_id"]),
            "kind": group["kind"],
            "gt": group["gt"],
            "split": group["split"],
            "observations": int(len(frame)),
            "eligible_policy_frames": int(len(eligible)),
            "accepted": accepted,
            "prediction": result.get("prediction"),
            "original_groups": group["original_groups"],
        }
        if accepted:
            accepted_groups += 1
            if group["kind"] == "numeric":
                accepted_numeric += 1
                is_correct = int(result["prediction"]) == int(group["gt"])
                correct += int(is_correct)
                decision["correct"] = is_correct
            else:
                accepted_unreadable += 1
                decision["correct"] = False
        decisions.append(decision)

    return {
        "tracks": len(subset),
        "numeric_tracks": len(numeric),
        "unreadable_tracks": len(unreadable),
        "accepted_groups": accepted_groups,
        "accepted_numeric_tracks": accepted_numeric,
        "accepted_unreadable_tracks": accepted_unreadable,
        "correct": correct,
        "insufficient_policy_frames": insufficient_policy_frames,
        "track_coverage": accepted_numeric / len(numeric) if numeric else None,
        "decision_precision": correct / accepted_groups if accepted_groups else None,
        "accepted_numeric_accuracy": correct / accepted_numeric if accepted_numeric else None,
        "unreadable_false_accept_rate": (
            accepted_unreadable / len(unreadable) if unreadable else None
        ),
        "correct_recall": correct / len(numeric) if numeric else None,
        "decisions": decisions,
    }


def _fragmentation_summary(gt: pd.DataFrame, matched_records: list[dict[str, Any]]) -> dict[str, Any]:
    all_groups = sorted(set(gt["original_group"].astype(str)))
    mapped: defaultdict[str, set[int]] = defaultdict(set)
    for row in matched_records:
        mapped[str(row["original_group"])].add(int(row["new_track_id"]))

    matched_groups = [group for group in all_groups if group in mapped]
    counts = [len(mapped[group]) for group in matched_groups]
    fragmented = [group for group in matched_groups if len(mapped[group]) > 1]
    return {
        "original_groups_total": len(all_groups),
        "original_groups_with_matches": len(matched_groups),
        "original_groups_without_matches": len(all_groups) - len(matched_groups),
        "groups_mapping_to_one_new_track": sum(len(mapped[group]) == 1 for group in matched_groups),
        "groups_mapping_to_multiple_new_tracks": len(fragmented),
        "fragmented_group_rate": len(fragmented) / len(matched_groups) if matched_groups else None,
        "new_tracks_per_matched_original_group_mean": mean(counts) if counts else None,
        "new_tracks_per_matched_original_group_median": median(counts) if counts else None,
        "max_new_tracks_for_one_original_group": max(counts) if counts else 0,
        "fragmented_groups": [
            {"original_group": group, "new_track_ids": sorted(mapped[group])}
            for group in fragmented
        ],
        "semantics": (
            "Diagnostic mapping from reviewed #37 groups to current v1.1.0 track IDs on the same labelled frames. "
            "It is a fragmentation proxy, not true physical-player ID-switch ground truth."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate the frozen #37 jersey track policy on the current v1.1.0 + BoT-SORT "
            "PlayerState operating point without retuning the jersey policy."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=ROOT / "data" / "jersey_ground_truth" / "porto_ovarense_v1" / "ground_truth.csv",
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "OwiQgAMqyUs.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "model_v1.1.0.pt")
    parser.add_argument(
        "--jersey-model",
        type=Path,
        default=ROOT / "weights" / "jersey_uncertainty_vit_b.pt",
    )
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--inference-conf", type=float, default=0.03)
    parser.add_argument("--player-conf", type=float, default=0.40)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    gt_path = args.ground_truth.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    jersey_model_path = args.jersey_model.expanduser().resolve()
    for path in (gt_path, video_path, model_path, jersey_model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if not (0.0 <= args.match_iou <= 1.0):
        raise ValueError("--match-iou must be between 0 and 1")

    gt = _load_gt(gt_path)
    gt_by_frame = {int(frame_id): group.reset_index(drop=True) for frame_id, group in gt.groupby("frame_id")}
    min_frame = int(gt["frame_id"].min())
    max_frame = int(gt["frame_id"].max())

    detector = YOLO(str(model_path))
    detector.to(args.device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    class_counts: Counter[str] = Counter()
    matched_records: list[dict[str, Any]] = []
    match_ious: list[float] = []
    matched_gt_ids: set[str] = set()
    unmatched_new_detections = 0
    processed = 0
    frame_id = min_frame

    try:
        while frame_id <= max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            results = detector.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=float(args.inference_conf),
                imgsz=int(args.imgsz),
                device=args.device,
                max_det=40,
                verbose=False,
            )
            result = results[0] if results else None
            players, frame_counts = _extract_detections(result, detector, float(args.player_conf))
            class_counts.update(frame_counts)
            processed += 1

            gt_rows = gt_by_frame.get(frame_id)
            if gt_rows is not None:
                gt_boxes = [
                    (
                        float(row["bbox_x1"]),
                        float(row["bbox_y1"]),
                        float(row["bbox_x2"]),
                        float(row["bbox_y2"]),
                    )
                    for _, row in gt_rows.iterrows()
                ]
                matches, _, unmatched_det = _greedy_match(gt_boxes, players, float(args.match_iou))
                unmatched_new_detections += len(unmatched_det)
                for gt_index, det_index, overlap in matches:
                    gt_row = gt_rows.iloc[gt_index]
                    det = players[det_index]
                    player_crop = _safe_crop(frame, det["box"])
                    if player_crop is None:
                        continue
                    torso = _torso_crop(player_crop)
                    if torso is None or torso.shape[0] < 16 or torso.shape[1] < 12:
                        continue
                    sample_id = str(gt_row["sample_id"])
                    matched_gt_ids.add(sample_id)
                    match_ious.append(overlap)
                    matched_records.append(
                        {
                            "sample_id": sample_id,
                            "frame_id": int(frame_id),
                            "timestamp_sec": frame_id / max(fps, 1e-6),
                            "original_track_id": str(gt_row["track_id"]),
                            "original_group": str(gt_row["original_group"]),
                            "identity_group": str(gt_row["identity_group"]),
                            "split": str(gt_row["split"]),
                            "label_kind": str(gt_row["label_kind"]),
                            "jersey_gt": str(gt_row["jersey_gt"]),
                            "new_track_id": int(det["track_id"]),
                            "new_player_confidence": float(det["confidence"]),
                            "match_iou": float(overlap),
                            "new_box": [float(v) for v in det["box"]],
                            "_torso_crop": torso,
                        }
                    )

            if processed % 250 == 0 or frame_id == max_frame:
                counts_text = " ".join(f"{name}={class_counts.get(name, 0)}" for name in P1_CLASSES)
                print(
                    f"[JERSEY REVALIDATION] frame={frame_id}/{max_frame} processed={processed} "
                    f"detections[{counts_text} total={sum(class_counts.values())}] "
                    f"matched_gt={len(matched_gt_ids)}/{len(gt)} new_tracks={len({r['new_track_id'] for r in matched_records})}",
                    flush=True,
                )
            frame_id += 1
    finally:
        cap.release()

    predictions, jersey_runtime = _infer_jerseys(
        matched_records,
        jersey_model_path,
        args.device,
        max(1, int(args.batch_size)),
    )
    groups, conflicts = _track_groups(predictions)
    evaluations = {split: _evaluate(groups, split) for split in ("train", "val", "test", "mixed", "all")}

    matched_numeric = sum(row["label_kind"] == "number" for row in predictions)
    gt_numeric = int((gt["label_kind"] == "number").sum())
    matched_unreadable = sum(row["label_kind"] == "unreadable" for row in predictions)
    gt_unreadable = int((gt["label_kind"] == "unreadable").sum())

    all_metrics = evaluations["all"]
    comparison = {
        "historical": HISTORICAL_SELECTED_POLICY,
        "current": {
            key: all_metrics.get(key)
            for key in (
                "numeric_tracks",
                "accepted_numeric_tracks",
                "correct",
                "track_coverage",
                "decision_precision",
                "correct_recall",
                "unreadable_false_accept_rate",
            )
        },
        "delta": {
            "track_coverage_pp": (
                100.0 * (all_metrics["track_coverage"] - HISTORICAL_SELECTED_POLICY["track_coverage"])
                if all_metrics["track_coverage"] is not None else None
            ),
            "decision_precision_pp": (
                100.0 * (
                    all_metrics["decision_precision"] - HISTORICAL_SELECTED_POLICY["decision_precision"]
                )
                if all_metrics["decision_precision"] is not None else None
            ),
            "correct_recall_pp": (
                100.0 * (all_metrics["correct_recall"] - HISTORICAL_SELECTED_POLICY["correct_recall"])
                if all_metrics["correct_recall"] is not None else None
            ),
        },
        "interpretation_guard": (
            "The current run reuses the same human-reviewed Porto–Ovarense evidence and frozen jersey policy. "
            "It is downstream revalidation / regression evidence, not independent held-out promotion evidence."
        ),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "evaluation" / "jersey_revalidation_v110" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_predictions = []
    for row in predictions:
        item = {key: value for key, value in row.items() if key != "_number_probs"}
        serializable_predictions.append(item)
    with (output_dir / "matched_observations.jsonl").open("w", encoding="utf-8") as handle:
        for item in serializable_predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    decision_rows = []
    for split, metrics in evaluations.items():
        for row in metrics["decisions"]:
            decision_rows.append(dict(row, evaluation_split=split))
    with (output_dir / "track_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for item in decision_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 Jersey Identification revalidation after PlayerState v1",
        "evidence_status": "same_reviewed_video_downstream_revalidation_not_independent_held_out",
        "runtime": {
            "video": str(video_path),
            "ground_truth": str(gt_path),
            "detector": str(model_path),
            "tracker": args.tracker,
            "imgsz": int(args.imgsz),
            "inference_confidence": float(args.inference_conf),
            "player_confidence": float(args.player_conf),
            "jersey_model": str(jersey_model_path),
            "jersey_device": jersey_runtime["device"],
            "jersey_checkpoint_coverage": jersey_runtime["checkpoint_coverage"],
            "match_iou_min": float(args.match_iou),
            "frozen_policy": FROZEN_POLICY,
        },
        "detector_diagnostics": {
            "frames_processed": processed,
            "frame_range": [min_frame, max_frame],
            "class_counts": {name: int(class_counts.get(name, 0)) for name in P1_CLASSES},
            "all_p1_class_detections": int(sum(class_counts.values())),
        },
        "gt_matching": {
            "ocr_rows": int(len(gt)),
            "matched_rows": int(len(predictions)),
            "match_rate": len(predictions) / len(gt) if len(gt) else None,
            "numeric_rows": gt_numeric,
            "matched_numeric_rows": matched_numeric,
            "numeric_match_rate": matched_numeric / gt_numeric if gt_numeric else None,
            "unreadable_rows": gt_unreadable,
            "matched_unreadable_rows": matched_unreadable,
            "unreadable_match_rate": matched_unreadable / gt_unreadable if gt_unreadable else None,
            "mean_iou": mean(match_ious) if match_ious else None,
            "median_iou": median(match_ious) if match_ious else None,
            "min_iou": min(match_ious) if match_ious else None,
            "unmatched_new_player_detections_on_reviewed_frames": unmatched_new_detections,
            "semantics": (
                "GT labels are transferred only when a current v1.1.0 tracked-player box overlaps a human-reviewed #37 player box on the exact same frame."
            ),
        },
        "track_mapping": _fragmentation_summary(gt, predictions),
        "new_track_label_conflicts": {
            "count": len(conflicts),
            "tracks": conflicts,
            "semantics": (
                "A current track matching more than one numeric human jersey label is flagged as a possible identity association failure and excluded from jersey decision metrics."
            ),
        },
        "evaluation": {
            split: {key: value for key, value in metrics.items() if key != "decisions"}
            for split, metrics in evaluations.items()
        },
        "historical_comparison": comparison,
        "scope_guard": {
            "jersey_policy_retuned": False,
            "player_tracker_retuned": False,
            "team_identity_resolved": False,
            "identity_consolidation_performed": False,
            "global_jersey_accuracy_claim": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "[JERSEY_REVALIDATION_SUMMARY] "
        f"match={100.0 * summary['gt_matching']['match_rate']:.1f}% "
        f"numeric_tracks={all_metrics['numeric_tracks']} accepted={all_metrics['accepted_numeric_tracks']} "
        f"precision={all_metrics['decision_precision']} coverage={all_metrics['track_coverage']} "
        f"fragmented_groups={summary['track_mapping']['groups_mapping_to_multiple_new_tracks']} "
        f"conflict_tracks={len(conflicts)}",
        flush=True,
    )
    print(f"[JERSEY_REVALIDATION_SUMMARY] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
