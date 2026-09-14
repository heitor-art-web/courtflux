from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

from tools.jersey_temporal_ocr_benchmark import (
    HF_LISTED_NUMBERS,
    _build_model,
    _device,
    _raw_prediction,
    _top_numbers,
    _transform,
    _uniform_indices,
)

P1_CLASSES = ("basketball", "player", "referee", "rim", "backboard")


def _class_name(model: YOLO, cls_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _canonical_class(name: str) -> str | None:
    value = str(name or "").strip().lower().replace("_", " ").replace("-", " ")
    if "basketball" in value or value in {"ball", "sports ball"}:
        return "basketball"
    if "referee" in value or value == "ref":
        return "referee"
    if "backboard" in value:
        return "backboard"
    if value == "rim" or "basket rim" in value:
        return "rim"
    if value in {"player", "basketball player", "person"}:
        return "player"
    return None


def _safe_crop(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def _torso_crop(player_crop: np.ndarray) -> np.ndarray | None:
    if player_crop is None or player_crop.size == 0:
        return None
    height, width = player_crop.shape[:2]
    y1 = int(round(height * 0.15))
    y2 = int(round(height * 0.65))
    x1 = int(round(width * 0.10))
    x2 = int(round(width * 0.90))
    crop = player_crop[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    return crop if crop.size else None


def _sequence_plan(sample_count: int) -> list[dict[str, Any]]:
    """Return up to three 8-frame sequence plans for one track.

    0-1 distinct samples: no prediction.
    2-7 samples: one repeated/resampled low-support sequence.
    >=8 samples: first/middle/last natural 8-frame windows.
    """
    if sample_count < 2:
        return []
    if sample_count < 8:
        return [
            {
                "support": "partial_repeated",
                "indices": _uniform_indices(sample_count, 8),
            }
        ]

    starts = sorted({0, max(0, (sample_count - 8) // 2), sample_count - 8})
    return [
        {
            "support": "full_8_distinct",
            "indices": list(range(start, start + 8)),
        }
        for start in starts
    ]


def _track_bucket(observations: int) -> str:
    if observations <= 4:
        return "short_1_4"
    if observations <= 14:
        return "medium_5_14"
    return "long_15_plus"


def _aggregate_track_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "prediction": None,
            "stable": None,
            "modal_fraction": None,
            "best_probability": None,
        }
    values = [str(row["prediction"]) for row in rows]
    counts = Counter(values)
    max_count = max(counts.values())
    tied = {value for value, count in counts.items() if count == max_count}
    best_row = max(
        (row for row in rows if str(row["prediction"]) in tied),
        key=lambda row: float(row["raw_probability"]),
    )
    prediction = str(best_row["prediction"])
    return {
        "prediction": prediction,
        "stable": len(counts) == 1 if len(rows) >= 2 else None,
        "modal_fraction": float(max_count / len(rows)),
        "best_probability": float(max(float(row["raw_probability"]) for row in rows)),
    }


def _predict_sequences(
    sequences: list[dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            batch = torch.stack([row["tensor"] for row in chunk]).to(device)
            tens_logits, ones_logits = model(batch)
            tens_probs = torch.softmax(tens_logits, dim=1).detach().cpu().numpy()
            ones_probs = torch.softmax(ones_logits, dim=1).detach().cpu().numpy()
            for index, source in enumerate(chunk):
                raw = _raw_prediction(tens_probs[index], ones_probs[index])
                top5 = _top_numbers(tens_probs[index], ones_probs[index], 5)
                outputs.append(
                    {
                        "track_id": int(source["track_id"]),
                        "sequence_index": int(source["sequence_index"]),
                        "support": source["support"],
                        "source_sample_count": int(source["source_sample_count"]),
                        "frame_ids": [int(value) for value in source["frame_ids"]],
                        **raw,
                        "top3_numbers": top5[:3],
                        "top5_numbers": top5,
                    }
                )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Initial no-GT Jersey validation using best_temporal.pt on current "
            "model_v1.1.0 + BoT-SORT player tracks. Reports operational support/stability only."
        )
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "OwiQgAMqyUs.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "model_v1.1.0.pt")
    parser.add_argument("--jersey-model", type=Path, default=ROOT / "weights" / "best_temporal.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--player-conf", type=float, default=0.40)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=120.0)
    parser.add_argument("--sample-every-sec", type=float, default=0.50)
    parser.add_argument("--max-samples-per-track", type=int, default=24)
    parser.add_argument("--crop", choices=["player", "torso"], default="player")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    detector_path = args.model.expanduser().resolve()
    jersey_path = args.jersey_model.expanduser().resolve()
    for path in (video_path, detector_path, jersey_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    if args.sample_every_sec <= 0:
        raise ValueError("--sample-every-sec must be > 0")

    detector = YOLO(str(detector_path))
    detector.to(args.device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(args.start * fps)))
    requested_end = int(round(args.end * fps))
    end_frame = min(total_frames - 1, requested_end) if total_frames > 0 else requested_end
    sample_gap = max(1, int(round(args.sample_every_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    class_counts: Counter[str] = Counter()
    track_observations: Counter[int] = Counter()
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    last_sample_frame: dict[int, int] = {}
    track_samples: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    invalid_crops = 0
    processed = 0
    frame_id = start_frame

    try:
        while frame_id <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            results = detector.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=float(args.player_conf),
                imgsz=int(args.imgsz),
                device=args.device,
                max_det=40,
                verbose=False,
            )
            result = results[0] if results else None
            processed += 1

            if result is not None and result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    canonical = _canonical_class(_class_name(detector, cls_id))
                    if canonical is not None:
                        class_counts[canonical] += 1
                    if canonical != "player" or box.id is None:
                        continue
                    track_id = int(box.id[0])
                    if track_id < 0:
                        continue
                    track_observations[track_id] += 1
                    first_frame.setdefault(track_id, frame_id)
                    last_frame[track_id] = frame_id

                    if len(track_samples[track_id]) >= max(1, int(args.max_samples_per_track)):
                        continue
                    previous = last_sample_frame.get(track_id)
                    if previous is not None and frame_id - previous < sample_gap:
                        continue

                    xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                    player_crop = _safe_crop(frame, xyxy)
                    crop = player_crop if args.crop == "player" else _torso_crop(player_crop)
                    if crop is None or crop.shape[0] < 16 or crop.shape[1] < 12:
                        invalid_crops += 1
                        continue
                    track_samples[track_id].append(
                        {
                            "frame_id": frame_id,
                            "timestamp_sec": frame_id / max(fps, 1e-6),
                            "crop": crop.copy(),
                        }
                    )
                    last_sample_frame[track_id] = frame_id

            if processed % 250 == 0 or frame_id == end_frame:
                counts = " ".join(f"{name}={class_counts.get(name, 0)}" for name in P1_CLASSES)
                print(
                    f"[JERSEY TEMPORAL] frame={frame_id}/{end_frame} processed={processed} "
                    f"detections[{counts} total={sum(class_counts.values())}] "
                    f"tracks={len(track_observations)} sampled_tracks={len(track_samples)}",
                    flush=True,
                )
            frame_id += 1
    finally:
        cap.release()

    device = _device(args.device)
    jersey_model, jersey_arch = _build_model(jersey_path, device)
    transform = _transform()

    sequences: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for track_id in sorted(track_observations):
        samples = track_samples.get(track_id, [])
        plans = _sequence_plan(len(samples))
        for sequence_index, plan in enumerate(plans):
            selected = [samples[index] for index in plan["indices"]]
            tensors = []
            for sample in selected:
                rgb = cv2.cvtColor(sample["crop"], cv2.COLOR_BGR2RGB)
                tensors.append(transform(Image.fromarray(rgb)))
            sequences.append(
                {
                    "track_id": track_id,
                    "sequence_index": sequence_index,
                    "support": plan["support"],
                    "source_sample_count": len(samples),
                    "frame_ids": [sample["frame_id"] for sample in selected],
                    "tensor": torch.stack(tensors),
                }
            )

    sequence_predictions = _predict_sequences(
        sequences,
        jersey_model,
        device,
        max(1, int(args.batch_size)),
    )
    by_track_predictions: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sequence_predictions:
        by_track_predictions[int(row["track_id"])].append(row)

    for track_id in sorted(track_observations):
        rows = by_track_predictions.get(track_id, [])
        samples = track_samples.get(track_id, [])
        aggregate = _aggregate_track_predictions(rows)
        support = (
            "full_8_distinct"
            if len(samples) >= 8
            else "partial_repeated"
            if len(samples) >= 2
            else "insufficient"
        )
        decisions.append(
            {
                "track_id": track_id,
                "observations": int(track_observations[track_id]),
                "track_bucket": _track_bucket(int(track_observations[track_id])),
                "first_frame": int(first_frame[track_id]),
                "last_frame": int(last_frame[track_id]),
                "sampled_frames": len(samples),
                "support": support,
                "sequence_predictions": len(rows),
                **aggregate,
                "prediction_in_hf_listed_subset": (
                    aggregate["prediction"] in HF_LISTED_NUMBERS
                    if aggregate["prediction"] is not None
                    else None
                ),
            }
        )

    total_tracks = len(decisions)
    predicted = [row for row in decisions if row["prediction"] is not None]
    full_support = [row for row in decisions if row["support"] == "full_8_distinct"]
    partial = [row for row in decisions if row["support"] == "partial_repeated"]
    insufficient = [row for row in decisions if row["support"] == "insufficient"]
    stability_rows = [row for row in decisions if row["stable"] is not None]
    modal_values = [float(row["modal_fraction"]) for row in predicted if row["modal_fraction"] is not None]
    probability_values = [float(row["best_probability"]) for row in predicted if row["best_probability"] is not None]
    listed_count = sum(bool(row["prediction_in_hf_listed_subset"]) for row in predicted)
    prediction_distribution = Counter(str(row["prediction"]) for row in predicted)

    buckets: dict[str, Any] = {}
    for bucket in ("short_1_4", "medium_5_14", "long_15_plus"):
        rows = [row for row in decisions if row["track_bucket"] == bucket]
        bucket_pred = [row for row in rows if row["prediction"] is not None]
        bucket_full = [row for row in rows if row["support"] == "full_8_distinct"]
        buckets[bucket] = {
            "tracks": len(rows),
            "predicted_tracks": len(bucket_pred),
            "full_support_tracks": len(bucket_full),
            "prediction_coverage": len(bucket_pred) / len(rows) if rows else None,
            "full_support_coverage": len(bucket_full) / len(rows) if rows else None,
        }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "evaluation" / "jersey_temporal_initial_v110" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "sequence_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in sequence_predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "track_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 Jersey initial operational validation with best_temporal.pt and no jersey GT",
        "evidence_status": "no_ground_truth_support_and_stability_only",
        "runtime": {
            "video": str(video_path),
            "detector": str(detector_path),
            "jersey_model": str(jersey_path),
            "tracker": args.tracker,
            "tracker_input_confidence": float(args.player_conf),
            "imgsz": int(args.imgsz),
            "device": args.device,
            "start_sec": float(args.start),
            "end_sec": float(args.end),
            "sample_every_sec": float(args.sample_every_sec),
            "max_samples_per_track": int(args.max_samples_per_track),
            "crop": args.crop,
            "sequence_frames": 8,
        },
        "jersey_checkpoint": jersey_arch,
        "detector_class_counts": {
            **{name: int(class_counts.get(name, 0)) for name in P1_CLASSES},
            "total": int(sum(class_counts.values())),
        },
        "tracking": {
            "processed_frames": processed,
            "tracked_player_ids": total_tracks,
            "invalid_jersey_crops": invalid_crops,
            "short_track_proxy_tracks": sum(row["track_bucket"] == "short_1_4" for row in decisions),
            "short_track_proxy_rate": (
                sum(row["track_bucket"] == "short_1_4" for row in decisions) / total_tracks
                if total_tracks
                else None
            ),
        },
        "jersey_operational": {
            "predicted_tracks": len(predicted),
            "unknown_tracks": total_tracks - len(predicted),
            "prediction_coverage": len(predicted) / total_tracks if total_tracks else None,
            "full_8_distinct_tracks": len(full_support),
            "full_support_coverage": len(full_support) / total_tracks if total_tracks else None,
            "partial_repeated_tracks": len(partial),
            "insufficient_tracks": len(insufficient),
            "tracks_with_multiple_sequence_predictions": len(stability_rows),
            "stable_multi_sequence_tracks": sum(bool(row["stable"]) for row in stability_rows),
            "multi_sequence_stability_rate": (
                sum(bool(row["stable"]) for row in stability_rows) / len(stability_rows)
                if stability_rows
                else None
            ),
            "mean_modal_fraction": mean(modal_values) if modal_values else None,
            "median_modal_fraction": median(modal_values) if modal_values else None,
            "mean_best_raw_probability": mean(probability_values) if probability_values else None,
            "median_best_raw_probability": median(probability_values) if probability_values else None,
            "predictions_in_hf_listed_subset": listed_count,
            "hf_listed_subset_fraction": listed_count / len(predicted) if predicted else None,
            "prediction_distribution": prediction_distribution.most_common(),
            "by_track_length": buckets,
        },
        "interpretation_guardrails": [
            "No human jersey ground truth is used; this run does not measure jersey-number accuracy or precision.",
            "A full-support prediction uses at least 8 distinct sampled crops from one current track.",
            "Tracks with 2-7 sampled crops are evaluated only as low-support repeated sequences and remain diagnostic.",
            "The best_temporal raw probability is not treated as calibrated confidence.",
            "Prediction concentration in the checkpoint's listed-number subset is diagnostic only, not proof of correctness.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"summary={summary_path}", flush=True)
    print(
        "[JERSEY TEMPORAL SUMMARY] "
        f"tracks={total_tracks} predicted={len(predicted)} full8={len(full_support)} "
        f"partial={len(partial)} insufficient={len(insufficient)} "
        f"stability={summary['jersey_operational']['multi_sequence_stability_rate']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
