from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jersey_revalidation_v110 import (
    FROZEN_POLICY,
    P1_CLASSES,
    _canonical_class,
    _class_name,
    _infer_jerseys,
    _safe_crop,
    _torso_crop,
)
from tools.jersey_uncertainty_track_policy_benchmark import _group_prediction


def _track_bucket(observations: int) -> str:
    if observations <= 4:
        return "short_1_4"
    if observations <= 14:
        return "medium_5_14"
    return "long_15_plus"


def _policy_eligible_count(frame: pd.DataFrame) -> int:
    eligible = frame
    if FROZEN_POLICY["require_not_absent"]:
        eligible = eligible[~eligible["absent_argmax"].astype(bool)]
    if FROZEN_POLICY["uncertainty_max"] is not None:
        eligible = eligible[
            eligible["uncertainty"].astype(float) <= float(FROZEN_POLICY["uncertainty_max"])
        ]
    if FROZEN_POLICY["pred_score_min"] is not None:
        eligible = eligible[
            eligible["pred_score"].astype(float) >= float(FROZEN_POLICY["pred_score_min"])
        ]
    return int(len(eligible))


def _build_track_decisions(
    predictions: list[dict[str, Any]],
    track_observations: dict[int, int],
) -> list[dict[str, Any]]:
    by_track: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_track[int(row["track_id"])].append(row)

    decisions: list[dict[str, Any]] = []
    for track_id in sorted(track_observations):
        rows = by_track.get(track_id, [])
        observations = int(track_observations[track_id])
        bucket = _track_bucket(observations)
        if not rows:
            decisions.append(
                {
                    "track_id": track_id,
                    "observations": observations,
                    "track_bucket": bucket,
                    "sampled_frames": 0,
                    "policy_eligible_frames": 0,
                    "accepted": False,
                    "prediction": None,
                    "unknown_reason": "no_valid_jersey_sample",
                    "unique_frame_predictions": 0,
                    "modal_frame_prediction": None,
                    "modal_frame_fraction": None,
                    "mean_uncertainty": None,
                    "median_uncertainty": None,
                    "best_uncertainty": None,
                }
            )
            continue

        frame = pd.DataFrame(rows)
        result = _group_prediction({"frame": frame}, FROZEN_POLICY)
        values = [int(value) for value in frame["pred_number"].tolist()]
        counts = Counter(values)
        modal_number, modal_count = counts.most_common(1)[0]
        uncertainties = [float(value) for value in frame["uncertainty"].tolist()]
        eligible_count = _policy_eligible_count(frame)
        accepted = bool(result.get("accepted"))

        decisions.append(
            {
                "track_id": track_id,
                "observations": observations,
                "track_bucket": bucket,
                "sampled_frames": int(len(frame)),
                "policy_eligible_frames": eligible_count,
                "accepted": accepted,
                "prediction": int(result["prediction"]) if accepted else None,
                "unknown_reason": None if accepted else "frozen_policy_abstention",
                "eligible_frames": int(result.get("eligible_frames", eligible_count)),
                "agg_margin": (
                    float(result["agg_margin"]) if result.get("agg_margin") is not None else None
                ),
                "vote_fraction": (
                    float(result["vote_fraction"]) if result.get("vote_fraction") is not None else None
                ),
                "unique_frame_predictions": int(len(counts)),
                "modal_frame_prediction": int(modal_number),
                "modal_frame_fraction": float(modal_count / len(values)),
                "mean_uncertainty": float(mean(uncertainties)),
                "median_uncertainty": float(median(uncertainties)),
                "best_uncertainty": float(min(uncertainties)),
            }
        )
    return decisions


def _summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(decisions)
    accepted = [row for row in decisions if row["accepted"]]
    sampled = [row for row in decisions if row["sampled_frames"] > 0]
    evaluable = [
        row
        for row in decisions
        if row["policy_eligible_frames"] >= int(FROZEN_POLICY["min_frames"])
    ]
    short = [row for row in decisions if row["track_bucket"] == "short_1_4"]

    buckets: dict[str, Any] = {}
    for bucket in ("short_1_4", "medium_5_14", "long_15_plus"):
        rows = [row for row in decisions if row["track_bucket"] == bucket]
        accepted_rows = [row for row in rows if row["accepted"]]
        buckets[bucket] = {
            "tracks": len(rows),
            "accepted": len(accepted_rows),
            "unknown": len(rows) - len(accepted_rows),
            "acceptance_rate": len(accepted_rows) / len(rows) if rows else None,
            "mean_sampled_frames": (
                mean(float(row["sampled_frames"]) for row in rows) if rows else None
            ),
        }

    vote_values = [
        float(row["vote_fraction"])
        for row in accepted
        if row.get("vote_fraction") is not None
    ]
    modal_values = [
        float(row["modal_frame_fraction"])
        for row in accepted
        if row.get("modal_frame_fraction") is not None
    ]
    uncertainty_values = [
        float(row["best_uncertainty"])
        for row in accepted
        if row.get("best_uncertainty") is not None
    ]

    return {
        "tracked_player_ids": total,
        "sampled_tracks": len(sampled),
        "policy_evaluable_tracks": len(evaluable),
        "accepted_tracks": len(accepted),
        "unknown_tracks": total - len(accepted),
        "operational_acceptance_rate": len(accepted) / total if total else None,
        "operational_unknown_rate": (total - len(accepted)) / total if total else None,
        "acceptance_rate_among_sampled": len(accepted) / len(sampled) if sampled else None,
        "short_track_proxy": {
            "tracks": len(short),
            "rate": len(short) / total if total else None,
            "definition": "tracked player ids with 1-4 detector observations; proxy only, not physical-player fragmentation GT",
        },
        "accepted_track_stability": {
            "mean_policy_vote_fraction": mean(vote_values) if vote_values else None,
            "median_policy_vote_fraction": median(vote_values) if vote_values else None,
            "mean_modal_frame_fraction": mean(modal_values) if modal_values else None,
            "median_modal_frame_fraction": median(modal_values) if modal_values else None,
            "tracks_modal_fraction_ge_0_50": sum(value >= 0.50 for value in modal_values),
            "tracks_modal_fraction_ge_0_75": sum(value >= 0.75 for value in modal_values),
            "mean_best_uncertainty": mean(uncertainty_values) if uncertainty_values else None,
        },
        "by_track_length": buckets,
    }


def _serialize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_number_probs"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Initial no-ground-truth Jersey Identification validation for the current P1 "
            "v1.1.0 + BoT-SORT player operating point. Reports coverage/stability only."
        )
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
    parser.add_argument("--player-conf", type=float, default=0.40)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=120.0)
    parser.add_argument("--sample-every-sec", type=float, default=1.0)
    parser.add_argument("--max-samples-per-track", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    detector_path = args.model.expanduser().resolve()
    jersey_model_path = args.jersey_model.expanduser().resolve()
    for path in (video_path, detector_path, jersey_model_path):
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
    requested_end_frame = int(round(args.end * fps))
    end_frame = min(total_frames - 1, requested_end_frame) if total_frames > 0 else requested_end_frame
    sample_gap_frames = max(1, int(round(args.sample_every_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    class_counts: Counter[str] = Counter()
    track_observations: Counter[int] = Counter()
    track_first_frame: dict[int, int] = {}
    track_last_frame: dict[int, int] = {}
    sampled_per_track: Counter[int] = Counter()
    last_sample_frame: dict[int, int] = {}
    samples: list[dict[str, Any]] = []
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
                    confidence = float(box.conf[0])
                    track_observations[track_id] += 1
                    track_first_frame.setdefault(track_id, frame_id)
                    track_last_frame[track_id] = frame_id

                    if sampled_per_track[track_id] >= max(1, int(args.max_samples_per_track)):
                        continue
                    previous = last_sample_frame.get(track_id)
                    if previous is not None and frame_id - previous < sample_gap_frames:
                        continue

                    player_box = tuple(float(v) for v in box.xyxy[0].tolist())
                    player_crop = _safe_crop(frame, player_box)
                    torso = _torso_crop(player_crop) if player_crop is not None else None
                    if torso is None or torso.shape[0] < 16 or torso.shape[1] < 12:
                        invalid_crops += 1
                        continue

                    samples.append(
                        {
                            "track_id": track_id,
                            "frame_id": frame_id,
                            "timestamp_sec": frame_id / max(fps, 1e-6),
                            "player_confidence": confidence,
                            "player_box": [float(v) for v in player_box],
                            "_torso_crop": torso,
                        }
                    )
                    sampled_per_track[track_id] += 1
                    last_sample_frame[track_id] = frame_id

            if processed % 250 == 0 or frame_id == end_frame:
                counts_text = " ".join(
                    f"{name}={class_counts.get(name, 0)}" for name in P1_CLASSES
                )
                print(
                    f"[JERSEY INITIAL] frame={frame_id}/{end_frame} processed={processed} "
                    f"detections[{counts_text} total={sum(class_counts.values())}] "
                    f"tracks={len(track_observations)} samples={len(samples)}",
                    flush=True,
                )
            frame_id += 1
    finally:
        cap.release()

    predictions, jersey_runtime = _infer_jerseys(
        samples,
        jersey_model_path,
        args.device,
        max(1, int(args.batch_size)),
    )
    decisions = _build_track_decisions(predictions, dict(track_observations))
    decision_summary = _summarize_decisions(decisions)

    accepted_numbers = Counter(
        int(row["prediction"])
        for row in decisions
        if row["accepted"] and row["prediction"] is not None
    )
    track_lifetimes = {
        str(track_id): {
            "first_frame": int(track_first_frame[track_id]),
            "last_frame": int(track_last_frame[track_id]),
            "observations": int(track_observations[track_id]),
            "span_frames": int(track_last_frame[track_id] - track_first_frame[track_id] + 1),
        }
        for track_id in track_observations
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "evaluation" / "jersey_initial_validation_v110" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "sample_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(_serialize_prediction(row), ensure_ascii=False) + "\n")
    with (output_dir / "track_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 Jersey Identification initial operational validation without jersey ground truth",
        "evidence_status": "no_ground_truth_operational_coverage_and_stability_only",
        "runtime": {
            "video": str(video_path),
            "detector": str(detector_path),
            "tracker": args.tracker,
            "imgsz": int(args.imgsz),
            "tracker_input_confidence": float(args.player_conf),
            "start_sec": float(args.start),
            "end_sec": float(args.end),
            "fps": fps,
            "sample_every_sec": float(args.sample_every_sec),
            "max_samples_per_track": int(args.max_samples_per_track),
            "jersey_model": str(jersey_model_path),
            "jersey_device": jersey_runtime["device"],
            "jersey_checkpoint_coverage": jersey_runtime["checkpoint_coverage"],
            "frozen_policy": FROZEN_POLICY,
        },
        "detector_diagnostics": {
            "frames_processed": processed,
            "frame_range": [start_frame, max(start_frame, frame_id - 1)],
            "class_counts": {name: int(class_counts.get(name, 0)) for name in P1_CLASSES},
            "all_p1_class_detections": int(sum(class_counts.values())),
        },
        "tracking": {
            "track_observations": int(sum(track_observations.values())),
            "unique_player_track_ids": int(len(track_observations)),
            "invalid_jersey_crops": int(invalid_crops),
            "track_lifetimes": track_lifetimes,
        },
        "jersey": {
            "sampled_crops": int(len(predictions)),
            "accepted_number_distribution": {
                str(number): int(count) for number, count in sorted(accepted_numbers.items())
            },
            **decision_summary,
        },
        "scope_guard": {
            "ground_truth_used": False,
            "jersey_policy_retuned": False,
            "player_tracker_retuned": False,
            "accuracy_claim": False,
            "selective_precision_claim": False,
            "identity_consolidation_performed": False,
            "interpretation": (
                "Accepted jersey numbers are operational hypotheses/evidence only. This run measures "
                "coverage, abstention and temporal stability; it cannot measure correctness without human GT."
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "[JERSEY_INITIAL_SUMMARY] "
        f"tracks={decision_summary['tracked_player_ids']} "
        f"accepted={decision_summary['accepted_tracks']} "
        f"unknown={decision_summary['unknown_tracks']} "
        f"acceptance={decision_summary['operational_acceptance_rate']} "
        f"short_track_rate={decision_summary['short_track_proxy']['rate']}",
        flush=True,
    )
    print(f"[JERSEY_INITIAL_SUMMARY] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
