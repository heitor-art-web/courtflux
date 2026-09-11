from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from evaluation.ball_temporal_v1_real_eval import (
    extract_ball_candidates,
    load_annotations,
    matches_truth,
    point_error,
    summarize,
)
from services.vision.ball_hybrid_v1 import TrajectoryAwareBallTrackerV1
from services.vision.ball_temporal_v1 import TemporalBallTrackerV1
from services.vision.ball_trajectory_consistency import BallTrajectoryConsistencyV1


TRAJECTORY_POLICY = {
    "id": "medium__score_070",
    "gate": {"distance": 0.10, "speed": 0.08, "acceleration": 0.08},
    "policy": {"min_score": 0.70, "protect_conf": None, "margin": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model_v1.0.0 and model_v1.1.0 across direct, temporal, and frozen trajectory policy."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model-v100", default="weights/model_v1.0.0.pt")
    parser.add_argument("--model-v110", default="weights/model_v1.1.0.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--ball-conf", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--point-margin-px", type=float, default=4.0)
    parser.add_argument("--max-prediction-gap", type=int, default=4)
    parser.add_argument("--association-gate-norm", type=float, default=0.12)
    parser.add_argument("--gate-growth-per-frame", type=float, default=0.02)
    parser.add_argument("--max-gate-norm", type=float, default=0.24)
    parser.add_argument("--velocity-alpha", type=float, default=0.45)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def make_record(
    *,
    frame_id: int,
    timestamp_sec: float | None,
    annotation: dict[str, Any],
    key: str,
    prediction: dict[str, Any] | None,
    source: str,
    point_margin_px: float,
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp_sec,
        "visible": bool(annotation["visible"]),
        key: prediction,
        f"{key}_correct": matches_truth(prediction, annotation, point_margin_px),
        f"{key}_point_error_px": point_error(prediction, annotation),
        f"{key}_source": source,
    }


def transitions(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        "recovered_visible_frames": sum(
            1
            for row in records
            if row["visible"] and not row["direct_correct"] and row[f"{key}_correct"]
        ),
        "regressed_visible_frames": sum(
            1
            for row in records
            if row["visible"] and row["direct_correct"] and not row[f"{key}_correct"]
        ),
    }


def run_model(
    *,
    label: str,
    model_path: Path,
    video_path: Path,
    annotations: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    first_frame, last_frame = min(annotations), max(annotations)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    model = YOLO(str(model_path))
    tracker_kwargs = {
        "max_prediction_gap": args.max_prediction_gap,
        "association_gate_norm": args.association_gate_norm,
        "gate_growth_per_frame": args.gate_growth_per_frame,
        "max_gate_norm": args.max_gate_norm,
        "velocity_alpha": args.velocity_alpha,
    }
    temporal = TemporalBallTrackerV1(**tracker_kwargs)
    trajectory = TrajectoryAwareBallTrackerV1(
        trajectory=BallTrajectoryConsistencyV1(
            max_distance_norm=TRAJECTORY_POLICY["gate"]["distance"],
            max_speed_residual_norm=TRAJECTORY_POLICY["gate"]["speed"],
            max_acceleration_norm=TRAJECTORY_POLICY["gate"]["acceleration"],
        ),
        min_trajectory_score=TRAJECTORY_POLICY["policy"]["min_score"],
        protect_direct_confidence=TRAJECTORY_POLICY["policy"]["protect_conf"],
        override_score_margin=TRAJECTORY_POLICY["policy"]["margin"],
        **tracker_kwargs,
    )

    direct_records: list[dict[str, Any]] = []
    temporal_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    temporal_sources: Counter[str] = Counter()
    trajectory_sources: Counter[str] = Counter()
    candidate_count_total = 0
    frames_with_candidates = 0

    total_frames = last_frame - first_frame + 1
    for offset, frame_id in enumerate(range(first_frame, last_frame + 1), start=1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")

        candidates = extract_ball_candidates(model, frame, args)
        candidate_count_total += len(candidates)
        if candidates:
            frames_with_candidates += 1
        direct = max(candidates, key=lambda item: item["confidence"]) if candidates else None
        temporal_decision = temporal.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
        trajectory_decision = trajectory.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
        temporal_sources[temporal_decision.source] += 1
        trajectory_sources[trajectory_decision.source] += 1

        if frame_id in annotations:
            annotation = annotations[frame_id]
            timestamp_sec = frame_id / fps if fps > 0 else annotation.get("timestamp_sec")
            direct_row = make_record(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                annotation=annotation,
                key="direct",
                prediction=direct,
                source="highest_confidence_direct",
                point_margin_px=args.point_margin_px,
            )
            temporal_row = make_record(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                annotation=annotation,
                key="temporal",
                prediction=temporal_decision.ball,
                source=temporal_decision.source,
                point_margin_px=args.point_margin_px,
            )
            trajectory_row = make_record(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                annotation=annotation,
                key="trajectory",
                prediction=trajectory_decision.ball,
                source=trajectory_decision.source,
                point_margin_px=args.point_margin_px,
            )
            direct_records.append(direct_row)
            temporal_records.append({**direct_row, **temporal_row})
            trajectory_records.append({**direct_row, **trajectory_row})

        if offset == 1 or offset % 100 == 0 or frame_id == last_frame:
            print(
                f"[{label}] [{offset}/{total_frames}] "
                f"{100.0 * offset / max(total_frames, 1):.1f}% frame={frame_id} candidates={len(candidates)}",
                flush=True,
            )

    capture.release()

    direct_summary = summarize(direct_records, "direct")
    temporal_summary = summarize(temporal_records, "temporal")
    trajectory_summary = summarize(trajectory_records, "trajectory")

    return {
        "label": label,
        "model": str(model_path),
        "processed_frames": total_frames,
        "annotated_frames": len(direct_records),
        "candidate_diagnostics": {
            "frames_with_candidates": frames_with_candidates,
            "candidate_frame_coverage": frames_with_candidates / total_frames if total_frames else 0.0,
            "mean_candidates_per_processed_frame": candidate_count_total / total_frames if total_frames else 0.0,
        },
        "direct": direct_summary,
        "temporal": temporal_summary,
        "trajectory": trajectory_summary,
        "transitions_vs_direct": {
            "temporal": transitions(temporal_records, "temporal"),
            "trajectory": transitions(trajectory_records, "trajectory"),
        },
        "source_counts": {
            "temporal": dict(temporal_sources),
            "trajectory": dict(trajectory_sources),
        },
        "trajectory_metadata": trajectory.metadata(),
    }


def metric_delta(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "visible_recall",
        "decision_precision",
        "output_coverage",
        "mean_point_error_px_on_correct",
    ]
    result: dict[str, Any] = {}
    for field in fields:
        a, b = new.get(field), old.get(field)
        result[field] = a - b if a is not None and b is not None else None
    result["correct_visible_frames"] = new["correct_visible_frames"] - old["correct_visible_frames"]
    result["wrong_visible_outputs"] = new["wrong_visible_outputs"] - old["wrong_visible_outputs"]
    result["false_positive_invisible_frames"] = (
        new["false_positive_invisible_frames"] - old["false_positive_invisible_frames"]
    )
    return result


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    model_v100 = Path(args.model_v100).expanduser().resolve()
    model_v110 = Path(args.model_v110).expanduser().resolve()
    for path in (video_path, annotations_path, model_v100, model_v110):
        if not path.exists():
            raise FileNotFoundError(path)

    annotations = load_annotations(annotations_path)
    first_frame, last_frame = min(annotations), max(annotations)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("evaluation") / "ball_model_v110_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_v100 = run_model(
        label="model_v1.0.0",
        model_path=model_v100,
        video_path=video_path,
        annotations=annotations,
        args=args,
    )
    results_v110 = run_model(
        label="model_v1.1.0",
        model_path=model_v110,
        video_path=video_path,
        annotations=annotations,
        args=args,
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "Compare frozen P1 ball policies across model_v1.0.0 and model_v1.1.0",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "frame_range": [first_frame, last_frame],
        "device": args.device,
        "imgsz": args.imgsz,
        "ball_conf": args.ball_conf,
        "trajectory_policy": TRAJECTORY_POLICY,
        "models": {
            "model_v1.0.0": results_v100,
            "model_v1.1.0": results_v110,
        },
        "delta_v110_minus_v100": {
            variant: metric_delta(results_v110[variant], results_v100[variant])
            for variant in ("direct", "temporal", "trajectory")
        },
        "notes": [
            "Both models use identical video, GT, imgsz, ball_conf, NMS, and tracker settings.",
            "Trajectory policy is frozen to medium__score_070 selected on model_v1.0.0 tuning evidence; no retuning is performed on model_v1.1.0.",
            "The model comparison measures detector impact separately from tracker impact.",
            "This Porto-Ovarense segment remains tuning/validation evidence, not independent held-out promotion evidence.",
        ],
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nmodel | variant | recall | precision | coverage | correct | wrong-visible | fp-invisible")
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---:")
    for model_label, model_result in (
        ("v1.0.0", results_v100),
        ("v1.1.0", results_v110),
    ):
        for variant in ("direct", "temporal", "trajectory"):
            m = model_result[variant]
            p = m["decision_precision"]
            p_text = f"{p:.3f}" if p is not None else "n/a"
            print(
                f"{model_label} | {variant} | {m['visible_recall']:.3f} | {p_text} | "
                f"{m['output_coverage']:.3f} | {m['correct_visible_frames']} | "
                f"{m['wrong_visible_outputs']} | {m['false_positive_invisible_frames']}"
            )

    print("\nv1.1.0 - v1.0.0 deltas")
    for variant, delta in summary["delta_v110_minus_v100"].items():
        print(
            f"{variant}: recall={delta['visible_recall']:+.3f} "
            f"precision={delta['decision_precision']:+.3f} "
            f"coverage={delta['output_coverage']:+.3f}"
        )
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
