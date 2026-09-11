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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct ball detection, TemporalBallTrackerV1, and "
            "TrajectoryAwareBallTrackerV1 on real annotated video."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", default="weights/model_v1.0.0.pt")
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


def _record(
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


def _delta(candidate: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    return {
        "visible_recall": candidate["visible_recall"] - direct["visible_recall"],
        "decision_precision": (
            candidate["decision_precision"] - direct["decision_precision"]
            if candidate["decision_precision"] is not None and direct["decision_precision"] is not None
            else None
        ),
        "output_coverage": candidate["output_coverage"] - direct["output_coverage"],
    }


def _transitions(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    recovered = sum(
        1
        for row in records
        if row["visible"] and not row["direct_correct"] and row[f"{key}_correct"]
    )
    regressed = sum(
        1
        for row in records
        if row["visible"] and row["direct_correct"] and not row[f"{key}_correct"]
    )
    return {"recovered_visible_frames": recovered, "regressed_visible_frames": regressed}


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    for path in (video_path, annotations_path, model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    annotations = load_annotations(annotations_path)
    first_frame, last_frame = min(annotations), max(annotations)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("evaluation") / "ball_trajectory_v1_real" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    temporal_tracker = TemporalBallTrackerV1(**tracker_kwargs)
    trajectory_tracker = TrajectoryAwareBallTrackerV1(**tracker_kwargs)

    direct_records: list[dict[str, Any]] = []
    temporal_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    temporal_sources: Counter[str] = Counter()
    trajectory_sources: Counter[str] = Counter()

    processed = 0
    total_frames = last_frame - first_frame + 1

    for frame_id in range(first_frame, last_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")

        candidates = extract_ball_candidates(model, frame, args)
        direct = max(candidates, key=lambda item: item["confidence"]) if candidates else None
        temporal_decision = temporal_tracker.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
        trajectory_decision = trajectory_tracker.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
        temporal_sources[temporal_decision.source] += 1
        trajectory_sources[trajectory_decision.source] += 1

        if frame_id in annotations:
            annotation = annotations[frame_id]
            timestamp_sec = frame_id / fps if fps > 0 else annotation.get("timestamp_sec")
            direct_row = _record(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                annotation=annotation,
                key="direct",
                prediction=direct,
                source="highest_confidence_direct",
                point_margin_px=args.point_margin_px,
            )
            temporal_row = _record(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                annotation=annotation,
                key="temporal",
                prediction=temporal_decision.ball,
                source=temporal_decision.source,
                point_margin_px=args.point_margin_px,
            )
            trajectory_row = _record(
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

        processed += 1
        if processed == 1 or processed % 100 == 0 or frame_id == last_frame:
            print(
                f"[{processed}/{total_frames}] {100.0 * processed / max(total_frames, 1):.1f}% "
                f"frame={frame_id} candidates={len(candidates)} "
                f"temporal={temporal_decision.source} trajectory={trajectory_decision.source}",
                flush=True,
            )

    capture.release()

    direct_summary = summarize(direct_records, "direct")
    temporal_summary = summarize(temporal_records, "temporal")
    trajectory_summary = summarize(trajectory_records, "trajectory")

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 Ball Trajectory Consistency v1 real-video comparison",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "model": str(model_path),
        "device": args.device,
        "imgsz": args.imgsz,
        "ball_conf": args.ball_conf,
        "frame_range": [first_frame, last_frame],
        "processed_frames": processed,
        "annotated_frames": len(direct_records),
        "direct_baseline": direct_summary,
        "temporal_tracker": temporal_summary,
        "trajectory_tracker": trajectory_summary,
        "delta_vs_direct": {
            "temporal": {**_delta(temporal_summary, direct_summary), **_transitions(temporal_records, "temporal")},
            "trajectory": {**_delta(trajectory_summary, direct_summary), **_transitions(trajectory_records, "trajectory")},
        },
        "delta_trajectory_vs_temporal": {
            "visible_recall": trajectory_summary["visible_recall"] - temporal_summary["visible_recall"],
            "decision_precision": (
                trajectory_summary["decision_precision"] - temporal_summary["decision_precision"]
                if trajectory_summary["decision_precision"] is not None and temporal_summary["decision_precision"] is not None
                else None
            ),
            "output_coverage": trajectory_summary["output_coverage"] - temporal_summary["output_coverage"],
        },
        "source_counts_all_processed_frames": {
            "temporal": dict(temporal_sources),
            "trajectory": dict(trajectory_sources),
        },
        "metadata": {
            "temporal": temporal_tracker.metadata(),
            "trajectory": trajectory_tracker.metadata(),
        },
        "notes": [
            "All trackers consume the exact same detector candidates from one detector pass per frame.",
            "Metrics are computed only on annotated frames while state is updated on every frame in the GT span.",
            "Direct baseline is the highest-confidence detector candidate at the configured ball_conf.",
            "Point truth uses the existing box-plus-margin contract; bbox truth uses IoU >= 0.25.",
            "Promotion should consider both direct-baseline quality and improvement over TemporalBallTrackerV1.",
        ],
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "trajectory_annotated_frames.json").write_text(
        json.dumps(trajectory_records, indent=2), encoding="utf-8"
    )

    print("\nvariant | recall | precision | coverage | wrong-visible | fp-invisible")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for name, metrics in (
        ("direct", direct_summary),
        ("temporal", temporal_summary),
        ("trajectory", trajectory_summary),
    ):
        precision = metrics["decision_precision"]
        precision_text = f"{precision:.3f}" if precision is not None else "n/a"
        print(
            f"{name} | {metrics['visible_recall']:.3f} | {precision_text} | "
            f"{metrics['output_coverage']:.3f} | {metrics['wrong_visible_outputs']} | "
            f"{metrics['false_positive_invisible_frames']}"
        )

    print(f"\nWrote: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
