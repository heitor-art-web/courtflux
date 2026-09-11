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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep conservative Ball Trajectory Consistency v1 gate/override policies "
            "using one detector pass."
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


def gate_presets() -> list[dict[str, Any]]:
    return [
        {"name": "loose", "distance": 0.16, "speed": 0.12, "acceleration": 0.14},
        {"name": "medium", "distance": 0.10, "speed": 0.08, "acceleration": 0.08},
        {"name": "tight", "distance": 0.06, "speed": 0.05, "acceleration": 0.05},
        {"name": "very_tight", "distance": 0.04, "speed": 0.035, "acceleration": 0.035},
    ]


def policy_presets() -> list[dict[str, Any]]:
    return [
        {"name": "legacy", "min_score": 0.0, "protect_conf": None, "margin": 0.0},
        {"name": "score_070", "min_score": 0.70, "protect_conf": None, "margin": 0.0},
        {"name": "protect_075", "min_score": 0.0, "protect_conf": 0.75, "margin": 0.0},
        {"name": "margin_005", "min_score": 0.0, "protect_conf": None, "margin": 0.05},
        {"name": "protect_075_margin_005", "min_score": 0.0, "protect_conf": 0.75, "margin": 0.05},
        {"name": "score_070_protect_075_margin_005", "min_score": 0.70, "protect_conf": 0.75, "margin": 0.05},
    ]


def record(
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
        else Path("evaluation") / "ball_trajectory_policy_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_kwargs = {
        "max_prediction_gap": args.max_prediction_gap,
        "association_gate_norm": args.association_gate_norm,
        "gate_growth_per_frame": args.gate_growth_per_frame,
        "max_gate_norm": args.max_gate_norm,
        "velocity_alpha": args.velocity_alpha,
    }

    configurations: list[dict[str, Any]] = []
    trackers: dict[str, TrajectoryAwareBallTrackerV1] = {}
    for gate in gate_presets():
        for policy in policy_presets():
            config_id = f"{gate['name']}__{policy['name']}"
            trajectory = BallTrajectoryConsistencyV1(
                max_distance_norm=gate["distance"],
                max_speed_residual_norm=gate["speed"],
                max_acceleration_norm=gate["acceleration"],
            )
            trackers[config_id] = TrajectoryAwareBallTrackerV1(
                trajectory=trajectory,
                min_trajectory_score=policy["min_score"],
                protect_direct_confidence=policy["protect_conf"],
                override_score_margin=policy["margin"],
                **tracker_kwargs,
            )
            configurations.append({"id": config_id, "gate": gate, "policy": policy})

    temporal_tracker = TemporalBallTrackerV1(**tracker_kwargs)
    records_by_config: dict[str, list[dict[str, Any]]] = {row["id"]: [] for row in configurations}
    sources_by_config: dict[str, Counter[str]] = {row["id"]: Counter() for row in configurations}
    direct_records: list[dict[str, Any]] = []
    temporal_records: list[dict[str, Any]] = []
    temporal_sources: Counter[str] = Counter()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    model = YOLO(str(model_path))

    processed = 0
    total_frames = last_frame - first_frame + 1
    for frame_id in range(first_frame, last_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")

        candidates = extract_ball_candidates(model, frame, args)
        direct = max(candidates, key=lambda item: item["confidence"]) if candidates else None
        temporal_decision = temporal_tracker.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
        temporal_sources[temporal_decision.source] += 1
        decisions = {
            config_id: tracker.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
            for config_id, tracker in trackers.items()
        }
        for config_id, decision in decisions.items():
            sources_by_config[config_id][decision.source] += 1

        if frame_id in annotations:
            annotation = annotations[frame_id]
            timestamp_sec = frame_id / fps if fps > 0 else annotation.get("timestamp_sec")
            direct_row = record(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                annotation=annotation,
                key="direct",
                prediction=direct,
                source="highest_confidence_direct",
                point_margin_px=args.point_margin_px,
            )
            direct_records.append(direct_row)
            temporal_row = record(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                annotation=annotation,
                key="temporal",
                prediction=temporal_decision.ball,
                source=temporal_decision.source,
                point_margin_px=args.point_margin_px,
            )
            temporal_records.append({**direct_row, **temporal_row})

            for config_id, decision in decisions.items():
                candidate_row = record(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    annotation=annotation,
                    key="trajectory",
                    prediction=decision.ball,
                    source=decision.source,
                    point_margin_px=args.point_margin_px,
                )
                records_by_config[config_id].append({**direct_row, **candidate_row})

        processed += 1
        if processed == 1 or processed % 100 == 0 or frame_id == last_frame:
            print(
                f"[{processed}/{total_frames}] {100.0 * processed / max(total_frames, 1):.1f}% "
                f"frame={frame_id} candidates={len(candidates)} configs={len(configurations)}",
                flush=True,
            )

    capture.release()

    direct_summary = summarize(direct_records, "direct")
    temporal_summary = summarize(temporal_records, "temporal")
    results: list[dict[str, Any]] = []
    for config in configurations:
        config_id = config["id"]
        records = records_by_config[config_id]
        metrics = summarize(records, "trajectory")
        change = transitions(records, "trajectory")
        results.append(
            {
                **config,
                "metrics": metrics,
                "delta_vs_direct": {
                    "visible_recall": metrics["visible_recall"] - direct_summary["visible_recall"],
                    "decision_precision": metrics["decision_precision"] - direct_summary["decision_precision"],
                    "output_coverage": metrics["output_coverage"] - direct_summary["output_coverage"],
                    **change,
                },
                "source_counts_all_processed_frames": dict(sources_by_config[config_id]),
            }
        )

    ranked = sorted(
        results,
        key=lambda row: (
            row["metrics"]["decision_precision"],
            row["metrics"]["visible_recall"],
            -row["delta_vs_direct"]["regressed_visible_frames"],
        ),
        reverse=True,
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 trajectory gate/override policy sweep using one detector pass",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "model": str(model_path),
        "device": args.device,
        "imgsz": args.imgsz,
        "ball_conf": args.ball_conf,
        "frame_range": [first_frame, last_frame],
        "processed_frames": processed,
        "annotated_frames": len(direct_records),
        "configuration_count": len(configurations),
        "direct_baseline": direct_summary,
        "temporal_baseline": temporal_summary,
        "temporal_source_counts": dict(temporal_sources),
        "ranked_results": ranked,
        "best_by_precision_then_recall": ranked[0]["id"] if ranked else None,
        "notes": [
            "YOLO inference is executed once per frame and all trajectory policies consume identical candidates.",
            "This is tuning evidence on the current Porto-Ovarense GT, not independent held-out validation.",
            "Promotion still requires comparison against the direct detector and validation on additional data.",
            "The sweep separates trajectory gates from promotion policy: minimum score, direct-confidence protection, and override score margin.",
        ],
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTop trajectory policies")
    print("rank | config | recall | precision | coverage | recovered | regressed")
    print("---: | --- | ---: | ---: | ---: | ---: | ---:")
    for rank, row in enumerate(ranked[:10], start=1):
        m = row["metrics"]
        d = row["delta_vs_direct"]
        print(
            f"{rank} | {row['id']} | {m['visible_recall']:.3f} | "
            f"{m['decision_precision']:.3f} | {m['output_coverage']:.3f} | "
            f"{d['recovered_visible_frames']} | {d['regressed_visible_frames']}"
        )

    print("\nBaselines")
    print(
        f"direct   recall={direct_summary['visible_recall']:.3f} "
        f"precision={direct_summary['decision_precision']:.3f} coverage={direct_summary['output_coverage']:.3f}"
    )
    print(
        f"temporal recall={temporal_summary['visible_recall']:.3f} "
        f"precision={temporal_summary['decision_precision']:.3f} coverage={temporal_summary['output_coverage']:.3f}"
    )
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
