from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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


FROZEN_V100_TRAJECTORY = {
    "gate": {"name": "medium", "distance": 0.10, "speed": 0.08, "acceleration": 0.08},
    "policy": {"name": "score_070", "min_score": 0.70, "protect_conf": None, "margin": 0.0},
    "max_prediction_gap": 4,
}


def parse_csv_floats(value: str) -> list[float]:
    values = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate model_v1.0.0 vs model_v1.1.0 ball confidence, then run a small "
            "v1.1.0 tracker-policy sweep without repeating detector inference."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model-v100", default="weights/model_v1.0.0.pt")
    parser.add_argument("--model-v110", default="weights/model_v1.1.0.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--thresholds",
        default="0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20",
        help="Ball confidence thresholds replayed from one low-confidence detector pass.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--point-margin-px", type=float, default=4.0)
    parser.add_argument("--association-gate-norm", type=float, default=0.12)
    parser.add_argument("--gate-growth-per-frame", type=float, default=0.02)
    parser.add_argument("--max-gate-norm", type=float, default=0.24)
    parser.add_argument("--velocity-alpha", type=float, default=0.45)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def clone_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        if isinstance(row.get("box"), list):
            row["box"] = list(row["box"])
        if isinstance(row.get("center"), list):
            row["center"] = list(row["center"])
        cloned.append(row)
    return cloned


def filter_candidates(candidates: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if float(candidate.get("confidence", 0.0)) >= threshold]


def f1(metrics: dict[str, Any]) -> float:
    precision = metrics.get("decision_precision")
    recall = metrics.get("visible_recall")
    if precision is None or recall is None or precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def metric_pack(metrics: dict[str, Any]) -> dict[str, Any]:
    return {**metrics, "f1_recall_precision": f1(metrics)}


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


def tracker_kwargs(args: argparse.Namespace, max_prediction_gap: int) -> dict[str, Any]:
    return {
        "max_prediction_gap": max_prediction_gap,
        "association_gate_norm": args.association_gate_norm,
        "gate_growth_per_frame": args.gate_growth_per_frame,
        "max_gate_norm": args.max_gate_norm,
        "velocity_alpha": args.velocity_alpha,
    }


def make_trajectory_tracker(
    *,
    args: argparse.Namespace,
    gate: dict[str, Any],
    policy: dict[str, Any],
    max_prediction_gap: int,
) -> TrajectoryAwareBallTrackerV1:
    return TrajectoryAwareBallTrackerV1(
        trajectory=BallTrajectoryConsistencyV1(
            max_distance_norm=gate["distance"],
            max_speed_residual_norm=gate["speed"],
            max_acceleration_norm=gate["acceleration"],
        ),
        min_trajectory_score=policy["min_score"],
        protect_direct_confidence=policy["protect_conf"],
        override_score_margin=policy["margin"],
        **tracker_kwargs(args, max_prediction_gap),
    )


def infer_cache(
    *,
    label: str,
    model_path: Path,
    video_path: Path,
    annotations: dict[int, dict[str, Any]],
    inference_conf: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    first_frame, last_frame = min(annotations), max(annotations)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    model = YOLO(str(model_path))
    infer_args = SimpleNamespace(
        ball_conf=inference_conf,
        nms_iou=args.nms_iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
    )

    frames: list[dict[str, Any]] = []
    total = last_frame - first_frame + 1
    for offset, frame_id in enumerate(range(first_frame, last_frame + 1), start=1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")
        candidates = extract_ball_candidates(model, frame, infer_args)
        frames.append(
            {
                "frame_id": frame_id,
                "frame_shape": tuple(frame.shape),
                "timestamp_sec": frame_id / fps if fps > 0 else None,
                "candidates": candidates,
            }
        )
        if offset == 1 or offset % 100 == 0 or offset == total:
            print(
                f"[{label}] inference [{offset}/{total}] {100.0 * offset / max(total, 1):.1f}% "
                f"frame={frame_id} candidates={len(candidates)}",
                flush=True,
            )
    capture.release()
    return {
        "label": label,
        "model": str(model_path),
        "inference_conf": inference_conf,
        "frames": frames,
    }


def evaluate_direct(
    *,
    cache: dict[str, Any],
    annotations: dict[int, dict[str, Any]],
    threshold: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for frame in cache["frames"]:
        frame_id = frame["frame_id"]
        if frame_id not in annotations:
            continue
        candidates = filter_candidates(frame["candidates"], threshold)
        direct = max(candidates, key=lambda item: item["confidence"]) if candidates else None
        records.append(
            make_record(
                frame_id=frame_id,
                timestamp_sec=frame["timestamp_sec"],
                annotation=annotations[frame_id],
                key="direct",
                prediction=direct,
                source="highest_confidence_direct",
                point_margin_px=args.point_margin_px,
            )
        )
    return metric_pack(summarize(records, "direct")), records


def evaluate_tracker(
    *,
    cache: dict[str, Any],
    annotations: dict[int, dict[str, Any]],
    threshold: float,
    tracker: Any,
    key: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    direct_metrics, direct_records = evaluate_direct(
        cache=cache,
        annotations=annotations,
        threshold=threshold,
        args=args,
    )
    direct_by_frame = {row["frame_id"]: row for row in direct_records}
    records: list[dict[str, Any]] = []
    sources: Counter[str] = Counter()

    for frame in cache["frames"]:
        candidates = filter_candidates(frame["candidates"], threshold)
        decision = tracker.update(
            clone_candidates(candidates),
            frame_id=frame["frame_id"],
            frame_shape=frame["frame_shape"],
        )
        sources[decision.source] += 1
        frame_id = frame["frame_id"]
        if frame_id not in annotations:
            continue
        row = make_record(
            frame_id=frame_id,
            timestamp_sec=frame["timestamp_sec"],
            annotation=annotations[frame_id],
            key=key,
            prediction=decision.ball,
            source=decision.source,
            point_margin_px=args.point_margin_px,
        )
        records.append({**direct_by_frame[frame_id], **row})

    metrics = metric_pack(summarize(records, key))
    metrics["delta_vs_direct"] = {
        "visible_recall": metrics["visible_recall"] - direct_metrics["visible_recall"],
        "decision_precision": metrics["decision_precision"] - direct_metrics["decision_precision"],
        "output_coverage": metrics["output_coverage"] - direct_metrics["output_coverage"],
        **transitions(records, key),
    }
    return metrics, dict(sources)


def threshold_sweep(
    *,
    cache: dict[str, Any],
    annotations: dict[int, dict[str, Any]],
    thresholds: list[float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        direct, _ = evaluate_direct(cache=cache, annotations=annotations, threshold=threshold, args=args)
        temporal_tracker = TemporalBallTrackerV1(**tracker_kwargs(args, 4))
        temporal, _ = evaluate_tracker(
            cache=cache,
            annotations=annotations,
            threshold=threshold,
            tracker=temporal_tracker,
            key="temporal",
            args=args,
        )
        trajectory_tracker = make_trajectory_tracker(
            args=args,
            gate=FROZEN_V100_TRAJECTORY["gate"],
            policy=FROZEN_V100_TRAJECTORY["policy"],
            max_prediction_gap=FROZEN_V100_TRAJECTORY["max_prediction_gap"],
        )
        trajectory, _ = evaluate_tracker(
            cache=cache,
            annotations=annotations,
            threshold=threshold,
            tracker=trajectory_tracker,
            key="trajectory",
            args=args,
        )
        rows.append(
            {
                "threshold": threshold,
                "direct": direct,
                "temporal_gap4": temporal,
                "trajectory_frozen_v100": trajectory,
            }
        )
    return rows


def best_threshold(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row[variant]["f1_recall_precision"],
            row[variant]["visible_recall"],
            row[variant]["decision_precision"] or -1.0,
            -row["threshold"],
        ),
    )


def v110_policy_sweep(
    *,
    cache: dict[str, Any],
    annotations: dict[int, dict[str, Any]],
    temporal_threshold: float,
    trajectory_threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    temporal_rows: list[dict[str, Any]] = []
    for gap in (1, 2, 3, 4):
        tracker = TemporalBallTrackerV1(**tracker_kwargs(args, gap))
        metrics, sources = evaluate_tracker(
            cache=cache,
            annotations=annotations,
            threshold=temporal_threshold,
            tracker=tracker,
            key="temporal",
            args=args,
        )
        temporal_rows.append(
            {
                "threshold": temporal_threshold,
                "max_prediction_gap": gap,
                "metrics": metrics,
                "source_counts": sources,
            }
        )

    gates = [
        {"name": "loose", "distance": 0.16, "speed": 0.12, "acceleration": 0.14},
        {"name": "medium", "distance": 0.10, "speed": 0.08, "acceleration": 0.08},
    ]
    policies = [
        {"name": "score_060", "min_score": 0.60, "protect_conf": None, "margin": 0.0},
        {"name": "score_065", "min_score": 0.65, "protect_conf": None, "margin": 0.0},
        {"name": "score_070", "min_score": 0.70, "protect_conf": None, "margin": 0.0},
        {"name": "score_070_protect_075", "min_score": 0.70, "protect_conf": 0.75, "margin": 0.0},
        {"name": "score_075_protect_075", "min_score": 0.75, "protect_conf": 0.75, "margin": 0.0},
    ]
    trajectory_rows: list[dict[str, Any]] = []
    for gap in (2, 4):
        for gate in gates:
            for policy in policies:
                tracker = make_trajectory_tracker(
                    args=args,
                    gate=gate,
                    policy=policy,
                    max_prediction_gap=gap,
                )
                metrics, sources = evaluate_tracker(
                    cache=cache,
                    annotations=annotations,
                    threshold=trajectory_threshold,
                    tracker=tracker,
                    key="trajectory",
                    args=args,
                )
                trajectory_rows.append(
                    {
                        "threshold": trajectory_threshold,
                        "max_prediction_gap": gap,
                        "gate": gate,
                        "policy": policy,
                        "metrics": metrics,
                        "source_counts": sources,
                    }
                )

    temporal_ranked = sorted(
        temporal_rows,
        key=lambda row: (
            row["metrics"]["f1_recall_precision"],
            row["metrics"]["visible_recall"],
            row["metrics"]["decision_precision"] or -1.0,
        ),
        reverse=True,
    )
    trajectory_ranked = sorted(
        trajectory_rows,
        key=lambda row: (
            row["metrics"]["f1_recall_precision"],
            row["metrics"]["visible_recall"],
            row["metrics"]["decision_precision"] or -1.0,
        ),
        reverse=True,
    )
    return {
        "temporal_threshold": temporal_threshold,
        "trajectory_threshold": trajectory_threshold,
        "temporal_ranked": temporal_ranked,
        "trajectory_ranked": trajectory_ranked,
        "best_temporal": temporal_ranked[0],
        "best_trajectory": trajectory_ranked[0],
    }


def summary_line(label: str, threshold: float, metrics: dict[str, Any]) -> str:
    precision = metrics["decision_precision"]
    return (
        f"{label}: conf={threshold:.3f} recall={metrics['visible_recall']:.3f} "
        f"precision={precision:.3f} f1={metrics['f1_recall_precision']:.3f} "
        f"coverage={metrics['output_coverage']:.3f}"
    )


def main() -> None:
    args = parse_args()
    thresholds = parse_csv_floats(args.thresholds)
    inference_conf = min(thresholds)
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
        else Path("evaluation") / "ball_model_calibration_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_v100 = infer_cache(
        label="model_v1.0.0",
        model_path=model_v100,
        video_path=video_path,
        annotations=annotations,
        inference_conf=inference_conf,
        args=args,
    )
    cache_v110 = infer_cache(
        label="model_v1.1.0",
        model_path=model_v110,
        video_path=video_path,
        annotations=annotations,
        inference_conf=inference_conf,
        args=args,
    )

    sweep_v100 = threshold_sweep(
        cache=cache_v100,
        annotations=annotations,
        thresholds=thresholds,
        args=args,
    )
    sweep_v110 = threshold_sweep(
        cache=cache_v110,
        annotations=annotations,
        thresholds=thresholds,
        args=args,
    )

    best_v100 = {
        variant: best_threshold(sweep_v100, variant)
        for variant in ("direct", "temporal_gap4", "trajectory_frozen_v100")
    }
    best_v110_threshold = {
        variant: best_threshold(sweep_v110, variant)
        for variant in ("direct", "temporal_gap4", "trajectory_frozen_v100")
    }

    v110_tuning = v110_policy_sweep(
        cache=cache_v110,
        annotations=annotations,
        temporal_threshold=best_v110_threshold["temporal_gap4"]["threshold"],
        trajectory_threshold=best_v110_threshold["trajectory_frozen_v100"]["threshold"],
        args=args,
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 detector calibration and small v1.1.0 tracker-policy sweep",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "frame_range": [first_frame, last_frame],
        "device": args.device,
        "imgsz": args.imgsz,
        "thresholds": thresholds,
        "inference_conf": inference_conf,
        "frozen_v100_trajectory_policy": FROZEN_V100_TRAJECTORY,
        "supervised_benchmark_context_from_project": {
            "model_v1.0.0_basketball_precision": 0.453,
            "model_v1.0.0_basketball_recall": 0.664,
            "model_v1.1.0_basketball_precision": 0.844,
            "model_v1.1.0_basketball_recall": 0.785,
            "note": "Context supplied by the project owner; this runner does not recompute the supervised detector benchmark.",
        },
        "threshold_sweep": {
            "model_v1.0.0": sweep_v100,
            "model_v1.1.0": sweep_v110,
        },
        "best_threshold_by_f1": {
            "model_v1.0.0": best_v100,
            "model_v1.1.0_before_tracker_retuning": best_v110_threshold,
        },
        "model_v1.1.0_small_tracker_retuning": v110_tuning,
        "comparison_guidance": {
            "selection_metric": "F1 over visible recall and decision precision; recall, precision, coverage and errors remain separately reported.",
            "v1.0.0_status": "Existing P1 integration reference; trajectory policy previously tuned on this GT.",
            "v1.1.0_status": "Detector challenger with stronger supervised benchmark; threshold and small tracker policy sweep performed here.",
        },
        "notes": [
            "Each detector is run once at the minimum requested confidence; higher thresholds are replayed by filtering the same post-NMS ball candidate cache.",
            "Threshold calibration is performed for both models so score calibration differences are not mistaken for detector capability differences.",
            "The v1.1.0 tracker sweep is intentionally small: temporal max_prediction_gap 1..4; trajectory uses loose/medium gates, gaps 2/4, and a small score/protection policy set.",
            "This same Porto-Ovarense GT is calibration/tuning evidence. Final promotion still requires evaluation on additional held-out videos with the chosen settings frozen.",
        ],
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest threshold by F1 (frozen tracker policies)")
    for model_label, best in (("v1.0.0", best_v100), ("v1.1.0", best_v110_threshold)):
        print(f"\n{model_label}")
        for variant in ("direct", "temporal_gap4", "trajectory_frozen_v100"):
            row = best[variant]
            print(summary_line(variant, row["threshold"], row[variant]))

    print("\nv1.1.0 retuned tracker candidates")
    temporal_best = v110_tuning["best_temporal"]
    print(
        summary_line(
            f"temporal gap={temporal_best['max_prediction_gap']}",
            temporal_best["threshold"],
            temporal_best["metrics"],
        )
    )
    trajectory_best = v110_tuning["best_trajectory"]
    print(
        summary_line(
            "trajectory "
            f"gate={trajectory_best['gate']['name']} gap={trajectory_best['max_prediction_gap']} "
            f"policy={trajectory_best['policy']['name']}",
            trajectory_best["threshold"],
            trajectory_best["metrics"],
        )
    )
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
