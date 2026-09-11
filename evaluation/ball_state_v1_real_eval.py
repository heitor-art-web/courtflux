from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
from ultralytics import YOLO

from core.ball_state import BallState, BallStateStatus
from core.config import Settings, settings
from evaluation.ball_temporal_v1_real_eval import (
    extract_ball_candidates,
    load_annotations,
    matches_truth,
    point_error,
    summarize,
)
from services.vision.ball_state_v1 import BallStateEstimatorV1


REQUIRED_PROVENANCE_KEYS = {
    "detector_model",
    "detector_version",
    "detector_baseline_status",
    "ball_candidate_confidence",
    "tracker",
    "tracker_source",
    "direct_observation",
    "short_gap_prediction",
    "prediction_updates_state",
    "reacquisition",
    "uncertainty_policy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate canonical P1 BallState v1 on annotated real video using the frozen "
            "model_v1.1.0 + trajectory configuration."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--imgsz", type=int, default=settings.imgsz)
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=settings.ball_candidate_confidence,
        help="Operational BallState candidate threshold. Default is the frozen P1 v1.1.0 value.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--point-margin-px", type=float, default=4.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory; defaults to evaluation/ball_state_v1_real/<timestamp>.",
    )
    return parser.parse_args()


def state_prediction(state: BallState) -> dict[str, Any] | None:
    if not state.available or state.box is None:
        return None
    return {
        "box": list(state.box),
        "center": state.position(),
        "confidence": state.confidence,
        "source": state.source,
        "interpolated": state.predicted,
    }


def status_metrics(records: list[dict[str, Any]], status: str) -> dict[str, Any]:
    rows = [row for row in records if row["state"]["status"] == status]
    total = len(rows)
    visible = sum(1 for row in rows if row["visible"])
    invisible = total - visible
    correct_visible = sum(1 for row in rows if row["visible"] and row["state_correct"])
    wrong_visible = sum(
        1
        for row in rows
        if row["visible"] and row["state_prediction"] is not None and not row["state_correct"]
    )
    false_positive_invisible = sum(
        1 for row in rows if not row["visible"] and row["state_prediction"] is not None
    )
    precision_den = correct_visible + wrong_visible + false_positive_invisible
    errors = [
        row["state_point_error_px"]
        for row in rows
        if row["state_correct"] and row["state_point_error_px"] is not None
    ]
    uncertainties = [float(row["state"]["uncertainty"]) for row in rows]
    return {
        "frames": total,
        "fraction_of_annotated": total / len(records) if records else 0.0,
        "visible_frames": visible,
        "invisible_frames": invisible,
        "correct_visible_frames": correct_visible,
        "wrong_visible_outputs": wrong_visible,
        "false_positive_invisible_frames": false_positive_invisible,
        "decision_precision": correct_visible / precision_den if precision_den else None,
        "visible_recall_contribution": (
            correct_visible / sum(1 for row in records if row["visible"])
            if any(row["visible"] for row in records)
            else 0.0
        ),
        "mean_point_error_px_on_correct": sum(errors) / len(errors) if errors else None,
        "mean_uncertainty": sum(uncertainties) / len(uncertainties) if uncertainties else None,
    }


def uncertainty_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in records:
        state = row["state"]
        status = state["status"]
        uncertainty = float(state["uncertainty"])
        if status == BallStateStatus.MISSING.value:
            groups["missing"].append(uncertainty)
            continue
        groups[status].append(uncertainty)
        if row["state_correct"]:
            groups["available_correct"].append(uncertainty)
            groups[f"{status}_correct"].append(uncertainty)
        else:
            groups["available_incorrect"].append(uncertainty)
            groups[f"{status}_incorrect"].append(uncertainty)

    return {
        key: {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
        for key, values in sorted(groups.items())
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
        else Path("evaluation") / "ball_state_v1_real" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Settings(
        model=str(model_path),
        detector_version=settings.detector_version,
        detector_baseline_status=settings.detector_baseline_status,
        device=args.device,
        imgsz=args.imgsz,
        ball_candidate_confidence=float(args.ball_conf),
        ball_max_prediction_gap=settings.ball_max_prediction_gap,
        ball_trajectory_max_distance_norm=settings.ball_trajectory_max_distance_norm,
        ball_trajectory_max_speed_residual_norm=settings.ball_trajectory_max_speed_residual_norm,
        ball_trajectory_max_acceleration_norm=settings.ball_trajectory_max_acceleration_norm,
        ball_trajectory_min_score=settings.ball_trajectory_min_score,
        ball_trajectory_protect_direct_confidence=settings.ball_trajectory_protect_direct_confidence,
        ball_trajectory_override_score_margin=settings.ball_trajectory_override_score_margin,
    )
    estimator = BallStateEstimatorV1(config=config)
    model = YOLO(str(model_path))
    infer_args = SimpleNamespace(
        ball_conf=float(args.ball_conf),
        nms_iou=float(args.nms_iou),
        imgsz=int(args.imgsz),
        max_det=int(args.max_det),
        device=args.device,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    status_counts_all: Counter[str] = Counter()
    candidate_total = 0
    provenance_complete = 0
    provenance_checked = 0
    processed = 0
    total_frames = last_frame - first_frame + 1

    for frame_id in range(first_frame, last_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")
        timestamp_sec = frame_id / fps if fps > 0 else None
        candidates = extract_ball_candidates(model, frame, infer_args)
        candidate_total += len(candidates)
        direct = max(candidates, key=lambda item: item["confidence"]) if candidates else None
        state = estimator.update(
            candidates,
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            frame_shape=frame.shape,
        )
        prediction = state_prediction(state)
        source_counts[state.source] += 1
        status_counts_all[state.status.value] += 1
        provenance_checked += 1
        if REQUIRED_PROVENANCE_KEYS.issubset(state.provenance):
            provenance_complete += 1
        processed += 1

        if frame_id in annotations:
            annotation = annotations[frame_id]
            direct_correct = matches_truth(direct, annotation, args.point_margin_px)
            state_correct = matches_truth(prediction, annotation, args.point_margin_px)
            records.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp_sec,
                    "visible": bool(annotation["visible"]),
                    "basketball_candidates": len(candidates),
                    "direct": direct,
                    "direct_correct": direct_correct,
                    "direct_point_error_px": point_error(direct, annotation),
                    "state": state.as_dict(),
                    "state_prediction": prediction,
                    "state_correct": state_correct,
                    "state_point_error_px": point_error(prediction, annotation),
                }
            )

        if processed == 1 or processed % 100 == 0 or processed == total_frames:
            counts = " ".join(
                f"{name}={status_counts_all.get(name, 0)}"
                for name in ("observed", "predicted", "missing")
            )
            print(
                f"[BallState] [{processed}/{total_frames}] "
                f"{100.0 * processed / max(total_frames, 1):.1f}% frame={frame_id} "
                f"basketball_candidates={len(candidates)} cumulative[{counts}]",
                flush=True,
            )

    capture.release()

    direct_metrics = summarize(records, "direct")
    state_metrics = summarize(records, "state_prediction")
    status_breakdown = {
        status.value: status_metrics(records, status.value)
        for status in BallStateStatus
    }
    annotated_status_counts = Counter(row["state"]["status"] for row in records)
    visible_frames = sum(1 for row in records if row["visible"])
    invisible_frames = len(records) - visible_frames
    missing_visible = sum(
        1
        for row in records
        if row["visible"] and row["state"]["status"] == BallStateStatus.MISSING.value
    )
    missing_invisible = sum(
        1
        for row in records
        if not row["visible"] and row["state"]["status"] == BallStateStatus.MISSING.value
    )
    recovered = sum(
        1
        for row in records
        if row["visible"] and not row["direct_correct"] and row["state_correct"]
    )
    regressed = sum(
        1
        for row in records
        if row["visible"] and row["direct_correct"] and not row["state_correct"]
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 BallState v1 real-video validation",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "frame_range": [first_frame, last_frame],
        "processed_frames": processed,
        "annotated_frames": len(records),
        "visible_annotated_frames": visible_frames,
        "invisible_annotated_frames": invisible_frames,
        "detector": {
            "model": str(model_path),
            "version": config.detector_version,
            "baseline_status": config.detector_baseline_status,
            "device": args.device,
            "imgsz": args.imgsz,
            "ball_candidate_confidence": args.ball_conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
        },
        "ball_state_metadata": estimator.metadata(),
        "all_processed_state_counts": dict(status_counts_all),
        "annotated_state_counts": dict(annotated_status_counts),
        "candidate_diagnostics": {
            "basketball_candidates_total": candidate_total,
            "basketball_candidates_mean_per_processed_frame": candidate_total / processed if processed else 0.0,
        },
        "direct_baseline": direct_metrics,
        "ball_state": state_metrics,
        "delta_ball_state_minus_direct": {
            "visible_recall": state_metrics["visible_recall"] - direct_metrics["visible_recall"],
            "decision_precision": (
                None
                if state_metrics["decision_precision"] is None or direct_metrics["decision_precision"] is None
                else state_metrics["decision_precision"] - direct_metrics["decision_precision"]
            ),
            "output_coverage": state_metrics["output_coverage"] - direct_metrics["output_coverage"],
            "recovered_visible_frames": recovered,
            "regressed_visible_frames": regressed,
        },
        "status_breakdown": status_breakdown,
        "missing_behavior": {
            "visible_frames_marked_missing": missing_visible,
            "visible_missing_rate": missing_visible / visible_frames if visible_frames else 0.0,
            "invisible_frames_marked_missing": missing_invisible,
            "invisible_abstention_rate": missing_invisible / invisible_frames if invisible_frames else 0.0,
        },
        "source_counts_all_processed_frames": dict(source_counts),
        "uncertainty": uncertainty_summary(records),
        "provenance": {
            "required_keys": sorted(REQUIRED_PROVENANCE_KEYS),
            "frames_checked": provenance_checked,
            "complete_frames": provenance_complete,
            "completeness_rate": provenance_complete / provenance_checked if provenance_checked else 0.0,
        },
        "scope_guard": {
            "p2_event_semantics": False,
            "possession": False,
            "pass": False,
            "shot": False,
            "rebound": False,
            "made_miss": False,
        },
        "notes": [
            "BallState observed means direct detector evidence selected by the trajectory-aware tracker.",
            "BallState predicted means short-gap temporal prediction and is never counted as a direct observation.",
            "BallState missing is explicit UNKNOWN and exposes no synthetic position.",
            "The direct baseline uses the same v1.1.0 detector pass and operational basketball confidence threshold.",
            "This Porto-Ovarense segment remains calibration/integration evidence, not independent P1 exit validation.",
        ],
    }

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    precision = state_metrics["decision_precision"]
    direct_precision = direct_metrics["decision_precision"]
    print("\nBallState v1 real-video summary")
    print(
        f"direct: recall={direct_metrics['visible_recall']:.3f} "
        f"precision={direct_precision:.3f} coverage={direct_metrics['output_coverage']:.3f}"
    )
    print(
        f"state:  recall={state_metrics['visible_recall']:.3f} "
        f"precision={precision:.3f} coverage={state_metrics['output_coverage']:.3f}"
    )
    print(
        "annotated status: "
        + " ".join(
            f"{status.value}={annotated_status_counts.get(status.value, 0)}"
            for status in BallStateStatus
        )
    )
    print(
        f"transitions: recovered={recovered} regressed={regressed} "
        f"invisible_abstention={summary['missing_behavior']['invisible_abstention_rate']:.3f}"
    )
    print(f"provenance completeness={summary['provenance']['completeness_rate']:.3f}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
