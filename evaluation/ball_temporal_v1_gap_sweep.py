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
from services.vision.ball_temporal_v1 import TemporalBallTrackerV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep TemporalBallTrackerV1 max_prediction_gap values on one real-video detector pass."
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
    parser.add_argument("--gaps", default="0,1,2,3,4")
    parser.add_argument("--association-gate-norm", type=float, default=0.12)
    parser.add_argument("--gate-growth-per-frame", type=float, default=0.02)
    parser.add_argument("--max-gate-norm", type=float, default=0.24)
    parser.add_argument("--velocity-alpha", type=float, default=0.45)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    for path in (video_path, annotations_path, model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gaps = sorted({max(0, int(value.strip())) for value in str(args.gaps).split(",") if value.strip()})
    if not gaps:
        raise ValueError("--gaps must contain at least one integer")

    annotations = load_annotations(annotations_path)
    first_frame, last_frame = min(annotations), max(annotations)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("evaluation") / "ball_temporal_v1_gap_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    model = YOLO(str(model_path))
    trackers = {
        gap: TemporalBallTrackerV1(
            max_prediction_gap=gap,
            association_gate_norm=args.association_gate_norm,
            gate_growth_per_frame=args.gate_growth_per_frame,
            max_gate_norm=args.max_gate_norm,
            velocity_alpha=args.velocity_alpha,
        )
        for gap in gaps
    }
    records_by_gap: dict[int, list[dict[str, Any]]] = {gap: [] for gap in gaps}
    source_counts: dict[int, Counter[str]] = {gap: Counter() for gap in gaps}
    direct_records: list[dict[str, Any]] = []

    processed = 0
    total_frames = last_frame - first_frame + 1

    for frame_id in range(first_frame, last_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")

        candidates = extract_ball_candidates(model, frame, args)
        direct = max(candidates, key=lambda item: item["confidence"]) if candidates else None
        decisions = {
            gap: tracker.update(candidates, frame_id=frame_id, frame_shape=frame.shape)
            for gap, tracker in trackers.items()
        }
        for gap, decision in decisions.items():
            source_counts[gap][decision.source] += 1

        if frame_id in annotations:
            annotation = annotations[frame_id]
            direct_correct = matches_truth(direct, annotation, args.point_margin_px)
            direct_record = {
                "frame_id": frame_id,
                "timestamp_sec": frame_id / fps if fps > 0 else annotation.get("timestamp_sec"),
                "visible": bool(annotation["visible"]),
                "direct": direct,
                "direct_correct": direct_correct,
                "direct_point_error_px": point_error(direct, annotation),
            }
            direct_records.append(direct_record)

            for gap, decision in decisions.items():
                temporal = decision.ball
                temporal_correct = matches_truth(temporal, annotation, args.point_margin_px)
                records_by_gap[gap].append(
                    {
                        "frame_id": frame_id,
                        "timestamp_sec": direct_record["timestamp_sec"],
                        "visible": direct_record["visible"],
                        "direct": direct,
                        "direct_correct": direct_correct,
                        "direct_point_error_px": direct_record["direct_point_error_px"],
                        "temporal": temporal,
                        "temporal_source": decision.source,
                        "temporal_correct": temporal_correct,
                        "temporal_point_error_px": point_error(temporal, annotation),
                    }
                )

        processed += 1
        if processed == 1 or processed % 100 == 0 or frame_id == last_frame:
            print(
                f"[{processed}/{total_frames}] {100.0 * processed / max(total_frames, 1):.1f}% "
                f"frame={frame_id} candidates={len(candidates)}",
                flush=True,
            )

    capture.release()

    direct_summary = summarize(direct_records, "direct")
    sweep: list[dict[str, Any]] = []

    for gap in gaps:
        records = records_by_gap[gap]
        temporal_summary = summarize(records, "temporal")
        bridged = [row for row in records if row["temporal_source"] == "predicted_short_gap"]
        bridged_visible = [row for row in bridged if row["visible"]]
        bridged_correct = sum(1 for row in bridged_visible if row["temporal_correct"])
        recovered = sum(
            1 for row in records if row["visible"] and not row["direct_correct"] and row["temporal_correct"]
        )
        regressions = sum(
            1 for row in records if row["visible"] and row["direct_correct"] and not row["temporal_correct"]
        )
        sweep.append(
            {
                "max_prediction_gap": gap,
                "temporal_tracker": temporal_summary,
                "delta": {
                    "visible_recall": temporal_summary["visible_recall"] - direct_summary["visible_recall"],
                    "decision_precision": (
                        temporal_summary["decision_precision"] - direct_summary["decision_precision"]
                        if temporal_summary["decision_precision"] is not None
                        and direct_summary["decision_precision"] is not None
                        else None
                    ),
                    "output_coverage": temporal_summary["output_coverage"] - direct_summary["output_coverage"],
                    "recovered_visible_frames": recovered,
                    "regressed_visible_frames": regressions,
                },
                "short_gap_predictions": {
                    "count": len(bridged),
                    "visible_count": len(bridged_visible),
                    "correct_visible_count": bridged_correct,
                    "visible_precision": bridged_correct / len(bridged_visible) if bridged_visible else None,
                },
                "source_counts_all_processed_frames": dict(source_counts[gap]),
            }
        )

    best = max(
        sweep,
        key=lambda row: (
            row["temporal_tracker"]["decision_precision"] if row["temporal_tracker"]["decision_precision"] is not None else -1.0,
            row["temporal_tracker"]["visible_recall"],
            -row["max_prediction_gap"],
        ),
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 TemporalBallTrackerV1 max_prediction_gap sweep using one detector pass",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "model": str(model_path),
        "device": args.device,
        "imgsz": args.imgsz,
        "ball_conf": args.ball_conf,
        "frame_range": [first_frame, last_frame],
        "processed_frames": processed,
        "annotated_frames": len(direct_records),
        "gaps": gaps,
        "direct_baseline": direct_summary,
        "sweep": sweep,
        "best_by_precision_then_recall": best["max_prediction_gap"],
        "notes": [
            "Detector inference is executed once per frame; all gap policies consume the same candidate list.",
            "Gap=0 disables short-gap prediction while retaining temporal association logic.",
            "Metrics are computed only on annotated frames; temporal state is updated across every frame in the GT span.",
            "The reported best value is descriptive only and must not be treated as held-out tuning evidence.",
        ],
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nmax_gap | recall | precision | coverage | recovered | regressed | short-gap precision")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in sweep:
        metrics = row["temporal_tracker"]
        delta = row["delta"]
        short = row["short_gap_predictions"]
        precision = metrics["decision_precision"]
        short_precision = short["visible_precision"]
        print(
            f"{row['max_prediction_gap']} | {metrics['visible_recall']:.3f} | "
            f"{precision:.3f} | {metrics['output_coverage']:.3f} | "
            f"{delta['recovered_visible_frames']} | {delta['regressed_visible_frames']} | "
            f"{short_precision:.3f}" if short_precision is not None else
            f"{row['max_prediction_gap']} | {metrics['visible_recall']:.3f} | "
            f"{precision:.3f} | {metrics['output_coverage']:.3f} | "
            f"{delta['recovered_visible_frames']} | {delta['regressed_visible_frames']} | n/a"
        )

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
