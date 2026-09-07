from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ball_error_review import (
    aggregate_production_summaries,
    analyze_production_case,
    resolve_video_path,
)


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NEW_RUN=GT_RUN")
    run_id, gt_run_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_run_id:
        raise argparse.ArgumentTypeError("Expected NEW_RUN=GT_RUN")
    return run_id, gt_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate smart-detector production runs against frozen CourtFlux ball GT. "
            "Use --pair NEW_RUN=GT_RUN repeatedly so new run ids can reuse the original annotations."
        )
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        required=True,
        help="Map one new smart run id to the frozen GT run id: NEW_RUN=GT_RUN. Repeat for a batch.",
    )
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--output", default="evaluation/p0_1_smart_benchmark")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--bbox-iou", type=float, default=0.25)
    parser.add_argument("--timestamp-tolerance-sec", type=float, default=0.001)
    parser.add_argument("--max-fn-frames", type=int, default=0)
    parser.add_argument("--max-fp-frames", type=int, default=0)
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Export review JPEGs. Disabled by default for faster A/B benchmarking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    mappings: list[dict[str, str]] = []

    for run_id, gt_run_id in args.pair:
        run_dir = run_root / run_id
        gt_dir = gt_root / gt_run_id
        frames_path = run_dir / "frames.jsonl"
        annotations_path = gt_dir / "annotations.jsonl"
        if not frames_path.exists():
            raise FileNotFoundError(frames_path)
        if not annotations_path.exists():
            raise FileNotFoundError(annotations_path)

        video_path = resolve_video_path(None, gt_dir, run_dir)
        run_output = output_root / run_id
        summary, rows = analyze_production_case(
            run_id=run_id,
            frames_path=frames_path,
            annotations_path=annotations_path,
            output_dir=run_output,
            video_path=video_path,
            gt_radius_px=max(0.0, float(args.gt_radius_px)),
            bbox_iou=max(0.0, float(args.bbox_iou)),
            timestamp_tolerance_sec=max(0.0, float(args.timestamp_tolerance_sec)),
            threshold=float(args.threshold),
            max_fn_frames=int(args.max_fn_frames),
            max_fp_frames=int(args.max_fp_frames),
            export_images=bool(args.with_images),
            oracle_model=None,
            oracle_crop_size=640,
            imgsz=960,
            device="mps",
            nms_iou=0.5,
            max_det=100,
        )
        summary["gt_run_id"] = gt_run_id
        summary["benchmark_pair"] = f"{run_id}={gt_run_id}"
        (run_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        for row in rows:
            row["gt_run_id"] = gt_run_id

        summaries.append(summary)
        all_rows.extend(rows)
        mappings.append({"run_id": run_id, "gt_run_id": gt_run_id})
        print(
            f"[{run_id} <= GT {gt_run_id}] "
            f"recall={summary['candidate_oracle']['visible_candidate_recall_pct']:.2f}% "
            f"FN={summary['candidate_oracle']['false_negative_frames']} "
            f"false/frame={summary['candidate_noise']['false_candidates_per_gameplay_frame']:.3f}"
        )

    aggregate = aggregate_production_summaries(summaries, output_root)
    aggregate["frozen_gt_mappings"] = mappings
    aggregate["smart_detector_benchmark"] = True
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(aggregate, indent=2))
    print(f"[SMART BALL BENCHMARK] {output_root}")


if __name__ == "__main__":
    main()
