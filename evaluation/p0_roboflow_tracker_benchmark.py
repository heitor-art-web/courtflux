from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import (
    align_production_truth,
    candidate_matches_truth,
    load_jsonl,
)

try:
    import supervision as sv
    from trackers import CBIoUTracker, OCSORTTracker
    from trackers.utils.iou import CIoU
except ImportError as exc:  # pragma: no cover - local optional dependency
    raise SystemExit(
        "Missing optional tracker dependencies. Install with: "
        "python3 -m pip install -r requirements-tracking.txt"
    ) from exc


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def pct(n: int | float, d: int | float) -> float:
    return round(100.0 * float(n) / max(float(d), 1.0), 2)


def _candidate_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    assoc = row.get("ball_association")
    if not isinstance(assoc, dict):
        return []
    candidates = assoc.get("candidates") or []
    return [c for c in candidates if isinstance(c, dict)]


def _box(candidate: dict[str, Any]) -> list[float] | None:
    box = candidate.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [float(v) for v in box]
    except (TypeError, ValueError):
        return None


def _confidence(candidate: dict[str, Any]) -> float:
    for key in ("detector_confidence", "confidence", "rank_score", "association_confidence"):
        value = candidate.get(key)
        try:
            if value is not None:
                return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def detections_from_row(row: dict[str, Any]) -> sv.Detections:
    boxes: list[list[float]] = []
    confidences: list[float] = []
    source_indices: list[int] = []
    for idx, candidate in enumerate(_candidate_rows(row)):
        box = _box(candidate)
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.append(box)
        confidences.append(_confidence(candidate))
        source_indices.append(idx)

    if not boxes:
        detections = sv.Detections.empty()
        detections.confidence = np.array([], dtype=np.float32)
        detections.data["courtflux_candidate_index"] = np.array([], dtype=np.int32)
        return detections

    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        confidence=np.asarray(confidences, dtype=np.float32),
        data={"courtflux_candidate_index": np.asarray(source_indices, dtype=np.int32)},
    )


def tracked_candidate_dict(det: sv.Detections, index: int) -> dict[str, Any]:
    xyxy = det.xyxy[index]
    return {
        "box": [float(v) for v in xyxy.tolist()],
        "center": (
            float((xyxy[0] + xyxy[2]) / 2.0),
            float((xyxy[1] + xyxy[3]) / 2.0),
        ),
        "confidence": (
            0.0 if det.confidence is None else float(det.confidence[index])
        ),
    }


def make_tracker(name: str, frame_rate: float):
    if name == "ocsort_ciou":
        # Fixed CourtFlux profile; this is not a sweep. CIoU is chosen because Roboflow's
        # tracker docs explicitly call it out for small/fast ball tracking. We keep a low
        # confidence admission threshold because the frozen candidate generator operates
        # at 0.10 and this benchmark is about association, not detector thresholding.
        return OCSORTTracker(
            frame_rate=float(frame_rate),
            minimum_consecutive_frames=3,
            lost_track_buffer=8,
            minimum_iou_threshold=-0.3,
            direction_consistency_weight=0.2,
            high_conf_det_threshold=0.10,
            delta_t=3,
            iou=CIoU(),
        )
    if name == "cbiou":
        return CBIoUTracker(
            frame_rate=float(frame_rate),
            lost_track_buffer=8,
            track_activation_threshold=0.10,
            minimum_consecutive_frames=3,
            high_conf_det_threshold=0.25,
            instant_first_frame_activation=False,
            buffer_ratio_first=0.3,
            buffer_ratio_second=0.5,
        )
    raise ValueError(f"Unknown tracker: {name}")


def evaluate_tracker(
    run_dir: Path,
    gt_path: Path,
    *,
    tracker_name: str,
    frame_rate: float,
    radius_px: float,
) -> dict[str, Any]:
    production_rows = load_jsonl(run_dir / "frames.jsonl")
    truth_rows = load_jsonl(gt_path)
    aligned, unmatched = align_production_truth(production_rows, truth_rows, tolerance_sec=0.001)

    tracker = make_tracker(tracker_name, frame_rate)
    visible = 0
    not_visible = 0
    candidate_true_frames = 0
    confirmed_true_frames = 0
    confirmed_any_frames = 0
    not_visible_confirmed_frames = 0
    confirmed_instances = 0
    confirmed_true_instances = 0
    confirmed_false_instances = 0
    true_track_ids: list[int] = []
    true_track_id_counts: Counter[int] = Counter()
    false_track_id_counts: Counter[int] = Counter()
    truth_by_track: dict[int, Counter[str]] = defaultdict(Counter)

    for row, truth, _delta in aligned:
        is_visible = bool(truth.get("visible"))
        visible += int(is_visible)
        not_visible += int(not is_visible)

        input_candidates = _candidate_rows(row)
        if is_visible:
            if any(candidate_matches_truth(c, truth, radius_px, 0.25)[0] for c in input_candidates):
                candidate_true_frames += 1

        detections = detections_from_row(row)
        tracked = tracker.update(detections)
        tids = (
            np.asarray(tracked.tracker_id, dtype=int)
            if tracked.tracker_id is not None
            else np.full(len(tracked), -1, dtype=int)
        )
        confirmed_indices = [idx for idx, tid in enumerate(tids.tolist()) if int(tid) >= 0]
        confirmed_instances += len(confirmed_indices)
        if confirmed_indices:
            confirmed_any_frames += 1

        frame_has_true_confirmed = False
        for idx in confirmed_indices:
            tid = int(tids[idx])
            candidate = tracked_candidate_dict(tracked, idx)
            correct, _distance = candidate_matches_truth(candidate, truth, radius_px, 0.25)
            if is_visible and correct:
                frame_has_true_confirmed = True
                confirmed_true_instances += 1
                true_track_ids.append(tid)
                true_track_id_counts[tid] += 1
                truth_by_track[tid]["true"] += 1
            else:
                confirmed_false_instances += 1
                false_track_id_counts[tid] += 1
                truth_by_track[tid]["false"] += 1

        if is_visible and frame_has_true_confirmed:
            confirmed_true_frames += 1
        if not is_visible and confirmed_indices:
            not_visible_confirmed_frames += 1

    unique_true_ids = sorted(set(true_track_ids))
    dominant_true_id_frames = max(true_track_id_counts.values(), default=0)
    track_purities = []
    for tid, counts in truth_by_track.items():
        total = counts["true"] + counts["false"]
        if total:
            track_purities.append((tid, counts["true"] / total, total))
    weighted_purity_num = sum(purity * total for _, purity, total in track_purities)
    weighted_purity_den = sum(total for _, _, total in track_purities)

    return {
        "run_id": run_dir.name,
        "tracker": tracker_name,
        "aligned_labelled_frames": len(aligned),
        "unmatched_gameplay_frames": int(unmatched),
        "counts": {
            "visible_frames": visible,
            "not_visible_frames": not_visible,
            "candidate_true_frames": candidate_true_frames,
            "confirmed_true_frames": confirmed_true_frames,
            "confirmed_any_frames": confirmed_any_frames,
            "not_visible_confirmed_frames": not_visible_confirmed_frames,
            "confirmed_instances": confirmed_instances,
            "confirmed_true_instances": confirmed_true_instances,
            "confirmed_false_instances": confirmed_false_instances,
            "unique_track_ids_ever_true": len(unique_true_ids),
            "dominant_true_track_frames": dominant_true_id_frames,
        },
        "metrics": {
            "candidate_recall_pct": pct(candidate_true_frames, visible),
            "confirmed_true_track_recall_pct": pct(confirmed_true_frames, visible),
            "confirmation_efficiency_pct": pct(confirmed_true_frames, candidate_true_frames),
            "confirmed_instance_precision_pct": pct(confirmed_true_instances, confirmed_instances),
            "not_visible_confirmed_track_frame_rate_pct": pct(not_visible_confirmed_frames, not_visible),
            "true_track_fragmentation_ids": len(unique_true_ids),
            "dominant_true_track_share_pct": pct(dominant_true_id_frames, confirmed_true_frames),
            "confirmed_track_weighted_purity_pct": round(
                100.0 * weighted_purity_num / max(weighted_purity_den, 1), 2
            ),
        },
        "true_track_id_counts": dict(true_track_id_counts),
        "false_track_id_counts": dict(false_track_id_counts),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for result in results:
        totals.update(result["counts"])
    visible = totals["visible_frames"]
    not_visible = totals["not_visible_frames"]
    return {
        "candidate_recall_micro_pct": pct(totals["candidate_true_frames"], visible),
        "confirmed_true_track_recall_micro_pct": pct(totals["confirmed_true_frames"], visible),
        "confirmation_efficiency_micro_pct": pct(
            totals["confirmed_true_frames"], totals["candidate_true_frames"]
        ),
        "confirmed_instance_precision_micro_pct": pct(
            totals["confirmed_true_instances"], totals["confirmed_instances"]
        ),
        "not_visible_confirmed_track_frame_rate_micro_pct": pct(
            totals["not_visible_confirmed_frames"], not_visible
        ),
        "unique_track_ids_ever_true_sum": int(totals["unique_track_ids_ever_true"]),
        "dominant_true_track_share_macro_pct": round(
            float(np.mean([r["metrics"]["dominant_true_track_share_pct"] for r in results])), 2
        ),
        "confirmed_track_weighted_purity_macro_pct": round(
            float(np.mean([r["metrics"]["confirmed_track_weighted_purity_pct"] for r in results])), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Roboflow OC-SORT+CIoU and C-BIoU as hypothesis trackers on "
            "frozen CourtFlux candidate traces. This does not choose a single ball identity."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--tracker", choices=["ocsort_ciou", "cbiou", "all"], default="all")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--output", default="evaluation/p0_roboflow_tracker_benchmark")
    args = parser.parse_args()

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    tracker_names = ["ocsort_ciou", "cbiou"] if args.tracker == "all" else [args.tracker]
    summary: dict[str, Any] = {
        "scope": "P0.2 hypothesis-tracker benchmark; no ball identity arbitration",
        "gt_radius_px": float(args.gt_radius_px),
        "profiles": {
            "ocsort_ciou": {
                "minimum_consecutive_frames": 3,
                "lost_track_buffer": 8,
                "minimum_iou_threshold": -0.3,
                "direction_consistency_weight": 0.2,
                "high_conf_det_threshold": 0.10,
                "delta_t": 3,
                "iou": "CIoU",
            },
            "cbiou": {
                "minimum_consecutive_frames": 3,
                "lost_track_buffer": 8,
                "track_activation_threshold": 0.10,
                "high_conf_det_threshold": 0.25,
                "instant_first_frame_activation": False,
                "buffer_ratio_first": 0.3,
                "buffer_ratio_second": 0.5,
            },
        },
        "trackers": {},
        "note": (
            "A confirmed track is not automatically the ball. The benchmark measures whether "
            "the correct candidate survives as a mature hypothesis and how much false-track "
            "state is created. Ball identity arbitration is intentionally deferred."
        ),
    }

    for tracker_name in tracker_names:
        results = []
        for run_id, gt_id in args.pair:
            results.append(
                evaluate_tracker(
                    run_root / run_id,
                    gt_root / gt_id / "annotations.jsonl",
                    tracker_name=tracker_name,
                    frame_rate=float(args.frame_rate),
                    radius_px=float(args.gt_radius_px),
                )
            )
        summary["trackers"][tracker_name] = {
            "runs": results,
            "aggregate": aggregate(results),
        }

    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("P0 ROBOFLOW TRACKER HYPOTHESIS BENCHMARK")
    print("=" * 86)
    for name, payload in summary["trackers"].items():
        metrics = payload["aggregate"]
        print(f"\n{name.upper()}")
        print("-" * 74)
        print(f"candidate recall:              {metrics['candidate_recall_micro_pct']:7.2f}%")
        print(f"confirmed true-track recall:   {metrics['confirmed_true_track_recall_micro_pct']:7.2f}%")
        print(f"confirmation efficiency:       {metrics['confirmation_efficiency_micro_pct']:7.2f}%")
        print(f"confirmed instance precision:  {metrics['confirmed_instance_precision_micro_pct']:7.2f}%")
        print(f"not-visible confirmed frames:  {metrics['not_visible_confirmed_track_frame_rate_micro_pct']:7.2f}%")
        print(f"true-track IDs (sum):           {metrics['unique_track_ids_ever_true_sum']:7d}")
        print(f"dominant true-track share:      {metrics['dominant_true_track_share_macro_pct']:7.2f}%")
        print(f"confirmed weighted purity:      {metrics['confirmed_track_weighted_purity_macro_pct']:7.2f}%")
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
