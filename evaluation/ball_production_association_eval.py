from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate P0.2 production ball association using instrumented CourtFlux "
            "frames.jsonl and sparse ground truth from ball_detection_eval.py per_frame.json."
        )
    )
    parser.add_argument("--frames", required=True, help="Instrumented production frames.jsonl")
    parser.add_argument(
        "--truth",
        required=True,
        help="Ground-truth per_frame.json produced by evaluation/ball_detection_eval.py",
    )
    parser.add_argument(
        "--gt-radius-px",
        type=float,
        default=40.0,
        help="Point-truth center radius used to call a candidate/trajectory correct.",
    )
    parser.add_argument(
        "--bbox-iou",
        type=float,
        default=0.25,
        help="Fallback IoU threshold when bbox truth is used instead of point truth.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to <frames dir>/p0_2_association_eval.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise ValueError(f"No frame rows found in {path}")
    return rows


def load_truth(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--truth must be a per_frame.json top-level list")
    rows = [dict(row) for row in payload if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("frame_id", 0)))
    if not rows:
        raise ValueError(f"No truth rows found in {path}")
    return rows


def center_distance(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def item_matches_truth(
    item: dict[str, Any] | None,
    truth: dict[str, Any],
    gt_radius_px: float,
    bbox_iou: float,
) -> tuple[bool, float | None]:
    if item is None or not bool(truth.get("visible")):
        return False, None

    center = item.get("center")
    point = truth.get("point")
    if (
        isinstance(center, (list, tuple))
        and len(center) >= 2
        and isinstance(point, list)
        and len(point) == 2
    ):
        distance = center_distance(center, point)
        return distance <= gt_radius_px, distance

    box = item.get("box")
    gt_box = truth.get("bbox")
    if (
        isinstance(box, list)
        and len(box) == 4
        and isinstance(gt_box, list)
        and len(gt_box) == 4
    ):
        return iou_xyxy(box, gt_box) >= bbox_iou, None

    return False, None


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / max(denominator, 1), 2)


def candidate_failure_mode(
    true_candidates: list[dict[str, Any]],
    selected_correct: bool,
) -> str | None:
    if not true_candidates or selected_correct:
        return None

    details = [candidate.get("association") for candidate in true_candidates]
    if all(detail is None for detail in details):
        return "initialization_or_reset_loss"

    scored = [detail for detail in details if isinstance(detail, dict)]
    if any(bool(detail.get("plausible")) for detail in scored):
        return "score_or_selection_loss"
    return "gate_loss"


def main() -> None:
    args = parse_args()
    frames_path = Path(args.frames).expanduser().resolve()
    truth_path = Path(args.truth).expanduser().resolve()
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else frames_path.parent / "p0_2_association_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    production_rows = load_jsonl(frames_path)
    truth_rows = load_truth(truth_path)
    production_by_frame = {
        int(row.get("frame_id", -1)): row
        for row in production_rows
        if row.get("frame_id") is not None
    }

    instrumented_rows = 0
    for row in production_rows:
        association = row.get("ball_association") or {}
        if isinstance(association, dict) and "candidates" in association:
            instrumented_rows += 1
    if instrumented_rows == 0:
        raise ValueError(
            "No P0.2 candidate traces found. Re-run CourtFlux after the "
            "P0 association instrumentation commit; old frames.jsonl files cannot "
            "recover production candidate boxes after the fact."
        )

    visible_frames = 0
    not_visible_frames = 0
    matched_truth_frames = 0
    true_candidate_available = 0
    correct_association_opportunities = 0
    direct_visible = 0
    direct_correct = 0
    direct_wrong = 0
    direct_not_visible = 0
    predicted_visible = 0
    missing_visible = 0
    final_trajectory_correct = 0
    final_trajectory_present = 0
    final_trajectory_not_visible = 0
    reacquired_frames = 0
    reacquired_correct = 0
    reacquired_wrong = 0
    reinitialized_frames = 0

    source_counts: Counter[str] = Counter()
    track_state_counts: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []

    direct_sources = {
        "detector_associated",
        "detector_reacquired",
        "detector_reinitialized",
        "detector_highest_conf",
    }

    for truth in truth_rows:
        frame_id = int(truth.get("frame_id", 0))
        visible = bool(truth.get("visible"))
        if visible:
            visible_frames += 1
        else:
            not_visible_frames += 1

        production = production_by_frame.get(frame_id)
        if production is None:
            failure_modes["missing_production_frame"] += 1
            output_rows.append(
                {
                    "frame_id": frame_id,
                    "visible": visible,
                    "matched_production": False,
                }
            )
            continue

        matched_truth_frames += 1
        association = production.get("ball_association") or {}
        if not isinstance(association, dict):
            association = {}
        candidates = [
            dict(candidate)
            for candidate in (association.get("candidates") or [])
            if isinstance(candidate, dict)
        ]
        source = str(association.get("source") or production.get("ball_source") or "missing")
        track_state = str(association.get("track_state") or "UNKNOWN")
        source_counts[source] += 1
        track_state_counts[track_state] += 1

        true_indices: list[int] = []
        true_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            correct, _ = item_matches_truth(
                candidate,
                truth,
                gt_radius_px=args.gt_radius_px,
                bbox_iou=args.bbox_iou,
            )
            if correct:
                true_indices.append(index)
                true_candidates.append(candidate)

        selected_index = association.get("selected_index")
        selected_candidate = None
        if isinstance(selected_index, int) and 0 <= selected_index < len(candidates):
            selected_candidate = candidates[selected_index]
        selected_correct, selected_distance = item_matches_truth(
            selected_candidate,
            truth,
            gt_radius_px=args.gt_radius_px,
            bbox_iou=args.bbox_iou,
        )

        trajectory_center = association.get("trajectory_center")
        trajectory_box = association.get("trajectory_box")
        trajectory_item = None
        if trajectory_center is not None or trajectory_box is not None:
            trajectory_item = {
                "center": trajectory_center,
                "box": trajectory_box,
            }
        elif production.get("selected_ball_center") is not None:
            # Backward-compatible fallback for direct observations only.
            trajectory_item = {
                "center": production.get("selected_ball_center"),
                "box": None,
            }
        trajectory_correct, trajectory_distance = item_matches_truth(
            trajectory_item,
            truth,
            gt_radius_px=args.gt_radius_px,
            bbox_iou=args.bbox_iou,
        )

        candidate_available = bool(true_indices)
        if visible and candidate_available:
            true_candidate_available += 1
            if selected_correct and source in direct_sources:
                correct_association_opportunities += 1
            else:
                failure_mode = candidate_failure_mode(true_candidates, selected_correct)
                if failure_mode:
                    failure_modes[failure_mode] += 1
                if source == "predicted_hold":
                    failure_modes["predicted_hold_despite_true_candidate"] += 1
                elif source == "missing":
                    failure_modes["missing_despite_true_candidate"] += 1

        if visible:
            if source in direct_sources:
                direct_visible += 1
                if selected_correct:
                    direct_correct += 1
                else:
                    direct_wrong += 1
            elif source == "predicted_hold":
                predicted_visible += 1
            else:
                missing_visible += 1

            if trajectory_item is not None:
                final_trajectory_present += 1
                if trajectory_correct:
                    final_trajectory_correct += 1
        else:
            if source in direct_sources:
                direct_not_visible += 1
            if trajectory_item is not None:
                final_trajectory_not_visible += 1

        if bool(association.get("reacquired")):
            reacquired_frames += 1
            if selected_correct:
                reacquired_correct += 1
            else:
                reacquired_wrong += 1
        if source == "detector_reinitialized":
            reinitialized_frames += 1

        output_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_sec": production.get("timestamp_sec"),
                "visible": visible,
                "matched_production": True,
                "candidate_count": len(candidates),
                "true_candidate_indices": true_indices,
                "true_candidate_available": candidate_available,
                "source": source,
                "track_state": track_state,
                "selected_index": selected_index,
                "selected_correct": bool(selected_correct),
                "selected_gt_distance_px": (
                    None if selected_distance is None else round(float(selected_distance), 4)
                ),
                "trajectory_present": trajectory_item is not None,
                "trajectory_correct": bool(trajectory_correct),
                "trajectory_gt_distance_px": (
                    None if trajectory_distance is None else round(float(trajectory_distance), 4)
                ),
                "reacquired": bool(association.get("reacquired", False)),
                "challenger_streak": int(association.get("challenger_streak", 0) or 0),
                "failure_mode": candidate_failure_mode(true_candidates, selected_correct),
                "candidates": candidates,
            }
        )

    summary = {
        "scope": "P0.2 production association against sparse ground truth",
        "frames_file": str(frames_path),
        "truth_file": str(truth_path),
        "gt_radius_px": float(args.gt_radius_px),
        "bbox_iou": float(args.bbox_iou),
        "truth_frames": len(truth_rows),
        "matched_truth_frames": matched_truth_frames,
        "instrumented_production_rows": instrumented_rows,
        "visible_frames": visible_frames,
        "not_visible_frames": not_visible_frames,
        "candidate_oracle": {
            "true_candidate_available_visible": true_candidate_available,
            "visible_candidate_recall_pct": pct(true_candidate_available, visible_frames),
        },
        "association": {
            "opportunities_true_candidate_available": true_candidate_available,
            "correct_when_true_candidate_available": correct_association_opportunities,
            "accuracy_conditioned_on_true_candidate_pct": pct(
                correct_association_opportunities,
                true_candidate_available,
            ),
            "direct_visible": direct_visible,
            "direct_correct_visible": direct_correct,
            "direct_wrong_visible": direct_wrong,
            "direct_precision_visible_pct": pct(direct_correct, direct_visible),
            "direct_recall_visible_pct": pct(direct_correct, visible_frames),
            "direct_not_visible": direct_not_visible,
            "source_counts": dict(source_counts),
            "track_state_counts": dict(track_state_counts),
            "failure_modes": dict(failure_modes),
        },
        "trajectory": {
            "present_visible": final_trajectory_present,
            "correct_visible": final_trajectory_correct,
            "visible_presence_pct": pct(final_trajectory_present, visible_frames),
            "visible_correct_recall_pct": pct(final_trajectory_correct, visible_frames),
            "present_not_visible": final_trajectory_not_visible,
            "predicted_visible": predicted_visible,
            "missing_visible": missing_visible,
        },
        "reacquisition": {
            "challenger_confirmed_frames": reacquired_frames,
            "correct": reacquired_correct,
            "wrong": reacquired_wrong,
            "precision_pct": pct(reacquired_correct, reacquired_frames),
            "reinitialized_frames": reinitialized_frames,
        },
    }

    summary_path = output_dir / "summary.json"
    frames_output_path = output_dir / "frames.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with frames_output_path.open("w", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"[P0.2 SUMMARY] {summary_path}")
    print(f"[P0.2 FRAMES] {frames_output_path}")


if __name__ == "__main__":
    main()
