from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_tracker import BallCandidateScore, TemporalBallTracker


WEIGHT_SETS = [
    ("current", 0.58, 0.32, 0.10),
    ("balanced", 0.45, 0.45, 0.10),
    ("temporal_lean", 0.35, 0.55, 0.10),
    ("temporal_strong", 0.25, 0.65, 0.10),
    ("size_aware", 0.35, 0.50, 0.15),
]

GATE_SETS = [
    ("current", 0.12, 0.025, 0.28, 0.85, 1.35),
    ("no_high_conf_boost", 0.12, 0.025, 0.28, 0.85, 1.00),
    ("medium", 0.10, 0.020, 0.24, 0.85, 1.00),
    ("tight", 0.08, 0.015, 0.18, 0.85, 1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay CourtFlux TemporalBallTracker against saved P0.1 detector candidates "
            "and ground-truth points. No YOLO inference is performed."
        )
    )
    parser.add_argument("--per-frame", required=True, help="ball_detection_eval.py per_frame.json")
    parser.add_argument(
        "--video",
        default=None,
        help="Optional source video used only to read frame width/height.",
    )
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    parser.add_argument(
        "--candidate-conf",
        type=float,
        default=0.01,
        help="Minimum confidence admitted into association replay. Default matches evaluator capture conf.",
    )
    parser.add_argument(
        "--gt-radius-px",
        type=float,
        default=40.0,
        help=(
            "A direct candidate is considered the real ball when its center is within this many pixels "
            "of one-click point truth. Center distance avoids huge boxes passing merely because they contain the click."
        ),
    )
    parser.add_argument("--bbox-iou", type=float, default=0.25)
    parser.add_argument("--max-hold-frames", type=int, default=5)
    parser.add_argument("--reset-after-frames", type=int, default=12)
    parser.add_argument("--velocity-alpha", type=float, default=0.40)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to <per_frame directory>/association_benchmark.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("per_frame.json must contain a top-level JSON list.")
    rows = [dict(row) for row in payload if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("frame_id", 0)))
    if not rows:
        raise ValueError("No per-frame rows found.")
    return rows


def resolve_frame_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.video:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {args.video}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not read video dimensions: {args.video}")
        return (height, width, 3)

    if args.frame_width and args.frame_height:
        if args.frame_width <= 0 or args.frame_height <= 0:
            raise ValueError("--frame-width and --frame-height must be positive.")
        return (int(args.frame_height), int(args.frame_width), 3)

    raise ValueError("Provide --video or both --frame-width and --frame-height.")


def box_center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def center_distance(a: tuple[float, float], b: list[float] | tuple[float, float]) -> float:
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
    return intersection / union if union > 0.0 else 0.0


def prepare_candidates(row: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for prediction in row.get("predictions") or []:
        confidence = float(prediction.get("confidence", 0.0))
        box = prediction.get("box")
        if confidence < threshold or not isinstance(box, list) or len(box) != 4:
            continue
        box_values = [float(value) for value in box]
        if box_values[2] <= box_values[0] or box_values[3] <= box_values[1]:
            continue
        candidates.append(
            {
                "box": box_values,
                "center": box_center(box_values),
                "confidence": confidence,
                "interpolated": False,
            }
        )
    return candidates


def candidate_matches_truth(
    candidate: dict[str, Any],
    row: dict[str, Any],
    gt_radius_px: float,
    bbox_iou: float,
) -> bool:
    if not bool(row.get("visible")):
        return False
    gt_point = row.get("point")
    if isinstance(gt_point, list) and len(gt_point) == 2:
        return center_distance(candidate["center"], gt_point) <= gt_radius_px
    gt_box = row.get("bbox")
    if isinstance(gt_box, list) and len(gt_box) == 4:
        return iou_xyxy(candidate["box"], gt_box) >= bbox_iou
    return False


def ball_matches_truth(
    ball: dict[str, Any] | None,
    row: dict[str, Any],
    gt_radius_px: float,
    bbox_iou: float,
) -> tuple[bool, float | None]:
    if ball is None or not bool(row.get("visible")):
        return False, None
    gt_point = row.get("point")
    if isinstance(gt_point, list) and len(gt_point) == 2:
        distance = center_distance(tuple(ball["center"]), gt_point)
        return distance <= gt_radius_px, distance
    gt_box = row.get("bbox")
    if isinstance(gt_box, list) and len(gt_box) == 4:
        overlap = iou_xyxy(ball["box"], gt_box)
        return overlap >= bbox_iou, None
    return False, None


def score_to_dict(score: BallCandidateScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    payload = asdict(score)
    return {
        key: round(float(value), 6) if isinstance(value, float) else value
        for key, value in payload.items()
    }


def safe_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def make_tracker(config: dict[str, Any], args: argparse.Namespace) -> TemporalBallTracker:
    return TemporalBallTracker(
        max_hold_frames=args.max_hold_frames,
        reset_after_frames=args.reset_after_frames,
        base_gate_norm=config["base_gate_norm"],
        gate_growth_per_frame=config["gate_growth_per_frame"],
        max_gate_norm=config["max_gate_norm"],
        velocity_alpha=args.velocity_alpha,
        score_confidence_weight=config["confidence_weight"],
        score_proximity_weight=config["proximity_weight"],
        score_size_weight=config["size_weight"],
        high_confidence_gate_threshold=config["high_confidence_gate_threshold"],
        high_confidence_gate_multiplier=config["high_confidence_gate_multiplier"],
    )


def evaluate_config(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    args: argparse.Namespace,
    frame_shape: tuple[int, int, int],
    keep_frames: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tracker = make_tracker(config, args)

    visible_frames = 0
    not_visible_frames = 0
    true_candidate_available = 0
    true_candidate_plausible = 0
    no_true_candidate = 0
    correct_direct = 0
    wrong_direct_visible = 0
    correct_predicted = 0
    wrong_predicted_visible = 0
    missing_visible = 0
    direct_not_visible = 0
    predicted_not_visible = 0
    missing_not_visible = 0
    score_loss_frames = 0
    gate_loss_frames = 0
    initialization_frames = 0
    initialization_errors = 0
    selected_high_conf_gate_false = 0

    correct_direct_distances: list[float] = []
    wrong_direct_distances: list[float] = []
    correct_selected_conf: list[float] = []
    wrong_selected_conf: list[float] = []
    correct_selected_proximity: list[float] = []
    wrong_selected_proximity: list[float] = []
    correct_selected_size: list[float] = []
    wrong_selected_size: list[float] = []
    score_margin_false_minus_true: list[float] = []
    frame_output: list[dict[str, Any]] = []

    for row in rows:
        frame_id = int(row.get("frame_id", 0))
        candidates = prepare_candidates(row, args.candidate_conf)
        visible = bool(row.get("visible"))
        if visible:
            visible_frames += 1
        else:
            not_visible_frames += 1

        true_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate_matches_truth(candidate, row, args.gt_radius_px, args.bbox_iou)
        ]
        if visible:
            if true_indices:
                true_candidate_available += 1
            else:
                no_true_candidate += 1

        pre_scores = [
            tracker.inspect_candidate(candidate, frame_id, frame_shape)
            for candidate in candidates
        ]
        temporal_scoring_active = any(score is not None for score in pre_scores)
        if candidates and not temporal_scoring_active:
            initialization_frames += 1

        plausible_true_indices = [
            index
            for index in true_indices
            if pre_scores[index] is None or bool(pre_scores[index].plausible)
        ]
        if visible and plausible_true_indices:
            true_candidate_plausible += 1

        decision = tracker.select(candidates, frame_id, frame_shape)
        correct, gt_distance = ball_matches_truth(
            decision.ball,
            row,
            args.gt_radius_px,
            args.bbox_iou,
        )

        selected_score_details = None
        if decision.selected_index is not None and 0 <= decision.selected_index < len(pre_scores):
            selected_score_details = pre_scores[decision.selected_index]

        if visible:
            if decision.source == "detector_associated":
                if correct:
                    correct_direct += 1
                    if gt_distance is not None:
                        correct_direct_distances.append(gt_distance)
                    if decision.ball is not None:
                        correct_selected_conf.append(float(decision.ball.get("confidence", 0.0)))
                    if selected_score_details is not None:
                        correct_selected_proximity.append(selected_score_details.proximity)
                        correct_selected_size.append(selected_score_details.size_similarity)
                else:
                    wrong_direct_visible += 1
                    if gt_distance is not None:
                        wrong_direct_distances.append(gt_distance)
                    if decision.ball is not None:
                        wrong_selected_conf.append(float(decision.ball.get("confidence", 0.0)))
                    if selected_score_details is not None:
                        wrong_selected_proximity.append(selected_score_details.proximity)
                        wrong_selected_size.append(selected_score_details.size_similarity)
                        if selected_score_details.high_confidence_gate_used:
                            selected_high_conf_gate_false += 1

                    if candidates and not temporal_scoring_active:
                        initialization_errors += 1
                    elif true_indices:
                        plausible_true_scores = [
                            pre_scores[index]
                            for index in true_indices
                            if pre_scores[index] is not None and pre_scores[index].plausible
                        ]
                        if plausible_true_scores:
                            score_loss_frames += 1
                            if selected_score_details is not None:
                                best_true_score = max(score.score for score in plausible_true_scores)
                                score_margin_false_minus_true.append(
                                    selected_score_details.score - best_true_score
                                )
                        else:
                            gate_loss_frames += 1
            elif decision.source == "predicted_hold":
                if correct:
                    correct_predicted += 1
                else:
                    wrong_predicted_visible += 1
                if true_indices:
                    if plausible_true_indices:
                        score_loss_frames += 1
                    else:
                        gate_loss_frames += 1
            else:
                missing_visible += 1
                if true_indices:
                    if plausible_true_indices:
                        score_loss_frames += 1
                    else:
                        gate_loss_frames += 1
        else:
            if decision.source == "detector_associated":
                direct_not_visible += 1
            elif decision.source == "predicted_hold":
                predicted_not_visible += 1
            else:
                missing_not_visible += 1

        if keep_frames:
            frame_output.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": row.get("timestamp_sec"),
                    "visible": visible,
                    "candidate_count": len(candidates),
                    "true_candidate_indices": true_indices,
                    "plausible_true_candidate_indices": plausible_true_indices,
                    "decision_source": decision.source,
                    "selected_index": decision.selected_index,
                    "selected_confidence": (
                        None if decision.ball is None else round(float(decision.ball.get("confidence", 0.0)), 6)
                    ),
                    "selected_gt_distance_px": rounded(gt_distance),
                    "selected_correct": bool(correct),
                    "association_score": rounded(decision.score),
                    "association_distance_norm": rounded(decision.distance_norm),
                    "gate_norm": rounded(decision.gate_norm),
                    "rejected_candidates": int(decision.rejected_candidates),
                    "selected_score_components": score_to_dict(selected_score_details),
                    "candidate_score_components": [score_to_dict(score) for score in pre_scores],
                }
            )

    tp = correct_direct
    fp = wrong_direct_visible + direct_not_visible
    fn = visible_frames - correct_direct
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = {
        "config_name": config["name"],
        "weight_set": config["weight_set"],
        "gate_set": config["gate_set"],
        "confidence_weight": round(config["confidence_weight"], 4),
        "proximity_weight": round(config["proximity_weight"], 4),
        "size_weight": round(config["size_weight"], 4),
        "base_gate_norm": round(config["base_gate_norm"], 4),
        "gate_growth_per_frame": round(config["gate_growth_per_frame"], 4),
        "max_gate_norm": round(config["max_gate_norm"], 4),
        "high_confidence_gate_multiplier": round(config["high_confidence_gate_multiplier"], 4),
        "visible_frames": visible_frames,
        "not_visible_frames": not_visible_frames,
        "true_candidate_available": true_candidate_available,
        "oracle_candidate_recall": round(true_candidate_available / visible_frames, 6) if visible_frames else 0.0,
        "true_candidate_plausible": true_candidate_plausible,
        "correct_direct": correct_direct,
        "wrong_direct_visible": wrong_direct_visible,
        "correct_predicted": correct_predicted,
        "wrong_predicted_visible": wrong_predicted_visible,
        "missing_visible": missing_visible,
        "direct_not_visible": direct_not_visible,
        "predicted_not_visible": predicted_not_visible,
        "missing_not_visible": missing_not_visible,
        "direct_precision": round(precision, 6),
        "direct_recall": round(recall, 6),
        "direct_f1": round(f1, 6),
        "association_accuracy_when_true_candidate_available": round(
            correct_direct / true_candidate_available, 6
        ) if true_candidate_available else 0.0,
        "score_loss_frames": score_loss_frames,
        "gate_loss_frames": gate_loss_frames,
        "initialization_frames": initialization_frames,
        "initialization_errors": initialization_errors,
        "selected_high_conf_gate_false": selected_high_conf_gate_false,
        "correct_direct_gt_distance_median_px": rounded(safe_median(correct_direct_distances)),
        "wrong_direct_gt_distance_median_px": rounded(safe_median(wrong_direct_distances)),
        "wrong_direct_gt_distance_p90_px": rounded(percentile(wrong_direct_distances, 0.90)),
        "correct_selected_conf_mean": rounded(safe_mean(correct_selected_conf)),
        "wrong_selected_conf_mean": rounded(safe_mean(wrong_selected_conf)),
        "correct_selected_proximity_mean": rounded(safe_mean(correct_selected_proximity)),
        "wrong_selected_proximity_mean": rounded(safe_mean(wrong_selected_proximity)),
        "correct_selected_size_similarity_mean": rounded(safe_mean(correct_selected_size)),
        "wrong_selected_size_similarity_mean": rounded(safe_mean(wrong_selected_size)),
        "wrong_minus_true_score_margin_mean": rounded(safe_mean(score_margin_false_minus_true)),
    }
    return metrics, frame_output


def build_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for weight_name, confidence_weight, proximity_weight, size_weight in WEIGHT_SETS:
        for (
            gate_name,
            base_gate_norm,
            gate_growth_per_frame,
            max_gate_norm,
            high_confidence_gate_threshold,
            high_confidence_gate_multiplier,
        ) in GATE_SETS:
            configs.append(
                {
                    "name": f"{weight_name}__{gate_name}",
                    "weight_set": weight_name,
                    "gate_set": gate_name,
                    "confidence_weight": confidence_weight,
                    "proximity_weight": proximity_weight,
                    "size_weight": size_weight,
                    "base_gate_norm": base_gate_norm,
                    "gate_growth_per_frame": gate_growth_per_frame,
                    "max_gate_norm": max_gate_norm,
                    "high_confidence_gate_threshold": high_confidence_gate_threshold,
                    "high_confidence_gate_multiplier": high_confidence_gate_multiplier,
                }
            )
    return configs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    per_frame_path = Path(args.per_frame).expanduser().resolve()
    rows = load_rows(per_frame_path)
    frame_shape = resolve_frame_shape(args)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else per_frame_path.parent / "association_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = [int(row.get("frame_id", 0)) for row in rows]
    frame_steps = [b - a for a, b in zip(frame_ids, frame_ids[1:]) if b > a]
    mean_frame_step = safe_mean([float(value) for value in frame_steps])

    grid = build_grid()
    grid_metrics: list[dict[str, Any]] = []
    for config in grid:
        metrics, _ = evaluate_config(rows, config, args, frame_shape, keep_frames=False)
        grid_metrics.append(metrics)

    grid_metrics.sort(
        key=lambda item: (
            float(item["direct_f1"]),
            float(item["direct_precision"]),
            int(item["correct_direct"]),
            -int(item["wrong_direct_visible"]),
        ),
        reverse=True,
    )

    config_by_name = {config["name"]: config for config in grid}
    current_name = "current__current"
    best_name = str(grid_metrics[0]["config_name"])
    current_metrics, current_frames = evaluate_config(
        rows,
        config_by_name[current_name],
        args,
        frame_shape,
        keep_frames=True,
    )
    best_metrics, best_frames = evaluate_config(
        rows,
        config_by_name[best_name],
        args,
        frame_shape,
        keep_frames=True,
    )

    write_csv(output_dir / "grid.csv", grid_metrics)
    (output_dir / "per_frame_current.json").write_text(
        json.dumps(current_frames, indent=2), encoding="utf-8"
    )
    (output_dir / "per_frame_best.json").write_text(
        json.dumps(best_frames, indent=2), encoding="utf-8"
    )

    summary = {
        "source": str(per_frame_path),
        "frame_shape": list(frame_shape),
        "frames_replayed": len(rows),
        "mean_annotated_frame_step": rounded(mean_frame_step),
        "candidate_conf": args.candidate_conf,
        "gt_radius_px": args.gt_radius_px,
        "bbox_iou": args.bbox_iou,
        "warning": (
            "This is a sparse replay over annotated evaluator frames, not an exact reproduction of the full-frame "
            "production tracker. Use it to diagnose scoring/gating and nominate configs; validate any winner on "
            "a separate clip and then in a full production run."
        ),
        "current": current_metrics,
        "best_on_this_replay": best_metrics,
        "top_5": grid_metrics[:5],
        "interpretation": {
            "score_loss_frames": (
                "A GT-consistent candidate was temporally plausible, but another direct candidate won the association score."
            ),
            "gate_loss_frames": (
                "A GT-consistent candidate existed in the captured pool but was outside the temporal gate."
            ),
            "initialization_errors": (
                "The tracker had no active temporal state/reset and highest confidence initialized it on the wrong object."
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote association benchmark to: {output_dir}")


if __name__ == "__main__":
    main()
