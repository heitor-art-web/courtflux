from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_tracker import BallCandidateScore, TemporalBallTracker

DIRECT_SOURCES = {
    "detector_associated",
    "detector_reacquired",
    "detector_reinitialized",
    "detector_highest_conf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine CourtFlux P0.2 recovery events and replay the exact persisted "
            "candidate stream with reacquisition OFF versus the current recovery policy."
        )
    )
    parser.add_argument("--frames", required=True, help="Instrumented production frames.jsonl")
    parser.add_argument("--truth", required=True, help="Sparse ball GT per_frame.json")
    parser.add_argument("--video", default=None, help="Optional source video for shape and contact sheets")
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    parser.add_argument("--video-frame-offset", type=int, default=0)
    parser.add_argument("--window-before", type=int, default=5)
    parser.add_argument("--window-after", type=int, default=10)
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--bbox-iou", type=float, default=0.25)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    rows.sort(key=lambda row: int(row.get("frame_id", 0)))
    if not rows:
        raise ValueError(f"No production rows found in {path}")
    return rows


def load_truth(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--truth must contain a top-level JSON list")
    rows = [dict(row) for row in payload if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("frame_id", 0)))
    if not rows:
        raise ValueError(f"No truth rows found in {path}")
    return rows


def resolve_frame_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.video:
        capture = cv2.VideoCapture(str(Path(args.video).expanduser()))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {args.video}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        if width > 0 and height > 0:
            return (height, width, 3)
    if args.frame_width and args.frame_height:
        if args.frame_width <= 0 or args.frame_height <= 0:
            raise ValueError("--frame-width and --frame-height must be positive")
        return (int(args.frame_height), int(args.frame_width), 3)
    raise ValueError("Provide --video or both --frame-width and --frame-height for exact replay")


def center_distance(a: Any, b: Any) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
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
    truth: dict[str, Any] | None,
    gt_radius_px: float,
    bbox_iou: float,
) -> tuple[bool, float | None]:
    if item is None or truth is None or not bool(truth.get("visible")):
        return False, None
    center, point = item.get("center"), truth.get("point")
    if isinstance(center, (list, tuple)) and len(center) >= 2 and isinstance(point, list) and len(point) == 2:
        distance = center_distance(center, point)
        return distance <= gt_radius_px, distance
    box, gt_box = item.get("box"), truth.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) == 4 and isinstance(gt_box, list) and len(gt_box) == 4:
        return iou_xyxy(list(box), gt_box) >= bbox_iou, None
    return False, None


def pct(num: int, den: int) -> float:
    return round(100.0 * num / max(den, 1), 2)


def rounded(value: Any, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def association(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("ball_association")
    return value if isinstance(value, dict) else {}


def persisted_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for trace in association(row).get("candidates") or []:
        if not isinstance(trace, dict):
            continue
        box, center = trace.get("box"), trace.get("center")
        if not isinstance(box, list) or len(box) != 4 or not isinstance(center, list) or len(center) < 2:
            continue
        output.append(
            {
                "box": [float(v) for v in box],
                "center": (float(center[0]), float(center[1])),
                "confidence": float(trace.get("confidence", 0.0)),
                "interpolated": False,
            }
        )
    return output


def persisted_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        config = association(row).get("config")
        if isinstance(config, dict):
            return config
    raise ValueError("No persisted tracker config found in frames.jsonl")


def make_tracker(config: dict[str, Any], reacquire_enabled: bool) -> TemporalBallTracker:
    weights = config.get("score_weights") or {}
    gate = config.get("gate") or {}
    recovery = config.get("reacquire") or {}
    return TemporalBallTracker(
        max_hold_frames=int(config.get("max_hold_frames", 5)),
        reset_after_frames=int(config.get("reset_after_frames", 12)),
        velocity_alpha=float(config.get("velocity_alpha", 0.40)),
        score_confidence_weight=float(weights.get("confidence", 0.58)),
        score_proximity_weight=float(weights.get("proximity", 0.32)),
        score_size_weight=float(weights.get("size", 0.10)),
        base_gate_norm=float(gate.get("base", 0.12)),
        gate_growth_per_frame=float(gate.get("growth_per_frame", 0.025)),
        max_gate_norm=float(gate.get("max", 0.28)),
        high_confidence_gate_threshold=float(gate.get("high_confidence_threshold", 0.85)),
        high_confidence_gate_multiplier=float(gate.get("high_confidence_multiplier", 1.35)),
        reacquire_enabled=reacquire_enabled,
        reacquire_min_confidence=float(recovery.get("min_confidence", 0.45)),
        reacquire_confidence_margin=float(recovery.get("confidence_margin", 0.15)),
        reacquire_confirm_frames=int(recovery.get("confirm_frames", 2)),
        reacquire_radius_norm=float(recovery.get("radius_norm", 0.04)),
        reacquire_min_size_similarity=float(recovery.get("min_size_similarity", 0.25)),
    )


def score_payload(score: BallCandidateScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "score": rounded(score.score),
        "confidence": rounded(score.confidence),
        "proximity": rounded(score.proximity),
        "size_similarity": rounded(score.size_similarity),
        "distance_norm": rounded(score.distance_norm),
        "gate_norm": rounded(score.gate_norm),
        "plausible": bool(score.plausible),
        "high_confidence_gate_used": bool(score.high_confidence_gate_used),
    }


def classify(source: str, ball: dict[str, Any] | None, correct: bool) -> str:
    if source in DIRECT_SOURCES:
        return "correct_direct" if correct else "wrong_direct"
    if source == "predicted_hold":
        return "correct_predicted" if correct else "wrong_predicted"
    return "missing" if ball is None else ("correct_other" if correct else "wrong_other")


def replay(
    rows: list[dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    frame_shape: tuple[int, int, int],
    config: dict[str, Any],
    reacquire_enabled: bool,
    gt_radius_px: float,
    bbox_iou: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    tracker = make_tracker(config, reacquire_enabled)
    records: dict[int, dict[str, Any]] = {}
    counts = Counter()
    source_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()

    for row in rows:
        frame_id = int(row.get("frame_id", 0))
        candidates = persisted_candidates(row)
        truth = truth_by_frame.get(frame_id)
        pre_state = {
            "track_state": str(tracker.track_state),
            "last_center": None if tracker.last_center is None else [rounded(v, 4) for v in tracker.last_center.tolist()],
            "last_frame_id": tracker.last_frame_id,
            "last_confidence": rounded(tracker.last_confidence),
            "last_area": rounded(tracker.last_area),
            "velocity": [rounded(v) for v in tracker.velocity.tolist()],
            "challenger_streak": int(tracker._challenger_streak),
            "challenger_confidence": rounded(tracker._challenger_confidence),
        }
        pre_scores = [tracker.inspect_candidate(c, frame_id, frame_shape) for c in candidates]
        previous_frame_id = tracker.last_frame_id
        reset = previous_frame_id is not None and frame_id - int(previous_frame_id) > tracker.reset_after_frames
        decision = tracker.select(candidates, frame_id, frame_shape)
        source = str(decision.source)
        if source == "detector_reacquired" and reset:
            source = "detector_reinitialized"
        correct, distance = item_matches_truth(decision.ball, truth, gt_radius_px, bbox_iou)
        true_indices = [i for i, c in enumerate(candidates) if item_matches_truth(c, truth, gt_radius_px, bbox_iou)[0]]

        if truth is not None:
            visible = bool(truth.get("visible"))
            counts["visible" if visible else "not_visible"] += 1
            if visible and true_indices:
                counts["candidate_available"] += 1
                if correct and source in DIRECT_SOURCES:
                    counts["conditioned_correct"] += 1
                else:
                    true_scores = [pre_scores[i] for i in true_indices if pre_scores[i] is not None]
                    if true_scores and any(s.plausible for s in true_scores):
                        counts["score_loss"] += 1
                    elif true_scores:
                        counts["gate_loss"] += 1
            if visible:
                if source in DIRECT_SOURCES:
                    counts["direct_visible"] += 1
                    counts["direct_correct" if correct else "direct_wrong"] += 1
                if decision.ball is not None:
                    counts["trajectory_present_visible"] += 1
                    if correct:
                        counts["trajectory_correct_visible"] += 1
            else:
                if source in DIRECT_SOURCES:
                    counts["direct_not_visible"] += 1
                if decision.ball is not None:
                    counts["trajectory_present_not_visible"] += 1
            if bool(decision.reacquired) and not reset:
                counts["reacquired"] += 1
                counts["reacquired_correct" if correct else "reacquired_wrong"] += 1

        source_counts[source] += 1
        state_counts[str(decision.track_state)] += 1
        selected_score = None
        if decision.selected_index is not None and 0 <= decision.selected_index < len(pre_scores):
            selected_score = pre_scores[decision.selected_index]
        records[frame_id] = {
            "frame_id": frame_id,
            "timestamp_sec": row.get("timestamp_sec"),
            "source": source,
            "track_state": str(decision.track_state),
            "selected_index": decision.selected_index,
            "selected_correct": bool(correct),
            "selected_distance_px": rounded(distance, 3),
            "classification": classify(source, decision.ball, correct),
            "trajectory_present": decision.ball is not None,
            "reacquired": bool(decision.reacquired and not reset),
            "candidate_count": len(candidates),
            "true_candidate_indices": true_indices,
            "true_candidate_available": bool(true_indices),
            "selected_score": score_payload(selected_score),
            "pre_state": pre_state,
        }

    visible = counts["visible"]
    available = counts["candidate_available"]
    metrics = {
        "reacquire_enabled": reacquire_enabled,
        "visible_frames": visible,
        "not_visible_frames": counts["not_visible"],
        "candidate_oracle": {
            "true_candidate_available_visible": available,
            "visible_candidate_recall_pct": pct(available, visible),
        },
        "association": {
            "opportunities_true_candidate_available": available,
            "correct_when_true_candidate_available": counts["conditioned_correct"],
            "accuracy_conditioned_on_true_candidate_pct": pct(counts["conditioned_correct"], available),
            "direct_visible": counts["direct_visible"],
            "direct_correct_visible": counts["direct_correct"],
            "direct_wrong_visible": counts["direct_wrong"],
            "direct_precision_visible_pct": pct(counts["direct_correct"], counts["direct_visible"]),
            "direct_recall_visible_pct": pct(counts["direct_correct"], visible),
            "direct_not_visible": counts["direct_not_visible"],
            "gate_loss": counts["gate_loss"],
            "score_or_selection_loss": counts["score_loss"],
            "source_counts": dict(source_counts),
            "track_state_counts": dict(state_counts),
        },
        "trajectory": {
            "present_visible": counts["trajectory_present_visible"],
            "correct_visible": counts["trajectory_correct_visible"],
            "visible_presence_pct": pct(counts["trajectory_present_visible"], visible),
            "visible_correct_recall_pct": pct(counts["trajectory_correct_visible"], visible),
            "present_not_visible": counts["trajectory_present_not_visible"],
        },
        "reacquisition": {
            "challenger_confirmed_frames": counts["reacquired"],
            "correct": counts["reacquired_correct"],
            "wrong": counts["reacquired_wrong"],
            "precision_pct": pct(counts["reacquired_correct"], counts["reacquired"]),
        },
    }
    return records, metrics


def event_frames(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row.get("frame_id", 0)) for row in rows if bool(association(row).get("reacquired"))]


def event_detail(
    frame_id: int,
    production_by_frame: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    before: int,
    after: int,
    gt_radius_px: float,
    bbox_iou: float,
) -> dict[str, Any]:
    row = production_by_frame[frame_id]
    assoc = association(row)
    truth = truth_by_frame.get(frame_id)
    traces = [dict(c) for c in assoc.get("candidates") or [] if isinstance(c, dict)]
    selected_index = assoc.get("selected_index")
    selected = traces[selected_index] if isinstance(selected_index, int) and 0 <= selected_index < len(traces) else None
    selected_correct = item_matches_truth(selected, truth, gt_radius_px, bbox_iou)[0]
    true_indices = [i for i, c in enumerate(traces) if item_matches_truth(c, truth, gt_radius_px, bbox_iou)[0]]
    selected_assoc = selected.get("association") if isinstance(selected, dict) and isinstance(selected.get("association"), dict) else {}
    distance_norm, gate_norm = selected_assoc.get("distance_norm"), selected_assoc.get("gate_norm")
    gate_ratio = None if distance_norm is None or gate_norm in (None, 0) else float(distance_norm) / float(gate_norm)
    selected_conf = None if selected is None else float(selected.get("confidence", 0.0))
    true_conf = [float(traces[i].get("confidence", 0.0)) for i in true_indices]
    best_true_conf = max(true_conf) if true_conf else None

    previous_ids = [fid for fid in production_by_frame if fid < frame_id]
    previous = production_by_frame[max(previous_ids)] if previous_ids else None
    previous_assoc = association(previous or {})

    off_event, on_event = off.get(frame_id), on.get(frame_id)
    if off_event and on_event and bool(truth and truth.get("visible")):
        if on_event["selected_correct"] and not off_event["selected_correct"]:
            label = "recovery_win"
        elif off_event["selected_correct"] and not on_event["selected_correct"]:
            label = "recovery_loss"
        elif on_event["selected_correct"] and off_event["selected_correct"]:
            label = "neutral_both_correct"
        else:
            label = "neutral_both_wrong"
    else:
        label = "not_comparable"

    timeline = []
    off_correct = on_correct = off_better = on_better = 0
    for truth_frame in sorted(fid for fid in truth_by_frame if frame_id - before <= fid <= frame_id + after):
        off_row, on_row = off.get(truth_frame), on.get(truth_frame)
        visible = bool(truth_by_frame[truth_frame].get("visible"))
        if visible and off_row and off_row["selected_correct"]:
            off_correct += 1
        if visible and on_row and on_row["selected_correct"]:
            on_correct += 1
        if visible and off_row and on_row:
            if off_row["selected_correct"] and not on_row["selected_correct"]:
                off_better += 1
            elif on_row["selected_correct"] and not off_row["selected_correct"]:
                on_better += 1
        timeline.append({"frame_id": truth_frame, "relative_frame": truth_frame - frame_id, "visible": visible, "off": off_row, "v2": on_row})

    return {
        "frame_id": frame_id,
        "timestamp_sec": row.get("timestamp_sec"),
        "production_reacquisition_correct": bool(selected_correct),
        "production_source": assoc.get("source"),
        "production_track_state": assoc.get("track_state"),
        "production_challenger_streak": int(assoc.get("challenger_streak", 0) or 0),
        "production_challenger_confidence": assoc.get("challenger_confidence"),
        "candidate_count": len(traces),
        "true_candidate_available": bool(true_indices),
        "true_candidate_indices": true_indices,
        "selected_index": selected_index,
        "selected_confidence": rounded(selected_conf),
        "best_true_confidence": rounded(best_true_conf),
        "selected_minus_best_true_confidence": None if selected_conf is None or best_true_conf is None else rounded(selected_conf - best_true_conf),
        "selected_distance_norm": rounded(distance_norm),
        "selected_gate_norm": rounded(gate_norm),
        "selected_gate_ratio": rounded(gate_ratio),
        "selected_size_similarity": rounded(selected_assoc.get("size_similarity")),
        "selected_proximity": rounded(selected_assoc.get("proximity")),
        "selected_score": rounded(selected_assoc.get("score")),
        "previous_production": {
            "frame_id": None if previous is None else int(previous.get("frame_id", 0)),
            "source": previous_assoc.get("source"),
            "track_state": previous_assoc.get("track_state"),
            "trajectory_confidence": previous_assoc.get("trajectory_confidence"),
            "hold_age_frames": previous_assoc.get("hold_age_frames"),
        },
        "v2_pre_state": None if on_event is None else on_event.get("pre_state"),
        "counterfactual_at_event": {"label": label, "off": off_event, "v2": on_event},
        "window": {
            "before_frames": before,
            "after_frames": after,
            "truth_samples": len(timeline),
            "off_correct_visible_truth_samples": off_correct,
            "v2_correct_visible_truth_samples": on_correct,
            "off_better_truth_samples": off_better,
            "v2_better_truth_samples": on_better,
            "net_v2_correct_delta": on_correct - off_correct,
            "timeline": timeline,
        },
    }


def flatten_event(event: dict[str, Any]) -> dict[str, Any]:
    pre = event.get("v2_pre_state") or {}
    prev = event.get("previous_production") or {}
    cf = event.get("counterfactual_at_event") or {}
    window = event.get("window") or {}
    return {
        "frame_id": event.get("frame_id"),
        "timestamp_sec": event.get("timestamp_sec"),
        "correct": event.get("production_reacquisition_correct"),
        "counterfactual_label": cf.get("label"),
        "candidate_count": event.get("candidate_count"),
        "true_candidate_available": event.get("true_candidate_available"),
        "selected_confidence": event.get("selected_confidence"),
        "best_true_confidence": event.get("best_true_confidence"),
        "selected_minus_best_true_confidence": event.get("selected_minus_best_true_confidence"),
        "distance_norm": event.get("selected_distance_norm"),
        "gate_norm": event.get("selected_gate_norm"),
        "gate_ratio": event.get("selected_gate_ratio"),
        "size_similarity": event.get("selected_size_similarity"),
        "proximity": event.get("selected_proximity"),
        "association_score": event.get("selected_score"),
        "challenger_streak": event.get("production_challenger_streak"),
        "challenger_confidence": event.get("production_challenger_confidence"),
        "pre_track_state": pre.get("track_state"),
        "pre_last_confidence": pre.get("last_confidence"),
        "pre_last_frame_id": pre.get("last_frame_id"),
        "previous_source": prev.get("source"),
        "previous_track_state": prev.get("track_state"),
        "previous_trajectory_confidence": prev.get("trajectory_confidence"),
        "window_truth_samples": window.get("truth_samples"),
        "off_correct_visible_truth_samples": window.get("off_correct_visible_truth_samples"),
        "v2_correct_visible_truth_samples": window.get("v2_correct_visible_truth_samples"),
        "off_better_truth_samples": window.get("off_better_truth_samples"),
        "v2_better_truth_samples": window.get("v2_better_truth_samples"),
        "net_v2_correct_delta": window.get("net_v2_correct_delta"),
    }


def aggregate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [e for e in events if e.get("production_reacquisition_correct")]
    wrong = [e for e in events if not e.get("production_reacquisition_correct")]

    def mean(group: list[dict[str, Any]], key: str) -> float | None:
        values = [float(e[key]) for e in group if e.get(key) is not None]
        return rounded(float(np.mean(values))) if values else None

    def window_mean(group: list[dict[str, Any]]) -> float | None:
        values = [float((e.get("window") or {}).get("net_v2_correct_delta")) for e in group if (e.get("window") or {}).get("net_v2_correct_delta") is not None]
        return rounded(float(np.mean(values))) if values else None

    labels = Counter(str((e.get("counterfactual_at_event") or {}).get("label")) for e in events)
    return {
        "events": len(events),
        "correct": len(correct),
        "wrong": len(wrong),
        "precision_pct": pct(len(correct), len(events)),
        "counterfactual_labels": dict(labels),
        "feature_means": {
            "correct": {
                "confidence": mean(correct, "selected_confidence"),
                "gate_ratio": mean(correct, "selected_gate_ratio"),
                "size_similarity": mean(correct, "selected_size_similarity"),
                "challenger_streak": mean(correct, "production_challenger_streak"),
                "net_v2_correct_delta": window_mean(correct),
            },
            "wrong": {
                "confidence": mean(wrong, "selected_confidence"),
                "gate_ratio": mean(wrong, "selected_gate_ratio"),
                "size_similarity": mean(wrong, "selected_size_similarity"),
                "challenger_streak": mean(wrong, "production_challenger_streak"),
                "net_v2_correct_delta": window_mean(wrong),
            },
        },
    }


def metric_delta(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    return {
        "association_accuracy_conditioned_pp": rounded(on["association"]["accuracy_conditioned_on_true_candidate_pct"] - off["association"]["accuracy_conditioned_on_true_candidate_pct"], 2),
        "direct_correct_visible": on["association"]["direct_correct_visible"] - off["association"]["direct_correct_visible"],
        "direct_precision_visible_pp": rounded(on["association"]["direct_precision_visible_pct"] - off["association"]["direct_precision_visible_pct"], 2),
        "direct_not_visible": on["association"]["direct_not_visible"] - off["association"]["direct_not_visible"],
        "gate_loss": on["association"]["gate_loss"] - off["association"]["gate_loss"],
        "trajectory_correct_recall_pp": rounded(on["trajectory"]["visible_correct_recall_pct"] - off["trajectory"]["visible_correct_recall_pct"], 2),
    }


def replay_agreement(rows: list[dict[str, Any]], on: dict[int, dict[str, Any]]) -> dict[str, Any]:
    comparable = index_match = source_match = 0
    for row in rows:
        frame_id = int(row.get("frame_id", 0))
        replay_row = on.get(frame_id)
        if replay_row is None:
            continue
        comparable += 1
        assoc = association(row)
        index_match += int(assoc.get("selected_index") == replay_row.get("selected_index"))
        source_match += int(str(assoc.get("source") or "missing") == replay_row.get("source"))
    return {
        "comparable_frames": comparable,
        "selected_index_match_pct": pct(index_match, comparable),
        "source_match_pct": pct(source_match, comparable),
    }


def draw_contact_sheet(
    video_path: Path,
    event: dict[str, Any],
    production_by_frame: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    output_path: Path,
    frame_offset: int,
    gt_radius_px: float,
    bbox_iou: float,
) -> None:
    event_id = int(event["frame_id"])
    window = event.get("window") or {}
    frame_ids = list(range(max(0, event_id - int(window.get("before_frames", 0))), event_id + int(window.get("after_frames", 0)) + 1))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    panels: list[np.ndarray] = []
    for frame_id in frame_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id + frame_offset)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        assoc = association(production_by_frame.get(frame_id, {}))
        traces = [c for c in assoc.get("candidates") or [] if isinstance(c, dict)]
        selected_index = assoc.get("selected_index")
        truth = truth_by_frame.get(frame_id)
        for index, candidate in enumerate(traces):
            box = candidate.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            is_true = item_matches_truth(candidate, truth, gt_radius_px, bbox_iou)[0]
            color, thickness = ((0, 180, 0), 1) if is_true else ((180, 180, 180), 1)
            if isinstance(selected_index, int) and index == selected_index:
                color, thickness = (0, 0, 255), 3
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        point = truth.get("point") if isinstance(truth, dict) else None
        if isinstance(point, list) and len(point) == 2:
            cv2.circle(frame, (int(round(float(point[0]))), int(round(float(point[1])))), max(4, int(round(gt_radius_px / 5))), (255, 0, 255), 2)
        label = f"f={frame_id} rel={frame_id - event_id:+d}" + (" REACQ" if frame_id == event_id else "")
        cv2.putText(frame, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"source={assoc.get('source', 'n/a')}", (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(frame, (480, 270)))
    capture.release()
    if not panels:
        return
    columns = min(4, len(panels))
    rows = int(math.ceil(len(panels) / columns))
    blank = np.zeros_like(panels[0])
    while len(panels) < rows * columns:
        panels.append(blank.copy())
    sheet = np.vstack([np.hstack(panels[i * columns:(i + 1) * columns]) for i in range(rows)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    frames_path = Path(args.frames).expanduser().resolve()
    truth_path = Path(args.truth).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else frames_path.parent / "recovery_error_mining"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(frames_path)
    truth_rows = load_truth(truth_path)
    frame_shape = resolve_frame_shape(args)
    production_by_frame = {int(row.get("frame_id", 0)): row for row in rows}
    truth_by_frame = {int(row.get("frame_id", 0)): row for row in truth_rows}
    if not any("candidates" in association(row) for row in rows):
        raise ValueError("No persisted P0.2 candidate traces found; use an instrumented run")

    config = persisted_config(rows)
    off, metrics_off = replay(rows, truth_by_frame, frame_shape, config, False, args.gt_radius_px, args.bbox_iou)
    on, metrics_on = replay(rows, truth_by_frame, frame_shape, config, True, args.gt_radius_px, args.bbox_iou)
    events = [
        event_detail(frame_id, production_by_frame, truth_by_frame, off, on, max(0, args.window_before), max(0, args.window_after), args.gt_radius_px, args.bbox_iou)
        for frame_id in event_frames(rows)
    ]

    probe = make_tracker(config, True)
    summary = {
        "scope": "P0.2 recovery error mining and counterfactual replay",
        "frames_file": str(frames_path),
        "truth_file": str(truth_path),
        "video_file": None if not args.video else str(Path(args.video).expanduser().resolve()),
        "frame_shape": list(frame_shape),
        "window_before": max(0, args.window_before),
        "window_after": max(0, args.window_after),
        "gt_radius_px": float(args.gt_radius_px),
        "bbox_iou": float(args.bbox_iou),
        "production_rows": len(rows),
        "truth_frames": len(truth_rows),
        "tracker_policy": getattr(probe, "reacquire_policy", None),
        "production_reacquisition_events": aggregate_events(events),
        "counterfactual": {
            "reacquire_off": metrics_off,
            "reacquire_on": metrics_on,
            "on_minus_off": metric_delta(metrics_off, metrics_on),
            "replay_vs_production_agreement": replay_agreement(rows, on),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "events.json").write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "events.csv", [flatten_event(event) for event in events])
    replay_truth = [{"frame_id": frame_id, "visible": bool(truth_by_frame[frame_id].get("visible")), "off": off.get(frame_id), "v2": on.get(frame_id)} for frame_id in sorted(truth_by_frame)]
    (output_dir / "replay_frames.json").write_text(json.dumps(replay_truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.video and not args.no_visuals:
        video_path = Path(args.video).expanduser().resolve()
        for event in events:
            status = "correct" if event["production_reacquisition_correct"] else "wrong"
            draw_contact_sheet(video_path, event, production_by_frame, truth_by_frame, output_dir / "visual" / f"event_{int(event['frame_id']):06d}_{status}.jpg", int(args.video_frame_offset), float(args.gt_radius_px), float(args.bbox_iou))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OUTPUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
