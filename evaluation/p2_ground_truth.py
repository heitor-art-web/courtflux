from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ground truth must be a JSON object")
    if str(payload.get("version") or "") != "courtflux_p2_gt_v1":
        raise ValueError("ground truth version must be courtflux_p2_gt_v1")
    return payload


def _event_frame(row: dict[str, Any]) -> int:
    raw = row.get("frame", row.get("frame_end", row.get("frame_start", 0)))
    return int(raw)


def _event_compatible(predicted: dict[str, Any], truth: dict[str, Any]) -> bool:
    if str(predicted.get("event_type") or "") != str(truth.get("event_type") or ""):
        return False

    truth_outcome = truth.get("outcome")
    if truth_outcome not in {None, "", "unknown", "UNKNOWN"}:
        predicted_outcome = dict(predicted.get("attributes", {}) or {}).get("outcome")
        if str(predicted_outcome or "") != str(truth_outcome):
            return False

    for key in ("team_id", "actor_track_id", "secondary_track_id"):
        expected = truth.get(key)
        if expected is not None and predicted.get(key) != expected:
            return False
    return True


def evaluate_events(
    predicted: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    *,
    tolerance_frames: int = 8,
) -> dict[str, Any]:
    tolerance = max(0, int(tolerance_frames))
    used_predictions: set[int] = set()
    matched_truth: set[int] = set()
    match_rows: list[dict[str, Any]] = []

    for truth_index, truth_row in enumerate(truth):
        if not isinstance(truth_row, dict):
            continue
        truth_frame = _event_frame(truth_row)
        candidates: list[tuple[int, int]] = []
        for pred_index, pred_row in enumerate(predicted):
            if pred_index in used_predictions or not isinstance(pred_row, dict):
                continue
            if not _event_compatible(pred_row, truth_row):
                continue
            delta = abs(_event_frame(pred_row) - truth_frame)
            if delta <= tolerance:
                candidates.append((delta, pred_index))
        if not candidates:
            continue
        delta, pred_index = min(candidates)
        used_predictions.add(pred_index)
        matched_truth.add(truth_index)
        match_rows.append(
            {
                "truth_index": truth_index,
                "prediction_index": pred_index,
                "event_type": str(truth_row.get("event_type") or "unknown"),
                "frame_error": delta,
            }
        )

    event_types = sorted(
        {
            str(row.get("event_type") or "unknown")
            for row in predicted + truth
            if isinstance(row, dict)
        }
    )
    by_type: dict[str, Any] = {}
    for event_type in event_types:
        pred_indices = {
            index
            for index, row in enumerate(predicted)
            if isinstance(row, dict) and str(row.get("event_type") or "unknown") == event_type
        }
        truth_indices = {
            index
            for index, row in enumerate(truth)
            if isinstance(row, dict) and str(row.get("event_type") or "unknown") == event_type
        }
        matched_pred = {
            int(row["prediction_index"])
            for row in match_rows
            if row["event_type"] == event_type
        }
        matched_gt = {
            int(row["truth_index"])
            for row in match_rows
            if row["event_type"] == event_type
        }
        tp = len(matched_gt)
        fp = len(pred_indices - matched_pred)
        fn = len(truth_indices - matched_gt)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        errors = [
            int(row["frame_error"])
            for row in match_rows
            if row["event_type"] == event_type
        ]
        by_type[event_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_abs_frame_error": (
                None if not errors else round(sum(errors) / len(errors), 3)
            ),
        }

    return {
        "tolerance_frames": tolerance,
        "predicted_events": len(predicted),
        "ground_truth_events": len(truth),
        "matched_events": len(match_rows),
        "by_type": by_type,
        "matches": match_rows,
    }


def _truth_state_by_frame(intervals: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in intervals:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or "unknown")
        start = int(row.get("frame_start", row.get("frame", 0)))
        end = int(row.get("frame_end", start))
        for frame_id in range(start, max(start, end) + 1):
            result[frame_id] = state
    return result


def evaluate_control_states(
    predicted_states: list[dict[str, Any]],
    truth_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_by_frame = _truth_state_by_frame(truth_intervals)
    predicted_by_frame = {
        int(row.get("frame_id", -1)): str(row.get("control_state") or "unknown")
        for row in predicted_states
        if isinstance(row, dict) and int(row.get("frame_id", -1)) >= 0
    }

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    evaluated = 0
    missing_predictions = 0
    for frame_id, truth_state in sorted(truth_by_frame.items()):
        predicted_state = predicted_by_frame.get(frame_id)
        if predicted_state is None:
            missing_predictions += 1
            continue
        evaluated += 1
        confusion[truth_state][predicted_state] += 1
        if predicted_state == truth_state:
            correct += 1

    return {
        "ground_truth_labeled_frames": len(truth_by_frame),
        "evaluated_frames": evaluated,
        "missing_prediction_frames": missing_predictions,
        "accuracy": round(correct / max(evaluated, 1), 4),
        "confusion": {
            truth_state: dict(sorted(counter.items()))
            for truth_state, counter in sorted(confusion.items())
        },
    }


def evaluate_payload(
    *,
    predicted_events: list[dict[str, Any]],
    predicted_states: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    tolerance_frames: int = 8,
) -> dict[str, Any]:
    return {
        "version": "courtflux_p2_gt_eval_v1",
        "events": evaluate_events(
            predicted_events,
            list(ground_truth.get("events", []) or []),
            tolerance_frames=tolerance_frames,
        ),
        "control_states": evaluate_control_states(
            predicted_states,
            list(ground_truth.get("control_intervals", []) or []),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P2 events/control states against human GT.")
    parser.add_argument("--events", required=True, help="p2_events.jsonl")
    parser.add_argument("--states", required=True, help="p2_states.jsonl")
    parser.add_argument("--ground-truth", required=True, help="courtflux_p2_gt_v1 JSON")
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_payload(
        predicted_events=load_jsonl(args.events),
        predicted_states=load_jsonl(args.states),
        ground_truth=load_ground_truth(args.ground_truth),
        tolerance_frames=args.tolerance_frames,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
