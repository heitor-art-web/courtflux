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


def _event_window(row: dict[str, Any]) -> tuple[int, int]:
    """Return a human-label event window.

    Point labels remain valid: ``frame`` becomes a one-frame window.
    Interval labels use ``frame_start``/``frame_end`` and intentionally do not
    require the human reviewer to identify the exact release frame.
    """
    if row.get("frame") is not None:
        frame = int(row["frame"])
        return frame, frame

    start_raw = row.get("frame_start", row.get("frame_end", 0))
    end_raw = row.get("frame_end", start_raw)
    start = int(start_raw)
    end = int(end_raw)
    if end < start:
        start, end = end, start
    return start, end


def _distance_to_window(frame: int, start: int, end: int) -> int:
    if start <= frame <= end:
        return 0
    if frame < start:
        return start - frame
    return frame - end


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
    fps: float | None = None,
) -> dict[str, Any]:
    """Match predicted event points against human point-or-window labels.

    A prediction anywhere inside a human interval has zero temporal error.
    ``tolerance_frames`` extends only beyond the human window edges.
    """
    tolerance = max(0, int(tolerance_frames))
    used_predictions: set[int] = set()
    matched_truth: set[int] = set()
    match_rows: list[dict[str, Any]] = []

    for truth_index, truth_row in enumerate(truth):
        if not isinstance(truth_row, dict):
            continue
        truth_start, truth_end = _event_window(truth_row)
        candidates: list[tuple[int, int]] = []
        for pred_index, pred_row in enumerate(predicted):
            if pred_index in used_predictions or not isinstance(pred_row, dict):
                continue
            if not _event_compatible(pred_row, truth_row):
                continue
            pred_frame = _event_frame(pred_row)
            delta = _distance_to_window(pred_frame, truth_start, truth_end)
            if delta <= tolerance:
                candidates.append((delta, pred_index))
        if not candidates:
            continue

        delta, pred_index = min(candidates)
        pred_frame = _event_frame(predicted[pred_index])
        used_predictions.add(pred_index)
        matched_truth.add(truth_index)
        match_rows.append(
            {
                "truth_index": truth_index,
                "prediction_index": pred_index,
                "event_type": str(truth_row.get("event_type") or "unknown"),
                "truth_frame_start": truth_start,
                "truth_frame_end": truth_end,
                "prediction_frame": pred_frame,
                "inside_truth_window": truth_start <= pred_frame <= truth_end,
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
            if isinstance(row, dict)
            and str(row.get("event_type") or "unknown") == event_type
        }
        truth_indices = {
            index
            for index, row in enumerate(truth)
            if isinstance(row, dict)
            and str(row.get("event_type") or "unknown") == event_type
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
        inside_count = sum(
            1
            for row in match_rows
            if row["event_type"] == event_type and row["inside_truth_window"]
        )
        mean_error_frames = None if not errors else round(sum(errors) / len(errors), 3)
        mean_error_sec = None
        if mean_error_frames is not None and fps and fps > 0:
            mean_error_sec = round(mean_error_frames / float(fps), 4)

        by_type[event_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "matched_inside_human_window": inside_count,
            "mean_distance_to_human_window_frames": mean_error_frames,
            "mean_distance_to_human_window_sec": mean_error_sec,
            "mean_abs_frame_error": mean_error_frames,
        }

    return {
        "matching_contract": "prediction_point_to_human_event_window",
        "tolerance_frames_beyond_window": tolerance,
        "tolerance_frames": tolerance,
        "predicted_events": len(predicted),
        "ground_truth_events": len(truth),
        "ground_truth_window_events": sum(
            1
            for row in truth
            if isinstance(row, dict)
            and row.get("frame") is None
            and (row.get("frame_start") is not None or row.get("frame_end") is not None)
        ),
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


def _predicted_state_by_frame(predicted_states: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(row.get("frame_id", -1)): str(row.get("control_state") or "unknown")
        for row in predicted_states
        if isinstance(row, dict) and int(row.get("frame_id", -1)) >= 0
    }


def _evaluate_control_intervals(
    predicted_by_frame: dict[int, str],
    truth_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate continuity per human interval, not only boundary-exact frames."""
    rows: list[dict[str, Any]] = []
    evaluated = 0
    dominant_correct = 0
    matching_rates: list[float] = []

    for truth_index, row in enumerate(truth_intervals):
        if not isinstance(row, dict):
            continue
        truth_state = str(row.get("state") or "unknown")
        start = int(row.get("frame_start", row.get("frame", 0)))
        end = int(row.get("frame_end", start))
        if end < start:
            start, end = end, start

        observed = [
            predicted_by_frame[frame_id]
            for frame_id in range(start, end + 1)
            if frame_id in predicted_by_frame
        ]
        if not observed:
            rows.append(
                {
                    "truth_index": truth_index,
                    "state": truth_state,
                    "frame_start": start,
                    "frame_end": end,
                    "observed_frames": 0,
                    "matching_rate": None,
                    "dominant_predicted_state": None,
                    "dominant_matches_truth": None,
                }
            )
            continue

        evaluated += 1
        counts = Counter(observed)
        dominant_state, _ = counts.most_common(1)[0]
        matching_rate = counts[truth_state] / len(observed)
        matching_rates.append(matching_rate)
        dominant_matches = dominant_state == truth_state
        if dominant_matches:
            dominant_correct += 1

        rows.append(
            {
                "truth_index": truth_index,
                "state": truth_state,
                "frame_start": start,
                "frame_end": end,
                "observed_frames": len(observed),
                "matching_rate": round(matching_rate, 4),
                "dominant_predicted_state": dominant_state,
                "dominant_matches_truth": dominant_matches,
            }
        )

    return {
        "contract": "human_interval_continuity",
        "ground_truth_intervals": len(
            [row for row in truth_intervals if isinstance(row, dict)]
        ),
        "evaluated_intervals": evaluated,
        "dominant_state_interval_accuracy": round(
            dominant_correct / max(evaluated, 1), 4
        ),
        "mean_interval_matching_rate": (
            None
            if not matching_rates
            else round(sum(matching_rates) / len(matching_rates), 4)
        ),
        "intervals": rows,
    }


def evaluate_control_states(
    predicted_states: list[dict[str, Any]],
    truth_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_by_frame = _truth_state_by_frame(truth_intervals)
    predicted_by_frame = _predicted_state_by_frame(predicted_states)

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
        "scoring_contract": "continuity_primary_frame_accuracy_diagnostic",
        "ground_truth_labeled_frames": len(truth_by_frame),
        "evaluated_frames": evaluated,
        "missing_prediction_frames": missing_predictions,
        "accuracy": round(correct / max(evaluated, 1), 4),
        "coverage_adjusted_accuracy": round(
            correct / max(len(truth_by_frame), 1), 4
        ),
        "confusion": {
            truth_state: dict(sorted(counter.items()))
            for truth_state, counter in sorted(confusion.items())
        },
        "continuity": _evaluate_control_intervals(
            predicted_by_frame,
            truth_intervals,
        ),
    }


def evaluate_payload(
    *,
    predicted_events: list[dict[str, Any]],
    predicted_states: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    tolerance_frames: int = 8,
) -> dict[str, Any]:
    video = dict(ground_truth.get("video", {}) or {})
    fps_raw = video.get("fps")
    fps = float(fps_raw) if fps_raw not in {None, ""} else None

    return {
        "version": "courtflux_p2_gt_eval_v1",
        "events": evaluate_events(
            predicted_events,
            list(ground_truth.get("events", []) or []),
            tolerance_frames=tolerance_frames,
            fps=fps,
        ),
        "control_states": evaluate_control_states(
            predicted_states,
            list(ground_truth.get("control_intervals", []) or []),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate P2 events/control states against human GT windows."
    )
    parser.add_argument("--events", required=True, help="p2_events.jsonl")
    parser.add_argument("--states", required=True, help="p2_states.jsonl")
    parser.add_argument("--ground-truth", required=True, help="courtflux_p2_gt_v1 JSON")
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=8,
        help="Tolerance outside human event-window edges; predictions inside a window have zero temporal error.",
    )
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
