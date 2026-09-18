from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from evaluation.p2_ground_truth import (
    _event_window,
    _interval_distance,
    evaluate_events,
    load_ground_truth,
    load_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_CLIPS = ("V1", "V2", "V3", "V4", "V5")
DEVELOPMENT_CLIPS = ("V1", "V3", "V4", "V5")
CHALLENGE_CLIPS = ("V2",)
SHOT_EVENT = "shot_attempt"

SHOT_RECALL_MIN = 0.80
SHOT_PRECISION_MIN = 0.80
SHOT_OUTCOME_ACCURACY_MIN = 0.75
SAME_GT_WINDOW_DUPLICATES_MAX = 0


def _run_events_path(run_id: str) -> Path:
    return PROJECT_ROOT / "evaluation" / "runs" / run_id / "p2_events.jsonl"


def _shot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("event_type") or "") == SHOT_EVENT
    ]


def _same_gt_window_duplicates(
    predicted_shots: list[dict[str, Any]],
    truth_shots: list[dict[str, Any]],
    *,
    tolerance_frames: int,
) -> dict[str, Any]:
    assigned: dict[int, list[int]] = {index: [] for index in range(len(truth_shots))}
    rows: list[dict[str, Any]] = []

    for pred_index, pred in enumerate(predicted_shots):
        pred_start, pred_end = _event_window(pred)
        candidates: list[tuple[int, int]] = []
        for truth_index, truth in enumerate(truth_shots):
            truth_start, truth_end = _event_window(truth)
            distance = _interval_distance(pred_start, pred_end, truth_start, truth_end)
            if distance <= tolerance_frames:
                candidates.append((distance, truth_index))
        if not candidates:
            continue
        distance, truth_index = min(candidates)
        assigned[truth_index].append(pred_index)
        rows.append(
            {
                "prediction_index": pred_index,
                "truth_index": truth_index,
                "distance_frames": distance,
            }
        )

    duplicate_predictions = sum(max(0, len(indices) - 1) for indices in assigned.values())
    duplicate_windows = sum(1 for indices in assigned.values() if len(indices) > 1)
    return {
        "contract": "multiple shot_attempt predictions associated with one GT shot window count as duplicates",
        "duplicate_predictions": duplicate_predictions,
        "duplicate_gt_windows": duplicate_windows,
        "assignments": rows,
    }


def _event_gt_readiness(gt: dict[str, Any], truth_shots: list[dict[str, Any]]) -> dict[str, Any]:
    control_intervals = [
        row
        for row in list(gt.get("control_intervals", []) or [])
        if isinstance(row, dict)
    ]
    shot_flight_intervals = sum(
        1 for row in control_intervals if str(row.get("state") or "") == "shot_flight"
    )
    human_notes = " ".join(str(item) for item in list(gt.get("human_source_notes", []) or []))
    human_notes_lower = human_notes.lower()
    human_shot_signal = any(token in human_notes_lower for token in ("shot", "dunk", "free throw"))

    # Gate B requires exhaustive shot-event GT. A clip with explicit human shot evidence
    # but no shot_attempt events is not ready to contribute negative evidence/precision.
    complete = not (len(truth_shots) == 0 and (shot_flight_intervals > 0 or human_shot_signal))
    return {
        "complete_for_gate_b": complete,
        "shot_attempt_events": len(truth_shots),
        "shot_flight_intervals": shot_flight_intervals,
        "human_shot_signal": human_shot_signal,
        "reason": None
        if complete
        else "human annotation contains shot evidence but events[] has no shot_attempt labels",
    }


def _clip_metrics(clip: str, run_id: str, tolerance_frames: int) -> dict[str, Any]:
    events_path = _run_events_path(run_id)
    gt_path = PROJECT_ROOT / "evaluation" / "ground_truth" / f"{clip.lower()}.json"
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    predicted_events = load_jsonl(events_path)
    gt = load_ground_truth(gt_path)
    truth_events = [row for row in list(gt.get("events", []) or []) if isinstance(row, dict)]
    fps_raw = dict(gt.get("video", {}) or {}).get("fps")
    fps = None if fps_raw in {None, ""} else float(fps_raw)
    event_eval = evaluate_events(
        predicted_events,
        truth_events,
        tolerance_frames=tolerance_frames,
        fps=fps,
    )
    shot = dict(event_eval.get("by_type", {}).get(SHOT_EVENT, {}) or {})
    predicted_shots = _shot_rows(predicted_events)
    truth_shots = _shot_rows(truth_events)
    duplicates = _same_gt_window_duplicates(
        predicted_shots,
        truth_shots,
        tolerance_frames=tolerance_frames,
    )
    shot_matches = [
        row
        for row in list(event_eval.get("matches", []) or [])
        if isinstance(row, dict) and str(row.get("event_type") or "") == SHOT_EVENT
    ]
    truth_outcomes = Counter(str(row.get("truth_outcome") or "unlabeled") for row in shot_matches)
    predicted_outcomes = Counter(str(row.get("predicted_outcome") or "missing") for row in shot_matches)
    outcome_pairs = Counter(
        f"{str(row.get('truth_outcome') or 'unlabeled')}->{str(row.get('predicted_outcome') or 'missing')}"
        for row in shot_matches
    )
    return {
        "clip": clip,
        "run_id": run_id,
        "ground_truth_shots": len(truth_shots),
        "predicted_shots": len(predicted_shots),
        "event_gt": _event_gt_readiness(gt, truth_shots),
        "shot": shot,
        "outcome_diagnostics": {
            "truth": dict(sorted(truth_outcomes.items())),
            "predicted": dict(sorted(predicted_outcomes.items())),
            "pairs": dict(sorted(outcome_pairs.items())),
        },
        "duplicates": duplicates,
    }


def _aggregate(clips: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    outcome_labeled = outcome_correct = 0
    duplicate_predictions = duplicate_windows = 0
    gt_shots = predicted_shots = 0
    predicted_outcomes: Counter[str] = Counter()
    outcome_pairs: Counter[str] = Counter()

    for clip in clips:
        shot = clip["shot"]
        tp += int(shot.get("tp", 0) or 0)
        fp += int(shot.get("fp", 0) or 0)
        fn += int(shot.get("fn", 0) or 0)
        outcome_labeled += int(shot.get("outcome_labeled_matches", 0) or 0)
        outcome_correct += int(shot.get("outcome_correct", 0) or 0)
        duplicate_predictions += int(clip["duplicates"]["duplicate_predictions"])
        duplicate_windows += int(clip["duplicates"]["duplicate_gt_windows"])
        gt_shots += int(clip["ground_truth_shots"])
        predicted_shots += int(clip["predicted_shots"])
        predicted_outcomes.update(dict(clip["outcome_diagnostics"]["predicted"]))
        outcome_pairs.update(dict(clip["outcome_diagnostics"]["pairs"]))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    outcome_accuracy = None if outcome_labeled == 0 else outcome_correct / outcome_labeled
    return {
        "ground_truth_shots": gt_shots,
        "predicted_shots": predicted_shots,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "outcome_labeled_matches": outcome_labeled,
        "outcome_correct": outcome_correct,
        "outcome_accuracy": None if outcome_accuracy is None else round(outcome_accuracy, 4),
        "predicted_outcomes_on_labeled_matches": dict(sorted(predicted_outcomes.items())),
        "outcome_pairs": dict(sorted(outcome_pairs.items())),
        "same_gt_window_duplicate_predictions": duplicate_predictions,
        "duplicate_gt_windows": duplicate_windows,
    }


def _gate_checks(aggregate: dict[str, Any], clips: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_accuracy = aggregate["outcome_accuracy"]
    incomplete_gt = [
        row["clip"]
        for row in clips
        if not bool(dict(row.get("event_gt", {}) or {}).get("complete_for_gate_b", False))
    ]
    checks = [
        {
            "name": "development_event_gt_complete",
            "actual": incomplete_gt,
            "expected": "all development clips have exhaustive shot-event GT",
            "passed": not incomplete_gt,
        },
        {
            "name": "shot_attempt_recall",
            "actual": aggregate["recall"],
            "expected": f">= {SHOT_RECALL_MIN}",
            "passed": aggregate["recall"] >= SHOT_RECALL_MIN,
        },
        {
            "name": "shot_attempt_precision",
            "actual": aggregate["precision"],
            "expected": f">= {SHOT_PRECISION_MIN}",
            "passed": aggregate["precision"] >= SHOT_PRECISION_MIN,
        },
        {
            "name": "shot_outcome_accuracy",
            "actual": outcome_accuracy,
            "expected": f">= {SHOT_OUTCOME_ACCURACY_MIN}",
            "passed": outcome_accuracy is not None and outcome_accuracy >= SHOT_OUTCOME_ACCURACY_MIN,
        },
        {
            "name": "same_gt_window_duplicate_predictions",
            "actual": aggregate["same_gt_window_duplicate_predictions"],
            "expected": f"<= {SAME_GT_WINDOW_DUPLICATES_MAX}",
            "passed": aggregate["same_gt_window_duplicate_predictions"] <= SAME_GT_WINDOW_DUPLICATES_MAX,
        },
    ]
    return {
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks": checks,
        "required_checks": len(checks),
        "passed_checks": sum(1 for row in checks if row["passed"]),
        "failed_checks": sum(1 for row in checks if not row["passed"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P2 Gate B shot/event correctness.")
    for clip in ALL_CLIPS:
        parser.add_argument(f"--{clip.lower()}-run", required=True)
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output", default="evaluation/gate_b_scoreboard.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_map = {clip: str(getattr(args, f"{clip.lower()}_run")) for clip in ALL_CLIPS}
    clips = [_clip_metrics(clip, run_map[clip], args.tolerance_frames) for clip in ALL_CLIPS]
    development = [row for row in clips if row["clip"] in DEVELOPMENT_CLIPS]
    challenge_clips = [row for row in clips if row["clip"] in CHALLENGE_CLIPS]

    aggregate = _aggregate(development)
    gate = _gate_checks(aggregate, development)
    challenge_aggregate = _aggregate(challenge_clips)
    challenge_gate = _gate_checks(challenge_aggregate, challenge_clips)

    result = {
        "version": "courtflux_p2_gate_b_shot_event_v2",
        "contract": (
            "Gate B development blockers use V1/V3/V4/V5; V2 is challenge-only; "
            "shot existence is matched one-to-one to human windows; outcome is scored "
            "separately on matched labeled attempts; extra predictions associated with "
            "one GT shot window are duplicate attempts; development clips must have "
            "exhaustive shot-event GT before Gate B can pass"
        ),
        "development_clips": list(DEVELOPMENT_CLIPS),
        "tolerance_frames": int(args.tolerance_frames),
        "runs": run_map,
        "gate": gate,
        "aggregate": aggregate,
        "challenge": {
            "clips": list(CHALLENGE_CLIPS),
            "blocking": False,
            "diagnostic_gate": challenge_gate,
            "aggregate": challenge_aggregate,
        },
        "clips": clips,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
