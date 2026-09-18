from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import evaluate_payload, load_ground_truth, load_jsonl

CLIPS = ("V1", "V2", "V3", "V4", "V5")
CORE_STATES = ("possessed", "pass_flight", "shot_flight")

COVERAGE_AGG_MIN = 0.98
COVERAGE_CLIP_MIN = 0.95
CORE_INTERVAL_RECALL_MIN = 0.70
POSSESSED_INTERVAL_RECALL_MIN = 0.70
PASS_INTERVAL_RECALL_MIN = 0.60
SHOT_INTERVAL_RECALL_MIN = 0.70
SEQUENCE_ORDER_ACCURACY_MIN = 0.90
CORE_UNRESOLVED_MAX = 0.25
PASS_AS_SHOT_MAX = 0.15
SHOT_AS_PASS_MAX = 0.15
CLIP_CORE_INTERVAL_RECALL_MIN = 0.40
CLIP_CORE_UNRESOLVED_MAX = 0.45


def _run_paths(run_id: str) -> tuple[Path, Path, Path]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    return run_dir, run_dir / "p2_events.jsonl", run_dir / "p2_states.jsonl"


def _interval_distance(a: dict[str, Any], b: dict[str, Any]) -> int:
    a_start, a_end = int(a["frame_start"]), int(a["frame_end"])
    b_start, b_end = int(b["frame_start"]), int(b["frame_end"])
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def _overlap_frames(a: dict[str, Any], b: dict[str, Any]) -> int:
    start = max(int(a["frame_start"]), int(b["frame_start"]))
    end = min(int(a["frame_end"]), int(b["frame_end"]))
    return max(0, end - start + 1)


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = _overlap_frames(a, b)
    if overlap <= 0:
        return 0.0
    start = min(int(a["frame_start"]), int(b["frame_start"]))
    end = max(int(a["frame_end"]), int(b["frame_end"]))
    return overlap / max(1, end - start + 1)


def _truth_intervals(gt: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(list(gt.get("control_intervals", []) or [])):
        if not isinstance(raw, dict):
            continue
        state = str(raw.get("state") or "unknown")
        start = int(raw.get("frame_start", raw.get("frame", 0)))
        end = int(raw.get("frame_end", start))
        if end < start:
            start, end = end, start
        rows.append(
            {
                "truth_index": index,
                "state": state,
                "frame_start": start,
                "frame_end": end,
            }
        )
    return rows


def _predicted_intervals(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames: list[tuple[int, str]] = []
    for row in states:
        if not isinstance(row, dict):
            continue
        try:
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if frame_id >= 0:
            frames.append((frame_id, str(row.get("control_state") or "unknown")))
    frames.sort()
    if not frames:
        return []

    intervals: list[dict[str, Any]] = []
    start, previous, state = frames[0][0], frames[0][0], frames[0][1]
    for frame_id, next_state in frames[1:]:
        if frame_id == previous + 1 and next_state == state:
            previous = frame_id
            continue
        intervals.append(
            {
                "prediction_index": len(intervals),
                "state": state,
                "frame_start": start,
                "frame_end": previous,
            }
        )
        start, previous, state = frame_id, frame_id, next_state
    intervals.append(
        {
            "prediction_index": len(intervals),
            "state": state,
            "frame_start": start,
            "frame_end": previous,
        }
    )
    return intervals


def _gt_window_coverage(
    truth: list[dict[str, Any]], predicted_states: list[dict[str, Any]]
) -> tuple[int, int, float]:
    predicted_frames = {
        int(row.get("frame_id", -1))
        for row in predicted_states
        if isinstance(row, dict) and int(row.get("frame_id", -1)) >= 0
    }
    gt_frames: set[int] = set()
    for row in truth:
        if row["state"] in CORE_STATES:
            gt_frames.update(range(int(row["frame_start"]), int(row["frame_end"]) + 1))
    observed = len(gt_frames & predicted_frames)
    return len(gt_frames), observed, round(observed / max(len(gt_frames), 1), 4)


def _match_intervals(
    truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    *,
    tolerance_frames: int,
) -> dict[str, Any]:
    core_truth = [row for row in truth if row["state"] in CORE_STATES]
    core_pred = [row for row in predicted if row["state"] in CORE_STATES]

    same_state_pairs: list[tuple[int, int, int, int, int]] = []
    for ti, t in enumerate(core_truth):
        for pi, p in enumerate(core_pred):
            if t["state"] != p["state"]:
                continue
            distance = _interval_distance(t, p)
            if distance <= tolerance_frames:
                same_state_pairs.append(
                    (distance, -_overlap_frames(t, p), ti, pi, int(p["frame_start"]))
                )

    used_truth: set[int] = set()
    used_pred: set[int] = set()
    correct_matches: dict[int, int] = {}
    for _distance, _neg_overlap, ti, pi, _start in sorted(same_state_pairs):
        if ti in used_truth or pi in used_pred:
            continue
        used_truth.add(ti)
        used_pred.add(pi)
        correct_matches[ti] = pi

    wrong_type_pairs: list[tuple[int, int, int, int, int]] = []
    for ti, t in enumerate(core_truth):
        if ti in used_truth:
            continue
        for pi, p in enumerate(core_pred):
            if pi in used_pred or t["state"] == p["state"]:
                continue
            distance = _interval_distance(t, p)
            if distance <= tolerance_frames:
                wrong_type_pairs.append(
                    (distance, -_overlap_frames(t, p), ti, pi, int(p["frame_start"]))
                )

    wrong_matches: dict[int, int] = {}
    for _distance, _neg_overlap, ti, pi, _start in sorted(wrong_type_pairs):
        if ti in used_truth or pi in used_pred:
            continue
        used_truth.add(ti)
        used_pred.add(pi)
        wrong_matches[ti] = pi

    by_state: dict[str, Counter[str]] = defaultdict(Counter)
    interval_rows: list[dict[str, Any]] = []
    boundary_ious: list[float] = []
    start_errors: list[int] = []
    end_errors: list[int] = []

    for ti, t in enumerate(core_truth):
        if ti in correct_matches:
            p = core_pred[correct_matches[ti]]
            resolution = "correct"
            predicted_state = p["state"]
            iou = _iou(t, p)
            boundary_ious.append(iou)
            start_errors.append(int(p["frame_start"]) - int(t["frame_start"]))
            end_errors.append(int(p["frame_end"]) - int(t["frame_end"]))
        elif ti in wrong_matches:
            p = core_pred[wrong_matches[ti]]
            resolution = "wrong_type"
            predicted_state = p["state"]
            iou = _iou(t, p)
        else:
            p = None
            resolution = "unresolved"
            predicted_state = None
            iou = None

        by_state[t["state"]][resolution] += 1
        if predicted_state is not None:
            by_state[t["state"]][f"predicted:{predicted_state}"] += 1
        interval_rows.append(
            {
                **t,
                "resolution": resolution,
                "predicted_state": predicted_state,
                "prediction_frame_start": None if p is None else p["frame_start"],
                "prediction_frame_end": None if p is None else p["frame_end"],
                "distance_frames": None if p is None else _interval_distance(t, p),
                "overlap_frames": None if p is None else _overlap_frames(t, p),
                "iou": None if iou is None else round(iou, 4),
            }
        )

    state_summary: dict[str, Any] = {}
    for state in CORE_STATES:
        counter = by_state.get(state, Counter())
        total = sum(counter[key] for key in ("correct", "wrong_type", "unresolved"))
        state_summary[state] = {
            "intervals": total,
            "correct": int(counter.get("correct", 0)),
            "wrong_type": int(counter.get("wrong_type", 0)),
            "unresolved": int(counter.get("unresolved", 0)),
            "recall": round(counter.get("correct", 0) / max(total, 1), 4),
            "unresolved_rate": round(counter.get("unresolved", 0) / max(total, 1), 4),
            "predicted_as": {
                pred_state: int(counter.get(f"predicted:{pred_state}", 0))
                for pred_state in CORE_STATES
                if counter.get(f"predicted:{pred_state}", 0) > 0
            },
        }

    core_total = len(core_truth)
    core_correct = sum(1 for row in interval_rows if row["resolution"] == "correct")
    core_unresolved = sum(1 for row in interval_rows if row["resolution"] == "unresolved")

    gt_transition_total = max(0, len(core_truth) - 1)
    eligible_detected = 0
    correct_order = 0
    transition_rows: list[dict[str, Any]] = []
    for left_index in range(gt_transition_total):
        right_index = left_index + 1
        left = core_truth[left_index]
        right = core_truth[right_index]
        left_pi = correct_matches.get(left_index)
        right_pi = correct_matches.get(right_index)
        eligible = left_pi is not None and right_pi is not None
        ordered = False
        if eligible:
            eligible_detected += 1
            ordered = int(core_pred[left_pi]["frame_start"]) <= int(
                core_pred[right_pi]["frame_start"]
            )
            if ordered:
                correct_order += 1
        transition_rows.append(
            {
                "from_state": left["state"],
                "to_state": right["state"],
                "eligible_detected": bool(eligible),
                "correct_order": bool(ordered) if eligible else None,
            }
        )

    pass_summary = state_summary["pass_flight"]
    shot_summary = state_summary["shot_flight"]
    pass_as_shot = int(by_state.get("pass_flight", Counter()).get("predicted:shot_flight", 0))
    shot_as_pass = int(by_state.get("shot_flight", Counter()).get("predicted:pass_flight", 0))

    return {
        "contract": "P3 semantic interval detection; sequence order is conditioned on both adjacent episodes being detected",
        "tolerance_frames": int(tolerance_frames),
        "core_intervals": core_total,
        "correct_core_intervals": core_correct,
        "core_interval_detection_recall": round(core_correct / max(core_total, 1), 4),
        "core_unresolved_intervals": core_unresolved,
        "core_unresolved_rate": round(core_unresolved / max(core_total, 1), 4),
        "by_state": state_summary,
        "critical_confusions": {
            "pass_interval_as_shot": round(pass_as_shot / max(pass_summary["intervals"], 1), 4),
            "shot_interval_as_pass": round(shot_as_pass / max(shot_summary["intervals"], 1), 4),
        },
        "semantic_transitions": {
            "gt_total": gt_transition_total,
            "eligible_detected": eligible_detected,
            "correct_order": correct_order,
            "detected_transition_coverage": round(
                eligible_detected / max(gt_transition_total, 1), 4
            ),
            "order_accuracy": None
            if eligible_detected == 0
            else round(correct_order / eligible_detected, 4),
            "rows": transition_rows,
        },
        "boundary_diagnostics": {
            "matched_intervals": len(boundary_ious),
            "mean_iou": None if not boundary_ious else round(sum(boundary_ious) / len(boundary_ious), 4),
            "mean_absolute_start_error_frames": None
            if not start_errors
            else round(sum(abs(value) for value in start_errors) / len(start_errors), 3),
            "mean_absolute_end_error_frames": None
            if not end_errors
            else round(sum(abs(value) for value in end_errors) / len(end_errors), 3),
        },
        "intervals": interval_rows,
    }


def _clip_metrics(clip: str, run_id: str, tolerance_frames: int) -> dict[str, Any]:
    _run_dir, events_path, states_path = _run_paths(run_id)
    gt_path = PROJECT_ROOT / "evaluation" / "ground_truth" / f"{clip.lower()}.json"
    missing = [str(path) for path in (events_path, states_path, gt_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{clip}: missing required files: {missing}")

    predicted_events = load_jsonl(events_path)
    predicted_states = load_jsonl(states_path)
    gt = load_ground_truth(gt_path)
    legacy = evaluate_payload(
        predicted_events=predicted_events,
        predicted_states=predicted_states,
        ground_truth=gt,
        tolerance_frames=tolerance_frames,
    )
    truth = _truth_intervals(gt)
    predicted = _predicted_intervals(predicted_states)
    timeline = _match_intervals(truth, predicted, tolerance_frames=tolerance_frames)
    gt_frames, observed_frames, coverage = _gt_window_coverage(truth, predicted_states)
    return {
        "clip": clip,
        "run_id": run_id,
        "gt_file": str(gt_path.relative_to(PROJECT_ROOT)),
        "gt_window_frames": gt_frames,
        "observed_gt_window_frames": observed_frames,
        "prediction_coverage": coverage,
        "timeline": timeline,
        "frame_diagnostics": {
            "frame_accuracy": legacy["control_states"]["accuracy"],
            "dominant_state_interval_accuracy": legacy["control_states"]["continuity"]["dominant_state_interval_accuracy"],
            "mean_interval_matching_rate": legacy["control_states"]["continuity"]["mean_interval_matching_rate"],
        },
    }


def _aggregate(clips: list[dict[str, Any]]) -> dict[str, Any]:
    gt_frames = sum(int(row["gt_window_frames"]) for row in clips)
    observed_frames = sum(int(row["observed_gt_window_frames"]) for row in clips)
    by_state: dict[str, Counter[str]] = defaultdict(Counter)
    core_total = core_correct = core_unresolved = 0
    gt_transitions = eligible_transitions = correct_order = 0

    for clip in clips:
        timeline = clip["timeline"]
        core_total += int(timeline["core_intervals"])
        core_correct += int(timeline["correct_core_intervals"])
        core_unresolved += int(timeline["core_unresolved_intervals"])
        semantic = timeline["semantic_transitions"]
        gt_transitions += int(semantic["gt_total"])
        eligible_transitions += int(semantic["eligible_detected"])
        correct_order += int(semantic["correct_order"])
        for state in CORE_STATES:
            row = timeline["by_state"][state]
            for key in ("intervals", "correct", "wrong_type", "unresolved"):
                by_state[state][key] += int(row[key])
            for pred_state, count in dict(row.get("predicted_as", {})).items():
                by_state[state][f"predicted:{pred_state}"] += int(count)

    state_summary: dict[str, Any] = {}
    for state in CORE_STATES:
        counter = by_state[state]
        total = int(counter["intervals"])
        state_summary[state] = {
            "intervals": total,
            "correct": int(counter["correct"]),
            "wrong_type": int(counter["wrong_type"]),
            "unresolved": int(counter["unresolved"]),
            "recall": round(counter["correct"] / max(total, 1), 4),
            "unresolved_rate": round(counter["unresolved"] / max(total, 1), 4),
        }

    pass_total = state_summary["pass_flight"]["intervals"]
    shot_total = state_summary["shot_flight"]["intervals"]
    pass_as_shot = int(by_state["pass_flight"].get("predicted:shot_flight", 0))
    shot_as_pass = int(by_state["shot_flight"].get("predicted:pass_flight", 0))
    order_accuracy = (
        None
        if eligible_transitions == 0
        else round(correct_order / eligible_transitions, 4)
    )

    return {
        "prediction_coverage": round(observed_frames / max(gt_frames, 1), 4),
        "core_intervals": core_total,
        "correct_core_intervals": core_correct,
        "core_interval_detection_recall": round(core_correct / max(core_total, 1), 4),
        "core_unresolved_intervals": core_unresolved,
        "core_unresolved_rate": round(core_unresolved / max(core_total, 1), 4),
        "by_state": state_summary,
        "semantic_transition_order_accuracy": order_accuracy,
        "semantic_transitions": {
            "gt_total": gt_transitions,
            "eligible_detected": eligible_transitions,
            "correct_order": correct_order,
            "detected_transition_coverage": round(
                eligible_transitions / max(gt_transitions, 1), 4
            ),
        },
        "critical_confusions": {
            "pass_interval_as_shot": round(pass_as_shot / max(pass_total, 1), 4),
            "shot_interval_as_pass": round(shot_as_pass / max(shot_total, 1), 4),
        },
    }


def _gate_checks(clips: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, expected: str, passed: bool) -> None:
        checks.append(
            {"name": name, "actual": actual, "expected": expected, "passed": bool(passed)}
        )

    add("aggregate_prediction_coverage", aggregate["prediction_coverage"], f">= {COVERAGE_AGG_MIN}", aggregate["prediction_coverage"] >= COVERAGE_AGG_MIN)
    add("aggregate_core_interval_detection_recall", aggregate["core_interval_detection_recall"], f">= {CORE_INTERVAL_RECALL_MIN}", aggregate["core_interval_detection_recall"] >= CORE_INTERVAL_RECALL_MIN)
    add("aggregate_possessed_interval_recall", aggregate["by_state"]["possessed"]["recall"], f">= {POSSESSED_INTERVAL_RECALL_MIN}", aggregate["by_state"]["possessed"]["recall"] >= POSSESSED_INTERVAL_RECALL_MIN)
    add("aggregate_pass_interval_recall", aggregate["by_state"]["pass_flight"]["recall"], f">= {PASS_INTERVAL_RECALL_MIN}", aggregate["by_state"]["pass_flight"]["recall"] >= PASS_INTERVAL_RECALL_MIN)
    add("aggregate_shot_interval_recall", aggregate["by_state"]["shot_flight"]["recall"], f">= {SHOT_INTERVAL_RECALL_MIN}", aggregate["by_state"]["shot_flight"]["recall"] >= SHOT_INTERVAL_RECALL_MIN)
    order_accuracy = aggregate["semantic_transition_order_accuracy"]
    add(
        "aggregate_semantic_transition_order_accuracy",
        order_accuracy,
        f">= {SEQUENCE_ORDER_ACCURACY_MIN}",
        order_accuracy is not None and order_accuracy >= SEQUENCE_ORDER_ACCURACY_MIN,
    )
    add("aggregate_core_unresolved_rate", aggregate["core_unresolved_rate"], f"<= {CORE_UNRESOLVED_MAX}", aggregate["core_unresolved_rate"] <= CORE_UNRESOLVED_MAX)
    add("pass_interval_as_shot", aggregate["critical_confusions"]["pass_interval_as_shot"], f"<= {PASS_AS_SHOT_MAX}", aggregate["critical_confusions"]["pass_interval_as_shot"] <= PASS_AS_SHOT_MAX)
    add("shot_interval_as_pass", aggregate["critical_confusions"]["shot_interval_as_pass"], f"<= {SHOT_AS_PASS_MAX}", aggregate["critical_confusions"]["shot_interval_as_pass"] <= SHOT_AS_PASS_MAX)

    for clip in clips:
        timeline = clip["timeline"]
        add(f"{clip['clip']}_prediction_coverage", clip["prediction_coverage"], f">= {COVERAGE_CLIP_MIN}", clip["prediction_coverage"] >= COVERAGE_CLIP_MIN)
        add(f"{clip['clip']}_core_interval_detection_recall", timeline["core_interval_detection_recall"], f">= {CLIP_CORE_INTERVAL_RECALL_MIN}", timeline["core_interval_detection_recall"] >= CLIP_CORE_INTERVAL_RECALL_MIN)
        add(f"{clip['clip']}_core_unresolved_rate", timeline["core_unresolved_rate"], f"<= {CLIP_CORE_UNRESOLVED_MAX}", timeline["core_unresolved_rate"] <= CLIP_CORE_UNRESOLVED_MAX)

    return {
        "status": "PASS" if checks and all(row["passed"] for row in checks) else "FAIL",
        "checks": checks,
        "required_checks": len(checks),
        "passed_checks": sum(1 for row in checks if row["passed"]),
        "failed_checks": sum(1 for row in checks if not row["passed"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Gate A P3-readiness semantics across the supplied V1-V5 development runs."
    )
    for clip in CLIPS:
        parser.add_argument(f"--{clip.lower()}-run", required=True)
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output", default="evaluation/gate_a_scoreboard.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_map = {clip: str(getattr(args, f"{clip.lower()}_run")) for clip in CLIPS}
    clips = [_clip_metrics(clip, run_map[clip], args.tolerance_frames) for clip in CLIPS]
    aggregate = _aggregate(clips)
    gate = _gate_checks(clips, aggregate)
    result = {
        "version": "courtflux_p2_gate_a_p3_readiness_v3",
        "contract": "interval/event semantics primary; sequence order conditioned on detected adjacent episodes; V1-V5 development analysis; V6-V7 final validation-only",
        "tolerance_frames": int(args.tolerance_frames),
        "runs": run_map,
        "clips": clips,
        "aggregate": aggregate,
        "gate": gate,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate,
                "aggregate": aggregate,
                "clips": [
                    {
                        "clip": row["clip"],
                        "run_id": row["run_id"],
                        "prediction_coverage": row["prediction_coverage"],
                        "core_interval_detection_recall": row["timeline"]["core_interval_detection_recall"],
                        "core_unresolved_rate": row["timeline"]["core_unresolved_rate"],
                        "semantic_transition_order_accuracy": row["timeline"]["semantic_transitions"]["order_accuracy"],
                        "detected_transition_coverage": row["timeline"]["semantic_transitions"]["detected_transition_coverage"],
                        "frame_accuracy_diagnostic": row["frame_diagnostics"]["frame_accuracy"],
                    }
                    for row in clips
                ],
                "output": str(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
