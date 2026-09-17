from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import evaluate_payload, load_ground_truth, load_jsonl


def _truth_state_by_frame(gt: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in list(gt.get("control_intervals", []) or []):
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or "unknown")
        start = int(row.get("frame_start", row.get("frame", 0)))
        end = int(row.get("frame_end", start))
        if end < start:
            start, end = end, start
        for frame_id in range(start, end + 1):
            result[frame_id] = state
    return result


def _project_challenger_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in states:
        candidate = deepcopy(row)
        candidate["control_state"] = str(
            row.get("soft_acquisition_challenger_state")
            or row.get("live_control_state")
            or row.get("control_state")
            or "unknown"
        )
        projected.append(candidate)
    return projected


def _control_summary(payload: dict[str, Any]) -> dict[str, Any]:
    control = dict(payload.get("control_states", {}) or {})
    continuity = dict(control.get("continuity", {}) or {})
    return {
        "frame_accuracy": control.get("accuracy"),
        "prediction_coverage": control.get("prediction_coverage"),
        "dominant_state_interval_accuracy": continuity.get(
            "dominant_state_interval_accuracy"
        ),
        "mean_interval_matching_rate": continuity.get("mean_interval_matching_rate"),
        "confusion": control.get("confusion", {}),
    }


def _delta(challenger: Any, baseline: Any) -> float | None:
    try:
        return round(float(challenger) - float(baseline), 4)
    except (TypeError, ValueError):
        return None


def evaluate(run_id: str, ground_truth_path: Path, tolerance_frames: int) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    events_path = run_dir / "p2_events.jsonl"
    for path in (states_path, events_path, ground_truth_path):
        if not path.exists():
            raise FileNotFoundError(path)

    states = load_jsonl(states_path)
    events = load_jsonl(events_path)
    gt = load_ground_truth(ground_truth_path)

    if not any("soft_acquisition_challenger_state" in row for row in states):
        raise RuntimeError(
            "run does not contain soft_acquisition_challenger_state; rerun the canonical probe after the challenger commit"
        )

    baseline_eval = evaluate_payload(
        predicted_events=events,
        predicted_states=states,
        ground_truth=gt,
        tolerance_frames=tolerance_frames,
    )
    challenger_states = _project_challenger_states(states)
    challenger_eval = evaluate_payload(
        predicted_events=events,
        predicted_states=challenger_states,
        ground_truth=gt,
        tolerance_frames=tolerance_frames,
    )

    baseline = _control_summary(baseline_eval)
    challenger = _control_summary(challenger_eval)
    truth = _truth_state_by_frame(gt)
    triggered_truth: Counter[str] = Counter()
    triggered_frames: list[dict[str, Any]] = []
    for row in states:
        if not bool(row.get("soft_acquisition_triggered", False)):
            continue
        frame_id = int(row.get("frame_id", -1))
        truth_state = truth.get(frame_id, "unlabeled")
        triggered_truth[truth_state] += 1
        triggered_frames.append(
            {
                "frame_id": frame_id,
                "truth_state": truth_state,
                "track_id": row.get("soft_acquisition_track_id"),
                "streak_frames": row.get("soft_acquisition_streak_frames"),
                "mean_score": row.get("soft_acquisition_mean_score"),
                "peak_score": row.get("soft_acquisition_peak_score"),
            }
        )

    return {
        "version": "courtflux_p2_gate_a_soft_acquisition_eval_v1",
        "run_id": run_id,
        "ground_truth": str(ground_truth_path.relative_to(PROJECT_ROOT)),
        "policy": {
            "min_streak_frames": 4,
            "min_mean_score": 0.35,
            "scope": "live UNKNOWN + engine eligible + owner absent + thresholded candidate absent + flight inactive + shot inactive",
            "diagnostic_only": True,
        },
        "baseline": baseline,
        "challenger": challenger,
        "delta": {
            "frame_accuracy": _delta(
                challenger.get("frame_accuracy"), baseline.get("frame_accuracy")
            ),
            "dominant_state_interval_accuracy": _delta(
                challenger.get("dominant_state_interval_accuracy"),
                baseline.get("dominant_state_interval_accuracy"),
            ),
            "mean_interval_matching_rate": _delta(
                challenger.get("mean_interval_matching_rate"),
                baseline.get("mean_interval_matching_rate"),
            ),
        },
        "triggered_frames": len(triggered_frames),
        "trigger_truth": dict(sorted(triggered_truth.items())),
        "trigger_details": triggered_frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gate A live control state with the soft-acquisition challenger."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", default="evaluation/ground_truth/v1.json")
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    result = evaluate(str(args.run), gt_path, int(args.tolerance_frames))
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
