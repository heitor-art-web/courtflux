from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import evaluate_payload, load_ground_truth, load_jsonl

CORE_STATES = {"possessed", "pass_flight", "shot_flight"}


def _load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("version") or "") != "courtflux_p2_gate_a_shot_preflight_policy_v1":
        raise ValueError("unexpected shot-preflight policy version")
    if str(payload.get("family") or "") != "above_distance":
        raise ValueError("frozen policy family must be above_distance")
    if str(payload.get("scope") or "") != "unknown_only":
        raise ValueError("frozen policy scope must be unknown_only")
    if bool(payload.get("require_direct_observed", False)):
        raise ValueError("frozen policy must not require direct observation")
    return payload


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


def _live_state(row: dict[str, Any]) -> str:
    return str(row.get("live_control_state") or row.get("control_state") or "unknown")


def _triggers(row: dict[str, Any], *, max_distance_norm: float) -> bool:
    if _live_state(row) != "unknown":
        return False
    if not bool(row.get("engine_eligible", False)):
        return False
    if not bool(row.get("flight_active", False)):
        return False
    if bool(row.get("pass_flight_receiver_evidence", False)):
        return False
    if bool(row.get("shot_active", False)):
        return False
    if not bool(row.get("shot_preflight_available", False)):
        return False
    if not bool(row.get("shot_preflight_above", False)):
        return False
    raw_distance = row.get("shot_preflight_ball_to_rim_distance_norm")
    try:
        distance = float(raw_distance)
    except (TypeError, ValueError):
        return False
    return distance <= float(max_distance_norm)


def _project(states: list[dict[str, Any]], *, max_distance_norm: float) -> tuple[list[dict[str, Any]], list[int]]:
    projected: list[dict[str, Any]] = []
    triggered_frames: list[int] = []
    for row in states:
        clone = deepcopy(row)
        clone["control_state"] = _live_state(row)
        if _triggers(row, max_distance_norm=max_distance_norm):
            clone["control_state"] = "shot_flight"
            triggered_frames.append(int(row.get("frame_id", -1)))
        projected.append(clone)
    return projected, triggered_frames


def _control_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    control = dict(payload.get("control_states", {}) or {})
    continuity = dict(control.get("continuity", {}) or {})
    confusion = dict(control.get("confusion", {}) or {})

    shot = dict(confusion.get("shot_flight", {}) or {})
    passed = dict(confusion.get("pass_flight", {}) or {})
    possessed = dict(confusion.get("possessed", {}) or {})

    shot_total = sum(int(value) for value in shot.values())
    pass_total = sum(int(value) for value in passed.values())
    possessed_total = sum(int(value) for value in possessed.values())

    core_total = 0
    core_unknown = 0
    for truth_state in CORE_STATES:
        row = dict(confusion.get(truth_state, {}) or {})
        core_total += sum(int(value) for value in row.values())
        core_unknown += int(row.get("unknown", 0) or 0)

    return {
        "accuracy": control.get("accuracy"),
        "prediction_coverage": control.get("prediction_coverage"),
        "dominant_state_interval_accuracy": continuity.get("dominant_state_interval_accuracy"),
        "mean_interval_matching_rate": continuity.get("mean_interval_matching_rate"),
        "shot_recall": round(int(shot.get("shot_flight", 0) or 0) / max(shot_total, 1), 4),
        "pass_to_shot_rate": round(int(passed.get("shot_flight", 0) or 0) / max(pass_total, 1), 4),
        "possessed_to_shot_rate": round(int(possessed.get("shot_flight", 0) or 0) / max(possessed_total, 1), 4),
        "core_unknown_rate": round(core_unknown / max(core_total, 1), 4),
        "confusion": confusion,
    }


def _delta(challenger: Any, baseline: Any) -> float | None:
    try:
        return round(float(challenger) - float(baseline), 4)
    except (TypeError, ValueError):
        return None


def evaluate(
    run_id: str,
    ground_truth_path: Path,
    policy_path: Path,
    tolerance_frames: int,
) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    events_path = run_dir / "p2_events.jsonl"
    for path in (states_path, events_path, ground_truth_path, policy_path):
        if not path.exists():
            raise FileNotFoundError(path)

    states = [row for row in load_jsonl(states_path) if isinstance(row, dict)]
    if not any("shot_preflight_policy" in row for row in states):
        raise RuntimeError(
            "run has no shot-preflight diagnostics; rerun canonical probe after instrumentation commit"
        )
    events = load_jsonl(events_path)
    gt = load_ground_truth(ground_truth_path)
    policy = _load_policy(policy_path)
    max_distance = float(dict(policy.get("params", {}) or {}).get("max_distance_norm", 8.0))

    baseline_payload = evaluate_payload(
        predicted_events=events,
        predicted_states=states,
        ground_truth=gt,
        tolerance_frames=tolerance_frames,
    )
    projected, triggered_frames = _project(states, max_distance_norm=max_distance)
    challenger_payload = evaluate_payload(
        predicted_events=events,
        predicted_states=projected,
        ground_truth=gt,
        tolerance_frames=tolerance_frames,
    )

    baseline = _control_metrics(baseline_payload)
    challenger = _control_metrics(challenger_payload)
    truth = _truth_state_by_frame(gt)
    trigger_truth: dict[str, int] = {}
    for frame_id in triggered_frames:
        key = truth.get(frame_id, "unlabeled")
        trigger_truth[key] = trigger_truth.get(key, 0) + 1

    max_pass_to_shot = float(
        dict(policy.get("frozen_gate_guardrail", {}) or {}).get(
            "pass_flight_to_shot_flight_max", 0.20
        )
    )

    return {
        "version": "courtflux_p2_gate_a_shot_preflight_eval_v1",
        "run_id": run_id,
        "ground_truth": str(ground_truth_path.relative_to(PROJECT_ROOT)),
        "policy": policy,
        "baseline": baseline,
        "challenger": challenger,
        "delta": {
            key: _delta(challenger.get(key), baseline.get(key))
            for key in (
                "accuracy",
                "dominant_state_interval_accuracy",
                "mean_interval_matching_rate",
                "shot_recall",
                "pass_to_shot_rate",
                "possessed_to_shot_rate",
                "core_unknown_rate",
            )
        },
        "triggered_frames": len(triggered_frames),
        "trigger_truth": dict(sorted(trigger_truth.items())),
        "pass_to_shot_guardrail_max": max_pass_to_shot,
        "pass_to_shot_guardrail_ok": bool(
            float(challenger.get("pass_to_shot_rate") or 0.0) <= max_pass_to_shot
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Gate A causal shot-preflight challenger against one human-GT clip."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--policy",
        default="evaluation/ground_truth/gate_a_shot_preflight_policy_v1.json",
    )
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = PROJECT_ROOT / policy_path

    result = evaluate(
        str(args.run),
        gt_path,
        policy_path,
        int(args.tolerance_frames),
    )
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
