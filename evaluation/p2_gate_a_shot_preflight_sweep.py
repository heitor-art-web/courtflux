from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import evaluate_control_states, load_ground_truth, load_jsonl

PolicyFn = Callable[[dict[str, Any]], bool]


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


def _as_float(row: dict[str, Any], field: str) -> float | None:
    raw = row.get(field)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_int(row: dict[str, Any], field: str) -> int:
    try:
        return int(row.get(field, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _eligible(row: dict[str, Any], *, scope: str, require_direct: bool) -> bool:
    live = str(row.get("live_control_state") or row.get("control_state") or "unknown")
    if scope == "unknown_only" and live != "unknown":
        return False
    if scope == "non_shot" and live == "shot_flight":
        return False
    return bool(
        row.get("shot_preflight_available", False)
        and row.get("flight_active", False)
        and not row.get("pass_flight_receiver_evidence", False)
        and not row.get("shot_active", False)
        and row.get("engine_eligible", False)
        and (not require_direct or row.get("shot_preflight_direct_observed", False))
    )


def _distance(row: dict[str, Any]) -> float | None:
    return _as_float(row, "shot_preflight_ball_to_rim_distance_norm")


def _policy_specs() -> list[tuple[str, dict[str, Any], PolicyFn]]:
    policies: list[tuple[str, dict[str, Any], PolicyFn]] = []

    policies.append(
        (
            "rim_band",
            {},
            lambda row: bool(row.get("shot_preflight_rim_band", False)),
        )
    )

    for max_distance in (4.0, 6.0, 8.0):
        policies.append(
            (
                "above_distance",
                {"max_distance_norm": max_distance},
                lambda row, max_distance=max_distance: bool(
                    row.get("shot_preflight_above", False)
                    and _distance(row) is not None
                    and float(_distance(row)) <= max_distance
                ),
            )
        )

    for max_distance in (1.0, 2.0, 3.0, 4.0):
        policies.append(
            (
                "near_distance",
                {"max_distance_norm": max_distance},
                lambda row, max_distance=max_distance: bool(
                    row.get("shot_preflight_near_x", False)
                    and _distance(row) is not None
                    and float(_distance(row)) <= max_distance
                ),
            )
        )

    for min_streak in (2, 3, 4):
        for max_distance in (4.0, 6.0, 8.0):
            policies.append(
                (
                    "approach_distance",
                    {
                        "min_approach_streak": min_streak,
                        "max_distance_norm": max_distance,
                    },
                    lambda row, min_streak=min_streak, max_distance=max_distance: bool(
                        _as_int(row, "shot_preflight_approach_streak") >= min_streak
                        and _distance(row) is not None
                        and float(_distance(row)) <= max_distance
                    ),
                )
            )

    for above_distance in (4.0, 6.0, 8.0):
        for near_distance in (1.0, 2.0, 3.0, 4.0):
            policies.append(
                (
                    "above_or_near_distance",
                    {
                        "above_max_distance_norm": above_distance,
                        "near_max_distance_norm": near_distance,
                    },
                    lambda row, above_distance=above_distance, near_distance=near_distance: bool(
                        _distance(row) is not None
                        and (
                            (
                                bool(row.get("shot_preflight_above", False))
                                and float(_distance(row)) <= above_distance
                            )
                            or (
                                bool(row.get("shot_preflight_near_x", False))
                                and float(_distance(row)) <= near_distance
                            )
                        )
                    ),
                )
            )

    return policies


def _project(
    states: list[dict[str, Any]],
    *,
    scope: str,
    require_direct: bool,
    policy: PolicyFn,
) -> tuple[list[dict[str, Any]], list[int]]:
    projected: list[dict[str, Any]] = []
    triggered: list[int] = []
    for row in states:
        clone = deepcopy(row)
        live = str(row.get("live_control_state") or row.get("control_state") or "unknown")
        clone["control_state"] = live
        if _eligible(row, scope=scope, require_direct=require_direct) and policy(row):
            clone["control_state"] = "shot_flight"
            try:
                triggered.append(int(row.get("frame_id", -1)))
            except (TypeError, ValueError):
                pass
        projected.append(clone)
    return projected, triggered


def _confusion_count(control: dict[str, Any], truth_state: str, predicted_state: str) -> int:
    confusion = dict(control.get("confusion", {}) or {})
    row = dict(confusion.get(truth_state, {}) or {})
    try:
        return int(row.get(predicted_state, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _truth_total(control: dict[str, Any], truth_state: str) -> int:
    confusion = dict(control.get("confusion", {}) or {})
    row = dict(confusion.get(truth_state, {}) or {})
    total = 0
    for value in row.values():
        try:
            total += int(value or 0)
        except (TypeError, ValueError):
            continue
    return total


def _metric_summary(control: dict[str, Any]) -> dict[str, Any]:
    continuity = dict(control.get("continuity", {}) or {})
    pass_total = _truth_total(control, "pass_flight")
    possessed_total = _truth_total(control, "possessed")
    shot_total = _truth_total(control, "shot_flight")
    core_total = pass_total + possessed_total + shot_total
    core_unknown = sum(
        _confusion_count(control, truth_state, "unknown")
        for truth_state in ("pass_flight", "possessed", "shot_flight")
    )
    return {
        "accuracy": control.get("accuracy"),
        "dominant_state_interval_accuracy": continuity.get(
            "dominant_state_interval_accuracy"
        ),
        "mean_interval_matching_rate": continuity.get("mean_interval_matching_rate"),
        "shot_recall": round(
            _confusion_count(control, "shot_flight", "shot_flight") / max(shot_total, 1), 4
        ),
        "pass_to_shot_rate": round(
            _confusion_count(control, "pass_flight", "shot_flight") / max(pass_total, 1), 4
        ),
        "possessed_to_shot_rate": round(
            _confusion_count(control, "possessed", "shot_flight")
            / max(possessed_total, 1),
            4,
        ),
        "core_unknown_rate": round(core_unknown / max(core_total, 1), 4),
    }


def _delta(challenger: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "accuracy",
        "dominant_state_interval_accuracy",
        "mean_interval_matching_rate",
        "shot_recall",
        "pass_to_shot_rate",
        "possessed_to_shot_rate",
        "core_unknown_rate",
    ):
        try:
            result[key] = round(float(challenger[key]) - float(baseline[key]), 4)
        except (TypeError, ValueError, KeyError):
            result[key] = None
    return result


def analyze(run_id: str, ground_truth_path: Path, top_n: int) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not ground_truth_path.exists():
        raise FileNotFoundError(ground_truth_path)

    states = [row for row in load_jsonl(states_path) if isinstance(row, dict)]
    gt = load_ground_truth(ground_truth_path)
    truth = _truth_state_by_frame(gt)
    intervals = list(gt.get("control_intervals", []) or [])

    if not any("shot_preflight_policy" in row for row in states):
        raise RuntimeError("run does not contain shot_preflight diagnostics")

    baseline_rows: list[dict[str, Any]] = []
    for row in states:
        clone = deepcopy(row)
        clone["control_state"] = str(
            row.get("live_control_state") or row.get("control_state") or "unknown"
        )
        baseline_rows.append(clone)
    baseline_control = evaluate_control_states(baseline_rows, intervals)
    baseline = _metric_summary(baseline_control)

    results: list[dict[str, Any]] = []
    policy_index = 0
    for scope in ("unknown_only", "non_shot"):
        for require_direct in (False, True):
            for family, params, policy in _policy_specs():
                policy_index += 1
                projected, triggered = _project(
                    states,
                    scope=scope,
                    require_direct=require_direct,
                    policy=policy,
                )
                control = evaluate_control_states(projected, intervals)
                metrics = _metric_summary(control)

                labeled_trigger_truth: Counter[str] = Counter()
                unlabeled_triggers = 0
                for frame_id in triggered:
                    truth_state = truth.get(frame_id)
                    if truth_state is None:
                        unlabeled_triggers += 1
                    else:
                        labeled_trigger_truth[truth_state] += 1

                tp = int(labeled_trigger_truth.get("shot_flight", 0))
                fp = sum(
                    count
                    for state, count in labeled_trigger_truth.items()
                    if state != "shot_flight"
                )
                labeled_positive_opportunities = sum(
                    1
                    for row in states
                    if _eligible(row, scope=scope, require_direct=require_direct)
                    and truth.get(int(row.get("frame_id", -1))) == "shot_flight"
                )
                precision = tp / max(tp + fp, 1)
                recall = tp / max(labeled_positive_opportunities, 1)
                f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

                results.append(
                    {
                        "policy_index": policy_index,
                        "family": family,
                        "params": params,
                        "scope": scope,
                        "require_direct_observed": require_direct,
                        "triggered_frames": len(triggered),
                        "unlabeled_triggers": unlabeled_triggers,
                        "trigger_truth": dict(sorted(labeled_trigger_truth.items())),
                        "labeled_classifier": {
                            "positive_opportunities": labeled_positive_opportunities,
                            "tp": tp,
                            "fp": fp,
                            "precision": round(precision, 4),
                            "recall": round(recall, 4),
                            "f1": round(f1, 4),
                        },
                        "metrics": metrics,
                        "delta": _delta(metrics, baseline),
                        "pass_to_shot_guardrail_ok": metrics["pass_to_shot_rate"] <= 0.20,
                    }
                )

    ranked = sorted(
        results,
        key=lambda row: (
            bool(row["pass_to_shot_guardrail_ok"]),
            float(row["delta"].get("mean_interval_matching_rate") or 0.0),
            float(row["delta"].get("shot_recall") or 0.0),
            float(row["labeled_classifier"].get("precision") or 0.0),
            -int(row.get("unlabeled_triggers", 0)),
        ),
        reverse=True,
    )

    return {
        "version": "courtflux_p2_gate_a_shot_preflight_sweep_v1",
        "run_id": run_id,
        "ground_truth": str(ground_truth_path.relative_to(PROJECT_ROOT)),
        "contract": {
            "development_clip": "V1 only",
            "labeled_precision": "unlabeled frames are excluded from TP/FP classification",
            "projection": "diagnostic only; selected policy changes projected control_state to shot_flight but never mutates runtime",
            "frozen_gate_guardrail": "pass_flight->shot_flight <= 0.20",
        },
        "baseline": baseline,
        "policies_evaluated": len(results),
        "top_policies": ranked[: max(1, int(top_n))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep causal Gate A shot-preflight challenger policies on V1."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", default="evaluation/ground_truth/v1.json")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    result = analyze(str(args.run), gt_path, int(args.top_n))
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
