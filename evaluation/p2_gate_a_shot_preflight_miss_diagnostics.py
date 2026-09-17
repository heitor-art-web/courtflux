from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import load_ground_truth, load_jsonl

MAX_DISTANCE_NORM = 8.0


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


def _distance(row: dict[str, Any]) -> float | None:
    raw = row.get("shot_preflight_ball_to_rim_distance_norm")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _frozen_trigger(row: dict[str, Any]) -> bool:
    distance = _distance(row)
    return bool(
        bool(row.get("engine_eligible", False))
        and str(row.get("live_control_state") or row.get("control_state") or "unknown") == "unknown"
        and bool(row.get("flight_active", False))
        and not bool(row.get("pass_flight_receiver_evidence", False))
        and not bool(row.get("shot_active", False))
        and bool(row.get("shot_preflight_available", False))
        and bool(row.get("shot_preflight_above", False))
        and distance is not None
        and distance <= MAX_DISTANCE_NORM
    )


def _primary_blocker(row: dict[str, Any]) -> str:
    live = str(row.get("live_control_state") or row.get("control_state") or "unknown")
    distance = _distance(row)
    if not bool(row.get("engine_eligible", False)):
        return "engine_ineligible"
    if live == "shot_flight":
        return "already_live_shot"
    if live != "unknown":
        return f"scope_live_{live}"
    if not bool(row.get("flight_active", False)):
        return "flight_inactive"
    if bool(row.get("pass_flight_receiver_evidence", False)):
        return "receiver_evidence_present"
    if bool(row.get("shot_active", False)):
        return "shot_active_but_live_not_shot"
    if not bool(row.get("shot_preflight_available", False)):
        return "preflight_unavailable"
    if not bool(row.get("shot_preflight_above", False)):
        return "not_above_rim"
    if distance is None:
        return "distance_missing"
    if distance > MAX_DISTANCE_NORM:
        return "distance_gt_8"
    return "would_trigger"


def _all_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    live = str(row.get("live_control_state") or row.get("control_state") or "unknown")
    distance = _distance(row)
    if not bool(row.get("engine_eligible", False)):
        blockers.append("engine_ineligible")
    if live != "unknown":
        blockers.append(f"scope_live_{live}")
    if not bool(row.get("flight_active", False)):
        blockers.append("flight_inactive")
    if bool(row.get("pass_flight_receiver_evidence", False)):
        blockers.append("receiver_evidence_present")
    if bool(row.get("shot_active", False)):
        blockers.append("shot_active")
    if not bool(row.get("shot_preflight_available", False)):
        blockers.append("preflight_unavailable")
    if not bool(row.get("shot_preflight_above", False)):
        blockers.append("not_above_rim")
    if distance is None:
        blockers.append("distance_missing")
    elif distance > MAX_DISTANCE_NORM:
        blockers.append("distance_gt_8")
    return blockers


def analyze(run_id: str, ground_truth_path: Path) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not ground_truth_path.exists():
        raise FileNotFoundError(ground_truth_path)

    states = [row for row in load_jsonl(states_path) if isinstance(row, dict)]
    truth = _truth_state_by_frame(load_ground_truth(ground_truth_path))
    if not any("shot_preflight_policy" in row for row in states):
        raise RuntimeError("run does not contain shot_preflight diagnostics")

    primary: Counter[str] = Counter()
    all_counts: Counter[str] = Counter()
    live_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    shot_truth_frames = 0
    already_correct = 0
    frozen_triggers = 0

    for row in states:
        try:
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if truth.get(frame_id) != "shot_flight":
            continue
        shot_truth_frames += 1
        live = str(row.get("live_control_state") or row.get("control_state") or "unknown")
        live_counts[live] += 1
        if live == "shot_flight":
            already_correct += 1
            continue
        if _frozen_trigger(row):
            frozen_triggers += 1
        blocker = _primary_blocker(row)
        primary[blocker] += 1
        for item in _all_blockers(row):
            all_counts[item] += 1
        if len(rows) < 80:
            rows.append(
                {
                    "frame_id": frame_id,
                    "live_control_state": live,
                    "primary_blocker": blocker,
                    "engine_eligible": bool(row.get("engine_eligible", False)),
                    "flight_active": bool(row.get("flight_active", False)),
                    "flight_supported": bool(row.get("flight_supported", False)),
                    "receiver_evidence": bool(row.get("pass_flight_receiver_evidence", False)),
                    "shot_active": bool(row.get("shot_active", False)),
                    "preflight_available": bool(row.get("shot_preflight_available", False)),
                    "above": bool(row.get("shot_preflight_above", False)),
                    "near_x": bool(row.get("shot_preflight_near_x", False)),
                    "distance_norm": _distance(row),
                    "relative_y": row.get("shot_preflight_relative_y_rim_heights"),
                    "approach_streak": row.get("shot_preflight_approach_streak"),
                    "rising_streak": row.get("shot_preflight_rising_streak"),
                }
            )

    missed = max(0, shot_truth_frames - already_correct)
    return {
        "version": "courtflux_p2_gate_a_shot_preflight_miss_diag_v1",
        "run_id": run_id,
        "ground_truth": str(ground_truth_path.relative_to(PROJECT_ROOT)),
        "frozen_policy": {
            "family": "above_distance",
            "max_distance_norm": MAX_DISTANCE_NORM,
            "scope": "unknown_only",
            "require_direct_observed": False,
        },
        "shot_truth_frames": shot_truth_frames,
        "already_live_shot_frames": already_correct,
        "missed_shot_truth_frames": missed,
        "frozen_policy_triggerable_missed_frames": frozen_triggers,
        "live_state_on_shot_truth": dict(sorted(live_counts.items())),
        "primary_blocker_counts": dict(primary.most_common()),
        "all_blocker_counts": dict(all_counts.most_common()),
        "examples": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain why frozen Gate A shot-preflight policy misses GT shot_flight frames."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    result = analyze(str(args.run), gt_path)
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
