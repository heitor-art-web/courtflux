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


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        raw = row.get("raw_best_score")
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def analyze(run_id: str, ground_truth_path: Path) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    states = [row for row in load_jsonl(states_path) if isinstance(row, dict)]
    gt = load_ground_truth(ground_truth_path)
    truth = _truth_state_by_frame(gt)

    shot_rows = [
        row
        for row in states
        if truth.get(int(row.get("frame_id", -1))) == "shot_flight"
    ]
    missed = [
        row
        for row in shot_rows
        if str(row.get("live_control_state") or row.get("control_state") or "unknown")
        != "shot_flight"
    ]
    flight_inactive = [row for row in missed if not bool(row.get("flight_active", False))]

    owner_context: Counter[str] = Counter()
    candidate_context: Counter[str] = Counter()
    control_context: Counter[str] = Counter()
    lost_bucket: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in flight_inactive:
        engine_eligible = bool(row.get("engine_eligible", False))
        owner_present = row.get("current_track_id") is not None
        candidate_present = row.get("candidate_track_id") is not None
        raw_present = row.get("raw_best_track_id") is not None
        control = str(row.get("live_control_state") or row.get("control_state") or "unknown")

        if not engine_eligible:
            owner_context["engine_ineligible"] += 1
        elif owner_present:
            owner_context["engine_eligible_owner_present"] += 1
        else:
            owner_context["engine_eligible_owner_absent"] += 1

        if candidate_present:
            candidate_context["thresholded_candidate_present"] += 1
        elif raw_present:
            candidate_context["raw_best_only"] += 1
        else:
            candidate_context["no_candidate_evidence"] += 1
        control_context[control] += 1

        try:
            lost = int(row.get("lost_frames", 0) or 0)
        except (TypeError, ValueError):
            lost = 0
        if lost <= 0:
            lost_bucket["0"] += 1
        elif lost <= 2:
            lost_bucket["1_2"] += 1
        elif lost <= 6:
            lost_bucket["3_6"] += 1
        else:
            lost_bucket["7_plus"] += 1

        if len(examples) < 20 and engine_eligible:
            examples.append(
                {
                    "frame_id": int(row.get("frame_id", -1)),
                    "control_state": control,
                    "current_track_id": row.get("current_track_id"),
                    "lost_frames": row.get("lost_frames"),
                    "candidate_track_id": row.get("candidate_track_id"),
                    "candidate_frames": row.get("candidate_frames"),
                    "candidate_score": row.get("candidate_score"),
                    "raw_best_track_id": row.get("raw_best_track_id"),
                    "raw_best_score": row.get("raw_best_score"),
                    "shot_preflight_available": row.get("shot_preflight_available"),
                    "shot_preflight_above": row.get("shot_preflight_above"),
                    "shot_preflight_ball_to_rim_distance_norm": row.get(
                        "shot_preflight_ball_to_rim_distance_norm"
                    ),
                }
            )

    owner_absent_rows = [
        row
        for row in flight_inactive
        if bool(row.get("engine_eligible", False)) and row.get("current_track_id") is None
    ]
    owner_present_rows = [
        row
        for row in flight_inactive
        if bool(row.get("engine_eligible", False)) and row.get("current_track_id") is not None
    ]

    return {
        "version": "courtflux_p2_gate_a_flight_entry_diag_v1",
        "run_id": run_id,
        "ground_truth": str(ground_truth_path.relative_to(PROJECT_ROOT)),
        "shot_truth_frames": len(shot_rows),
        "missed_shot_truth_frames": len(missed),
        "flight_inactive_missed_frames": len(flight_inactive),
        "owner_context": dict(sorted(owner_context.items())),
        "candidate_context": dict(sorted(candidate_context.items())),
        "control_context": dict(sorted(control_context.items())),
        "lost_frames_bucket": dict(sorted(lost_bucket.items())),
        "owner_absent_raw_score": _score_summary(owner_absent_rows),
        "owner_present_raw_score": _score_summary(owner_present_rows),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose why GT shot_flight frames fail to enter possession flight."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

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
