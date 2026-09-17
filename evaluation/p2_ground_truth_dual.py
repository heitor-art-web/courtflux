from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from evaluation.p2_ground_truth import (  # noqa: E402
    evaluate_control_states,
    evaluate_events,
    load_ground_truth,
    load_jsonl,
)


VERSION = "courtflux_p2_gt_eval_dual_v1"


def _states_for_field(
    states: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, dict):
            continue
        clone = dict(state)
        clone["control_state"] = str(
            state.get(field, state.get("control_state")) or "unknown"
        )
        rows.append(clone)
    return rows


def evaluate_payload_dual(
    *,
    predicted_events: list[dict[str, Any]],
    predicted_states: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    tolerance_frames: int = 8,
) -> dict[str, Any]:
    video = dict(ground_truth.get("video", {}) or {})
    fps_raw = video.get("fps")
    fps = float(fps_raw) if fps_raw not in {None, ""} else None
    truth_intervals = list(ground_truth.get("control_intervals", []) or [])

    live_rows = _states_for_field(predicted_states, "live_control_state")
    resolved_rows = _states_for_field(predicted_states, "resolved_control_state")
    resolved_field_rows = sum(
        1
        for row in predicted_states
        if isinstance(row, dict) and "resolved_control_state" in row
    )
    provisional_frames = sum(
        1
        for row in predicted_states
        if isinstance(row, dict) and bool(row.get("resolved_control_provisional", False))
    )
    backfill_activations = sum(
        1
        for row in predicted_states
        if isinstance(row, dict) and bool(row.get("resolved_shot_backfill_applied", False))
    )

    return {
        "version": VERSION,
        "events": evaluate_events(
            predicted_events,
            list(ground_truth.get("events", []) or []),
            tolerance_frames=tolerance_frames,
            fps=fps,
        ),
        "control_state_contract": {
            "live": "online control_state; never retroactively changed",
            "resolved": (
                "additive provisional evaluation channel; late shot activation may "
                "backfill preceding generic flight; receiver evidence abstains"
            ),
            "resolved_field_rows": resolved_field_rows,
            "resolved_field_coverage": round(
                resolved_field_rows / max(len(predicted_states), 1), 4
            ),
            "resolved_provisional_frames": provisional_frames,
            "resolved_shot_backfill_activations": backfill_activations,
        },
        "control_states": evaluate_control_states(live_rows, truth_intervals),
        "resolved_control_states": evaluate_control_states(
            resolved_rows,
            truth_intervals,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate P2 event windows plus live and provisional resolved control-state "
            "continuity against human GT."
        )
    )
    parser.add_argument("--events", required=True, help="p2_events.jsonl")
    parser.add_argument("--states", required=True, help="p2_states.jsonl")
    parser.add_argument("--ground-truth", required=True, help="courtflux_p2_gt_v1 JSON")
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_payload_dual(
        predicted_events=load_jsonl(args.events),
        predicted_states=load_jsonl(args.states),
        ground_truth=load_ground_truth(args.ground_truth),
        tolerance_frames=args.tolerance_frames,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"[P2_GT_DUAL_FILE] {output.resolve()}")
    print(rendered, end="")


if __name__ == "__main__":
    main()
