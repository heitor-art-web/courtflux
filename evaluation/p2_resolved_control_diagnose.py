from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.p2_ground_truth import evaluate_control_states, load_ground_truth, load_jsonl


VERSION = "courtflux_p2_resolved_control_diagnose_v1"


def _copy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clone = dict(row)
        clone["live_control_state"] = str(row.get("control_state") or "unknown")
        clone["resolved_control_state"] = clone["live_control_state"]
        copied.append(clone)
    copied.sort(key=lambda row: int(row.get("frame_id", -1)))
    return copied


def _same_reset(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a.get("reset_count", 0) or 0) == int(b.get("reset_count", 0) or 0)


def _flight_start_from_row(row: dict[str, Any]) -> int | None:
    if not bool(row.get("flight_active", False)):
        return None
    try:
        frame_id = int(row.get("frame_id", -1))
        flight_frames = max(1, int(row.get("flight_frames", 0) or 0))
    except (TypeError, ValueError):
        return None
    if frame_id < 0:
        return None
    return max(0, frame_id - flight_frames + 1)


def resolve_states(
    states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _copy_rows(states)
    windows: list[dict[str, Any]] = []
    previous_shot_active = False

    for index, row in enumerate(rows):
        shot_active = bool(row.get("shot_active", False))
        if shot_active and not previous_shot_active:
            activation_frame = int(row.get("frame_id", -1))
            last_flight_index: int | None = None

            for probe in range(index - 1, -1, -1):
                candidate = rows[probe]
                if not _same_reset(candidate, row):
                    break
                if bool(candidate.get("flight_active", False)):
                    last_flight_index = probe
                    break

            if last_flight_index is not None:
                last_flight = rows[last_flight_index]
                flight_start = _flight_start_from_row(last_flight)
                last_flight_frame = int(last_flight.get("frame_id", -1))

                # A receiver observation means the generic airborne segment already has
                # pass-oriented evidence. In that case retrospective shot relabeling must
                # abstain instead of overriding it.
                receiver_evidence = False
                if flight_start is not None:
                    for candidate in rows:
                        frame_id = int(candidate.get("frame_id", -1))
                        if frame_id < flight_start or frame_id > last_flight_frame:
                            continue
                        if not _same_reset(candidate, row):
                            continue
                        if bool(candidate.get("pass_flight_receiver_evidence", False)):
                            receiver_evidence = True
                            break

                if flight_start is not None and not receiver_evidence:
                    relabeled = 0
                    for candidate in rows:
                        frame_id = int(candidate.get("frame_id", -1))
                        if frame_id < flight_start or frame_id > activation_frame:
                            continue
                        if not _same_reset(candidate, row):
                            continue
                        candidate["resolved_control_state"] = "shot_flight"
                        relabeled += 1
                    windows.append(
                        {
                            "shot_activation_frame": activation_frame,
                            "flight_start_frame": flight_start,
                            "last_flight_frame": last_flight_frame,
                            "gap_to_shot_activation_frames": max(
                                0, activation_frame - last_flight_frame - 1
                            ),
                            "relabeled_frames": relabeled,
                            "receiver_evidence": False,
                        }
                    )
                else:
                    windows.append(
                        {
                            "shot_activation_frame": activation_frame,
                            "flight_start_frame": flight_start,
                            "last_flight_frame": last_flight_frame,
                            "gap_to_shot_activation_frames": max(
                                0, activation_frame - last_flight_frame - 1
                            ),
                            "relabeled_frames": 0,
                            "receiver_evidence": receiver_evidence,
                        }
                    )

        previous_shot_active = shot_active

    return rows, windows


def diagnose(
    states: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    resolved_rows, windows = resolve_states(states)
    truth_intervals = list(ground_truth.get("control_intervals", []) or [])

    live_rows = [
        {**row, "control_state": row["live_control_state"]}
        for row in resolved_rows
    ]
    resolved_eval_rows = [
        {**row, "control_state": row["resolved_control_state"]}
        for row in resolved_rows
    ]

    live = evaluate_control_states(live_rows, truth_intervals)
    resolved = evaluate_control_states(resolved_eval_rows, truth_intervals)

    return {
        "version": VERSION,
        "contract": (
            "diagnostic_only; live state unchanged; generic flight may be relabeled "
            "as shot_flight only after later shot activation and only without receiver evidence"
        ),
        "state_rows": len(resolved_rows),
        "resolution_windows": windows,
        "live": {
            "accuracy": live.get("accuracy"),
            "continuity": live.get("continuity"),
            "confusion": live.get("confusion"),
        },
        "resolved": {
            "accuracy": resolved.get("accuracy"),
            "continuity": resolved.get("continuity"),
            "confusion": resolved.get("confusion"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure retrospective shot-flight resolution without changing runtime state."
    )
    parser.add_argument("--states", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = diagnose(load_jsonl(args.states), load_ground_truth(args.ground_truth))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"[P2_RESOLVED_CONTROL_DIAG_FILE] {output.resolve()}")
    print(rendered, end="")


if __name__ == "__main__":
    main()
