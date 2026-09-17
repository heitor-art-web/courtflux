from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.p2_ground_truth import evaluate_control_states, load_ground_truth, load_jsonl


VERSION = "courtflux_p2_resolved_pass_diagnose_v1"


def _same_reset(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        return int(a.get("reset_count", 0) or 0) == int(b.get("reset_count", 0) or 0)
    except (TypeError, ValueError):
        return False


def _flight_start(row: dict[str, Any]) -> int | None:
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


def resolve_passes(states: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for raw in states:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        live = str(row.get("control_state") or "unknown")
        row["diagnostic_live_control_state"] = live
        row["diagnostic_resolved_control_state"] = str(
            row.get("resolved_control_state") or live
        )
        rows.append(row)
    rows.sort(key=lambda row: int(row.get("frame_id", -1)))

    windows: list[dict[str, Any]] = []
    for index in range(1, len(rows)):
        receiver = rows[index]
        previous = rows[index - 1]
        if not _same_reset(previous, receiver):
            continue
        if not bool(previous.get("flight_active", False)):
            continue

        receiver_track = receiver.get("current_track_id")
        origin_track = previous.get("flight_origin_track_id")
        if receiver_track is None or origin_track is None:
            continue
        try:
            receiver_track_int = int(receiver_track)
            origin_track_int = int(origin_track)
        except (TypeError, ValueError):
            continue
        if receiver_track_int == origin_track_int:
            continue

        start = _flight_start(previous)
        if start is None:
            continue
        try:
            end = int(previous.get("frame_id", -1))
            receiver_frame = int(receiver.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if end < start or receiver_frame != end + 1:
            continue

        candidate_rows = [
            row
            for row in rows
            if start <= int(row.get("frame_id", -1)) <= end
            and _same_reset(row, receiver)
        ]
        shot_resolved_frames = sum(
            str(row.get("diagnostic_resolved_control_state") or "unknown") == "shot_flight"
            for row in candidate_rows
        )
        relabeled = 0
        for row in candidate_rows:
            # Shot resolution has precedence over pass resolution.
            if str(row.get("diagnostic_resolved_control_state") or "unknown") == "shot_flight":
                continue
            row["diagnostic_resolved_control_state"] = "pass_flight"
            relabeled += 1

        windows.append(
            {
                "flight_start_frame": start,
                "flight_end_frame": end,
                "receiver_frame": receiver_frame,
                "origin_track_id": origin_track_int,
                "receiver_track_id": receiver_track_int,
                "flight_supported_at_end": bool(previous.get("flight_supported", False)),
                "receiver_evidence_at_end": bool(
                    previous.get("pass_flight_receiver_evidence", False)
                ),
                "shot_resolved_frames_preserved": shot_resolved_frames,
                "relabeled_frames": relabeled,
            }
        )

    return rows, windows


def diagnose(states: list[dict[str, Any]], ground_truth: dict[str, Any]) -> dict[str, Any]:
    rows, windows = resolve_passes(states)
    truth = list(ground_truth.get("control_intervals", []) or [])
    live_rows = [
        {**row, "control_state": row["diagnostic_live_control_state"]}
        for row in rows
    ]
    starting_resolved_rows = [
        {
            **row,
            "control_state": str(
                row.get("resolved_control_state")
                or row["diagnostic_live_control_state"]
            ),
        }
        for row in rows
    ]
    pass_resolved_rows = [
        {**row, "control_state": row["diagnostic_resolved_control_state"]}
        for row in rows
    ]
    return {
        "version": VERSION,
        "contract": (
            "diagnostic_only; pass backfill requires an immediately following locked handler "
            "different from flight origin; existing resolved shot_flight has precedence"
        ),
        "resolution_windows": windows,
        "live": evaluate_control_states(live_rows, truth),
        "starting_resolved": evaluate_control_states(starting_resolved_rows, truth),
        "pass_resolved": evaluate_control_states(pass_resolved_rows, truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure retrospective pass-flight resolution from confirmed receiver locks."
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
        print(f"[P2_RESOLVED_PASS_DIAG_FILE] {output.resolve()}")
    print(rendered, end="")


if __name__ == "__main__":
    main()
