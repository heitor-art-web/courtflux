from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.p2_ground_truth import _event_window, _interval_distance, load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_TYPE_BY_STATE = {
    "pass_flight": "pass_candidate",
    "shot_flight": "shot_attempt",
}


def _run_path(run_id: str, filename: str) -> Path:
    return PROJECT_ROOT / "evaluation" / "runs" / run_id / filename


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = Counter(
        str(row.get(key))
        for row in rows
        if row.get(key) not in {None, "", "unknown", "UNKNOWN"}
    )
    return dict(values.most_common())


def _true_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row.get(key, False)) for row in rows)


def _max_int(rows: list[dict[str, Any]], key: str) -> int:
    values: list[int] = []
    for row in rows:
        try:
            values.append(int(row.get(key, 0) or 0))
        except (TypeError, ValueError):
            pass
    return max(values, default=0)


def _nearby_events(
    events: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    radius: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        pred_start, pred_end = _event_window(event)
        distance = _interval_distance(start, end, pred_start, pred_end)
        if distance > radius:
            continue
        attrs = dict(event.get("attributes", {}) or {})
        result.append(
            {
                "index": index,
                "type": event.get("event_type"),
                "start": pred_start,
                "end": pred_end,
                "distance": distance,
                "team": event.get("team_id"),
                "actor": event.get("actor_track_id"),
                "secondary": event.get("secondary_track_id"),
                "publication_reason": attrs.get("team_attribution_publication_reason"),
                "publication_source": attrs.get("team_attribution_publication_source"),
                "internal_team": attrs.get("internal_team_id_before_publication"),
            }
        )
    return sorted(result, key=lambda row: (int(row["distance"]), int(row["start"]), int(row["index"])))


def _state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames": len(rows),
        "control": _counter(rows, "control_state"),
        "resolved_control": _counter(rows, "resolved_control_state"),
        "live_source": _counter(rows, "live_control_source"),
        "current_team": _counter(rows, "current_team_id"),
        "internal_team": _counter(rows, "internal_current_team_id"),
        "current_track": _counter(rows, "current_track_id"),
        "candidate_track": _counter(rows, "candidate_track_id"),
        "raw_best_track": _counter(rows, "raw_best_track_id"),
        "candidate_frames_max": _max_int(rows, "candidate_frames"),
        "lost_frames_max": _max_int(rows, "lost_frames"),
        "flight_active_frames": _true_count(rows, "flight_active"),
        "flight_supported_frames": _true_count(rows, "flight_supported"),
        "receiver_evidence_frames": _true_count(rows, "pass_flight_receiver_evidence"),
        "shot_active_frames": _true_count(rows, "shot_active"),
        "shot_preflight_signal_frames": _true_count(rows, "shot_preflight_signal"),
        "shot_preflight_trajectory_frames": _true_count(rows, "shot_preflight_trajectory_start"),
    }


def audit(scoreboard_path: Path, *, radius: int) -> dict[str, Any]:
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    artifacts: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    failures: list[dict[str, Any]] = []

    for clip in list(scoreboard.get("clips", []) or []):
        clip_name = str(clip.get("clip") or "")
        for sample in list(clip.get("sample_results", []) or []):
            if bool(sample.get("attributed", False)):
                continue
            run_key = str(sample.get("run_key") or "")
            run_id = str(sample.get("run_id") or "")
            cache_key = f"{run_key}:{run_id}"
            if cache_key not in artifacts:
                artifacts[cache_key] = (
                    load_jsonl(_run_path(run_id, "p2_states.jsonl")),
                    load_jsonl(_run_path(run_id, "p2_events.jsonl")),
                )
            states, events = artifacts[cache_key]
            start = int(sample.get("frame_start", 0))
            end = int(sample.get("frame_end", start))
            window_rows = [
                row
                for row in states
                if start - radius <= int(row.get("frame_id", -1)) <= end + radius
            ]
            nearby = _nearby_events(events, start=start, end=end, radius=radius)
            expected_event = EVENT_TYPE_BY_STATE.get(str(sample.get("state") or ""))
            expected_matches = [row for row in nearby if row.get("type") == expected_event]
            other_types = [row for row in nearby if row.get("type") != expected_event]
            failures.append(
                {
                    "clip": clip_name,
                    "sample_index": int(sample.get("sample_index", -1)),
                    "run_key": run_key,
                    "run_id": run_id,
                    "state": sample.get("state"),
                    "truth_team": sample.get("truth_team_id"),
                    "window": [start, end],
                    "expected_event_type": expected_event,
                    "expected_event_candidates": expected_matches,
                    "other_nearby_events": other_types,
                    "states": _state_summary(window_rows),
                }
            )

    return {
        "version": "courtflux_p2_gate_c_coverage_recovery_audit_v1",
        "scoreboard": str(scoreboard_path),
        "radius_frames": int(radius),
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit all current Gate C UNKNOWN failures in one pass before coverage recovery."
    )
    parser.add_argument(
        "--scoreboard",
        default="evaluation/gate_c_scoreboard_latest.json",
    )
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument(
        "--output",
        default="evaluation/gate_c_coverage_recovery_audit.json",
    )
    args = parser.parse_args()

    report = audit(Path(args.scoreboard), radius=max(0, int(args.radius)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("===== GATE C COVERAGE RECOVERY AUDIT =====")
    for row in report["failures"]:
        states = row["states"]
        nearby = row["expected_event_candidates"] + row["other_nearby_events"]
        print(
            f"{row['clip']}#{row['sample_index']} {row['state']} truth={row['truth_team']} "
            f"window={row['window']} events={nearby} "
            f"control={states['control']} teams={states['current_team']} "
            f"tracks={states['current_track']} candidate={states['candidate_track']} "
            f"candidate_max={states['candidate_frames_max']} "
            f"flight={states['flight_active_frames']}/{states['flight_supported_frames']} "
            f"receiver={states['receiver_evidence_frames']} shot={states['shot_active_frames']}"
        )
    print(f"[COVERAGE_RECOVERY_AUDIT] {output.resolve()}")


if __name__ == "__main__":
    main()
