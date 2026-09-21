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
                "from_team": attrs.get("from_team_id"),
                "to_team": attrs.get("to_team_id"),
                "transition_kind": attrs.get("transition_kind"),
                "flight_supported": attrs.get("flight_supported"),
                "flight_frames": attrs.get("flight_frames"),
                "flight_direct": attrs.get("flight_direct_observations"),
                "flight_predicted": attrs.get("flight_predicted_observations"),
                "flight_displacement_heights": attrs.get("flight_displacement_player_heights"),
                "fresh_hint_tracks": attrs.get("team_attribution_fresh_hint_tracks"),
                "fresh_hint_ages": attrs.get("team_attribution_fresh_hint_ages"),
            }
        )
    return sorted(
        result,
        key=lambda row: (int(row["distance"]), int(row["start"]), int(row["index"])),
    )


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


def _track_ids(window_rows: list[dict[str, Any]], nearby: list[dict[str, Any]]) -> list[int]:
    values: set[int] = set()
    for event in nearby:
        for key in ("actor", "secondary"):
            try:
                raw = event.get(key)
                if raw is not None:
                    values.add(int(raw))
            except (TypeError, ValueError):
                pass
    for row in window_rows:
        for key in ("current_track_id", "candidate_track_id", "raw_best_track_id"):
            try:
                raw = row.get(key)
                if raw is not None:
                    values.add(int(raw))
            except (TypeError, ValueError):
                pass
    return sorted(values)


def _track_evidence(
    rows: list[dict[str, Any]],
    track_ids: list[int],
) -> dict[str, Any]:
    wanted = set(track_ids)
    evidence: dict[int, dict[str, Any]] = {}
    for frame in rows:
        try:
            frame_id = int(frame.get("frame_id", -1))
        except (TypeError, ValueError):
            frame_id = -1
        for prediction in list(frame.get("team_predictions", []) or []):
            if not isinstance(prediction, dict):
                continue
            try:
                track_id = int(prediction.get("track_id"))
            except (TypeError, ValueError):
                continue
            if track_id not in wanted:
                continue
            bucket = evidence.setdefault(
                track_id,
                {
                    "frames": 0,
                    "player_team": Counter(),
                    "fused_team": Counter(),
                    "rescored_frames": 0,
                    "active_on_court": Counter(),
                    "max_team_confidence": 0.0,
                    "max_fused_confidence": 0.0,
                    "max_fused_margin": 0.0,
                    "recent": [],
                },
            )
            bucket["frames"] += 1
            team = prediction.get("team")
            if team not in {None, "", "unknown", "UNKNOWN"}:
                bucket["player_team"][str(team)] += 1
            fused = prediction.get("fused_best_team")
            if fused not in {None, "", "unknown", "UNKNOWN"}:
                bucket["fused_team"][str(fused)] += 1
            if bool(prediction.get("rescored", False)):
                bucket["rescored_frames"] += 1
            active = prediction.get("active_on_court")
            active_key = "true" if active is True else "false" if active is False else "none"
            bucket["active_on_court"][active_key] += 1
            for key, target in (
                ("team_confidence", "max_team_confidence"),
                ("fused_confidence", "max_fused_confidence"),
                ("fused_margin", "max_fused_margin"),
            ):
                try:
                    bucket[target] = max(float(bucket[target]), float(prediction.get(key, 0.0) or 0.0))
                except (TypeError, ValueError):
                    pass
            if bool(prediction.get("rescored", False)) or fused is not None:
                bucket["recent"].append(
                    {
                        "frame": frame_id,
                        "team": team,
                        "team_conf": prediction.get("team_confidence"),
                        "fused": fused,
                        "fused_conf": prediction.get("fused_confidence"),
                        "margin": prediction.get("fused_margin"),
                        "rescored": bool(prediction.get("rescored", False)),
                        "active": active,
                    }
                )

    result: dict[str, Any] = {}
    for track_id in track_ids:
        bucket = evidence.get(track_id)
        if bucket is None:
            continue
        result[str(track_id)] = {
            "frames": int(bucket["frames"]),
            "player_team": dict(bucket["player_team"].most_common()),
            "fused_team": dict(bucket["fused_team"].most_common()),
            "rescored_frames": int(bucket["rescored_frames"]),
            "active_on_court": dict(bucket["active_on_court"]),
            "max_team_confidence": round(float(bucket["max_team_confidence"]), 6),
            "max_fused_confidence": round(float(bucket["max_fused_confidence"]), 6),
            "max_fused_margin": round(float(bucket["max_fused_margin"]), 6),
            "recent": list(bucket["recent"])[-8:],
        }
    return result


def _control_changes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous: tuple[Any, ...] | None = None
    for row in rows:
        snapshot = (
            row.get("control_state"),
            row.get("current_track_id"),
            row.get("current_team_id"),
            row.get("candidate_track_id"),
            row.get("candidate_frames"),
            row.get("raw_best_track_id"),
            bool(row.get("flight_supported", False)),
            bool(row.get("pass_flight_receiver_evidence", False)),
            bool(row.get("shot_active", False)),
        )
        if snapshot == previous:
            continue
        previous = snapshot
        result.append(
            {
                "frame": row.get("frame_id"),
                "control": snapshot[0],
                "owner": snapshot[1],
                "team": snapshot[2],
                "candidate": snapshot[3],
                "candidate_frames": snapshot[4],
                "raw_best": snapshot[5],
                "flight_supported": snapshot[6],
                "receiver": snapshot[7],
                "shot": snapshot[8],
            }
        )
    if len(result) <= 24:
        return result
    return result[:12] + [{"omitted_changes": len(result) - 24}] + result[-12:]


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
            tracks = _track_ids(window_rows, nearby)
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
                    "track_evidence": _track_evidence(window_rows, tracks),
                    "control_changes": _control_changes(window_rows),
                }
            )

    return {
        "version": "courtflux_p2_gate_c_coverage_recovery_audit_v2",
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

    print("===== GATE C COVERAGE RECOVERY AUDIT V2 =====")
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
            f"receiver={states['receiver_evidence_frames']} shot={states['shot_active_frames']} "
            f"track_evidence={row['track_evidence']}"
        )
    print(f"[COVERAGE_RECOVERY_AUDIT] {output.resolve()}")


if __name__ == "__main__":
    main()
