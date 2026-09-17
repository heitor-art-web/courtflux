from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "courtflux_p2_control_diagnose_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _state_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: int(row.get("frame_id", 0)))
    runs: list[dict[str, Any]] = []
    run_state = str(ordered[0].get("control_state") or "unknown")
    run_start = int(ordered[0].get("frame_id", 0))
    run_end = run_start

    for row in ordered[1:]:
        frame_id = int(row.get("frame_id", 0))
        state = str(row.get("control_state") or "unknown")
        if state == run_state and frame_id == run_end + 1:
            run_end = frame_id
            continue
        runs.append(
            {
                "state": run_state,
                "frame_start": run_start,
                "frame_end": run_end,
                "frames": run_end - run_start + 1,
            }
        )
        run_state = state
        run_start = frame_id
        run_end = frame_id

    runs.append(
        {
            "state": run_state,
            "frame_start": run_start,
            "frame_end": run_end,
            "frames": run_end - run_start + 1,
        }
    )
    return runs


def _track_switches(rows: list[dict[str, Any]], key: str) -> int:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(1 for previous, current in zip(values, values[1:]) if previous != current)


def _reset_increments(rows: list[dict[str, Any]]) -> int:
    values: list[int] = []
    for row in rows:
        try:
            values.append(max(0, int(row.get("reset_count", 0) or 0)))
        except (TypeError, ValueError):
            values.append(0)
    if not values:
        return 0
    return max(values) - min(values)


def _optional_candidate_signals(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not any("candidate_track_id" in row for row in rows):
        return None

    present = [row for row in rows if row.get("candidate_track_id") is not None]
    below_threshold = 0
    above_threshold = 0
    lock_pending = 0
    lock_ready = 0
    receiver_evidence = 0
    distinct_from_origin = 0
    same_as_origin = 0
    supported_distinct_receiver = 0
    supported_without_receiver = 0
    candidate_scores: list[float] = []

    for row in present:
        score = row.get("candidate_score")
        threshold = row.get("candidate_threshold")
        frames = row.get("candidate_frames")
        required = row.get("candidate_required_frames")
        try:
            if score is not None:
                candidate_scores.append(float(score))
            if score is not None and threshold is not None:
                if float(score) < float(threshold):
                    below_threshold += 1
                else:
                    above_threshold += 1
        except (TypeError, ValueError):
            pass
        try:
            if (
                score is not None
                and threshold is not None
                and float(score) >= float(threshold)
                and frames is not None
                and required is not None
            ):
                if int(frames) < int(required):
                    lock_pending += 1
                else:
                    lock_ready += 1
        except (TypeError, ValueError):
            pass

        if bool(row.get("pass_flight_receiver_evidence", False)):
            receiver_evidence += 1

        candidate_track = row.get("candidate_track_id")
        origin_track = row.get("flight_origin_track_id")
        if candidate_track is not None and origin_track is not None:
            try:
                if int(candidate_track) != int(origin_track):
                    distinct_from_origin += 1
                    if bool(row.get("flight_supported", False)):
                        supported_distinct_receiver += 1
                else:
                    same_as_origin += 1
            except (TypeError, ValueError):
                pass
        if bool(row.get("flight_supported", False)) and not bool(
            row.get("pass_flight_receiver_evidence", False)
        ):
            supported_without_receiver += 1

    return {
        "candidate_present_frames": len(present),
        "candidate_track_switches": _track_switches(rows, "candidate_track_id"),
        "candidate_below_threshold_frames": below_threshold,
        "candidate_above_threshold_frames": above_threshold,
        "candidate_lock_pending_frames": lock_pending,
        "candidate_lock_ready_frames": lock_ready,
        "candidate_distinct_from_flight_origin_frames": distinct_from_origin,
        "candidate_same_as_flight_origin_frames": same_as_origin,
        "receiver_evidence_frames": receiver_evidence,
        "flight_supported_with_distinct_receiver_frames": supported_distinct_receiver,
        "flight_supported_without_receiver_frames": supported_without_receiver,
        "max_candidate_score": (
            None if not candidate_scores else round(max(candidate_scores), 6)
        ),
        "mean_candidate_score": (
            None
            if not candidate_scores
            else round(sum(candidate_scores) / len(candidate_scores), 6)
        ),
    }


def _frame_input_signals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ball_sources = Counter(str(row.get("ball_source") or "unknown") for row in rows)
    gameplay_statuses: Counter[str] = Counter()
    selected_confidences: list[float] = []
    selected_ball_frames = 0
    player_zero_frames = 0

    for row in rows:
        if bool(row.get("selected_ball", False)):
            selected_ball_frames += 1
        raw_confidence = row.get("selected_ball_conf")
        try:
            if raw_confidence is not None:
                selected_confidences.append(float(raw_confidence))
        except (TypeError, ValueError):
            pass
        try:
            if int(row.get("players_final", 0) or 0) <= 0:
                player_zero_frames += 1
        except (TypeError, ValueError):
            player_zero_frames += 1
        court_state = row.get("court_state")
        if isinstance(court_state, dict):
            gameplay_statuses[str(court_state.get("gameplay_status") or "unknown")] += 1
        else:
            gameplay_statuses["unknown"] += 1

    return {
        "artifact_frames": len(rows),
        "selected_ball_frames": selected_ball_frames,
        "selected_ball_rate": _rate(selected_ball_frames, len(rows)),
        "ball_source_counts": dict(sorted(ball_sources.items())),
        "predicted_hold_frames": int(ball_sources.get("predicted_hold", 0)),
        "missing_ball_frames": int(ball_sources.get("missing", 0)),
        "player_zero_frames": player_zero_frames,
        "gameplay_status_counts": dict(sorted(gameplay_statuses.items())),
        "mean_selected_ball_confidence": (
            None
            if not selected_confidences
            else round(sum(selected_confidences) / len(selected_confidences), 4)
        ),
    }


def analyze_interval(
    truth_index: int,
    truth: dict[str, Any],
    rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame_start = int(truth["frame_start"])
    frame_end = int(truth["frame_end"])
    truth_state = str(truth["state"])
    interval_rows = [
        row
        for row in rows
        if frame_start <= int(row.get("frame_id", -1)) <= frame_end
    ]
    interval_rows.sort(key=lambda row: int(row.get("frame_id", 0)))

    state_counts = Counter(str(row.get("control_state") or "unknown") for row in interval_rows)
    dominant_state = None
    if state_counts:
        dominant_state = sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    observed = len(interval_rows)
    matching = int(state_counts.get(truth_state, 0))
    engine_eligible = sum(bool(row.get("engine_eligible", False)) for row in interval_rows)
    handler_frames = sum(row.get("current_track_id") is not None for row in interval_rows)
    lost_frames_positive = sum(int(row.get("lost_frames", 0) or 0) > 0 for row in interval_rows)
    flight_active = sum(bool(row.get("flight_active", False)) for row in interval_rows)
    flight_supported = sum(bool(row.get("flight_supported", False)) for row in interval_rows)
    receiver_evidence = sum(
        bool(row.get("pass_flight_receiver_evidence", False)) for row in interval_rows
    )
    shot_active = sum(bool(row.get("shot_active", False)) for row in interval_rows)
    predicted_shot_flight = int(state_counts.get("shot_flight", 0))
    predicted_unknown = int(state_counts.get("unknown", 0))

    result: dict[str, Any] = {
        "truth_index": truth_index,
        "truth_state": truth_state,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "observed_frames": observed,
        "state_counts": dict(sorted(state_counts.items())),
        "dominant_state": dominant_state,
        "matching_frames": matching,
        "matching_rate": _rate(matching, observed),
        "engine_eligible_frames": engine_eligible,
        "engine_ineligible_frames": max(0, observed - engine_eligible),
        "current_handler_frames": handler_frames,
        "no_current_handler_frames": max(0, observed - handler_frames),
        "lost_frames_positive": lost_frames_positive,
        "flight_active_frames": flight_active,
        "flight_supported_frames": flight_supported,
        "receiver_evidence_frames": receiver_evidence,
        "shot_active_frames": shot_active,
        "predicted_shot_flight_frames": predicted_shot_flight,
        "predicted_unknown_frames": predicted_unknown,
        "current_track_switches": _track_switches(interval_rows, "current_track_id"),
        "reset_increments": _reset_increments(interval_rows),
        "state_runs": _state_runs(interval_rows),
    }

    if truth_state == "possessed":
        result["diagnostic_signals"] = {
            "possessed_truth_without_handler_frames": max(0, observed - handler_frames),
            "possessed_truth_with_lost_counter_frames": lost_frames_positive,
            "possessed_truth_predicted_shot_flight_frames": predicted_shot_flight,
            "possessed_truth_engine_ineligible_frames": max(0, observed - engine_eligible),
        }
    elif truth_state == "pass_flight":
        result["diagnostic_signals"] = {
            "pass_truth_without_locked_origin_frames": sum(
                row.get("current_track_id") is None and not bool(row.get("flight_active", False))
                for row in interval_rows
            ),
            "pass_truth_with_flight_active_frames": flight_active,
            "pass_truth_with_flight_supported_frames": flight_supported,
            "pass_truth_with_receiver_evidence_frames": receiver_evidence,
            "pass_truth_predicted_shot_flight_frames": predicted_shot_flight,
            "pass_truth_predicted_unknown_frames": predicted_unknown,
        }
    elif truth_state == "shot_flight":
        result["diagnostic_signals"] = {
            "shot_truth_predicted_shot_flight_frames": predicted_shot_flight,
            "shot_truth_shot_tracker_active_frames": shot_active,
            "shot_truth_not_predicted_shot_flight_frames": max(
                0, observed - predicted_shot_flight
            ),
            "shot_truth_with_handler_still_locked_frames": handler_frames,
        }
    else:
        result["diagnostic_signals"] = {}

    candidate_signals = _optional_candidate_signals(interval_rows)
    if candidate_signals is not None:
        result["candidate_signals"] = candidate_signals

    if frame_rows is not None:
        matching_frame_rows = [
            row
            for row in frame_rows
            if frame_start <= int(row.get("frame_id", -1)) <= frame_end
        ]
        matching_frame_rows.sort(key=lambda row: int(row.get("frame_id", 0)))
        result["p1_input_signals"] = _frame_input_signals(matching_frame_rows)
    return result


def diagnose(
    states: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    frame_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    intervals_raw = ground_truth.get("control_intervals", [])
    if not isinstance(intervals_raw, list):
        raise ValueError("ground truth control_intervals must be a list")

    intervals = [
        analyze_interval(index, interval, states, frame_rows)
        for index, interval in enumerate(intervals_raw)
        if isinstance(interval, dict)
    ]

    aggregate: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        truth_state = str(interval["truth_state"])
        bucket = aggregate.setdefault(
            truth_state,
            {
                "intervals": 0,
                "observed_frames": 0,
                "matching_frames": 0,
                "engine_ineligible_frames": 0,
                "no_current_handler_frames": 0,
                "lost_frames_positive": 0,
                "flight_active_frames": 0,
                "flight_supported_frames": 0,
                "receiver_evidence_frames": 0,
                "shot_active_frames": 0,
                "predicted_shot_flight_frames": 0,
                "predicted_unknown_frames": 0,
            },
        )
        bucket["intervals"] += 1
        for key in (
            "observed_frames",
            "matching_frames",
            "engine_ineligible_frames",
            "no_current_handler_frames",
            "lost_frames_positive",
            "flight_active_frames",
            "flight_supported_frames",
            "receiver_evidence_frames",
            "shot_active_frames",
            "predicted_shot_flight_frames",
            "predicted_unknown_frames",
        ):
            bucket[key] += int(interval.get(key, 0) or 0)

    for bucket in aggregate.values():
        bucket["matching_rate"] = _rate(bucket["matching_frames"], bucket["observed_frames"])

    return {
        "version": VERSION,
        "contract": "diagnostic_only_no_runtime_decision_changes",
        "state_rows": len(states),
        "frame_rows": None if frame_rows is None else len(frame_rows),
        "ground_truth_intervals": len(intervals),
        "aggregate_by_truth_state": dict(sorted(aggregate.items())),
        "intervals": intervals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose P2 control-state failures against human GT intervals.",
    )
    parser.add_argument("--states", required=True, help="P2 state JSONL artifact.")
    parser.add_argument("--ground-truth", required=True, help="Human P2 GT JSON.")
    parser.add_argument(
        "--frames",
        help="Optional analyzer frames JSONL for P1 ball/player/gameplay input correlation.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states_path = Path(args.states)
    ground_truth_path = Path(args.ground_truth)
    frame_rows = None if not args.frames else _load_jsonl(Path(args.frames))
    report = diagnose(
        _load_jsonl(states_path),
        _load_json(ground_truth_path),
        frame_rows,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"[P2_CONTROL_DIAG_FILE] {output_path.resolve()}")
    print(rendered)


if __name__ == "__main__":
    main()
