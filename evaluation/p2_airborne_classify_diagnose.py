from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "courtflux_p2_airborne_classify_diagnose_v1"
RimBBox = tuple[float, float, float, float]


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


def _direct_ball(row: dict[str, Any]) -> tuple[float, float] | None:
    if not bool(row.get("selected_ball", False)):
        return None
    center = row.get("selected_ball_center")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    try:
        return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None


def _rim_candidates(row: dict[str, Any]) -> list[RimBBox]:
    court = row.get("court_state")
    if not isinstance(court, dict):
        return []
    rims: list[RimBBox] = []
    for evidence in list(court.get("basket_evidence", []) or []):
        if not isinstance(evidence, dict):
            continue
        if str(evidence.get("kind") or "").lower() != "rim":
            continue
        raw = evidence.get("bbox_px", evidence.get("box", evidence.get("bbox")))
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in raw[:4]]
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            rims.append((x1, y1, x2, y2))
    return rims


def _nearest_rim(ball: tuple[float, float], rims: list[RimBBox]) -> RimBBox | None:
    if not rims:
        return None
    bx, by = ball
    return min(
        rims,
        key=lambda rim: (
            ((rim[0] + rim[2]) / 2.0 - bx) ** 2
            + ((rim[1] + rim[3]) / 2.0 - by) ** 2
        ),
    )


def _rim_center(rim: RimBBox) -> tuple[float, float]:
    return (rim[0] + rim[2]) / 2.0, (rim[1] + rim[3]) / 2.0


def _rim_boxes_overlap(a: RimBBox, b: RimBBox) -> bool:
    return bool(
        min(a[2], b[2]) > max(a[0], b[0])
        and min(a[3], b[3]) > max(a[1], b[1])
    )


def _relative(ball: tuple[float, float], rim: RimBBox) -> tuple[float, float]:
    cx, cy = _rim_center(rim)
    return ball[0] - cx, ball[1] - cy


def _near_x(ball: tuple[float, float], rim: RimBBox) -> bool:
    cx, _ = _rim_center(rim)
    width = max(1.0, rim[2] - rim[0])
    # Deliberately reuse ShotEventTrackerV1's existing near_x geometry contract.
    return abs(ball[0] - cx) <= 2.5 * width


def diagnose(
    states: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    state_by_frame = {int(row.get("frame_id", -1)): row for row in states}
    frame_by_id = {int(row.get("frame_id", -1)): row for row in frames}

    motion_by_frame: dict[int, dict[str, Any]] = {}
    previous_geometry: tuple[int, tuple[float, float], RimBBox] | None = None

    for frame_id in sorted(frame_by_id):
        frame = frame_by_id[frame_id]
        ball = _direct_ball(frame)
        rim = None if ball is None else _nearest_rim(ball, _rim_candidates(frame))
        state = state_by_frame.get(frame_id, {})

        row: dict[str, Any] = {
            "frame_id": frame_id,
            "geometry_available": bool(ball is not None and rim is not None),
            "adjacent_geometry": False,
            "rim_geometry_coherent": False,
            "near_x": False,
            "rising_relative_to_rim": False,
            "distance_to_rim_decreasing": False,
            "flight_active": bool(state.get("flight_active", False)),
            "flight_supported": bool(state.get("flight_supported", False)),
            "receiver_evidence": bool(state.get("pass_flight_receiver_evidence", False)),
            "shot_active": bool(state.get("shot_active", False)),
            "early_shot_signal": False,
            "supported_early_shot_signal": False,
        }

        if ball is not None and rim is not None:
            row["near_x"] = _near_x(ball, rim)
            if previous_geometry is not None:
                previous_frame, previous_ball, previous_rim = previous_geometry
                adjacent = frame_id == previous_frame + 1
                coherent = adjacent and _rim_boxes_overlap(previous_rim, rim)
                row["adjacent_geometry"] = adjacent
                row["rim_geometry_coherent"] = coherent
                if coherent:
                    prev_rel = _relative(previous_ball, previous_rim)
                    cur_rel = _relative(ball, rim)
                    prev_distance = math.hypot(prev_rel[0], prev_rel[1])
                    cur_distance = math.hypot(cur_rel[0], cur_rel[1])
                    rising = cur_rel[1] < prev_rel[1]
                    toward = cur_distance < prev_distance
                    row["rising_relative_to_rim"] = rising
                    row["distance_to_rim_decreasing"] = toward
                    early = bool(
                        row["flight_active"]
                        and row["near_x"]
                        and rising
                        and toward
                    )
                    row["early_shot_signal"] = early
                    row["supported_early_shot_signal"] = bool(
                        early and row["flight_supported"]
                    )
            previous_geometry = (frame_id, ball, rim)
        else:
            # Do not bridge diagnostic motion through missing direct geometry. This keeps
            # the signal conservative and avoids manufacturing direction across gaps.
            previous_geometry = None

        motion_by_frame[frame_id] = row

    truth_intervals = list(ground_truth.get("control_intervals", []) or [])
    interval_reports: list[dict[str, Any]] = []
    aggregate: dict[str, Counter[str]] = {}

    for truth_index, truth in enumerate(truth_intervals):
        if not isinstance(truth, dict):
            continue
        truth_state = str(truth.get("state") or "unknown")
        start = int(truth["frame_start"])
        end = int(truth["frame_end"])
        rows = [
            motion_by_frame[frame_id]
            for frame_id in range(start, end + 1)
            if frame_id in motion_by_frame
        ]
        counts = Counter()
        for row in rows:
            counts["frames"] += 1
            for key in (
                "geometry_available",
                "adjacent_geometry",
                "rim_geometry_coherent",
                "near_x",
                "rising_relative_to_rim",
                "distance_to_rim_decreasing",
                "flight_active",
                "flight_supported",
                "receiver_evidence",
                "shot_active",
                "early_shot_signal",
                "supported_early_shot_signal",
            ):
                if bool(row.get(key, False)):
                    counts[key] += 1

        signals = [
            row["frame_id"] for row in rows if bool(row.get("early_shot_signal", False))
        ]
        supported_signals = [
            row["frame_id"]
            for row in rows
            if bool(row.get("supported_early_shot_signal", False))
        ]
        shot_active_frames = [
            row["frame_id"] for row in rows if bool(row.get("shot_active", False))
        ]

        report = {
            "truth_index": truth_index,
            "truth_state": truth_state,
            "frame_start": start,
            "frame_end": end,
            "observed_frames": counts["frames"],
            "geometry_available_frames": counts["geometry_available"],
            "flight_active_frames": counts["flight_active"],
            "flight_supported_frames": counts["flight_supported"],
            "receiver_evidence_frames": counts["receiver_evidence"],
            "shot_active_frames": counts["shot_active"],
            "early_shot_signal_frames": counts["early_shot_signal"],
            "supported_early_shot_signal_frames": counts[
                "supported_early_shot_signal"
            ],
            "first_early_shot_signal_frame": signals[0] if signals else None,
            "first_supported_early_shot_signal_frame": (
                supported_signals[0] if supported_signals else None
            ),
            "first_existing_shot_active_frame": (
                shot_active_frames[0] if shot_active_frames else None
            ),
        }
        interval_reports.append(report)

        counter = aggregate.setdefault(truth_state, Counter())
        counter.update(counts)
        counter["intervals"] += 1

    aggregate_report: dict[str, Any] = {}
    for state, counts in sorted(aggregate.items()):
        frames = int(counts.get("frames", 0))
        aggregate_report[state] = {
            "intervals": int(counts.get("intervals", 0)),
            "observed_frames": frames,
            "geometry_available_frames": int(counts.get("geometry_available", 0)),
            "flight_active_frames": int(counts.get("flight_active", 0)),
            "flight_supported_frames": int(counts.get("flight_supported", 0)),
            "receiver_evidence_frames": int(counts.get("receiver_evidence", 0)),
            "shot_active_frames": int(counts.get("shot_active", 0)),
            "early_shot_signal_frames": int(counts.get("early_shot_signal", 0)),
            "early_shot_signal_rate": _rate(
                int(counts.get("early_shot_signal", 0)), frames
            ),
            "supported_early_shot_signal_frames": int(
                counts.get("supported_early_shot_signal", 0)
            ),
            "supported_early_shot_signal_rate": _rate(
                int(counts.get("supported_early_shot_signal", 0)), frames
            ),
        }

    return {
        "version": VERSION,
        "contract": "diagnostic_only_no_runtime_decision_changes",
        "signal_contract": (
            "direct_adjacent_ball_and_overlapping_rim_geometry;"
            "flight_active;existing_near_x;relative_rise;distance_to_rim_decreasing"
        ),
        "states_rows": len(states),
        "frame_rows": len(frames),
        "ground_truth_intervals": len(truth_intervals),
        "aggregate_by_truth_state": aggregate_report,
        "intervals": interval_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose generic airborne evidence vs early shot-oriented motion."
    )
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = diagnose(
        _load_jsonl(args.states),
        _load_jsonl(args.frames),
        _load_json(args.ground_truth),
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"[P2_AIRBORNE_DIAG_FILE] {args.output.resolve()}")
    print(rendered, end="")


if __name__ == "__main__":
    main()
