from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _parse_case(raw: str) -> tuple[str, str, int]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--case must be LABEL:RUN_ID:FRAME_ID")
    label, run_id, frame_raw = parts
    if not label.strip() or not run_id.strip():
        raise argparse.ArgumentTypeError("case label and run id cannot be empty")
    try:
        frame_id = int(frame_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame must be an integer") from exc
    return label.strip(), run_id.strip(), frame_id


def _track_prediction(row: dict[str, Any], track_id: Any) -> dict[str, Any] | None:
    if track_id is None:
        return None
    try:
        wanted = int(track_id)
    except (TypeError, ValueError):
        return None
    for prediction in list(row.get("team_predictions") or []):
        if not isinstance(prediction, dict):
            continue
        try:
            current = int(prediction.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if current == wanted:
            return prediction
    return None


def _round(value: Any, digits: int = 3) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    current = row.get("current_track_id")
    current_prediction = _track_prediction(row, current)
    raw_scores = dict((current_prediction or {}).get("raw_scores") or {})
    return {
        "frame": row.get("frame_id"),
        "time": _round(row.get("timestamp_sec"), 3),
        "state": row.get("live_control_state", row.get("control_state")),
        "current": current,
        "current_conf": _round(row.get("current_confidence"), 3),
        "candidate": row.get("candidate_track_id"),
        "candidate_frames": row.get("candidate_frames"),
        "candidate_score": _round(row.get("candidate_score"), 3),
        "candidate_peak": _round(row.get("candidate_peak_score"), 3),
        "candidate_threshold": _round(row.get("candidate_threshold"), 3),
        "raw_best": row.get("raw_best_track_id"),
        "raw_score": _round(row.get("raw_best_score"), 3),
        "raw_above": bool(row.get("raw_best_above_threshold", False)),
        "flight": bool(row.get("flight_active", False)),
        "flight_origin": row.get("flight_origin_track_id"),
        "shot": bool(row.get("shot_active", False)),
        "lost": row.get("lost_frames"),
        "reset": row.get("reset_count"),
        "current_team_raw": {key: _round(value, 3) for key, value in raw_scores.items()},
    }


def _fmt_track(value: Any) -> str:
    return "-" if value is None else str(value)


def _fmt_num(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Y" if value else "N"
    return str(value)


def _print_case(label: str, run_id: str, target: int, radius: int) -> dict[str, Any]:
    states_path = PROJECT_ROOT / "evaluation" / "runs" / run_id / "p2_states.jsonl"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    rows = _load_jsonl(states_path)
    selected = [
        row for row in rows
        if isinstance(row.get("frame_id"), int)
        and target - radius <= int(row["frame_id"]) <= target + radius
    ]
    if not selected:
        raise ValueError(f"No P2 state rows found around frame {target} in run {run_id}")

    compact = [_compact_row(row) for row in selected]
    print(f"\n===== {label} / {run_id} / target={target} =====")
    print("mark frame time state      current conf  cand cfr cscore raw rscore above flight origin shot lost reset")
    for row in compact:
        marker = ">>>" if int(row["frame"]) == target else "   "
        print(
            f"{marker} {int(row['frame']):5d} "
            f"{_fmt_num(row['time']):>6} "
            f"{str(row['state'] or '-'):>10} "
            f"{_fmt_track(row['current']):>7} "
            f"{_fmt_num(row['current_conf']):>5} "
            f"{_fmt_track(row['candidate']):>5} "
            f"{_fmt_num(row['candidate_frames']):>3} "
            f"{_fmt_num(row['candidate_score']):>6} "
            f"{_fmt_track(row['raw_best']):>4} "
            f"{_fmt_num(row['raw_score']):>6} "
            f"{_fmt_num(row['raw_above']):>5} "
            f"{_fmt_num(row['flight']):>6} "
            f"{_fmt_track(row['flight_origin']):>6} "
            f"{_fmt_num(row['shot']):>4} "
            f"{_fmt_num(row['lost']):>4} "
            f"{_fmt_num(row['reset']):>5}"
        )

    target_rows = [row for row in compact if int(row["frame"]) == target]
    if target_rows:
        target_row = target_rows[0]
        print("TARGET_DETAIL:", json.dumps(target_row, ensure_ascii=False, sort_keys=True))

    current_sequence = [row["current"] for row in compact]
    raw_sequence = [row["raw_best"] for row in compact]
    summary = {
        "label": label,
        "run_id": run_id,
        "target_frame": target,
        "radius": radius,
        "rows": len(compact),
        "unique_current_tracks": sorted({int(v) for v in current_sequence if v is not None}),
        "unique_raw_best_tracks": sorted({int(v) for v in raw_sequence if v is not None}),
        "target": target_rows[0] if target_rows else None,
    }
    print("CASE_SUMMARY:", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect P2 possession-holder temporal evidence around selected frames."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_parse_case,
        metavar="LABEL:RUN_ID:FRAME_ID",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=12,
        help="Frames before/after the target to print (default: 12).",
    )
    args = parser.parse_args()
    if args.radius < 0:
        parser.error("--radius must be >= 0")

    summaries = [
        _print_case(label, run_id, frame_id, args.radius)
        for label, run_id, frame_id in args.case
    ]
    print("\n===== HOLDER AUDIT SUMMARY =====")
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
