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

NUMERIC_FIELDS = (
    "shot_preflight_ball_to_rim_distance_norm",
    "shot_preflight_horizontal_distance_rim_widths",
    "shot_preflight_relative_y_rim_heights",
    "shot_preflight_delta_distance_norm",
    "shot_preflight_delta_relative_y",
    "shot_preflight_approach_streak",
    "shot_preflight_rising_streak",
)
BOOLEAN_FIELDS = (
    "shot_preflight_available",
    "shot_preflight_direct_observed",
    "shot_preflight_near_x",
    "shot_preflight_above",
    "shot_preflight_rim_band",
    "shot_preflight_trajectory_start",
)


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


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 4)


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p75": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": round(max(values), 4),
    }


def _bool_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    total = len(rows)
    positive = sum(1 for row in rows if bool(row.get(field, False)))
    return {
        "frames": total,
        "true": positive,
        "share": round(positive / max(total, 1), 4),
    }


def _summarize(rows: list[dict[str, Any]], truth: dict[int, str]) -> dict[str, Any]:
    truth_counts: Counter[str] = Counter()
    for row in rows:
        try:
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        truth_counts[truth.get(frame_id, "unlabeled")] += 1
    return {
        "frames": len(rows),
        "truth": dict(sorted(truth_counts.items())),
        "boolean": {field: _bool_summary(rows, field) for field in BOOLEAN_FIELDS},
        "numeric": {field: _numeric_summary(rows, field) for field in NUMERIC_FIELDS},
    }


def _eligible_preflight(row: dict[str, Any]) -> bool:
    return bool(
        row.get("shot_preflight_available", False)
        and row.get("flight_active", False)
        and not row.get("pass_flight_receiver_evidence", False)
        and not row.get("shot_active", False)
        and row.get("engine_eligible", False)
    )


def analyze(
    run_id: str,
    ground_truth_path: Path,
    *,
    lookback_frames: int,
    negative_exclusion_frames: int,
) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not ground_truth_path.exists():
        raise FileNotFoundError(ground_truth_path)

    states = [row for row in load_jsonl(states_path) if isinstance(row, dict)]
    by_frame = {
        int(row.get("frame_id", -1)): row
        for row in states
        if int(row.get("frame_id", -1)) >= 0
    }
    truth = _truth_state_by_frame(load_ground_truth(ground_truth_path))

    if not any("shot_preflight_policy" in row for row in states):
        raise RuntimeError(
            "run does not contain shot_preflight diagnostics; rerun the canonical probe after the shot-preflight instrumentation commit"
        )

    activation_frames: list[int] = []
    previous_active = False
    for row in sorted(states, key=lambda item: int(item.get("frame_id", -1))):
        active = bool(row.get("shot_active", False))
        if active and not previous_active:
            activation_frames.append(int(row.get("frame_id", -1)))
        previous_active = active

    confirmed_activations = [
        frame_id
        for frame_id in activation_frames
        if bool(by_frame.get(frame_id, {}).get("resolved_shot_backfill_applied", False))
    ]

    positive_frames: set[int] = set()
    activation_windows: list[dict[str, Any]] = []
    for activation in confirmed_activations:
        activation_row = by_frame.get(activation, {})
        flight_start_raw = activation_row.get("resolved_shot_backfill_flight_start_frame")
        try:
            flight_start = int(flight_start_raw)
        except (TypeError, ValueError):
            flight_start = max(0, activation - lookback_frames)
        start = max(flight_start, activation - max(1, int(lookback_frames)))
        rows: list[dict[str, Any]] = []
        for frame_id in range(start, activation):
            row = by_frame.get(frame_id)
            if row is None or not _eligible_preflight(row):
                continue
            positive_frames.add(frame_id)
            rows.append(row)
        activation_windows.append(
            {
                "activation_frame": activation,
                "resolved_flight_start_frame": flight_start,
                "window_start_frame": start,
                "eligible_frames": [int(row.get("frame_id", -1)) for row in rows],
                "summary": _summarize(rows, truth),
                "last_rows": [
                    {
                        "frame_id": int(row.get("frame_id", -1)),
                        "truth_state": truth.get(int(row.get("frame_id", -1)), "unlabeled"),
                        "distance_norm": row.get("shot_preflight_ball_to_rim_distance_norm"),
                        "horizontal_rim_widths": row.get("shot_preflight_horizontal_distance_rim_widths"),
                        "relative_y": row.get("shot_preflight_relative_y_rim_heights"),
                        "delta_distance": row.get("shot_preflight_delta_distance_norm"),
                        "delta_relative_y": row.get("shot_preflight_delta_relative_y"),
                        "approach_streak": row.get("shot_preflight_approach_streak"),
                        "rising_streak": row.get("shot_preflight_rising_streak"),
                        "near_x": row.get("shot_preflight_near_x"),
                        "above": row.get("shot_preflight_above"),
                    }
                    for row in rows[-8:]
                ],
            }
        )

    exclusion: set[int] = set()
    radius = max(0, int(negative_exclusion_frames))
    for activation in activation_frames:
        exclusion.update(range(max(0, activation - radius), activation + radius + 1))

    negative_rows = [
        row
        for row in states
        if _eligible_preflight(row)
        and int(row.get("frame_id", -1)) not in positive_frames
        and int(row.get("frame_id", -1)) not in exclusion
    ]
    positive_rows = [by_frame[frame_id] for frame_id in sorted(positive_frames) if frame_id in by_frame]

    return {
        "version": "courtflux_p2_gate_a_shot_preflight_diag_v1",
        "run_id": run_id,
        "ground_truth": str(ground_truth_path.relative_to(PROJECT_ROOT)),
        "lookback_frames": int(lookback_frames),
        "negative_exclusion_frames": int(negative_exclusion_frames),
        "shot_activation_frames": activation_frames,
        "confirmed_backfill_activation_frames": confirmed_activations,
        "positive_pre_shot": _summarize(positive_rows, truth),
        "negative_generic_flight": _summarize(negative_rows, truth),
        "activation_windows": activation_windows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare causal shot-preflight geometry before confirmed shot activations against generic flight."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", default="evaluation/ground_truth/v1.json")
    parser.add_argument("--lookback-frames", type=int, default=12)
    parser.add_argument("--negative-exclusion-frames", type=int, default=12)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    result = analyze(
        str(args.run),
        gt_path,
        lookback_frames=int(args.lookback_frames),
        negative_exclusion_frames=int(args.negative_exclusion_frames),
    )
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
