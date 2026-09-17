from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import load_ground_truth, load_jsonl

CORE_STATES = {"possessed", "pass_flight", "shot_flight"}
FOCUS_CONTEXTS = {"flight_supported_no_receiver", "no_owner_no_candidate"}


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unknown_context(row: dict[str, Any]) -> str:
    if not bool(row.get("engine_eligible", False)):
        return "engine_ineligible"
    if bool(row.get("shot_active", False)):
        return "shot_active_but_unknown"
    if bool(row.get("flight_supported", False)):
        if bool(row.get("pass_flight_receiver_evidence", False)):
            return "flight_supported_receiver_present_but_unknown"
        return "flight_supported_no_receiver"
    if bool(row.get("flight_active", False)):
        return "flight_active_not_supported"
    if row.get("current_track_id") is not None:
        return "owner_present_but_unknown"
    if row.get("candidate_track_id") is not None:
        return "thresholded_candidate_present"
    return "no_owner_no_candidate"


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
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(value, 4)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": round(max(values), 4),
    }


def _streak_summary(values: list[int]) -> dict[str, Any]:
    base = _summary([float(value) for value in values])
    base.update(
        {
            "frames_with_streak_ge_2": sum(1 for value in values if value >= 2),
            "frames_with_streak_ge_3": sum(1 for value in values if value >= 3),
            "frames_with_streak_ge_4": sum(1 for value in values if value >= 4),
        }
    )
    return base


def _raw_bucket(row: dict[str, Any]) -> str:
    raw_track = row.get("raw_best_track_id")
    if raw_track is None:
        return "raw_best_none"
    raw_score = _as_float(row.get("raw_best_score"), 0.0)
    threshold = _as_float(row.get("candidate_threshold"), 0.0)
    if bool(row.get("flight_active", False)):
        origin = row.get("flight_origin_track_id")
        if origin is not None and int(raw_track) == int(origin):
            return "raw_best_is_flight_origin"
        if raw_score >= threshold:
            return "raw_best_distinct_at_or_above_threshold"
        return "raw_best_distinct_below_threshold"
    if raw_score >= threshold:
        return "raw_best_at_or_above_threshold"
    return "raw_best_below_threshold"


def _raw_streak_by_frame(states: list[dict[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    previous_frame: int | None = None
    previous_track: int | None = None
    streak = 0
    for row in sorted(states, key=lambda item: int(item.get("frame_id", -1))):
        frame_id = int(row.get("frame_id", -1))
        if frame_id < 0:
            continue
        raw_track = row.get("raw_best_track_id")
        if raw_track is None:
            streak = 0
            previous_track = None
        else:
            track_id = int(raw_track)
            if (
                previous_frame is not None
                and frame_id == previous_frame + 1
                and previous_track is not None
                and track_id == previous_track
            ):
                streak += 1
            else:
                streak = 1
            previous_track = track_id
        result[frame_id] = streak
        previous_frame = frame_id
    return result


def analyze(run_id: str, ground_truth: Path) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not ground_truth.exists():
        raise FileNotFoundError(ground_truth)

    states = load_jsonl(states_path)
    by_frame = {
        int(row.get("frame_id", -1)): row
        for row in states
        if isinstance(row, dict) and int(row.get("frame_id", -1)) >= 0
    }
    raw_streak = _raw_streak_by_frame([row for row in states if isinstance(row, dict)])
    truth = _truth_state_by_frame(load_ground_truth(ground_truth))

    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    buckets_by_truth: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    scores: dict[str, list[float]] = defaultdict(list)
    threshold_gaps: dict[str, list[float]] = defaultdict(list)
    streaks: dict[str, list[int]] = defaultdict(list)
    score_by_truth: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    gap_by_truth: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    streak_by_truth: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame_id, truth_state in sorted(truth.items()):
        if truth_state not in CORE_STATES:
            continue
        row = by_frame.get(frame_id)
        if row is None:
            continue
        predicted = str(row.get("live_control_state", row.get("control_state")) or "unknown")
        if predicted != "unknown":
            continue
        context = _unknown_context(row)
        if context not in FOCUS_CONTEXTS:
            continue

        bucket = _raw_bucket(row)
        buckets[context][bucket] += 1
        buckets_by_truth[context][truth_state][bucket] += 1

        if row.get("raw_best_track_id") is not None:
            score = _as_float(row.get("raw_best_score"), 0.0)
            threshold = _as_float(row.get("candidate_threshold"), 0.0)
            gap = threshold - score
            streak = int(raw_streak.get(frame_id, 0))
            scores[context].append(score)
            threshold_gaps[context].append(gap)
            streaks[context].append(streak)
            score_by_truth[context][truth_state].append(score)
            gap_by_truth[context][truth_state].append(gap)
            streak_by_truth[context][truth_state].append(streak)

        if len(examples[context]) < 12:
            examples[context].append(
                {
                    "frame_id": frame_id,
                    "truth_state": truth_state,
                    "raw_bucket": bucket,
                    "raw_best_track_id": row.get("raw_best_track_id"),
                    "raw_best_score": row.get("raw_best_score"),
                    "raw_best_streak_frames": raw_streak.get(frame_id, 0),
                    "candidate_threshold": row.get("candidate_threshold"),
                    "flight_origin_track_id": row.get("flight_origin_track_id"),
                    "flight_frames": row.get("flight_frames"),
                    "current_track_id": row.get("current_track_id"),
                }
            )

    def render(counter: Counter[str]) -> dict[str, Any]:
        total = sum(counter.values())
        return {
            key: {"frames": int(value), "share": round(value / max(total, 1), 4)}
            for key, value in counter.most_common()
        }

    contexts: dict[str, Any] = {}
    for context in sorted(FOCUS_CONTEXTS):
        contexts[context] = {
            "raw_candidate_breakdown": render(buckets[context]),
            "raw_score": _summary(scores[context]),
            "threshold_minus_raw_score": _summary(threshold_gaps[context]),
            "raw_best_streak_frames": _streak_summary(streaks[context]),
            "by_truth_state": {
                truth_state: {
                    "raw_candidate_breakdown": render(counter),
                    "raw_score": _summary(score_by_truth[context][truth_state]),
                    "threshold_minus_raw_score": _summary(gap_by_truth[context][truth_state]),
                    "raw_best_streak_frames": _streak_summary(
                        streak_by_truth[context][truth_state]
                    ),
                }
                for truth_state, counter in sorted(buckets_by_truth[context].items())
            },
            "examples": examples[context],
        }

    return {
        "version": "courtflux_p2_gate_a_raw_candidate_diag_v2",
        "run_id": run_id,
        "ground_truth": str(ground_truth.relative_to(PROJECT_ROOT)),
        "contexts": contexts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pre-threshold possession candidates for Gate A UNKNOWN frames."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", default="evaluation/ground_truth/v1.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
