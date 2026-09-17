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

DEFAULT_RUNS = {
    "V1": "20260917_133606_837768",
    "V2": "20260917_141201_062495",
    "V3": "20260917_142607_777551",
    "V4": "20260917_143501_261725",
    "V5": "20260917_144307_273207",
}
CORE_STATES = {"possessed", "pass_flight", "shot_flight"}


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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _unknown_context(row: dict[str, Any], *, reset_on_frame: bool) -> str:
    if not bool(row.get("engine_eligible", False)):
        return "engine_ineligible"
    if reset_on_frame:
        return "reset_on_frame"
    if bool(row.get("shot_active", False)):
        return "shot_active_but_unknown"
    if bool(row.get("flight_supported", False)):
        if bool(row.get("pass_flight_receiver_evidence", False)):
            return "flight_supported_receiver_present_but_unknown"
        return "flight_supported_no_receiver"
    if bool(row.get("flight_active", False)):
        return "flight_active_not_supported"

    current_track_id = row.get("current_track_id")
    if current_track_id is not None:
        if _as_int(row.get("lost_frames"), 0) > 0:
            return "owner_retained_during_loss"
        return "owner_present_but_unknown"

    candidate_track_id = row.get("candidate_track_id")
    if candidate_track_id is not None:
        score = _as_float(row.get("candidate_score"), 0.0)
        threshold = _as_float(row.get("candidate_threshold"), 0.0)
        frames = _as_int(row.get("candidate_frames"), 0)
        required = max(1, _as_int(row.get("candidate_required_frames"), 1))
        if score >= threshold:
            if frames < required:
                return "candidate_above_threshold_maturing"
            return "candidate_above_threshold_ready_but_unknown"
        return "candidate_below_threshold"

    return "no_owner_no_candidate"


def _flight_owner_bucket(row: dict[str, Any]) -> str:
    if row.get("current_track_id") is None:
        return "owner_absent"
    lost = _as_int(row.get("lost_frames"), 0)
    if lost <= 0:
        return "owner_present_lost_0"
    if lost <= 2:
        return "owner_present_lost_1_2"
    if lost <= 6:
        return "owner_present_lost_3_6"
    return "owner_present_lost_gt6"


def _clip_diagnostics(clip: str, run_id: str) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    gt_path = PROJECT_ROOT / "evaluation" / "ground_truth" / f"{clip.lower()}.json"
    for path in (states_path, gt_path):
        if not path.exists():
            raise FileNotFoundError(f"{clip}: missing {path}")

    states = load_jsonl(states_path)
    by_frame = {int(row.get("frame_id", -1)): row for row in states if int(row.get("frame_id", -1)) >= 0}
    truth = _truth_state_by_frame(load_ground_truth(gt_path))

    counts: Counter[str] = Counter()
    by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    flight_owner: Counter[str] = Counter()
    flight_owner_by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    reset_reasons: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    core_frames = 0
    core_unknown = 0

    for frame_id, truth_state in sorted(truth.items()):
        if truth_state not in CORE_STATES:
            continue
        core_frames += 1
        row = by_frame.get(frame_id)
        if row is None:
            counts["missing_prediction"] += 1
            by_truth[truth_state]["missing_prediction"] += 1
            continue

        reset_count = _as_int(row.get("reset_count"), 0)
        previous_row = by_frame.get(frame_id - 1)
        previous_reset_count = (
            reset_count
            if previous_row is None
            else _as_int(previous_row.get("reset_count"), reset_count)
        )
        reset_on_frame = previous_row is not None and reset_count > previous_reset_count

        predicted = str(row.get("live_control_state", row.get("control_state")) or "unknown")
        if predicted != "unknown":
            continue

        core_unknown += 1
        context = _unknown_context(row, reset_on_frame=reset_on_frame)
        counts[context] += 1
        by_truth[truth_state][context] += 1

        if context == "flight_supported_no_receiver":
            bucket = _flight_owner_bucket(row)
            flight_owner[bucket] += 1
            flight_owner_by_truth[truth_state][bucket] += 1

        if reset_on_frame:
            reset_reasons[str(row.get("reset_reason") or "unknown")] += 1
        if len(examples[context]) < 5:
            examples[context].append(
                {
                    "frame_id": frame_id,
                    "truth_state": truth_state,
                    "gameplay_status": row.get("gameplay_status"),
                    "current_track_id": row.get("current_track_id"),
                    "lost_frames": row.get("lost_frames"),
                    "candidate_track_id": row.get("candidate_track_id"),
                    "candidate_frames": row.get("candidate_frames"),
                    "candidate_score": row.get("candidate_score"),
                    "candidate_threshold": row.get("candidate_threshold"),
                    "flight_active": row.get("flight_active"),
                    "flight_supported": row.get("flight_supported"),
                    "pass_flight_receiver_evidence": row.get("pass_flight_receiver_evidence"),
                    "shot_active": row.get("shot_active"),
                    "reset_count": row.get("reset_count"),
                    "reset_reason": row.get("reset_reason"),
                }
            )

    def render_counter(counter: Counter[str]) -> dict[str, Any]:
        total = sum(counter.values())
        return {
            key: {
                "frames": int(value),
                "share": round(value / max(total, 1), 4),
            }
            for key, value in counter.most_common()
        }

    return {
        "clip": clip,
        "run_id": run_id,
        "core_truth_frames": core_frames,
        "core_unknown_frames": core_unknown,
        "core_unknown_rate": round(core_unknown / max(core_frames, 1), 4),
        "unknown_contexts": render_counter(counts),
        "unknown_contexts_by_truth_state": {
            state: render_counter(counter) for state, counter in sorted(by_truth.items())
        },
        "flight_supported_no_receiver_owner_breakdown": render_counter(flight_owner),
        "flight_supported_no_receiver_owner_breakdown_by_truth_state": {
            state: render_counter(counter)
            for state, counter in sorted(flight_owner_by_truth.items())
        },
        "reset_reasons_on_unknown_frames": dict(reset_reasons.most_common()),
        "examples": dict(examples),
    }


def _aggregate(clips: list[dict[str, Any]]) -> dict[str, Any]:
    contexts: Counter[str] = Counter()
    by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    flight_owner: Counter[str] = Counter()
    flight_owner_by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    total_core = 0
    total_unknown = 0

    for clip in clips:
        total_core += int(clip["core_truth_frames"])
        total_unknown += int(clip["core_unknown_frames"])
        for context, row in dict(clip["unknown_contexts"]).items():
            contexts[context] += int(row["frames"])
        for truth_state, mapping in dict(clip["unknown_contexts_by_truth_state"]).items():
            for context, row in dict(mapping).items():
                by_truth[truth_state][context] += int(row["frames"])
        for bucket, row in dict(clip["flight_supported_no_receiver_owner_breakdown"]).items():
            flight_owner[bucket] += int(row["frames"])
        for truth_state, mapping in dict(
            clip["flight_supported_no_receiver_owner_breakdown_by_truth_state"]
        ).items():
            for bucket, row in dict(mapping).items():
                flight_owner_by_truth[truth_state][bucket] += int(row["frames"])

    def render(counter: Counter[str]) -> dict[str, Any]:
        total = sum(counter.values())
        return {
            key: {"frames": int(value), "share": round(value / max(total, 1), 4)}
            for key, value in counter.most_common()
        }

    owner_retained = sum(
        value for key, value in flight_owner.items() if key.startswith("owner_present_")
    )

    return {
        "core_truth_frames": total_core,
        "core_unknown_frames": total_unknown,
        "core_unknown_rate": round(total_unknown / max(total_core, 1), 4),
        "unknown_contexts": render(contexts),
        "unknown_contexts_by_truth_state": {
            state: render(counter) for state, counter in sorted(by_truth.items())
        },
        "flight_supported_no_receiver_owner_breakdown": render(flight_owner),
        "flight_supported_no_receiver_owner_breakdown_by_truth_state": {
            state: render(counter) for state, counter in sorted(flight_owner_by_truth.items())
        },
        "flight_supported_no_receiver_owner_retained_frames": int(owner_retained),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break Gate A core->UNKNOWN frames into observable runtime contexts.")
    for clip, default in DEFAULT_RUNS.items():
        parser.add_argument(f"--{clip.lower()}-run", default=default)
    parser.add_argument("--output", default="evaluation/gate_a_unknown_diagnostics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_map = {clip: str(getattr(args, f"{clip.lower()}_run")) for clip in DEFAULT_RUNS}
    clips = [_clip_diagnostics(clip, run_map[clip]) for clip in DEFAULT_RUNS]
    aggregate = _aggregate(clips)
    result = {
        "version": "courtflux_p2_gate_a_unknown_diag_v1",
        "contract": "diagnostic only; classifies observable snapshots for live core-truth frames predicted UNKNOWN",
        "runs": run_map,
        "aggregate": aggregate,
        "clips": clips,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "aggregate": aggregate,
        "clips": [
            {
                "clip": clip["clip"],
                "core_unknown_rate": clip["core_unknown_rate"],
                "unknown_contexts": clip["unknown_contexts"],
                "flight_supported_no_receiver_owner_breakdown": clip[
                    "flight_supported_no_receiver_owner_breakdown"
                ],
            }
            for clip in clips
        ],
        "output": str(output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
