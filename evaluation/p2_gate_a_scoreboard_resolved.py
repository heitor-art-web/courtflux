from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation import p2_gate_a_scoreboard_v3 as _base

PROJECT_ROOT = _base.PROJECT_ROOT
CLIPS = _base.CLIPS


def _predicted_intervals(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the final P2 timeline into semantic intervals for Gate A.

    Gate A measures the timeline that downstream P3 consumes after P2 processing. Prefer
    ``resolved_control_state`` when present because it includes P2's retrospective temporal
    resolution (for example late shot activation backfill). Fall back to the live channel
    for historical rows that predate the resolved field.
    """
    frames: list[tuple[int, str]] = []
    for row in states:
        if not isinstance(row, dict):
            continue
        try:
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if frame_id < 0:
            continue
        state = str(
            row.get("resolved_control_state")
            or row.get("live_control_state")
            or row.get("control_state")
            or "unknown"
        )
        frames.append((frame_id, state))

    frames.sort()
    if not frames:
        return []

    intervals: list[dict[str, Any]] = []
    start, previous, state = frames[0][0], frames[0][0], frames[0][1]
    for frame_id, next_state in frames[1:]:
        if frame_id == previous + 1 and next_state == state:
            previous = frame_id
            continue
        intervals.append(
            {
                "prediction_index": len(intervals),
                "state": state,
                "frame_start": start,
                "frame_end": previous,
            }
        )
        start, previous, state = frame_id, frame_id, next_state
    intervals.append(
        {
            "prediction_index": len(intervals),
            "state": state,
            "frame_start": start,
            "frame_end": previous,
        }
    )
    return intervals


# Reuse the frozen matcher/checks while changing only the P3-facing state channel.
_base._predicted_intervals = _predicted_intervals
_match_intervals = _base._match_intervals
_aggregate = _base._aggregate
_gate_checks = _base._gate_checks


def main() -> None:
    args = _base.parse_args()
    run_map = {clip: str(getattr(args, f"{clip.lower()}_run")) for clip in CLIPS}
    clips = [
        _base._clip_metrics(clip, run_map[clip], args.tolerance_frames)
        for clip in CLIPS
    ]
    aggregate = _base._aggregate(clips)
    gate = _base._gate_checks(clips, aggregate)
    result = {
        "version": "courtflux_p2_gate_a_p3_readiness_v4",
        "contract": (
            "P3 final timeline uses resolved_control_state when available; live control "
            "remains diagnostic; interval semantics primary; sequence order conditioned "
            "on detected adjacent episodes; V1-V5 development analysis; V6-V7 final "
            "validation-only"
        ),
        "timeline_state_channel": "resolved_control_state_then_live_fallback",
        "tolerance_frames": int(args.tolerance_frames),
        "runs": run_map,
        "clips": clips,
        "aggregate": aggregate,
        "gate": gate,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "timeline_state_channel": result["timeline_state_channel"],
                "gate": gate,
                "aggregate": aggregate,
                "clips": [
                    {
                        "clip": row["clip"],
                        "run_id": row["run_id"],
                        "prediction_coverage": row["prediction_coverage"],
                        "core_interval_detection_recall": row["timeline"]["core_interval_detection_recall"],
                        "core_unresolved_rate": row["timeline"]["core_unresolved_rate"],
                        "semantic_transition_order_accuracy": row["timeline"]["semantic_transitions"]["order_accuracy"],
                        "detected_transition_coverage": row["timeline"]["semantic_transitions"]["detected_transition_coverage"],
                        "frame_accuracy_diagnostic": row["frame_diagnostics"]["frame_accuracy"],
                    }
                    for row in clips
                ],
                "output": str(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
