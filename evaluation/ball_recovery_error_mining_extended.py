from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from evaluation import ball_recovery_error_mining_base as base


DIRECT_SOURCES = base.DIRECT_SOURCES


def parse_horizons(raw: str) -> list[int]:
    values: set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid horizon {token!r}; expected comma-separated positive integers") from exc
        if value <= 0:
            raise ValueError(f"Horizons must be positive; got {value}")
        values.add(value)
    if not values:
        raise ValueError("At least one positive horizon is required")
    return sorted(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine CourtFlux P0.2 recovery events, replay the exact persisted candidate "
            "stream with reacquisition OFF versus ON, and attribute downstream damage "
            "or benefit to non-overlapping trajectory-divergence episodes."
        )
    )
    parser.add_argument("--frames", required=True, help="Instrumented production frames.jsonl")
    parser.add_argument("--truth", required=True, help="Sparse ball GT per_frame.json")
    parser.add_argument("--video", default=None, help="Optional source video for shape and visuals")
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    parser.add_argument("--video-frame-offset", type=int, default=0)
    parser.add_argument("--window-before", type=int, default=5)
    parser.add_argument("--window-after", type=int, default=10)
    parser.add_argument(
        "--horizons",
        default="1,3,5,10,20,30",
        help="Comma-separated downstream event horizons in production frames.",
    )
    parser.add_argument(
        "--episode-convergence-frames",
        type=int,
        default=3,
        help=(
            "Consecutive materially converged frames required to close a divergence "
            "episode. Shorter converged gaps remain inside one episode."
        ),
    )
    parser.add_argument(
        "--divergence-center-tolerance-px",
        type=float,
        default=1.0,
        help="Output/state center separation above which replay states are materially different.",
    )
    parser.add_argument(
        "--episode-visual-panels",
        type=int,
        default=12,
        help="Maximum sampled panels in each episode contact sheet.",
    )
    parser.add_argument(
        "--top-episodes",
        type=int,
        default=10,
        help="Maximum harmful/beneficial episodes listed in report.md.",
    )
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--bbox-iou", type=float, default=0.25)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def vector(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return np.asarray([float(value[0]), float(value[1])], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def distance(a: Any, b: Any) -> float | None:
    va, vb = vector(a), vector(b)
    if va is None or vb is None:
        return None
    return float(np.linalg.norm(va - vb))


def describe(values: Iterable[Any]) -> dict[str, Any]:
    cleaned = [float(v) for v in values if safe_float(v) is not None]
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(cleaned, dtype=np.float64)
    return {
        "count": len(cleaned),
        "mean": base.rounded(np.mean(array)),
        "median": base.rounded(np.median(array)),
        "p10": base.rounded(np.percentile(array, 10)),
        "p25": base.rounded(np.percentile(array, 25)),
        "p75": base.rounded(np.percentile(array, 75)),
        "p90": base.rounded(np.percentile(array, 90)),
        "min": base.rounded(np.min(array)),
        "max": base.rounded(np.max(array)),
    }


def _candidate_trace_stats(row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    assoc = base.association(row)
    traces = [dict(c) for c in assoc.get("candidates") or [] if isinstance(c, dict)]
    selected_index = safe_int(assoc.get("selected_index"))
    selected = (
        traces[selected_index]
        if selected_index is not None and 0 <= selected_index < len(traces)
        else None
    )
    confidences = [safe_float(c.get("confidence")) or 0.0 for c in traces]
    plausible = 0
    outside = 0
    high_conf_extended = 0
    score_values: list[tuple[int, float]] = []
    confidence_values: list[tuple[int, float]] = []
    for index, trace in enumerate(traces):
        confidence_values.append((index, safe_float(trace.get("confidence")) or 0.0))
        details = trace.get("association")
        if isinstance(details, dict):
            if bool(details.get("plausible")):
                plausible += 1
            else:
                outside += 1
            if bool(details.get("high_confidence_gate_used")):
                high_conf_extended += 1
            score = safe_float(details.get("score"))
            if score is not None:
                score_values.append((index, score))

    confidence_values.sort(key=lambda item: item[1], reverse=True)
    score_values.sort(key=lambda item: item[1], reverse=True)
    conf_rank = None
    score_rank = None
    if selected_index is not None:
        conf_rank = next(
            (rank for rank, (index, _) in enumerate(confidence_values, start=1) if index == selected_index),
            None,
        )
        score_rank = next(
            (rank for rank, (index, _) in enumerate(score_values, start=1) if index == selected_index),
            None,
        )

    selected_conf = safe_float(selected.get("confidence")) if isinstance(selected, dict) else None
    nonselected_conf = [
        value for index, value in confidence_values if selected_index is None or index != selected_index
    ]
    next_best_conf = max(nonselected_conf) if nonselected_conf else None
    selected_details = (
        selected.get("association")
        if isinstance(selected, dict) and isinstance(selected.get("association"), dict)
        else {}
    )
    size_similarity = safe_float(selected_details.get("size_similarity"))
    distance_norm = safe_float(selected_details.get("distance_norm"))
    gate_norm = safe_float(selected_details.get("gate_norm"))
    excess = (
        None
        if distance_norm is None or gate_norm is None
        else distance_norm - gate_norm
    )

    return {
        "candidate_count": len(traces),
        "confidence": describe(confidences),
        "plausible_count": plausible,
        "outside_gate_count": outside,
        "high_confidence_extended_gate_count": high_conf_extended,
        "selected_confidence_rank": conf_rank,
        "selected_score_rank": score_rank,
        "selected_margin_to_next_confidence": (
            None
            if selected_conf is None or next_best_conf is None
            else base.rounded(selected_conf - next_best_conf)
        ),
        "selected_confidence_x_size_similarity": (
            None
            if selected_conf is None or size_similarity is None
            else base.rounded(selected_conf * size_similarity)
        ),
        "selected_distance_excess_norm": base.rounded(excess),
        "selected_plausible": (
            None if not selected_details else bool(selected_details.get("plausible"))
        ),
        "selected_high_confidence_gate_used": (
            None
            if not selected_details
            else bool(selected_details.get("high_confidence_gate_used"))
        ),
        "true_candidate_count": len(event.get("true_candidate_indices") or []),
    }


def _output_center(
    record: dict[str, Any] | None,
    row: dict[str, Any] | None,
    frame_shape: tuple[int, int, int],
) -> list[float] | None:
    if not isinstance(record, dict):
        return None
    source = str(record.get("source") or "missing")
    if source in DIRECT_SOURCES:
        selected_index = safe_int(record.get("selected_index"))
        candidates = base.persisted_candidates(row or {})
        if selected_index is not None and 0 <= selected_index < len(candidates):
            center = candidates[selected_index].get("center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                return [float(center[0]), float(center[1])]
        return None
    if source != "predicted_hold":
        return None

    pre = record.get("pre_state") or {}
    last_center = vector(pre.get("last_center"))
    velocity = vector(pre.get("velocity"))
    last_frame_id = safe_int(pre.get("last_frame_id"))
    frame_id = safe_int(record.get("frame_id"))
    if last_center is None or velocity is None or last_frame_id is None or frame_id is None:
        return None
    dt = max(frame_id - last_frame_id, 0)
    predicted = last_center + velocity * float(dt)
    height, width = frame_shape[:2]
    predicted[0] = float(np.clip(predicted[0], 0, max(width - 1, 0)))
    predicted[1] = float(np.clip(predicted[1], 0, max(height - 1, 0)))
    return [float(predicted[0]), float(predicted[1])]


def enrich_replay_records(
    rows: list[dict[str, Any]],
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    frame_shape: tuple[int, int, int],
) -> None:
    by_frame = {int(row.get("frame_id", 0)): row for row in rows}
    for replay in (off, on):
        for frame_id, record in replay.items():
            record["output_center"] = _output_center(
                record,
                by_frame.get(frame_id),
                frame_shape,
            )


def compare_replays(
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    center_tolerance_px: float = 1.0,
) -> list[dict[str, Any]]:
    tolerance = max(float(center_tolerance_px), 0.0)
    records: list[dict[str, Any]] = []
    for frame_id in sorted(set(off) | set(on)):
        a, b = off.get(frame_id), on.get(frame_id)
        if a is None or b is None:
            records.append(
                {
                    "frame_id": frame_id,
                    "material_diverged": True,
                    "missing_counterfactual_side": True,
                }
            )
            continue

        center_sep = distance(a.get("output_center"), b.get("output_center"))
        presence_mismatch = bool(a.get("trajectory_present")) != bool(b.get("trajectory_present"))
        output_diverged = presence_mismatch or (
            center_sep is not None and center_sep > tolerance
        )

        pre_a, pre_b = a.get("pre_state") or {}, b.get("pre_state") or {}
        state_center_sep = distance(pre_a.get("last_center"), pre_b.get("last_center"))
        velocity_sep = distance(pre_a.get("velocity"), pre_b.get("velocity"))
        last_frame_mismatch = pre_a.get("last_frame_id") != pre_b.get("last_frame_id")
        state_core_diverged = (
            last_frame_mismatch
            or (state_center_sep is not None and state_center_sep > tolerance)
            or (velocity_sep is not None and velocity_sep > tolerance)
        )

        source_diverged = str(a.get("source")) != str(b.get("source"))
        index_diverged = a.get("selected_index") != b.get("selected_index")
        reacquired_diverged = bool(a.get("reacquired")) != bool(b.get("reacquired"))
        decision_diverged = source_diverged or index_diverged or reacquired_diverged

        track_state_diverged = str(a.get("track_state")) != str(b.get("track_state"))
        challenger_diverged = (
            int(pre_a.get("challenger_streak") or 0)
            != int(pre_b.get("challenger_streak") or 0)
        )
        diagnostic_state_diverged = state_core_diverged or track_state_diverged or challenger_diverged

        material = bool(output_diverged or decision_diverged or state_core_diverged)
        truth = truth_by_frame.get(frame_id)
        truth_visible = None if truth is None else bool(truth.get("visible"))
        off_correct = None if truth is None else bool(a.get("selected_correct"))
        on_correct = None if truth is None else bool(b.get("selected_correct"))
        effect = "unlabeled"
        if truth is not None:
            if not truth_visible:
                effect = "ball_not_visible"
            elif on_correct and not off_correct:
                effect = "v2_better"
            elif off_correct and not on_correct:
                effect = "off_better"
            elif on_correct and off_correct:
                effect = "both_correct"
            else:
                effect = "both_wrong"

        records.append(
            {
                "frame_id": frame_id,
                "timestamp_sec": b.get("timestamp_sec"),
                "truth_available": truth is not None,
                "truth_visible": truth_visible,
                "candidate_count": b.get("candidate_count"),
                "true_candidate_available": b.get("true_candidate_available"),
                "off_source": a.get("source"),
                "v2_source": b.get("source"),
                "off_track_state": a.get("track_state"),
                "v2_track_state": b.get("track_state"),
                "off_selected_index": a.get("selected_index"),
                "v2_selected_index": b.get("selected_index"),
                "off_reacquired": bool(a.get("reacquired")),
                "v2_reacquired": bool(b.get("reacquired")),
                "off_output_center": a.get("output_center"),
                "v2_output_center": b.get("output_center"),
                "output_center_separation_px": base.rounded(center_sep, 3),
                "pre_state_center_separation_px": base.rounded(state_center_sep, 3),
                "pre_state_velocity_separation_px_per_frame": base.rounded(velocity_sep, 6),
                "output_diverged": bool(output_diverged),
                "decision_diverged": bool(decision_diverged),
                "state_core_diverged": bool(state_core_diverged),
                "diagnostic_state_diverged": bool(diagnostic_state_diverged),
                "track_state_diverged": bool(track_state_diverged),
                "challenger_state_diverged": bool(challenger_diverged),
                "material_diverged": material,
                "off_correct": off_correct,
                "v2_correct": on_correct,
                "truth_effect": effect,
            }
        )
    return records


def _effect_metrics(
    frame_ids: Iterable[int],
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    labeled_ids: list[int] = []
    visible_ids: list[int] = []
    off_better_ids: list[int] = []
    v2_better_ids: list[int] = []

    for frame_id in sorted(set(int(v) for v in frame_ids)):
        truth = truth_by_frame.get(frame_id)
        if truth is None:
            continue
        a, b = off.get(frame_id), on.get(frame_id)
        if a is None or b is None:
            continue
        labeled_ids.append(frame_id)
        visible = bool(truth.get("visible"))
        counts["visible" if visible else "not_visible"] += 1
        if visible:
            visible_ids.append(frame_id)
            a_correct = bool(a.get("selected_correct"))
            b_correct = bool(b.get("selected_correct"))
            counts["off_correct"] += int(a_correct)
            counts["v2_correct"] += int(b_correct)
            if a_correct and not b_correct:
                counts["off_better"] += 1
                off_better_ids.append(frame_id)
            elif b_correct and not a_correct:
                counts["v2_better"] += 1
                v2_better_ids.append(frame_id)
            elif a_correct and b_correct:
                counts["both_correct"] += 1
            else:
                counts["both_wrong"] += 1

            if bool(a.get("true_candidate_available")):
                counts["candidate_available"] += 1
                counts["off_conditioned_correct"] += int(
                    a_correct and str(a.get("source")) in DIRECT_SOURCES
                )
                counts["v2_conditioned_correct"] += int(
                    b_correct and str(b.get("source")) in DIRECT_SOURCES
                )
        else:
            counts["off_direct_not_visible"] += int(str(a.get("source")) in DIRECT_SOURCES)
            counts["v2_direct_not_visible"] += int(str(b.get("source")) in DIRECT_SOURCES)
            counts["off_output_not_visible"] += int(bool(a.get("trajectory_present")))
            counts["v2_output_not_visible"] += int(bool(b.get("trajectory_present")))

    return {
        "truth_samples": len(labeled_ids),
        "visible_truth_samples": counts["visible"],
        "not_visible_truth_samples": counts["not_visible"],
        "candidate_available_visible_truth_samples": counts["candidate_available"],
        "off_correct_visible": counts["off_correct"],
        "v2_correct_visible": counts["v2_correct"],
        "net_v2_correct_delta": counts["v2_correct"] - counts["off_correct"],
        "off_better_visible": counts["off_better"],
        "v2_better_visible": counts["v2_better"],
        "both_correct_visible": counts["both_correct"],
        "both_wrong_visible": counts["both_wrong"],
        "off_conditioned_correct": counts["off_conditioned_correct"],
        "v2_conditioned_correct": counts["v2_conditioned_correct"],
        "conditioned_correct_delta": (
            counts["v2_conditioned_correct"] - counts["off_conditioned_correct"]
        ),
        "off_direct_not_visible": counts["off_direct_not_visible"],
        "v2_direct_not_visible": counts["v2_direct_not_visible"],
        "direct_not_visible_delta": (
            counts["v2_direct_not_visible"] - counts["off_direct_not_visible"]
        ),
        "off_output_not_visible": counts["off_output_not_visible"],
        "v2_output_not_visible": counts["v2_output_not_visible"],
        "output_not_visible_delta": (
            counts["v2_output_not_visible"] - counts["off_output_not_visible"]
        ),
        "first_off_better_frame": off_better_ids[0] if off_better_ids else None,
        "last_off_better_frame": off_better_ids[-1] if off_better_ids else None,
        "first_v2_better_frame": v2_better_ids[0] if v2_better_ids else None,
        "last_v2_better_frame": v2_better_ids[-1] if v2_better_ids else None,
    }


def enrich_events(
    events: list[dict[str, Any]],
    production_by_frame: dict[int, dict[str, Any]],
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    horizons: list[int],
) -> list[dict[str, Any]]:
    event_ids = sorted(int(event["frame_id"]) for event in events)
    max_frame = max(set(off) | set(on)) if off or on else 0
    for position, event in enumerate(sorted(events, key=lambda item: int(item["frame_id"]))):
        frame_id = int(event["frame_id"])
        event["event_id"] = f"R{position + 1:03d}"
        previous_event = event_ids[position - 1] if position > 0 else None
        next_event = event_ids[position + 1] if position + 1 < len(event_ids) else None
        event["recovery_spacing"] = {
            "previous_recovery_frame": previous_event,
            "frames_since_previous_recovery": (
                None if previous_event is None else frame_id - previous_event
            ),
            "next_recovery_frame": next_event,
            "frames_to_next_recovery": (
                None if next_event is None else next_event - frame_id
            ),
        }
        event["candidate_pool"] = _candidate_trace_stats(
            production_by_frame.get(frame_id, {}),
            event,
        )
        horizon_payload: dict[str, Any] = {}
        for horizon in horizons:
            end = min(frame_id + int(horizon), max_frame)
            metrics = _effect_metrics(
                range(frame_id + 1, end + 1),
                off,
                on,
                truth_by_frame,
            )
            metrics["production_frames_requested"] = int(horizon)
            metrics["production_frames_available"] = max(0, end - frame_id)
            metrics["crosses_next_recovery"] = bool(
                next_event is not None and next_event <= end
            )
            horizon_payload[str(horizon)] = metrics
        event["downstream_horizons"] = horizon_payload

        isolated_end = (
            min(next_event - 1, max_frame)
            if next_event is not None
            else max_frame
        )
        event["isolated_until_next_recovery"] = {
            "start_frame": frame_id + 1,
            "end_frame": isolated_end if isolated_end >= frame_id + 1 else None,
            **_effect_metrics(
                range(frame_id + 1, isolated_end + 1)
                if isolated_end >= frame_id + 1
                else [],
                off,
                on,
                truth_by_frame,
            ),
        }
    return events


def event_window_overlap_summary(
    events: list[dict[str, Any]],
    before: int,
    after: int,
) -> dict[str, Any]:
    if not events:
        return {
            "events": 0,
            "overlapping_pairs": 0,
            "events_in_any_overlap": 0,
            "max_window_multiplicity": 0,
        }
    intervals = [
        (
            int(event["frame_id"]),
            int(event["frame_id"]) - max(0, int(before)),
            int(event["frame_id"]) + max(0, int(after)),
        )
        for event in events
    ]
    overlapping_pairs = 0
    overlapping_event_ids: set[int] = set()
    for i, (event_a, start_a, end_a) in enumerate(intervals):
        for event_b, start_b, end_b in intervals[i + 1 :]:
            if max(start_a, start_b) <= min(end_a, end_b):
                overlapping_pairs += 1
                overlapping_event_ids.update((event_a, event_b))

    multiplicity = Counter()
    for _, start, end in intervals:
        for frame_id in range(start, end + 1):
            multiplicity[frame_id] += 1

    return {
        "events": len(events),
        "window_before": max(0, int(before)),
        "window_after": max(0, int(after)),
        "overlapping_pairs": overlapping_pairs,
        "events_in_any_overlap": len(overlapping_event_ids),
        "max_window_multiplicity": max(multiplicity.values(), default=0),
        "frames_covered_by_multiple_event_windows": sum(
            1 for count in multiplicity.values() if count > 1
        ),
    }


def _classify_episode(effect: dict[str, Any]) -> str:
    if int(effect.get("truth_samples") or 0) <= 0:
        return "unlabeled"
    net = int(effect.get("net_v2_correct_delta") or 0)
    if net < 0:
        return "harmful"
    if net > 0:
        return "beneficial"
    direct_fp = int(effect.get("direct_not_visible_delta") or 0)
    if direct_fp > 0:
        return "harmful_false_positive"
    if direct_fp < 0:
        return "beneficial_false_positive"
    conditioned = int(effect.get("conditioned_correct_delta") or 0)
    if conditioned < 0:
        return "harmful_conditioned"
    if conditioned > 0:
        return "beneficial_conditioned"
    return "neutral"


def build_divergence_episodes(
    comparisons: list[dict[str, Any]],
    events: list[dict[str, Any]],
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    convergence_frames: int = 3,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    convergence_target = max(1, int(convergence_frames))
    by_frame = {int(record["frame_id"]): record for record in comparisons}
    frame_ids = sorted(by_frame)
    event_by_frame = {int(event["frame_id"]): event for event in events}
    episodes_ranges: list[tuple[int, int]] = []
    start: int | None = None
    last_material: int | None = None
    converged_run = 0

    for frame_id in frame_ids:
        material = bool(by_frame[frame_id].get("material_diverged"))
        if start is None:
            if material:
                start = frame_id
                last_material = frame_id
                converged_run = 0
            continue

        if material:
            last_material = frame_id
            converged_run = 0
        else:
            converged_run += 1
            if converged_run >= convergence_target:
                assert last_material is not None
                episodes_ranges.append((start, last_material))
                start = None
                last_material = None
                converged_run = 0

    if start is not None and last_material is not None:
        episodes_ranges.append((start, last_material))

    episodes: list[dict[str, Any]] = []
    frame_to_episode: dict[int, str] = {}
    all_event_frames = sorted(event_by_frame)
    last_frame = frame_ids[-1] if frame_ids else 0

    for index, (start_frame, end_frame) in enumerate(episodes_ranges, start=1):
        episode_id = f"E{index:03d}"
        interval_ids = list(range(start_frame, end_frame + 1))
        material_ids = [
            frame_id
            for frame_id in interval_ids
            if bool(by_frame.get(frame_id, {}).get("material_diverged"))
        ]
        events_inside = [
            event_by_frame[frame_id]
            for frame_id in all_event_frames
            if start_frame <= frame_id <= end_frame
        ]
        recent_events = [
            event_by_frame[frame_id]
            for frame_id in all_event_frames
            if start_frame - 2 <= frame_id < start_frame
        ]
        trigger = events_inside[0] if events_inside else (recent_events[-1] if recent_events else None)
        trigger_frame = None if trigger is None else int(trigger["frame_id"])
        effect = _effect_metrics(interval_ids, off, on, truth_by_frame)

        center_seps = [
            safe_float(by_frame.get(frame_id, {}).get("output_center_separation_px"))
            for frame_id in interval_ids
        ]
        center_seps = [value for value in center_seps if value is not None]
        state_center_seps = [
            safe_float(by_frame.get(frame_id, {}).get("pre_state_center_separation_px"))
            for frame_id in interval_ids
        ]
        state_center_seps = [value for value in state_center_seps if value is not None]
        velocity_seps = [
            safe_float(
                by_frame.get(frame_id, {}).get(
                    "pre_state_velocity_separation_px_per_frame"
                )
            )
            for frame_id in interval_ids
        ]
        velocity_seps = [value for value in velocity_seps if value is not None]

        first_timestamp = safe_float(on.get(start_frame, {}).get("timestamp_sec"))
        last_timestamp = safe_float(on.get(end_frame, {}).get("timestamp_sec"))
        duration_sec = (
            None
            if first_timestamp is None or last_timestamp is None
            else max(0.0, last_timestamp - first_timestamp)
        )

        end_position = frame_ids.index(end_frame) if end_frame in frame_ids else -1
        convergence_following = (
            end_position >= 0
            and end_position + convergence_target < len(frame_ids)
        )

        episode = {
            "episode_id": episode_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": end_frame - start_frame + 1,
            "duration_sec": base.rounded(duration_sec, 4),
            "materially_divergent_frames": len(material_ids),
            "interior_converged_gap_frames": (end_frame - start_frame + 1) - len(material_ids),
            "convergence_frames_required": convergence_target,
            "converged_before_run_end": bool(
                end_frame < last_frame and convergence_following
            ),
            "trigger_recovery_event_id": None if trigger is None else trigger.get("event_id"),
            "trigger_recovery_frame": trigger_frame,
            "trigger_lag_frames": (
                None if trigger_frame is None else start_frame - trigger_frame
            ),
            "recovery_event_count": len(events_inside),
            "recovery_event_ids": [event.get("event_id") for event in events_inside],
            "recovery_event_frames": [int(event["frame_id"]) for event in events_inside],
            "effect": _classify_episode(effect),
            "ground_truth_effect": effect,
            "separation": {
                "output_center_px": describe(center_seps),
                "state_center_px": describe(state_center_seps),
                "velocity_px_per_frame": describe(velocity_seps),
            },
            "start_comparison": by_frame.get(start_frame),
            "end_comparison": by_frame.get(end_frame),
            "trigger_features": None if trigger is None else {
                "selected_confidence": trigger.get("selected_confidence"),
                "selected_gate_ratio": trigger.get("selected_gate_ratio"),
                "selected_size_similarity": trigger.get("selected_size_similarity"),
                "selected_minus_best_true_confidence": trigger.get(
                    "selected_minus_best_true_confidence"
                ),
                "candidate_count": trigger.get("candidate_count"),
                "challenger_streak": trigger.get("production_challenger_streak"),
                "challenger_confidence": trigger.get("production_challenger_confidence"),
                "pre_track_state": (trigger.get("v2_pre_state") or {}).get("track_state"),
                "pre_last_confidence": (trigger.get("v2_pre_state") or {}).get(
                    "last_confidence"
                ),
                "previous_source": (trigger.get("previous_production") or {}).get("source"),
                "previous_track_state": (trigger.get("previous_production") or {}).get(
                    "track_state"
                ),
                "candidate_pool": trigger.get("candidate_pool"),
            },
        }
        episodes.append(episode)
        for frame_id in interval_ids:
            frame_to_episode[frame_id] = episode_id

    for event in events:
        frame_id = int(event["frame_id"])
        episode_id = frame_to_episode.get(frame_id)
        if episode_id is None:
            candidates = [
                episode["episode_id"]
                for episode in episodes
                if int(episode["start_frame"]) - 2 <= frame_id <= int(episode["end_frame"])
            ]
            episode_id = candidates[0] if candidates else None
        event["episode_id"] = episode_id

    return episodes, frame_to_episode


def _path_value(payload: dict[str, Any], path: str) -> float | None:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return safe_float(value)


def threshold_scan(
    episodes: list[dict[str, Any]],
    feature_path: str,
) -> dict[str, Any]:
    samples: list[tuple[float, bool]] = []
    for episode in episodes:
        effect = str(episode.get("effect"))
        if effect not in {"harmful", "beneficial"}:
            continue
        value = _path_value(episode, feature_path)
        if value is not None:
            samples.append((value, effect == "harmful"))

    harmful = sum(1 for _, label in samples if label)
    beneficial = sum(1 for _, label in samples if not label)
    result = {
        "feature": feature_path,
        "diagnostic_only": True,
        "samples": len(samples),
        "harmful_samples": harmful,
        "beneficial_samples": beneficial,
        "best_threshold": None,
        "harmful_direction": None,
        "balanced_accuracy_pct": None,
    }
    if harmful == 0 or beneficial == 0 or len({value for value, _ in samples}) < 2:
        return result

    unique = sorted({value for value, _ in samples})
    thresholds = [
        (a + b) / 2.0
        for a, b in zip(unique, unique[1:])
        if a != b
    ]
    best: tuple[float, float, str] | None = None
    for threshold in thresholds:
        for direction in ("gte", "lte"):
            tp = tn = fp = fn = 0
            for value, is_harmful in samples:
                predicts_harmful = value >= threshold if direction == "gte" else value <= threshold
                if is_harmful and predicts_harmful:
                    tp += 1
                elif is_harmful:
                    fn += 1
                elif predicts_harmful:
                    fp += 1
                else:
                    tn += 1
            tpr = tp / harmful
            tnr = tn / beneficial
            balanced = (tpr + tnr) / 2.0
            candidate = (balanced, threshold, direction)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is not None:
        result["best_threshold"] = base.rounded(best[1])
        result["harmful_direction"] = best[2]
        result["balanced_accuracy_pct"] = base.rounded(best[0] * 100.0, 2)
    return result


def aggregate_episodes(
    episodes: list[dict[str, Any]],
    metrics_off: dict[str, Any],
    metrics_on: dict[str, Any],
    total_production_frames: int,
) -> dict[str, Any]:
    effects = Counter(str(episode.get("effect")) for episode in episodes)
    durations = [int(episode.get("duration_frames") or 0) for episode in episodes]
    material_frames = sum(int(episode.get("materially_divergent_frames") or 0) for episode in episodes)
    covered_frames = sum(int(episode.get("duration_frames") or 0) for episode in episodes)

    net_correct = sum(
        int((episode.get("ground_truth_effect") or {}).get("net_v2_correct_delta") or 0)
        for episode in episodes
    )
    conditioned_delta = sum(
        int((episode.get("ground_truth_effect") or {}).get("conditioned_correct_delta") or 0)
        for episode in episodes
    )
    direct_not_visible_delta = sum(
        int((episode.get("ground_truth_effect") or {}).get("direct_not_visible_delta") or 0)
        for episode in episodes
    )

    global_correct_delta = (
        int(metrics_on["trajectory"]["correct_visible"])
        - int(metrics_off["trajectory"]["correct_visible"])
    )
    global_conditioned_delta = (
        int(metrics_on["association"]["correct_when_true_candidate_available"])
        - int(metrics_off["association"]["correct_when_true_candidate_available"])
    )
    global_direct_not_visible_delta = (
        int(metrics_on["association"]["direct_not_visible"])
        - int(metrics_off["association"]["direct_not_visible"])
    )

    harmful = [
        episode
        for episode in episodes
        if str(episode.get("effect")).startswith("harmful")
    ]
    beneficial = [
        episode
        for episode in episodes
        if str(episode.get("effect")).startswith("beneficial")
    ]

    def ranked(group: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
        ordered = sorted(
            group,
            key=lambda episode: int(
                (episode.get("ground_truth_effect") or {}).get("net_v2_correct_delta") or 0
            ),
            reverse=reverse,
        )
        return [
            {
                "episode_id": episode.get("episode_id"),
                "start_frame": episode.get("start_frame"),
                "end_frame": episode.get("end_frame"),
                "effect": episode.get("effect"),
                "net_v2_correct_delta": (
                    episode.get("ground_truth_effect") or {}
                ).get("net_v2_correct_delta"),
                "direct_not_visible_delta": (
                    episode.get("ground_truth_effect") or {}
                ).get("direct_not_visible_delta"),
                "trigger_recovery_frame": episode.get("trigger_recovery_frame"),
            }
            for episode in ordered[:10]
        ]

    diagnostics = [
        threshold_scan(episodes, "trigger_features.selected_confidence"),
        threshold_scan(episodes, "trigger_features.selected_gate_ratio"),
        threshold_scan(episodes, "trigger_features.selected_size_similarity"),
        threshold_scan(
            episodes,
            "trigger_features.selected_minus_best_true_confidence",
        ),
        threshold_scan(episodes, "trigger_features.pre_last_confidence"),
        threshold_scan(
            episodes,
            "trigger_features.candidate_pool.selected_margin_to_next_confidence",
        ),
        threshold_scan(
            episodes,
            "trigger_features.candidate_pool.selected_confidence_x_size_similarity",
        ),
    ]

    return {
        "episodes_total": len(episodes),
        "effect_counts": dict(effects),
        "frames_inside_episode_intervals": covered_frames,
        "materially_divergent_frames": material_frames,
        "material_divergence_pct_of_production": base.rounded(
            100.0 * material_frames / max(total_production_frames, 1),
            2,
        ),
        "duration_frames": describe(durations),
        "attribution": {
            "episode_net_visible_correct_delta_sum": net_correct,
            "global_visible_correct_delta": global_correct_delta,
            "visible_correct_residual": global_correct_delta - net_correct,
            "episode_conditioned_correct_delta_sum": conditioned_delta,
            "global_conditioned_correct_delta": global_conditioned_delta,
            "conditioned_correct_residual": global_conditioned_delta - conditioned_delta,
            "episode_direct_not_visible_delta_sum": direct_not_visible_delta,
            "global_direct_not_visible_delta": global_direct_not_visible_delta,
            "direct_not_visible_residual": (
                global_direct_not_visible_delta - direct_not_visible_delta
            ),
        },
        "top_harmful": ranked(harmful, reverse=False),
        "top_beneficial": ranked(beneficial, reverse=True),
        "trigger_feature_discrimination": diagnostics,
        "guardrail": (
            "Threshold scans are descriptive diagnostics on this run only. "
            "Do not promote them to production policy without multi-video validation."
        ),
    }


def flatten_event(event: dict[str, Any], horizons: list[int]) -> dict[str, Any]:
    row = base.flatten_event(event)
    pool = event.get("candidate_pool") or {}
    spacing = event.get("recovery_spacing") or {}
    isolated = event.get("isolated_until_next_recovery") or {}
    row.update(
        {
            "event_id": event.get("event_id"),
            "episode_id": event.get("episode_id"),
            "previous_recovery_frame": spacing.get("previous_recovery_frame"),
            "frames_since_previous_recovery": spacing.get("frames_since_previous_recovery"),
            "next_recovery_frame": spacing.get("next_recovery_frame"),
            "frames_to_next_recovery": spacing.get("frames_to_next_recovery"),
            "plausible_candidate_count": pool.get("plausible_count"),
            "outside_gate_candidate_count": pool.get("outside_gate_count"),
            "selected_confidence_rank": pool.get("selected_confidence_rank"),
            "selected_score_rank": pool.get("selected_score_rank"),
            "selected_margin_to_next_confidence": pool.get(
                "selected_margin_to_next_confidence"
            ),
            "selected_confidence_x_size_similarity": pool.get(
                "selected_confidence_x_size_similarity"
            ),
            "selected_distance_excess_norm": pool.get("selected_distance_excess_norm"),
            "selected_plausible": pool.get("selected_plausible"),
            "selected_high_confidence_gate_used": pool.get(
                "selected_high_confidence_gate_used"
            ),
            "isolated_net_v2_correct_delta": isolated.get("net_v2_correct_delta"),
            "isolated_conditioned_correct_delta": isolated.get(
                "conditioned_correct_delta"
            ),
            "isolated_direct_not_visible_delta": isolated.get(
                "direct_not_visible_delta"
            ),
        }
    )
    for horizon in horizons:
        metrics = (event.get("downstream_horizons") or {}).get(str(horizon)) or {}
        prefix = f"h{horizon}_"
        row[prefix + "visible_truth_samples"] = metrics.get("visible_truth_samples")
        row[prefix + "net_v2_correct_delta"] = metrics.get("net_v2_correct_delta")
        row[prefix + "conditioned_correct_delta"] = metrics.get(
            "conditioned_correct_delta"
        )
        row[prefix + "direct_not_visible_delta"] = metrics.get(
            "direct_not_visible_delta"
        )
        row[prefix + "crosses_next_recovery"] = metrics.get("crosses_next_recovery")
    return row


def flatten_episode(episode: dict[str, Any]) -> dict[str, Any]:
    effect = episode.get("ground_truth_effect") or {}
    separation = episode.get("separation") or {}
    trigger = episode.get("trigger_features") or {}
    pool = trigger.get("candidate_pool") or {}
    return {
        "episode_id": episode.get("episode_id"),
        "effect": episode.get("effect"),
        "start_frame": episode.get("start_frame"),
        "end_frame": episode.get("end_frame"),
        "duration_frames": episode.get("duration_frames"),
        "duration_sec": episode.get("duration_sec"),
        "materially_divergent_frames": episode.get("materially_divergent_frames"),
        "interior_converged_gap_frames": episode.get("interior_converged_gap_frames"),
        "trigger_recovery_event_id": episode.get("trigger_recovery_event_id"),
        "trigger_recovery_frame": episode.get("trigger_recovery_frame"),
        "trigger_lag_frames": episode.get("trigger_lag_frames"),
        "recovery_event_count": episode.get("recovery_event_count"),
        "truth_samples": effect.get("truth_samples"),
        "visible_truth_samples": effect.get("visible_truth_samples"),
        "off_correct_visible": effect.get("off_correct_visible"),
        "v2_correct_visible": effect.get("v2_correct_visible"),
        "net_v2_correct_delta": effect.get("net_v2_correct_delta"),
        "conditioned_correct_delta": effect.get("conditioned_correct_delta"),
        "direct_not_visible_delta": effect.get("direct_not_visible_delta"),
        "output_not_visible_delta": effect.get("output_not_visible_delta"),
        "off_better_visible": effect.get("off_better_visible"),
        "v2_better_visible": effect.get("v2_better_visible"),
        "first_off_better_frame": effect.get("first_off_better_frame"),
        "last_off_better_frame": effect.get("last_off_better_frame"),
        "first_v2_better_frame": effect.get("first_v2_better_frame"),
        "last_v2_better_frame": effect.get("last_v2_better_frame"),
        "output_center_separation_p90_px": (
            separation.get("output_center_px") or {}
        ).get("p90"),
        "output_center_separation_max_px": (
            separation.get("output_center_px") or {}
        ).get("max"),
        "state_center_separation_max_px": (
            separation.get("state_center_px") or {}
        ).get("max"),
        "velocity_separation_max_px_per_frame": (
            separation.get("velocity_px_per_frame") or {}
        ).get("max"),
        "trigger_confidence": trigger.get("selected_confidence"),
        "trigger_gate_ratio": trigger.get("selected_gate_ratio"),
        "trigger_size_similarity": trigger.get("selected_size_similarity"),
        "trigger_selected_minus_best_true_confidence": trigger.get(
            "selected_minus_best_true_confidence"
        ),
        "trigger_candidate_count": trigger.get("candidate_count"),
        "trigger_pre_track_state": trigger.get("pre_track_state"),
        "trigger_pre_last_confidence": trigger.get("pre_last_confidence"),
        "trigger_selected_margin_to_next_confidence": pool.get(
            "selected_margin_to_next_confidence"
        ),
        "trigger_selected_confidence_x_size_similarity": pool.get(
            "selected_confidence_x_size_similarity"
        ),
    }


def flatten_comparison(record: dict[str, Any], episode_id: str | None) -> dict[str, Any]:
    return {
        "frame_id": record.get("frame_id"),
        "timestamp_sec": record.get("timestamp_sec"),
        "episode_id": episode_id,
        "material_diverged": record.get("material_diverged"),
        "output_diverged": record.get("output_diverged"),
        "decision_diverged": record.get("decision_diverged"),
        "state_core_diverged": record.get("state_core_diverged"),
        "diagnostic_state_diverged": record.get("diagnostic_state_diverged"),
        "truth_available": record.get("truth_available"),
        "truth_visible": record.get("truth_visible"),
        "truth_effect": record.get("truth_effect"),
        "off_source": record.get("off_source"),
        "v2_source": record.get("v2_source"),
        "off_track_state": record.get("off_track_state"),
        "v2_track_state": record.get("v2_track_state"),
        "off_selected_index": record.get("off_selected_index"),
        "v2_selected_index": record.get("v2_selected_index"),
        "off_reacquired": record.get("off_reacquired"),
        "v2_reacquired": record.get("v2_reacquired"),
        "output_center_separation_px": record.get("output_center_separation_px"),
        "pre_state_center_separation_px": record.get(
            "pre_state_center_separation_px"
        ),
        "pre_state_velocity_separation_px_per_frame": record.get(
            "pre_state_velocity_separation_px_per_frame"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_visual_frame_ids(
    episode: dict[str, Any],
    comparisons_by_frame: dict[int, dict[str, Any]],
    max_panels: int,
) -> list[int]:
    start = int(episode["start_frame"])
    end = int(episode["end_frame"])
    important: set[int] = {start, end}
    trigger = safe_int(episode.get("trigger_recovery_frame"))
    if trigger is not None and start <= trigger <= end:
        important.add(trigger)

    for frame_id in range(start, end + 1):
        effect = str(comparisons_by_frame.get(frame_id, {}).get("truth_effect"))
        if effect in {"off_better", "v2_better"}:
            important.add(frame_id)

    separations = sorted(
        (
            (
                safe_float(comparisons_by_frame.get(frame_id, {}).get(
                    "output_center_separation_px"
                ))
                or 0.0,
                frame_id,
            )
            for frame_id in range(start, end + 1)
        ),
        reverse=True,
    )
    for _, frame_id in separations[:3]:
        important.add(frame_id)

    max_panels = max(4, int(max_panels))
    if len(important) < max_panels and end > start:
        needed = max_panels - len(important)
        samples = np.linspace(start, end, num=max(needed + 2, 2), dtype=int)
        important.update(int(value) for value in samples)

    ordered = sorted(frame_id for frame_id in important if start <= frame_id <= end)
    if len(ordered) <= max_panels:
        return ordered

    mandatory = {start, end}
    if trigger is not None:
        mandatory.add(trigger)
    ranked = [
        frame_id
        for _, frame_id in separations
        if frame_id not in mandatory
    ]
    result = list(sorted(mandatory))
    for frame_id in ranked:
        if len(result) >= max_panels:
            break
        if frame_id not in result:
            result.append(frame_id)
    return sorted(result[:max_panels])


def draw_episode_contact_sheet(
    video_path: Path,
    episode: dict[str, Any],
    comparisons_by_frame: dict[int, dict[str, Any]],
    production_by_frame: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    output_path: Path,
    frame_offset: int,
    gt_radius_px: float,
    bbox_iou: float,
    max_panels: int,
) -> None:
    frame_ids = _episode_visual_frame_ids(episode, comparisons_by_frame, max_panels)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    panels: list[np.ndarray] = []

    for frame_id in frame_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id + int(frame_offset))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        production = production_by_frame.get(frame_id, {})
        assoc = base.association(production)
        truth = truth_by_frame.get(frame_id)
        for candidate in assoc.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            box = candidate.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            is_true = base.item_matches_truth(
                candidate,
                truth,
                float(gt_radius_px),
                float(bbox_iou),
            )[0]
            color = (0, 150, 0) if is_true else (150, 150, 150)
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        point = truth.get("point") if isinstance(truth, dict) else None
        if isinstance(point, list) and len(point) == 2:
            cv2.circle(
                frame,
                (int(round(float(point[0]))), int(round(float(point[1])))),
                max(4, int(round(float(gt_radius_px) / 5.0))),
                (255, 0, 255),
                2,
            )

        comparison = comparisons_by_frame.get(frame_id, {})
        off_center = vector(comparison.get("off_output_center"))
        v2_center = vector(comparison.get("v2_output_center"))
        if off_center is not None:
            cv2.circle(
                frame,
                (int(round(off_center[0])), int(round(off_center[1]))),
                8,
                (255, 120, 0),
                3,
            )
        if v2_center is not None:
            cv2.circle(
                frame,
                (int(round(v2_center[0])), int(round(v2_center[1]))),
                8,
                (0, 0, 255),
                3,
            )

        label = (
            f"{episode['episode_id']} f={frame_id} "
            f"effect={comparison.get('truth_effect', 'unlabeled')}"
        )
        cv2.putText(
            frame,
            label,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"OFF={comparison.get('off_source')} V2={comparison.get('v2_source')}",
            (16, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(cv2.resize(frame, (480, 270)))

    capture.release()
    if not panels:
        return
    columns = min(4, len(panels))
    row_count = int(math.ceil(len(panels) / columns))
    blank = np.zeros_like(panels[0])
    while len(panels) < row_count * columns:
        panels.append(blank.copy())
    sheet = np.vstack(
        [
            np.hstack(panels[row * columns : (row + 1) * columns])
            for row in range(row_count)
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def build_report(
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    top_n: int,
) -> str:
    cf = summary.get("counterfactual") or {}
    delta = cf.get("on_minus_off") or {}
    agreement = cf.get("replay_vs_production_agreement") or {}
    episode_summary = summary.get("divergence_episodes") or {}
    attribution = episode_summary.get("attribution") or {}
    event_summary = summary.get("production_reacquisition_events") or {}

    lines = [
        "# P0.2 Recovery Error Mining Report",
        "",
        "This report compares the exact persisted production candidate stream under "
        "reacquisition OFF and the current recovery policy. Sparse ground truth is used "
        "only where annotations exist.",
        "",
        "## Replay fidelity",
        "",
        f"- Comparable production frames: {agreement.get('comparable_frames')}",
        f"- Selected-index agreement with production: {agreement.get('selected_index_match_pct')}%",
        f"- Source agreement with production: {agreement.get('source_match_pct')}%",
        "",
        "## Global counterfactual delta (ON - OFF)",
        "",
        f"- Association accuracy conditioned: {delta.get('association_accuracy_conditioned_pp')} pp",
        f"- Direct correct visible: {delta.get('direct_correct_visible')}",
        f"- Direct precision visible: {delta.get('direct_precision_visible_pp')} pp",
        f"- Direct detections while ball not visible: {delta.get('direct_not_visible')}",
        f"- Gate loss: {delta.get('gate_loss')}",
        f"- Trajectory correct recall: {delta.get('trajectory_correct_recall_pp')} pp",
        "",
        "## Production reacquisition events",
        "",
        f"- Total events: {event_summary.get('events_total_production')}",
        f"- Events with sparse truth: {event_summary.get('events_with_sparse_truth')}",
        f"- Events without sparse truth: {event_summary.get('events_without_sparse_truth')}",
        f"- Instantaneous labeled precision: {event_summary.get('precision_pct_on_labeled_events')}%",
        "",
        "## Non-overlapping divergence episodes",
        "",
        f"- Episodes: {episode_summary.get('episodes_total')}",
        f"- Effect counts: `{json.dumps(episode_summary.get('effect_counts') or {}, ensure_ascii=False)}`",
        f"- Materially divergent frames: {episode_summary.get('materially_divergent_frames')}",
        f"- Material divergence share: {episode_summary.get('material_divergence_pct_of_production')}%",
        f"- Sum of episode visible-correct deltas: {attribution.get('episode_net_visible_correct_delta_sum')}",
        f"- Global visible-correct delta: {attribution.get('global_visible_correct_delta')}",
        f"- Attribution residual: {attribution.get('visible_correct_residual')}",
        "",
        "Episodes are non-overlapping, so their downstream effects are safer to interpret "
        "than overlapping per-event windows.",
        "",
    ]

    harmful = sorted(
        [
            episode
            for episode in episodes
            if str(episode.get("effect")).startswith("harmful")
        ],
        key=lambda episode: int(
            (episode.get("ground_truth_effect") or {}).get("net_v2_correct_delta") or 0
        ),
    )[: max(0, int(top_n))]
    beneficial = sorted(
        [
            episode
            for episode in episodes
            if str(episode.get("effect")).startswith("beneficial")
        ],
        key=lambda episode: int(
            (episode.get("ground_truth_effect") or {}).get("net_v2_correct_delta") or 0
        ),
        reverse=True,
    )[: max(0, int(top_n))]

    def episode_table(title: str, group: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Episode | Frames | Effect | Net correct | Direct invisible Δ | Trigger | Conf | Gate ratio | Size sim |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not group:
            lines.append("| — | — | none | — | — | — | — | — | — |")
        for episode in group:
            effect = episode.get("ground_truth_effect") or {}
            trigger = episode.get("trigger_features") or {}
            lines.append(
                "| {id} | {start}-{end} | {effect_name} | {net} | {fp} | {trigger_frame} | {conf} | {gate} | {size} |".format(
                    id=episode.get("episode_id"),
                    start=episode.get("start_frame"),
                    end=episode.get("end_frame"),
                    effect_name=episode.get("effect"),
                    net=effect.get("net_v2_correct_delta"),
                    fp=effect.get("direct_not_visible_delta"),
                    trigger_frame=episode.get("trigger_recovery_frame"),
                    conf=trigger.get("selected_confidence"),
                    gate=trigger.get("selected_gate_ratio"),
                    size=trigger.get("selected_size_similarity"),
                )
            )
        lines.append("")

    episode_table("Most harmful episodes", harmful)
    episode_table("Most beneficial episodes", beneficial)

    lines.extend(
        [
            "## Exploratory trigger-feature diagnostics",
            "",
            "These scans are descriptive only and are intentionally **not** production "
            "threshold recommendations. The sample is one video with sparse GT.",
            "",
            "| Feature | Samples | Harmful | Beneficial | Best threshold | Harmful direction | Balanced acc. |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for diagnostic in episode_summary.get("trigger_feature_discrimination") or []:
        lines.append(
            "| {feature} | {samples} | {harmful} | {beneficial} | {threshold} | {direction} | {accuracy}% |".format(
                feature=diagnostic.get("feature"),
                samples=diagnostic.get("samples"),
                harmful=diagnostic.get("harmful_samples"),
                beneficial=diagnostic.get("beneficial_samples"),
                threshold=diagnostic.get("best_threshold"),
                direction=diagnostic.get("harmful_direction"),
                accuracy=diagnostic.get("balanced_accuracy_pct"),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Instantaneous recovery correctness is not equivalent to downstream trajectory quality.",
            "- Event windows may overlap; use divergence episodes for causal downstream attribution.",
            "- Events without sparse truth must not be labeled correct or wrong.",
            "- Single-video feature thresholds are diagnostic hypotheses only.",
            "- Keep detector candidate threshold fixed while comparing recovery policies.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    frames_path = Path(args.frames).expanduser().resolve()
    truth_path = Path(args.truth).expanduser().resolve()
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else frames_path.parent / "recovery_error_mining"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = parse_horizons(args.horizons)
    rows = base.load_jsonl(frames_path)
    truth_rows = base.load_truth(truth_path)
    frame_shape = base.resolve_frame_shape(args)
    production_by_frame = {int(row.get("frame_id", 0)): row for row in rows}
    truth_by_frame = {int(row.get("frame_id", 0)): row for row in truth_rows}
    if not any("candidates" in base.association(row) for row in rows):
        raise ValueError("No persisted P0.2 candidate traces found; use an instrumented run")

    config = base.persisted_config(rows)
    off, metrics_off = base.replay(
        rows,
        truth_by_frame,
        frame_shape,
        config,
        False,
        args.gt_radius_px,
        args.bbox_iou,
    )
    on, metrics_on = base.replay(
        rows,
        truth_by_frame,
        frame_shape,
        config,
        True,
        args.gt_radius_px,
        args.bbox_iou,
    )
    enrich_replay_records(rows, off, on, frame_shape)

    events = [
        base.event_detail(
            frame_id,
            production_by_frame,
            truth_by_frame,
            off,
            on,
            max(0, args.window_before),
            max(0, args.window_after),
            args.gt_radius_px,
            args.bbox_iou,
        )
        for frame_id in base.event_frames(rows)
    ]
    enrich_events(
        events,
        production_by_frame,
        off,
        on,
        truth_by_frame,
        horizons,
    )

    comparisons = compare_replays(
        off,
        on,
        truth_by_frame,
        center_tolerance_px=max(0.0, args.divergence_center_tolerance_px),
    )
    comparisons_by_frame = {
        int(record["frame_id"]): record for record in comparisons
    }
    episodes, frame_to_episode = build_divergence_episodes(
        comparisons,
        events,
        off,
        on,
        truth_by_frame,
        convergence_frames=max(1, args.episode_convergence_frames),
    )

    probe = base.make_tracker(config, True)
    event_summary = base.aggregate_events(events)
    event_summary["window_overlap"] = event_window_overlap_summary(
        events,
        max(0, args.window_before),
        max(0, args.window_after),
    )
    episode_summary = aggregate_episodes(
        episodes,
        metrics_off,
        metrics_on,
        total_production_frames=len(rows),
    )

    summary = {
        "scope": "P0.2 recovery error mining, counterfactual replay, and divergence episodes",
        "frames_file": str(frames_path),
        "truth_file": str(truth_path),
        "video_file": (
            None
            if not args.video
            else str(Path(args.video).expanduser().resolve())
        ),
        "frame_shape": list(frame_shape),
        "window_before": max(0, args.window_before),
        "window_after": max(0, args.window_after),
        "downstream_horizons": horizons,
        "episode_convergence_frames": max(1, args.episode_convergence_frames),
        "divergence_center_tolerance_px": max(
            0.0,
            float(args.divergence_center_tolerance_px),
        ),
        "gt_radius_px": float(args.gt_radius_px),
        "bbox_iou": float(args.bbox_iou),
        "production_rows": len(rows),
        "truth_frames": len(truth_rows),
        "tracker_policy": getattr(probe, "reacquire_policy", None),
        "production_reacquisition_events": event_summary,
        "divergence_episodes": episode_summary,
        "counterfactual": {
            "reacquire_off": metrics_off,
            "reacquire_on": metrics_on,
            "on_minus_off": base.metric_delta(metrics_off, metrics_on),
            "replay_vs_production_agreement": base.replay_agreement(rows, on),
        },
        "methodology": {
            "material_divergence": (
                "Episode membership starts when OFF and ON differ in output, selected "
                "decision/reacquisition, or core direct-observation state (last center, "
                "last frame, velocity). Track-state/challenger-only differences are "
                "recorded diagnostically but do not alone start an episode."
            ),
            "episode_closure": (
                f"{max(1, args.episode_convergence_frames)} consecutive materially "
                "converged frames close an episode; short convergence gaps remain inside "
                "the same non-overlapping episode."
            ),
            "sparse_truth": (
                "Correctness deltas use only annotated truth frames. Unannotated recovery "
                "events remain unlabeled."
            ),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "events.csv",
        [flatten_event(event, horizons) for event in events],
    )
    (output_dir / "episodes.json").write_text(
        json.dumps(episodes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "episodes.csv",
        [flatten_episode(episode) for episode in episodes],
    )
    write_jsonl(
        output_dir / "divergence_frames.jsonl",
        (
            {
                **record,
                "episode_id": frame_to_episode.get(int(record["frame_id"])),
            }
            for record in comparisons
            if bool(record.get("material_diverged"))
            or bool(record.get("diagnostic_state_diverged"))
        ),
    )
    write_csv(
        output_dir / "divergence_frames.csv",
        [
            flatten_comparison(
                record,
                frame_to_episode.get(int(record["frame_id"])),
            )
            for record in comparisons
            if bool(record.get("material_diverged"))
            or bool(record.get("diagnostic_state_diverged"))
        ],
    )

    replay_truth = [
        {
            "frame_id": frame_id,
            "visible": bool(truth_by_frame[frame_id].get("visible")),
            "episode_id": frame_to_episode.get(frame_id),
            "comparison": comparisons_by_frame.get(frame_id),
            "off": off.get(frame_id),
            "v2": on.get(frame_id),
        }
        for frame_id in sorted(truth_by_frame)
    ]
    (output_dir / "replay_frames.json").write_text(
        json.dumps(replay_truth, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    diagnostics = {
        "trigger_feature_discrimination": episode_summary.get(
            "trigger_feature_discrimination"
        ),
        "event_window_overlap": event_summary.get("window_overlap"),
        "episode_attribution": episode_summary.get("attribution"),
        "guardrail": episode_summary.get("guardrail"),
    }
    (output_dir / "feature_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary, episodes, max(0, args.top_episodes)),
        encoding="utf-8",
    )

    if args.video and not args.no_visuals:
        video_path = Path(args.video).expanduser().resolve()
        for event in events:
            if not event.get("truth_available"):
                continue
            correctness = event.get("production_reacquisition_correct")
            status = "correct" if correctness is True else "wrong"
            base.draw_contact_sheet(
                video_path,
                event,
                production_by_frame,
                truth_by_frame,
                output_dir
                / "visual"
                / "events"
                / f"{event.get('event_id')}_frame_{int(event['frame_id']):06d}_{status}.jpg",
                int(args.video_frame_offset),
                float(args.gt_radius_px),
                float(args.bbox_iou),
            )

        for episode in episodes:
            draw_episode_contact_sheet(
                video_path,
                episode,
                comparisons_by_frame,
                production_by_frame,
                truth_by_frame,
                output_dir
                / "visual"
                / "episodes"
                / f"{episode['episode_id']}_{episode['effect']}_{int(episode['start_frame']):06d}_{int(episode['end_frame']):06d}.jpg",
                int(args.video_frame_offset),
                float(args.gt_radius_px),
                float(args.bbox_iou),
                max(4, args.episode_visual_panels),
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OUTPUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
