from __future__ import annotations

import csv
import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import ball_recovery_error_mining_base as base
from evaluation import ball_recovery_error_mining_extended as extended


def _counterfactual_label(
    truth: dict[str, Any] | None,
    off_record: dict[str, Any] | None,
    on_record: dict[str, Any] | None,
) -> str:
    if truth is None:
        return "unlabeled_event"
    if not bool(truth.get("visible")):
        return "ball_not_visible"
    off_correct = bool((off_record or {}).get("selected_correct"))
    on_correct = bool((on_record or {}).get("selected_correct"))
    if on_correct and not off_correct:
        return "recovery_win"
    if off_correct and not on_correct:
        return "recovery_loss"
    if off_correct and on_correct:
        return "neutral_both_correct"
    return "neutral_both_wrong"


def _restore_score_probe(
    config: dict[str, Any],
    pre_state: dict[str, Any],
):
    tracker = base.make_tracker(config, True)
    center = pre_state.get("last_center")
    tracker.last_center = (
        None
        if not isinstance(center, (list, tuple)) or len(center) < 2
        else np.asarray([float(center[0]), float(center[1])], dtype=np.float64)
    )
    tracker.last_frame_id = extended.safe_int(pre_state.get("last_frame_id"))
    tracker.last_confidence = extended.safe_float(pre_state.get("last_confidence"))
    tracker.last_area = extended.safe_float(pre_state.get("last_area"))
    velocity = pre_state.get("velocity")
    if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
        tracker.velocity = np.asarray(
            [float(velocity[0]), float(velocity[1])],
            dtype=np.float64,
        )
    tracker.track_state = str(pre_state.get("track_state") or tracker.track_state)
    tracker._challenger_streak = int(pre_state.get("challenger_streak") or 0)
    tracker._challenger_confidence = extended.safe_float(
        pre_state.get("challenger_confidence")
    )
    return tracker


def _replay_candidate_pool(
    row: dict[str, Any],
    record: dict[str, Any],
    config: dict[str, Any],
    frame_shape: tuple[int, int, int],
) -> dict[str, Any]:
    candidates = base.persisted_candidates(row)
    frame_id = int(record.get("frame_id", 0))
    probe = _restore_score_probe(config, record.get("pre_state") or {})
    scores = [probe.inspect_candidate(candidate, frame_id, frame_shape) for candidate in candidates]
    selected_index = extended.safe_int(record.get("selected_index"))

    confidence_values = sorted(
        [
            (index, float(candidate.get("confidence", 0.0)))
            for index, candidate in enumerate(candidates)
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    score_values = sorted(
        [
            (index, float(score.score))
            for index, score in enumerate(scores)
            if score is not None
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    selected_candidate = (
        candidates[selected_index]
        if selected_index is not None and 0 <= selected_index < len(candidates)
        else None
    )
    selected_score = (
        scores[selected_index]
        if selected_index is not None and 0 <= selected_index < len(scores)
        else None
    )
    selected_confidence = (
        None
        if selected_candidate is None
        else float(selected_candidate.get("confidence", 0.0))
    )
    next_best_confidence = max(
        [
            value
            for index, value in confidence_values
            if selected_index is None or index != selected_index
        ],
        default=None,
    )
    confidence_rank = (
        None
        if selected_index is None
        else next(
            (
                rank
                for rank, (index, _) in enumerate(confidence_values, start=1)
                if index == selected_index
            ),
            None,
        )
    )
    score_rank = (
        None
        if selected_index is None
        else next(
            (
                rank
                for rank, (index, _) in enumerate(score_values, start=1)
                if index == selected_index
            ),
            None,
        )
    )
    plausible_count = sum(1 for score in scores if score is not None and score.plausible)
    outside_count = sum(1 for score in scores if score is not None and not score.plausible)
    gate_ratio = None
    if selected_score is not None and selected_score.gate_norm > 0:
        gate_ratio = selected_score.distance_norm / selected_score.gate_norm

    return {
        "candidate_count": len(candidates),
        "confidence": extended.describe(value for _, value in confidence_values),
        "plausible_count": plausible_count,
        "outside_gate_count": outside_count,
        "selected_confidence_rank": confidence_rank,
        "selected_score_rank": score_rank,
        "selected_margin_to_next_confidence": (
            None
            if selected_confidence is None or next_best_confidence is None
            else base.rounded(selected_confidence - next_best_confidence)
        ),
        "selected_confidence_x_size_similarity": (
            None
            if selected_confidence is None or selected_score is None
            else base.rounded(selected_confidence * selected_score.size_similarity)
        ),
        "selected_gate_ratio": base.rounded(gate_ratio),
        "selected_distance_excess_norm": (
            None
            if selected_score is None
            else base.rounded(selected_score.distance_norm - selected_score.gate_norm)
        ),
        "selected_plausible": (
            None if selected_score is None else bool(selected_score.plausible)
        ),
        "selected_high_confidence_gate_used": (
            None
            if selected_score is None
            else bool(selected_score.high_confidence_gate_used)
        ),
        "selected_size_similarity": (
            None if selected_score is None else base.rounded(selected_score.size_similarity)
        ),
        "selected_confidence": selected_confidence,
        "true_candidate_count": len(record.get("true_candidate_indices") or []),
    }


def _enrich_replay_event_windows(
    events: list[dict[str, Any]],
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    horizons: list[int],
) -> None:
    ordered = sorted(events, key=lambda item: int(item["frame_id"]))
    event_frames = [int(event["frame_id"]) for event in ordered]
    max_frame = max(set(off) | set(on)) if off or on else 0
    for position, event in enumerate(ordered):
        frame_id = int(event["frame_id"])
        previous_frame = event_frames[position - 1] if position > 0 else None
        next_frame = event_frames[position + 1] if position + 1 < len(event_frames) else None
        event["recovery_spacing"] = {
            "previous_recovery_frame": previous_frame,
            "frames_since_previous_recovery": (
                None if previous_frame is None else frame_id - previous_frame
            ),
            "next_recovery_frame": next_frame,
            "frames_to_next_recovery": (
                None if next_frame is None else next_frame - frame_id
            ),
        }
        downstream: dict[str, Any] = {}
        for horizon in horizons:
            end = min(frame_id + int(horizon), max_frame)
            metrics = extended._effect_metrics(
                range(frame_id + 1, end + 1),
                off,
                on,
                truth_by_frame,
            )
            metrics["production_frames_requested"] = int(horizon)
            metrics["production_frames_available"] = max(0, end - frame_id)
            metrics["crosses_next_replay_recovery"] = bool(
                next_frame is not None and next_frame <= end
            )
            downstream[str(horizon)] = metrics
        event["downstream_horizons"] = downstream

        isolated_end = min(next_frame - 1, max_frame) if next_frame is not None else max_frame
        event["isolated_until_next_replay_recovery"] = {
            "start_frame": frame_id + 1,
            "end_frame": isolated_end if isolated_end >= frame_id + 1 else None,
            **extended._effect_metrics(
                range(frame_id + 1, isolated_end + 1)
                if isolated_end >= frame_id + 1
                else [],
                off,
                on,
                truth_by_frame,
            ),
        }


def extract_replay_recovery_events(
    rows: list[dict[str, Any]],
    truth_by_frame: dict[int, dict[str, Any]],
    off: dict[int, dict[str, Any]],
    on: dict[int, dict[str, Any]],
    config: dict[str, Any],
    frame_shape: tuple[int, int, int],
    horizons: list[int],
    policy: str,
) -> list[dict[str, Any]]:
    production_by_frame = {int(row.get("frame_id", 0)): row for row in rows}
    events: list[dict[str, Any]] = []
    for frame_id in sorted(on):
        record = on[frame_id]
        if not bool(record.get("reacquired")):
            continue
        row = production_by_frame.get(frame_id, {})
        truth = truth_by_frame.get(frame_id)
        off_record = off.get(frame_id)
        pool = _replay_candidate_pool(row, record, config, frame_shape)
        assoc = base.association(row)
        event = {
            "event_id": f"RR{len(events) + 1:03d}",
            "frame_id": frame_id,
            "timestamp_sec": record.get("timestamp_sec"),
            "policy": policy,
            "truth_available": truth is not None,
            "truth_visible": None if truth is None else bool(truth.get("visible")),
            "selected_correct": None if truth is None else bool(record.get("selected_correct")),
            "counterfactual_label": _counterfactual_label(truth, off_record, record),
            "source": record.get("source"),
            "track_state": record.get("track_state"),
            "selected_index": record.get("selected_index"),
            "selected_score": record.get("selected_score"),
            "selected_confidence": pool.get("selected_confidence"),
            "selected_gate_ratio": pool.get("selected_gate_ratio"),
            "selected_size_similarity": pool.get("selected_size_similarity"),
            "pre_state": record.get("pre_state"),
            "candidate_pool": pool,
            "output_center": record.get("output_center"),
            "production_source_same_frame": assoc.get("source"),
            "production_reacquired_same_frame": bool(assoc.get("reacquired")),
            "off_at_event": off_record,
            "on_at_event": record,
            "episode_id": None,
        }
        events.append(event)

    _enrich_replay_event_windows(events, off, on, truth_by_frame, horizons)
    return events


def aggregate_replay_recovery_events(
    events: list[dict[str, Any]],
    before: int,
    after: int,
    policy: str,
) -> dict[str, Any]:
    labels = Counter(str(event.get("counterfactual_label")) for event in events)
    labeled_visible = [
        event
        for event in events
        if bool(event.get("truth_available")) and bool(event.get("truth_visible"))
    ]
    correct = sum(bool(event.get("selected_correct")) for event in labeled_visible)
    wrong = len(labeled_visible) - correct

    def feature_mean(group: list[dict[str, Any]], key: str) -> float | None:
        values = [extended.safe_float(event.get(key)) for event in group]
        values = [value for value in values if value is not None]
        return None if not values else base.rounded(float(np.mean(values)))

    correct_events = [event for event in labeled_visible if bool(event.get("selected_correct"))]
    wrong_events = [event for event in labeled_visible if not bool(event.get("selected_correct"))]
    return {
        "policy": policy,
        "events_total_replay": len(events),
        "event_frames": [int(event["frame_id"]) for event in events],
        "events_with_sparse_truth": sum(bool(event.get("truth_available")) for event in events),
        "events_without_sparse_truth": sum(not bool(event.get("truth_available")) for event in events),
        "visible_labeled_events": len(labeled_visible),
        "correct_on_visible_labeled_events": correct,
        "wrong_on_visible_labeled_events": wrong,
        "precision_pct_on_visible_labeled_events": base.pct(correct, len(labeled_visible)),
        "counterfactual_labels": dict(labels),
        "feature_means": {
            "correct": {
                "confidence": feature_mean(correct_events, "selected_confidence"),
                "gate_ratio": feature_mean(correct_events, "selected_gate_ratio"),
                "size_similarity": feature_mean(correct_events, "selected_size_similarity"),
            },
            "wrong": {
                "confidence": feature_mean(wrong_events, "selected_confidence"),
                "gate_ratio": feature_mean(wrong_events, "selected_gate_ratio"),
                "size_similarity": feature_mean(wrong_events, "selected_size_similarity"),
            },
        },
        "window_overlap": extended.event_window_overlap_summary(events, before, after),
    }


def _trigger_features(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if event is None:
        return None
    pre = event.get("pre_state") or {}
    pool = event.get("candidate_pool") or {}
    return {
        "selected_confidence": event.get("selected_confidence"),
        "selected_gate_ratio": event.get("selected_gate_ratio"),
        "selected_size_similarity": event.get("selected_size_similarity"),
        "selected_minus_best_true_confidence": None,
        "candidate_count": pool.get("candidate_count"),
        "challenger_streak": pre.get("challenger_streak"),
        "challenger_confidence": pre.get("challenger_confidence"),
        "pre_track_state": pre.get("track_state"),
        "pre_last_confidence": pre.get("last_confidence"),
        "previous_source": None,
        "previous_track_state": None,
        "candidate_pool": pool,
    }


def attach_replay_events_to_episodes(
    episodes: list[dict[str, Any]],
    replay_events: list[dict[str, Any]],
) -> None:
    ordered = sorted(replay_events, key=lambda item: int(item["frame_id"]))
    for episode in episodes:
        start = int(episode["start_frame"])
        end = int(episode["end_frame"])

        episode["production_trigger_recovery_event_id"] = episode.get(
            "trigger_recovery_event_id"
        )
        episode["production_trigger_recovery_frame"] = episode.get(
            "trigger_recovery_frame"
        )
        episode["production_trigger_lag_frames"] = episode.get("trigger_lag_frames")
        episode["production_recovery_event_count"] = episode.get("recovery_event_count", 0)
        episode["production_recovery_event_ids"] = list(
            episode.get("recovery_event_ids") or []
        )
        episode["production_recovery_event_frames"] = list(
            episode.get("recovery_event_frames") or []
        )

        inside = [
            event for event in ordered if start <= int(event["frame_id"]) <= end
        ]
        recent = [
            event for event in ordered if start - 2 <= int(event["frame_id"]) < start
        ]
        trigger = inside[0] if inside else (recent[-1] if recent else None)
        trigger_frame = None if trigger is None else int(trigger["frame_id"])

        episode["replay_recovery_event_count"] = len(inside)
        episode["replay_recovery_event_ids"] = [event["event_id"] for event in inside]
        episode["replay_recovery_event_frames"] = [int(event["frame_id"]) for event in inside]
        episode["trigger_replay_recovery_event_id"] = (
            None if trigger is None else trigger.get("event_id")
        )
        episode["trigger_replay_recovery_frame"] = trigger_frame
        episode["trigger_replay_lag_frames"] = (
            None if trigger_frame is None else start - trigger_frame
        )

        # Generic trigger fields now describe the current replay policy. The historical
        # production/v2 trigger remains available in the production_* fields above.
        episode["trigger_recovery_event_id"] = (
            None if trigger is None else trigger.get("event_id")
        )
        episode["trigger_recovery_frame"] = trigger_frame
        episode["trigger_lag_frames"] = (
            None if trigger_frame is None else start - trigger_frame
        )
        episode["recovery_event_count"] = len(inside)
        episode["recovery_event_ids"] = [event["event_id"] for event in inside]
        episode["recovery_event_frames"] = [int(event["frame_id"]) for event in inside]
        episode["trigger_features"] = _trigger_features(trigger)

        for event in inside:
            event["episode_id"] = episode.get("episode_id")
        if trigger is not None and trigger not in inside and trigger.get("episode_id") is None:
            trigger["episode_id"] = episode.get("episode_id")


def flatten_replay_event(event: dict[str, Any], horizons: list[int]) -> dict[str, Any]:
    pool = event.get("candidate_pool") or {}
    isolated = event.get("isolated_until_next_replay_recovery") or {}
    row = {
        "event_id": event.get("event_id"),
        "episode_id": event.get("episode_id"),
        "frame_id": event.get("frame_id"),
        "timestamp_sec": event.get("timestamp_sec"),
        "policy": event.get("policy"),
        "truth_available": event.get("truth_available"),
        "truth_visible": event.get("truth_visible"),
        "selected_correct": event.get("selected_correct"),
        "counterfactual_label": event.get("counterfactual_label"),
        "selected_index": event.get("selected_index"),
        "selected_confidence": event.get("selected_confidence"),
        "selected_gate_ratio": event.get("selected_gate_ratio"),
        "selected_size_similarity": event.get("selected_size_similarity"),
        "candidate_count": pool.get("candidate_count"),
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
        "production_source_same_frame": event.get("production_source_same_frame"),
        "production_reacquired_same_frame": event.get("production_reacquired_same_frame"),
        "isolated_net_on_correct_delta": isolated.get("net_v2_correct_delta"),
        "isolated_conditioned_correct_delta": isolated.get("conditioned_correct_delta"),
        "isolated_direct_not_visible_delta": isolated.get("direct_not_visible_delta"),
    }
    for horizon in horizons:
        metrics = (event.get("downstream_horizons") or {}).get(str(horizon)) or {}
        prefix = f"h{horizon}_"
        row[prefix + "net_on_correct_delta"] = metrics.get("net_v2_correct_delta")
        row[prefix + "conditioned_correct_delta"] = metrics.get("conditioned_correct_delta")
        row[prefix + "direct_not_visible_delta"] = metrics.get("direct_not_visible_delta")
        row[prefix + "crosses_next_replay_recovery"] = metrics.get(
            "crosses_next_replay_recovery"
        )
    return row


def _comparison_with_on_aliases(
    record: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    return {
        **record,
        "on_policy": policy,
        "on_source": record.get("v2_source"),
        "on_track_state": record.get("v2_track_state"),
        "on_selected_index": record.get("v2_selected_index"),
        "on_reacquired": record.get("v2_reacquired"),
        "on_output_center": record.get("v2_output_center"),
        "on_correct": record.get("v2_correct"),
    }


def _build_report(
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    top_n: int,
) -> str:
    report = extended.build_report(summary, episodes, top_n)
    replay = summary.get("replay_reacquisition_events") or {}
    section = "\n".join(
        [
            "## Current replay-policy reacquisition events",
            "",
            f"- Policy: `{replay.get('policy')}`",
            f"- Total replay recoveries: {replay.get('events_total_replay')}",
            f"- Recovery frames: `{replay.get('event_frames')}`",
            f"- Events with sparse truth: {replay.get('events_with_sparse_truth')}",
            f"- Visible labeled precision: {replay.get('precision_pct_on_visible_labeled_events')}%",
            f"- Counterfactual labels: `{json.dumps(replay.get('counterfactual_labels') or {}, ensure_ascii=False)}`",
            "",
            "These are recoveries actually emitted by the current ON replay policy. "
            "They are intentionally separate from historical production recoveries "
            "persisted in the source frames file.",
            "",
        ]
    )
    marker = "## Non-overlapping divergence episodes"
    if marker in report:
        return report.replace(marker, section + "\n" + marker, 1)
    return report + "\n" + section


def main() -> None:
    # Keep the mature miner implementation as the primary pipeline, then augment its
    # outputs with policy-correct replay recovery reporting. Redirect the first summary so
    # users see one final, corrected summary on stdout.
    with redirect_stdout(io.StringIO()):
        extended.main()

    args = extended.parse_args()
    frames_path = Path(args.frames).expanduser().resolve()
    truth_path = Path(args.truth).expanduser().resolve()
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else frames_path.parent / "recovery_error_mining"
    )
    horizons = extended.parse_horizons(args.horizons)
    rows = base.load_jsonl(frames_path)
    truth_rows = base.load_truth(truth_path)
    truth_by_frame = {int(row.get("frame_id", 0)): row for row in truth_rows}
    frame_shape = base.resolve_frame_shape(args)
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
    extended.enrich_replay_records(rows, off, on, frame_shape)
    probe = base.make_tracker(config, True)
    policy = str(getattr(probe, "reacquire_policy", "unknown"))

    replay_events = extract_replay_recovery_events(
        rows,
        truth_by_frame,
        off,
        on,
        config,
        frame_shape,
        horizons,
        policy,
    )
    historical_events = json.loads((output_dir / "events.json").read_text(encoding="utf-8"))
    episodes = json.loads((output_dir / "episodes.json").read_text(encoding="utf-8"))
    attach_replay_events_to_episodes(episodes, replay_events)

    comparisons = extended.compare_replays(
        off,
        on,
        truth_by_frame,
        center_tolerance_px=max(0.0, args.divergence_center_tolerance_px),
    )
    comparisons_by_frame = {int(record["frame_id"]): record for record in comparisons}
    frame_to_episode: dict[int, str] = {}
    for episode in episodes:
        for frame_id in range(int(episode["start_frame"]), int(episode["end_frame"]) + 1):
            frame_to_episode[frame_id] = str(episode["episode_id"])

    replay_summary = aggregate_replay_recovery_events(
        replay_events,
        max(0, args.window_before),
        max(0, args.window_after),
        policy,
    )
    episode_summary = extended.aggregate_episodes(
        episodes,
        metrics_off,
        metrics_on,
        total_production_frames=len(rows),
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary["tracker_policy"] = policy
    summary["replay_reacquisition_events"] = replay_summary
    summary["divergence_episodes"] = episode_summary
    summary.setdefault("counterfactual", {})["reacquire_off"] = metrics_off
    summary["counterfactual"]["reacquire_on"] = metrics_on
    summary["counterfactual"]["on_minus_off"] = base.metric_delta(metrics_off, metrics_on)
    summary["counterfactual"]["replay_policy"] = policy
    summary.setdefault("methodology", {})["recovery_event_sources"] = (
        "production_reacquisition_events are historical events persisted in the source "
        "frames file; replay_reacquisition_events are recoveries actually emitted by "
        "the current ON replay policy. Episode trigger_recovery_* fields now refer to "
        "the current replay policy, while production_trigger_recovery_* preserves the "
        "historical source-run trigger."
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "replay_reacquisition_events.json").write_text(
        json.dumps(replay_events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    extended.write_csv(
        output_dir / "replay_reacquisition_events.csv",
        [flatten_replay_event(event, horizons) for event in replay_events],
    )
    (output_dir / "episodes.json").write_text(
        json.dumps(episodes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    extended.write_csv(
        output_dir / "episodes.csv",
        [extended.flatten_episode(episode) for episode in episodes],
    )

    comparison_rows = [
        _comparison_with_on_aliases(record, policy)
        for record in comparisons
        if bool(record.get("material_diverged"))
        or bool(record.get("diagnostic_state_diverged"))
    ]
    extended.write_jsonl(
        output_dir / "divergence_frames.jsonl",
        (
            {
                **record,
                "episode_id": frame_to_episode.get(int(record["frame_id"])),
            }
            for record in comparison_rows
        ),
    )
    extended.write_csv(
        output_dir / "divergence_frames.csv",
        [
            {
                **extended.flatten_comparison(
                    record,
                    frame_to_episode.get(int(record["frame_id"])),
                ),
                "on_policy": policy,
                "on_source": record.get("on_source"),
                "on_track_state": record.get("on_track_state"),
                "on_selected_index": record.get("on_selected_index"),
                "on_reacquired": record.get("on_reacquired"),
            }
            for record in comparison_rows
        ],
    )

    replay_truth = [
        {
            "frame_id": frame_id,
            "visible": bool(truth_by_frame[frame_id].get("visible")),
            "episode_id": frame_to_episode.get(frame_id),
            "comparison": _comparison_with_on_aliases(
                comparisons_by_frame.get(frame_id, {}),
                policy,
            ),
            "off": off.get(frame_id),
            "on_policy": policy,
            "on": on.get(frame_id),
            "v2": on.get(frame_id),
        }
        for frame_id in sorted(truth_by_frame)
    ]
    (output_dir / "replay_frames.json").write_text(
        json.dumps(replay_truth, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    diagnostics = {
        "tracker_policy": policy,
        "production_reacquisition_events": summary.get("production_reacquisition_events"),
        "replay_reacquisition_events": replay_summary,
        "trigger_feature_discrimination": episode_summary.get(
            "trigger_feature_discrimination"
        ),
        "event_window_overlap": (
            summary.get("production_reacquisition_events") or {}
        ).get("window_overlap"),
        "replay_event_window_overlap": replay_summary.get("window_overlap"),
        "episode_attribution": episode_summary.get("attribution"),
        "guardrail": episode_summary.get("guardrail"),
    }
    (output_dir / "feature_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _build_report(summary, episodes, max(0, args.top_episodes)),
        encoding="utf-8",
    )

    # Keep the historical production event file unchanged; it describes the source run.
    # Re-write it only to ensure a stable JSON formatting after the upstream pass.
    (output_dir / "events.json").write_text(
        json.dumps(historical_events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OUTPUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
