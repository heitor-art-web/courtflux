from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * max(0.0, min(float(q), 1.0))))
    return round(ordered[index], 6)


def _rate(numerator: float, denominator: float) -> float:
    return round(100.0 * float(numerator) / max(float(denominator), 1.0), 2)


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return round(sum(data) / len(data), 6) if data else None


def _per_unit(numerator: float, denominator: float, unit: float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) * float(unit) / float(denominator), 6)


def _iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _point(value: Any) -> tuple[float, float] | None:
    try:
        if value is None or len(value) < 2:
            return None
        x, y = float(value[0]), float(value[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return x, y
    except (TypeError, ValueError):
        return None


def _player_records(frame: dict[str, Any]) -> list[dict[str, Any]]:
    values = frame.get("players")
    if values is None:
        values = frame.get("team_predictions", [])
    return [item for item in (values or []) if isinstance(item, dict)]


def _track_id(player: dict[str, Any]) -> int:
    try:
        return int(player.get("track_id", -1))
    except (TypeError, ValueError):
        return -1


def _player_box_height_ratio(player: dict[str, Any]) -> float | None:
    value = player.get("player_box_height_ratio")
    try:
        if value is not None:
            ratio = float(value)
            return ratio if math.isfinite(ratio) and ratio >= 0.0 else None
    except (TypeError, ValueError):
        pass
    box = player.get("box")
    frame_height = player.get("frame_height")
    try:
        if not isinstance(box, list) or len(box) < 4 or frame_height is None:
            return None
        height = max(0.0, float(box[3]) - float(box[1]))
        denom = float(frame_height)
        if denom <= 0.0:
            return None
        return height / denom
    except (TypeError, ValueError):
        return None


def _is_analytics_eligible(players: list[dict[str, Any]], *, minimum_player_boxes: int, median_box_height_ratio_max: float) -> tuple[bool, float | None]:
    ratios = [ratio for ratio in (_player_box_height_ratio(player) for player in players) if ratio is not None]
    median_ratio = statistics.median(ratios) if ratios else None
    eligible = bool(len(ratios) >= int(minimum_player_boxes) and median_ratio is not None and median_ratio <= float(median_box_height_ratio_max))
    return eligible, (round(float(median_ratio), 6) if median_ratio is not None else None)


def _frame_scope_summary(counts: list[int]) -> dict[str, Any]:
    median_players = statistics.median(counts) if counts else 0.0
    low_player_threshold = max(0, int(math.floor(median_players)) - 2)
    low_player_frames = sum(1 for value in counts if value <= low_player_threshold)
    return {
        "frames": len(counts),
        "players_per_frame_mean": _mean(counts),
        "players_per_frame_p10": _quantile([float(value) for value in counts], 0.10),
        "players_per_frame_median": _quantile([float(value) for value in counts], 0.50),
        "players_per_frame_p90": _quantile([float(value) for value in counts], 0.90),
        "low_player_count_proxy_threshold": low_player_threshold,
        "low_player_count_proxy_frames": low_player_frames,
        "low_player_count_proxy_rate_pct": _rate(low_player_frames, len(counts)),
    }


def _resolve_video_fps(frames: list[dict[str, Any]], explicit_video_fps: float | None) -> tuple[float | None, str]:
    try:
        if explicit_video_fps is not None:
            fps = float(explicit_video_fps)
            if math.isfinite(fps) and fps > 0.0:
                return fps, "run_summary"
    except (TypeError, ValueError):
        pass
    estimates: list[float] = []
    ordered = sorted((int(row.get("frame_id", 0) or 0), float(row.get("timestamp_sec", 0.0) or 0.0)) for row in frames if row.get("frame_id") is not None and row.get("timestamp_sec") is not None)
    for (left_frame, left_ts), (right_frame, right_ts) in zip(ordered, ordered[1:]):
        delta_frames = right_frame - left_frame
        delta_time = right_ts - left_ts
        if delta_frames > 0 and delta_time > 1e-6:
            estimate = delta_frames / delta_time
            if math.isfinite(estimate) and 1.0 <= estimate <= 240.0:
                estimates.append(estimate)
    if estimates:
        return round(float(statistics.median(estimates)), 6), "frame_timestamps"
    return None, "unavailable"


def load_frames(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def evaluate_frames(frames: list[dict[str, Any]], duplicate_iou_threshold: float = 0.75, court_jump_m_per_frame: float = 2.5, short_track_frames: int = 5, short_track_seconds: float = 0.20, video_fps: float | None = None, wide_view_minimum_player_boxes: int = 2, wide_view_median_box_height_ratio_max: float = 0.45) -> dict[str, Any]:
    gameplay = [row for row in frames if bool(row.get("valid_gameplay", False))]
    resolved_fps, fps_source = _resolve_video_fps(frames, video_fps)
    gameplay_exposure_sec = len(gameplay) / resolved_fps if resolved_fps is not None and resolved_fps > 0 else None
    all_player_counts: list[int] = []
    eligible_player_counts: list[int] = []
    wide_view_median_ratios: list[float] = []
    detected_observations = tracked_observations = untracked_observations = 0
    frames_with_player_detections = frames_without_player_detections = 0
    frames_with_any_untracked = frames_all_detections_untracked = 0
    eligible_frames_with_player_detections = eligible_frames_all_detections_untracked = 0
    duplicate_track_id_frames = duplicate_box_pairs = 0
    detector_timing_ms: list[float] = []
    known_team_observations = 0
    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in gameplay:
        frame_id = int(row.get("frame_id", 0) or 0)
        players = _player_records(row)
        player_count = len(players)
        all_player_counts.append(player_count)
        eligible, median_height_ratio = _is_analytics_eligible(players, minimum_player_boxes=wide_view_minimum_player_boxes, median_box_height_ratio_max=wide_view_median_box_height_ratio_max)
        if median_height_ratio is not None:
            wide_view_median_ratios.append(median_height_ratio)
        if eligible:
            eligible_player_counts.append(player_count)
        try:
            detected_this_frame = int(row.get("accepted_player_candidates", player_count) or 0)
        except (TypeError, ValueError):
            detected_this_frame = player_count
        detected_observations += max(0, detected_this_frame)
        if detected_this_frame > 0:
            frames_with_player_detections += 1
            if eligible:
                eligible_frames_with_player_detections += 1
        else:
            frames_without_player_detections += 1
        seen_ids: list[int] = []
        boxes: list[tuple[int, list[float] | None]] = []
        tracked_this_frame = untracked_this_frame = 0
        for player in players:
            track_id = _track_id(player)
            box = player.get("box")
            boxes.append((track_id, box if isinstance(box, list) else None))
            team = player.get("team")
            if team and str(team).lower() not in {"unknown", "none"}:
                known_team_observations += 1
            if track_id < 0:
                untracked_observations += 1
                untracked_this_frame += 1
                continue
            tracked_observations += 1
            tracked_this_frame += 1
            seen_ids.append(track_id)
            observations[track_id].append({"frame_id": frame_id, "timestamp_sec": float(row.get("timestamp_sec", 0.0) or 0.0), "box": box, "court_xy": player.get("court_xy"), "court_xy_valid": bool(player.get("court_xy_valid", False)), "homography_source": player.get("homography_source"), "team": team, "confidence": player.get("confidence")})
        if untracked_this_frame > 0:
            frames_with_any_untracked += 1
        if detected_this_frame > 0 and tracked_this_frame == 0:
            frames_all_detections_untracked += 1
            if eligible:
                eligible_frames_all_detections_untracked += 1
        if len(seen_ids) != len(set(seen_ids)):
            duplicate_track_id_frames += 1
        for index, (track_a, box_a) in enumerate(boxes):
            if track_a < 0:
                continue
            for track_b, box_b in boxes[index + 1:]:
                if track_b < 0 or track_a == track_b:
                    continue
                if _iou(box_a, box_b) >= duplicate_iou_threshold:
                    duplicate_box_pairs += 1
        timing = row.get("timing_ms", {}) or {}
        if isinstance(timing, dict):
            total = 0.0
            found = False
            for value in timing.values():
                try:
                    total += float(value)
                    found = True
                except (TypeError, ValueError):
                    pass
            if found:
                detector_timing_ms.append(total)

    track_observed_lengths: list[int] = []
    track_span_lengths: list[int] = []
    track_observed_durations_sec: list[float] = []
    track_span_durations_sec: list[float] = []
    track_coverage: list[float] = []
    interruption_events = short_tracks = temporal_short_tracks = 0
    team_change_events = 0
    team_majority_shares: list[float] = []
    court_steps_m_per_frame: list[float] = []
    court_jump_events = tracks_with_court = 0
    invalid_or_missing_court_samples = nonconsecutive_court_pairs_skipped = 0
    per_track: dict[str, Any] = {}

    for track_id, values in sorted(observations.items()):
        ordered = sorted(values, key=lambda item: item["frame_id"])
        frame_ids = [int(item["frame_id"]) for item in ordered]
        observed = len(ordered)
        span = max(frame_ids[-1] - frame_ids[0] + 1, 1)
        gaps = [max(0, right - left - 1) for left, right in zip(frame_ids, frame_ids[1:])]
        interruptions = sum(1 for gap in gaps if gap > 0)
        interruption_events += interruptions
        track_observed_lengths.append(observed)
        track_span_lengths.append(span)
        coverage = observed / span
        track_coverage.append(coverage)
        if observed <= short_track_frames:
            short_tracks += 1
        observed_duration_sec = span_duration_sec = max_internal_gap_sec = None
        if resolved_fps is not None:
            observed_duration_sec = observed / resolved_fps
            span_duration_sec = span / resolved_fps
            max_internal_gap_sec = (max(gaps) if gaps else 0) / resolved_fps
            track_observed_durations_sec.append(observed_duration_sec)
            track_span_durations_sec.append(span_duration_sec)
            if observed_duration_sec <= float(short_track_seconds) + 1e-9:
                temporal_short_tracks += 1
        known_teams = [str(item.get("team")) for item in ordered if item.get("team") and str(item.get("team")).lower() not in {"unknown", "none"}]
        team_changes = sum(1 for left, right in zip(known_teams, known_teams[1:]) if left != right)
        team_change_events += team_changes
        majority_share = None
        if known_teams:
            counts = Counter(known_teams)
            majority_share = max(counts.values()) / len(known_teams)
            team_majority_shares.append(majority_share)
        court_values = 0
        previous = None
        track_steps: list[float] = []
        for item in ordered:
            current = _point(item.get("court_xy")) if item.get("court_xy_valid") else None
            if current is None:
                invalid_or_missing_court_samples += 1
                continue
            court_values += 1
            if previous is not None:
                previous_frame, previous_point = previous
                delta_frames = int(item["frame_id"]) - previous_frame
                if delta_frames != 1:
                    nonconsecutive_court_pairs_skipped += 1
                else:
                    dx = current[0] - previous_point[0]
                    dy = current[1] - previous_point[1]
                    step = math.sqrt(dx * dx + dy * dy)
                    track_steps.append(step)
                    court_steps_m_per_frame.append(step)
                    if step > court_jump_m_per_frame:
                        court_jump_events += 1
            previous = (int(item["frame_id"]), current)
        if court_values >= 2:
            tracks_with_court += 1
        per_track[str(track_id)] = {"observed_frames": observed, "span_frames": span, "observed_duration_sec": round(observed_duration_sec, 6) if observed_duration_sec is not None else None, "span_duration_sec": round(span_duration_sec, 6) if span_duration_sec is not None else None, "observation_coverage_pct": round(coverage * 100.0, 2), "interruptions": interruptions, "max_internal_gap_frames": max(gaps) if gaps else 0, "max_internal_gap_sec": round(max_internal_gap_sec, 6) if max_internal_gap_sec is not None else None, "team_changes": team_changes, "team_majority_share_pct": round(majority_share * 100.0, 2) if majority_share is not None else None, "valid_court_samples": court_values, "court_step_m_per_consecutive_frame_p90": _quantile(track_steps, 0.90)}

    all_observations = tracked_observations + untracked_observations
    all_scope = _frame_scope_summary(all_player_counts)
    eligible_scope = _frame_scope_summary(eligible_player_counts)
    gameplay_exposure_minutes = gameplay_exposure_sec / 60.0 if gameplay_exposure_sec is not None else None
    temporal_metrics_available = resolved_fps is not None
    tracking_time_normalized = {"video_fps": resolved_fps, "fps_source": fps_source, "gameplay_exposure_sec": round(gameplay_exposure_sec, 6) if gameplay_exposure_sec is not None else None, "short_track_threshold_sec": float(short_track_seconds), "short_tracks": temporal_short_tracks if temporal_metrics_available else None, "short_track_rate_pct": _rate(temporal_short_tracks, len(observations)) if temporal_metrics_available else None, "observed_track_duration_median_sec": _quantile(track_observed_durations_sec, 0.50), "observed_track_duration_p90_sec": _quantile(track_observed_durations_sec, 0.90), "observed_track_duration_max_sec": round(max(track_observed_durations_sec), 6) if track_observed_durations_sec else None, "track_span_median_sec": _quantile(track_span_durations_sec, 0.50), "same_id_interruptions_per_gameplay_minute": _per_unit(interruption_events, gameplay_exposure_minutes, 1.0) if gameplay_exposure_minutes is not None else None, "unique_track_ids_per_gameplay_minute": _per_unit(len(observations), gameplay_exposure_minutes, 1.0) if gameplay_exposure_minutes is not None else None, "unique_track_ids_per_1000_tracked_observations": _per_unit(len(observations), tracked_observations, 1000.0), "same_id_interruptions_per_1000_tracked_observations": _per_unit(interruption_events, tracked_observations, 1000.0), "quality_gate_eligible": temporal_metrics_available, "semantics": "fps_normalized_cross_video_tracking_diagnostics"}

    return {
        "schema_version": 3,
        "scope": "P1.1_player_tracking_quality",
        "ground_truth_status": {"true_id_switches": "requires_player_identity_ground_truth", "true_track_fragmentation": "requires_player_identity_ground_truth", "player_recall": "requires_player_presence_ground_truth", "team_classification_accuracy": "requires_correct_match_context_and_team_ground_truth", "reported_interruptions": "same_track_id_gap_proxy_only", "reported_duplicate_boxes": "high_iou_distinct_track_proxy_only"},
        "frames": {"gameplay_frames": len(gameplay), "analytics_eligible_frames": len(eligible_player_counts), "analytics_eligible_rate_pct": _rate(len(eligible_player_counts), len(gameplay)), "analytics_eligibility_proxy": {"minimum_player_boxes": int(wide_view_minimum_player_boxes), "median_player_box_height_ratio_max": float(wide_view_median_box_height_ratio_max), "median_observed_player_box_height_ratio": _quantile(wide_view_median_ratios, 0.50), "quality_gate_eligible": False, "semantics": "diagnostic_wide_view_proxy_not_gameplay_policy"}, "all_gameplay": all_scope, "analytics_eligible": eligible_scope, "players_per_frame_mean": all_scope["players_per_frame_mean"], "players_per_frame_p10": all_scope["players_per_frame_p10"], "players_per_frame_median": all_scope["players_per_frame_median"], "players_per_frame_p90": all_scope["players_per_frame_p90"], "low_player_count_proxy_threshold": all_scope["low_player_count_proxy_threshold"], "low_player_count_proxy_frames": all_scope["low_player_count_proxy_frames"], "low_player_count_proxy_rate_pct": all_scope["low_player_count_proxy_rate_pct"]},
        "detector_tracker_bridge": {"detected_player_observations": detected_observations, "tracked_player_observations": tracked_observations, "untracked_player_observations": untracked_observations, "tracker_assignment_rate_pct": _rate(tracked_observations, detected_observations), "untracked_observation_rate_pct": _rate(untracked_observations, all_observations), "frames_with_player_detections": frames_with_player_detections, "frames_without_player_detections": frames_without_player_detections, "frames_with_any_untracked": frames_with_any_untracked, "frames_all_detections_untracked": frames_all_detections_untracked, "frames_all_detections_untracked_rate_pct": _rate(frames_all_detections_untracked, frames_with_player_detections), "analytics_eligible_frames_with_player_detections": eligible_frames_with_player_detections, "analytics_eligible_frames_all_detections_untracked": eligible_frames_all_detections_untracked, "analytics_eligible_all_untracked_rate_pct": _rate(eligible_frames_all_detections_untracked, eligible_frames_with_player_detections)},
        "tracking": {"unique_track_ids": len(observations), "tracked_observations": tracked_observations, "untracked_observations": untracked_observations, "untracked_observation_rate_pct": _rate(untracked_observations, all_observations), "duplicate_track_id_frames": duplicate_track_id_frames, "duplicate_track_id_frame_rate_pct": _rate(duplicate_track_id_frames, len(gameplay)), "high_iou_distinct_track_pairs": duplicate_box_pairs, "short_track_threshold_frames": short_track_frames, "short_tracks": short_tracks, "short_track_rate_pct": _rate(short_tracks, len(observations)), "observed_track_length_median_frames": _quantile([float(v) for v in track_observed_lengths], 0.50), "observed_track_length_p90_frames": _quantile([float(v) for v in track_observed_lengths], 0.90), "observed_track_length_max_frames": max(track_observed_lengths) if track_observed_lengths else None, "track_span_median_frames": _quantile([float(v) for v in track_span_lengths], 0.50), "track_observation_coverage_median_pct": round(100.0 * statistics.median(track_coverage), 2) if track_coverage else None, "same_id_interruption_events": interruption_events, "time_normalized": tracking_time_normalized},
        "team_stability": {"known_team_observations": known_team_observations, "team_assignment_coverage_pct": _rate(known_team_observations, all_observations), "team_change_events": team_change_events, "track_team_majority_share_median_pct": round(100.0 * statistics.median(team_majority_shares), 2) if team_majority_shares else None, "quality_gate_eligible": False, "accuracy_claim": "not_measured"},
        "court_continuity": {"tracks_with_multiple_valid_court_samples": tracks_with_court, "invalid_or_missing_court_samples": invalid_or_missing_court_samples, "nonconsecutive_valid_pairs_skipped": nonconsecutive_court_pairs_skipped, "court_step_m_per_consecutive_frame_median": _quantile(court_steps_m_per_frame, 0.50), "court_step_m_per_consecutive_frame_p90": _quantile(court_steps_m_per_frame, 0.90), "court_step_m_per_consecutive_frame_max": round(max(court_steps_m_per_frame), 6) if court_steps_m_per_frame else None, "jump_threshold_m_per_frame": court_jump_m_per_frame, "jump_events": court_jump_events, "jump_rate_pct": _rate(court_jump_events, len(court_steps_m_per_frame)), "quality_gate_eligible": False, "semantics": "valid_plausible_homography_samples_and_consecutive_frames_only"},
        "performance": {"detector_timing_sum_ms_per_gameplay_frame_mean": _mean(detector_timing_ms)},
        "per_track": per_track,
    }


def evaluate_run(run: Path, benchmark_context: dict[str, Any] | None = None) -> dict[str, Any]:
    run = Path(run)
    frames_path = run / "frames.jsonl" if run.is_dir() else run
    run_dir = frames_path.parent
    context = dict(benchmark_context or {})
    proxy = context.get("analytics_eligibility_proxy", {}) or {}
    time_normalization = context.get("time_normalization", {}) or {}
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    performance = summary.get("performance", {}) or {}
    video_fps = performance.get("video_fps")
    report = evaluate_frames(load_frames(frames_path), video_fps=video_fps, short_track_seconds=float(time_normalization.get("short_track_threshold_sec", 0.20) or 0.20), wide_view_minimum_player_boxes=int(proxy.get("minimum_player_boxes", 2) or 2), wide_view_median_box_height_ratio_max=float(proxy.get("median_player_box_height_ratio_max", 0.45) or 0.45))
    report["benchmark_context"] = context
    report["team_stability"]["quality_gate_eligible"] = bool(context.get("team_metrics_in_quality_gate", False))
    report["run_dir"] = str(run_dir)
    report["frames_file"] = str(frames_path)
    if config_path.exists():
        report["run_config"] = json.loads(config_path.read_text(encoding="utf-8"))
    if summary:
        report["run_summary_excerpt"] = {"run_id": summary.get("run_id"), "video_source": summary.get("video_source"), "model": summary.get("model"), "imgsz": summary.get("imgsz"), "tracker": summary.get("tracker"), "device": summary.get("device"), "elapsed_sec": summary.get("elapsed_sec"), "performance": performance}
    return report


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    team_eligible = [report for report in reports if bool(report.get("team_stability", {}).get("quality_gate_eligible", False)) and report.get("team_stability", {}).get("track_team_majority_share_median_pct") is not None]
    temporal_eligible = [report for report in reports if bool(report.get("tracking", {}).get("time_normalized", {}).get("quality_gate_eligible", False))]
    def temporal_values(key: str) -> list[float]:
        values: list[float] = []
        for report in temporal_eligible:
            value = report["tracking"]["time_normalized"].get(key)
            if value is not None:
                values.append(float(value))
        return values
    temporal_short_rates = temporal_values("short_track_rate_pct")
    interruptions_per_minute = temporal_values("same_id_interruptions_per_gameplay_minute")
    unique_ids_per_minute = temporal_values("unique_track_ids_per_gameplay_minute")
    ids_per_1000 = temporal_values("unique_track_ids_per_1000_tracked_observations")
    return {"runs": len(reports), "gameplay_frames_total": sum(int(r["frames"]["gameplay_frames"]) for r in reports), "analytics_eligible_frames_total": sum(int(r["frames"]["analytics_eligible_frames"]) for r in reports), "analytics_eligible_rate_macro_pct": _mean(r["frames"]["analytics_eligible_rate_pct"] for r in reports), "all_gameplay_players_per_frame_macro_mean": _mean(r["frames"]["all_gameplay"]["players_per_frame_mean"] for r in reports if r["frames"]["all_gameplay"]["players_per_frame_mean"] is not None), "analytics_eligible_players_per_frame_macro_mean": _mean(r["frames"]["analytics_eligible"]["players_per_frame_mean"] for r in reports if r["frames"]["analytics_eligible"]["players_per_frame_mean"] is not None), "unique_track_ids_macro_mean": _mean(r["tracking"]["unique_track_ids"] for r in reports), "short_track_rate_macro_pct": _mean(r["tracking"]["short_track_rate_pct"] for r in reports), "tracker_assignment_rate_macro_pct": _mean(r["detector_tracker_bridge"]["tracker_assignment_rate_pct"] for r in reports), "all_untracked_frame_rate_macro_pct": _mean(r["detector_tracker_bridge"]["frames_all_detections_untracked_rate_pct"] for r in reports), "untracked_observation_rate_macro_pct": _mean(r["tracking"]["untracked_observation_rate_pct"] for r in reports), "track_observation_coverage_median_macro_pct": _mean(r["tracking"]["track_observation_coverage_median_pct"] for r in reports if r["tracking"]["track_observation_coverage_median_pct"] is not None), "time_normalized_quality_gate_eligible_runs": len(temporal_eligible), "time_normalized_short_track_rate_macro_pct": _mean(temporal_short_rates), "time_normalized_same_id_interruptions_per_gameplay_minute_macro": _mean(interruptions_per_minute), "time_normalized_unique_track_ids_per_gameplay_minute_macro": _mean(unique_ids_per_minute), "time_normalized_unique_track_ids_per_1000_tracked_observations_macro": _mean(ids_per_1000), "worst_run_time_normalized_short_track_rate_pct": max(temporal_short_rates) if temporal_short_rates else None, "team_majority_share_median_macro_pct": _mean(r["team_stability"]["track_team_majority_share_median_pct"] for r in team_eligible), "team_quality_gate_eligible_runs": len(team_eligible), "court_valid_jump_rate_macro_pct": _mean(r["court_continuity"]["jump_rate_pct"] for r in reports), "court_continuity_quality_gate_eligible": False, "worst_run_short_track_rate_pct": max(r["tracking"]["short_track_rate_pct"] for r in reports), "worst_run_untracked_observation_rate_pct": max(r["tracking"]["untracked_observation_rate_pct"] for r in reports)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CourtFlux P1.1 player tracking diagnostics")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or frames.jsonl files")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path")
    args = parser.parse_args()
    reports = [evaluate_run(path) for path in args.runs]
    payload = {"created_at": datetime.now().isoformat(), "scope": "P1.1_player_tracking_quality", "schema_version": 3, "reports": reports, "aggregate": aggregate_reports(reports)}
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("evaluation") / "p1_tracking_reports" / f"p1_tracking_v3_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
