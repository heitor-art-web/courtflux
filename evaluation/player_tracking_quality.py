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
    # Future-proof for an explicit players field while remaining compatible with
    # P1.1 instrumentation, which enriches the already-serialized team_predictions.
    values = frame.get("players")
    if values is None:
        values = frame.get("team_predictions", [])
    return [item for item in (values or []) if isinstance(item, dict)]


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


def evaluate_frames(
    frames: list[dict[str, Any]],
    duplicate_iou_threshold: float = 0.75,
    court_jump_m_per_frame: float = 2.5,
    short_track_frames: int = 5,
) -> dict[str, Any]:
    gameplay = [row for row in frames if bool(row.get("valid_gameplay", False))]
    player_counts: list[int] = []
    untracked_observations = 0
    tracked_observations = 0
    duplicate_track_id_frames = 0
    duplicate_box_pairs = 0
    detector_timing_ms: list[float] = []

    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in gameplay:
        frame_id = int(row.get("frame_id", 0) or 0)
        players = _player_records(row)
        player_counts.append(len(players))

        seen_ids: list[int] = []
        boxes: list[tuple[int, list[float] | None]] = []
        for player in players:
            try:
                track_id = int(player.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1
            box = player.get("box")
            boxes.append((track_id, box if isinstance(box, list) else None))
            if track_id < 0:
                untracked_observations += 1
                continue
            tracked_observations += 1
            seen_ids.append(track_id)
            observations[track_id].append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": float(row.get("timestamp_sec", 0.0) or 0.0),
                    "box": box,
                    "court_xy": player.get("court_xy"),
                    "team": player.get("team"),
                    "confidence": player.get("confidence"),
                }
            )

        if len(seen_ids) != len(set(seen_ids)):
            duplicate_track_id_frames += 1

        for index, (track_a, box_a) in enumerate(boxes):
            if track_a < 0:
                continue
            for track_b, box_b in boxes[index + 1 :]:
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
    track_coverage: list[float] = []
    interruption_events = 0
    short_tracks = 0
    team_change_events = 0
    team_majority_shares: list[float] = []
    court_steps_m_per_frame: list[float] = []
    court_jump_events = 0
    tracks_with_court = 0

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

        known_teams = [
            str(item.get("team"))
            for item in ordered
            if item.get("team") and str(item.get("team")).lower() not in {"unknown", "none"}
        ]
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
            current = _point(item.get("court_xy"))
            if current is None:
                continue
            court_values += 1
            if previous is not None:
                previous_frame, previous_point = previous
                delta_frames = max(int(item["frame_id"]) - previous_frame, 1)
                dx = current[0] - previous_point[0]
                dy = current[1] - previous_point[1]
                step = math.sqrt(dx * dx + dy * dy) / delta_frames
                track_steps.append(step)
                court_steps_m_per_frame.append(step)
                if step > court_jump_m_per_frame:
                    court_jump_events += 1
            previous = (int(item["frame_id"]), current)
        if court_values >= 2:
            tracks_with_court += 1

        per_track[str(track_id)] = {
            "observed_frames": observed,
            "span_frames": span,
            "observation_coverage_pct": round(coverage * 100.0, 2),
            "interruptions": interruptions,
            "max_internal_gap_frames": max(gaps) if gaps else 0,
            "team_changes": team_changes,
            "team_majority_share_pct": (
                round(majority_share * 100.0, 2) if majority_share is not None else None
            ),
            "court_samples": court_values,
            "court_step_m_per_frame_p90": _quantile(track_steps, 0.90),
        }

    median_players = statistics.median(player_counts) if player_counts else 0.0
    low_player_threshold = max(0, int(math.floor(median_players)) - 2)
    low_player_frames = sum(1 for value in player_counts if value <= low_player_threshold)
    all_observations = tracked_observations + untracked_observations

    return {
        "schema_version": 1,
        "scope": "P1.1_player_tracking_quality",
        "ground_truth_status": {
            "true_id_switches": "requires_player_identity_ground_truth",
            "true_track_fragmentation": "requires_player_identity_ground_truth",
            "player_recall": "requires_player_presence_ground_truth",
            "reported_interruptions": "same_track_id_gap_proxy_only",
            "reported_duplicate_boxes": "high_iou_distinct_track_proxy_only",
        },
        "frames": {
            "gameplay_frames": len(gameplay),
            "players_per_frame_mean": _mean(player_counts),
            "players_per_frame_p10": _quantile([float(v) for v in player_counts], 0.10),
            "players_per_frame_median": _quantile([float(v) for v in player_counts], 0.50),
            "players_per_frame_p90": _quantile([float(v) for v in player_counts], 0.90),
            "low_player_count_proxy_threshold": low_player_threshold,
            "low_player_count_proxy_frames": low_player_frames,
            "low_player_count_proxy_rate_pct": _rate(low_player_frames, len(gameplay)),
        },
        "tracking": {
            "unique_track_ids": len(observations),
            "tracked_observations": tracked_observations,
            "untracked_observations": untracked_observations,
            "untracked_observation_rate_pct": _rate(untracked_observations, all_observations),
            "duplicate_track_id_frames": duplicate_track_id_frames,
            "duplicate_track_id_frame_rate_pct": _rate(duplicate_track_id_frames, len(gameplay)),
            "high_iou_distinct_track_pairs": duplicate_box_pairs,
            "short_track_threshold_frames": short_track_frames,
            "short_tracks": short_tracks,
            "short_track_rate_pct": _rate(short_tracks, len(observations)),
            "observed_track_length_median_frames": _quantile([float(v) for v in track_observed_lengths], 0.50),
            "observed_track_length_p90_frames": _quantile([float(v) for v in track_observed_lengths], 0.90),
            "observed_track_length_max_frames": max(track_observed_lengths) if track_observed_lengths else None,
            "track_span_median_frames": _quantile([float(v) for v in track_span_lengths], 0.50),
            "track_observation_coverage_median_pct": (
                round(100.0 * statistics.median(track_coverage), 2) if track_coverage else None
            ),
            "same_id_interruption_events": interruption_events,
        },
        "team_stability": {
            "team_change_events": team_change_events,
            "track_team_majority_share_median_pct": (
                round(100.0 * statistics.median(team_majority_shares), 2)
                if team_majority_shares
                else None
            ),
        },
        "court_continuity": {
            "tracks_with_multiple_court_samples": tracks_with_court,
            "court_step_m_per_frame_median": _quantile(court_steps_m_per_frame, 0.50),
            "court_step_m_per_frame_p90": _quantile(court_steps_m_per_frame, 0.90),
            "court_step_m_per_frame_max": (
                round(max(court_steps_m_per_frame), 6) if court_steps_m_per_frame else None
            ),
            "jump_threshold_m_per_frame": court_jump_m_per_frame,
            "jump_events": court_jump_events,
            "jump_rate_pct": _rate(court_jump_events, len(court_steps_m_per_frame)),
        },
        "performance": {
            "detector_timing_sum_ms_per_gameplay_frame_mean": _mean(detector_timing_ms),
        },
        "per_track": per_track,
    }


def evaluate_run(run: Path) -> dict[str, Any]:
    run = Path(run)
    frames_path = run / "frames.jsonl" if run.is_dir() else run
    run_dir = frames_path.parent
    report = evaluate_frames(load_frames(frames_path))
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    report["run_dir"] = str(run_dir)
    report["frames_file"] = str(frames_path)
    if config_path.exists():
        report["run_config"] = json.loads(config_path.read_text(encoding="utf-8"))
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report["run_summary_excerpt"] = {
            "run_id": summary.get("run_id"),
            "video_source": summary.get("video_source"),
            "model": summary.get("model"),
            "imgsz": summary.get("imgsz"),
            "tracker": summary.get("tracker"),
            "device": summary.get("device"),
            "elapsed_sec": summary.get("elapsed_sec"),
            "performance": summary.get("performance", {}),
        }
    return report


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    return {
        "runs": len(reports),
        "gameplay_frames_total": sum(int(r["frames"]["gameplay_frames"]) for r in reports),
        "unique_track_ids_macro_mean": _mean(r["tracking"]["unique_track_ids"] for r in reports),
        "short_track_rate_macro_pct": _mean(r["tracking"]["short_track_rate_pct"] for r in reports),
        "untracked_observation_rate_macro_pct": _mean(
            r["tracking"]["untracked_observation_rate_pct"] for r in reports
        ),
        "track_observation_coverage_median_macro_pct": _mean(
            r["tracking"]["track_observation_coverage_median_pct"]
            for r in reports
            if r["tracking"]["track_observation_coverage_median_pct"] is not None
        ),
        "team_majority_share_median_macro_pct": _mean(
            r["team_stability"]["track_team_majority_share_median_pct"]
            for r in reports
            if r["team_stability"]["track_team_majority_share_median_pct"] is not None
        ),
        "court_jump_rate_macro_pct": _mean(r["court_continuity"]["jump_rate_pct"] for r in reports),
        "worst_run_short_track_rate_pct": max(r["tracking"]["short_track_rate_pct"] for r in reports),
        "worst_run_untracked_observation_rate_pct": max(
            r["tracking"]["untracked_observation_rate_pct"] for r in reports
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CourtFlux P1.1 player tracking diagnostics")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or frames.jsonl files")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path")
    args = parser.parse_args()

    reports = [evaluate_run(path) for path in args.runs]
    payload = {
        "created_at": datetime.now().isoformat(),
        "scope": "P1.1_player_tracking_quality",
        "reports": reports,
        "aggregate": aggregate_reports(reports),
    }
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("evaluation") / "p1_tracking_reports" / f"p1_tracking_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
