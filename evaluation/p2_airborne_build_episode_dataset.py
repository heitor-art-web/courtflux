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

from services.p2.airborne_classifier_v1 import AirborneClassifierV1 as _FrameFeatureClassifier
from services.p2.airborne_classifier_v1_1 import AirborneClassifierV1

FRAME_FEATURES = _FrameFeatureClassifier.FEATURE_NAMES
FEATURES = AirborneClassifierV1.FEATURE_NAMES
CORE_TRAIN_STATES = ("possessed", "pass_flight", "shot_flight")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _truth_by_frame(gt: dict[str, Any]) -> dict[int, tuple[str, int]]:
    result: dict[int, tuple[str, int]] = {}
    for interval_index, row in enumerate(list(gt.get("control_intervals", []) or [])):
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or "unknown")
        start = int(row.get("frame_start", row.get("frame", 0)))
        end = int(row.get("frame_end", start))
        if end < start:
            start, end = end, start
        for frame_id in range(start, end + 1):
            result[frame_id] = (state, interval_index)
    return result


def _features_from_existing_row(row: dict[str, Any], *, fps: float) -> dict[str, float]:
    """Reconstruct the causal frame feature contract from historical p2_states rows."""
    fps = max(1.0, float(fps))
    flight_frames = max(0.0, _float(row.get("flight_frames"), 0.0))
    direct_observations = max(0.0, _float(row.get("flight_direct_observations"), 0.0))
    delta_distance = _float(row.get("shot_preflight_delta_distance_norm"), 0.0)
    delta_relative_y = _float(row.get("shot_preflight_delta_relative_y"), 0.0)
    approach_streak = max(0.0, _float(row.get("shot_preflight_approach_streak"), 0.0))
    rising_streak = max(0.0, _float(row.get("shot_preflight_rising_streak"), 0.0))
    values = {
        "flight_supported": 1.0 if bool(row.get("flight_supported", False)) else 0.0,
        "flight_duration_sec": flight_frames / fps,
        "flight_direct_observation_fraction": direct_observations / max(flight_frames, 1.0),
        "flight_displacement_player_heights": _float(
            row.get("flight_displacement_player_heights"), 0.0
        ),
        "receiver_evidence": 1.0
        if bool(row.get("pass_flight_receiver_evidence", False))
        else 0.0,
        "raw_best_distinct_from_origin": 1.0
        if bool(row.get("raw_best_distinct_from_flight_origin", False))
        else 0.0,
        "raw_best_score": _float(row.get("raw_best_score"), 0.0),
        "candidate_score": _float(row.get("candidate_score"), 0.0),
        "shot_active": 1.0 if bool(row.get("shot_active", False)) else 0.0,
        "shot_preflight_available": 1.0
        if bool(row.get("shot_preflight_available", False))
        else 0.0,
        "shot_preflight_direct_observed": 1.0
        if bool(row.get("shot_preflight_direct_observed", False))
        else 0.0,
        "shot_preflight_near_x": 1.0
        if bool(row.get("shot_preflight_near_x", False))
        else 0.0,
        "shot_preflight_above": 1.0
        if bool(row.get("shot_preflight_above", False))
        else 0.0,
        "shot_preflight_ball_to_rim_distance_norm": _float(
            row.get("shot_preflight_ball_to_rim_distance_norm"), 0.0
        ),
        "shot_preflight_horizontal_distance_rim_widths": _float(
            row.get("shot_preflight_horizontal_distance_rim_widths"), 0.0
        ),
        "shot_preflight_relative_y_rim_heights": _float(
            row.get("shot_preflight_relative_y_rim_heights"), 0.0
        ),
        "shot_preflight_distance_velocity_norm_per_sec": delta_distance * fps,
        "shot_preflight_vertical_velocity_heights_per_sec": delta_relative_y * fps,
        "shot_preflight_approach_duration_sec": approach_streak / fps,
        "shot_preflight_rising_duration_sec": rising_streak / fps,
    }
    return {name: float(values.get(name, 0.0)) for name in FRAME_FEATURES}


def _flight_key(row: dict[str, Any]) -> tuple[int, int, int | None] | None:
    if not bool(row.get("engine_eligible", False)) or not bool(row.get("flight_active", False)):
        return None
    frame_id = _int(row.get("frame_id"))
    if frame_id is None or frame_id < 0:
        return None
    flight_frames = max(1, _int(row.get("flight_frames")) or 1)
    start = max(0, frame_id - flight_frames + 1)
    reset_count = _int(row.get("reset_count")) or 0
    origin = _int(row.get("flight_origin_track_id"))
    return reset_count, start, origin


def _finalize_episode(
    *,
    clip_id: str,
    video_id: str,
    run_id: str,
    fps: float,
    ordinal: int,
    key: tuple[int, int, int | None],
    frame_ids: list[int],
    frame_features: list[dict[str, float]],
    truth: dict[int, tuple[str, int]],
) -> dict[str, Any] | None:
    if not frame_ids or not frame_features:
        return None

    overlaps: Counter[str] = Counter()
    truth_intervals: set[int] = set()
    for frame_id in frame_ids:
        target = truth.get(frame_id)
        if target is None:
            continue
        state, interval_index = target
        overlaps[state] += 1
        truth_intervals.add(int(interval_index))

    airborne_frames = int(overlaps["pass_flight"] + overlaps["shot_flight"])
    possessed_frames = int(overlaps["possessed"])
    if airborne_frames <= 0 and possessed_frames <= 0:
        return None

    stage1_label = "airborne" if airborne_frames > 0 else "continuity"
    pass_frames = int(overlaps["pass_flight"])
    shot_frames = int(overlaps["shot_flight"])
    if stage1_label == "airborne" and pass_frames > 0 and shot_frames == 0:
        stage2_label: str | None = "pass_flight"
    elif stage1_label == "airborne" and shot_frames > 0 and pass_frames == 0:
        stage2_label = "shot_flight"
    else:
        # A flight hypothesis spanning both human PASS and SHOT windows is useful for
        # Stage 1 but has no clean Stage-2 target and is excluded from Stage-2 fitting.
        stage2_label = None

    accumulator = AirborneClassifierV1.new_accumulator()
    prefixes: list[dict[str, Any]] = []
    final_features: dict[str, float] | None = None
    for frame_id, features in zip(frame_ids, frame_features):
        final_features = AirborneClassifierV1.update_accumulator(accumulator, features)
        prefixes.append(
            {
                "frame_id": int(frame_id),
                "elapsed_sec": round(float(final_features["episode_duration_sec"]), 6),
                "features": {name: float(final_features[name]) for name in FEATURES},
            }
        )
    if final_features is None:
        return None

    reset_count, inferred_start, origin_track_id = key
    return {
        "episode_id": f"{clip_id}:flight:{ordinal}:{reset_count}:{inferred_start}:{origin_track_id}",
        "clip_id": clip_id,
        "video_id": video_id,
        "run_id": run_id,
        "fps": float(fps),
        "frame_start": int(frame_ids[0]),
        "frame_end": int(frame_ids[-1]),
        "frames": len(frame_ids),
        "flight_key": {
            "reset_count": int(reset_count),
            "inferred_start_frame": int(inferred_start),
            "origin_track_id": origin_track_id,
        },
        "truth_overlap_frames": {
            state: int(overlaps.get(state, 0)) for state in CORE_TRAIN_STATES
        },
        "truth_interval_indices": sorted(truth_intervals),
        "stage1_label": stage1_label,
        "stage2_label": stage2_label,
        "features": {name: float(final_features[name]) for name in FEATURES},
        "prefixes": prefixes,
    }


def build(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if str(manifest.get("version") or "") != "courtflux_airborne_dev_manifest_v2":
        raise ValueError("development manifest version must be courtflux_airborne_dev_manifest_v2")

    episodes: list[dict[str, Any]] = []
    clip_summary: list[dict[str, Any]] = []
    video_ids: set[str] = set()

    for clip in list(manifest.get("clips", []) or []):
        if not isinstance(clip, dict):
            continue
        clip_id = str(clip.get("id") or "")
        video_id = str(clip.get("video_id") or "")
        run_id = str(clip.get("run_id") or "")
        gt_rel = str(clip.get("gt_file") or "")
        if not clip_id or not video_id or not run_id or not gt_rel:
            raise ValueError(f"invalid development clip entry: {clip}")
        if video_id in video_ids:
            raise ValueError(f"{clip_id}: each development clip must use a distinct video")
        video_ids.add(video_id)

        states_path = PROJECT_ROOT / "evaluation" / "runs" / run_id / "p2_states.jsonl"
        gt_path = PROJECT_ROOT / gt_rel
        if not states_path.exists():
            raise FileNotFoundError(f"{clip_id}: missing {states_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"{clip_id}: missing {gt_path}")

        gt = _load_json(gt_path)
        fps = max(1.0, _float(dict(gt.get("video", {}) or {}).get("fps"), 30.0))
        truth = _truth_by_frame(gt)
        state_rows = sorted(_load_jsonl(states_path), key=lambda row: int(row.get("frame_id", -1)))

        current_key: tuple[int, int, int | None] | None = None
        current_frames: list[int] = []
        current_features: list[dict[str, float]] = []
        previous_frame: int | None = None
        ordinal = 0
        clip_episodes: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal ordinal, current_key, current_frames, current_features
            if current_key is None:
                return
            episode = _finalize_episode(
                clip_id=clip_id,
                video_id=video_id,
                run_id=run_id,
                fps=fps,
                ordinal=ordinal,
                key=current_key,
                frame_ids=current_frames,
                frame_features=current_features,
                truth=truth,
            )
            ordinal += 1
            if episode is not None:
                episodes.append(episode)
                clip_episodes.append(episode)
            current_key = None
            current_frames = []
            current_features = []

        for row in state_rows:
            frame_id = _int(row.get("frame_id"))
            key = _flight_key(row)
            if frame_id is None or key is None:
                flush()
                previous_frame = frame_id
                continue
            if (
                current_key is not None
                and (key != current_key or previous_frame is None or frame_id != previous_frame + 1)
            ):
                flush()
            if current_key is None:
                current_key = key
            current_frames.append(frame_id)
            current_features.append(_features_from_existing_row(row, fps=fps))
            previous_frame = frame_id
        flush()

        stage1_counts = Counter(str(row["stage1_label"]) for row in clip_episodes)
        stage2_counts = Counter(
            str(row["stage2_label"])
            for row in clip_episodes
            if row.get("stage2_label") is not None
        )
        clip_summary.append(
            {
                "clip_id": clip_id,
                "video_id": video_id,
                "run_id": run_id,
                "fps": fps,
                "episodes": len(clip_episodes),
                "stage1_labels": dict(sorted(stage1_counts.items())),
                "stage2_labels": dict(sorted(stage2_counts.items())),
                "ambiguous_stage2": sum(
                    1
                    for row in clip_episodes
                    if row["stage1_label"] == "airborne" and row.get("stage2_label") is None
                ),
            }
        )

    if len(video_ids) < 3:
        raise ValueError("grouped cross-validation requires at least three development videos")
    if not episodes:
        raise ValueError("development dataset contains no labelled flight episodes")

    stage1_counts = Counter(str(row["stage1_label"]) for row in episodes)
    stage2_counts = Counter(
        str(row["stage2_label"]) for row in episodes if row.get("stage2_label") is not None
    )
    for label in AirborneClassifierV1.STAGE1_LABELS:
        if stage1_counts.get(label, 0) == 0:
            raise ValueError(f"episode dataset missing Stage-1 class: {label}")
    for label in AirborneClassifierV1.STAGE2_LABELS:
        if stage2_counts.get(label, 0) == 0:
            raise ValueError(f"episode dataset missing Stage-2 class: {label}")

    summary = {
        "version": "courtflux_airborne_episode_dataset_v1_1",
        "contract": "one contiguous flight hypothesis = one training example; causal prefixes retained for runtime-equivalent evaluation; V1-V5 development; V6-V7 validation-only",
        "episodes": len(episodes),
        "stage1_label_counts": dict(sorted(stage1_counts.items())),
        "stage2_label_counts": dict(sorted(stage2_counts.items())),
        "ambiguous_stage2_episodes": sum(
            1
            for row in episodes
            if row["stage1_label"] == "airborne" and row.get("stage2_label") is None
        ),
        "features": list(FEATURES),
        "development_video_ids": sorted(video_ids),
        "clips": clip_summary,
    }
    return episodes, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AirborneClassifierV1.1 episode dataset from V1-V5 runs."
    )
    parser.add_argument("--manifest", default="evaluation/airborne_dev/manifest.json")
    parser.add_argument("--output", default="evaluation/airborne_dev/airborne_dataset.jsonl")
    parser.add_argument("--summary", default="evaluation/airborne_dev/airborne_dataset_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    summary_path = Path(args.summary)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path

    episodes, summary = build(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in episodes) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
