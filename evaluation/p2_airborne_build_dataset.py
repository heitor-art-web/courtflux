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

from services.p2.airborne_classifier_v1 import AirborneClassifierV1

FEATURES = AirborneClassifierV1.FEATURE_NAMES


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


def _truth_by_frame(gt: dict[str, Any]) -> dict[int, str]:
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


def _label(state: str) -> str:
    if state == "pass_flight":
        return "pass_flight"
    if state == "shot_flight":
        return "shot_flight"
    return "unresolved"


def _receiver_track(row: dict[str, Any]) -> int | None:
    candidate = _int(row.get("candidate_track_id"))
    if candidate is not None:
        return candidate
    raw = _int(row.get("raw_best_track_id"))
    origin = _int(row.get("flight_origin_track_id"))
    if raw is not None and raw != origin:
        return raw
    return None


def _features_from_existing_row(
    row: dict[str, Any],
    *,
    receiver_streak: int,
) -> dict[str, float]:
    delta_distance = _float(row.get("shot_preflight_delta_distance_norm"), 0.0)
    delta_relative_y = _float(row.get("shot_preflight_delta_relative_y"), 0.0)
    values = {
        "flight_supported": 1.0 if bool(row.get("flight_supported", False)) else 0.0,
        "flight_frames": _float(row.get("flight_frames"), 0.0),
        "flight_direct_observations": _float(row.get("flight_direct_observations"), 0.0),
        "flight_displacement_player_heights": _float(
            row.get("flight_displacement_player_heights"), 0.0
        ),
        "receiver_evidence": 1.0 if bool(row.get("pass_flight_receiver_evidence", False)) else 0.0,
        "receiver_thresholded": 1.0 if row.get("candidate_track_id") is not None else 0.0,
        "raw_best_distinct_from_origin": 1.0
        if bool(row.get("raw_best_distinct_from_flight_origin", False))
        else 0.0,
        "raw_best_score": _float(row.get("raw_best_score"), 0.0),
        "candidate_score": _float(row.get("candidate_score"), 0.0),
        "receiver_same_track_streak": float(receiver_streak),
        # These two receiver-geometry features were not persisted in the existing
        # V1-V5 runs. Keep them neutral; constant training columns produce zero weight.
        "receiver_approach_delta_heights": 0.0,
        "receiver_motion_cosine": 0.0,
        "shot_active": 1.0 if bool(row.get("shot_active", False)) else 0.0,
        # Semantic shot evidence was not persisted in the historical state rows. Strong
        # semantic shot evidence still has runtime precedence outside the classifier.
        "semantic_shot_signal": 0.0,
        "rim_available": 1.0 if bool(row.get("shot_preflight_available", False)) else 0.0,
        "rim_above": 1.0 if bool(row.get("shot_preflight_above", False)) else 0.0,
        "rim_distance_norm": _float(row.get("shot_preflight_ball_to_rim_distance_norm"), 0.0),
        # Runtime classifier expresses approach/rise as positive progress. Existing
        # diagnostics store current-minus-previous, so invert the sign here.
        "rim_approach_delta_norm": -delta_distance,
        "rim_rising_delta_heights": -delta_relative_y,
        "direct_ball_observed": 1.0
        if bool(row.get("shot_preflight_direct_observed", False))
        else 0.0,
        # Not persisted in the existing V1-V5 runs; neutral and therefore zero-weighted.
        "recent_owner_age_frames": 0.0,
    }
    return {name: float(values[name]) for name in FEATURES}


def build(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if str(manifest.get("version") or "") != "courtflux_airborne_dev_manifest_v2":
        raise ValueError("development manifest version must be courtflux_airborne_dev_manifest_v2")

    rows_out: list[dict[str, Any]] = []
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
            raise ValueError(
                f"{clip_id}: each development clip must use a distinct video for grouped CV"
            )
        video_ids.add(video_id)

        states_path = PROJECT_ROOT / "evaluation" / "runs" / run_id / "p2_states.jsonl"
        gt_path = PROJECT_ROOT / gt_rel
        if not states_path.exists():
            raise FileNotFoundError(f"{clip_id}: missing {states_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"{clip_id}: missing {gt_path}")

        gt = _load_json(gt_path)
        truth = _truth_by_frame(gt)
        state_rows = sorted(
            _load_jsonl(states_path), key=lambda row: int(row.get("frame_id", -1))
        )
        counts: Counter[str] = Counter()
        included = 0
        previous_receiver: int | None = None
        previous_frame: int | None = None
        receiver_streak = 0

        for state_row in state_rows:
            try:
                frame_id = int(state_row.get("frame_id", -1))
            except (TypeError, ValueError):
                continue

            receiver = _receiver_track(state_row)
            if (
                receiver is not None
                and receiver == previous_receiver
                and previous_frame is not None
                and frame_id == previous_frame + 1
            ):
                receiver_streak += 1
            elif receiver is not None:
                receiver_streak = 1
            else:
                receiver_streak = 0
            previous_receiver = receiver
            previous_frame = frame_id

            if frame_id not in truth:
                continue
            if not bool(state_row.get("engine_eligible", False)):
                continue
            if not bool(state_row.get("flight_active", False)):
                continue

            label = _label(truth[frame_id])
            rows_out.append(
                {
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "truth_state": truth[frame_id],
                    "label": label,
                    "features": _features_from_existing_row(
                        state_row, receiver_streak=receiver_streak
                    ),
                }
            )
            counts[label] += 1
            included += 1

        clip_summary.append(
            {
                "clip_id": clip_id,
                "video_id": video_id,
                "run_id": run_id,
                "rows": included,
                "labels": dict(sorted(counts.items())),
            }
        )

    if len(video_ids) < 3:
        raise ValueError("grouped cross-validation requires at least three development videos")
    if not rows_out:
        raise ValueError("development dataset contains no eligible airborne rows")

    label_counts = Counter(str(row["label"]) for row in rows_out)
    missing = [label for label in AirborneClassifierV1.CLASSES if label_counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"development dataset missing classifier classes: {missing}")

    summary = {
        "version": "courtflux_airborne_dataset_v2",
        "policy": "V1-V5 development corpus; grouped leave-one-video-out CV; V6-V7 reserved for final Gate A",
        "rows": len(rows_out),
        "features": list(FEATURES),
        "label_counts": dict(sorted(label_counts.items())),
        "development_video_ids": sorted(video_ids),
        "clips": clip_summary,
    }
    return rows_out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AirborneClassifierV1 dataset from existing V1-V5 state runs."
    )
    parser.add_argument("--manifest", default="evaluation/airborne_dev/manifest.json")
    parser.add_argument("--output", default="evaluation/airborne_dev/airborne_dataset.jsonl")
    parser.add_argument(
        "--summary", default="evaluation/airborne_dev/airborne_dataset_summary.json"
    )
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

    rows, summary = build(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
