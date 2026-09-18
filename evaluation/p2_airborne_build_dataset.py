from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FIELDS = (
    "flight_supported",
    "flight_frames",
    "flight_direct_observations",
    "flight_displacement_player_heights",
    "pass_flight_receiver_evidence",
    "raw_best_distinct_from_flight_origin",
    "raw_best_score",
    "candidate_score",
    "shot_active",
    "shot_preflight_available",
    "shot_preflight_direct_observed",
    "shot_preflight_near_x",
    "shot_preflight_above",
    "shot_preflight_ball_to_rim_distance_norm",
    "shot_preflight_horizontal_distance_rim_widths",
    "shot_preflight_relative_y_rim_heights",
    "shot_preflight_delta_distance_norm",
    "shot_preflight_delta_relative_y",
    "shot_preflight_approach_streak",
    "shot_preflight_rising_streak",
)

MODEL_FEATURE_NAMES = (
    "flight_supported",
    "flight_frames",
    "flight_direct_observations",
    "flight_displacement_player_heights",
    "receiver_evidence",
    "raw_best_distinct_from_origin",
    "raw_best_score",
    "candidate_score",
    "shot_active",
    "shot_preflight_available",
    "shot_preflight_direct_observed",
    "shot_preflight_near_x",
    "shot_preflight_above",
    "shot_preflight_ball_to_rim_distance_norm",
    "shot_preflight_horizontal_distance_rim_widths",
    "shot_preflight_relative_y_rim_heights",
    "shot_preflight_delta_distance_norm",
    "shot_preflight_delta_relative_y",
    "shot_preflight_approach_streak",
    "shot_preflight_rising_streak",
)

FIELD_MAP = {
    "receiver_evidence": "pass_flight_receiver_evidence",
    "raw_best_distinct_from_origin": "raw_best_distinct_from_flight_origin",
}


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
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _validation_video_ids() -> set[str]:
    manifest = _load_json(PROJECT_ROOT / "evaluation" / "ground_truth" / "manifest.json")
    return {
        str(row.get("video_id"))
        for row in list(manifest.get("clips", []) or [])
        if isinstance(row, dict) and str(row.get("id") or "").upper() in {"V1", "V2", "V3", "V4", "V5"}
    }


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


def _feature_value(row: dict[str, Any], model_name: str) -> float:
    source_name = FIELD_MAP.get(model_name, model_name)
    value = row.get(source_name)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if str(manifest.get("version") or "") != "courtflux_airborne_dev_manifest_v1":
        raise ValueError("development manifest version must be courtflux_airborne_dev_manifest_v1")

    forbidden = _validation_video_ids()
    rows_out: list[dict[str, Any]] = []
    clip_summary: list[dict[str, Any]] = []
    train_video_ids: set[str] = set()
    calibration_video_ids: set[str] = set()

    for clip in list(manifest.get("clips", []) or []):
        if not isinstance(clip, dict):
            continue
        clip_id = str(clip.get("id") or "")
        split = str(clip.get("split") or "")
        video_id = str(clip.get("video_id") or "")
        run_id = clip.get("run_id")
        gt_rel = str(clip.get("gt_file") or "")
        if split not in {"train", "calibration"}:
            raise ValueError(f"{clip_id}: split must be train or calibration")
        if video_id in forbidden:
            raise ValueError(f"{clip_id}: validation video {video_id} is forbidden in development data")
        if not run_id:
            raise ValueError(f"{clip_id}: run_id is required after the development probe is generated")
        if not gt_rel:
            raise ValueError(f"{clip_id}: gt_file is required")

        (train_video_ids if split == "train" else calibration_video_ids).add(video_id)
        run_path = PROJECT_ROOT / "evaluation" / "runs" / str(run_id) / "p2_states.jsonl"
        gt_path = PROJECT_ROOT / gt_rel
        if not run_path.exists():
            raise FileNotFoundError(f"{clip_id}: missing {run_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"{clip_id}: missing human GT {gt_path}")

        gt = _load_json(gt_path)
        truth = _truth_by_frame(gt)
        state_rows = _load_jsonl(run_path)
        counts = {"unresolved": 0, "pass_flight": 0, "shot_flight": 0}
        included = 0
        for state_row in state_rows:
            try:
                frame_id = int(state_row.get("frame_id", -1))
            except (TypeError, ValueError):
                continue
            if frame_id not in truth:
                continue
            if not bool(state_row.get("flight_active", False)):
                continue
            if not bool(state_row.get("engine_eligible", False)):
                continue
            label = _label(truth[frame_id])
            feature_values = {
                name: _feature_value(state_row, name)
                for name in MODEL_FEATURE_NAMES
            }
            rows_out.append(
                {
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "split": split,
                    "run_id": str(run_id),
                    "frame_id": frame_id,
                    "truth_state": truth[frame_id],
                    "label": label,
                    "features": feature_values,
                }
            )
            counts[label] += 1
            included += 1
        clip_summary.append(
            {
                "clip_id": clip_id,
                "split": split,
                "video_id": video_id,
                "run_id": str(run_id),
                "rows": included,
                "labels": counts,
            }
        )

    overlap = train_video_ids & calibration_video_ids
    if overlap:
        raise ValueError(f"train/calibration must be video-disjoint; overlap={sorted(overlap)}")
    if not rows_out:
        raise ValueError("development dataset contains no eligible airborne rows")

    summary = {
        "version": "courtflux_airborne_dataset_v1",
        "rows": len(rows_out),
        "features": list(MODEL_FEATURE_NAMES),
        "clips": clip_summary,
        "train_video_ids": sorted(train_video_ids),
        "calibration_video_ids": sorted(calibration_video_ids),
        "forbidden_validation_video_ids": sorted(forbidden),
    }
    return rows_out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AirborneClassifierV1 development dataset.")
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

    rows, summary = build(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
