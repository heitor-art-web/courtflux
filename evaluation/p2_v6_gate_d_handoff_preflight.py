from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_v6_gate_d_reid_preflight as base
from evaluation import p2_v6_gate_d_reid_sports_preflight as sports  # noqa: F401

DEFAULT_STRIDE_FRAMES = 8
DEFAULT_CONTEXT_SAMPLES = 4
DEFAULT_MIN_TRACK_SECONDS = 2.5
DEFAULT_MIN_ABS_DRIFT = 0.055
DEFAULT_MIN_ROBUST_Z = 3.5
DEFAULT_NMS_SECONDS = 0.55


def _load_gt(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected GT object: {path}")
    return payload


def _mapped_gt(gt: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in gt.get("mapped_observations") or []:
        if not isinstance(item, dict):
            continue
        endpoint = item.get("endpoint") or {}
        if not isinstance(endpoint, dict):
            continue
        if endpoint.get("track_id") is None or endpoint.get("frame_id") is None:
            continue
        rows.append(
            {
                "id": str(item.get("id") or ""),
                "human_player": str(item.get("human_player") or ""),
                "human_source_sec": float(item.get("human_source_sec") or 0.0),
                "track_id": int(endpoint["track_id"]),
                "frame_id": int(endpoint["frame_id"]),
                "timestamp_sec": float(endpoint.get("timestamp_sec") or item.get("human_source_sec") or 0.0),
            }
        )
    return rows


def _identity_intervals(gt_rows: list[dict[str, Any]], *, changed: bool) -> list[dict[str, Any]]:
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        by_track[int(row["track_id"])].append(row)
    result: list[dict[str, Any]] = []
    for track_id, rows in by_track.items():
        rows.sort(key=lambda row: float(row["timestamp_sec"]))
        for left, right in zip(rows, rows[1:]):
            is_changed = left["human_player"] != right["human_player"]
            if is_changed != bool(changed):
                continue
            result.append(
                {
                    "track_id": int(track_id),
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "left_player": left["human_player"],
                    "right_player": right["human_player"],
                    "start_sec": float(left["timestamp_sec"]),
                    "end_sec": float(right["timestamp_sec"]),
                }
            )
    return result


def _bbox_quality(box: list[float], confidence: float) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = height / width
    area_ratio = width * height / float(1920 * 1080)
    quality = float(max(0.0, min(1.0, confidence)))
    if aspect < 1.05:
        quality *= max(0.05, aspect / 1.05)
    if aspect > 5.5:
        quality *= max(0.05, 5.5 / aspect)
    if area_ratio > 0.16:
        quality *= max(0.05, 0.16 / area_ratio)
    if height < 90.0:
        quality *= max(0.05, height / 90.0)
    return float(quality)


def _collect_raw_tracks(states_path: Path) -> dict[int, list[dict[str, Any]]]:
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
            frame_id = int(row.get("frame_id", -1))
            timestamp_sec = row.get("timestamp_sec")
            if frame_id < 0 or timestamp_sec is None:
                continue
            for prediction in row.get("team_predictions") or []:
                if not isinstance(prediction, dict):
                    continue
                track_id = prediction.get("track_id")
                box = prediction.get("box")
                if track_id is None or not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                box_f = [float(v) for v in box]
                confidence = float(prediction.get("confidence") or 0.0)
                tracks[int(track_id)].append(
                    {
                        "frame_id": frame_id,
                        "timestamp_sec": float(timestamp_sec),
                        "box": box_f,
                        "quality": _bbox_quality(box_f, confidence),
                    }
                )
    return tracks


def _risk_tracks(
    tracks: dict[int, list[dict[str, Any]]],
    gt_track_ids: set[int],
    *,
    min_seconds: float,
) -> list[int]:
    selected: set[int] = set(gt_track_ids)
    for track_id, rows in tracks.items():
        if len(rows) < 2:
            continue
        duration = float(rows[-1]["timestamp_sec"]) - float(rows[0]["timestamp_sec"])
        if duration >= float(min_seconds):
            selected.add(int(track_id))
    return sorted(selected)


def _sample_track(rows: list[dict[str, Any]], *, stride_frames: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: int(row["frame_id"]))
    selected: list[dict[str, Any]] = []
    last_frame = -10**9
    for row in rows:
        frame_id = int(row["frame_id"])
        if frame_id - last_frame < int(stride_frames):
            if selected and float(row["quality"]) > float(selected[-1]["quality"]) + 0.20:
                selected[-1] = row
                last_frame = frame_id
            continue
        if float(row["quality"]) < 0.20:
            continue
        selected.append(row)
        last_frame = frame_id
    return selected


def _read_full_body(capture: cv2.VideoCapture, sample: dict[str, Any]) -> np.ndarray:
    timestamp_sec = float(sample["timestamp_sec"])
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"Could not decode video at {timestamp_sec:.3f}s")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in sample["box"]]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    ix1 = max(0, min(width - 1, int(math.floor(x1 - 0.02 * bw))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2 + 0.02 * bw))))
    iy1 = max(0, min(height - 1, int(math.floor(y1 - 0.01 * bh))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2 + 0.01 * bh))))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop at {timestamp_sec:.3f}s")
    return crop


def _embed_samples(
    model: torch.nn.Module,
    device: torch.device,
    video_path: Path,
    sampled: dict[int, list[dict[str, Any]]],
    *,
    batch_size: int,
) -> dict[int, list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    metadata: list[tuple[int, dict[str, Any]]] = []
    tensors: list[torch.Tensor] = []
    try:
        for track_id in sorted(sampled):
            for sample in sampled[track_id]:
                tensors.append(base._preprocess(_read_full_body(capture, sample)))
                metadata.append((track_id, sample))
    finally:
        capture.release()
    if not tensors:
        return {}
    matrix = base._embed_batches(model, tensors, device=device, batch_size=int(batch_size))
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (track_id, sample), embedding in zip(metadata, matrix):
        row = dict(sample)
        row["embedding"] = np.asarray(embedding, dtype=np.float32)
        result[int(track_id)].append(row)
    return result


def _unit_mean(vectors: list[np.ndarray]) -> np.ndarray:
    vector = np.asarray(vectors, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _local_drift(rows: list[dict[str, Any]], *, context_samples: int) -> list[dict[str, Any]]:
    k = int(context_samples)
    if len(rows) < 2 * k + 1:
        return []
    drifts: list[dict[str, Any]] = []
    for split in range(k, len(rows) - k + 1):
        left = rows[split - k : split]
        right = rows[split : split + k]
        if len(left) < k or len(right) < k:
            continue
        left_center = _unit_mean([row["embedding"] for row in left])
        right_center = _unit_mean([row["embedding"] for row in right])
        cosine = float(np.dot(left_center, right_center))
        drifts.append(
            {
                "timestamp_sec": 0.5 * (float(left[-1]["timestamp_sec"]) + float(right[0]["timestamp_sec"])),
                "frame_left": int(left[-1]["frame_id"]),
                "frame_right": int(right[0]["frame_id"]),
                "cosine": cosine,
                "drift": float(1.0 - cosine),
            }
        )
    return drifts


def _robust_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median, max(1e-6, 1.4826 * mad)


def _flag_changes(
    by_track: dict[int, list[dict[str, Any]]],
    *,
    min_abs_drift: float,
    min_robust_z: float,
    nms_seconds: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, float]]:
    pooled = [float(row["drift"]) for rows in by_track.values() for row in rows]
    global_median, global_sigma = _robust_stats(pooled)
    flagged: dict[int, list[dict[str, Any]]] = {}
    for track_id, rows in by_track.items():
        local_median, local_sigma = _robust_stats([float(row["drift"]) for row in rows])
        candidates: list[dict[str, Any]] = []
        for row in rows:
            drift = float(row["drift"])
            z_local = (drift - local_median) / local_sigma
            z_global = (drift - global_median) / global_sigma
            robust_z = min(z_local, z_global)
            if drift < float(min_abs_drift) or robust_z < float(min_robust_z):
                continue
            item = dict(row)
            item["robust_z"] = float(robust_z)
            candidates.append(item)
        candidates.sort(key=lambda row: float(row["drift"]), reverse=True)
        kept: list[dict[str, Any]] = []
        for row in candidates:
            if any(
                abs(float(row["timestamp_sec"]) - float(prev["timestamp_sec"])) < float(nms_seconds)
                for prev in kept
            ):
                continue
            kept.append(row)
        flagged[int(track_id)] = sorted(kept, key=lambda row: float(row["timestamp_sec"]))
    return flagged, {
        "global_drift_median": global_median,
        "global_drift_robust_sigma": global_sigma,
    }


def _evaluate_intervals(
    intervals: list[dict[str, Any]],
    flagged: dict[int, list[dict[str, Any]]],
    *,
    padding_sec: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for interval in intervals:
        start = float(interval["start_sec"]) - float(padding_sec)
        end = float(interval["end_sec"]) + float(padding_sec)
        hits = [
            row
            for row in flagged.get(int(interval["track_id"]), [])
            if start <= float(row["timestamp_sec"]) <= end
        ]
        item = dict(interval)
        item["detected"] = bool(hits)
        item["detections"] = [
            {
                "timestamp_sec": round(float(row["timestamp_sec"]), 6),
                "drift": round(float(row["drift"]), 6),
                "robust_z": round(float(row["robust_z"]), 3),
            }
            for row in hits
        ]
        result.append(item)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt_path = Path(args.gt).expanduser().resolve()
    states_path = Path(args.states).expanduser().resolve() if args.states else base._latest_states_file()
    video_path = Path(args.video).expanduser().resolve()
    gt_rows = _mapped_gt(_load_gt(gt_path))
    handoff_gt = _identity_intervals(gt_rows, changed=True)
    same_gt = _identity_intervals(gt_rows, changed=False)
    gt_track_ids = {int(row["track_id"]) for row in gt_rows}

    print("[Handoff preflight] 1/6 scanning raw tracks", flush=True)
    tracks = _collect_raw_tracks(states_path)
    selected_tracks = _risk_tracks(tracks, gt_track_ids, min_seconds=float(args.min_track_seconds))
    sampled = {
        track_id: _sample_track(tracks[track_id], stride_frames=int(args.stride_frames))
        for track_id in selected_tracks
        if track_id in tracks
    }
    minimum = 2 * int(args.context_samples) + 1
    sampled = {track_id: rows for track_id, rows in sampled.items() if len(rows) >= minimum}

    print(
        f"[Handoff preflight] 2/6 loading SportsReID ({len(sampled)} tracks, "
        f"{sum(len(v) for v in sampled.values())} crops)",
        flush=True,
    )
    device = base._device(args.device)
    model, model_meta = base._build_model(device, "sportsreid_osnet_x1_0_soccernet")

    print("[Handoff preflight] 3/6 embedding local track samples", flush=True)
    embedded = _embed_samples(model, device, video_path, sampled, batch_size=int(args.batch_size))

    print("[Handoff preflight] 4/6 computing label-free local drift", flush=True)
    drift_by_track = {
        track_id: _local_drift(rows, context_samples=int(args.context_samples))
        for track_id, rows in embedded.items()
    }
    drift_by_track = {track_id: rows for track_id, rows in drift_by_track.items() if rows}
    flagged, stats = _flag_changes(
        drift_by_track,
        min_abs_drift=float(args.min_abs_drift),
        min_robust_z=float(args.min_robust_z),
        nms_seconds=float(args.nms_seconds),
    )

    print("[Handoff preflight] 5/6 evaluating frozen human intervals", flush=True)
    handoff_eval = _evaluate_intervals(handoff_gt, flagged, padding_sec=float(args.gt_padding_sec))
    same_eval = _evaluate_intervals(same_gt, flagged, padding_sec=0.0)
    handoff_hits = sum(bool(row["detected"]) for row in handoff_eval)
    same_false = sum(bool(row["detected"]) for row in same_eval)
    handoff_recall = handoff_hits / len(handoff_eval) if handoff_eval else 0.0
    same_false_rate = same_false / len(same_eval) if same_eval else 0.0
    total_flags = sum(len(rows) for rows in flagged.values())

    report = {
        "version": "courtflux_gate_d_v6_local_handoff_preflight_v1",
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "anti_leakage": {
            "human_identity_used_to_compute_drift_or_threshold": False,
            "game_player_id_used": False,
            "human_gt_used_only_to_evaluate_detected_change_points": True,
        },
        "sources": {"ground_truth": str(gt_path), "states": str(states_path), "video": str(video_path)},
        "model": model_meta,
        "policy": {
            "stride_frames": int(args.stride_frames),
            "context_samples": int(args.context_samples),
            "min_track_seconds": float(args.min_track_seconds),
            "min_abs_drift": float(args.min_abs_drift),
            "min_robust_z": float(args.min_robust_z),
            "nms_seconds": float(args.nms_seconds),
            "thresholding": "pooled + per-track median/MAD; labels excluded",
        },
        "drift_stats": stats,
        "selected_track_count": len(sampled),
        "embedded_crop_count": sum(len(v) for v in embedded.values()),
        "flagged_change_count": total_flags,
        "human_handoff_intervals": handoff_eval,
        "human_same_identity_same_track_intervals": same_eval,
        "metrics": {
            "handoff_intervals": len(handoff_eval),
            "handoff_detected": handoff_hits,
            "handoff_recall": handoff_recall,
            "same_identity_intervals": len(same_eval),
            "same_identity_false_alarm_intervals": same_false,
            "same_identity_false_alarm_rate": same_false_rate,
        },
        "flagged_by_gt_track": {
            str(track_id): [
                {
                    "timestamp_sec": round(float(row["timestamp_sec"]), 6),
                    "drift": round(float(row["drift"]), 6),
                    "robust_z": round(float(row["robust_z"]), 3),
                }
                for row in flagged.get(track_id, [])
            ]
            for track_id in sorted(gt_track_ids)
        },
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else states_path.parent / "p2_v6_gate_d_handoff_preflight.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("[Handoff preflight] 6/6 report", flush=True)
    print("\nV6 Gate D local-handoff preflight", flush=True)
    print(f"human handoff recall: {handoff_hits}/{len(handoff_eval)} = {100.0 * handoff_recall:.2f}%", flush=True)
    print(
        f"same-identity interval false alarms: {same_false}/{len(same_eval)} = "
        f"{100.0 * same_false_rate:.2f}%",
        flush=True,
    )
    print(f"label-free change points flagged: {total_flags}", flush=True)
    for row in handoff_eval:
        print(
            f"handoff track {row['track_id']} {row['left_player']}->{row['right_player']} "
            f"[{row['start_sec']:.3f},{row['end_sec']:.3f}] detected={row['detected']} "
            f"hits={row['detections']}",
            flush=True,
        )
    print(f"report: {output_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V6 Gate-D local tracker-handoff diagnostic. SportsReID is used only as a "
            "short-horizon change detector inside raw tracks; human identities are evaluation-only."
        )
    )
    parser.add_argument("--gt", default=str(base.DEFAULT_GT))
    parser.add_argument("--states", default=None)
    parser.add_argument("--video", default=str(base.DEFAULT_VIDEO))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--stride-frames", type=int, default=DEFAULT_STRIDE_FRAMES)
    parser.add_argument("--context-samples", type=int, default=DEFAULT_CONTEXT_SAMPLES)
    parser.add_argument("--min-track-seconds", type=float, default=DEFAULT_MIN_TRACK_SECONDS)
    parser.add_argument("--min-abs-drift", type=float, default=DEFAULT_MIN_ABS_DRIFT)
    parser.add_argument("--min-robust-z", type=float, default=DEFAULT_MIN_ROBUST_Z)
    parser.add_argument("--nms-seconds", type=float, default=DEFAULT_NMS_SECONDS)
    parser.add_argument("--gt-padding-sec", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.stride_frames < 1 or args.context_samples < 2 or args.batch_size < 1:
        raise SystemExit("Invalid stride/context/batch parameters")
    run(args)


if __name__ == "__main__":
    main()
