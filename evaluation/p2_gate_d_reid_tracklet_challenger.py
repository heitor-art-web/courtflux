from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluation.p2_gate_d_reid_challenger import (
    build_osnet_extractor,
    cosine_similarity,
    _clip_box,
    _load_json,
    _load_states,
    _read_frame_at_timestamp,
)


@dataclass(frozen=True, slots=True)
class TrackletSample:
    frame_id: int
    timestamp_sec: float
    track_id: int
    episode_index: int | None
    box: tuple[int, int, int, int]
    max_iou: float
    quality: float
    crop: np.ndarray


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = float(iw * ih)
    if intersection <= 0.0:
        return 0.0
    left_area = float(max(0, lx2 - lx1) * max(0, ly2 - ly1))
    right_area = float(max(0, rx2 - rx1) * max(0, ry2 - ry1))
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _matching_prediction(
    state: dict[str, Any],
    *,
    track_id: int,
    episode_index: int | None,
) -> dict[str, Any] | None:
    matches = []
    for row in list(state.get("game_player_predictions", []) or []):
        if not isinstance(row, dict):
            continue
        if int(row.get("track_id", -1) or -1) != track_id:
            continue
        if episode_index is not None:
            if row.get("episode_index") is None or int(row["episode_index"]) != episode_index:
                continue
        matches.append(row)
    return matches[0] if len(matches) == 1 else None


def _sample_quality(
    box: tuple[int, int, int, int],
    *,
    max_iou: float,
) -> float:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = float(width * height)
    aspect = float(height) / float(width)
    aspect_penalty = abs(math.log(max(aspect, 1e-6) / 2.2))
    return math.log1p(area) - 2.0 * max_iou - 0.25 * aspect_penalty


def collect_tracklet_samples(
    *,
    capture: cv2.VideoCapture,
    states: dict[int, dict[str, Any]],
    endpoint: dict[str, Any],
    window_sec: float,
    max_samples: int,
    min_spacing_sec: float,
) -> list[TrackletSample]:
    endpoint_frame = int(endpoint["frame_id"])
    endpoint_state = states.get(endpoint_frame)
    if endpoint_state is None:
        raise ValueError(f"Missing state frame {endpoint_frame}")
    endpoint_ts = float(endpoint_state["timestamp_sec"])
    track_id = int(endpoint["track_id"])
    episode_index = endpoint.get("episode_index")
    episode_index = None if episode_index is None else int(episode_index)

    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    if width <= 0 or height <= 0:
        probe = _read_frame_at_timestamp(capture, endpoint_ts)
        height, width = probe.shape[:2]

    candidates: list[tuple[float, int, float, tuple[int, int, int, int], float]] = []
    for frame_id, state in states.items():
        timestamp_sec = float(state.get("timestamp_sec", -1.0))
        if abs(timestamp_sec - endpoint_ts) > window_sec:
            continue
        prediction = _matching_prediction(
            state,
            track_id=track_id,
            episode_index=episode_index,
        )
        if prediction is None:
            continue
        box = _clip_box(prediction.get("box", []), width=width, height=height)
        other_boxes: list[tuple[int, int, int, int]] = []
        for other in list(state.get("game_player_predictions", []) or []):
            if not isinstance(other, dict) or other is prediction:
                continue
            try:
                other_boxes.append(_clip_box(other.get("box", []), width=width, height=height))
            except ValueError:
                continue
        max_iou = max((_iou(box, other_box) for other_box in other_boxes), default=0.0)
        quality = _sample_quality(box, max_iou=max_iou)
        candidates.append((quality, frame_id, timestamp_sec, box, max_iou))

    if not candidates:
        raise ValueError(
            f"No tracklet samples for frame={endpoint_frame} track={track_id} episode={episode_index}"
        )

    # Prefer large, isolated, person-shaped crops, while retaining temporal diversity.
    candidates.sort(key=lambda item: (item[0], -abs(item[2] - endpoint_ts)), reverse=True)
    selected: list[tuple[float, int, float, tuple[int, int, int, int], float]] = []
    for candidate in candidates:
        timestamp_sec = candidate[2]
        if selected and any(abs(timestamp_sec - row[2]) < min_spacing_sec for row in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_samples:
            break
    if not selected:
        selected.append(candidates[0])

    samples: list[TrackletSample] = []
    for quality, frame_id, timestamp_sec, box, max_iou in sorted(selected, key=lambda item: item[2]):
        frame = _read_frame_at_timestamp(capture, timestamp_sec)
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        samples.append(
            TrackletSample(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                track_id=track_id,
                episode_index=episode_index,
                box=box,
                max_iou=max_iou,
                quality=quality,
                crop=crop,
            )
        )
    if not samples:
        raise ValueError(
            f"All selected tracklet crops were empty for frame={endpoint_frame} track={track_id}"
        )
    return samples


def aggregate_tracklet_features(features: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    array = array / np.maximum(norms, 1e-12)
    centroid = array.mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    coherence = float(np.mean(array @ centroid))
    return centroid.astype(np.float32, copy=False), coherence


def tracklet_pair_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("tracklet features must be 2D")
    left_norm = left_array / np.maximum(np.linalg.norm(left_array, axis=1, keepdims=True), 1e-12)
    right_norm = right_array / np.maximum(np.linalg.norm(right_array, axis=1, keepdims=True), 1e-12)
    left_centroid, left_coherence = aggregate_tracklet_features(left_norm)
    right_centroid, right_coherence = aggregate_tracklet_features(right_norm)
    pairwise = left_norm @ right_norm.T
    flat = np.sort(pairwise.reshape(-1))[::-1]
    top_count = max(1, int(math.ceil(float(flat.size) * 0.25)))
    return {
        "centroid_cosine_similarity": round(cosine_similarity(left_centroid, right_centroid), 6),
        "pairwise_top_quartile_mean": round(float(np.mean(flat[:top_count])), 6),
        "pairwise_median": round(float(np.median(flat)), 6),
        "pairwise_max": round(float(np.max(flat)), 6),
        "left_coherence": round(left_coherence, 6),
        "right_coherence": round(right_coherence, 6),
    }


def evaluate_clip(
    *,
    clip: str,
    video_path: Path,
    states_path: Path,
    observations_path: Path,
    model_path: Path,
    device: str,
    window_sec: float,
    max_samples: int,
) -> dict[str, Any]:
    observations_payload = _load_json(observations_path)
    states = _load_states(states_path)
    observations = [
        row
        for row in list(observations_payload.get("observations", []) or [])
        if isinstance(row, dict) and str(row.get("clip")) == clip
    ]
    if not observations:
        raise ValueError(f"No frozen Gate-D observations for clip {clip}")

    extractor = build_osnet_extractor(model_path=model_path, device=device)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    rows: list[dict[str, Any]] = []
    min_spacing_sec = max(0.04, window_sec / max(max_samples * 2, 1))
    try:
        for observation in observations:
            endpoints = observation.get("endpoints")
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise ValueError(f"Observation {observation.get('id')} is not fully mapped")
            endpoint_samples = [
                collect_tracklet_samples(
                    capture=capture,
                    states=states,
                    endpoint=endpoint,
                    window_sec=window_sec,
                    max_samples=max_samples,
                    min_spacing_sec=min_spacing_sec,
                )
                for endpoint in endpoints
            ]
            endpoint_features = [extractor([sample.crop for sample in samples]) for samples in endpoint_samples]
            metrics = tracklet_pair_metrics(endpoint_features[0], endpoint_features[1])

            left_state = states[int(endpoints[0]["frame_id"])]
            right_state = states[int(endpoints[1]["frame_id"])]
            gap_sec = abs(float(right_state["timestamp_sec"]) - float(left_state["timestamp_sec"]))
            cross_track = int(endpoints[0]["track_id"]) != int(endpoints[1]["track_id"])
            positive_reid_eligible = bool(cross_track and gap_sec >= 0.5)

            rows.append(
                {
                    "id": observation.get("id"),
                    "clip": observation.get("clip"),
                    "label": observation.get("label"),
                    "cross_track": cross_track,
                    "gap_sec": round(gap_sec, 6),
                    "positive_reid_eligible": positive_reid_eligible,
                    **metrics,
                    "endpoints": [
                        {
                            "frame_id": int(endpoint["frame_id"]),
                            "track_id": int(endpoint["track_id"]),
                            "episode_index": endpoint.get("episode_index"),
                            "sample_count": len(samples),
                            "samples": [
                                {
                                    "frame_id": sample.frame_id,
                                    "timestamp_sec": round(sample.timestamp_sec, 6),
                                    "box": list(sample.box),
                                    "max_iou": round(sample.max_iou, 6),
                                    "quality": round(sample.quality, 6),
                                }
                                for sample in samples
                            ],
                        }
                        for endpoint, samples in zip(endpoints, endpoint_samples)
                    ],
                }
            )
    finally:
        capture.release()

    eligible_same = [
        float(row["centroid_cosine_similarity"])
        for row in rows
        if row.get("positive_reid_eligible") and row.get("label") == "SAME"
    ]
    eligible_different = [
        float(row["centroid_cosine_similarity"])
        for row in rows
        if row.get("positive_reid_eligible") and row.get("label") == "DIFFERENT"
    ]
    return {
        "version": "courtflux_gate_d_reid_tracklet_challenger_v1",
        "model": "osnet_x1_0",
        "window_sec": window_sec,
        "max_samples_per_endpoint": max_samples,
        "pairs": rows,
        "eligible_same_similarity_min": None if not eligible_same else round(min(eligible_same), 6),
        "eligible_different_similarity_max": (
            None if not eligible_different else round(max(eligible_different), 6)
        ),
        "eligible_separation_margin": (
            None
            if not eligible_same or not eligible_different
            else round(min(eligible_same) - max(eligible_different), 6)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tracklet-aggregated OSNet evidence on frozen Gate-D human pairs. "
            "This is a challenger only and does not mutate game_player_id decisions."
        )
    )
    parser.add_argument("--clip", required=True, choices=("v1", "v3", "v4", "v5"))
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("evaluation/ground_truth/gate_d_human_observations_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_sec <= 0.0:
        raise ValueError("--window-sec must be > 0")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    result = evaluate_clip(
        clip=args.clip,
        video_path=args.video,
        states_path=args.states,
        observations_path=args.observations,
        model_path=args.model_path,
        device=args.device,
        window_sec=float(args.window_sec),
        max_samples=int(args.max_samples),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
