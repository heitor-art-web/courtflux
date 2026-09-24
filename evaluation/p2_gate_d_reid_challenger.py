from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class EndpointCrop:
    observation_id: str
    endpoint_index: int
    frame_id: int
    timestamp_sec: float
    track_id: int
    episode_index: int | None
    box: tuple[int, int, int, int]
    crop: np.ndarray


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(tuple(left), dtype=np.float32)
    b = np.asarray(tuple(right), dtype=np.float32)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("embeddings must be non-empty 1D vectors with equal shape")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        raise ValueError("embedding norm must be non-zero")
    return float(np.dot(a, b) / denom)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_states(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict) and row.get("frame_id") is not None:
                rows[int(row["frame_id"])] = row
    return rows


def _prediction_for_endpoint(state: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    track_id = int(endpoint["track_id"])
    episode_index = endpoint.get("episode_index")
    candidates = [
        row
        for row in list(state.get("game_player_predictions", []) or [])
        if isinstance(row, dict) and int(row.get("track_id", -1) or -1) == track_id
    ]
    if episode_index is not None:
        candidates = [
            row
            for row in candidates
            if row.get("episode_index") is not None
            and int(row["episode_index"]) == int(episode_index)
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one prediction for frame={endpoint['frame_id']} "
            f"track={track_id} episode={episode_index}; got {len(candidates)}"
        )
    return candidates[0]


def _clip_box(box: Iterable[Any], *, width: int, height: int) -> tuple[int, int, int, int]:
    values = list(box)
    if len(values) != 4:
        raise ValueError(f"Expected xyxy box with four values, got {values}")
    x1, y1, x2, y2 = (int(round(float(value))) for value in values)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _read_frame_at_timestamp(capture: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode video at {timestamp_sec:.6f}s")
    return frame


def collect_endpoint_crops(
    *,
    clip: str,
    video_path: Path,
    states_path: Path,
    observations_path: Path,
) -> list[tuple[dict[str, Any], list[EndpointCrop]]]:
    observations_payload = _load_json(observations_path)
    states = _load_states(states_path)
    observations = [
        row
        for row in list(observations_payload.get("observations", []) or [])
        if isinstance(row, dict) and str(row.get("clip")) == clip
    ]
    if not observations:
        raise ValueError(f"No frozen Gate-D observations for clip {clip}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        pairs: list[tuple[dict[str, Any], list[EndpointCrop]]] = []
        for observation in observations:
            endpoints = observation.get("endpoints")
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise ValueError(f"Observation {observation.get('id')} is not fully mapped")
            crops: list[EndpointCrop] = []
            for endpoint_index, endpoint in enumerate(endpoints):
                frame_id = int(endpoint["frame_id"])
                state = states.get(frame_id)
                if state is None:
                    raise ValueError(f"Missing state frame {frame_id} for {observation.get('id')}")
                prediction = _prediction_for_endpoint(state, endpoint)
                timestamp_sec = float(state["timestamp_sec"])
                frame = _read_frame_at_timestamp(capture, timestamp_sec)
                height, width = frame.shape[:2]
                box = _clip_box(prediction.get("box", []), width=width, height=height)
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise ValueError(f"Empty crop for {observation.get('id')} endpoint {endpoint_index}")
                crops.append(
                    EndpointCrop(
                        observation_id=str(observation.get("id")),
                        endpoint_index=endpoint_index,
                        frame_id=frame_id,
                        timestamp_sec=timestamp_sec,
                        track_id=int(endpoint["track_id"]),
                        episode_index=(
                            None
                            if endpoint.get("episode_index") is None
                            else int(endpoint["episode_index"])
                        ),
                        box=box,
                        crop=crop,
                    )
                )
            pairs.append((observation, crops))
        return pairs
    finally:
        capture.release()


def build_osnet_extractor(*, model_path: Path, device: str) -> Callable[[list[np.ndarray]], np.ndarray]:
    try:
        from torchreid.utils import FeatureExtractor
    except Exception as exc:  # pragma: no cover - depends on optional research package
        raise RuntimeError(
            "Torchreid is required for the OSNet challenger. Install KaiyangZhou/deep-person-reid "
            "in the current environment, then rerun this script."
        ) from exc

    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=str(model_path),
        device=device,
        verbose=False,
    )

    def _extract(crops_bgr: list[np.ndarray]) -> np.ndarray:
        # Torchreid accepts numpy arrays in RGB order.
        crops_rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops_bgr]
        features = extractor(crops_rgb)
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(crops_bgr):
            raise RuntimeError(f"Unexpected OSNet feature shape: {array.shape}")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return array / norms

    return _extract


def evaluate_pairs(
    pairs: list[tuple[dict[str, Any], list[EndpointCrop]]],
    *,
    extractor: Callable[[list[np.ndarray]], np.ndarray],
) -> dict[str, Any]:
    crops = [endpoint.crop for _, endpoints in pairs for endpoint in endpoints]
    features = extractor(crops)
    cursor = 0
    rows: list[dict[str, Any]] = []
    for observation, endpoints in pairs:
        left_feature = features[cursor]
        right_feature = features[cursor + 1]
        cursor += 2
        similarity = cosine_similarity(left_feature, right_feature)
        rows.append(
            {
                "id": observation.get("id"),
                "clip": observation.get("clip"),
                "label": observation.get("label"),
                "osnet_cosine_similarity": round(similarity, 6),
                "endpoints": [
                    {
                        "frame_id": endpoint.frame_id,
                        "timestamp_sec": round(endpoint.timestamp_sec, 6),
                        "track_id": endpoint.track_id,
                        "episode_index": endpoint.episode_index,
                        "box": list(endpoint.box),
                    }
                    for endpoint in endpoints
                ],
            }
        )

    same_scores = [row["osnet_cosine_similarity"] for row in rows if row.get("label") == "SAME"]
    different_scores = [
        row["osnet_cosine_similarity"] for row in rows if row.get("label") == "DIFFERENT"
    ]
    return {
        "version": "courtflux_gate_d_reid_challenger_v1",
        "model": "osnet_x1_0",
        "pairs": rows,
        "same_similarity_mean": (
            None if not same_scores else round(float(np.mean(same_scores)), 6)
        ),
        "different_similarity_mean": (
            None if not different_scores else round(float(np.mean(different_scores)), 6)
        ),
        "same_similarity_min": None if not same_scores else round(min(same_scores), 6),
        "different_similarity_max": (
            None if not different_scores else round(max(different_scores), 6)
        ),
        "separation_margin": (
            None
            if not same_scores or not different_scores
            else round(min(same_scores) - max(different_scores), 6)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OSNet person-ReID challenger on frozen Gate-D human pairs."
    )
    parser.add_argument("--clip", required=True, choices=("v1", "v3", "v4", "v5"))
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("evaluation/ground_truth/gate_d_human_observations_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    pairs = collect_endpoint_crops(
        clip=args.clip,
        video_path=args.video,
        states_path=args.states,
        observations_path=args.observations,
    )
    extractor = build_osnet_extractor(model_path=args.model_path, device=args.device)
    result = evaluate_pairs(pairs, extractor=extractor)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
