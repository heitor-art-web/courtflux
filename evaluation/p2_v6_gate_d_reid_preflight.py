from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = REPO_ROOT / "evaluation" / "ground_truth" / "gate_d_v6_human_ball_carrier_gt_v1.json"
DEFAULT_VIDEO = REPO_ROOT / "data" / "VLZugJeVOSU.mp4"
DEFAULT_WINDOW_FRAMES = 12
DEFAULT_SAMPLES_PER_OBSERVATION = 5
SAME_RECALL_TARGET = 0.85
DIFFERENT_FALSE_MERGE_MAX = 0.05


def _latest_states_file() -> Path:
    candidates = sorted(
        path
        for path in (REPO_ROOT / "evaluation" / "runs").glob("*/p2_states.jsonl")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "No evaluation/runs/*/p2_states.jsonl found. Pass --states explicitly."
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is unavailable")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mapped_observations(gt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = gt.get("mapped_observations") or []
    observations: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        observation_id = str(row.get("id") or "").strip()
        endpoint = row.get("endpoint") or {}
        if not observation_id or not isinstance(endpoint, dict):
            continue
        frame_id = endpoint.get("frame_id")
        track_id = endpoint.get("track_id")
        if frame_id is None or track_id is None:
            continue
        observations[observation_id] = {
            "id": observation_id,
            "human_player": str(row.get("human_player") or ""),
            "team_id": row.get("team_id"),
            "human_source_sec": row.get("human_source_sec"),
            "target_frame_id": int(frame_id),
            "track_id": int(track_id),
            "gt_episode_index": endpoint.get("episode_index"),
        }
    if not observations:
        raise ValueError("GT contains no mapped_observations")
    return observations


def _prediction_for_track(row: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    predictions = row.get("game_player_predictions") or []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        try:
            candidate_track = int(prediction.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if candidate_track == int(track_id):
            return prediction
    return None


def _collect_crop_candidates(
    states_path: Path,
    observations: dict[str, dict[str, Any]],
    *,
    window_frames: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in observations}
    targets = list(observations.values())

    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {states_path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                continue
            try:
                frame_id = int(row.get("frame_id", -10**9))
            except (TypeError, ValueError):
                continue

            for observation in targets:
                target_frame = int(observation["target_frame_id"])
                if abs(frame_id - target_frame) > int(window_frames):
                    continue
                prediction = _prediction_for_track(row, int(observation["track_id"]))
                if prediction is None:
                    continue
                box = prediction.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                timestamp = row.get("timestamp_sec")
                if timestamp is None:
                    continue
                result[observation["id"]].append(
                    {
                        "frame_id": frame_id,
                        "timestamp_sec": float(timestamp),
                        "box": [float(value) for value in box],
                        "runtime_episode_index": prediction.get("episode_index"),
                    }
                )
    return result


def _select_samples(
    candidates: list[dict[str, Any]],
    *,
    target_frame: int,
    sample_count: int,
    window_frames: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    span = max(1, int(window_frames))
    preferred_offsets = [0]
    step = max(1, span // 2)
    preferred_offsets.extend([-step, step, -span, span])
    for offset in range(1, span + 1):
        preferred_offsets.extend([-offset, offset])

    selected: list[dict[str, Any]] = []
    used_frames: set[int] = set()
    for offset in preferred_offsets:
        if len(selected) >= int(sample_count):
            break
        desired = int(target_frame) + int(offset)
        available = [
            item for item in candidates if int(item["frame_id"]) not in used_frames
        ]
        if not available:
            break
        best = min(
            available,
            key=lambda item: (
                abs(int(item["frame_id"]) - desired),
                abs(int(item["frame_id"]) - int(target_frame)),
            ),
        )
        selected.append(best)
        used_frames.add(int(best["frame_id"]))

    selected.sort(key=lambda item: int(item["frame_id"]))
    return selected


def _read_crop(
    capture: cv2.VideoCapture,
    *,
    timestamp_sec: float,
    box: list[float],
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_sec)) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"Could not decode video at {timestamp_sec:.3f}s")

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    x_pad = box_width * 0.03
    y_pad = box_height * 0.02
    ix1 = max(0, min(width - 1, int(math.floor(x1 - x_pad))))
    iy1 = max(0, min(height - 1, int(math.floor(y1 - y_pad))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2 + x_pad))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2 + y_pad))))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        raise RuntimeError(
            f"Empty crop at {timestamp_sec:.3f}s box={box} frame={width}x{height}"
        )
    return crop


def _preprocess(crop: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    tensor = tensor.div_(255.0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(256, 128),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _build_model(device: torch.device) -> torch.nn.Module:
    print("[ReID preflight] loading torchvision ResNet50 ImageNet weights...", flush=True)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    return model.to(device).eval()


def _embed_batches(
    model: torch.nn.Module,
    tensors: list[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), int(batch_size)):
            batch = torch.stack(tensors[start : start + int(batch_size)]).to(device)
            features = model(batch)
            features = F.normalize(features.float(), p=2, dim=1)
            outputs.append(features.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _pair_rows(
    gt: dict[str, Any],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_rows: list[dict[str, Any]] = []
    different_rows: list[dict[str, Any]] = []
    for key, label, destination in (
        ("same_pairs", "SAME", same_rows),
        ("different_pairs", "DIFFERENT", different_rows),
    ):
        for pair in gt.get(key) or []:
            if not isinstance(pair, dict):
                continue
            left = str(pair.get("left") or "")
            right = str(pair.get("right") or "")
            if left not in embeddings or right not in embeddings:
                raise RuntimeError(
                    f"Missing embedding for pair {pair.get('id')}: {left} / {right}"
                )
            destination.append(
                {
                    "id": str(pair.get("id") or ""),
                    "label": label,
                    "left": left,
                    "right": right,
                    "similarity": round(_cosine(embeddings[left], embeddings[right]), 6),
                    "time_gap_sec": pair.get("time_gap_sec"),
                }
            )
    if not same_rows or not different_rows:
        raise ValueError("GT must contain SAME and DIFFERENT pairs")
    return same_rows, different_rows


def _candidate_thresholds(scores: list[float]) -> list[float]:
    values = sorted(set(float(score) for score in scores), reverse=True)
    if not values:
        return [1.0]
    thresholds = [values[0] + 1e-6]
    for left, right in zip(values, values[1:]):
        thresholds.append((left + right) / 2.0)
    thresholds.append(values[-1] - 1e-6)
    return thresholds


def _operating_point(
    same_rows: list[dict[str, Any]],
    different_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    same_scores = [float(row["similarity"]) for row in same_rows]
    different_scores = [float(row["similarity"]) for row in different_rows]
    candidates = _candidate_thresholds(same_scores + different_scores)

    scored: list[dict[str, Any]] = []
    for threshold in candidates:
        same_correct = sum(score >= threshold for score in same_scores)
        false_merges = sum(score >= threshold for score in different_scores)
        same_recall = same_correct / len(same_scores)
        false_merge_rate = false_merges / len(different_scores)
        scored.append(
            {
                "threshold": float(threshold),
                "same_correct": int(same_correct),
                "same_total": len(same_scores),
                "same_recall": float(same_recall),
                "different_false_merges": int(false_merges),
                "different_total": len(different_scores),
                "different_false_merge_rate": float(false_merge_rate),
            }
        )

    within_false_merge_gate = [
        row
        for row in scored
        if float(row["different_false_merge_rate"]) <= DIFFERENT_FALSE_MERGE_MAX + 1e-12
    ]
    pool = within_false_merge_gate or scored
    best = max(
        pool,
        key=lambda row: (
            float(row["same_recall"]),
            -float(row["different_false_merge_rate"]),
            float(row["threshold"]),
        ),
    )
    best = dict(best)
    best["signal_feasible_for_gate_d"] = bool(
        float(best["same_recall"]) >= SAME_RECALL_TARGET
        and float(best["different_false_merge_rate"]) <= DIFFERENT_FALSE_MERGE_MAX
    )
    return best


def _auc(same_rows: list[dict[str, Any]], different_rows: list[dict[str, Any]]) -> float:
    wins = 0.0
    total = 0
    for same in same_rows:
        same_score = float(same["similarity"])
        for different in different_rows:
            different_score = float(different["similarity"])
            total += 1
            if same_score > different_score:
                wins += 1.0
            elif same_score == different_score:
                wins += 0.5
    return float(wins / total) if total else 0.0


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(array.min()), 6),
        "p25": round(float(np.percentile(array, 25)), 6),
        "median": round(float(np.median(array)), 6),
        "mean": round(float(array.mean()), 6),
        "p75": round(float(np.percentile(array, 75)), 6),
        "max": round(float(array.max()), 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt_path = Path(args.gt).expanduser().resolve()
    states_path = (
        Path(args.states).expanduser().resolve() if args.states else _latest_states_file()
    )
    video_path = Path(args.video).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    gt = _load_json(gt_path)
    observations = _mapped_observations(gt)
    candidates = _collect_crop_candidates(
        states_path,
        observations,
        window_frames=int(args.window_frames),
    )

    selected_by_observation: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for observation_id, observation in observations.items():
        selected = _select_samples(
            candidates.get(observation_id, []),
            target_frame=int(observation["target_frame_id"]),
            sample_count=int(args.samples_per_observation),
            window_frames=int(args.window_frames),
        )
        selected_by_observation[observation_id] = selected
        if not selected:
            missing.append(observation_id)
    if missing:
        raise RuntimeError(
            "Could not find raw-track crop evidence for GT observations: "
            + ", ".join(missing)
            + ". This preflight deliberately matches the human-mapped raw track, not the "
            "current game_player_id or episode index."
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    tensors: list[torch.Tensor] = []
    sample_owner: list[str] = []
    sample_diagnostics: dict[str, list[dict[str, Any]]] = {
        key: [] for key in observations
    }
    try:
        for observation_id in observations:
            for sample in selected_by_observation[observation_id]:
                crop = _read_crop(
                    capture,
                    timestamp_sec=float(sample["timestamp_sec"]),
                    box=list(sample["box"]),
                )
                tensors.append(_preprocess(crop))
                sample_owner.append(observation_id)
                sample_diagnostics[observation_id].append(
                    {
                        "frame_id": int(sample["frame_id"]),
                        "timestamp_sec": round(float(sample["timestamp_sec"]), 6),
                        "runtime_episode_index": sample.get("runtime_episode_index"),
                        "crop_width": int(crop.shape[1]),
                        "crop_height": int(crop.shape[0]),
                    }
                )
    finally:
        capture.release()

    device = _device(str(args.device))
    model = _build_model(device)
    sample_embeddings = _embed_batches(
        model,
        tensors,
        device=device,
        batch_size=int(args.batch_size),
    )

    grouped: dict[str, list[np.ndarray]] = {key: [] for key in observations}
    for owner, embedding in zip(sample_owner, sample_embeddings):
        grouped[owner].append(np.asarray(embedding, dtype=np.float32))

    observation_embeddings: dict[str, np.ndarray] = {}
    for observation_id, rows in grouped.items():
        matrix = np.stack(rows, axis=0)
        vector = matrix.mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise RuntimeError(f"Zero embedding for {observation_id}")
        observation_embeddings[observation_id] = vector / norm

    same_rows, different_rows = _pair_rows(gt, observation_embeddings)
    operating = _operating_point(same_rows, different_rows)
    threshold = float(operating["threshold"])
    for row in same_rows + different_rows:
        row["predicted_same"] = bool(float(row["similarity"]) >= threshold)
        row["correct_at_selected_threshold"] = bool(
            row["predicted_same"] == (row["label"] == "SAME")
        )

    same_scores = [float(row["similarity"]) for row in same_rows]
    different_scores = [float(row["similarity"]) for row in different_rows]
    output = {
        "version": "courtflux_gate_d_v6_reid_preflight_v1",
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "purpose": (
            "Measure whether visual appearance embeddings contain enough same-person "
            "signal on the frozen V6 human identity pairs before runtime integration."
        ),
        "model": {
            "backend": "torchvision_resnet50_imagenet_v2",
            "person_reid_specific": False,
            "input_size": [256, 128],
            "feature_dimensions": int(next(iter(observation_embeddings.values())).shape[0]),
            "device": str(device),
            "note": (
                "This is a zero-new-dependency baseline. If it does not meet the signal "
                "test, use a person-ReID-specific OSNet/MSMT17 backend next."
            ),
        },
        "sources": {
            "ground_truth": str(gt_path),
            "states": str(states_path),
            "video": str(video_path),
        },
        "crop_policy": {
            "raw_track_only": True,
            "game_player_id_used_for_crop_selection": False,
            "episode_index_used_for_crop_selection": False,
            "window_frames": int(args.window_frames),
            "samples_per_observation": int(args.samples_per_observation),
            "observation_count": len(observations),
        },
        "observation_samples": {
            observation_id: {
                "human_player": observations[observation_id]["human_player"],
                "team_id": observations[observation_id]["team_id"],
                "target_frame_id": observations[observation_id]["target_frame_id"],
                "track_id": observations[observation_id]["track_id"],
                "gt_episode_index_reference_only": observations[observation_id][
                    "gt_episode_index"
                ],
                "samples": sample_diagnostics[observation_id],
            }
            for observation_id in observations
        },
        "similarity": {
            "metric": "cosine",
            "same": _stats(same_scores),
            "different": _stats(different_scores),
            "pairwise_auc": round(_auc(same_rows, different_rows), 6),
        },
        "selected_operating_point": {
            **operating,
            "same_recall_target": SAME_RECALL_TARGET,
            "different_false_merge_rate_max": DIFFERENT_FALSE_MERGE_MAX,
            "selection_policy": (
                "maximize SAME recall subject to DIFFERENT false-merge <= 5%; "
                "diagnostic feasibility only, not a frozen production threshold"
            ),
        },
        "same_pairs": same_rows,
        "different_pairs": different_rows,
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else states_path.parent / "p2_v6_gate_d_reid_preflight.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    selected = output["selected_operating_point"]
    print("\nV6 Gate D visual ReID preflight", flush=True)
    print(f"model: torchvision ResNet50 ImageNet V2 ({device})", flush=True)
    print(
        f"SAME: {selected['same_correct']}/{selected['same_total']} "
        f"= {100.0 * float(selected['same_recall']):.2f}%",
        flush=True,
    )
    print(
        f"DIFFERENT false merges: {selected['different_false_merges']}/"
        f"{selected['different_total']} = "
        f"{100.0 * float(selected['different_false_merge_rate']):.2f}%",
        flush=True,
    )
    print(f"pairwise AUC: {100.0 * float(output['similarity']['pairwise_auc']):.2f}%", flush=True)
    print(f"selected cosine threshold: {threshold:.6f}", flush=True)
    print(
        "signal feasible for Gate D: "
        + ("YES" if selected["signal_feasible_for_gate_d"] else "NO"),
        flush=True,
    )
    print(f"report: {output_path}", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cheap V6 Gate-D visual identity preflight on the frozen human SAME/DIFFERENT "
            "pairs. This does not mutate runtime identity or score the formal gate."
        )
    )
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--states", default=None)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument(
        "--samples-per-observation", type=int, default=DEFAULT_SAMPLES_PER_OBSERVATION
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.window_frames < 0:
        raise SystemExit("--window-frames must be >= 0")
    if args.samples_per_observation < 1:
        raise SystemExit("--samples-per-observation must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
