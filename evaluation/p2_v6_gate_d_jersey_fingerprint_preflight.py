from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_v6_gate_d_reid_preflight as base
from evaluation.p2_v6_gate_d_reid_preflight_rawtrack import _raw_track_prediction
from services.vision.performance_jersey_analyzer import PerformanceJerseyAnalyzer

VERSION = "courtflux_gate_d_v6_jersey_fingerprint_preflight_v1"
DEFAULT_MATCH_CONFIG = REPO_ROOT / "evaluation" / "match_configs" / "gate_c_v6.json"
DEFAULT_MODEL = REPO_ROOT / "weights" / "jersey_number_recon.pt"
DEFAULT_WINDOW_FRAMES = 24
DEFAULT_SAMPLES_PER_OBSERVATION = 8


def _collect_raw_track_candidates(
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
                prediction = _raw_track_prediction(row, int(observation["track_id"]))
                if prediction is None:
                    continue
                box = prediction.get("box")
                timestamp = row.get("timestamp_sec")
                if not isinstance(box, (list, tuple)) or len(box) != 4 or timestamp is None:
                    continue
                result[observation["id"]].append(
                    {
                        "frame_id": frame_id,
                        "timestamp_sec": float(timestamp),
                        "box": [float(value) for value in box],
                        "crop_source": prediction.get("reid_crop_source"),
                    }
                )
    return result


def _torso(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        raise ValueError("empty player crop")
    height, width = crop.shape[:2]
    ymin, ymax = int(height * 0.15), int(height * 0.65)
    xmin, xmax = int(width * 0.10), int(width * 0.90)
    torso = crop[ymin:ymax, xmin:xmax]
    if torso.size == 0:
        raise ValueError(f"empty torso crop from {width}x{height}")
    return torso


def _bc(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Bhattacharyya inputs must be same non-zero shape")
    left = np.clip(left, 0.0, None)
    right = np.clip(right, 0.0, None)
    lsum = float(left.sum())
    rsum = float(right.sum())
    if lsum <= 1e-12 or rsum <= 1e-12:
        return 0.0
    left /= lsum
    right /= rsum
    return float(np.clip(np.sqrt(left * right).sum(), 0.0, 1.0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("cosine inputs must be same non-zero shape")
    ln = float(np.linalg.norm(left))
    rn = float(np.linalg.norm(right))
    if ln <= 1e-12 or rn <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / (ln * rn), -1.0, 1.0))


def _pair_rows_for_representation(
    gt: dict[str, Any],
    fingerprints: dict[str, dict[str, np.ndarray]],
    representation: str,
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
            if left not in fingerprints or right not in fingerprints:
                raise RuntimeError(f"Missing fingerprint for pair {pair.get('id')}")
            if representation == "digit_posterior":
                lt = fingerprints[left]["tens"]
                rt = fingerprints[right]["tens"]
                lo = fingerprints[left]["ones"]
                ro = fingerprints[right]["ones"]
                similarity = 0.5 * (_bc(lt, rt) + _bc(lo, ro))
            elif representation == "projection":
                similarity = _cosine(
                    fingerprints[left]["projection"],
                    fingerprints[right]["projection"],
                )
            elif representation == "equal_fusion":
                posterior = 0.5 * (
                    _bc(fingerprints[left]["tens"], fingerprints[right]["tens"])
                    + _bc(fingerprints[left]["ones"], fingerprints[right]["ones"])
                )
                projection = _cosine(
                    fingerprints[left]["projection"],
                    fingerprints[right]["projection"],
                )
                similarity = 0.5 * (posterior + projection)
            else:
                raise ValueError(f"unknown representation: {representation}")

            destination.append(
                {
                    "id": str(pair.get("id") or ""),
                    "label": label,
                    "left": left,
                    "right": right,
                    "similarity": round(float(similarity), 6),
                    "time_gap_sec": pair.get("time_gap_sec"),
                }
            )
    if not same_rows or not different_rows:
        raise ValueError("GT must contain SAME and DIFFERENT pairs")
    return same_rows, different_rows


def _score_representation(
    gt: dict[str, Any],
    fingerprints: dict[str, dict[str, np.ndarray]],
    representation: str,
) -> dict[str, Any]:
    same_rows, different_rows = _pair_rows_for_representation(
        gt, fingerprints, representation
    )
    operating = base._operating_point(same_rows, different_rows)
    threshold = float(operating["threshold"])
    for row in same_rows + different_rows:
        row["predicted_same"] = bool(float(row["similarity"]) >= threshold)
        row["correct_at_selected_threshold"] = bool(
            row["predicted_same"] == (row["label"] == "SAME")
        )
    same_scores = [float(row["similarity"]) for row in same_rows]
    different_scores = [float(row["similarity"]) for row in different_rows]
    return {
        "representation": representation,
        "similarity": {
            "same": base._stats(same_scores),
            "different": base._stats(different_scores),
            "pairwise_auc": round(base._auc(same_rows, different_rows), 6),
        },
        "selected_operating_point": {
            **operating,
            "same_recall_target": base.SAME_RECALL_TARGET,
            "different_false_merge_rate_max": base.DIFFERENT_FALSE_MERGE_MAX,
            "selection_policy": (
                "oracle diagnostic only: maximize SAME recall subject to "
                "DIFFERENT false-merge <= 5%; never a production threshold"
            ),
        },
        "same_pairs": same_rows,
        "different_pairs": different_rows,
    }


def _model_forward(
    analyzer: PerformanceJerseyAnalyzer,
    tensors: list[torch.Tensor],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = analyzer.number_model
    for attr in ("backbone", "proj", "head_tens", "head_ones"):
        if not hasattr(model, attr):
            raise RuntimeError(
                "Jersey checkpoint model does not expose checkpoint-native "
                f"{attr!r}; anonymous fingerprint preflight cannot inspect it safely."
            )

    projections: list[np.ndarray] = []
    tens_rows: list[np.ndarray] = []
    ones_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), int(batch_size)):
            batch = torch.stack(tensors[start : start + int(batch_size)]).to(
                analyzer.device
            )
            features = model.backbone(batch)
            projected = torch.relu(model.proj(features))
            tens_logits = model.head_tens(projected)
            ones_logits = model.head_ones(projected)
            projected = torch.nn.functional.normalize(projected.float(), p=2, dim=1)
            tens_probs = torch.softmax(tens_logits.float(), dim=1)
            ones_probs = torch.softmax(ones_logits.float(), dim=1)
            projections.append(projected.detach().cpu().numpy())
            tens_rows.append(tens_probs.detach().cpu().numpy())
            ones_rows.append(ones_probs.detach().cpu().numpy())

    return (
        np.concatenate(projections, axis=0),
        np.concatenate(tens_rows, axis=0),
        np.concatenate(ones_rows, axis=0),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt_path = Path(args.gt).expanduser().resolve()
    states_path = (
        Path(args.states).expanduser().resolve()
        if args.states
        else base._latest_states_file()
    )
    video_path = Path(args.video).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    match_config = Path(args.match_config).expanduser().resolve()

    for path in (gt_path, states_path, video_path, model_path, match_config):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[Jersey fingerprint preflight] 1/5 scanning frozen raw-track crops", flush=True)
    gt = base._load_json(gt_path)
    observations = base._mapped_observations(gt)
    candidates = _collect_raw_track_candidates(
        states_path, observations, window_frames=int(args.window_frames)
    )

    selected_by_observation: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for observation_id, observation in observations.items():
        selected = base._select_samples(
            candidates.get(observation_id, []),
            target_frame=int(observation["target_frame_id"]),
            sample_count=int(args.samples_per_observation),
            window_frames=int(args.window_frames),
        )
        selected_by_observation[observation_id] = selected
        if len(selected) < int(args.min_samples_per_observation):
            missing.append(f"{observation_id}({len(selected)})")
    if missing:
        raise RuntimeError(
            "Insufficient raw-track crop evidence for jersey fingerprint: "
            + ", ".join(missing)
        )

    print("[Jersey fingerprint preflight] 2/5 decoding torso crops", flush=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    torso_arrays: list[np.ndarray] = []
    sample_owner: list[str] = []
    sample_diagnostics: dict[str, list[dict[str, Any]]] = {
        key: [] for key in observations
    }
    try:
        for observation_id in observations:
            for sample in selected_by_observation[observation_id]:
                crop = base._read_crop(
                    capture,
                    timestamp_sec=float(sample["timestamp_sec"]),
                    box=list(sample["box"]),
                )
                torso = _torso(crop)
                torso_arrays.append(torso)
                sample_owner.append(observation_id)
                sample_diagnostics[observation_id].append(
                    {
                        "frame_id": int(sample["frame_id"]),
                        "timestamp_sec": round(float(sample["timestamp_sec"]), 6),
                        "crop_source": sample.get("crop_source"),
                        "player_crop_width": int(crop.shape[1]),
                        "player_crop_height": int(crop.shape[0]),
                        "torso_width": int(torso.shape[1]),
                        "torso_height": int(torso.shape[0]),
                    }
                )
    finally:
        capture.release()

    print("[Jersey fingerprint preflight] 3/5 loading checkpoint-native jersey model", flush=True)
    requested = str(args.device)
    analyzer = PerformanceJerseyAnalyzer(
        profiling_path=str(match_config),
        device=(
            "mps"
            if requested == "auto" and torch.backends.mps.is_available()
            else "cuda"
            if requested == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested == "auto"
            else requested
        ),
        model_path=str(model_path),
        sequence_length=int(args.samples_per_observation),
    )

    print("[Jersey fingerprint preflight] 4/5 embedding all samples once", flush=True)
    tensors: list[torch.Tensor] = []
    from PIL import Image

    for torso in torso_arrays:
        rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
        tensors.append(analyzer.transform(Image.fromarray(rgb)))

    projections, tens_probs, ones_probs = _model_forward(
        analyzer, tensors, batch_size=int(args.batch_size)
    )

    grouped: dict[str, dict[str, list[np.ndarray]]] = {
        key: {"projection": [], "tens": [], "ones": []} for key in observations
    }
    for index, owner in enumerate(sample_owner):
        grouped[owner]["projection"].append(projections[index])
        grouped[owner]["tens"].append(tens_probs[index])
        grouped[owner]["ones"].append(ones_probs[index])

    fingerprints: dict[str, dict[str, np.ndarray]] = {}
    observation_quality: dict[str, dict[str, Any]] = {}
    for observation_id, parts in grouped.items():
        projection = np.stack(parts["projection"], axis=0).mean(axis=0)
        projection_norm = float(np.linalg.norm(projection))
        if projection_norm <= 1e-12:
            raise RuntimeError(f"zero projection fingerprint for {observation_id}")
        projection = projection / projection_norm
        tens = np.stack(parts["tens"], axis=0).mean(axis=0)
        ones = np.stack(parts["ones"], axis=0).mean(axis=0)
        tens /= max(float(tens.sum()), 1e-12)
        ones /= max(float(ones.sum()), 1e-12)
        fingerprints[observation_id] = {
            "projection": projection,
            "tens": tens,
            "ones": ones,
        }
        observation_quality[observation_id] = {
            "samples": len(parts["projection"]),
            "tens_peak_probability": round(float(tens.max()), 6),
            "ones_peak_probability": round(float(ones.max()), 6),
            "tens_entropy": round(
                float(-(tens * np.log(np.clip(tens, 1e-12, 1.0))).sum()), 6
            ),
            "ones_entropy": round(
                float(-(ones * np.log(np.clip(ones, 1e-12, 1.0))).sum()), 6
            ),
        }

    print("[Jersey fingerprint preflight] 5/5 scoring frozen SAME/DIFFERENT pairs", flush=True)
    representations = {
        name: _score_representation(gt, fingerprints, name)
        for name in ("digit_posterior", "projection", "equal_fusion")
    }
    best_name, best_result = max(
        representations.items(),
        key=lambda item: (
            float(item[1]["selected_operating_point"]["same_recall"]),
            -float(item[1]["selected_operating_point"]["different_false_merge_rate"]),
            float(item[1]["similarity"]["pairwise_auc"]),
        ),
    )

    report = {
        "version": VERSION,
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "runtime_code_modified": False,
        "purpose": (
            "Test whether the existing jersey recognizer contains anonymous same-person "
            "signal before any Gate-D integration. No roster or decoded jersey number is "
            "used as an identity key."
        ),
        "anti_leakage": {
            "human_identity_used_for_crop_selection": False,
            "game_player_id_used": False,
            "episode_index_used_for_crop_selection": False,
            "raw_track_geometry_only_for_crop_selection": True,
            "roster_decode_used_for_similarity": False,
            "decoded_jersey_number_used_for_similarity": False,
            "human_gt_used_only_for_pair_scoring": True,
            "representation_weights_tuned_on_v6": False,
            "equal_fusion_weights": [0.5, 0.5],
            "selected_thresholds_are_oracle_diagnostic_only": True,
        },
        "sources": {
            "ground_truth": str(gt_path),
            "states": str(states_path),
            "video": str(video_path),
            "model": str(model_path),
            "match_config": str(match_config),
        },
        "model": {
            **dict(analyzer.checkpoint_diagnostics),
            "device": str(analyzer.device),
            "fingerprint_semantics": "anonymous projection + digit posterior; no roster decode",
        },
        "crop_policy": {
            "raw_track_only": True,
            "window_frames": int(args.window_frames),
            "samples_per_observation": int(args.samples_per_observation),
            "min_samples_per_observation": int(args.min_samples_per_observation),
            "observation_count": len(observations),
            "torso_policy": "player crop y=15..65%, x=10..90% (checkpoint-native)",
        },
        "observation_samples": {
            observation_id: {
                "human_player": observations[observation_id]["human_player"],
                "team_id": observations[observation_id]["team_id"],
                "target_frame_id": observations[observation_id]["target_frame_id"],
                "track_id": observations[observation_id]["track_id"],
                "samples": sample_diagnostics[observation_id],
                "fingerprint_quality": observation_quality[observation_id],
            }
            for observation_id in observations
        },
        "representations": representations,
        "best_diagnostic_representation": best_name,
        "signal_feasible_for_gate_d": bool(
            best_result["selected_operating_point"]["signal_feasible_for_gate_d"]
        ),
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else states_path.parent / "p2_v6_gate_d_jersey_fingerprint_preflight.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nV6 Gate D jersey-fingerprint preflight", flush=True)
    for name in ("digit_posterior", "projection", "equal_fusion"):
        result = representations[name]
        selected = result["selected_operating_point"]
        print(
            f"{name}: SAME={selected['same_correct']}/{selected['same_total']} "
            f"({100.0 * float(selected['same_recall']):.2f}%) "
            f"DIFF_false_merge={selected['different_false_merges']}/"
            f"{selected['different_total']} "
            f"({100.0 * float(selected['different_false_merge_rate']):.2f}%) "
            f"AUC={100.0 * float(result['similarity']['pairwise_auc']):.2f}%",
            flush=True,
        )
    print(f"best diagnostic representation: {best_name}", flush=True)
    print(
        "signal feasible for Gate D: "
        + ("YES" if report["signal_feasible_for_gate_d"] else "NO"),
        flush=True,
    )
    print(f"report: {output_path}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic anonymous jersey-fingerprint capacity test for Gate D V6. "
            "No runtime mutation, roster decode or production threshold selection."
        )
    )
    parser.add_argument("--gt", default=str(base.DEFAULT_GT))
    parser.add_argument("--states")
    parser.add_argument("--video", default=str(base.DEFAULT_VIDEO))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--match-config", default=str(DEFAULT_MATCH_CONFIG))
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument(
        "--samples-per-observation", type=int, default=DEFAULT_SAMPLES_PER_OBSERVATION
    )
    parser.add_argument("--min-samples-per-observation", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.window_frames < 0:
        raise SystemExit("--window-frames must be >= 0")
    if args.samples_per_observation < 2:
        raise SystemExit("--samples-per-observation must be >= 2")
    if args.min_samples_per_observation < 1:
        raise SystemExit("--min-samples-per-observation must be >= 1")
    if args.min_samples_per_observation > args.samples_per_observation:
        raise SystemExit("--min-samples-per-observation cannot exceed samples-per-observation")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    run(args)
