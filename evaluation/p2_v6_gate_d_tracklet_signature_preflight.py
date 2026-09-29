from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_v6_gate_d_reid_preflight as base
from evaluation.p2_v6_gate_d_reid_preflight_rawtrack import _raw_track_prediction


DEFAULT_WINDOW_FRAMES = 120
DEFAULT_SAMPLES = 12


def _bbox_quality(box: list[float], prediction: dict[str, Any]) -> float:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    area_ratio = (w * h) / float(1920 * 1080)
    aspect = h / w
    confidence = float(prediction.get("confidence") or 0.0)
    shape = 1.0
    if aspect < 1.15:
        shape *= max(0.05, aspect / 1.15)
    if aspect > 5.0:
        shape *= max(0.05, 5.0 / aspect)
    if area_ratio > 0.12:
        shape *= max(0.05, 0.12 / area_ratio)
    if h < 90.0:
        shape *= max(0.05, h / 90.0)
    return float(max(0.0, min(1.0, confidence * shape)))


def _collect_candidates(
    states_path: Path,
    observations: dict[str, dict[str, Any]],
    *,
    window_frames: int,
) -> tuple[dict[str, list[dict[str, Any]]], set[tuple[int, int]]]:
    result = {key: [] for key in observations}
    targets = list(observations.values())
    target_track_ids = {int(row["track_id"]) for row in targets}
    co_visible: set[tuple[int, int]] = set()

    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            try:
                frame_id = int(row.get("frame_id", -10**9))
            except (TypeError, ValueError):
                continue

            visible_tracks: list[int] = []
            for prediction in row.get("team_predictions") or []:
                if not isinstance(prediction, dict):
                    continue
                try:
                    track_id = int(prediction.get("track_id", -1))
                except (TypeError, ValueError):
                    continue
                if track_id in target_track_ids:
                    visible_tracks.append(track_id)
            for left, right in combinations(sorted(set(visible_tracks)), 2):
                co_visible.add((left, right))

            timestamp = row.get("timestamp_sec")
            if timestamp is None:
                continue
            for observation in targets:
                target_frame = int(observation["target_frame_id"])
                if abs(frame_id - target_frame) > int(window_frames):
                    continue
                prediction = _raw_track_prediction(row, int(observation["track_id"]))
                if prediction is None:
                    continue
                box = prediction.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                box_f = [float(value) for value in box]
                result[observation["id"]].append(
                    {
                        "frame_id": frame_id,
                        "timestamp_sec": float(timestamp),
                        "box": box_f,
                        "quality": _bbox_quality(box_f, prediction),
                        "source": prediction.get("reid_crop_source", "raw_track"),
                    }
                )
    return result, co_visible


def _select_samples(
    candidates: list[dict[str, Any]],
    *,
    target_frame: int,
    sample_count: int,
    window_frames: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    # Prefer player-shaped boxes, while retaining temporal locality.
    span = max(1, int(window_frames))
    ranked = sorted(
        candidates,
        key=lambda item: (
            -(float(item["quality"]) - 0.35 * abs(int(item["frame_id"]) - target_frame) / span),
            abs(int(item["frame_id"]) - target_frame),
        ),
    )
    chosen: list[dict[str, Any]] = []
    used: list[int] = []
    min_spacing = max(2, span // max(4, sample_count))
    for item in ranked:
        frame_id = int(item["frame_id"])
        if any(abs(frame_id - previous) < min_spacing for previous in used):
            continue
        chosen.append(item)
        used.append(frame_id)
        if len(chosen) >= int(sample_count):
            break
    if len(chosen) < min(4, sample_count):
        for item in ranked:
            frame_id = int(item["frame_id"])
            if frame_id in used:
                continue
            chosen.append(item)
            used.append(frame_id)
            if len(chosen) >= int(sample_count):
                break
    return sorted(chosen, key=lambda item: int(item["frame_id"]))


def _decode_torso(
    capture: cv2.VideoCapture,
    *,
    timestamp_sec: float,
    box: list[float],
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"Could not decode video at {timestamp_sec:.3f}s")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(2.0, x2 - x1)
    h = max(2.0, y2 - y1)
    # Upper torso: enough context for shoulder/skin/jersey pattern but excludes most floor/legs.
    tx1 = x1 + 0.12 * w
    tx2 = x1 + 0.88 * w
    ty1 = y1 + 0.08 * h
    ty2 = y1 + 0.64 * h
    ix1 = max(0, min(width - 1, int(math.floor(tx1))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(tx2))))
    iy1 = max(0, min(height - 1, int(math.floor(ty1))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(ty2))))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        raise RuntimeError(f"Empty torso crop at {timestamp_sec:.3f}s")
    return crop


def _l2(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _descriptor(crop: np.ndarray) -> np.ndarray:
    image = cv2.resize(crop, (64, 96), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Illumination-normalised low-resolution structure: retains jersey glyph/outline signal.
    small = cv2.resize(gray, (16, 24), interpolation=cv2.INTER_AREA)
    small = (small - float(small.mean())) / (float(small.std()) + 1e-5)
    structure = _l2(small)

    # Spatial gradient histograms capture number/letter edges without requiring OCR.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    grad_parts: list[np.ndarray] = []
    for gy_idx in range(4):
        for gx_idx in range(3):
            y0, y1 = gy_idx * 24, (gy_idx + 1) * 24
            x0 = int(round(gx_idx * 64 / 3.0))
            x1 = int(round((gx_idx + 1) * 64 / 3.0))
            cell_angle = angle[y0:y1, x0:x1]
            cell_mag = mag[y0:y1, x0:x1]
            hist, _ = np.histogram(
                cell_angle,
                bins=8,
                range=(0.0, 360.0),
                weights=cell_mag,
            )
            grad_parts.append(hist.astype(np.float32))
    gradients = _l2(np.concatenate(grad_parts))

    # Coarse LAB colour by horizontal bands; weakly weighted because team colour is shared.
    color_parts: list[np.ndarray] = []
    for band in np.array_split(lab, 3, axis=0):
        for channel in range(3):
            hist = cv2.calcHist([band], [channel], None, [8], [0, 256]).reshape(-1)
            color_parts.append(hist.astype(np.float32))
    color = _l2(np.concatenate(color_parts))

    combined = np.concatenate([
        0.52 * gradients,
        0.38 * structure,
        0.10 * color,
    ])
    return _l2(combined)


def _sample_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _tracklet_similarity(
    left: list[np.ndarray],
    right: list[np.ndarray],
) -> float:
    scores = sorted(
        (_sample_similarity(a, b) for a in left for b in right),
        reverse=True,
    )
    if not scores:
        return -1.0
    # A few pose-compatible frames should agree; one accidental match must not dominate.
    keep = max(2, int(math.ceil(len(scores) * 0.20)))
    return float(np.median(np.asarray(scores[:keep], dtype=np.float32)))


def _pair_rows(
    gt: dict[str, Any],
    embeddings: dict[str, list[np.ndarray]],
    observations: dict[str, dict[str, Any]],
    co_visible: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_rows: list[dict[str, Any]] = []
    different_rows: list[dict[str, Any]] = []
    for key, label, destination in (
        ("same_pairs", "SAME", same_rows),
        ("different_pairs", "DIFFERENT", different_rows),
    ):
        for pair in gt.get(key) or []:
            left_id = str(pair.get("left") or "")
            right_id = str(pair.get("right") or "")
            left_track = int(observations[left_id]["track_id"])
            right_track = int(observations[right_id]["track_id"])
            cannot_link = tuple(sorted((left_track, right_track))) in co_visible and left_track != right_track
            visual = _tracklet_similarity(embeddings[left_id], embeddings[right_id])
            similarity = -1.0 if cannot_link else visual
            destination.append(
                {
                    "id": str(pair.get("id") or ""),
                    "label": label,
                    "left": left_id,
                    "right": right_id,
                    "left_track_id": left_track,
                    "right_track_id": right_track,
                    "co_visibility_cannot_link": bool(cannot_link),
                    "visual_similarity": round(float(visual), 6),
                    "similarity": round(float(similarity), 6),
                    "time_gap_sec": pair.get("time_gap_sec"),
                }
            )
    return same_rows, different_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt_path = Path(args.gt).expanduser().resolve()
    states_path = Path(args.states).expanduser().resolve() if args.states else base._latest_states_file()
    video_path = Path(args.video).expanduser().resolve()
    gt = base._load_json(gt_path)
    observations = base._mapped_observations(gt)

    print("[Tracklet preflight] 1/4 scanning raw tracks + co-visibility", flush=True)
    candidates, co_visible = _collect_candidates(
        states_path,
        observations,
        window_frames=int(args.window_frames),
    )
    selected: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for observation_id, observation in observations.items():
        rows = _select_samples(
            candidates.get(observation_id, []),
            target_frame=int(observation["target_frame_id"]),
            sample_count=int(args.samples),
            window_frames=int(args.window_frames),
        )
        selected[observation_id] = rows
        if not rows:
            missing.append(observation_id)
    if missing:
        raise RuntimeError("No raw-track samples for: " + ", ".join(missing))

    print("[Tracklet preflight] 2/4 decoding torso samples", flush=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    descriptors: dict[str, list[np.ndarray]] = {key: [] for key in observations}
    sample_diag: dict[str, list[dict[str, Any]]] = {key: [] for key in observations}
    try:
        for observation_id in observations:
            for sample in selected[observation_id]:
                crop = _decode_torso(
                    capture,
                    timestamp_sec=float(sample["timestamp_sec"]),
                    box=list(sample["box"]),
                )
                descriptors[observation_id].append(_descriptor(crop))
                sample_diag[observation_id].append(
                    {
                        "frame_id": int(sample["frame_id"]),
                        "timestamp_sec": round(float(sample["timestamp_sec"]), 6),
                        "bbox_quality": round(float(sample["quality"]), 6),
                        "torso_width": int(crop.shape[1]),
                        "torso_height": int(crop.shape[0]),
                    }
                )
    finally:
        capture.release()

    print("[Tracklet preflight] 3/4 scoring frozen SAME/DIFFERENT pairs", flush=True)
    same_rows, different_rows = _pair_rows(
        gt, descriptors, observations, co_visible
    )
    operating = base._operating_point(same_rows, different_rows)
    threshold = float(operating["threshold"])
    for row in same_rows + different_rows:
        row["predicted_same"] = bool(float(row["similarity"]) >= threshold)
        row["correct_at_selected_threshold"] = bool(
            row["predicted_same"] == (row["label"] == "SAME")
        )

    same_scores = [float(row["similarity"]) for row in same_rows]
    diff_scores = [float(row["similarity"]) for row in different_rows]
    output = {
        "version": "courtflux_gate_d_v6_tracklet_signature_preflight_v1",
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "sources": {
            "ground_truth": str(gt_path),
            "states": str(states_path),
            "video": str(video_path),
        },
        "policy": {
            "human_identity_used_for_descriptor_or_pair_selection": False,
            "game_player_id_used": False,
            "raw_track_geometry_used": True,
            "co_visibility_is_hard_cannot_link": True,
            "torso_signature_semantic_jersey_reading": False,
            "window_frames": int(args.window_frames),
            "samples_per_observation": int(args.samples),
        },
        "observation_samples": {
            observation_id: {
                "track_id": observations[observation_id]["track_id"],
                "target_frame_id": observations[observation_id]["target_frame_id"],
                "samples": sample_diag[observation_id],
            }
            for observation_id in observations
        },
        "similarity": {
            "metric": "tracklet torso signature + co-visibility cannot-link",
            "same": base._stats(same_scores),
            "different": base._stats(diff_scores),
            "pairwise_auc": round(base._auc(same_rows, different_rows), 6),
        },
        "selected_operating_point": {
            **operating,
            "same_recall_target": base.SAME_RECALL_TARGET,
            "different_false_merge_rate_max": base.DIFFERENT_FALSE_MERGE_MAX,
            "selection_policy": "diagnostic feasibility only; maximize SAME recall subject to <=5% false merges",
        },
        "same_pairs": same_rows,
        "different_pairs": different_rows,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else states_path.parent / "p2_v6_gate_d_tracklet_signature_preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[Tracklet preflight] 4/4 complete", flush=True)
    selected_op = output["selected_operating_point"]
    print("\nV6 Gate D tracklet-signature preflight", flush=True)
    print(
        f"SAME: {selected_op['same_correct']}/{selected_op['same_total']} = "
        f"{100.0 * float(selected_op['same_recall']):.2f}%",
        flush=True,
    )
    print(
        f"DIFFERENT false merges: {selected_op['different_false_merges']}/"
        f"{selected_op['different_total']} = "
        f"{100.0 * float(selected_op['different_false_merge_rate']):.2f}%",
        flush=True,
    )
    print(f"pairwise AUC: {100.0 * float(output['similarity']['pairwise_auc']):.2f}%", flush=True)
    print(f"selected signature threshold: {threshold:.6f}", flush=True)
    print(
        "signal feasible for Gate D: "
        + ("YES" if selected_op["signal_feasible_for_gate_d"] else "NO"),
        flush=True,
    )
    print(f"report: {output_path}", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cheap V6 Gate-D temporal tracklet/torso-signature preflight")
    parser.add_argument("--gt", default=str(base.DEFAULT_GT))
    parser.add_argument("--states", default=None)
    parser.add_argument("--video", default=str(base.DEFAULT_VIDEO))
    parser.add_argument("--output", default=None)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.window_frames < 1:
        raise SystemExit("--window-frames must be >= 1")
    if args.samples < 2:
        raise SystemExit("--samples must be >= 2")
    run(args)


if __name__ == "__main__":
    main()
