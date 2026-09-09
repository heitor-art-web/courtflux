from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import align_production_truth, candidate_matches_truth, load_jsonl
from evaluation.p0_roboflow_tracker_benchmark import (
    _candidate_rows,
    _confidence,
    detections_from_row,
    make_tracker,
    tracked_candidate_dict,
)


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def _float(candidate: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    try:
        value = candidate.get(key)
        return fallback if value is None else float(value)
    except (TypeError, ValueError):
        return fallback


def _center_from_box(box: np.ndarray) -> np.ndarray:
    return np.asarray(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0), dtype=np.float64)


def _area_from_box(box: np.ndarray) -> float:
    return max(float(box[2] - box[0]), 1.0) * max(float(box[3] - box[1]), 1.0)


def _auc(values: list[float], labels: list[int]) -> float | None:
    """Rank-sum ROC AUC without sklearn. Ties receive average ranks."""
    if not values or len(values) != len(labels):
        return None
    arr = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    finite = np.isfinite(arr)
    arr = arr[finite]
    y = y[finite]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(arr, kind="mergesort")
    sorted_values = arr[order]
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and sorted_values[j] == sorted_values[i]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _median(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return None if not finite else float(np.median(np.asarray(finite, dtype=np.float64)))


def _best_recall_at_precision(
    values: list[float], labels: list[int], precision_target: float = 0.90
) -> dict[str, Any]:
    """Diagnostic univariate separability in both directions; not a production threshold search."""
    arr = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    finite = np.isfinite(arr)
    arr = arr[finite]
    y = y[finite]
    positives = int(np.sum(y == 1))
    if positives == 0 or len(arr) == 0:
        return {"recall_pct": 0.0, "precision_pct": 0.0, "direction": None, "threshold": None}

    unique = np.unique(arr)
    if len(unique) > 512:
        unique = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, 513)))

    best = (0.0, 0.0, None, None)
    for direction in ("ge", "le"):
        for threshold in unique:
            pred = arr >= threshold if direction == "ge" else arr <= threshold
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            if tp == 0:
                continue
            precision = tp / max(tp + fp, 1)
            recall = tp / positives
            if precision + 1e-12 < precision_target:
                continue
            if recall > best[0] or (recall == best[0] and precision > best[1]):
                best = (recall, precision, direction, float(threshold))
    return {
        "recall_pct": round(100.0 * best[0], 2),
        "precision_pct": round(100.0 * best[1], 2),
        "direction": best[2],
        "threshold": best[3],
    }


@dataclass
class TrackHistory:
    observations: int = 0
    last_step: int | None = None
    streak: int = 0
    confidences: list[float] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    last_center: np.ndarray | None = None
    last_observation_step: int | None = None
    true_instances: int = 0
    false_instances: int = 0
    visible_true_frames: int = 0
    not_visible_instances: int = 0

    def update(self, step: int, confidence: float, area: float, center: np.ndarray) -> dict[str, float]:
        self.observations += 1
        self.streak = self.streak + 1 if self.last_step is not None and step == self.last_step + 1 else 1
        self.last_step = step
        self.confidences.append(float(confidence))
        self.areas.append(float(area))

        speed = 0.0
        if self.last_center is not None and self.last_observation_step is not None:
            dt = max(step - self.last_observation_step, 1)
            speed = float(np.linalg.norm(center - self.last_center) / float(dt))
        self.last_center = center
        self.last_observation_step = step

        mean_conf = float(np.mean(self.confidences))
        area_mean = float(np.mean(self.areas))
        area_std = float(np.std(self.areas)) if len(self.areas) > 1 else 0.0
        area_cv = area_std / max(area_mean, 1e-6)
        return {
            "track_observations": float(self.observations),
            "track_streak": float(self.streak),
            "track_mean_confidence": mean_conf,
            "track_area_cv": float(area_cv),
            "track_speed_px_per_step": speed,
        }


def analyze_pair(
    run_dir: Path,
    gt_path: Path,
    *,
    frame_rate: float,
    radius_px: float,
) -> dict[str, Any]:
    production_rows = load_jsonl(run_dir / "frames.jsonl")
    truth_rows = load_jsonl(gt_path)
    aligned, unmatched = align_production_truth(production_rows, truth_rows, tolerance_sec=0.001)
    tracker = make_tracker("ocsort_ciou", frame_rate)

    histories: dict[int, TrackHistory] = defaultdict(TrackHistory)
    observations: list[dict[str, Any]] = []
    visible_frames = 0
    visible_with_true_confirmed = 0
    not_visible_frames = 0

    for step, (row, truth, _delta) in enumerate(aligned):
        is_visible = bool(truth.get("visible"))
        visible_frames += int(is_visible)
        not_visible_frames += int(not is_visible)

        source_candidates = _candidate_rows(row)
        detections = detections_from_row(row)
        tracked = tracker.update(detections)
        tids = (
            np.asarray(tracked.tracker_id, dtype=int)
            if tracked.tracker_id is not None
            else np.full(len(tracked), -1, dtype=int)
        )
        source_indices = tracked.data.get("courtflux_candidate_index")
        if source_indices is None:
            source_indices = np.full(len(tracked), -1, dtype=int)
        source_indices = np.asarray(source_indices, dtype=int)

        confirmed_indices = [i for i, tid in enumerate(tids.tolist()) if int(tid) >= 0]
        frame_true = False
        frame_confidences = []
        for i in confirmed_indices:
            source_idx = int(source_indices[i]) if i < len(source_indices) else -1
            source = source_candidates[source_idx] if 0 <= source_idx < len(source_candidates) else {}
            frame_confidences.append(_confidence(source))

        for i in confirmed_indices:
            tid = int(tids[i])
            source_idx = int(source_indices[i]) if i < len(source_indices) else -1
            source = source_candidates[source_idx] if 0 <= source_idx < len(source_candidates) else {}
            candidate = tracked_candidate_dict(tracked, i)
            correct, _distance = candidate_matches_truth(candidate, truth, radius_px, 0.25)
            label = int(is_visible and correct)
            frame_true = frame_true or bool(label)

            box = np.asarray(tracked.xyxy[i], dtype=np.float64)
            center = _center_from_box(box)
            area = _area_from_box(box)
            detector_conf = _confidence(source)
            history_features = histories[tid].update(step, detector_conf, area, center)
            if label:
                histories[tid].true_instances += 1
                histories[tid].visible_true_frames += 1
            else:
                histories[tid].false_instances += 1
                if not is_visible:
                    histories[tid].not_visible_instances += 1

            sorted_conf = sorted(frame_confidences, reverse=True)
            conf_rank = 1 + sum(1 for value in sorted_conf if value > detector_conf)
            observations.append(
                {
                    "run_id": run_dir.name,
                    "step": step,
                    "tracker_id": tid,
                    "label": label,
                    "visible": is_visible,
                    "detector_confidence": float(detector_conf),
                    "rank_score": _float(source, "rank_score", detector_conf),
                    "association_confidence": _float(source, "association_confidence", detector_conf),
                    "confidence_rank_in_confirmed": float(conf_rank),
                    "confirmed_competitors": float(len(confirmed_indices)),
                    "box_area": float(area),
                    **history_features,
                }
            )

        visible_with_true_confirmed += int(is_visible and frame_true)

    track_rows: list[dict[str, Any]] = []
    for tid, history in histories.items():
        total = history.true_instances + history.false_instances
        track_rows.append(
            {
                "tracker_id": tid,
                "instances": total,
                "true_instances": history.true_instances,
                "false_instances": history.false_instances,
                "purity_pct": round(100.0 * history.true_instances / max(total, 1), 2),
                "mean_confidence": float(np.mean(history.confidences)) if history.confidences else 0.0,
                "area_cv": float(np.std(history.areas) / max(np.mean(history.areas), 1e-6)) if history.areas else 0.0,
                "not_visible_instances": history.not_visible_instances,
            }
        )

    return {
        "run_id": run_dir.name,
        "aligned_frames": len(aligned),
        "unmatched": int(unmatched),
        "visible_frames": visible_frames,
        "not_visible_frames": not_visible_frames,
        "visible_with_true_confirmed": visible_with_true_confirmed,
        "observations": observations,
        "tracks": track_rows,
    }


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [obs for pair in pairs for obs in pair["observations"]]
    labels = [int(obs["label"]) for obs in observations]
    positives = sum(labels)
    negatives = len(labels) - positives

    feature_names = [
        "detector_confidence",
        "rank_score",
        "association_confidence",
        "confidence_rank_in_confirmed",
        "confirmed_competitors",
        "track_observations",
        "track_streak",
        "track_mean_confidence",
        "track_area_cv",
        "track_speed_px_per_step",
        "box_area",
    ]
    features: dict[str, Any] = {}
    for name in feature_names:
        values = [float(obs[name]) for obs in observations]
        pos_values = [value for value, label in zip(values, labels) if label == 1]
        neg_values = [value for value, label in zip(values, labels) if label == 0]
        raw_auc = _auc(values, labels)
        features[name] = {
            "true_median": _median(pos_values),
            "false_median": _median(neg_values),
            "auc_raw": None if raw_auc is None else round(raw_auc, 4),
            "auc_best_direction": None if raw_auc is None else round(max(raw_auc, 1.0 - raw_auc), 4),
            "best_univariate_at_90_precision": _best_recall_at_precision(values, labels, 0.90),
        }

    all_tracks = [track for pair in pairs for track in pair["tracks"]]
    maturity = {}
    for minimum_instances in (3, 5, 8, 12, 20):
        eligible = [track for track in all_tracks if int(track["instances"]) >= minimum_instances]
        true = sum(int(track["true_instances"]) for track in eligible)
        total = sum(int(track["instances"]) for track in eligible)
        maturity[str(minimum_instances)] = {
            "tracks": len(eligible),
            "instance_precision_pct": round(100.0 * true / max(total, 1), 2),
            "true_instances": true,
            "instances": total,
        }

    total_visible = sum(int(pair["visible_frames"]) for pair in pairs)
    confirmed_true_frames = sum(int(pair["visible_with_true_confirmed"]) for pair in pairs)
    track_purity_buckets = Counter()
    for track in all_tracks:
        purity = float(track["purity_pct"])
        if purity >= 90.0:
            bucket = ">=90%"
        elif purity >= 50.0:
            bucket = "50-<90%"
        elif purity > 0.0:
            bucket = ">0-<50%"
        else:
            bucket = "0%"
        track_purity_buckets[bucket] += 1

    return {
        "scope": "OC-SORT+CIoU confirmed-hypothesis identity-signal analysis",
        "confirmed_observations": len(observations),
        "true_confirmed_observations": positives,
        "false_confirmed_observations": negatives,
        "confirmed_instance_precision_pct": round(100.0 * positives / max(len(observations), 1), 2),
        "confirmed_true_frame_ceiling_pct": round(100.0 * confirmed_true_frames / max(total_visible, 1), 2),
        "features": features,
        "track_maturity_curve": maturity,
        "track_purity_buckets": dict(track_purity_buckets),
        "note": (
            "Feature labels use GT only for evaluation. The measured features themselves do not use GT. "
            "best_univariate_at_90_precision is a diagnostic separability bound, not a production threshold recommendation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether OC-SORT confirmed ball hypotheses expose enough non-GT signal to distinguish "
            "true-ball observations from false tracks before adding court/player context."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--output", default="evaluation/p0_ocsort_identity_signal_analysis")
    args = parser.parse_args()

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    pairs = []
    for run_id, gt_id in args.pair:
        pairs.append(
            analyze_pair(
                run_root / run_id,
                gt_root / gt_id / "annotations.jsonl",
                frame_rate=float(args.frame_rate),
                radius_px=float(args.gt_radius_px),
            )
        )

    summary = summarize(pairs)
    summary["runs"] = [
        {key: value for key, value in pair.items() if key not in {"observations", "tracks"}}
        for pair in pairs
    ]
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("P0 OC-SORT BALL IDENTITY SIGNAL ANALYSIS")
    print("=" * 104)
    print(f"confirmed observations:       {summary['confirmed_observations']:7d}")
    print(f"confirmed instance precision: {summary['confirmed_instance_precision_pct']:7.2f}%")
    print(f"true-frame ceiling:            {summary['confirmed_true_frame_ceiling_pct']:7.2f}%")
    print("\nFEATURE SEPARABILITY")
    print("-" * 104)
    print(f"{'feature':30s} {'true_med':>10s} {'false_med':>10s} {'best_auc':>9s} {'recall@P90':>11s}")
    for name, metrics in summary["features"].items():
        true_med = metrics["true_median"]
        false_med = metrics["false_median"]
        best_auc = metrics["auc_best_direction"]
        p90 = metrics["best_univariate_at_90_precision"]["recall_pct"]
        print(
            f"{name:30s} "
            f"{('n/a' if true_med is None else f'{true_med:.3f}'):>10s} "
            f"{('n/a' if false_med is None else f'{false_med:.3f}'):>10s} "
            f"{('n/a' if best_auc is None else f'{best_auc:.3f}'):>9s} "
            f"{p90:10.2f}%"
        )

    print("\nTRACK MATURITY DIAGNOSTIC")
    print("-" * 72)
    for minimum, metrics in summary["track_maturity_curve"].items():
        print(
            f"instances>={minimum:>2s}: tracks={metrics['tracks']:4d} "
            f"precision={metrics['instance_precision_pct']:6.2f}% "
            f"true_instances={metrics['true_instances']:5d}/{metrics['instances']:5d}"
        )
    print(f"\ntrack purity buckets: {summary['track_purity_buckets']}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
