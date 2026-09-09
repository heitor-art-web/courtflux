from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.p0_ocsort_identity_signal_analysis import _auc, analyze_pair, parse_pair

FEATURES = [
    "detector_confidence",
    "rank_score",
    "confidence_rank_in_confirmed",
    "confirmed_competitors",
    "track_observations",
    "track_streak",
    "track_mean_confidence",
    "track_area_cv",
    "track_speed_px_per_step",
    "box_area",
]


def _feature_vector(obs: dict[str, Any]) -> list[float]:
    return [
        float(obs["detector_confidence"]),
        float(obs["rank_score"]),
        float(obs["confidence_rank_in_confirmed"]),
        float(obs["confirmed_competitors"]),
        math.log1p(max(float(obs["track_observations"]), 0.0)),
        math.log1p(max(float(obs["track_streak"]), 0.0)),
        float(obs["track_mean_confidence"]),
        math.log1p(max(float(obs["track_area_cv"]), 0.0)),
        math.log1p(max(float(obs["track_speed_px_per_step"]), 0.0)),
        math.log1p(max(float(obs["box_area"]), 0.0)),
    ]


def _matrix(observations: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([_feature_vector(obs) for obs in observations], dtype=np.float64)
    y = np.asarray([int(obs["label"]) for obs in observations], dtype=np.float64)
    return x, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 0.02,
    learning_rate: float = 0.08,
    steps: int = 1800,
) -> dict[str, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    z = np.column_stack([np.ones(len(z), dtype=np.float64), z])

    positives = max(int(np.sum(y == 1.0)), 1)
    negatives = max(int(np.sum(y == 0.0)), 1)
    sample_weights = np.where(
        y == 1.0,
        len(y) / (2.0 * positives),
        len(y) / (2.0 * negatives),
    )

    weights = np.zeros(z.shape[1], dtype=np.float64)
    for _ in range(steps):
        p = _sigmoid(z @ weights)
        residual = (p - y) * sample_weights
        grad = (z.T @ residual) / max(len(y), 1)
        grad[1:] += ridge * weights[1:]
        weights -= learning_rate * grad

    return {"mean": mean, "scale": scale, "weights": weights}


def _predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    z = np.column_stack([np.ones(len(z), dtype=np.float64), z])
    return _sigmoid(z @ model["weights"])


def _threshold_for_precision(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target: float = 0.90,
) -> dict[str, float]:
    order = np.argsort(-probabilities, kind="mergesort")
    p = probabilities[order]
    y = labels[order]
    tp = np.cumsum(y == 1.0)
    fp = np.cumsum(y == 0.0)
    precision = tp / np.maximum(tp + fp, 1)
    total_pos = max(int(np.sum(y == 1.0)), 1)
    recall = tp / total_pos
    valid = np.where((precision >= target) & (tp > 0))[0]
    if len(valid) == 0:
        return {"threshold": float("inf"), "precision": 0.0, "recall": 0.0}
    best = valid[int(np.argmax(recall[valid]))]
    return {
        "threshold": float(p[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def _evaluate_fold(
    train_pairs: list[dict[str, Any]],
    test_pair: dict[str, Any],
) -> dict[str, Any]:
    train_obs = [obs for pair in train_pairs for obs in pair["observations"]]
    test_obs = list(test_pair["observations"])
    x_train, y_train = _matrix(train_obs)
    x_test, y_test = _matrix(test_obs)

    model = _fit_logistic(x_train, y_train)
    train_prob = _predict(model, x_train)
    test_prob = _predict(model, x_test)
    calibration = _threshold_for_precision(train_prob, y_train, 0.90)
    threshold = float(calibration["threshold"])

    auc = _auc(test_prob.tolist(), y_test.astype(int).tolist())

    # Production-like arbitration: at most one concrete ball output per frame.
    by_step: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for probability, obs in zip(test_prob.tolist(), test_obs):
        by_step.setdefault(int(obs["step"]), []).append((float(probability), obs))

    selected = []
    for step, candidates in by_step.items():
        probability, obs = max(candidates, key=lambda item: item[0])
        if probability >= threshold:
            selected.append((step, probability, obs))

    selected_true = sum(int(obs["label"]) for _, _, obs in selected)
    selected_total = len(selected)
    visible_true_steps = {
        step for step, _, obs in selected if bool(obs["visible"]) and int(obs["label"]) == 1
    }
    not_visible_false_steps = {
        step for step, _, obs in selected if not bool(obs["visible"])
    }

    visible_frames = int(test_pair["visible_frames"])
    not_visible_frames = int(test_pair["not_visible_frames"])
    return {
        "test_run_id": test_pair["run_id"],
        "train_calibration": {
            "threshold": None if not math.isfinite(threshold) else round(threshold, 6),
            "precision_pct": round(100.0 * calibration["precision"], 2),
            "recall_pct": round(100.0 * calibration["recall"], 2),
        },
        "test_observation_auc": None if auc is None else round(float(auc), 4),
        "selected_output_frames": selected_total,
        "selected_true_frames": selected_true,
        "visible_frames": visible_frames,
        "not_visible_frames": not_visible_frames,
        "output_precision_pct": round(100.0 * selected_true / max(selected_total, 1), 2),
        "visible_coverage_pct": round(100.0 * len(visible_true_steps) / max(visible_frames, 1), 2),
        "not_visible_false_state_pct": round(
            100.0 * len(not_visible_false_steps) / max(not_visible_frames, 1), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-video-out diagnostic for a multivariate OC-SORT ball identity arbiter. "
            "Threshold is calibrated to >=90% precision on four training videos and applied "
            "unchanged to the held-out fifth video. No court/player context is used."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--output", default="evaluation/p0_ocsort_multivariate_arbiter_cv")
    args = parser.parse_args()

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    pairs = [
        analyze_pair(
            run_root / run_id,
            gt_root / gt_id / "annotations.jsonl",
            frame_rate=float(args.frame_rate),
            radius_px=float(args.gt_radius_px),
        )
        for run_id, gt_id in args.pair
    ]
    if len(pairs) < 2:
        raise SystemExit("Need at least two run/GT pairs for held-out validation")

    folds = []
    for idx, test_pair in enumerate(pairs):
        train_pairs = [pair for j, pair in enumerate(pairs) if j != idx]
        folds.append(_evaluate_fold(train_pairs, test_pair))

    selected_total = sum(int(f["selected_output_frames"]) for f in folds)
    selected_true = sum(int(f["selected_true_frames"]) for f in folds)
    visible_total = sum(int(f["visible_frames"]) for f in folds)
    not_visible_total = sum(int(f["not_visible_frames"]) for f in folds)
    # Derive counts back from percentages only where exact frame counts are already stored.
    visible_true_total = sum(
        int(round(float(f["visible_coverage_pct"]) * int(f["visible_frames"]) / 100.0))
        for f in folds
    )
    nv_false_total = sum(
        int(round(float(f["not_visible_false_state_pct"]) * int(f["not_visible_frames"]) / 100.0))
        for f in folds
    )

    aucs = [float(f["test_observation_auc"]) for f in folds if f["test_observation_auc"] is not None]
    summary = {
        "scope": "LOVO multivariate OC-SORT identity arbiter; temporal/detector features only",
        "features": FEATURES,
        "folds": folds,
        "aggregate": {
            "test_auc_macro": None if not aucs else round(float(np.mean(aucs)), 4),
            "output_precision_micro_pct": round(100.0 * selected_true / max(selected_total, 1), 2),
            "visible_coverage_micro_pct": round(100.0 * visible_true_total / max(visible_total, 1), 2),
            "not_visible_false_state_micro_pct": round(100.0 * nv_false_total / max(not_visible_total, 1), 2),
            "selected_output_frames": selected_total,
        },
        "decision_rule": (
            "If held-out precision cannot stay near 90% with meaningful coverage, temporal/detector "
            "features are insufficient for Ball Identity and court/player context should be added."
        ),
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("P0 OC-SORT MULTIVARIATE BALL IDENTITY — LEAVE-ONE-VIDEO-OUT")
    print("=" * 102)
    for fold in folds:
        cal = fold["train_calibration"]
        print(
            f"{fold['test_run_id']}  "
            f"AUC={fold['test_observation_auc']}  "
            f"trainP={cal['precision_pct']:.2f}%/R={cal['recall_pct']:.2f}%  "
            f"testP={fold['output_precision_pct']:.2f}%  "
            f"cov={fold['visible_coverage_pct']:.2f}%  "
            f"NVfalse={fold['not_visible_false_state_pct']:.2f}%"
        )
    agg = summary["aggregate"]
    print("\nAGGREGATE")
    print("-" * 72)
    print(f"held-out AUC macro:        {agg['test_auc_macro']}")
    print(f"output precision micro:    {agg['output_precision_micro_pct']:.2f}%")
    print(f"visible coverage micro:    {agg['visible_coverage_micro_pct']:.2f}%")
    print(f"not-visible false state:   {agg['not_visible_false_state_micro_pct']:.2f}%")
    print(f"selected output frames:    {agg['selected_output_frames']}")
    print(f"summary: {path}")


if __name__ == "__main__":
    main()
