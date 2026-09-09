from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.p0_ocsort_identity_signal_analysis import _auc, analyze_pair, parse_pair
from evaluation.p0_ocsort_multivariate_arbiter_cv import (
    FEATURES,
    _fit_logistic,
    _matrix,
    _predict,
    _threshold_for_precision,
)
from services.vision.ball_hypothesis_tracker import IDENTITY_FEATURES


def _out_of_fold_predictions(pairs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    probabilities: list[float] = []
    labels: list[float] = []
    aucs: list[float] = []
    for index, test_pair in enumerate(pairs):
        train_pairs = [pair for j, pair in enumerate(pairs) if j != index]
        train_observations = [obs for pair in train_pairs for obs in pair["observations"]]
        test_observations = list(test_pair["observations"])
        x_train, y_train = _matrix(train_observations)
        x_test, y_test = _matrix(test_observations)
        model = _fit_logistic(x_train, y_train)
        test_probability = _predict(model, x_test)
        probabilities.extend(test_probability.tolist())
        labels.extend(y_test.tolist())
        auc = _auc(test_probability.tolist(), y_test.astype(int).tolist())
        if auc is not None:
            aucs.append(float(auc))
    return (
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        aucs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the frozen OC-SORT temporal identity seed model on V1-V5. The output "
            "is an experimental seed/veto model, not a standalone ball detector and not "
            "the P0 quality-gate definition."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument(
        "--seed-precision-target",
        type=float,
        default=0.90,
        help=(
            "Conservative operating point for high-confidence seeds only. This is "
            "independent from the P0 system output-precision gate."
        ),
    )
    parser.add_argument(
        "--output",
        default="evaluation/p0_ocsort_seed_model/temporal_seed_model.json",
    )
    args = parser.parse_args()

    if FEATURES != IDENTITY_FEATURES:
        raise SystemExit(
            "Feature schema drift between evaluation and production hypothesis tracker: "
            f"{FEATURES} != {IDENTITY_FEATURES}"
        )
    if not 0.5 <= float(args.seed_precision_target) <= 1.0:
        raise SystemExit("--seed-precision-target must be between 0.5 and 1.0")

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
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
        raise SystemExit("Need at least two run/GT pairs")

    oof_probability, oof_labels, oof_aucs = _out_of_fold_predictions(pairs)
    calibration = _threshold_for_precision(
        oof_probability,
        oof_labels,
        float(args.seed_precision_target),
    )
    threshold = float(calibration["threshold"])
    if not np.isfinite(threshold):
        raise SystemExit(
            "No out-of-fold threshold reached the requested seed precision target. "
            "Do not export a model; lower the target only as an explicit experiment."
        )

    all_observations = [obs for pair in pairs for obs in pair["observations"]]
    x_all, y_all = _matrix(all_observations)
    final_model = _fit_logistic(x_all, y_all)
    final_probability = _predict(final_model, x_all)
    final_auc = _auc(final_probability.tolist(), y_all.astype(int).tolist())

    payload = {
        "version": "ocsort_temporal_seed_v1",
        "scope": (
            "High-confidence temporal seed/veto model for OC-SORT hypotheses. "
            "Not the P0 quality gate and not sufficient as full ball-state output."
        ),
        "features": FEATURES,
        "feature_transform": "p0_ocsort_multivariate_arbiter_cv_v1",
        "mean": [float(v) for v in final_model["mean"].tolist()],
        "scale": [float(v) for v in final_model["scale"].tolist()],
        "weights": [float(v) for v in final_model["weights"].tolist()],
        "threshold": threshold,
        "seed_precision_target": float(args.seed_precision_target),
        "oof_precision_pct": round(100.0 * float(calibration["precision"]), 2),
        "oof_recall_pct": round(100.0 * float(calibration["recall"]), 2),
        "oof_auc_macro": (
            None if not oof_aucs else round(float(np.mean(oof_aucs)), 4)
        ),
        "fit_auc": None if final_auc is None else round(float(final_auc), 4),
        "training_runs": [pair["run_id"] for pair in pairs],
        "gt_radius_px": float(args.gt_radius_px),
        "safety": {
            "lost_can_emit_concrete_output": False,
            "tentative_can_emit_concrete_output": False,
            "hard_reacquisition": False,
        },
    }

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("P0 OC-SORT TEMPORAL SEED MODEL")
    print("=" * 78)
    print(f"OOF AUC macro:        {payload['oof_auc_macro']}")
    print(f"OOF seed precision:   {payload['oof_precision_pct']:.2f}%")
    print(f"OOF seed recall:      {payload['oof_recall_pct']:.2f}%")
    print(f"seed threshold:       {payload['threshold']:.6f}")
    print(f"model:                {output}")
    print("\nThis threshold is for conservative seeds only; P0 system output precision is a separate gate.")


if __name__ == "__main__":
    main()
