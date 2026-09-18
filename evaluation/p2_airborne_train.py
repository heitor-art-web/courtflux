from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.p2.airborne_classifier_v1 import AirborneClassifierV1

CLASSES = AirborneClassifierV1.CLASSES
FEATURES = AirborneClassifierV1.FEATURE_NAMES
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(dict(row.get("features", {}))[name]) for name in FEATURES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([CLASS_TO_INDEX[str(row.get("label"))] for row in rows], dtype=np.int64)
    return x, y


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _predict(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    minimum_probability: float,
) -> np.ndarray:
    probabilities = _softmax(x @ weights.T + bias)
    indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    unresolved = CLASS_TO_INDEX["unresolved"]
    return np.where(confidence >= minimum_probability, indices, unresolved)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, guess in zip(y.tolist(), pred.tolist()):
        confusion[int(truth), int(guess)] += 1

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, label in enumerate(CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[label] = {
            "support": int(confusion[index, :].sum()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "rows": int(len(y)),
        "accuracy": round(float((y == pred).mean()) if len(y) else 0.0, 4),
        "macro_f1": round(sum(f1_values) / max(len(f1_values), 1), 4),
        "per_class": per_class,
        "confusion": {
            truth_label: {
                pred_label: int(confusion[truth_index, pred_index])
                for pred_index, pred_label in enumerate(CLASSES)
            }
            for truth_index, truth_label in enumerate(CLASSES)
        },
    }


def _sample_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    """Give comparable influence to intervals, videos and classes.

    Frame rows are highly autocorrelated. First divide each human interval's total weight
    equally across its rows, then normalize each source video to equal total influence,
    finally class-balance the resulting effective mass.
    """
    interval_counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        key = (str(row.get("video_id") or ""), int(row.get("truth_interval_index", -1)))
        interval_counts[key] += 1

    base: list[float] = []
    video_mass: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        video_id = str(row.get("video_id") or "")
        key = (video_id, int(row.get("truth_interval_index", -1)))
        weight = 1.0 / max(interval_counts[key], 1)
        base.append(weight)
        video_mass[video_id] += weight

    video_balanced: list[float] = []
    for row, weight in zip(rows, base):
        video_id = str(row.get("video_id") or "")
        video_balanced.append(weight / max(video_mass[video_id], 1e-12))

    class_mass: defaultdict[str, float] = defaultdict(float)
    for row, weight in zip(rows, video_balanced):
        class_mass[str(row.get("label"))] += weight
    total_mass = sum(video_balanced)
    class_factor = {
        label: total_mass / (len(CLASSES) * max(class_mass[label], 1e-12))
        for label in CLASSES
    }
    result = np.asarray(
        [
            weight * class_factor[str(row.get("label"))]
            for row, weight in zip(rows, video_balanced)
        ],
        dtype=np.float64,
    )
    return result / max(float(result.mean()), 1e-12)


def _fit(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("cannot fit on an empty dataset")
    counts = Counter(str(row.get("label")) for row in rows)
    missing = [label for label in CLASSES if counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"training fold missing classes: {missing}")

    x_raw, y = _matrix(rows)
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x = (x_raw - mean) / scale

    weights = np.zeros((len(CLASSES), len(FEATURES)), dtype=np.float64)
    bias = np.zeros(len(CLASSES), dtype=np.float64)
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[y]
    sample_weight = _sample_weights(rows)

    for _ in range(max(1, int(epochs))):
        probabilities = _softmax(x @ weights.T + bias)
        error = (probabilities - one_hot) * sample_weight[:, None]
        grad_w = error.T @ x / max(len(x), 1) + float(l2) * weights
        grad_b = error.mean(axis=0)
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b

    return mean, scale, weights, bias


def _predict_rows(
    rows: list[dict[str, Any]],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    minimum_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_raw, y = _matrix(rows)
    x = (x_raw - mean) / scale
    pred = _predict(x, weights, bias, minimum_probability=minimum_probability)
    return y, pred


def train(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    minimum_probability: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("dataset is empty")

    video_ids = sorted({str(row.get("video_id") or "") for row in rows})
    if len(video_ids) < 3:
        raise ValueError("grouped cross-validation requires at least three videos")

    overall_counts = Counter(str(row.get("label")) for row in rows)
    missing = [label for label in CLASSES if overall_counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"development dataset missing classes: {missing}")

    fold_reports: list[dict[str, Any]] = []
    cv_truth: list[int] = []
    cv_pred: list[int] = []

    for held_out in video_ids:
        train_rows = [row for row in rows if str(row.get("video_id")) != held_out]
        test_rows = [row for row in rows if str(row.get("video_id")) == held_out]
        mean, scale, weights, bias = _fit(
            train_rows,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        truth, pred = _predict_rows(
            test_rows,
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            minimum_probability=minimum_probability,
        )
        cv_truth.extend(int(value) for value in truth.tolist())
        cv_pred.extend(int(value) for value in pred.tolist())
        fold_reports.append(
            {
                "held_out_video_id": held_out,
                "train_video_ids": [value for value in video_ids if value != held_out],
                "metrics": _metrics(truth, pred),
            }
        )

    cv_truth_array = np.asarray(cv_truth, dtype=np.int64)
    cv_pred_array = np.asarray(cv_pred, dtype=np.int64)
    cv_report = _metrics(cv_truth_array, cv_pred_array)

    mean, scale, weights, bias = _fit(
        rows,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    train_truth, train_pred = _predict_rows(
        rows,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=bias,
        minimum_probability=minimum_probability,
    )

    interval_count = len(
        {
            (str(row.get("video_id")), int(row.get("truth_interval_index", -1)))
            for row in rows
        }
    )
    report = {
        "contract": "V1-V5 development corpus; leave-one-video-out grouped cross-validation; V6-V7 final validation only",
        "development_video_ids": video_ids,
        "label_counts": dict(sorted(overall_counts.items())),
        "truth_intervals": interval_count,
        "minimum_probability": float(minimum_probability),
        "feature_time_basis": "FPS-normalized physical time",
        "training_weighting": "equal interval mass -> equal video mass -> class balance",
        "grouped_cross_validation": {
            "aggregate": cv_report,
            "folds": fold_reports,
        },
        "fit_on_all_V1_V5": _metrics(train_truth, train_pred),
    }

    artifact = {
        "version": AirborneClassifierV1.VERSION,
        "classes": list(CLASSES),
        "features": list(FEATURES),
        "feature_mean": [round(float(value), 10) for value in mean.tolist()],
        "feature_scale": [round(float(value), 10) for value in scale.tolist()],
        "weights": [
            [round(float(value), 10) for value in row]
            for row in weights.tolist()
        ],
        "bias": [round(float(value), 10) for value in bias.tolist()],
        "minimum_probability": round(float(minimum_probability), 6),
        "training": {
            "algorithm": "interval_video_class_balanced_multinomial_logistic_regression",
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "l2": float(l2),
            "rows": len(rows),
            "truth_intervals": interval_count,
            "development_video_ids": video_ids,
            "development_clips": ["V1", "V2", "V3", "V4", "V5"],
            "grouped_cross_validation": "leave_one_video_out",
            "feature_time_basis": "FPS-normalized physical time",
            "final_validation_clips": ["V6", "V7"],
        },
        "development_report": report,
    }
    return artifact, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AirborneClassifierV1 on V1-V5 with grouped video cross-validation."
    )
    parser.add_argument("--dataset", default="evaluation/airborne_dev/airborne_dataset.jsonl")
    parser.add_argument("--output", default="weights/p2_airborne_classifier_v1.json")
    parser.add_argument("--report", default="evaluation/airborne_dev/training_report.json")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--minimum-probability", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.minimum_probability <= 1.0):
        raise ValueError("--minimum-probability must be in (0, 1]")

    dataset = Path(args.dataset)
    output = Path(args.output)
    report_path = Path(args.report)
    if not dataset.is_absolute():
        dataset = PROJECT_ROOT / dataset
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    rows = _load_jsonl(dataset)
    artifact, report = train(
        rows,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        minimum_probability=args.minimum_probability,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"model={output}")


if __name__ == "__main__":
    main()
