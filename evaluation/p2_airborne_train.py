from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from services.p2.airborne_classifier_v1 import AirborneClassifierV1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _softmax(x @ weights.T + bias)
    indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    unresolved = CLASS_TO_INDEX["unresolved"]
    indices = np.where(confidence >= minimum_probability, indices, unresolved)
    return indices, probabilities


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


def train(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    minimum_probability: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_rows = [row for row in rows if str(row.get("split")) == "train"]
    calibration_rows = [row for row in rows if str(row.get("split")) == "calibration"]
    if not train_rows or not calibration_rows:
        raise ValueError("dataset requires non-empty train and calibration splits")

    train_videos = {str(row.get("video_id")) for row in train_rows}
    calibration_videos = {str(row.get("video_id")) for row in calibration_rows}
    overlap = train_videos & calibration_videos
    if overlap:
        raise ValueError(f"train/calibration video leakage: {sorted(overlap)}")

    train_counts = Counter(str(row.get("label")) for row in train_rows)
    missing_train = [label for label in CLASSES if train_counts.get(label, 0) == 0]
    if missing_train:
        raise ValueError(f"training split missing classes: {missing_train}")
    calibration_counts = Counter(str(row.get("label")) for row in calibration_rows)
    for required in ("pass_flight", "shot_flight"):
        if calibration_counts.get(required, 0) == 0:
            raise ValueError(f"calibration split needs at least one {required} row")

    x_train_raw, y_train = _matrix(train_rows)
    x_cal_raw, y_cal = _matrix(calibration_rows)
    mean = x_train_raw.mean(axis=0)
    scale = x_train_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_train = (x_train_raw - mean) / scale
    x_cal = (x_cal_raw - mean) / scale

    weights = np.zeros((len(CLASSES), len(FEATURES)), dtype=np.float64)
    bias = np.zeros(len(CLASSES), dtype=np.float64)
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[y_train]
    class_weight = {
        CLASS_TO_INDEX[label]: len(train_rows) / (len(CLASSES) * count)
        for label, count in train_counts.items()
    }
    sample_weight = np.asarray([class_weight[int(index)] for index in y_train], dtype=np.float64)
    sample_weight = sample_weight / sample_weight.mean()

    for _ in range(max(1, int(epochs))):
        probabilities = _softmax(x_train @ weights.T + bias)
        error = (probabilities - one_hot) * sample_weight[:, None]
        grad_w = error.T @ x_train / max(len(x_train), 1) + float(l2) * weights
        grad_b = error.mean(axis=0)
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b

    train_pred, _ = _predict(
        x_train, weights, bias, minimum_probability=minimum_probability
    )
    cal_pred, _ = _predict(
        x_cal, weights, bias, minimum_probability=minimum_probability
    )
    report = {
        "train": _metrics(y_train, train_pred),
        "calibration": _metrics(y_cal, cal_pred),
        "train_label_counts": dict(sorted(train_counts.items())),
        "calibration_label_counts": dict(sorted(calibration_counts.items())),
        "train_video_ids": sorted(train_videos),
        "calibration_video_ids": sorted(calibration_videos),
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
            "algorithm": "class_balanced_multinomial_logistic_regression_full_batch_gradient_descent",
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "l2": float(l2),
            "train_rows": len(train_rows),
            "calibration_rows": len(calibration_rows),
            "train_video_ids": sorted(train_videos),
            "calibration_video_ids": sorted(calibration_videos),
            "V1_V5_used_for_training_or_calibration": false,
        },
        "development_report": report,
    }
    return artifact, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AirborneClassifierV1 on disjoint development data.")
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
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"model={output}")


if __name__ == "__main__":
    main()
