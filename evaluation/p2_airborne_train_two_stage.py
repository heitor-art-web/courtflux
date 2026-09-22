from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.p2.airborne_classifier_two_stage_v1 import AirborneClassifierV1

FEATURES = AirborneClassifierV1.FEATURE_NAMES
OUTPUT_CLASSES = AirborneClassifierV1.CLASSES
OUTPUT_INDEX = {label: index for index, label in enumerate(OUTPUT_CLASSES)}


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


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(dict(row.get("features", {}))[name]) for name in FEATURES] for row in rows],
        dtype=np.float64,
    )


def _stage1_target(row: dict[str, Any]) -> int:
    return 1 if str(row.get("label")) in {"pass_flight", "shot_flight"} else 0


def _stage2_target(row: dict[str, Any]) -> int:
    return 1 if str(row.get("label")) == "shot_flight" else 0


def _output_target(row: dict[str, Any]) -> int:
    return OUTPUT_INDEX[str(row.get("label"))]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _sample_weights(
    rows: list[dict[str, Any]],
    target_fn: Callable[[dict[str, Any]], int],
) -> np.ndarray:
    """Equal interval mass -> equal video mass -> binary class balance."""
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

    class_mass: defaultdict[int, float] = defaultdict(float)
    for row, weight in zip(rows, video_balanced):
        class_mass[target_fn(row)] += weight
    if class_mass[0] <= 0.0 or class_mass[1] <= 0.0:
        raise ValueError("binary training stage requires both classes")

    total_mass = sum(video_balanced)
    class_factor = {
        value: total_mass / (2.0 * max(class_mass[value], 1e-12)) for value in (0, 1)
    }
    result = np.asarray(
        [weight * class_factor[target_fn(row)] for row, weight in zip(rows, video_balanced)],
        dtype=np.float64,
    )
    return result / max(float(result.mean()), 1e-12)


def _fit_binary(
    rows: list[dict[str, Any]],
    *,
    target_fn: Callable[[dict[str, Any]], int],
    learning_rate: float,
    epochs: int,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not rows:
        raise ValueError("cannot fit binary stage on an empty dataset")
    y = np.asarray([target_fn(row) for row in rows], dtype=np.float64)
    if len(set(int(value) for value in y.tolist())) < 2:
        raise ValueError("binary training stage requires both classes")

    x_raw = _feature_matrix(rows)
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x = (x_raw - mean) / scale
    weights = np.zeros(len(FEATURES), dtype=np.float64)
    bias = 0.0
    sample_weight = _sample_weights(rows, target_fn)

    for _ in range(max(1, int(epochs))):
        probabilities = _sigmoid(x @ weights + bias)
        error = (probabilities - y) * sample_weight
        grad_w = (x.T @ error) / max(len(x), 1) + float(l2) * weights
        grad_b = float(error.mean())
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b

    return mean, scale, weights, bias


def _binary_probabilities(
    rows: list[dict[str, Any]],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    x = (_feature_matrix(rows) - mean) / scale
    return _sigmoid(x @ weights + bias)


def _binary_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    labels: tuple[str, str],
) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for expected, actual in zip(truth.tolist(), pred.tolist()):
        confusion[int(expected), int(actual)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, label in enumerate(labels):
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
        "rows": int(len(truth)),
        "accuracy": round(float((truth == pred).mean()) if len(truth) else 0.0, 4),
        "macro_f1": round(sum(f1_values) / 2.0, 4),
        "per_class": per_class,
        "confusion": {
            labels[truth_index]: {
                labels[pred_index]: int(confusion[truth_index, pred_index])
                for pred_index in range(2)
            }
            for truth_index in range(2)
        },
    }


def _stage2_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    labels = ("pass_flight", "shot_flight")
    confusion = {
        label: {"pass_flight": 0, "shot_flight": 0, "unresolved": 0} for label in labels
    }
    for expected, actual in zip(truth.tolist(), pred.tolist()):
        truth_label = labels[int(expected)]
        pred_label = "unresolved" if int(actual) < 0 else labels[int(actual)]
        confusion[truth_label][pred_label] += 1

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, label in enumerate(labels):
        other = labels[1 - index]
        tp = confusion[label][label]
        fp = confusion[other][label]
        fn = confusion[label][other] + confusion[label]["unresolved"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    unresolved = sum(row["unresolved"] for row in confusion.values())
    correct = confusion["pass_flight"]["pass_flight"] + confusion["shot_flight"]["shot_flight"]
    return {
        "rows": int(len(truth)),
        "accuracy": round(correct / max(len(truth), 1), 4),
        "macro_f1": round(sum(f1_values) / 2.0, 4),
        "abstention_rate": round(unresolved / max(len(truth), 1), 4),
        "per_class": per_class,
        "confusion": confusion,
    }


def _final_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((len(OUTPUT_CLASSES), len(OUTPUT_CLASSES)), dtype=np.int64)
    for expected, actual in zip(truth.tolist(), pred.tolist()):
        confusion[int(expected), int(actual)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, label in enumerate(OUTPUT_CLASSES):
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
        "rows": int(len(truth)),
        "accuracy": round(float((truth == pred).mean()) if len(truth) else 0.0, 4),
        "macro_f1": round(sum(f1_values) / max(len(f1_values), 1), 4),
        "per_class": per_class,
        "confusion": {
            truth_label: {
                pred_label: int(confusion[truth_index, pred_index])
                for pred_index, pred_label in enumerate(OUTPUT_CLASSES)
            }
            for truth_index, truth_label in enumerate(OUTPUT_CLASSES)
        },
    }


def _fit_stages(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    stage1 = _fit_binary(
        rows,
        target_fn=_stage1_target,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    airborne_rows = [row for row in rows if _stage1_target(row) == 1]
    stage2 = _fit_binary(
        airborne_rows,
        target_fn=_stage2_target,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return {"airborne": stage1, "type": stage2}


def _predict_stages(
    rows: list[dict[str, Any]],
    stages: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    *,
    minimum_probability: float,
) -> dict[str, np.ndarray]:
    stage1_mean, stage1_scale, stage1_weights, stage1_bias = stages["airborne"]
    p_airborne = _binary_probabilities(
        rows,
        mean=stage1_mean,
        scale=stage1_scale,
        weights=stage1_weights,
        bias=stage1_bias,
    )
    stage1_truth = np.asarray([_stage1_target(row) for row in rows], dtype=np.int64)
    stage1_pred = (p_airborne >= minimum_probability).astype(np.int64)

    stage2_mean, stage2_scale, stage2_weights, stage2_bias = stages["type"]
    p_shot = _binary_probabilities(
        rows,
        mean=stage2_mean,
        scale=stage2_scale,
        weights=stage2_weights,
        bias=stage2_bias,
    )
    type_confidence = np.maximum(p_shot, 1.0 - p_shot)
    stage2_all_pred = np.where(
        type_confidence < minimum_probability,
        -1,
        (p_shot >= 0.5).astype(np.int64),
    )

    final_truth = np.asarray([_output_target(row) for row in rows], dtype=np.int64)
    final_pred = np.zeros(len(rows), dtype=np.int64)
    for index in range(len(rows)):
        if stage1_pred[index] == 0 or stage2_all_pred[index] < 0:
            final_pred[index] = OUTPUT_INDEX["unresolved"]
        elif stage2_all_pred[index] == 0:
            final_pred[index] = OUTPUT_INDEX["pass_flight"]
        else:
            final_pred[index] = OUTPUT_INDEX["shot_flight"]

    airborne_indices = [index for index, row in enumerate(rows) if _stage1_target(row) == 1]
    stage2_truth = np.asarray([_stage2_target(rows[index]) for index in airborne_indices], dtype=np.int64)
    stage2_pred = np.asarray([stage2_all_pred[index] for index in airborne_indices], dtype=np.int64)
    return {
        "stage1_truth": stage1_truth,
        "stage1_pred": stage1_pred,
        "stage2_truth": stage2_truth,
        "stage2_pred": stage2_pred,
        "final_truth": final_truth,
        "final_pred": final_pred,
    }


def _report_predictions(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "stage1_airborne_vs_continuity": _binary_metrics(
            predictions["stage1_truth"],
            predictions["stage1_pred"],
            labels=AirborneClassifierV1.STAGE1_LABELS,
        ),
        "stage2_pass_vs_shot": _stage2_metrics(
            predictions["stage2_truth"], predictions["stage2_pred"]
        ),
        "end_to_end": _final_metrics(
            predictions["final_truth"], predictions["final_pred"]
        ),
    }


def _stage_artifact(
    fit: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    *,
    labels: tuple[str, str],
    minimum_probability: float,
) -> dict[str, Any]:
    mean, scale, weights, bias = fit
    return {
        "labels": list(labels),
        "feature_mean": [round(float(value), 10) for value in mean.tolist()],
        "feature_scale": [round(float(value), 10) for value in scale.tolist()],
        "weights": [round(float(value), 10) for value in weights.tolist()],
        "bias": round(float(bias), 10),
        "minimum_probability": round(float(minimum_probability), 6),
    }


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

    label_counts = Counter(str(row.get("label")) for row in rows)
    for label in OUTPUT_CLASSES:
        if label_counts.get(label, 0) == 0:
            raise ValueError(f"development dataset missing classifier class: {label}")

    folds: list[dict[str, Any]] = []
    aggregate_predictions: dict[str, list[int]] = {
        "stage1_truth": [],
        "stage1_pred": [],
        "stage2_truth": [],
        "stage2_pred": [],
        "final_truth": [],
        "final_pred": [],
    }
    for held_out in video_ids:
        train_rows = [row for row in rows if str(row.get("video_id")) != held_out]
        test_rows = [row for row in rows if str(row.get("video_id")) == held_out]
        stages = _fit_stages(
            train_rows,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        predictions = _predict_stages(
            test_rows,
            stages,
            minimum_probability=minimum_probability,
        )
        for key in aggregate_predictions:
            aggregate_predictions[key].extend(int(value) for value in predictions[key].tolist())
        folds.append(
            {
                "held_out_video_id": held_out,
                "train_video_ids": [value for value in video_ids if value != held_out],
                "metrics": _report_predictions(predictions),
            }
        )

    aggregate_numpy = {
        key: np.asarray(values, dtype=np.int64) for key, values in aggregate_predictions.items()
    }
    cv_report = _report_predictions(aggregate_numpy)

    final_stages = _fit_stages(
        rows,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    fit_predictions = _predict_stages(
        rows,
        final_stages,
        minimum_probability=minimum_probability,
    )
    fit_report = _report_predictions(fit_predictions)

    interval_count = len(
        {
            (str(row.get("video_id")), int(row.get("truth_interval_index", -1)))
            for row in rows
        }
    )
    report = {
        "contract": "V1-V5 development; Stage1 AIRBORNE-vs-CONTINUITY; Stage2 PASS-vs-SHOT; leave-one-video-out CV; V6-V7 validation-only",
        "development_video_ids": video_ids,
        "label_counts": dict(sorted(label_counts.items())),
        "truth_intervals": interval_count,
        "minimum_probability": float(minimum_probability),
        "feature_time_basis": "FPS-normalized physical time",
        "training_weighting": "equal interval mass -> equal video mass -> binary class balance per stage",
        "grouped_cross_validation": {
            "aggregate": cv_report,
            "folds": folds,
        },
        "fit_on_all_V1_V5": fit_report,
    }

    artifact = {
        "version": AirborneClassifierV1.VERSION,
        "classes": list(OUTPUT_CLASSES),
        "features": list(FEATURES),
        "stages": {
            "airborne": _stage_artifact(
                final_stages["airborne"],
                labels=AirborneClassifierV1.STAGE1_LABELS,
                minimum_probability=minimum_probability,
            ),
            "type": _stage_artifact(
                final_stages["type"],
                labels=AirborneClassifierV1.STAGE2_LABELS,
                minimum_probability=minimum_probability,
            ),
        },
        "training": {
            "algorithm": "two_stage_interval_video_balanced_binary_logistic_regression",
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "l2": float(l2),
            "rows": len(rows),
            "truth_intervals": interval_count,
            "development_video_ids": video_ids,
            "development_clips": ["V1", "V2", "V3", "V4", "V5"],
            "grouped_cross_validation": "leave_one_video_out",
            "final_validation_clips": ["V6", "V7"],
        },
        "development_report": report,
    }
    return artifact, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two-stage AirborneClassifierV1 on V1-V5 with grouped CV."
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
    if not (0.5 <= args.minimum_probability <= 1.0):
        raise ValueError("--minimum-probability must be in [0.5, 1.0]")
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
