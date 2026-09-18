from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.p2.airborne_classifier_v1_1 import AirborneClassifierV1

FEATURES = AirborneClassifierV1.FEATURE_NAMES
FINAL_LABELS = ("continuity", "pass_flight", "shot_flight")


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


def _matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(dict(row.get("features", {}))[name]) for name in FEATURES] for row in rows],
        dtype=np.float64,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _sample_weights(rows: list[dict[str, Any]], labels: list[str]) -> np.ndarray:
    video_counts = Counter(str(row.get("video_id") or "") for row in rows)
    base = np.asarray(
        [1.0 / max(video_counts[str(row.get("video_id") or "")], 1) for row in rows],
        dtype=np.float64,
    )
    class_mass: defaultdict[str, float] = defaultdict(float)
    for label, weight in zip(labels, base.tolist()):
        class_mass[label] += float(weight)
    total = float(base.sum())
    classes = sorted(set(labels))
    factors = {
        label: total / (len(classes) * max(class_mass[label], 1e-12)) for label in classes
    }
    result = np.asarray(
        [weight * factors[label] for label, weight in zip(labels, base.tolist())],
        dtype=np.float64,
    )
    return result / max(float(result.mean()), 1e-12)


def _fit_binary(
    rows: list[dict[str, Any]],
    *,
    label_fn: Callable[[dict[str, Any]], str],
    negative_label: str,
    positive_label: str,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot fit binary stage on empty rows")
    labels = [label_fn(row) for row in rows]
    counts = Counter(labels)
    if counts.get(negative_label, 0) == 0 or counts.get(positive_label, 0) == 0:
        raise ValueError(
            f"binary stage requires both classes: {negative_label}={counts.get(negative_label, 0)}, "
            f"{positive_label}={counts.get(positive_label, 0)}"
        )
    x_raw = _matrix(rows)
    y = np.asarray([1.0 if label == positive_label else 0.0 for label in labels], dtype=np.float64)
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x = (x_raw - mean) / scale
    sample_weight = _sample_weights(rows, labels)
    weights = np.zeros(len(FEATURES), dtype=np.float64)
    bias = 0.0
    for _ in range(max(1, int(epochs))):
        probability = _sigmoid(x @ weights + bias)
        error = (probability - y) * sample_weight
        grad_w = x.T @ error / max(len(x), 1) + float(l2) * weights
        grad_b = float(error.mean())
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b
    return {
        "labels": (negative_label, positive_label),
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "bias": float(bias),
    }


def _probability(features: dict[str, Any], model: dict[str, Any]) -> float:
    vector = np.asarray([float(features[name]) for name in FEATURES], dtype=np.float64)
    x = (vector - model["mean"]) / model["scale"]
    logit = float(x @ model["weights"] + model["bias"])
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _binary_metrics(
    truth: list[str],
    pred: list[str],
    *,
    negative_label: str,
    positive_label: str,
) -> dict[str, Any]:
    labels = (negative_label, positive_label)
    confusion = {label: {guess: 0 for guess in labels} for label in labels}
    for expected, guess in zip(truth, pred):
        confusion[expected][guess] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
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
    accuracy = sum(int(a == b) for a, b in zip(truth, pred)) / max(len(truth), 1)
    return {
        "episodes": len(truth),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "per_class": per_class,
        "confusion": confusion,
    }


def _stage1_predict(rows: list[dict[str, Any]], model: dict[str, Any], threshold: float) -> list[str]:
    return [
        "airborne" if _probability(dict(row["features"]), model) >= threshold else "continuity"
        for row in rows
    ]


def _stage2_predict(rows: list[dict[str, Any]], model: dict[str, Any], threshold: float) -> list[str]:
    result: list[str] = []
    for row in rows:
        shot_probability = _probability(dict(row["features"]), model)
        confidence = max(shot_probability, 1.0 - shot_probability)
        if confidence < threshold:
            result.append("unresolved")
        else:
            result.append("shot_flight" if shot_probability >= 0.5 else "pass_flight")
    return result


def _stage2_metrics(truth: list[str], pred: list[str]) -> dict[str, Any]:
    labels = ("pass_flight", "shot_flight")
    confusion = {label: {guess: 0 for guess in (*labels, "unresolved")} for label in labels}
    for expected, guess in zip(truth, pred):
        confusion[expected][guess] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in (*labels, "unresolved") if other != label)
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
    accuracy = sum(int(a == b) for a, b in zip(truth, pred)) / max(len(truth), 1)
    abstention = sum(int(value == "unresolved") for value in pred) / max(len(pred), 1)
    return {
        "episodes": len(truth),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "abstention_rate": round(abstention, 4),
        "per_class": per_class,
        "confusion": confusion,
    }


def _evaluable_final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("stage1_label") == "continuity" or row.get("stage2_label") in AirborneClassifierV1.STAGE2_LABELS
    ]


def _final_truth(row: dict[str, Any]) -> str:
    return "continuity" if row.get("stage1_label") == "continuity" else str(row.get("stage2_label"))


def _terminal_prediction(
    row: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    threshold: float,
) -> str:
    features = dict(row["features"])
    airborne_probability = _probability(features, stage1)
    if airborne_probability < threshold:
        return "continuity"
    shot_probability = _probability(features, stage2)
    if max(shot_probability, 1.0 - shot_probability) < threshold:
        return "continuity"
    return "shot_flight" if shot_probability >= 0.5 else "pass_flight"


def _causal_prediction(
    row: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    threshold: float,
) -> tuple[str, float | None]:
    for prefix in list(row.get("prefixes", []) or []):
        features = dict(prefix.get("features", {}) or {})
        if not features:
            continue
        airborne_probability = _probability(features, stage1)
        if airborne_probability < threshold:
            continue
        shot_probability = _probability(features, stage2)
        if max(shot_probability, 1.0 - shot_probability) < threshold:
            continue
        label = "shot_flight" if shot_probability >= 0.5 else "pass_flight"
        return label, float(prefix.get("elapsed_sec", 0.0) or 0.0)
    return "continuity", None


def _final_metrics(truth: list[str], pred: list[str], latencies: list[float | None] | None = None) -> dict[str, Any]:
    labels = FINAL_LABELS
    confusion = {label: {guess: 0 for guess in labels} for label in labels}
    for expected, guess in zip(truth, pred):
        confusion[expected][guess] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
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
    result = {
        "episodes": len(truth),
        "accuracy": round(sum(int(a == b) for a, b in zip(truth, pred)) / max(len(truth), 1), 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "per_class": per_class,
        "confusion": confusion,
    }
    if latencies is not None:
        correct_airborne_latencies = [
            latency
            for expected, guess, latency in zip(truth, pred, latencies)
            if expected in ("pass_flight", "shot_flight") and expected == guess and latency is not None
        ]
        result["correct_airborne_decision_latency_sec"] = {
            "count": len(correct_airborne_latencies),
            "mean": None
            if not correct_airborne_latencies
            else round(sum(correct_airborne_latencies) / len(correct_airborne_latencies), 4),
            "max": None if not correct_airborne_latencies else round(max(correct_airborne_latencies), 4),
        }
    return result


def _fit_stages(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1 = _fit_binary(
        rows,
        label_fn=lambda row: str(row.get("stage1_label")),
        negative_label="continuity",
        positive_label="airborne",
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    stage2_rows = [row for row in rows if row.get("stage2_label") in AirborneClassifierV1.STAGE2_LABELS]
    stage2 = _fit_binary(
        stage2_rows,
        label_fn=lambda row: str(row.get("stage2_label")),
        negative_label="pass_flight",
        positive_label="shot_flight",
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    return stage1, stage2


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    stage1_truth = [str(row.get("stage1_label")) for row in rows]
    stage1_pred = _stage1_predict(rows, stage1, threshold)

    stage2_rows = [row for row in rows if row.get("stage2_label") in AirborneClassifierV1.STAGE2_LABELS]
    stage2_truth = [str(row.get("stage2_label")) for row in stage2_rows]
    stage2_pred = _stage2_predict(stage2_rows, stage2, threshold)

    final_rows = _evaluable_final_rows(rows)
    final_truth = [_final_truth(row) for row in final_rows]
    terminal_pred = [_terminal_prediction(row, stage1, stage2, threshold) for row in final_rows]
    causal_pairs = [_causal_prediction(row, stage1, stage2, threshold) for row in final_rows]
    causal_pred = [label for label, _latency in causal_pairs]
    latencies = [latency for _label, latency in causal_pairs]

    return {
        "stage1_airborne_vs_continuity": _binary_metrics(
            stage1_truth,
            stage1_pred,
            negative_label="continuity",
            positive_label="airborne",
        ),
        "stage2_pass_vs_shot": _stage2_metrics(stage2_truth, stage2_pred),
        "terminal_episode_end_to_end": _final_metrics(final_truth, terminal_pred),
        "causal_episode_end_to_end": _final_metrics(final_truth, causal_pred, latencies),
    }


def _stage_artifact(model: dict[str, Any], threshold: float) -> dict[str, Any]:
    return {
        "labels": list(model["labels"]),
        "feature_mean": [round(float(value), 10) for value in model["mean"].tolist()],
        "feature_scale": [round(float(value), 10) for value in model["scale"].tolist()],
        "weights": [round(float(value), 10) for value in model["weights"].tolist()],
        "bias": round(float(model["bias"]), 10),
        "minimum_probability": round(float(threshold), 6),
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
        raise ValueError("episode dataset is empty")
    video_ids = sorted({str(row.get("video_id") or "") for row in rows})
    if len(video_ids) < 3:
        raise ValueError("leave-one-video-out CV requires at least three videos")

    fold_reports: list[dict[str, Any]] = []
    aggregate_truth: list[str] = []
    aggregate_terminal: list[str] = []
    aggregate_causal: list[str] = []
    aggregate_latencies: list[float | None] = []
    stage1_truth_all: list[str] = []
    stage1_pred_all: list[str] = []
    stage2_truth_all: list[str] = []
    stage2_pred_all: list[str] = []

    for held_out in video_ids:
        train_rows = [row for row in rows if str(row.get("video_id")) != held_out]
        test_rows = [row for row in rows if str(row.get("video_id")) == held_out]
        stage1, stage2 = _fit_stages(
            train_rows,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        metrics = _evaluate(
            test_rows,
            stage1=stage1,
            stage2=stage2,
            threshold=minimum_probability,
        )

        stage1_truth = [str(row.get("stage1_label")) for row in test_rows]
        stage1_pred = _stage1_predict(test_rows, stage1, minimum_probability)
        stage1_truth_all.extend(stage1_truth)
        stage1_pred_all.extend(stage1_pred)
        s2_rows = [row for row in test_rows if row.get("stage2_label") in AirborneClassifierV1.STAGE2_LABELS]
        stage2_truth_all.extend(str(row.get("stage2_label")) for row in s2_rows)
        stage2_pred_all.extend(_stage2_predict(s2_rows, stage2, minimum_probability))
        final_rows = _evaluable_final_rows(test_rows)
        aggregate_truth.extend(_final_truth(row) for row in final_rows)
        aggregate_terminal.extend(
            _terminal_prediction(row, stage1, stage2, minimum_probability) for row in final_rows
        )
        causal = [_causal_prediction(row, stage1, stage2, minimum_probability) for row in final_rows]
        aggregate_causal.extend(label for label, _ in causal)
        aggregate_latencies.extend(latency for _, latency in causal)

        fold_reports.append(
            {
                "held_out_video_id": held_out,
                "train_video_ids": [video for video in video_ids if video != held_out],
                "train_episodes": len(train_rows),
                "test_episodes": len(test_rows),
                "metrics": metrics,
            }
        )

    stage1_final, stage2_final = _fit_stages(
        rows,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    fit_all = _evaluate(
        rows,
        stage1=stage1_final,
        stage2=stage2_final,
        threshold=minimum_probability,
    )

    aggregate = {
        "stage1_airborne_vs_continuity": _binary_metrics(
            stage1_truth_all,
            stage1_pred_all,
            negative_label="continuity",
            positive_label="airborne",
        ),
        "stage2_pass_vs_shot": _stage2_metrics(stage2_truth_all, stage2_pred_all),
        "terminal_episode_end_to_end": _final_metrics(aggregate_truth, aggregate_terminal),
        "causal_episode_end_to_end": _final_metrics(
            aggregate_truth, aggregate_causal, aggregate_latencies
        ),
    }

    stage1_counts = Counter(str(row.get("stage1_label")) for row in rows)
    stage2_counts = Counter(
        str(row.get("stage2_label")) for row in rows if row.get("stage2_label") in AirborneClassifierV1.STAGE2_LABELS
    )
    report = {
        "contract": "AirborneClassifierV1.1; one flight episode = one training unit; causal prefix replay; leave-one-video-out V1-V5; V6-V7 validation-only",
        "development_video_ids": video_ids,
        "episodes": len(rows),
        "stage1_label_counts": dict(sorted(stage1_counts.items())),
        "stage2_label_counts": dict(sorted(stage2_counts.items())),
        "minimum_probability": float(minimum_probability),
        "training_weighting": "equal episode unit -> equal video mass -> binary class balance per stage",
        "grouped_cross_validation": {
            "aggregate": aggregate,
            "folds": fold_reports,
        },
        "fit_on_all_V1_V5": fit_all,
    }

    artifact = {
        "version": AirborneClassifierV1.VERSION,
        "classes": list(AirborneClassifierV1.CLASSES),
        "features": list(FEATURES),
        "stages": {
            "airborne": _stage_artifact(stage1_final, minimum_probability),
            "type": _stage_artifact(stage2_final, minimum_probability),
        },
        "training": {
            "algorithm": "episode_level_video_balanced_two_stage_logistic_regression",
            "episodes": len(rows),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "l2": float(l2),
            "minimum_probability": float(minimum_probability),
            "development_clips": ["V1", "V2", "V3", "V4", "V5"],
            "grouped_cross_validation": "leave_one_video_out",
            "runtime_evaluation": "causal_prefix_replay_first_committed_decision",
            "final_validation_clips": ["V6", "V7"],
        },
        "development_report": report,
    }
    return artifact, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AirborneClassifierV1.1 on episode-level V1-V5 data."
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
    if not 0.5 <= args.minimum_probability <= 1.0:
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
