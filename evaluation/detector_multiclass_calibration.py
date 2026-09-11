from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import yaml
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ("basketball", "player", "referee", "rim", "backboard")
DEFAULT_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


ALIASES = {
    "basketball": "basketball",
    "basketballs": "basketball",
    "ball": "basketball",
    "balls": "basketball",
    "player": "player",
    "players": "player",
    "basketball player": "player",
    "basketball players": "player",
    "ref": "referee",
    "refs": "referee",
    "referee": "referee",
    "referees": "referee",
    "official": "referee",
    "officials": "referee",
    "rim": "rim",
    "rims": "rim",
    "basketball rim": "rim",
    "basketball rims": "rim",
    "hoop": "rim",
    "hoops": "rim",
    "basketball hoop": "rim",
    "basketball hoops": "rim",
    "backboard": "backboard",
    "backboards": "backboard",
}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare model_v1.0.0 and model_v1.1.0 on a common supervised YOLO split "
            "with per-class threshold calibration and cross-class confusion analysis."
        )
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "YOLO data.yaml used for the common supervised benchmark. If omitted, the "
            "script tries the data path embedded in the model checkpoints."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-v100", default="weights/model_v1.0.0.pt")
    parser.add_argument("--model-v110", default="weights/model_v1.1.0.pt")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--inference-conf", type=float, default=0.005)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return ALIASES.get(text, text)


def names_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(names)}
    raise TypeError(f"Unsupported names format: {type(names).__name__}")


def resolve_data_yaml(explicit: str | None, model_paths: list[Path]) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates: list[Path] = []
    for model_path in model_paths:
        model = YOLO(str(model_path))
        ckpt = getattr(model, "ckpt", None) or {}
        train_args = ckpt.get("train_args") or {}
        embedded = train_args.get("data")
        if embedded:
            candidate = Path(str(embedded)).expanduser()
            candidates.append(candidate)
            if candidate.exists():
                return candidate.resolve()
        overrides = getattr(model, "overrides", {}) or {}
        embedded = overrides.get("data")
        if embedded:
            candidate = Path(str(embedded)).expanduser()
            candidates.append(candidate)
            if candidate.exists():
                return candidate.resolve()

    tried = ", ".join(str(path) for path in candidates) or "<none embedded>"
    raise FileNotFoundError(
        "Could not resolve the supervised dataset YAML automatically. "
        f"Embedded candidates: {tried}. Re-run with --data /path/to/data.yaml"
    )


def resolve_dataset_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    raw = config.get("path")
    if not raw:
        return data_yaml.parent.resolve()
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (data_yaml.parent / path).resolve()


def _resolve_path(value: str, *, root: Path, anchor: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [(root / path).resolve(), (anchor / path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def expand_split_images(data_yaml: Path, config: dict[str, Any], split: str) -> list[Path]:
    if split not in config:
        raise KeyError(f"Dataset YAML has no '{split}' split")
    root = resolve_dataset_root(data_yaml, config)
    raw = config[split]
    specs = raw if isinstance(raw, list) else [raw]
    images: list[Path] = []

    for spec in specs:
        path = _resolve_path(str(spec), root=root, anchor=data_yaml.parent)
        if path.is_dir():
            images.extend(
                sorted(
                    p.resolve()
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
            )
            continue
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image_path = _resolve_path(line, root=root, anchor=path.parent)
                if image_path.suffix.lower() in IMAGE_EXTS:
                    images.append(image_path.resolve())
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path.resolve())
            continue
        matches = [
            p.resolve()
            for p in root.glob(str(spec))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        images.extend(sorted(matches))

    unique = list(dict.fromkeys(images))
    if not unique:
        raise FileNotFoundError(f"No images resolved for split '{split}' from {data_yaml}")
    return unique


def infer_label_path(image_path: Path, dataset_root: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [idx for idx, value in enumerate(parts) if value == "images"]
    if image_indices:
        idx = image_indices[-1]
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        relative = Path(image_path.name)
    if relative.parts and relative.parts[0] == "images":
        relative = Path("labels", *relative.parts[1:])
    else:
        relative = Path("labels") / relative
    return (dataset_root / relative).with_suffix(".txt")


def read_yolo_ground_truth(
    image_path: Path,
    label_path: Path,
    dataset_names: dict[int, str],
) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    if not label_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"{label_path}:{line_no}: expected at least 5 YOLO fields")
        class_id = int(float(fields[0]))
        if class_id not in dataset_names:
            raise ValueError(f"{label_path}:{line_no}: unknown class id {class_id}")
        xc, yc, bw, bh = (float(value) for value in fields[1:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        rows.append(
            {
                "class_id": class_id,
                "raw_class_name": dataset_names[class_id],
                "class_name": normalize_name(dataset_names[class_id]),
                "box": Box(x1, y1, x2, y2).as_list(),
            }
        )
    return rows


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    intersection = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_match(
    gt: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    match_iou: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    pairs: list[tuple[float, float, int, int]] = []
    for gt_idx, truth in enumerate(gt):
        for pred_idx, pred in enumerate(predictions):
            overlap = iou(truth["box"], pred["box"])
            if overlap >= match_iou:
                pairs.append((overlap, float(pred.get("confidence", 0.0)), gt_idx, pred_idx))
    pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, _confidence, gt_idx, pred_idx in pairs:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, overlap))
    return matches, matched_gt, matched_pred


def binary_metrics(
    records: list[dict[str, Any]],
    class_name: str,
    threshold: float,
    match_iou: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    for row in records:
        gt = [item for item in row["ground_truth"] if item["class_name"] == class_name]
        preds = [
            item
            for item in row["predictions"]
            if item["class_name"] == class_name and item["confidence"] >= threshold
        ]
        matches, matched_gt, matched_pred = greedy_match(gt, preds, match_iou=match_iou)
        tp += len(matches)
        fp += len(preds) - len(matched_pred)
        fn += len(gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_curve_point(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in curve if row["f1"] is not None]
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            row["f1"],
            row["precision"] if row["precision"] is not None else -1.0,
            row["recall"] if row["recall"] is not None else -1.0,
            row["threshold"],
        ),
    )


def best_recall_at_precision(
    curve: list[dict[str, Any]], minimum_precision: float
) -> dict[str, Any] | None:
    valid = [
        row
        for row in curve
        if row["precision"] is not None
        and row["recall"] is not None
        and row["precision"] >= minimum_precision
    ]
    if not valid:
        return None
    return max(valid, key=lambda row: (row["recall"], row["precision"], row["threshold"]))


def best_precision_at_recall(
    curve: list[dict[str, Any]], minimum_recall: float
) -> dict[str, Any] | None:
    valid = [
        row
        for row in curve
        if row["precision"] is not None
        and row["recall"] is not None
        and row["recall"] >= minimum_recall
    ]
    if not valid:
        return None
    return max(valid, key=lambda row: (row["precision"], row["recall"], row["threshold"]))


def confusion_matrix(
    records: list[dict[str, Any]],
    classes: list[str],
    thresholds: dict[str, float],
    *,
    match_iou: float,
) -> dict[str, Any]:
    columns = list(classes) + ["__miss__"]
    rows = list(classes) + ["__background__"]
    matrix = {row: {column: 0 for column in columns} for row in rows}

    for record in records:
        gt = [item for item in record["ground_truth"] if item["class_name"] in classes]
        preds = [
            item
            for item in record["predictions"]
            if item["class_name"] in classes
            and item["confidence"] >= thresholds[item["class_name"]]
        ]
        matches, matched_gt, matched_pred = greedy_match(gt, preds, match_iou=match_iou)

        for gt_idx, pred_idx, _overlap in matches:
            matrix[gt[gt_idx]["class_name"]][preds[pred_idx]["class_name"]] += 1
        for gt_idx, truth in enumerate(gt):
            if gt_idx not in matched_gt:
                matrix[truth["class_name"]]["__miss__"] += 1
        for pred_idx, pred in enumerate(preds):
            if pred_idx not in matched_pred:
                matrix["__background__"][pred["class_name"]] += 1

    cross_confusions: dict[str, int] = {}
    for gt_class in classes:
        for pred_class in classes:
            if gt_class == pred_class:
                continue
            count = matrix[gt_class][pred_class]
            if count:
                cross_confusions[f"{gt_class}_as_{pred_class}"] = count

    return {
        "rows_gt": rows,
        "columns_pred": columns,
        "matrix": matrix,
        "cross_confusions": cross_confusions,
    }


def predictions_from_result(result: Any, model_names: dict[int, str]) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    rows: list[dict[str, Any]] = []
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    for box, confidence, raw_class in zip(boxes, confs, classes):
        class_id = int(raw_class)
        raw_name = model_names.get(class_id, str(class_id))
        rows.append(
            {
                "class_id": class_id,
                "raw_class_name": raw_name,
                "class_name": normalize_name(raw_name),
                "confidence": float(confidence),
                "box": [float(value) for value in box],
            }
        )
    return rows


def run_detector_once(
    *,
    label: str,
    model_path: Path,
    image_paths: list[Path],
    ground_truth_by_path: dict[str, list[dict[str, Any]]],
    requested_classes: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = YOLO(str(model_path))
    model_names = names_dict(model.names)
    canonical_model_names = sorted({normalize_name(value) for value in model_names.values()})
    records: list[dict[str, Any]] = []
    detections_out = output_dir / f"detections_{label}.jsonl"

    results = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        conf=args.inference_conf,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )

    with detections_out.open("w", encoding="utf-8") as handle:
        for index, result in enumerate(results, start=1):
            image_path = Path(result.path).resolve()
            preds = predictions_from_result(result, model_names)
            row = {
                "image": str(image_path),
                "ground_truth": ground_truth_by_path[str(image_path)],
                "predictions": preds,
            }
            records.append(row)
            handle.write(json.dumps(row) + "\n")
            if index == 1 or index % 100 == 0 or index == len(image_paths):
                print(
                    f"[{label}] [{index}/{len(image_paths)}] "
                    f"{100.0 * index / max(len(image_paths), 1):.1f}% predictions={len(preds)}",
                    flush=True,
                )

    diagnostics = {
        "model": str(model_path),
        "model_names": model_names,
        "canonical_model_names": canonical_model_names,
        "missing_requested_classes_in_model": [
            name for name in requested_classes if name not in canonical_model_names
        ],
        "low_conf_detection_file": str(detections_out),
    }
    return records, diagnostics


def summarize_model(
    *,
    records: list[dict[str, Any]],
    classes: list[str],
    thresholds: list[float],
    match_iou: float,
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    best_thresholds: dict[str, float] = {}

    for class_name in classes:
        curve = [
            binary_metrics(records, class_name, threshold, match_iou)
            for threshold in thresholds
        ]
        best = best_curve_point(curve)
        if best is not None:
            best_thresholds[class_name] = float(best["threshold"])
        per_class[class_name] = {
            "curve": curve,
            "best_by_f1": best,
            "best_recall_at_precision_080": best_recall_at_precision(curve, 0.80),
            "best_recall_at_precision_090": best_recall_at_precision(curve, 0.90),
            "best_precision_at_recall_080": best_precision_at_recall(curve, 0.80),
        }

    default_threshold = min(thresholds)
    matrix_thresholds = {
        class_name: best_thresholds.get(class_name, default_threshold)
        for class_name in classes
    }
    matrix = confusion_matrix(
        records,
        classes,
        matrix_thresholds,
        match_iou=match_iou,
    )

    return {
        "per_class": per_class,
        "best_thresholds_by_f1": matrix_thresholds,
        "confusion_at_best_f1_thresholds": matrix,
    }


def metric_delta(new: dict[str, Any] | None, old: dict[str, Any] | None) -> dict[str, Any] | None:
    if not new or not old:
        return None
    result: dict[str, Any] = {}
    for field in ("precision", "recall", "f1"):
        a, b = new.get(field), old.get(field)
        result[field] = a - b if a is not None and b is not None else None
    result["threshold_new"] = new.get("threshold")
    result["threshold_old"] = old.get("threshold")
    result["tp_delta"] = new.get("tp", 0) - old.get("tp", 0)
    result["fp_delta"] = new.get("fp", 0) - old.get("fp", 0)
    result["fn_delta"] = new.get("fn", 0) - old.get("fn", 0)
    return result


def main() -> None:
    args = parse_args()
    model_v100 = Path(args.model_v100).expanduser().resolve()
    model_v110 = Path(args.model_v110).expanduser().resolve()
    for path in (model_v100, model_v110):
        if not path.exists():
            raise FileNotFoundError(path)

    data_yaml = resolve_data_yaml(args.data, [model_v100, model_v110])
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in dataset YAML: {data_yaml}")
    dataset_names = names_dict(config.get("names"))
    canonical_dataset_names = sorted({normalize_name(value) for value in dataset_names.values()})

    requested_classes = [
        normalize_name(value)
        for value in str(args.classes).split(",")
        if value.strip()
    ]
    requested_classes = list(dict.fromkeys(requested_classes))
    available_classes = [
        name for name in requested_classes if name in canonical_dataset_names
    ]
    missing_dataset_classes = [
        name for name in requested_classes if name not in canonical_dataset_names
    ]
    if not available_classes:
        raise ValueError(
            "None of the requested classes exist in the dataset. "
            f"Dataset names: {canonical_dataset_names}"
        )

    thresholds = sorted(
        {
            float(value.strip())
            for value in str(args.thresholds).split(",")
            if value.strip()
        }
    )
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value")
    if args.inference_conf > min(thresholds):
        raise ValueError(
            f"--inference-conf ({args.inference_conf}) must be <= minimum sweep threshold ({min(thresholds)})"
        )

    image_paths = expand_split_images(data_yaml, config, args.split)
    dataset_root = resolve_dataset_root(data_yaml, config)
    ground_truth_by_path: dict[str, list[dict[str, Any]]] = {}
    missing_label_files = 0
    gt_counts: Counter[str] = Counter()
    for index, image_path in enumerate(image_paths, start=1):
        label_path = infer_label_path(image_path, dataset_root)
        if not label_path.exists():
            missing_label_files += 1
        truth = read_yolo_ground_truth(image_path, label_path, dataset_names)
        ground_truth_by_path[str(image_path)] = truth
        gt_counts.update(item["class_name"] for item in truth)
        if index == 1 or index % 500 == 0 or index == len(image_paths):
            print(
                f"[GT] [{index}/{len(image_paths)}] labels={sum(gt_counts.values())}",
                flush=True,
            )

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("evaluation")
        / "detector_multiclass_calibration"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records_v100, diagnostics_v100 = run_detector_once(
        label="model_v1.0.0",
        model_path=model_v100,
        image_paths=image_paths,
        ground_truth_by_path=ground_truth_by_path,
        requested_classes=available_classes,
        args=args,
        output_dir=output_dir,
    )
    records_v110, diagnostics_v110 = run_detector_once(
        label="model_v1.1.0",
        model_path=model_v110,
        image_paths=image_paths,
        ground_truth_by_path=ground_truth_by_path,
        requested_classes=available_classes,
        args=args,
        output_dir=output_dir,
    )

    summary_v100 = summarize_model(
        records=records_v100,
        classes=available_classes,
        thresholds=thresholds,
        match_iou=args.match_iou,
    )
    summary_v110 = summarize_model(
        records=records_v110,
        classes=available_classes,
        thresholds=thresholds,
        match_iou=args.match_iou,
    )

    comparison = {
        class_name: metric_delta(
            summary_v110["per_class"][class_name]["best_by_f1"],
            summary_v100["per_class"][class_name]["best_by_f1"],
        )
        for class_name in available_classes
    }

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": (
            "Common supervised multi-class detector calibration and confusion analysis "
            "for model_v1.0.0 vs model_v1.1.0"
        ),
        "dataset": str(data_yaml),
        "split": args.split,
        "image_count": len(image_paths),
        "dataset_names": dataset_names,
        "canonical_dataset_names": canonical_dataset_names,
        "requested_classes": requested_classes,
        "evaluated_classes": available_classes,
        "missing_requested_classes_in_dataset": missing_dataset_classes,
        "ground_truth_counts": dict(gt_counts),
        "missing_label_files": missing_label_files,
        "settings": {
            "inference_conf": args.inference_conf,
            "thresholds": thresholds,
            "match_iou": args.match_iou,
            "nms_iou": args.nms_iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "max_det": args.max_det,
        },
        "models": {
            "model_v1.0.0": {
                "diagnostics": diagnostics_v100,
                **summary_v100,
            },
            "model_v1.1.0": {
                "diagnostics": diagnostics_v110,
                **summary_v110,
            },
        },
        "delta_v110_minus_v100_at_each_models_best_f1": comparison,
        "notes": [
            "Each model is inferred once at the low inference_conf; threshold sweeps are computed offline from identical model outputs.",
            "Per-class precision/recall/F1 matches predictions only to same-class GT using one-to-one IoU matching.",
            "The confusion matrix is class-agnostic matching by IoU after applying each predicted class's best-F1 threshold; off-diagonal cells expose class confusion such as referee->player.",
            "Unmatched GT appears in the __miss__ column; unmatched predictions appear in the __background__ row.",
            "This evaluates detector calibration/class separation on the supervised split, not downstream temporal tracking quality.",
        ],
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest per-class detector calibration (F1)")
    print("class | v1.0.0 thr | P | R | F1 | v1.1.0 thr | P | R | F1 | delta F1")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for class_name in available_classes:
        a = summary_v100["per_class"][class_name]["best_by_f1"]
        b = summary_v110["per_class"][class_name]["best_by_f1"]

        def fmt(row: dict[str, Any] | None, key: str) -> str:
            if not row or row.get(key) is None:
                return "n/a"
            return f"{row[key]:.3f}"

        delta = comparison[class_name]
        delta_f1 = delta["f1"] if delta and delta["f1"] is not None else None
        delta_text = f"{delta_f1:+.3f}" if delta_f1 is not None else "n/a"
        print(
            f"{class_name} | {fmt(a, 'threshold')} | {fmt(a, 'precision')} | {fmt(a, 'recall')} | {fmt(a, 'f1')} | "
            f"{fmt(b, 'threshold')} | {fmt(b, 'precision')} | {fmt(b, 'recall')} | {fmt(b, 'f1')} | {delta_text}"
        )

    for model_label, model_summary in (
        ("v1.0.0", summary_v100),
        ("v1.1.0", summary_v110),
    ):
        cross = model_summary["confusion_at_best_f1_thresholds"]["cross_confusions"]
        print(f"\n{model_label} cross-class confusions at calibrated thresholds:")
        if not cross:
            print("  none")
        else:
            for key, value in sorted(cross.items(), key=lambda item: (-item[1], item[0])):
                print(f"  {key}: {value}")

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
