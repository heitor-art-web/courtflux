from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ultralytics import YOLO

import evaluation.detector_multiclass_calibration as base


def _format_class_counts(counts: Counter[str], requested_classes: list[str]) -> str:
    ordered = [
        f"{class_name}={counts.get(class_name, 0)}"
        for class_name in requested_classes
    ]
    return " ".join(ordered)


def run_detector_once_safe(
    *,
    label: str,
    model_path: Path,
    image_paths: list[Path],
    ground_truth_by_path: dict[str, list[dict[str, Any]]],
    requested_classes: list[str],
    args: Any,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one image at a time so Ultralytics NMS never sees the full split as one batch.

    The calibration intentionally uses a very low inference confidence. Feeding the entire
    split as a list can create a large NMS batch and trigger Ultralytics' time-limit guard,
    which may truncate detections and invalidate the low-confidence threshold sweep.
    """
    model = YOLO(str(model_path))
    model_names = base.names_dict(model.names)
    canonical_model_names = sorted(
        {base.normalize_name(value) for value in model_names.values()}
    )
    records: list[dict[str, Any]] = []
    detections_out = output_dir / f"detections_{label}.jsonl"
    cumulative_counts: Counter[str] = Counter()

    with detections_out.open("w", encoding="utf-8") as handle:
        for index, source_path in enumerate(image_paths, start=1):
            results = model.predict(
                source=str(source_path),
                stream=False,
                conf=args.inference_conf,
                iou=args.nms_iou,
                imgsz=args.imgsz,
                device=args.device,
                max_det=args.max_det,
                verbose=False,
            )
            if len(results) != 1:
                raise RuntimeError(
                    f"Expected exactly one result for {source_path}, got {len(results)}"
                )
            result = results[0]
            image_path = Path(result.path).resolve()
            preds = base.predictions_from_result(result, model_names)
            row = {
                "image": str(image_path),
                "ground_truth": ground_truth_by_path[str(image_path)],
                "predictions": preds,
            }
            records.append(row)
            handle.write(json.dumps(row) + "\n")

            frame_counts: Counter[str] = Counter(
                pred["class_name"]
                for pred in preds
                if pred["class_name"] in requested_classes
            )
            cumulative_counts.update(frame_counts)

            if index == 1 or index % 25 == 0 or index == len(image_paths):
                print(
                    f"[{label}] [{index}/{len(image_paths)}] "
                    f"{100.0 * index / max(len(image_paths), 1):.1f}% "
                    f"frame[{_format_class_counts(frame_counts, requested_classes)} total={sum(frame_counts.values())}] "
                    f"cumulative[{_format_class_counts(cumulative_counts, requested_classes)} total={sum(cumulative_counts.values())}]",
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
        "inference_mode": "single_image_to_avoid_nms_batch_timeout",
        "low_conf_detection_counts_by_class": dict(cumulative_counts),
    }
    return records, diagnostics


def main() -> None:
    base.run_detector_once = run_detector_once_safe
    base.main()


if __name__ == "__main__":
    main()
