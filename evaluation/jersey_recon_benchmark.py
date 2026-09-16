from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UNKNOWN_VALUES = {None, "", "UNKNOWN", "UNREADABLE", "NONE", "N/A", "NA"}


def canonical_number(value: Any) -> str | None:
    if value in UNKNOWN_VALUES:
        return None
    text = str(value).strip().upper()
    if text in UNKNOWN_VALUES:
        return None
    if text == "00":
        return "00"
    if not text.isdigit():
        raise ValueError(f"invalid jersey number: {value!r}")
    number = int(text)
    if not 0 <= number <= 99:
        raise ValueError(f"jersey number outside 0..99: {value!r}")
    return str(number)


def roi_iou(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    if len(a) != 4 or len(b) != 4:
        raise ValueError("ROI must be [x1, y1, x2, y2]")
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        raise ValueError("ROI coordinates must define a positive-area box")
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Row:
    track_id: str
    gt_number: str | None
    visible: bool
    oracle_roi: list[float] | None
    auto_roi: list[float] | None
    oracle_prediction: str | None
    auto_prediction: str | None
    final_track_prediction: str | None

    @classmethod
    def from_json(cls, payload: dict[str, Any], line_number: int) -> "Row":
        if "track_id" not in payload:
            raise ValueError(f"line {line_number}: missing track_id")
        gt = canonical_number(payload.get("gt_number"))
        visible = bool(payload.get("visible", False))
        if visible and gt is None:
            raise ValueError(f"line {line_number}: visible track requires gt_number")
        return cls(
            track_id=str(payload["track_id"]),
            gt_number=gt,
            visible=visible,
            oracle_roi=payload.get("oracle_roi"),
            auto_roi=payload.get("auto_roi"),
            oracle_prediction=canonical_number(payload.get("oracle_prediction")),
            auto_prediction=canonical_number(payload.get("auto_prediction")),
            final_track_prediction=canonical_number(payload.get("final_track_prediction")),
        )


def read_manifest(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number}: expected JSON object")
        rows.append(Row.from_json(payload, line_number))
    if not rows:
        raise ValueError(f"manifest contains no benchmark rows: {path}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        if row.track_id in seen:
            duplicates.add(row.track_id)
        seen.add(row.track_id)
    if duplicates:
        raise ValueError("track_id must be unique; duplicates: " + ", ".join(sorted(duplicates)))
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _prediction_metrics(rows: Iterable[Row], field: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.visible and row.gt_number is not None]
    predictions = [getattr(row, field) for row in eligible]
    predicted = sum(value is not None for value in predictions)
    correct = sum(
        value is not None and value == row.gt_number
        for row, value in zip(eligible, predictions, strict=True)
    )
    return {
        "eligible": len(eligible),
        "predicted": predicted,
        "correct": correct,
        "coverage": _ratio(predicted, len(eligible)),
        # UNKNOWN/abstention counts as incorrect here: this is end-to-end conditional
        # accuracy given that a human says the number is actually visible.
        "accuracy_on_visible": _ratio(correct, len(eligible)),
        "accuracy_when_predicted": _ratio(correct, predicted),
    }


def score(rows: list[Row], iou_threshold: float) -> dict[str, Any]:
    labeled = [row for row in rows if row.gt_number is not None]
    visible = [row for row in labeled if row.visible]

    roi_rows: list[tuple[Row, float]] = []
    for row in visible:
        overlap = roi_iou(row.oracle_roi, row.auto_roi)
        if overlap is not None:
            roi_rows.append((row, overlap))
    roi_hits = sum(overlap >= iou_threshold for _, overlap in roi_rows)

    oracle = _prediction_metrics(rows, "oracle_prediction")
    automatic = _prediction_metrics(rows, "auto_prediction")

    final_predicted = sum(row.final_track_prediction is not None for row in labeled)
    final_correct = sum(
        row.final_track_prediction is not None
        and row.final_track_prediction == row.gt_number
        for row in labeled
    )

    oracle_accuracy = oracle["accuracy_on_visible"]
    auto_accuracy = automatic["accuracy_on_visible"]
    input_gap = (
        round(float(oracle_accuracy) - float(auto_accuracy), 6)
        if oracle_accuracy is not None and auto_accuracy is not None
        else None
    )

    return {
        "schema": "courtflux_jersey_recon_benchmark_v1",
        "tracks": {
            "total": len(rows),
            "labeled": len(labeled),
            "visible": len(visible),
            "visibility_rate": _ratio(len(visible), len(labeled)),
        },
        "roi_localization": {
            "iou_threshold": iou_threshold,
            "evaluable": len(roi_rows),
            "hits": roi_hits,
            "recall": _ratio(roi_hits, len(roi_rows)),
            "mean_iou": (
                round(sum(overlap for _, overlap in roi_rows) / len(roi_rows), 6)
                if roi_rows
                else None
            ),
        },
        "oracle_recognizer": oracle,
        "automatic_recognizer": automatic,
        "oracle_minus_auto_accuracy_gap": input_gap,
        "final_track_identification": {
            "eligible": len(labeled),
            "predicted": final_predicted,
            "correct": final_correct,
            "coverage": _ratio(final_predicted, len(labeled)),
            "accuracy": _ratio(final_correct, len(labeled)),
            "accuracy_when_predicted": _ratio(final_correct, final_predicted),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score Jersey reconstruction stages independently: visibility, ROI retrieval, "
            "recognition on human-selected oracle crops, automatic recognition and final "
            "track-level identity."
        )
    )
    parser.add_argument("manifest", type=Path, help="JSONL benchmark manifest")
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be between 0 and 1")

    result = score(read_manifest(args.manifest), args.iou_threshold)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
