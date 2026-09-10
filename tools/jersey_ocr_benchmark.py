from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from services.vision.performance_jersey_analyzer import PerformanceJerseyAnalyzer


NUMERIC_COLUMNS = [
    "torso_sharpness",
    "torso_width",
    "torso_height",
    "player_width",
    "player_height",
    "player_conf",
    "team_score",
    "team_margin",
]


def _float_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "torso_width" in out.columns and "torso_height" in out.columns:
        out["torso_area"] = out["torso_width"] * out["torso_height"]
    return out


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if series.empty:
        return {key: None for key in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")}
    return {
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def _readability_auc(frame: pd.DataFrame) -> float | None:
    readable = frame["label_kind"] == "number"
    unreadable = frame["label_kind"] == "unreadable"
    subset = frame[readable | unreadable].copy()
    n_pos = int((subset["label_kind"] == "number").sum())
    n_neg = int((subset["label_kind"] == "unreadable").sum())
    if not n_pos or not n_neg:
        return None
    ranks = pd.to_numeric(subset["raw_probability"], errors="coerce").rank(method="average")
    pos_rank_sum = float(ranks[subset["label_kind"] == "number"].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _number_scores(
    analyzer: PerformanceJerseyAnalyzer,
    tens_probs: list[float],
    ones_probs: list[float],
) -> list[tuple[str, float]]:
    scored = [
        (str(number), analyzer._candidate_probability(str(number), tens_probs, ones_probs))
        for number in range(100)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _numeric_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame[frame["label_kind"] == "number"].copy()
    if numeric.empty:
        return {
            "rows": 0,
            "exact_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
            "mean_raw_probability": None,
            "median_raw_probability": None,
        }
    exact = numeric["raw_canonical"] == numeric["jersey_gt_canonical"]
    top3 = numeric["top3_hit"].astype(bool)
    top5 = numeric["top5_hit"].astype(bool)
    probabilities = pd.to_numeric(numeric["raw_probability"], errors="coerce")
    return {
        "rows": int(len(numeric)),
        "correct": int(exact.sum()),
        "exact_accuracy": float(exact.mean()),
        "top3_accuracy": float(top3.mean()),
        "top5_accuracy": float(top5.mean()),
        "mean_raw_probability": float(probabilities.mean()),
        "median_raw_probability": float(probabilities.median()),
    }


def _candidate_keep(frame: pd.DataFrame, candidate: dict[str, Any]) -> pd.Series:
    family = candidate.get("family")
    if family == "sharpness":
        return frame["torso_sharpness"] >= float(candidate["threshold"])
    if family == "torso_area":
        return frame["torso_area"] >= float(candidate["threshold"])
    if family == "torso_height":
        return frame["torso_height"] >= float(candidate["threshold"])
    if family == "torso_width":
        return frame["torso_width"] >= float(candidate["threshold"])
    if family == "sharpness_and_area":
        return (
            (frame["torso_sharpness"] >= float(candidate["sharpness_threshold"]))
            & (frame["torso_area"] >= float(candidate["area_threshold"]))
        )
    raise ValueError(f"Unsupported quality-gate family: {family}")


def _gate_metrics(frame: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any]:
    numeric = frame[frame["label_kind"] == "number"].copy()
    if numeric.empty:
        return {"rows": 0, "kept_rows": 0, "coverage": None, "accuracy_kept": None}
    keep = _candidate_keep(numeric, candidate).fillna(False).astype(bool)
    kept = numeric[keep]
    correct_all = numeric["raw_canonical"] == numeric["jersey_gt_canonical"]
    correct_kept = kept["raw_canonical"] == kept["jersey_gt_canonical"]
    return {
        "rows": int(len(numeric)),
        "kept_rows": int(keep.sum()),
        "rejected_rows": int((~keep).sum()),
        "coverage": float(keep.mean()),
        "baseline_accuracy": float(correct_all.mean()),
        "accuracy_kept": float(correct_kept.mean()) if len(kept) else None,
        "correct_rows_kept": int(correct_kept.sum()),
        "correct_recall_vs_all_numeric": _safe_div(float(correct_kept.sum()), float(len(numeric))),
    }


def _temporal_metrics(
    frame: pd.DataFrame,
    analyzer: PerformanceJerseyAnalyzer,
) -> dict[str, Any]:
    numeric = frame[frame["label_kind"] == "number"].copy()
    if numeric.empty:
        return {"groups": 0, "exact_accuracy": None, "multi_crop_groups": 0, "multi_crop_accuracy": None}

    def group_key(row: pd.Series) -> str:
        identity = str(row.get("identity_group", "") or "").strip()
        base = identity or f"{row.get('video', '')}|track:{row.get('track_id', '')}"
        # A confirmed tracker ID switch can legitimately leave multiple jersey labels on
        # one track. Keep temporal diagnostics label-homogeneous rather than mixing people.
        return f"{base}|gt:{row['jersey_gt_canonical']}"

    numeric["benchmark_group"] = numeric.apply(group_key, axis=1)
    rows = []
    for key, group in numeric.groupby("benchmark_group", sort=False):
        tens = np.mean(np.stack(group["_tens_probs"].to_list()), axis=0)
        ones = np.mean(np.stack(group["_ones_probs"].to_list()), axis=0)
        raw = analyzer._raw_prediction(tens.tolist(), ones.tolist())
        gt = str(group.iloc[0]["jersey_gt_canonical"])
        rows.append(
            {
                "group": key,
                "gt": gt,
                "prediction": str(raw.get("raw_canonical") or ""),
                "correct": str(raw.get("raw_canonical") or "") == gt,
                "crops": int(len(group)),
                "raw_probability": float(raw.get("raw_probability", 0.0) or 0.0),
            }
        )
    groups = pd.DataFrame(rows)
    multi = groups[groups["crops"] >= 2]
    return {
        "groups": int(len(groups)),
        "exact_accuracy": float(groups["correct"].mean()) if len(groups) else None,
        "multi_crop_groups": int(len(multi)),
        "multi_crop_accuracy": float(multi["correct"].mean()) if len(multi) else None,
        "note": (
            "Diagnostic upper bound using only human-readable numeric crops, grouped by identity/track and GT label. "
            "It is not production temporal accuracy because unreadable frames are excluded by definition."
        ),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    overall = report["numeric_metrics"]["all"]
    lines = [
        "# Jersey OCR ground-truth benchmark",
        "",
        "This is a roster-free benchmark of the local jersey checkpoint. Raw digit predictions are compared directly with human ground truth; roster decoding is deliberately excluded.",
        "",
        "## Dataset",
        "",
        f"- OCR player crops inferred: {report['dataset']['ocr_rows']}",
        f"- Readable numeric GT crops: {report['dataset']['numeric_rows']}",
        f"- Unreadable player crops: {report['dataset']['unreadable_rows']}",
        f"- Checkpoint: `{report['model']['checkpoint']}`",
        f"- Device: `{report['model']['device']}`",
        f"- Inference wall time: {report['model']['inference_seconds']:.3f}s",
        "",
        "## Raw OCR accuracy",
        "",
        f"- Exact accuracy: **{_pct(overall['exact_accuracy'])}** ({overall.get('correct', 0)}/{overall['rows']})",
        f"- Top-3 number accuracy: **{_pct(overall['top3_accuracy'])}**",
        f"- Top-5 number accuracy: **{_pct(overall['top5_accuracy'])}**",
        f"- Median raw probability on readable crops: {overall['median_raw_probability']:.4f}",
        "",
        "### By split",
        "",
        "| Split | Rows | Exact | Top-3 | Top-5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        item = report["numeric_metrics"][split]
        lines.append(
            f"| {split} | {item['rows']} | {_pct(item['exact_accuracy'])} | {_pct(item['top3_accuracy'])} | {_pct(item['top5_accuracy'])} |"
        )

    lines += [
        "",
        "### By visibility",
        "",
        "| Visibility | Rows | Exact | Top-3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for visibility, item in report["visibility_metrics"].items():
        lines.append(
            f"| {visibility} | {item['rows']} | {_pct(item['exact_accuracy'])} | {_pct(item['top3_accuracy'])} |"
        )

    lines += [
        "",
        "## Confidence vs readability",
        "",
        f"- Readability AUROC using raw probability alone: {_pct(report['confidence']['readability_auroc'])}",
        f"- Readable median raw probability: {report['confidence']['readable']['median']:.4f}",
        f"- Unreadable median raw probability: {report['confidence']['unreadable']['median']:.4f}",
        "",
        "A readability AUROC near 50% means raw model confidence does not separate readable from unreadable crops well; a high value would support an additional confidence-based abstention rule.",
        "",
        "## Temporal readable-crop aggregation",
        "",
        f"- Groups: {report['temporal']['groups']}",
        f"- Exact group accuracy: {_pct(report['temporal']['exact_accuracy'])}",
        f"- Multi-crop groups: {report['temporal']['multi_crop_groups']}",
        f"- Multi-crop exact accuracy: {_pct(report['temporal']['multi_crop_accuracy'])}",
        "",
        report["temporal"]["note"],
        "",
    ]

    if report.get("quality_gates"):
        lines += [
            "## Quality-gate effect on readable OCR accuracy",
            "",
            "Thresholds come from the existing quality-gate calibration report; they are not re-optimized against OCR accuracy here.",
            "",
            "| Target | Rule | All coverage | All kept accuracy | Test coverage | Test kept accuracy |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for gate in report["quality_gates"]:
            all_metrics = gate["metrics"]["all"]
            test_metrics = gate["metrics"]["test"]
            lines.append(
                f"| {100 * gate['recall_target']:.0f}% | {gate['rule']} | {_pct(all_metrics['coverage'])} | "
                f"{_pct(all_metrics['accuracy_kept'])} | {_pct(test_metrics['coverage'])} | {_pct(test_metrics['accuracy_kept'])} |"
            )
        lines.append("")

    lines += [
        "## Prediction distribution",
        "",
        "Most common raw predictions on readable GT crops:",
        "",
    ]
    for prediction, count in report["prediction_distribution"][:12]:
        lines.append(f"- `{prediction}`: {count}")

    lines += ["", "## Largest confusions", ""]
    if report["confusions"]:
        for item in report["confusions"][:15]:
            lines.append(f"- GT `{item['gt']}` → predicted `{item['prediction']}`: {item['count']}")
    else:
        lines.append("- None")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the local jersey OCR checkpoint against human ground truth")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--match-config", required=True)
    parser.add_argument("--model", default="weights/jersey_number_recon.pt")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quality-gate-report")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth).expanduser().resolve()
    match_config = Path(args.match_config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    for path in (gt_path, match_config, model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else gt_path.parent / "ocr_benchmark_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    required = {
        "sample_id", "subject_type", "jersey_gt", "visibility", "label_kind",
        "dataset_role", "split", "torso_crop", "torso_sharpness", "torso_width", "torso_height",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ground truth is missing required columns: {missing}")
    frame = _float_frame(frame)
    ocr = frame[
        (frame["dataset_role"] == "ocr")
        & (frame["subject_type"] == "player")
        & frame["label_kind"].isin(["number", "unreadable"])
    ].copy().reset_index(drop=True)
    if ocr.empty:
        raise ValueError("No OCR-eligible player rows found")
    ocr["jersey_gt_canonical"] = ocr["jersey_gt"].map(
        lambda value: str(int(value)) if str(value).isdigit() else ""
    )

    analyzer = PerformanceJerseyAnalyzer(
        profiling_path=str(match_config),
        device=args.device,
        model_path=str(model_path),
        sequence_length=8,
    )

    tensors: list[torch.Tensor] = []
    valid_indices: list[int] = []
    for index, row in ocr.iterrows():
        crop_path = gt_path.parent / str(row["torso_crop"])
        crop = cv2.imread(str(crop_path))
        if crop is None or crop.size == 0:
            raise FileNotFoundError(f"Missing/unreadable torso crop: {crop_path}")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(analyzer.transform(Image.fromarray(rgb)))
        valid_indices.append(index)

    batch_size = max(1, int(args.batch_size))
    tens_prob_rows: list[np.ndarray] = []
    ones_prob_rows: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(analyzer.device)
            tens_logits, ones_logits = analyzer.number_model(batch)
            tens_probs = torch.softmax(tens_logits, dim=1).detach().cpu().numpy()
            ones_probs = torch.softmax(ones_logits, dim=1).detach().cpu().numpy()
            tens_prob_rows.extend(tens_probs)
            ones_prob_rows.extend(ones_probs)
    inference_seconds = time.perf_counter() - started

    records = []
    for position, index in enumerate(valid_indices):
        row = ocr.iloc[index]
        tens = tens_prob_rows[position].astype(float).tolist()
        ones = ones_prob_rows[position].astype(float).tolist()
        raw = analyzer._raw_prediction(tens, ones)
        scored = _number_scores(analyzer, tens, ones)
        top_numbers = [number for number, _ in scored[:5]]
        gt = str(row["jersey_gt_canonical"])
        record = row.to_dict()
        record.update(raw)
        record["top1_product_number"] = top_numbers[0]
        record["top3_numbers"] = "|".join(top_numbers[:3])
        record["top5_numbers"] = "|".join(top_numbers)
        record["top3_hit"] = bool(gt and gt in top_numbers[:3])
        record["top5_hit"] = bool(gt and gt in top_numbers)
        record["exact_correct"] = bool(gt and str(raw.get("raw_canonical") or "") == gt)
        record["_tens_probs"] = np.asarray(tens, dtype=float)
        record["_ones_probs"] = np.asarray(ones, dtype=float)
        records.append(record)
    results = pd.DataFrame(records)

    numeric_metrics = {"all": _numeric_metrics(results)}
    for split in ("train", "val", "test"):
        numeric_metrics[split] = _numeric_metrics(results[results["split"] == split])

    visibility_metrics = {}
    for visibility in sorted(results.loc[results["label_kind"] == "number", "visibility"].unique()):
        visibility_metrics[str(visibility)] = _numeric_metrics(results[results["visibility"] == visibility])

    readable_prob = pd.to_numeric(results.loc[results["label_kind"] == "number", "raw_probability"], errors="coerce")
    unreadable_prob = pd.to_numeric(results.loc[results["label_kind"] == "unreadable", "raw_probability"], errors="coerce")

    numeric = results[results["label_kind"] == "number"].copy()
    prediction_counts = Counter(str(value or "invalid") for value in numeric["raw_canonical"].tolist())
    confusion_counts = Counter(
        (str(row["jersey_gt_canonical"]), str(row["raw_canonical"] or "invalid"))
        for _, row in numeric.iterrows()
        if str(row["jersey_gt_canonical"]) != str(row["raw_canonical"] or "invalid")
    )
    confusions = [
        {"gt": gt, "prediction": prediction, "count": count}
        for (gt, prediction), count in confusion_counts.most_common()
    ]

    class_metrics = []
    for gt, group in numeric.groupby("jersey_gt_canonical", sort=lambda values: values):
        exact = group["raw_canonical"] == group["jersey_gt_canonical"]
        class_metrics.append(
            {
                "jersey_gt": str(gt),
                "rows": int(len(group)),
                "correct": int(exact.sum()),
                "accuracy": float(exact.mean()),
            }
        )
    class_metrics.sort(key=lambda item: int(item["jersey_gt"]))

    quality_gates = []
    gate_path = None
    if args.quality_gate_report:
        gate_path = Path(args.quality_gate_report).expanduser().resolve()
    else:
        candidate = gt_path.parent / "quality_gate_v1" / "quality_gate_report.json"
        if candidate.exists():
            gate_path = candidate
    if gate_path is not None:
        gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
        for candidate in gate_report.get("selected", []):
            gate = {
                "recall_target": float(candidate["recall_target"]),
                "family": candidate["family"],
                "rule": candidate["rule"],
                "metrics": {"all": _gate_metrics(results, candidate)},
            }
            for split in ("train", "val", "test"):
                gate["metrics"][split] = _gate_metrics(results[results["split"] == split], candidate)
            quality_gates.append(gate)

    report = {
        "dataset": {
            "ocr_rows": int(len(results)),
            "numeric_rows": int((results["label_kind"] == "number").sum()),
            "unreadable_rows": int((results["label_kind"] == "unreadable").sum()),
        },
        "model": {
            "checkpoint": str(model_path),
            "device": analyzer.device.type,
            "batch_size": batch_size,
            "inference_seconds": float(inference_seconds),
            "milliseconds_per_crop": float(1000.0 * inference_seconds / len(results)),
            "checkpoint_diagnostics": analyzer.checkpoint_diagnostics,
        },
        "numeric_metrics": numeric_metrics,
        "visibility_metrics": visibility_metrics,
        "class_metrics": class_metrics,
        "confidence": {
            "readable": _quantiles(readable_prob),
            "unreadable": _quantiles(unreadable_prob),
            "readability_auroc": _readability_auc(results),
        },
        "temporal": _temporal_metrics(results, analyzer),
        "prediction_distribution": [[key, count] for key, count in prediction_counts.most_common()],
        "confusions": confusions,
        "quality_gate_report": str(gate_path) if gate_path else None,
        "quality_gates": quality_gates,
        "benchmark_contract": {
            "raw_ocr": "direct checkpoint digit output; no roster decoding and no roster-based rewriting",
            "numeric_accuracy": "human-readable numeric GT crops only",
            "unreadable": "used only for confidence/readability diagnostics, not counted as an OCR number error",
            "quality_gate": "uses pre-existing calibrated thresholds without re-optimizing against OCR accuracy",
        },
    }

    serializable = results.drop(columns=["_tens_probs", "_ones_probs"], errors="ignore").copy()
    serializable.to_csv(output_dir / "jersey_ocr_predictions.csv", index=False)
    pd.DataFrame(confusions).to_csv(output_dir / "jersey_ocr_confusions.csv", index=False)
    (output_dir / "jersey_ocr_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(report, output_dir / "jersey_ocr_benchmark.md")

    print(f"benchmark={output_dir / 'jersey_ocr_benchmark.md'}")
    print(f"predictions={output_dir / 'jersey_ocr_predictions.csv'}")
    print(f"exact_accuracy={numeric_metrics['all']['exact_accuracy']:.6f}")
    print(f"top3_accuracy={numeric_metrics['all']['top3_accuracy']:.6f}")
    print(f"inference_ms_per_crop={report['model']['milliseconds_per_crop']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
