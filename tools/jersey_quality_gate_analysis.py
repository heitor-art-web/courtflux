from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
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
    for column in NUMERIC_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["torso_area"] = out["torso_width"] * out["torso_height"]
    out["torso_min_side"] = out[["torso_width", "torso_height"]].min(axis=1)
    return out


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {key: None for key in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")}
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _describe_group(frame: pd.DataFrame) -> dict:
    result = {"rows": int(len(frame))}
    for feature in ("torso_sharpness", "torso_width", "torso_height", "torso_area"):
        result[feature] = _quantiles(frame[feature])
    return result


def _candidate_values(series: pd.Series, max_values: int = 140) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return [0.0]
    unique = np.unique(values.to_numpy())
    if len(unique) <= max_values:
        candidates = unique
    else:
        quantiles = np.linspace(0.0, 1.0, max_values)
        candidates = np.unique(np.quantile(unique, quantiles))
    floor = min(0.0, float(np.min(unique)))
    return sorted({floor, *(float(value) for value in candidates)})


def _metrics(frame: pd.DataFrame, keep: pd.Series) -> dict:
    keep = keep.fillna(False).astype(bool)
    readable = frame["is_readable"]
    unreadable = ~readable

    readable_total = int(readable.sum())
    unreadable_total = int(unreadable.sum())
    total = int(len(frame))
    kept_total = int(keep.sum())
    kept_readable = int((keep & readable).sum())
    kept_unreadable = int((keep & unreadable).sum())
    rejected_readable = readable_total - kept_readable
    rejected_unreadable = unreadable_total - kept_unreadable

    return {
        "rows": total,
        "readable_rows": readable_total,
        "unreadable_rows": unreadable_total,
        "kept_rows": kept_total,
        "rejected_rows": total - kept_total,
        "readable_kept": kept_readable,
        "readable_rejected": rejected_readable,
        "unreadable_kept": kept_unreadable,
        "unreadable_rejected": rejected_unreadable,
        "readable_recall": kept_readable / readable_total if readable_total else None,
        "unreadable_rejection_rate": rejected_unreadable / unreadable_total if unreadable_total else None,
        "player_ocr_call_reduction": (total - kept_total) / total if total else None,
        "kept_readable_fraction": kept_readable / kept_total if kept_total else None,
    }


def _rule_label(family: str, sharpness: float | None = None, area: float | None = None, value: float | None = None) -> str:
    if family == "sharpness":
        return f"keep if torso_sharpness >= {value:.3f}"
    if family == "torso_area":
        return f"keep if torso_area >= {value:.1f}"
    if family == "torso_height":
        return f"keep if torso_height >= {value:.1f}"
    if family == "torso_width":
        return f"keep if torso_width >= {value:.1f}"
    if family == "sharpness_and_area":
        return f"keep if torso_sharpness >= {sharpness:.3f} AND torso_area >= {area:.1f}"
    return family


def _single_candidates(frame: pd.DataFrame, feature: str, family: str) -> list[dict]:
    candidates: list[dict] = []
    for threshold in _candidate_values(frame[feature]):
        keep = frame[feature] >= threshold
        candidates.append(
            {
                "family": family,
                "threshold": float(threshold),
                "rule": _rule_label(family, value=float(threshold)),
                "calibration": _metrics(frame, keep),
            }
        )
    return candidates


def _combined_candidates(frame: pd.DataFrame) -> list[dict]:
    sharpness_values = _candidate_values(frame["torso_sharpness"], max_values=90)
    area_values = _candidate_values(frame["torso_area"], max_values=90)
    candidates: list[dict] = []
    for sharpness in sharpness_values:
        sharp_keep = frame["torso_sharpness"] >= sharpness
        for area in area_values:
            keep = sharp_keep & (frame["torso_area"] >= area)
            candidates.append(
                {
                    "family": "sharpness_and_area",
                    "sharpness_threshold": float(sharpness),
                    "area_threshold": float(area),
                    "rule": _rule_label(
                        "sharpness_and_area",
                        sharpness=float(sharpness),
                        area=float(area),
                    ),
                    "calibration": _metrics(frame, keep),
                }
            )
    return candidates


def _candidate_keep(frame: pd.DataFrame, candidate: dict) -> pd.Series:
    family = candidate["family"]
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
    raise ValueError(f"Unknown family: {family}")


def _best_for_target(candidates: list[dict], recall_target: float) -> dict | None:
    eligible = []
    for candidate in candidates:
        metrics = candidate["calibration"]
        recall = metrics.get("readable_recall")
        if recall is None or recall + 1e-12 < recall_target:
            continue
        unreadable_rejection = metrics.get("unreadable_rejection_rate") or 0.0
        reduction = metrics.get("player_ocr_call_reduction") or 0.0
        rejected_readable = metrics.get("readable_rejected") or 0
        eligible.append((unreadable_rejection, reduction, -rejected_readable, candidate))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return dict(eligible[0][3])


def _evaluate_candidate(candidate: dict, splits: dict[str, pd.DataFrame]) -> dict:
    result = {key: value for key, value in candidate.items() if key != "calibration"}
    result["metrics"] = {}
    for name, frame in splits.items():
        result["metrics"][name] = _metrics(frame, _candidate_keep(frame, candidate))
    return result


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Jersey OCR quality-gate calibration",
        "",
        "This report treats numeric player labels as readable OCR opportunities and `unreadable` player labels as crops that should ideally be skipped before neural OCR.",
        "",
        "The thresholds are calibration candidates, not production truth. They were selected on the configured calibration split and must be rechecked on additional games before deployment.",
        "",
        "## Dataset",
        "",
        f"- Reviewed detections: {report['dataset']['reviewed_rows']}",
        f"- True-player OCR samples: {report['dataset']['ocr_rows']}",
        f"- Readable numeric samples: {report['dataset']['readable_rows']}",
        f"- Unreadable player samples: {report['dataset']['unreadable_rows']}",
        f"- Detector negatives: {report['dataset']['detector_negative_rows']}",
        f"- Unknown subjects: {report['dataset']['unknown_subject_rows']}",
        f"- Theoretical call reduction from a perfect subject-type gate alone: {_pct(report['dataset']['subject_gate_theoretical_reduction'])}",
        f"- Threshold selection split: {report['selection']['split']}",
        "",
        "## Selected crop-quality gates",
        "",
        "| Readable recall target | Rule | Calibration readable recall | Calibration unreadable rejected | Calibration OCR-call reduction | Test readable recall | Test unreadable rejected | Test OCR-call reduction |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["selected"]:
        test = item["metrics"].get("test", {})
        calibration = item["metrics"].get(report["selection"]["split"], item["metrics"].get("all", {}))
        lines.append(
            "| "
            + f"{100 * item['recall_target']:.0f}% | {item['rule']} | "
            + f"{_pct(calibration.get('readable_recall'))} | "
            + f"{_pct(calibration.get('unreadable_rejection_rate'))} | "
            + f"{_pct(calibration.get('player_ocr_call_reduction'))} | "
            + f"{_pct(test.get('readable_recall'))} | "
            + f"{_pct(test.get('unreadable_rejection_rate'))} | "
            + f"{_pct(test.get('player_ocr_call_reduction'))} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Prefer the highest-reduction rule whose readable-number recall remains acceptable on held-out data. With this small first-game dataset, 95% readable recall is a useful candidate operating point to inspect, but it should not be hard-coded until it survives another game with different camera motion, lighting and jersey designs.",
        "",
        "`torso_sharpness` is a cheap Laplacian-variance proxy. The combined rule tests whether adding crop area improves rejection beyond blur alone.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantitatively calibrate a cheap pre-OCR jersey crop quality gate")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--selection-split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--recall-targets", default="0.99,0.95,0.90")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else gt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    required = {
        "subject_type",
        "label_kind",
        "dataset_role",
        "split",
        "torso_sharpness",
        "torso_width",
        "torso_height",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ground truth is missing required columns: {missing}")

    frame = _float_frame(frame)
    ocr = frame[
        (frame["dataset_role"] == "ocr")
        & (frame["subject_type"] == "player")
        & frame["label_kind"].isin(["number", "unreadable"])
    ].copy()
    ocr["is_readable"] = ocr["label_kind"] == "number"
    ocr = ocr.dropna(subset=["torso_sharpness", "torso_width", "torso_height", "torso_area"])

    if ocr.empty or int(ocr["is_readable"].sum()) == 0 or int((~ocr["is_readable"]).sum()) == 0:
        raise ValueError("Need both readable numeric and unreadable player samples for calibration")

    splits = {"all": ocr}
    for split in ("train", "val", "test"):
        splits[split] = ocr[ocr["split"] == split].copy()

    selection_name = args.selection_split
    selection = splits[selection_name]
    selection_fallback = False
    if selection_name != "all":
        readable_count = int(selection["is_readable"].sum())
        unreadable_count = int((~selection["is_readable"]).sum())
        if readable_count < 5 or unreadable_count < 5:
            selection_name = "all"
            selection = ocr
            selection_fallback = True

    recall_targets = []
    for value in args.recall_targets.split(","):
        target = float(value.strip())
        if not (0.0 < target <= 1.0):
            raise ValueError("Recall targets must be in (0, 1]")
        recall_targets.append(target)

    families: dict[str, list[dict]] = {
        "sharpness": _single_candidates(selection, "torso_sharpness", "sharpness"),
        "torso_area": _single_candidates(selection, "torso_area", "torso_area"),
        "torso_height": _single_candidates(selection, "torso_height", "torso_height"),
        "torso_width": _single_candidates(selection, "torso_width", "torso_width"),
        "sharpness_and_area": _combined_candidates(selection),
    }

    selected: list[dict] = []
    candidate_rows: list[dict] = []
    for target in recall_targets:
        family_winners = []
        for family, candidates in families.items():
            winner = _best_for_target(candidates, target)
            if winner is None:
                continue
            evaluated = _evaluate_candidate(winner, splits)
            evaluated["recall_target"] = target
            evaluated["scope"] = "family_winner"
            family_winners.append(evaluated)
            row = {
                "recall_target": target,
                "scope": "family_winner",
                "family": family,
                "rule": evaluated["rule"],
            }
            for split_name, metrics in evaluated["metrics"].items():
                for key in ("readable_recall", "unreadable_rejection_rate", "player_ocr_call_reduction", "readable_rejected"):
                    row[f"{split_name}_{key}"] = metrics.get(key)
            candidate_rows.append(row)

        if not family_winners:
            continue
        family_winners.sort(
            key=lambda item: (
                item["metrics"][selection_name].get("unreadable_rejection_rate") or 0.0,
                item["metrics"][selection_name].get("player_ocr_call_reduction") or 0.0,
                -(item["metrics"][selection_name].get("readable_rejected") or 0),
            ),
            reverse=True,
        )
        best = family_winners[0]
        best["scope"] = "best_overall"
        selected.append(best)

        row = {
            "recall_target": target,
            "scope": "best_overall",
            "family": best["family"],
            "rule": best["rule"],
        }
        for split_name, metrics in best["metrics"].items():
            for key in ("readable_recall", "unreadable_rejection_rate", "player_ocr_call_reduction", "readable_rejected"):
                row[f"{split_name}_{key}"] = metrics.get(key)
        candidate_rows.append(row)

    reviewed_rows = int(len(frame))
    detector_negatives = int((frame["dataset_role"] == "detector_negative").sum())
    unknown_rows = int((frame["dataset_role"] == "excluded_unknown").sum())
    report = {
        "ground_truth": str(gt_path),
        "dataset": {
            "reviewed_rows": reviewed_rows,
            "ocr_rows": int(len(ocr)),
            "readable_rows": int(ocr["is_readable"].sum()),
            "unreadable_rows": int((~ocr["is_readable"]).sum()),
            "detector_negative_rows": detector_negatives,
            "unknown_subject_rows": unknown_rows,
            "subject_gate_theoretical_reduction": (
                (detector_negatives + unknown_rows) / reviewed_rows if reviewed_rows else None
            ),
            "readable_fraction_within_players": float(ocr["is_readable"].mean()),
        },
        "selection": {
            "requested_split": args.selection_split,
            "split": selection_name,
            "fallback_to_all": selection_fallback,
            "rows": int(len(selection)),
            "readable_rows": int(selection["is_readable"].sum()),
            "unreadable_rows": int((~selection["is_readable"]).sum()),
        },
        "distributions": {
            "readable": _describe_group(ocr[ocr["is_readable"]]),
            "unreadable": _describe_group(ocr[~ocr["is_readable"]]),
        },
        "selected": selected,
        "caution": (
            "This is a first-game calibration. Do not freeze a production threshold until the chosen gate "
            "preserves readable recall on at least one additional game with different conditions."
        ),
    }

    json_path = output_dir / "quality_gate_report.json"
    csv_path = output_dir / "quality_gate_candidates.csv"
    md_path = output_dir / "quality_gate_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(candidate_rows).to_csv(csv_path, index=False)
    _write_markdown(report, md_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"quality_gate_report={json_path}")
    print(f"quality_gate_candidates={csv_path}")
    print(f"quality_gate_markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
