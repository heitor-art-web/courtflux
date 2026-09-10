from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_PRECISIONS = (0.70, 0.80, 0.85, 0.90)
MODES = ("official_player", "torso_direct")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    required = {
        "mode",
        "split",
        "label_kind",
        "jersey_gt_int",
        "pred_number",
        "pred_score",
        "uncertainty",
        "absent_argmax",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {missing}")

    out = out[out["label_kind"].isin(["number", "unreadable"])].copy()
    out["pred_number"] = pd.to_numeric(out["pred_number"], errors="coerce")
    out["jersey_gt_int"] = pd.to_numeric(out["jersey_gt_int"], errors="coerce")
    out["pred_score"] = pd.to_numeric(out["pred_score"], errors="coerce")
    out["uncertainty"] = pd.to_numeric(out["uncertainty"], errors="coerce")
    out["absent_argmax"] = (
        out["absent_argmax"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    out["is_numeric"] = out["label_kind"] == "number"
    out["number_correct"] = (
        out["is_numeric"]
        & out["pred_number"].notna()
        & out["jersey_gt_int"].notna()
        & (out["pred_number"].astype("Int64") == out["jersey_gt_int"].astype("Int64"))
    )
    return out


def _accept_mask(
    frame: pd.DataFrame,
    *,
    uncertainty_max: float | None,
    pred_score_min: float | None,
    require_not_absent: bool,
) -> pd.Series:
    keep = pd.Series(True, index=frame.index, dtype=bool)
    if require_not_absent:
        keep &= ~frame["absent_argmax"].astype(bool)
    if uncertainty_max is not None:
        keep &= frame["uncertainty"] <= float(uncertainty_max)
    if pred_score_min is not None:
        keep &= frame["pred_score"] >= float(pred_score_min)
    return keep.fillna(False)


def _metrics(frame: pd.DataFrame, keep: pd.Series) -> dict[str, Any]:
    numeric = frame["is_numeric"].astype(bool)
    unreadable = ~numeric
    accepted = keep.astype(bool)
    accepted_numeric = accepted & numeric
    accepted_unreadable = accepted & unreadable
    correct_accepted = accepted & frame["number_correct"].astype(bool)

    n_numeric = int(numeric.sum())
    n_unreadable = int(unreadable.sum())
    n_accepted = int(accepted.sum())
    n_accepted_numeric = int(accepted_numeric.sum())
    n_correct = int(correct_accepted.sum())

    return {
        "rows": int(len(frame)),
        "numeric_rows": n_numeric,
        "unreadable_rows": n_unreadable,
        "accepted_rows": n_accepted,
        "accepted_numeric_rows": n_accepted_numeric,
        "correct_accepted": n_correct,
        "numeric_coverage": n_accepted_numeric / n_numeric if n_numeric else None,
        "decision_precision": n_correct / n_accepted if n_accepted else None,
        "accepted_numeric_accuracy": n_correct / n_accepted_numeric if n_accepted_numeric else None,
        "unreadable_false_accept_rate": int(accepted_unreadable.sum()) / n_unreadable if n_unreadable else None,
        "correct_recall": n_correct / n_numeric if n_numeric else None,
    }


def _candidate_values(series: pd.Series, direction: str) -> list[float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return [None]
    quantiles = np.unique(values.quantile(np.linspace(0.02, 1.0, 50)).to_numpy())
    vals = [float(value) for value in quantiles]
    if direction == "min":
        vals = sorted(vals)
    else:
        vals = sorted(vals, reverse=True)
    return vals


def _calibrate(train: pd.DataFrame, target_precision: float) -> dict[str, Any]:
    uncertainty_values = _candidate_values(train["uncertainty"], "max")
    score_values = _candidate_values(train["pred_score"], "min")

    candidates: list[dict[str, Any]] = []

    # Strict absent-aware baseline and one-dimensional thresholds are included
    # explicitly. The combined grid then tests whether confidence adds useful
    # information beyond uncertainty without ever looking at val/test labels.
    policies = [(None, None)]
    policies += [(u, None) for u in uncertainty_values]
    policies += [(None, s) for s in score_values]

    # A compact quantile grid avoids fitting hundreds of nearly identical cuts.
    u_grid = uncertainty_values[::5] or uncertainty_values
    s_grid = score_values[::5] or score_values
    policies += [(u, s) for u in u_grid for s in s_grid]

    seen: set[tuple[float | None, float | None]] = set()
    for uncertainty_max, pred_score_min in policies:
        key = (uncertainty_max, pred_score_min)
        if key in seen:
            continue
        seen.add(key)
        keep = _accept_mask(
            train,
            uncertainty_max=uncertainty_max,
            pred_score_min=pred_score_min,
            require_not_absent=True,
        )
        metrics = _metrics(train, keep)
        precision = metrics["decision_precision"]
        if precision is None or precision < target_precision:
            continue
        candidates.append(
            {
                "uncertainty_max": uncertainty_max,
                "pred_score_min": pred_score_min,
                "require_not_absent": True,
                "train_metrics": metrics,
            }
        )

    if not candidates:
        return {
            "target_precision": float(target_precision),
            "found": False,
            "reason": "No train-only threshold met the requested decision precision.",
        }

    # Primary objective: maximize readable-number coverage. Tie-breakers prefer
    # higher decision precision, higher correct recall, then the simpler policy.
    def ranking(item: dict[str, Any]) -> tuple[float, float, float, int]:
        metrics = item["train_metrics"]
        complexity = int(item["uncertainty_max"] is not None) + int(item["pred_score_min"] is not None)
        return (
            float(metrics["numeric_coverage"] or 0.0),
            float(metrics["decision_precision"] or 0.0),
            float(metrics["correct_recall"] or 0.0),
            -complexity,
        )

    best = max(candidates, key=ranking)
    return {
        "target_precision": float(target_precision),
        "found": True,
        **best,
    }


def _evaluate_policy(frame: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    if not policy.get("found"):
        return {"found": False}
    result: dict[str, Any] = {"found": True}
    for split in ("train", "val", "test", "all"):
        subset = frame if split == "all" else frame[frame["split"] == split]
        keep = _accept_mask(
            subset,
            uncertainty_max=policy.get("uncertainty_max"),
            pred_score_min=policy.get("pred_score_min"),
            require_not_absent=bool(policy.get("require_not_absent", True)),
        )
        result[split] = _metrics(subset, keep)
    return result


def _absent_only(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val", "test", "all"):
        subset = frame if split == "all" else frame[frame["split"] == split]
        keep = _accept_mask(
            subset,
            uncertainty_max=None,
            pred_score_min=None,
            require_not_absent=True,
        )
        result[split] = _metrics(subset, keep)
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Uncertainty-aware jersey OCR production-policy benchmark",
        "",
        "This benchmark treats the model's `absent` output as a real abstention and calibrates confidence thresholds on the CourtFlux **train split only**. Frozen thresholds are then evaluated on val/test without re-optimization.",
        "",
        "Decision precision counts both wrong jersey numbers and accepted `unreadable` crops as errors. Numeric coverage is the fraction of human-readable numeric crops for which the policy emits a jersey number.",
        "",
    ]

    for mode_report in report["modes"]:
        lines += [
            f"## {mode_report['mode']}",
            "",
            "### Absent-aware baseline",
            "",
            "| Split | Numeric coverage | Decision precision | Accepted-number accuracy | Unreadable false accept | Correct recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for split in ("train", "val", "test", "all"):
            m = mode_report["absent_only"][split]
            lines.append(
                f"| {split} | {_pct(m['numeric_coverage'])} | {_pct(m['decision_precision'])} | "
                f"{_pct(m['accepted_numeric_accuracy'])} | {_pct(m['unreadable_false_accept_rate'])} | {_pct(m['correct_recall'])} |"
            )

        lines += [
            "",
            "### Train-calibrated policies",
            "",
            "| Train target | u max | score min | Train cov. | Train precision | Test cov. | Test precision | Test num. acc. | Test unreadable FA | Test correct recall |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in mode_report["policies"]:
            calibration = item["calibration"]
            if not calibration.get("found"):
                lines.append(
                    f"| {_pct(calibration['target_precision'])} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            train_m = item["evaluation"]["train"]
            test_m = item["evaluation"]["test"]
            u = calibration.get("uncertainty_max")
            s = calibration.get("pred_score_min")
            lines.append(
                f"| {_pct(calibration['target_precision'])} | "
                f"{('n/a' if u is None else f'{u:.6f}')} | "
                f"{('n/a' if s is None else f'{s:.6f}')} | "
                f"{_pct(train_m['numeric_coverage'])} | {_pct(train_m['decision_precision'])} | "
                f"{_pct(test_m['numeric_coverage'])} | {_pct(test_m['decision_precision'])} | "
                f"{_pct(test_m['accepted_numeric_accuracy'])} | {_pct(test_m['unreadable_false_accept_rate'])} | "
                f"{_pct(test_m['correct_recall'])} |"
            )

        lines += [
            "",
            "Threshold selection contract: all threshold search and target-precision decisions use only rows whose split is `train`. `val`, `test`, and `all` are evaluation-only.",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate an absent-aware uncertainty policy on train and evaluate frozen thresholds on CourtFlux val/test"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    predictions_path = Path(args.predictions).expanduser().resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else predictions_path.parent / "policy_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _prepare(pd.read_csv(predictions_path))
    mode_reports = []
    for mode in MODES:
        subset = frame[frame["mode"] == mode].copy()
        if subset.empty:
            continue
        train = subset[subset["split"] == "train"].copy()
        if train.empty:
            raise ValueError(f"No train rows for mode={mode}")

        policies = []
        for target in TARGET_PRECISIONS:
            calibration = _calibrate(train, target)
            policies.append(
                {
                    "calibration": calibration,
                    "evaluation": _evaluate_policy(subset, calibration),
                }
            )
        mode_reports.append(
            {
                "mode": mode,
                "rows": int(len(subset)),
                "absent_only": _absent_only(subset),
                "policies": policies,
            }
        )

    if not mode_reports:
        raise ValueError("No supported modes found in predictions CSV")

    report = {
        "source_predictions": str(predictions_path),
        "target_precisions": list(TARGET_PRECISIONS),
        "modes": mode_reports,
    }
    json_path = output_dir / "jersey_uncertainty_policy_benchmark.json"
    md_path = output_dir / "jersey_uncertainty_policy_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)

    print(f"benchmark={md_path}")
    for mode in mode_reports:
        test = mode["absent_only"]["test"]
        print(
            f"{mode['mode']}: absent_only_test_coverage={test['numeric_coverage']} "
            f"precision={test['decision_precision']} correct_recall={test['correct_recall']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
