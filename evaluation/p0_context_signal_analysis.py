from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import align_production_truth, center_distance, load_jsonl
from evaluation.p0_ocsort_identity_signal_analysis import _auc


FEATURES = [
    "detector_confidence",
    "rank_score",
    "court_inside_margin_m",
    "inside_court_1m_margin",
    "nearest_hoop_apparent_distance_m",
    "apparent_court_speed_m_per_step",
    "nearest_player_box_distance_norm",
    "inside_player_box",
    "nearest_player_court_distance_m",
]


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tracked_hypotheses(production: dict[str, Any]) -> list[dict[str, Any]]:
    association = production.get("ball_association") or {}
    hypotheses = (association.get("ball_hypotheses") or {}).get("hypotheses") or []
    return [
        item
        for item in hypotheses
        if isinstance(item, dict)
        and item.get("lifecycle") == "TRACKED"
        and bool(item.get("observed", False))
    ]


def _label(hypothesis: dict[str, Any], truth: dict[str, Any], radius_px: float) -> int:
    if not bool(truth.get("visible", False)):
        return 0
    center = hypothesis.get("center")
    point = truth.get("point")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return 0
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return 0
    return int(center_distance(center, point) <= float(radius_px))


def _feature_value(hypothesis: dict[str, Any], feature: str) -> float | None:
    if feature in {"detector_confidence", "rank_score"}:
        return _number(hypothesis.get(feature))
    context = hypothesis.get("context") or {}
    return _number(context.get(feature))


def analyze_pair(
    frames_path: Path,
    annotations_path: Path,
    *,
    radius_px: float,
) -> dict[str, Any]:
    production_rows = load_jsonl(frames_path)
    truth_rows = load_jsonl(annotations_path)
    aligned, unmatched = align_production_truth(production_rows, truth_rows, tolerance_sec=0.001)

    observations: list[dict[str, Any]] = []
    visible_frames = 0
    not_visible_frames = 0
    true_steps: set[int] = set()
    context_valid_frames = 0
    homography_sources: dict[str, int] = {}

    for step, (production, truth, _delta) in enumerate(aligned):
        visible = bool(truth.get("visible", False))
        visible_frames += int(visible)
        not_visible_frames += int(not visible)
        hypotheses = _tracked_hypotheses(production)
        frame_context_valid = False
        for hypothesis in hypotheses:
            context = hypothesis.get("context") or {}
            if bool(context.get("homography_valid", False)):
                frame_context_valid = True
            source = str(context.get("homography_source") or "unknown")
            homography_sources[source] = homography_sources.get(source, 0) + 1
            label = _label(hypothesis, truth, radius_px)
            if label and visible:
                true_steps.add(step)
            observations.append(
                {
                    "step": step,
                    "visible": visible,
                    "label": label,
                    "features": {
                        feature: _feature_value(hypothesis, feature) for feature in FEATURES
                    },
                }
            )
        context_valid_frames += int(frame_context_valid)

    positives = sum(int(obs["label"]) for obs in observations)
    total = len(observations)
    feature_rows: dict[str, Any] = {}
    for feature in FEATURES:
        usable = [
            (obs["features"][feature], int(obs["label"]))
            for obs in observations
            if obs["features"][feature] is not None
        ]
        values = [float(value) for value, _ in usable]
        labels = [label for _, label in usable]
        true_values = [value for value, label in usable if label == 1]
        false_values = [value for value, label in usable if label == 0]
        auc = _auc(values, labels) if values and len(set(labels)) > 1 else None
        feature_rows[feature] = {
            "n": len(usable),
            "coverage_pct": round(100.0 * len(usable) / max(total, 1), 2),
            "true_median": None if not true_values else round(float(statistics.median(true_values)), 6),
            "false_median": None if not false_values else round(float(statistics.median(false_values)), 6),
            "auc": None if auc is None else round(float(auc), 4),
            "best_auc": None if auc is None else round(float(max(auc, 1.0 - auc)), 4),
        }

    return {
        "frames_path": str(frames_path),
        "annotations_path": str(annotations_path),
        "aligned_frames": len(aligned),
        "unmatched_gameplay_frames": int(unmatched),
        "visible_frames": visible_frames,
        "not_visible_frames": not_visible_frames,
        "confirmed_observations": total,
        "true_confirmed_observations": positives,
        "confirmed_instance_precision_pct": round(100.0 * positives / max(total, 1), 2),
        "true_track_frame_ceiling_pct": round(100.0 * len(true_steps) / max(visible_frames, 1), 2),
        "context_valid_frame_pct": round(100.0 * context_valid_frames / max(len(aligned), 1), 2),
        "homography_sources": homography_sources,
        "features": feature_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure separability of diagnostic court/player context on observed TRACKED "
            "OC-SORT ball hypotheses. Context never sees GT during tracking."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--output", default="evaluation/p0_context_signal_analysis")
    args = parser.parse_args()

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    results = []
    for run_id, gt_id in args.pair:
        result = analyze_pair(
            run_root / run_id / "frames.jsonl",
            gt_root / gt_id / "annotations.jsonl",
            radius_px=float(args.gt_radius_px),
        )
        result["run_id"] = run_id
        result["gt_run_id"] = gt_id
        results.append(result)

    total_observations = sum(int(result["confirmed_observations"]) for result in results)
    total_true = sum(int(result["true_confirmed_observations"]) for result in results)
    total_visible = sum(int(result["visible_frames"]) for result in results)
    weighted_ceiling_true = sum(
        int(round(float(result["true_track_frame_ceiling_pct"]) * int(result["visible_frames"]) / 100.0))
        for result in results
    )
    macro_context = statistics.fmean(float(result["context_valid_frame_pct"]) for result in results)

    aggregate_features: dict[str, Any] = {}
    for feature in FEATURES:
        best_aucs = [
            float(result["features"][feature]["best_auc"])
            for result in results
            if result["features"][feature]["best_auc"] is not None
        ]
        coverages = [float(result["features"][feature]["coverage_pct"]) for result in results]
        aggregate_features[feature] = {
            "best_auc_macro": None if not best_aucs else round(float(statistics.fmean(best_aucs)), 4),
            "feature_coverage_macro_pct": round(float(statistics.fmean(coverages)), 2),
        }

    summary = {
        "scope": "OC-SORT observed TRACKED hypotheses + diagnostic court/player context",
        "features": FEATURES,
        "runs": results,
        "aggregate": {
            "confirmed_observations": total_observations,
            "confirmed_instance_precision_micro_pct": round(100.0 * total_true / max(total_observations, 1), 2),
            "true_track_frame_ceiling_micro_pct": round(100.0 * weighted_ceiling_true / max(total_visible, 1), 2),
            "context_valid_frame_macro_pct": round(float(macro_context), 2),
            "features": aggregate_features,
        },
        "decision_rule": (
            "Use context in the multivariate Ball Identity arbiter only if it adds held-out "
            "separability beyond detector/temporal features; ball homography coordinates remain "
            "planar-apparent soft evidence, never a hard ground-plane constraint."
        ),
    }
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("P0 COURT / PLAYER CONTEXT SIGNAL ANALYSIS")
    print("=" * 96)
    agg = summary["aggregate"]
    print(f"confirmed observations:          {agg['confirmed_observations']}")
    print(f"confirmed instance precision:   {agg['confirmed_instance_precision_micro_pct']:.2f}%")
    print(f"true-track frame ceiling:       {agg['true_track_frame_ceiling_micro_pct']:.2f}%")
    print(f"context-valid frames macro:     {agg['context_valid_frame_macro_pct']:.2f}%")
    print("\nFEATURE SEPARABILITY")
    print("-" * 96)
    print(f"{'feature':38s} {'best_auc':>10s} {'coverage':>10s}")
    for feature in FEATURES:
        row = agg["features"][feature]
        auc = row["best_auc_macro"]
        auc_text = "n/a" if auc is None else f"{auc:.4f}"
        print(f"{feature:38s} {auc_text:>10s} {row['feature_coverage_macro_pct']:9.2f}%")
    print(f"\nsummary: {path}")


if __name__ == "__main__":
    main()
