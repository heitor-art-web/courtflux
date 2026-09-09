from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import align_production_truth, center_distance, load_jsonl
from evaluation.p0_ocsort_identity_signal_analysis import _auc
from evaluation.p0_ocsort_multivariate_arbiter_cv import _fit_logistic, _predict, _threshold_for_precision


TEMPORAL_FEATURES = [
    "detector_confidence",
    "rank_score",
    "confidence_rank_in_confirmed",
    "confirmed_competitors",
    "track_observations",
    "track_streak",
    "track_mean_confidence",
    "track_area_cv",
    "track_speed_px_per_step",
    "box_area",
]

CONTEXT_FEATURES = [
    "inside_court_1m_margin",
    "nearest_player_box_distance_norm",
    "inside_player_box",
    "court_inside_margin_m",
    "nearest_hoop_apparent_distance_m",
    "apparent_court_speed_m_per_step",
    "nearest_player_court_distance_m",
    "homography_inlier_ratio",
    "homography_rmse_px",
    "homography_keypoint_age_frames",
]

LOG1P_FEATURES = {
    "track_observations",
    "track_streak",
    "track_area_cv",
    "track_speed_px_per_step",
    "box_area",
    "nearest_player_box_distance_norm",
    "nearest_hoop_apparent_distance_m",
    "apparent_court_speed_m_per_step",
    "nearest_player_court_distance_m",
    "homography_rmse_px",
    "homography_keypoint_age_frames",
}


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
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _transform(feature: str, value: float | None) -> float | None:
    if value is None:
        return None
    if feature in LOG1P_FEATURES:
        return math.log1p(max(float(value), 0.0))
    return float(value)


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


def _extract_features(hypothesis: dict[str, Any]) -> dict[str, float | None]:
    history = hypothesis.get("features") or {}
    context = hypothesis.get("context") or {}
    result: dict[str, float | None] = {}
    for feature in TEMPORAL_FEATURES:
        if feature in {"detector_confidence", "rank_score"}:
            value = _number(hypothesis.get(feature))
        else:
            value = _number(history.get(feature))
        result[feature] = _transform(feature, value)
    for feature in CONTEXT_FEATURES:
        result[feature] = _transform(feature, _number(context.get(feature)))
    return result


def load_pair(frames_path: Path, annotations_path: Path, *, radius_px: float) -> dict[str, Any]:
    production_rows = load_jsonl(frames_path)
    truth_rows = load_jsonl(annotations_path)
    aligned, unmatched = align_production_truth(production_rows, truth_rows, tolerance_sec=0.001)

    observations: list[dict[str, Any]] = []
    visible_frames = 0
    not_visible_frames = 0
    for step, (production, truth, _delta) in enumerate(aligned):
        visible = bool(truth.get("visible", False))
        visible_frames += int(visible)
        not_visible_frames += int(not visible)
        for hypothesis in _tracked_hypotheses(production):
            observations.append(
                {
                    "step": int(step),
                    "visible": visible,
                    "label": _label(hypothesis, truth, radius_px),
                    "features": _extract_features(hypothesis),
                }
            )

    return {
        "frames_path": str(frames_path),
        "annotations_path": str(annotations_path),
        "aligned_frames": len(aligned),
        "unmatched_gameplay_frames": int(unmatched),
        "visible_frames": int(visible_frames),
        "not_visible_frames": int(not_visible_frames),
        "observations": observations,
    }


def _feature_names(profile: str) -> list[str]:
    if profile == "temporal":
        return list(TEMPORAL_FEATURES)
    if profile == "temporal_context":
        return list(TEMPORAL_FEATURES) + list(CONTEXT_FEATURES)
    raise ValueError(profile)


def _fit_preprocessor(observations: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    medians: dict[str, float] = {}
    for feature in features:
        values = [
            float(obs["features"][feature])
            for obs in observations
            if obs["features"].get(feature) is not None
        ]
        medians[feature] = float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0
    return {"features": list(features), "medians": medians}


def _matrix(
    observations: list[dict[str, Any]],
    preprocessor: dict[str, Any],
    *,
    add_context_missing_indicators: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    features = list(preprocessor["features"])
    medians = dict(preprocessor["medians"])
    for obs in observations:
        row: list[float] = []
        for feature in features:
            value = obs["features"].get(feature)
            row.append(float(medians[feature] if value is None else value))
        if add_context_missing_indicators:
            for feature in CONTEXT_FEATURES:
                if feature in features:
                    row.append(1.0 if obs["features"].get(feature) is None else 0.0)
        rows.append(row)
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray([int(obs["label"]) for obs in observations], dtype=np.float64)
    return x, y


def _selection_metrics(
    probabilities: np.ndarray,
    observations: list[dict[str, Any]],
    *,
    threshold: float,
    visible_frames: int,
    not_visible_frames: int,
) -> dict[str, Any]:
    by_step: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for probability, obs in zip(probabilities.tolist(), observations):
        by_step.setdefault(int(obs["step"]), []).append((float(probability), obs))

    selected: list[tuple[int, float, dict[str, Any]]] = []
    if math.isfinite(threshold):
        for step, candidates in by_step.items():
            probability, obs = max(candidates, key=lambda item: item[0])
            if probability >= threshold:
                selected.append((step, probability, obs))

    selected_true = sum(int(obs["label"]) for _, _, obs in selected)
    visible_true_steps = {
        step for step, _, obs in selected if bool(obs["visible"]) and int(obs["label"]) == 1
    }
    not_visible_false_steps = {step for step, _, obs in selected if not bool(obs["visible"])}
    return {
        "selected_output_frames": int(len(selected)),
        "selected_true_frames": int(selected_true),
        "visible_true_frames": int(len(visible_true_steps)),
        "not_visible_false_frames": int(len(not_visible_false_steps)),
        "output_precision_pct": round(100.0 * selected_true / max(len(selected), 1), 2),
        "visible_coverage_pct": round(100.0 * len(visible_true_steps) / max(visible_frames, 1), 2),
        "not_visible_false_state_pct": round(
            100.0 * len(not_visible_false_steps) / max(not_visible_frames, 1), 2
        ),
    }


def _evaluate_profile(
    train_pairs: list[dict[str, Any]],
    test_pair: dict[str, Any],
    *,
    profile: str,
    targets: list[float],
) -> dict[str, Any]:
    train_obs = [obs for pair in train_pairs for obs in pair["observations"]]
    test_obs = list(test_pair["observations"])
    features = _feature_names(profile)
    preprocessor = _fit_preprocessor(train_obs, features)
    add_missing = profile == "temporal_context"
    x_train, y_train = _matrix(train_obs, preprocessor, add_context_missing_indicators=add_missing)
    x_test, y_test = _matrix(test_obs, preprocessor, add_context_missing_indicators=add_missing)

    if len(train_obs) == 0 or len(test_obs) == 0 or len(set(y_train.astype(int).tolist())) < 2:
        raise RuntimeError(f"Insufficient observations/classes for profile={profile}")

    model = _fit_logistic(x_train, y_train)
    train_prob = _predict(model, x_train)
    test_prob = _predict(model, x_test)
    auc = _auc(test_prob.tolist(), y_test.astype(int).tolist())

    target_results: dict[str, Any] = {}
    for target in targets:
        calibration = _threshold_for_precision(train_prob, y_train, float(target))
        threshold = float(calibration["threshold"])
        metrics = _selection_metrics(
            test_prob,
            test_obs,
            threshold=threshold,
            visible_frames=int(test_pair["visible_frames"]),
            not_visible_frames=int(test_pair["not_visible_frames"]),
        )
        target_results[f"p{int(round(target * 100))}"] = {
            "target_precision_pct": round(100.0 * target, 2),
            "train_threshold": None if not math.isfinite(threshold) else round(threshold, 6),
            "train_precision_pct": round(100.0 * calibration["precision"], 2),
            "train_recall_pct": round(100.0 * calibration["recall"], 2),
            **metrics,
        }

    return {
        "profile": profile,
        "features": features,
        "missing_indicator_features": list(CONTEXT_FEATURES) if add_missing else [],
        "test_observation_auc": None if auc is None else round(float(auc), 4),
        "targets": target_results,
    }


def _aggregate(folds: list[dict[str, Any]], profile: str, target_key: str) -> dict[str, Any]:
    profile_rows = [fold["profiles"][profile] for fold in folds]
    aucs = [float(row["test_observation_auc"]) for row in profile_rows if row["test_observation_auc"] is not None]
    selected_total = sum(int(row["targets"][target_key]["selected_output_frames"]) for row in profile_rows)
    selected_true = sum(int(row["targets"][target_key]["selected_true_frames"]) for row in profile_rows)
    visible_true = sum(int(row["targets"][target_key]["visible_true_frames"]) for row in profile_rows)
    nv_false = sum(int(row["targets"][target_key]["not_visible_false_frames"]) for row in profile_rows)
    visible_total = sum(int(fold["visible_frames"]) for fold in folds)
    nv_total = sum(int(fold["not_visible_frames"]) for fold in folds)
    return {
        "test_auc_macro": None if not aucs else round(float(np.mean(aucs)), 4),
        "selected_output_frames": int(selected_total),
        "output_precision_micro_pct": round(100.0 * selected_true / max(selected_total, 1), 2),
        "visible_coverage_micro_pct": round(100.0 * visible_true / max(visible_total, 1), 2),
        "not_visible_false_state_micro_pct": round(100.0 * nv_false / max(nv_total, 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-video-out comparison of temporal-only versus temporal + court/player "
            "Ball Identity arbitration on diagnostic OC-SORT context runs. Missing context is "
            "imputed using training-fold medians only and accompanied by missingness indicators."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--engineering-precision", type=float, default=0.80)
    parser.add_argument("--north-star-precision", type=float, default=0.90)
    parser.add_argument("--output", default="evaluation/p0_context_multivariate_arbiter_cv")
    args = parser.parse_args()

    targets = [float(args.engineering_precision), float(args.north_star_precision)]
    if any(target <= 0.0 or target > 1.0 for target in targets):
        raise SystemExit("Precision targets must be in (0, 1]")

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    pairs: list[dict[str, Any]] = []
    for run_id, gt_id in args.pair:
        pair = load_pair(
            run_root / run_id / "frames.jsonl",
            gt_root / gt_id / "annotations.jsonl",
            radius_px=float(args.gt_radius_px),
        )
        pair["run_id"] = run_id
        pair["gt_run_id"] = gt_id
        pairs.append(pair)

    if len(pairs) < 3:
        raise SystemExit("Need at least three context run/GT pairs for held-out validation")

    folds: list[dict[str, Any]] = []
    for idx, test_pair in enumerate(pairs):
        train_pairs = [pair for j, pair in enumerate(pairs) if j != idx]
        fold = {
            "test_run_id": test_pair["run_id"],
            "gt_run_id": test_pair["gt_run_id"],
            "visible_frames": int(test_pair["visible_frames"]),
            "not_visible_frames": int(test_pair["not_visible_frames"]),
            "confirmed_observations": int(len(test_pair["observations"])),
            "profiles": {},
        }
        for profile in ("temporal", "temporal_context"):
            fold["profiles"][profile] = _evaluate_profile(
                train_pairs,
                test_pair,
                profile=profile,
                targets=targets,
            )
        folds.append(fold)

    engineering_key = f"p{int(round(float(args.engineering_precision) * 100))}"
    north_star_key = f"p{int(round(float(args.north_star_precision) * 100))}"
    aggregate: dict[str, Any] = {}
    for profile in ("temporal", "temporal_context"):
        aggregate[profile] = {
            engineering_key: _aggregate(folds, profile, engineering_key),
            north_star_key: _aggregate(folds, profile, north_star_key),
        }

    temporal_eng = aggregate["temporal"][engineering_key]
    context_eng = aggregate["temporal_context"][engineering_key]
    aggregate["engineering_delta_context_minus_temporal"] = {
        "auc_macro": (
            None
            if temporal_eng["test_auc_macro"] is None or context_eng["test_auc_macro"] is None
            else round(float(context_eng["test_auc_macro"] - temporal_eng["test_auc_macro"]), 4)
        ),
        "precision_pp": round(
            float(context_eng["output_precision_micro_pct"] - temporal_eng["output_precision_micro_pct"]), 2
        ),
        "visible_coverage_pp": round(
            float(context_eng["visible_coverage_micro_pct"] - temporal_eng["visible_coverage_micro_pct"]), 2
        ),
        "not_visible_false_state_pp": round(
            float(context_eng["not_visible_false_state_micro_pct"] - temporal_eng["not_visible_false_state_micro_pct"]), 2
        ),
    }

    summary = {
        "scope": "LOVO OC-SORT Ball Identity: temporal-only vs temporal + court/player context",
        "engineering_precision_target_pct": round(100.0 * float(args.engineering_precision), 2),
        "north_star_precision_target_pct": round(100.0 * float(args.north_star_precision), 2),
        "context_semantics": "homography ball coordinates are planar-apparent soft evidence only",
        "folds": folds,
        "aggregate": aggregate,
        "decision_rule": (
            "Promote court/player context toward step 6 only if it improves held-out AUC and/or "
            "engineering-target coverage without materially worsening held-out precision, not-visible "
            "false state, or cross-video stability."
        ),
    }

    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("P0 BALL IDENTITY — TEMPORAL VS COURT/PLAYER CONTEXT LOVO")
    print("=" * 108)
    for fold in folds:
        temporal = fold["profiles"]["temporal"]
        context = fold["profiles"]["temporal_context"]
        t80 = temporal["targets"][engineering_key]
        c80 = context["targets"][engineering_key]
        print(
            f"{fold['test_run_id']}  "
            f"AUC temporal={temporal['test_observation_auc']} context={context['test_observation_auc']}  "
            f"P{int(round(args.engineering_precision * 100))} "
            f"temporal P={t80['output_precision_pct']:.2f}/C={t80['visible_coverage_pct']:.2f}/NV={t80['not_visible_false_state_pct']:.2f}  "
            f"context P={c80['output_precision_pct']:.2f}/C={c80['visible_coverage_pct']:.2f}/NV={c80['not_visible_false_state_pct']:.2f}"
        )

    print("\nAGGREGATE — ENGINEERING TARGET")
    print("-" * 88)
    for profile in ("temporal", "temporal_context"):
        row = aggregate[profile][engineering_key]
        print(
            f"{profile:18s} AUC={row['test_auc_macro']}  "
            f"P={row['output_precision_micro_pct']:.2f}%  "
            f"coverage={row['visible_coverage_micro_pct']:.2f}%  "
            f"NVfalse={row['not_visible_false_state_micro_pct']:.2f}%  "
            f"outputs={row['selected_output_frames']}"
        )
    delta = aggregate["engineering_delta_context_minus_temporal"]
    print(
        "context delta: "
        f"AUC={delta['auc_macro']}  P={delta['precision_pp']:+.2f}pp  "
        f"coverage={delta['visible_coverage_pp']:+.2f}pp  "
        f"NVfalse={delta['not_visible_false_state_pp']:+.2f}pp"
    )

    print("\nNORTH STAR REFERENCE")
    print("-" * 88)
    for profile in ("temporal", "temporal_context"):
        row = aggregate[profile][north_star_key]
        print(
            f"{profile:18s} P={row['output_precision_micro_pct']:.2f}%  "
            f"coverage={row['visible_coverage_micro_pct']:.2f}%  "
            f"NVfalse={row['not_visible_false_state_micro_pct']:.2f}%"
        )
    print(f"summary: {path}")


if __name__ == "__main__":
    main()
