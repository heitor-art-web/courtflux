from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.ball_model_calibration_sweep import (
    best_threshold,
    infer_cache,
    parse_csv_floats,
    threshold_sweep,
    v110_policy_sweep,
)
from evaluation.ball_temporal_v1_real_eval import load_annotations


VARIANT_LABELS = {
    "direct": "direct",
    "temporal_gap4": "temporal_gap4",
    "trajectory_frozen_v100": "trajectory_frozen_v100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess whether model_v1.1.0 can replace model_v1.0.0 as the current P1 "
            "ball detector baseline after per-model threshold calibration and a small "
            "v1.1.0 tracker retuning sweep."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--detector-summary", default=None)
    parser.add_argument("--model-v100", default="weights/model_v1.0.0.pt")
    parser.add_argument("--model-v110", default="weights/model_v1.1.0.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--thresholds",
        default="0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--point-margin-px", type=float, default=4.0)
    parser.add_argument("--association-gate-norm", type=float, default=0.12)
    parser.add_argument("--gate-growth-per-frame", type=float, default=0.02)
    parser.add_argument("--max-gate-norm", type=float, default=0.24)
    parser.add_argument("--velocity-alpha", type=float, default=0.45)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--balanced-max-recall-drop-pp",
        type=float,
        default=1.0,
        help="Maximum recall regression allowed by the balanced promotion gate.",
    )
    parser.add_argument(
        "--balanced-max-precision-drop-pp",
        type=float,
        default=2.0,
        help="Maximum precision regression allowed by the balanced promotion gate.",
    )
    parser.add_argument(
        "--balanced-max-f1-drop-pp",
        type=float,
        default=0.5,
        help="Maximum F1 regression allowed by the balanced promotion gate.",
    )
    return parser.parse_args()


def safe_float(value: Any, fallback: float = 0.0) -> float:
    return float(value) if value is not None else fallback


def operational_candidate(
    *,
    model: str,
    variant: str,
    threshold: float,
    metrics: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "variant": variant,
        "threshold": float(threshold),
        "config": config or {},
        "metrics": metrics,
    }


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = candidate["metrics"]
    return (
        safe_float(metrics.get("f1_recall_precision")),
        safe_float(metrics.get("visible_recall")),
        safe_float(metrics.get("decision_precision"), -1.0),
        -float(candidate["threshold"]),
    )


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate["metrics"]
    return {
        "model": candidate["model"],
        "variant": candidate["variant"],
        "threshold": candidate["threshold"],
        "config": candidate.get("config", {}),
        "visible_recall": metrics.get("visible_recall"),
        "decision_precision": metrics.get("decision_precision"),
        "f1_recall_precision": metrics.get("f1_recall_precision"),
        "output_coverage": metrics.get("output_coverage"),
        "correct_visible": metrics.get("correct_visible"),
        "wrong_visible": metrics.get("wrong_visible"),
        "false_positive_invisible": metrics.get("false_positive_invisible"),
        "mean_correct_point_error_px": metrics.get("mean_correct_point_error_px"),
        "delta_vs_direct": metrics.get("delta_vs_direct"),
    }


def load_detector_context(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    v100 = models.get("model_v1.0.0", {})
    v110 = models.get("model_v1.1.0", {})
    if not v100 or not v110:
        raise ValueError("--detector-summary must contain model_v1.0.0 and model_v1.1.0")

    def class_best(model: dict[str, Any], class_name: str) -> dict[str, Any] | None:
        return model.get("per_class", {}).get(class_name, {}).get("best_by_f1")

    def macro_f1(model: dict[str, Any]) -> float:
        rows = []
        for class_name in ("basketball", "player", "referee", "rim", "backboard"):
            best = class_best(model, class_name)
            rows.append(safe_float(best.get("f1") if best else None))
        return sum(rows) / len(rows)

    ball100 = class_best(v100, "basketball")
    ball110 = class_best(v110, "basketball")
    if not ball100 or not ball110:
        raise ValueError("--detector-summary is missing basketball best_by_f1 metrics")

    return {
        "path": str(path),
        "dataset": payload.get("dataset"),
        "split": payload.get("split"),
        "image_count": payload.get("image_count"),
        "basketball": {
            "model_v1.0.0": ball100,
            "model_v1.1.0": ball110,
            "v1.1.0_improves_precision": safe_float(ball110.get("precision")) > safe_float(ball100.get("precision")),
            "v1.1.0_improves_recall": safe_float(ball110.get("recall")) > safe_float(ball100.get("recall")),
            "v1.1.0_improves_f1": safe_float(ball110.get("f1")) > safe_float(ball100.get("f1")),
        },
        "macro_best_f1": {
            "model_v1.0.0": macro_f1(v100),
            "model_v1.1.0": macro_f1(v110),
        },
    }


def promotion_assessment(
    *,
    reference: dict[str, Any],
    challenger: dict[str, Any],
    detector_context: dict[str, Any] | None,
    max_recall_drop_pp: float,
    max_precision_drop_pp: float,
    max_f1_drop_pp: float,
) -> dict[str, Any]:
    ref = reference["metrics"]
    new = challenger["metrics"]

    recall_delta_pp = 100.0 * (
        safe_float(new.get("visible_recall")) - safe_float(ref.get("visible_recall"))
    )
    precision_delta_pp = 100.0 * (
        safe_float(new.get("decision_precision")) - safe_float(ref.get("decision_precision"))
    )
    f1_delta_pp = 100.0 * (
        safe_float(new.get("f1_recall_precision")) - safe_float(ref.get("f1_recall_precision"))
    )
    coverage_delta_pp = 100.0 * (
        safe_float(new.get("output_coverage")) - safe_float(ref.get("output_coverage"))
    )

    supervised_required = detector_context is not None
    supervised_pass = True
    if detector_context is not None:
        basketball = detector_context["basketball"]
        supervised_pass = bool(
            basketball["v1.1.0_improves_precision"]
            and basketball["v1.1.0_improves_recall"]
            and basketball["v1.1.0_improves_f1"]
            and detector_context["macro_best_f1"]["model_v1.1.0"]
            > detector_context["macro_best_f1"]["model_v1.0.0"]
        )

    strict_operational_pass = (
        recall_delta_pp >= 0.0
        and precision_delta_pp >= 0.0
        and f1_delta_pp >= 0.0
    )
    balanced_operational_pass = (
        recall_delta_pp >= -max_recall_drop_pp
        and precision_delta_pp >= -max_precision_drop_pp
        and f1_delta_pp >= -max_f1_drop_pp
    )

    strict_pass = supervised_pass and strict_operational_pass
    balanced_pass = supervised_pass and balanced_operational_pass

    if strict_pass:
        recommendation = "PROMOTE_V1.1.0_CURRENT_P1_BASELINE_STRICT"
    elif balanced_pass:
        recommendation = "PROMOTE_V1.1.0_CURRENT_P1_BASELINE_BALANCED"
    else:
        recommendation = "KEEP_V1.0.0_OPERATIONAL_BALL_BASELINE"

    return {
        "reference": summarize_candidate(reference),
        "challenger": summarize_candidate(challenger),
        "delta_pp_v1.1.0_minus_v1.0.0": {
            "visible_recall": recall_delta_pp,
            "decision_precision": precision_delta_pp,
            "f1_recall_precision": f1_delta_pp,
            "output_coverage": coverage_delta_pp,
        },
        "supervised_detector_evidence_required": supervised_required,
        "supervised_detector_pass": supervised_pass,
        "strict_operational_pass": strict_operational_pass,
        "balanced_operational_pass": balanced_operational_pass,
        "strict_pass": strict_pass,
        "balanced_pass": balanced_pass,
        "balanced_tolerances_pp": {
            "max_recall_drop": max_recall_drop_pp,
            "max_precision_drop": max_precision_drop_pp,
            "max_f1_drop": max_f1_drop_pp,
        },
        "recommendation": recommendation,
        "evidence_status": "calibration_tuning_evidence_not_independent_held_out_promotion_evidence",
    }


def main() -> None:
    args = parse_args()
    thresholds = parse_csv_floats(args.thresholds)
    inference_conf = min(thresholds)

    video_path = Path(args.video).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    model_v100 = Path(args.model_v100).expanduser().resolve()
    model_v110 = Path(args.model_v110).expanduser().resolve()
    detector_summary = (
        Path(args.detector_summary).expanduser().resolve()
        if args.detector_summary
        else None
    )
    for path in (video_path, annotations_path, model_v100, model_v110):
        if not path.exists():
            raise FileNotFoundError(path)
    if detector_summary is not None and not detector_summary.exists():
        raise FileNotFoundError(detector_summary)

    annotations = load_annotations(annotations_path)
    first_frame, last_frame = min(annotations), max(annotations)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("evaluation")
        / "ball_model_v110_promotion"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_context = load_detector_context(detector_summary)

    cache_v100 = infer_cache(
        label="model_v1.0.0",
        model_path=model_v100,
        video_path=video_path,
        annotations=annotations,
        inference_conf=inference_conf,
        args=args,
    )
    cache_v110 = infer_cache(
        label="model_v1.1.0",
        model_path=model_v110,
        video_path=video_path,
        annotations=annotations,
        inference_conf=inference_conf,
        args=args,
    )

    sweep_v100 = threshold_sweep(
        cache=cache_v100,
        annotations=annotations,
        thresholds=thresholds,
        args=args,
    )
    sweep_v110 = threshold_sweep(
        cache=cache_v110,
        annotations=annotations,
        thresholds=thresholds,
        args=args,
    )

    best_v100_rows = {
        variant: best_threshold(sweep_v100, variant)
        for variant in VARIANT_LABELS
    }
    best_v110_rows = {
        variant: best_threshold(sweep_v110, variant)
        for variant in VARIANT_LABELS
    }

    v100_candidates = [
        operational_candidate(
            model="model_v1.0.0",
            variant=variant,
            threshold=row["threshold"],
            metrics=row[variant],
            config={"policy_status": "existing_or_frozen_v1.0.0_reference"},
        )
        for variant, row in best_v100_rows.items()
    ]

    v110_tuning = v110_policy_sweep(
        cache=cache_v110,
        annotations=annotations,
        temporal_threshold=best_v110_rows["temporal_gap4"]["threshold"],
        trajectory_threshold=best_v110_rows["trajectory_frozen_v100"]["threshold"],
        args=args,
    )

    best_temporal = v110_tuning["best_temporal"]
    best_trajectory = v110_tuning["best_trajectory"]
    direct_row = best_v110_rows["direct"]

    v110_candidates = [
        operational_candidate(
            model="model_v1.1.0",
            variant="direct",
            threshold=direct_row["threshold"],
            metrics=direct_row["direct"],
            config={"policy_status": "direct_calibrated"},
        ),
        operational_candidate(
            model="model_v1.1.0",
            variant="temporal_retuned",
            threshold=best_temporal["threshold"],
            metrics=best_temporal["metrics"],
            config={
                "max_prediction_gap": best_temporal["max_prediction_gap"],
                "source_counts": best_temporal.get("source_counts"),
            },
        ),
        operational_candidate(
            model="model_v1.1.0",
            variant="trajectory_retuned",
            threshold=best_trajectory["threshold"],
            metrics=best_trajectory["metrics"],
            config={
                "max_prediction_gap": best_trajectory["max_prediction_gap"],
                "gate": best_trajectory["gate"],
                "policy": best_trajectory["policy"],
                "source_counts": best_trajectory.get("source_counts"),
            },
        ),
    ]

    best_v100_operational = max(v100_candidates, key=candidate_rank)
    best_v110_operational = max(v110_candidates, key=candidate_rank)

    assessment = promotion_assessment(
        reference=best_v100_operational,
        challenger=best_v110_operational,
        detector_context=detector_context,
        max_recall_drop_pp=args.balanced_max_recall_drop_pp,
        max_precision_drop_pp=args.balanced_max_precision_drop_pp,
        max_f1_drop_pp=args.balanced_max_f1_drop_pp,
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 operational promotion assessment for model_v1.1.0",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "frame_range": [first_frame, last_frame],
        "device": args.device,
        "imgsz": args.imgsz,
        "thresholds": thresholds,
        "inference_conf": inference_conf,
        "detector_supervised_context": detector_context,
        "model_v1.0.0_candidates": [summarize_candidate(row) for row in v100_candidates],
        "model_v1.1.0_candidates": [summarize_candidate(row) for row in v110_candidates],
        "model_v1.1.0_retuning": v110_tuning,
        "promotion_assessment": assessment,
        "roadmap": {
            "current_goal": "Promote model_v1.1.0 as the current P1 detector baseline if supervised superiority transfers to an operational ball configuration without material regression.",
            "p1_exit_goal": "Train/evaluate model_v1.2.0 and rerun the full P1 regression suite across detector, ball, player, referee, rim/backboard, jersey identity, BallState, PlayerState and CourtState before P1 exit.",
            "backboard_policy": "Treat backboard as improving but non-blocking auxiliary evidence; expect later detector versions to improve it while rim remains the stronger basket-geometry signal.",
        },
        "notes": [
            "Each detector is inferred once at the minimum confidence and all higher thresholds are replayed from the same candidate cache.",
            "model_v1.0.0 is allowed to use its best calibrated direct/temporal/frozen-trajectory result; model_v1.1.0 is allowed a small threshold, temporal-gap and trajectory-policy retuning sweep.",
            "The strict gate requires no operational regression in recall, precision or F1.",
            "The balanced gate allows small configured operational regressions only when supervised detector evidence confirms v1.1.0 improves basketball precision, recall, F1 and multi-class macro F1.",
            "This Porto-Ovarense segment is still calibration/tuning evidence, not an independent held-out promotion set. A promotion here is a current P1 baseline decision, not final P1 exit validation.",
        ],
    }

    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ref = assessment["reference"]
    new = assessment["challenger"]
    delta = assessment["delta_pp_v1.1.0_minus_v1.0.0"]
    print("\nP1 ball promotion comparison")
    print(
        f"v1.0.0 best: {ref['variant']} conf={ref['threshold']:.3f} "
        f"recall={ref['visible_recall']:.3f} precision={ref['decision_precision']:.3f} "
        f"f1={ref['f1_recall_precision']:.3f} coverage={ref['output_coverage']:.3f}"
    )
    print(
        f"v1.1.0 best: {new['variant']} conf={new['threshold']:.3f} "
        f"recall={new['visible_recall']:.3f} precision={new['decision_precision']:.3f} "
        f"f1={new['f1_recall_precision']:.3f} coverage={new['output_coverage']:.3f}"
    )
    print(
        "delta v1.1.0-v1.0.0: "
        f"recall={delta['visible_recall']:+.3f}pp "
        f"precision={delta['decision_precision']:+.3f}pp "
        f"f1={delta['f1_recall_precision']:+.3f}pp "
        f"coverage={delta['output_coverage']:+.3f}pp"
    )
    print(
        f"strict={assessment['strict_pass']} balanced={assessment['balanced_pass']} "
        f"recommendation={assessment['recommendation']}"
    )
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
