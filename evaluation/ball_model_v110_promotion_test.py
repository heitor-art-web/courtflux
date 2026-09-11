from __future__ import annotations

from evaluation.ball_model_v110_promotion import operational_candidate, promotion_assessment


def metrics(*, recall: float, precision: float, f1: float, coverage: float) -> dict:
    return {
        "visible_recall": recall,
        "decision_precision": precision,
        "f1_recall_precision": f1,
        "output_coverage": coverage,
        "correct_visible": 100,
        "wrong_visible": 5,
        "false_positive_invisible": 10,
        "mean_correct_point_error_px": 8.0,
    }


def detector_context() -> dict:
    return {
        "basketball": {
            "v1.1.0_improves_precision": True,
            "v1.1.0_improves_recall": True,
            "v1.1.0_improves_f1": True,
        },
        "macro_best_f1": {
            "model_v1.0.0": 0.60,
            "model_v1.1.0": 0.80,
        },
    }


def main() -> None:
    reference = operational_candidate(
        model="model_v1.0.0",
        variant="trajectory",
        threshold=0.03,
        metrics=metrics(recall=0.88, precision=0.74, f1=0.80, coverage=0.87),
    )

    strict_challenger = operational_candidate(
        model="model_v1.1.0",
        variant="trajectory_retuned",
        threshold=0.02,
        metrics=metrics(recall=0.89, precision=0.75, f1=0.81, coverage=0.88),
    )
    strict = promotion_assessment(
        reference=reference,
        challenger=strict_challenger,
        detector_context=detector_context(),
        max_recall_drop_pp=1.0,
        max_precision_drop_pp=2.0,
        max_f1_drop_pp=0.5,
    )
    assert strict["strict_pass"] is True
    assert strict["recommendation"] == "PROMOTE_V1.1.0_CURRENT_P1_BASELINE_STRICT"

    balanced_challenger = operational_candidate(
        model="model_v1.1.0",
        variant="temporal_retuned",
        threshold=0.02,
        metrics=metrics(recall=0.875, precision=0.725, f1=0.798, coverage=0.89),
    )
    balanced = promotion_assessment(
        reference=reference,
        challenger=balanced_challenger,
        detector_context=detector_context(),
        max_recall_drop_pp=1.0,
        max_precision_drop_pp=2.0,
        max_f1_drop_pp=0.5,
    )
    assert balanced["strict_pass"] is False
    assert balanced["balanced_pass"] is True
    assert balanced["recommendation"] == "PROMOTE_V1.1.0_CURRENT_P1_BASELINE_BALANCED"

    failing_challenger = operational_candidate(
        model="model_v1.1.0",
        variant="direct",
        threshold=0.10,
        metrics=metrics(recall=0.82, precision=0.80, f1=0.75, coverage=0.70),
    )
    failing = promotion_assessment(
        reference=reference,
        challenger=failing_challenger,
        detector_context=detector_context(),
        max_recall_drop_pp=1.0,
        max_precision_drop_pp=2.0,
        max_f1_drop_pp=0.5,
    )
    assert failing["strict_pass"] is False
    assert failing["balanced_pass"] is False
    assert failing["recommendation"] == "KEEP_V1.0.0_OPERATIONAL_BALL_BASELINE"

    print("ball_model_v110_promotion_test: PASS")


if __name__ == "__main__":
    main()
