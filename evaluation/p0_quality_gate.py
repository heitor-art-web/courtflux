from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import (
    align_production_truth,
    candidate_matches_truth,
    center_distance,
    load_jsonl,
)


@dataclass(frozen=True)
class GateThresholds:
    visible_coverage_micro_min_pct: float = 80.0
    visible_coverage_macro_min_pct: float = 78.0
    visible_coverage_worst_min_pct: float = 70.0
    output_precision_min_pct: float = 80.0
    not_visible_false_state_max_pct: float = 5.0
    unsupported_gap_p90_max_frames_exclusive: float = 8.0
    catastrophic_error_events_max: int = 0
    direct_candidate_recall_supporting_min_pct: float = 65.0


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * float(numerator) / max(float(denominator), 1.0), 2)


def percentile(values: list[int] | list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(float(q), 1.0))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _association(production: dict[str, Any]) -> dict[str, Any]:
    value = production.get("ball_association")
    return value if isinstance(value, dict) else {}


def _candidate_rows(production: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = _association(production).get("candidates") or []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _output(production: dict[str, Any]) -> tuple[tuple[float, float] | None, str]:
    """Return concrete output location and coarse source: direct/predicted/missing."""
    association = _association(production)
    source = str(
        production.get("ball_source")
        or association.get("source")
        or ("detector" if production.get("selected_ball") else "missing")
    ).strip().lower()

    direct_center = _point(production.get("selected_ball_center"))
    if direct_center is not None and bool(production.get("selected_ball", False)):
        return direct_center, "direct"

    trajectory_center = _point(association.get("trajectory_center"))
    if trajectory_center is None:
        trajectory_center = _point(production.get("trajectory_ball_center"))

    if source == "predicted_hold" and trajectory_center is not None:
        return trajectory_center, "predicted"

    # Older diagnostics may expose a direct associated location only through the
    # trajectory payload. Require an actual selected index/direct-looking source so a
    # stale hold is never silently reclassified as a direct observation.
    if (
        trajectory_center is not None
        and association.get("selected_index") is not None
        and source not in {"missing", "predicted_hold"}
    ):
        return trajectory_center, "direct"

    return None, "missing"


def _matches_output(
    center: tuple[float, float] | None,
    truth: dict[str, Any],
    radius_px: float,
) -> tuple[bool, float | None]:
    if center is None or not bool(truth.get("visible")):
        return False, None
    point = truth.get("point")
    if isinstance(point, list) and len(point) == 2:
        distance = center_distance(center, point)
        return distance <= float(radius_px), float(distance)

    candidate = {"center": center}
    ok, distance = candidate_matches_truth(candidate, truth, radius_px, 0.25)
    return bool(ok), distance


def _close_gap(current: int, output: list[int]) -> int:
    if current > 0:
        output.append(int(current))
    return 0


def _summary_counts_template() -> dict[str, int]:
    return {
        "labelled_frames": 0,
        "visible_frames": 0,
        "not_visible_frames": 0,
        "candidate_frames": 0,
        "candidate_instances": 0,
        "candidate_true_instances": 0,
        "candidate_false_instances": 0,
        "visible_candidate_correct_frames": 0,
        "not_visible_candidate_frames": 0,
        "direct_output_frames": 0,
        "predicted_output_frames": 0,
        "missing_output_frames": 0,
        "visible_direct_correct_frames": 0,
        "visible_direct_wrong_frames": 0,
        "visible_predicted_correct_frames": 0,
        "visible_predicted_wrong_frames": 0,
        "visible_missing_frames": 0,
        "visible_usable_correct_frames": 0,
        "concrete_output_frames": 0,
        "concrete_correct_frames": 0,
        "not_visible_false_state_frames": 0,
        "not_visible_direct_false_frames": 0,
        "not_visible_predicted_false_frames": 0,
        "association_opportunity_frames": 0,
        "association_opportunity_missed_frames": 0,
        "catastrophic_error_frames": 0,
        "catastrophic_error_events": 0,
        "reacquisition_events": 0,
    }


def evaluate_pair(
    frames_path: Path,
    annotations_path: Path,
    *,
    radius_px: float = 40.0,
    catastrophic_radius_px: float = 160.0,
    timestamp_tolerance_sec: float = 0.001,
) -> dict[str, Any]:
    production_rows = load_jsonl(frames_path)
    truth_rows = load_jsonl(annotations_path)
    aligned, unmatched = align_production_truth(
        production_rows,
        truth_rows,
        tolerance_sec=float(timestamp_tolerance_sec),
    )

    counts = _summary_counts_template()
    unsupported_gaps: list[int] = []
    current_unsupported_gap = 0
    catastrophic_active = False
    hold_ages: list[int] = []
    visible_errors: list[float] = []

    for production, truth, _delta in aligned:
        visible = bool(truth.get("visible"))
        counts["labelled_frames"] += 1
        if visible:
            counts["visible_frames"] += 1
        else:
            counts["not_visible_frames"] += 1

        candidates = _candidate_rows(production)
        if candidates:
            counts["candidate_frames"] += 1
        counts["candidate_instances"] += len(candidates)

        true_candidate_count = 0
        for candidate in candidates:
            ok, _distance = candidate_matches_truth(
                candidate,
                truth,
                float(radius_px),
                0.25,
            )
            if ok:
                true_candidate_count += 1
        counts["candidate_true_instances"] += true_candidate_count
        counts["candidate_false_instances"] += max(
            len(candidates) - true_candidate_count,
            0,
        )

        candidate_correct = bool(visible and true_candidate_count > 0)
        if candidate_correct:
            counts["visible_candidate_correct_frames"] += 1
        if not visible and candidates:
            counts["not_visible_candidate_frames"] += 1

        center, output_kind = _output(production)
        if output_kind == "direct":
            counts["direct_output_frames"] += 1
        elif output_kind == "predicted":
            counts["predicted_output_frames"] += 1
            age = _association(production).get("hold_age_frames")
            try:
                if age is not None and int(age) > 0:
                    hold_ages.append(int(age))
            except (TypeError, ValueError):
                pass
        else:
            counts["missing_output_frames"] += 1

        association = _association(production)
        source = str(
            association.get("source") or production.get("ball_source") or ""
        ).lower()
        if bool(association.get("reacquired", False)) or source == "detector_reacquired":
            counts["reacquisition_events"] += 1

        if visible:
            correct, distance = _matches_output(center, truth, float(radius_px))
            if distance is not None:
                visible_errors.append(float(distance))

            if center is not None:
                counts["concrete_output_frames"] += 1
            if correct:
                counts["concrete_correct_frames"] += 1
                counts["visible_usable_correct_frames"] += 1

            if output_kind == "direct":
                if correct:
                    counts["visible_direct_correct_frames"] += 1
                else:
                    counts["visible_direct_wrong_frames"] += 1
            elif output_kind == "predicted":
                if correct:
                    counts["visible_predicted_correct_frames"] += 1
                else:
                    counts["visible_predicted_wrong_frames"] += 1
            else:
                counts["visible_missing_frames"] += 1

            if candidate_correct:
                counts["association_opportunity_frames"] += 1
                if not correct:
                    counts["association_opportunity_missed_frames"] += 1

            if correct:
                current_unsupported_gap = _close_gap(
                    current_unsupported_gap,
                    unsupported_gaps,
                )
                catastrophic_active = False
            else:
                current_unsupported_gap += 1
                catastrophic = bool(
                    center is not None
                    and distance is not None
                    and float(distance) > float(catastrophic_radius_px)
                )
                if catastrophic:
                    counts["catastrophic_error_frames"] += 1
                    if not catastrophic_active:
                        counts["catastrophic_error_events"] += 1
                    catastrophic_active = True
                else:
                    catastrophic_active = False
        else:
            # Correct not-visible state is no concrete location. Any emitted point is a
            # false state and therefore also counts against output precision.
            current_unsupported_gap = _close_gap(
                current_unsupported_gap,
                unsupported_gaps,
            )
            catastrophic_active = False
            if center is not None:
                counts["concrete_output_frames"] += 1
                counts["not_visible_false_state_frames"] += 1
                if output_kind == "direct":
                    counts["not_visible_direct_false_frames"] += 1
                elif output_kind == "predicted":
                    counts["not_visible_predicted_false_frames"] += 1

    _close_gap(current_unsupported_gap, unsupported_gaps)

    visible = counts["visible_frames"]
    not_visible = counts["not_visible_frames"]
    emitted = counts["concrete_output_frames"]
    opportunity = counts["association_opportunity_frames"]
    direct_emitted = (
        counts["visible_direct_correct_frames"]
        + counts["visible_direct_wrong_frames"]
        + counts["not_visible_direct_false_frames"]
    )
    predicted_emitted = (
        counts["visible_predicted_correct_frames"]
        + counts["visible_predicted_wrong_frames"]
        + counts["not_visible_predicted_false_frames"]
    )

    direct_coverage = pct(counts["visible_direct_correct_frames"], visible)
    usable_coverage = pct(counts["visible_usable_correct_frames"], visible)

    metrics = {
        "p0_1_detector_candidates": {
            "visible_candidate_recall_pct": pct(
                counts["visible_candidate_correct_frames"],
                visible,
            ),
            "candidate_instance_precision_pct": pct(
                counts["candidate_true_instances"],
                counts["candidate_instances"],
            ),
            "not_visible_candidate_frame_rate_pct": pct(
                counts["not_visible_candidate_frames"],
                not_visible,
            ),
            "candidate_instances_per_labelled_frame": round(
                counts["candidate_instances"] / max(counts["labelled_frames"], 1),
                4,
            ),
        },
        "p0_2_association": {
            "direct_correct_coverage_pct": direct_coverage,
            "direct_output_precision_pct": pct(
                counts["visible_direct_correct_frames"],
                direct_emitted,
            ),
            "association_opportunity_frames": opportunity,
            "association_opportunity_miss_rate_pct": pct(
                counts["association_opportunity_missed_frames"],
                opportunity,
            ),
        },
        "p0_3_trajectory": {
            "usable_correct_visible_coverage_pct": usable_coverage,
            "coverage_lift_over_direct_pp": round(usable_coverage - direct_coverage, 2),
            "predicted_correct_visible_frames": counts["visible_predicted_correct_frames"],
            "predicted_wrong_visible_frames": counts["visible_predicted_wrong_frames"],
            "predicted_output_precision_pct": pct(
                counts["visible_predicted_correct_frames"],
                predicted_emitted,
            ),
            "hold_age_p90_frames": (
                None
                if not hold_ages
                else round(float(percentile(hold_ages, 0.90) or 0.0), 3)
            ),
        },
        "p0_4_state_robustness": {
            "not_visible_false_state_rate_pct": pct(
                counts["not_visible_false_state_frames"],
                not_visible,
            ),
            "not_visible_direct_false_frames": counts["not_visible_direct_false_frames"],
            "not_visible_predicted_false_frames": counts[
                "not_visible_predicted_false_frames"
            ],
            "catastrophic_radius_px": float(catastrophic_radius_px),
            "catastrophic_error_frames": counts["catastrophic_error_frames"],
            "catastrophic_error_events": counts["catastrophic_error_events"],
            "reacquisition_events": counts["reacquisition_events"],
        },
        "p0_5_system": {
            "visible_usable_correct_coverage_pct": usable_coverage,
            "concrete_output_precision_pct": pct(
                counts["concrete_correct_frames"],
                emitted,
            ),
            "unsupported_gap_count": len(unsupported_gaps),
            "unsupported_gap_p90_frames": (
                None
                if not unsupported_gaps
                else round(float(percentile(unsupported_gaps, 0.90) or 0.0), 3)
            ),
            "unsupported_gap_max_frames": (
                None if not unsupported_gaps else max(unsupported_gaps)
            ),
            "visible_location_error_median_px": (
                None
                if not visible_errors
                else round(float(statistics.median(visible_errors)), 3)
            ),
            "visible_location_error_p90_px": (
                None
                if not visible_errors
                else round(float(percentile(visible_errors, 0.90) or 0.0), 3)
            ),
        },
    }

    return {
        "frames_path": str(frames_path),
        "annotations_path": str(annotations_path),
        "gt_radius_px": float(radius_px),
        "catastrophic_radius_px": float(catastrophic_radius_px),
        "aligned_labelled_frames": counts["labelled_frames"],
        "unmatched_gameplay_frames": int(unmatched),
        "counts": counts,
        "unsupported_gaps_frames": unsupported_gaps,
        "metrics": metrics,
    }


def _sum_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    totals = _summary_counts_template()
    for result in results:
        counts = result["counts"]
        for key in totals:
            totals[key] += int(counts.get(key, 0) or 0)
    return totals


def _aggregate_metrics(
    results: list[dict[str, Any]],
    totals: dict[str, int],
    *,
    catastrophic_radius_px: float,
) -> dict[str, Any]:
    visible = totals["visible_frames"]
    not_visible = totals["not_visible_frames"]
    emitted = totals["concrete_output_frames"]
    direct_emitted = (
        totals["visible_direct_correct_frames"]
        + totals["visible_direct_wrong_frames"]
        + totals["not_visible_direct_false_frames"]
    )
    predicted_emitted = (
        totals["visible_predicted_correct_frames"]
        + totals["visible_predicted_wrong_frames"]
        + totals["not_visible_predicted_false_frames"]
    )

    coverage_by_run = [
        float(result["metrics"]["p0_5_system"]["visible_usable_correct_coverage_pct"])
        for result in results
    ]
    gap_lengths = [
        int(gap)
        for result in results
        for gap in result.get("unsupported_gaps_frames", [])
    ]

    direct_coverage = pct(totals["visible_direct_correct_frames"], visible)
    usable_coverage = pct(totals["visible_usable_correct_frames"], visible)

    return {
        "p0_1_detector_candidates": {
            "visible_candidate_recall_micro_pct": pct(
                totals["visible_candidate_correct_frames"],
                visible,
            ),
            "candidate_instance_precision_micro_pct": pct(
                totals["candidate_true_instances"],
                totals["candidate_instances"],
            ),
            "not_visible_candidate_frame_rate_micro_pct": pct(
                totals["not_visible_candidate_frames"],
                not_visible,
            ),
        },
        "p0_2_association": {
            "direct_correct_coverage_micro_pct": direct_coverage,
            "direct_output_precision_micro_pct": pct(
                totals["visible_direct_correct_frames"],
                direct_emitted,
            ),
            "association_opportunity_miss_rate_micro_pct": pct(
                totals["association_opportunity_missed_frames"],
                totals["association_opportunity_frames"],
            ),
        },
        "p0_3_trajectory": {
            "usable_correct_visible_coverage_micro_pct": usable_coverage,
            "coverage_lift_over_direct_micro_pp": round(
                usable_coverage - direct_coverage,
                2,
            ),
            "predicted_output_precision_micro_pct": pct(
                totals["visible_predicted_correct_frames"],
                predicted_emitted,
            ),
        },
        "p0_4_state_robustness": {
            "not_visible_false_state_rate_micro_pct": pct(
                totals["not_visible_false_state_frames"],
                not_visible,
            ),
            "catastrophic_radius_px": float(catastrophic_radius_px),
            "catastrophic_error_frames": totals["catastrophic_error_frames"],
            "catastrophic_error_events": totals["catastrophic_error_events"],
            "reacquisition_events": totals["reacquisition_events"],
        },
        "p0_5_system": {
            "visible_usable_correct_coverage_micro_pct": usable_coverage,
            "visible_usable_correct_coverage_macro_pct": (
                None
                if not coverage_by_run
                else round(float(statistics.fmean(coverage_by_run)), 2)
            ),
            "visible_usable_correct_coverage_worst_pct": (
                None if not coverage_by_run else round(min(coverage_by_run), 2)
            ),
            "concrete_output_precision_micro_pct": pct(
                totals["concrete_correct_frames"],
                emitted,
            ),
            "unsupported_gap_p90_frames": (
                None
                if not gap_lengths
                else round(float(percentile(gap_lengths, 0.90) or 0.0), 3)
            ),
            "unsupported_gap_max_frames": (
                None if not gap_lengths else max(gap_lengths)
            ),
        },
    }


def evaluate_gate(
    aggregate: dict[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    system = aggregate["p0_5_system"]
    state = aggregate["p0_4_state_robustness"]
    detector = aggregate["p0_1_detector_candidates"]
    gap_p90 = system.get("unsupported_gap_p90_frames")

    blocking_checks = {
        "coverage_micro": {
            "value": system["visible_usable_correct_coverage_micro_pct"],
            "op": ">=",
            "threshold": thresholds.visible_coverage_micro_min_pct,
            "pass": system["visible_usable_correct_coverage_micro_pct"]
            >= thresholds.visible_coverage_micro_min_pct,
        },
        "coverage_macro": {
            "value": system["visible_usable_correct_coverage_macro_pct"],
            "op": ">=",
            "threshold": thresholds.visible_coverage_macro_min_pct,
            "pass": (
                system["visible_usable_correct_coverage_macro_pct"] is not None
                and system["visible_usable_correct_coverage_macro_pct"]
                >= thresholds.visible_coverage_macro_min_pct
            ),
        },
        "coverage_worst_video": {
            "value": system["visible_usable_correct_coverage_worst_pct"],
            "op": ">=",
            "threshold": thresholds.visible_coverage_worst_min_pct,
            "pass": (
                system["visible_usable_correct_coverage_worst_pct"] is not None
                and system["visible_usable_correct_coverage_worst_pct"]
                >= thresholds.visible_coverage_worst_min_pct
            ),
        },
        "output_precision": {
            "value": system["concrete_output_precision_micro_pct"],
            "op": ">=",
            "threshold": thresholds.output_precision_min_pct,
            "pass": system["concrete_output_precision_micro_pct"]
            >= thresholds.output_precision_min_pct,
        },
        "not_visible_false_state": {
            "value": state["not_visible_false_state_rate_micro_pct"],
            "op": "<=",
            "threshold": thresholds.not_visible_false_state_max_pct,
            "pass": state["not_visible_false_state_rate_micro_pct"]
            <= thresholds.not_visible_false_state_max_pct,
        },
        "unsupported_gap_p90": {
            "value": gap_p90,
            "op": "<",
            "threshold": thresholds.unsupported_gap_p90_max_frames_exclusive,
            "pass": (
                gap_p90 is None
                or gap_p90 < thresholds.unsupported_gap_p90_max_frames_exclusive
            ),
        },
        "catastrophic_error_events": {
            "value": state["catastrophic_error_events"],
            "op": "<=",
            "threshold": thresholds.catastrophic_error_events_max,
            "pass": state["catastrophic_error_events"]
            <= thresholds.catastrophic_error_events_max,
        },
    }

    supporting_checks = {
        "direct_candidate_recall": {
            "value": detector["visible_candidate_recall_micro_pct"],
            "op": ">=",
            "threshold": thresholds.direct_candidate_recall_supporting_min_pct,
            "pass": detector["visible_candidate_recall_micro_pct"]
            >= thresholds.direct_candidate_recall_supporting_min_pct,
            "blocking": False,
        }
    }

    return {
        "status": (
            "PASS"
            if all(bool(check["pass"]) for check in blocking_checks.values())
            else "FAIL"
        ),
        "blocking_checks": blocking_checks,
        "supporting_checks": supporting_checks,
        "thresholds": asdict(thresholds),
        "note": (
            "Direct detector recall is supporting evidence, not a standalone P0 blocker. "
            "Performance remains informational until an explicit throughput threshold is frozen."
        ),
    }


def build_summary(
    results: list[dict[str, Any]],
    *,
    thresholds: GateThresholds,
    catastrophic_radius_px: float,
) -> dict[str, Any]:
    totals = _sum_counts(results)
    aggregate = _aggregate_metrics(
        results,
        totals,
        catastrophic_radius_px=float(catastrophic_radius_px),
    )
    gate = evaluate_gate(aggregate, thresholds)
    return {
        "scope": "CourtFlux P0 system quality gate",
        "metric_contract": {
            "official_gt_match": "<=40px center distance by default",
            "coverage": (
                "visible GT frames with a correct direct or predicted concrete ball location"
            ),
            "precision": (
                "correct visible concrete outputs / all concrete outputs on labelled visible + not-visible frames"
            ),
            "not_visible_false_state": (
                "not-visible GT frames where a concrete location was emitted"
            ),
            "catastrophic_event": (
                "new contiguous visible-error episode with emitted location farther than catastrophic_radius_px from GT"
            ),
            "unsupported_gap": (
                "contiguous visible GT frames without a correct usable location; not-visible frames terminate a gap"
            ),
        },
        "runs": results,
        "totals": totals,
        "aggregate": aggregate,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CourtFlux P0.1-P0.5 against frozen point GT and apply the system-level engineering gate."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--catastrophic-radius-px", type=float, default=160.0)
    parser.add_argument("--timestamp-tolerance-sec", type=float, default=0.001)
    parser.add_argument("--coverage-micro-min", type=float, default=80.0)
    parser.add_argument("--coverage-macro-min", type=float, default=78.0)
    parser.add_argument("--coverage-worst-min", type=float, default=70.0)
    parser.add_argument("--precision-min", type=float, default=80.0)
    parser.add_argument("--not-visible-false-max", type=float, default=5.0)
    parser.add_argument("--unsupported-gap-p90-max", type=float, default=8.0)
    parser.add_argument("--catastrophic-events-max", type=int, default=0)
    parser.add_argument(
        "--direct-candidate-recall-supporting-min",
        type=float,
        default=65.0,
    )
    parser.add_argument("--output", default="evaluation/p0_quality_gate")
    args = parser.parse_args()

    thresholds = GateThresholds(
        visible_coverage_micro_min_pct=float(args.coverage_micro_min),
        visible_coverage_macro_min_pct=float(args.coverage_macro_min),
        visible_coverage_worst_min_pct=float(args.coverage_worst_min),
        output_precision_min_pct=float(args.precision_min),
        not_visible_false_state_max_pct=float(args.not_visible_false_max),
        unsupported_gap_p90_max_frames_exclusive=float(args.unsupported_gap_p90_max),
        catastrophic_error_events_max=int(args.catastrophic_events_max),
        direct_candidate_recall_supporting_min_pct=float(
            args.direct_candidate_recall_supporting_min
        ),
    )

    run_root = Path(args.run_root).resolve()
    gt_root = Path(args.gt_root).resolve()
    results: list[dict[str, Any]] = []
    for run_id, gt_id in args.pair:
        result = evaluate_pair(
            run_root / run_id / "frames.jsonl",
            gt_root / gt_id / "annotations.jsonl",
            radius_px=float(args.gt_radius_px),
            catastrophic_radius_px=float(args.catastrophic_radius_px),
            timestamp_tolerance_sec=float(args.timestamp_tolerance_sec),
        )
        result["run_id"] = run_id
        result["gt_run_id"] = gt_id
        results.append(result)

    summary = build_summary(
        results,
        thresholds=thresholds,
        catastrophic_radius_px=float(args.catastrophic_radius_px),
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[P0 QUALITY GATE] {summary['gate']['status']} {summary_path}")


if __name__ == "__main__":
    main()
