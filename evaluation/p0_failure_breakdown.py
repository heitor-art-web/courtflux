from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def f(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a compact per-video diagnosis from p0_quality_gate summary.json."
    )
    parser.add_argument(
        "summary",
        nargs="?",
        default="evaluation/p0_quality_gate_smart_v1/summary.json",
    )
    args = parser.parse_args()

    path = Path(args.summary)
    data = json.loads(path.read_text(encoding="utf-8"))

    print("P0 FAILURE BREAKDOWN")
    print("=" * 118)
    print(
        f"{'run':27} {'cand%':>7} {'direct%':>8} {'usable%':>8} {'dir_prec%':>10} "
        f"{'pred_prec%':>11} {'assoc_miss%':>11} {'nv_false%':>10} {'gap_p90':>8} {'cat_evt':>8}"
    )
    print("-" * 118)

    worst: tuple[float, str] | None = None
    for run in data.get("runs", []):
        metrics = run.get("metrics", {})
        p01 = metrics.get("p0_1_detector_candidates", {})
        p02 = metrics.get("p0_2_association", {})
        p03 = metrics.get("p0_3_trajectory", {})
        p04 = metrics.get("p0_4_state_robustness", {})
        p05 = metrics.get("p0_5_system", {})
        run_id = str(run.get("run_id", "?"))
        usable = float(p05.get("visible_usable_correct_coverage_pct", 0.0) or 0.0)
        if worst is None or usable < worst[0]:
            worst = (usable, run_id)
        print(
            f"{run_id:27} "
            f"{f(p01.get('visible_candidate_recall_pct')):>7} "
            f"{f(p02.get('direct_correct_coverage_pct')):>8} "
            f"{f(usable):>8} "
            f"{f(p02.get('direct_output_precision_pct')):>10} "
            f"{f(p03.get('predicted_output_precision_pct')):>11} "
            f"{f(p02.get('association_opportunity_miss_rate_pct')):>11} "
            f"{f(p04.get('not_visible_false_state_rate_pct')):>10} "
            f"{f(p05.get('unsupported_gap_p90_frames')):>8} "
            f"{f(p04.get('catastrophic_error_events')):>8}"
        )

    aggregate = data.get("aggregate", {})
    p01 = aggregate.get("p0_1_detector_candidates", {})
    p02 = aggregate.get("p0_2_association", {})
    p03 = aggregate.get("p0_3_trajectory", {})
    p04 = aggregate.get("p0_4_state_robustness", {})
    p05 = aggregate.get("p0_5_system", {})
    totals = data.get("totals", {})

    print("=" * 118)
    print("AGGREGATE")
    print(f"candidate recall micro:          {f(p01.get('visible_candidate_recall_micro_pct'))}%")
    print(f"direct correct coverage micro:   {f(p02.get('direct_correct_coverage_micro_pct'))}%")
    print(f"usable correct coverage micro:   {f(p05.get('visible_usable_correct_coverage_micro_pct'))}%")
    print(f"trajectory lift over direct:     {f(p03.get('coverage_lift_over_direct_micro_pp'))} pp")
    print(f"direct output precision:         {f(p02.get('direct_output_precision_micro_pct'))}%")
    print(f"predicted output precision:      {f(p03.get('predicted_output_precision_micro_pct'))}%")
    print(f"association opportunity miss:    {f(p02.get('association_opportunity_miss_rate_micro_pct'))}%")
    print(f"not-visible false state:         {f(p04.get('not_visible_false_state_rate_micro_pct'))}%")
    print(f"unsupported gap P90:             {f(p05.get('unsupported_gap_p90_frames'))} frames")
    print(f"catastrophic error events:       {f(p04.get('catastrophic_error_events'))}")
    print(f"reacquisition events:            {f(p04.get('reacquisition_events'))}")
    print(f"not-visible direct false frames: {f(totals.get('not_visible_direct_false_frames'))}")
    print(f"not-visible pred false frames:   {f(totals.get('not_visible_predicted_false_frames'))}")
    print(f"visible direct wrong frames:     {f(totals.get('visible_direct_wrong_frames'))}")
    print(f"visible predicted wrong frames:  {f(totals.get('visible_predicted_wrong_frames'))}")
    print(f"visible missing frames:          {f(totals.get('visible_missing_frames'))}")
    if worst is not None:
        print(f"worst video:                     {worst[1]} ({worst[0]:.2f}% usable coverage)")

    print("\nINTERPRETATION")
    candidate_recall = float(p01.get("visible_candidate_recall_micro_pct", 0.0) or 0.0)
    direct_coverage = float(p02.get("direct_correct_coverage_micro_pct", 0.0) or 0.0)
    usable = float(p05.get("visible_usable_correct_coverage_micro_pct", 0.0) or 0.0)
    assoc_loss = max(candidate_recall - direct_coverage, 0.0)
    temporal_lift = max(usable - direct_coverage, 0.0)
    print(f"candidate->direct association loss: {assoc_loss:.2f} pp")
    print(f"predicted-hold net coverage lift:   {temporal_lift:.2f} pp")
    print(
        "Candidate recall is an upper bound on direct association coverage: if candidate recall remains below the "
        "supporting target, association alone cannot close P0."
    )


if __name__ == "__main__":
    main()
