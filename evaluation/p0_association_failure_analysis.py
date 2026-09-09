from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import align_production_truth, candidate_center, load_jsonl


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def dist(center: tuple[float, float] | None, point: list[float] | None) -> float | None:
    if center is None or not isinstance(point, list) or len(point) != 2:
        return None
    return math.hypot(float(center[0]) - float(point[0]), float(center[1]) - float(point[1]))


def association(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("ball_association")
    return value if isinstance(value, dict) else {}


def traces(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in association(row).get("candidates", []) if isinstance(x, dict)]


def output_center(row: dict[str, Any]) -> tuple[float, float] | None:
    center = row.get("selected_ball_center")
    if isinstance(center, (list, tuple)) and len(center) >= 2 and row.get("selected_ball"):
        return float(center[0]), float(center[1])
    a = association(row)
    center = a.get("trajectory_center") or row.get("trajectory_ball_center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    return None


def selected_trace(row: dict[str, Any]) -> dict[str, Any] | None:
    a = association(row)
    idx = a.get("selected_index")
    items = traces(row)
    try:
        index = int(idx)
    except (TypeError, ValueError):
        return None
    if 0 <= index < len(items):
        return items[index]
    return None


def classify_true_candidate_state(
    row: dict[str, Any], truth: dict[str, Any], radius_px: float
) -> tuple[list[dict[str, Any]], str]:
    point = truth.get("point")
    true_traces: list[dict[str, Any]] = []
    for item in traces(row):
        d = dist(candidate_center(item), point)
        if d is not None and d <= radius_px:
            true_traces.append(item)

    if not true_traces:
        return [], "no_true_candidate"

    details = [item.get("association") for item in true_traces]
    if all(value is None for value in details):
        return true_traces, "no_temporal_score"

    plausible = [
        value
        for value in details
        if isinstance(value, dict) and bool(value.get("plausible", False))
    ]
    if plausible:
        return true_traces, "true_candidate_plausible"
    return true_traces, "true_candidate_outside_gate"


def evaluate(
    frames_path: Path,
    gt_path: Path,
    *,
    radius_px: float,
    catastrophic_radius_px: float,
    tolerance_sec: float,
) -> dict[str, Any]:
    production = load_jsonl(frames_path)
    truth = load_jsonl(gt_path)
    aligned, unmatched = align_production_truth(production, truth, tolerance_sec=tolerance_sec)

    opportunity_causes: Counter[str] = Counter()
    wrong_direct_causes: Counter[str] = Counter()
    catastrophic_causes: Counter[str] = Counter()
    not_visible_direct_causes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    visible = 0
    opportunities = 0
    opportunity_misses = 0
    wrong_direct = 0
    catastrophic_direct = 0
    not_visible_direct = 0

    for row, gt, _delta in aligned:
        a = association(row)
        source = str(a.get("source") or row.get("ball_source") or "missing").lower()
        source_counts[source] += 1
        center = output_center(row)
        point = gt.get("point")
        d = dist(center, point)
        visible_gt = bool(gt.get("visible"))
        selected = selected_trace(row)
        selected_details = selected.get("association") if selected else None

        if visible_gt:
            visible += 1
            true_items, true_state = classify_true_candidate_state(row, gt, radius_px)
            correct = bool(d is not None and d <= radius_px)

            if true_items:
                opportunities += 1
                if not correct:
                    opportunity_misses += 1
                    if true_state == "no_temporal_score":
                        cause = "initialization_or_reset_no_temporal_score"
                    elif true_state == "true_candidate_outside_gate":
                        cause = "gate_loss_true_candidate_outside_gate"
                    elif source == "predicted_hold":
                        cause = "plausible_true_candidate_but_predicted_hold"
                    elif center is None:
                        cause = "plausible_true_candidate_but_missing"
                    else:
                        cause = "score_loss_plausible_true_candidate_not_selected"
                    opportunity_causes[cause] += 1

            if source not in {"missing", "predicted_hold"} and center is not None and not correct:
                wrong_direct += 1
                if not true_items:
                    wrong_direct_causes["detector_ceiling_no_true_candidate"] += 1
                elif true_state == "no_temporal_score":
                    wrong_direct_causes["initialization_or_reset_no_temporal_score"] += 1
                elif true_state == "true_candidate_outside_gate":
                    wrong_direct_causes["gate_loss_true_candidate_outside_gate"] += 1
                else:
                    wrong_direct_causes["score_loss_plausible_true_candidate_not_selected"] += 1

                if d is not None and d > catastrophic_radius_px:
                    catastrophic_direct += 1
                    if not true_items:
                        catastrophic_causes["detector_ceiling_no_true_candidate"] += 1
                    elif true_state == "no_temporal_score":
                        catastrophic_causes["initialization_or_reset_no_temporal_score"] += 1
                    elif true_state == "true_candidate_outside_gate":
                        catastrophic_causes["gate_loss_true_candidate_outside_gate"] += 1
                    else:
                        catastrophic_causes["score_loss_plausible_true_candidate_not_selected"] += 1

        elif source not in {"missing", "predicted_hold"} and center is not None:
            not_visible_direct += 1
            if selected_details is None:
                not_visible_direct_causes["initialization_or_reset_no_temporal_score"] += 1
            elif isinstance(selected_details, dict) and bool(selected_details.get("plausible", False)):
                not_visible_direct_causes["continued_in_gate_false_track"] += 1
            else:
                not_visible_direct_causes["other_direct_false_state"] += 1

    return {
        "frames_path": str(frames_path),
        "gt_path": str(gt_path),
        "unmatched_gameplay_frames": unmatched,
        "visible_frames": visible,
        "association_opportunities": opportunities,
        "association_opportunity_misses": opportunity_misses,
        "association_opportunity_miss_rate_pct": round(100.0 * opportunity_misses / max(opportunities, 1), 2),
        "opportunity_miss_causes": dict(opportunity_causes),
        "wrong_direct_frames": wrong_direct,
        "wrong_direct_causes": dict(wrong_direct_causes),
        "catastrophic_direct_frames": catastrophic_direct,
        "catastrophic_direct_causes": dict(catastrophic_causes),
        "not_visible_direct_false_frames": not_visible_direct,
        "not_visible_direct_causes": dict(not_visible_direct_causes),
        "source_counts": dict(source_counts),
    }


def merge_counters(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    for result in results:
        total.update(result.get(key, {}))
    return dict(total)


def print_counter(title: str, values: dict[str, int]) -> None:
    total = sum(values.values())
    print(f"\n{title} (n={total})")
    print("-" * 86)
    for name, count in sorted(values.items(), key=lambda item: item[1], reverse=True):
        pct = 100.0 * count / max(total, 1)
        print(f"{name:58s} {count:6d}  {pct:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribute P0 association failures to initialization, gate, score, or detector ceiling using saved candidate traces."
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--catastrophic-radius-px", type=float, default=160.0)
    parser.add_argument("--timestamp-tolerance-sec", type=float, default=0.001)
    parser.add_argument("--output", default="evaluation/p0_association_failure_analysis")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    gt_root = Path(args.gt_root).resolve()
    results: list[dict[str, Any]] = []
    for run_id, gt_id in args.pair:
        result = evaluate(
            run_root / run_id / "frames.jsonl",
            gt_root / gt_id / "annotations.jsonl",
            radius_px=float(args.gt_radius_px),
            catastrophic_radius_px=float(args.catastrophic_radius_px),
            tolerance_sec=float(args.timestamp_tolerance_sec),
        )
        result["run_id"] = run_id
        result["gt_run_id"] = gt_id
        results.append(result)

    aggregate = {
        "association_opportunities": sum(r["association_opportunities"] for r in results),
        "association_opportunity_misses": sum(r["association_opportunity_misses"] for r in results),
        "opportunity_miss_causes": merge_counters(results, "opportunity_miss_causes"),
        "wrong_direct_frames": sum(r["wrong_direct_frames"] for r in results),
        "wrong_direct_causes": merge_counters(results, "wrong_direct_causes"),
        "catastrophic_direct_frames": sum(r["catastrophic_direct_frames"] for r in results),
        "catastrophic_direct_causes": merge_counters(results, "catastrophic_direct_causes"),
        "not_visible_direct_false_frames": sum(r["not_visible_direct_false_frames"] for r in results),
        "not_visible_direct_causes": merge_counters(results, "not_visible_direct_causes"),
    }
    aggregate["association_opportunity_miss_rate_pct"] = round(
        100.0 * aggregate["association_opportunity_misses"] / max(aggregate["association_opportunities"], 1), 2
    )

    payload = {"runs": results, "aggregate": aggregate}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("P0 ASSOCIATION FAILURE ATTRIBUTION")
    print("=" * 86)
    print(f"opportunities: {aggregate['association_opportunities']}")
    print(f"misses:        {aggregate['association_opportunity_misses']} ({aggregate['association_opportunity_miss_rate_pct']:.2f}%)")
    print_counter("Opportunity miss causes", aggregate["opportunity_miss_causes"])
    print_counter("Wrong direct output causes", aggregate["wrong_direct_causes"])
    print_counter("Catastrophic direct-frame causes", aggregate["catastrophic_direct_causes"])
    print_counter("Not-visible direct false-state causes", aggregate["not_visible_direct_causes"])
    print(f"\n[P0 ASSOCIATION ATTRIBUTION] {path}")


if __name__ == "__main__":
    main()
