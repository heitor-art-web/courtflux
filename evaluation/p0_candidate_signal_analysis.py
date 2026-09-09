from __future__ import annotations

import argparse
import bisect
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import (
    align_production_truth,
    candidate_center,
    center_distance,
    load_jsonl,
)

SIGNALS = (
    "detector_confidence",
    "association_confidence",
    "verification_score",
    "rank_score",
    "temporal_score",
    "geometry_score",
)


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(float(q), 1.0))
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    alpha = position - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p10": None, "median": None, "p90": None, "mean": None}
    return {
        "n": len(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "mean": statistics.fmean(values),
    }


def auc_from_scores(positive: list[float], negative: list[float]) -> float | None:
    """Mann-Whitney ROC AUC with average ranks for ties; higher score means positive."""
    if not positive or not negative:
        return None
    labelled = [(float(value), 1) for value in positive] + [
        (float(value), 0) for value in negative
    ]
    labelled.sort(key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(labelled):
        end = index + 1
        while end < len(labelled) and labelled[end][0] == labelled[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positives_in_tie = sum(label for _, label in labelled[index:end])
        rank_sum_positive += average_rank * positives_in_tie
        index = end
    n_pos = len(positive)
    n_neg = len(negative)
    u = rank_sum_positive - n_pos * (n_pos + 1) / 2.0
    return u / float(n_pos * n_neg)


def numeric(trace: dict[str, Any], key: str) -> float | None:
    value = trace.get(key)
    if value is None and key == "detector_confidence":
        value = trace.get("confidence")
    if value is None and key == "association_confidence":
        value = trace.get("rank_score", trace.get("confidence"))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def candidate_is_true(trace: dict[str, Any], truth: dict[str, Any], radius_px: float) -> bool:
    if not bool(truth.get("visible")):
        return False
    point = truth.get("point")
    center = candidate_center(trace)
    if not isinstance(point, list) or len(point) != 2 or center is None:
        return False
    return center_distance(center, point) <= float(radius_px)


def selected_trace(row: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    association = row.get("ball_association") or {}
    selected_index = association.get("selected_index")
    try:
        selected_index = int(selected_index)
    except (TypeError, ValueError):
        return None
    if 0 <= selected_index < len(traces):
        return traces[selected_index]
    return None


def analyze_pair(
    frames_path: Path,
    gt_path: Path,
    *,
    radius_px: float,
    tolerance_sec: float,
) -> dict[str, Any]:
    production_rows = load_jsonl(frames_path)
    truth_rows = load_jsonl(gt_path)
    aligned, unmatched = align_production_truth(
        production_rows,
        truth_rows,
        tolerance_sec=float(tolerance_sec),
    )

    signal_values: dict[str, dict[str, list[float]]] = {
        signal: {"true": [], "false": []} for signal in SIGNALS
    }
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    temporal_plausible = Counter()
    verified = Counter()
    selected_profiles: dict[str, dict[str, list[float]]] = {
        group: {signal: [] for signal in SIGNALS}
        for group in ("correct_visible", "wrong_visible", "not_visible")
    }
    selected_counts = Counter()

    candidate_total = 0
    true_total = 0
    false_total = 0

    for row, truth, _delta in aligned:
        association = row.get("ball_association") or {}
        traces = [
            trace
            for trace in (association.get("candidates") or [])
            if isinstance(trace, dict)
        ]
        labels: list[bool] = []
        for trace in traces:
            is_true = candidate_is_true(trace, truth, radius_px)
            labels.append(is_true)
            label = "true" if is_true else "false"
            candidate_total += 1
            true_total += int(is_true)
            false_total += int(not is_true)

            source = str(trace.get("source") or "unknown")
            source_counts[source][label] += 1
            if trace.get("temporal_plausible") is not None:
                temporal_plausible[f"{label}:{bool(trace.get('temporal_plausible'))}"] += 1
            if trace.get("verified") is not None:
                verified[f"{label}:{bool(trace.get('verified'))}"] += 1

            for signal in SIGNALS:
                value = numeric(trace, signal)
                if value is not None:
                    signal_values[signal][label].append(value)

        selected = selected_trace(row, traces)
        if selected is None:
            continue
        visible = bool(truth.get("visible"))
        selected_true = candidate_is_true(selected, truth, radius_px)
        if visible and selected_true:
            group = "correct_visible"
        elif visible:
            group = "wrong_visible"
        else:
            group = "not_visible"
        selected_counts[group] += 1
        for signal in SIGNALS:
            value = numeric(selected, signal)
            if value is not None:
                selected_profiles[group][signal].append(value)

    signal_summary: dict[str, Any] = {}
    for signal in SIGNALS:
        positives = signal_values[signal]["true"]
        negatives = signal_values[signal]["false"]
        true_dist = distribution(positives)
        false_dist = distribution(negatives)
        auc = auc_from_scores(positives, negatives)
        true_median = true_dist["median"]
        false_median = false_dist["median"]
        signal_summary[signal] = {
            "true": true_dist,
            "false": false_dist,
            "auc_higher_is_true": auc,
            "median_delta_true_minus_false": (
                None
                if true_median is None or false_median is None
                else float(true_median) - float(false_median)
            ),
        }

    sources = {}
    for source, counts in sorted(source_counts.items()):
        total = counts["true"] + counts["false"]
        sources[source] = {
            "total": total,
            "true": counts["true"],
            "false": counts["false"],
            "candidate_precision_pct": 100.0 * counts["true"] / max(total, 1),
        }

    selected_summary: dict[str, Any] = {}
    for group, values_by_signal in selected_profiles.items():
        selected_summary[group] = {
            "count": selected_counts[group],
            "signals": {
                signal: distribution(values_by_signal[signal]) for signal in SIGNALS
            },
        }

    return {
        "frames_path": str(frames_path),
        "gt_path": str(gt_path),
        "aligned_frames": len(aligned),
        "unmatched_gameplay_frames": unmatched,
        "candidate_total": candidate_total,
        "true_candidates": true_total,
        "false_candidates": false_total,
        "signals": signal_summary,
        "sources": sources,
        "temporal_plausible": dict(temporal_plausible),
        "verified": dict(verified),
        "selected_profiles": selected_summary,
    }


def merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    # Recompute aggregate distributions from the saved per-run summaries is impossible
    # without raw values, so main() collects all pairs once more into one virtual stream
    # by concatenating source paths through analyze_pair's logic. This helper only emits
    # high-level totals for convenient reporting.
    return {
        "candidate_total": sum(int(result["candidate_total"]) for result in results),
        "true_candidates": sum(int(result["true_candidates"]) for result in results),
        "false_candidates": sum(int(result["false_candidates"]) for result in results),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how well existing Smart-v1 candidate signals separate true-ball "
            "candidates from false candidates on frozen CourtFlux GT. No inference and "
            "no threshold sweep are performed."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--timestamp-tolerance-sec", type=float, default=0.001)
    args = parser.parse_args()

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    results: list[dict[str, Any]] = []
    for run_id, gt_id in args.pair:
        result = analyze_pair(
            run_root / run_id / "frames.jsonl",
            gt_root / gt_id / "annotations.jsonl",
            radius_px=float(args.gt_radius_px),
            tolerance_sec=float(args.timestamp_tolerance_sec),
        )
        result["run_id"] = run_id
        result["gt_run_id"] = gt_id
        results.append(result)

    totals = merge_results(results)
    print("\nP0 CANDIDATE SIGNAL SEPARABILITY")
    print("=" * 104)
    print(
        f"candidates={totals['candidate_total']}  true={totals['true_candidates']}  "
        f"false={totals['false_candidates']}"
    )
    print("\nPer-run signal AUC (0.5=random, 1.0=perfect; higher signal assumed more ball-like)")
    print("-" * 104)
    header = f"{'run':28s}" + "".join(f" {signal[:11]:>11s}" for signal in SIGNALS)
    print(header)
    print("-" * 104)
    for result in results:
        values = []
        for signal in SIGNALS:
            values.append(fmt(result["signals"][signal]["auc_higher_is_true"], 3))
        print(f"{result['run_id']:28s}" + "".join(f" {value:>11s}" for value in values))

    print("\nMedian true - false by signal")
    print("-" * 104)
    for result in results:
        deltas = [
            fmt(result["signals"][signal]["median_delta_true_minus_false"], 3)
            for signal in SIGNALS
        ]
        print(f"{result['run_id']:28s}" + "".join(f" {value:>11s}" for value in deltas))

    print("\nCandidate precision by source")
    print("-" * 104)
    aggregate_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        for source, payload in result["sources"].items():
            aggregate_sources[source]["true"] += int(payload["true"])
            aggregate_sources[source]["false"] += int(payload["false"])
    for source, counts in sorted(
        aggregate_sources.items(), key=lambda item: item[1]["true"] + item[1]["false"], reverse=True
    ):
        total = counts["true"] + counts["false"]
        precision = 100.0 * counts["true"] / max(total, 1)
        print(
            f"{source:32s} total={total:6d} true={counts['true']:5d} "
            f"false={counts['false']:6d} precision={precision:6.2f}%"
        )

    print("\nSelected-candidate profiles (per run medians)")
    print("-" * 104)
    for result in results:
        print(f"\n{result['run_id']}")
        for group in ("correct_visible", "wrong_visible", "not_visible"):
            payload = result["selected_profiles"][group]
            medians = [
                fmt(payload["signals"][signal]["median"], 3) for signal in SIGNALS
            ]
            print(
                f"  {group:16s} n={payload['count']:4d} "
                + " ".join(
                    f"{signal[:6]}={value}" for signal, value in zip(SIGNALS, medians)
                )
            )

    print("\nINTERPRETATION")
    print("AUC materially above ~0.7 is useful evidence of candidate separability.")
    print("AUC near 0.5 means the current signal is not discriminative on its own.")
    print("AUC below 0.5 means higher values are, if anything, associated with false candidates.")
    print("This analysis does not tune thresholds and does not alter the production tracker.")


if __name__ == "__main__":
    main()
