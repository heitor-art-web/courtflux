from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import evaluate_payload, load_ground_truth, load_jsonl

DEFAULT_RUNS = {
    "V1": "20260917_133606_837768",
    "V2": "20260917_141201_062495",
    "V3": "20260917_142607_777551",
    "V4": "20260917_143501_261725",
    "V5": "20260917_144307_273207",
}

COVERAGE_AGG_MIN = 0.98
COVERAGE_CLIP_MIN = 0.95
DOMINANT_AGG_MIN = 0.70
DOMINANT_CLIP_MIN = 0.50
MATCHING_AGG_MIN = 0.65
MATCHING_CLIP_MIN = 0.45
STATE_RECALL_MIN = {
    "possessed": 0.60,
    "pass_flight": 0.60,
    "shot_flight": 0.60,
}
LOOSE_RECALL_MIN = 0.50
PASS_AS_SHOT_MAX = 0.20
SHOT_AS_PASS_MAX = 0.20
CORE_UNKNOWN_MAX = 0.30


def _run_paths(run_id: str) -> tuple[Path, Path, Path]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    return run_dir, run_dir / "p2_events.jsonl", run_dir / "p2_states.jsonl"


def _clip_metrics(clip: str, run_id: str, tolerance_frames: int) -> dict[str, Any]:
    run_dir, events_path, states_path = _run_paths(run_id)
    gt_path = PROJECT_ROOT / "evaluation" / "ground_truth" / f"{clip.lower()}.json"
    missing = [str(p) for p in (events_path, states_path, gt_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{clip}: missing required files: {missing}")

    result = evaluate_payload(
        predicted_events=load_jsonl(events_path),
        predicted_states=load_jsonl(states_path),
        ground_truth=load_ground_truth(gt_path),
        tolerance_frames=tolerance_frames,
    )
    output_path = run_dir / "p2_gt_eval.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    control = result["control_states"]
    continuity = control["continuity"]
    return {
        "clip": clip,
        "run_id": run_id,
        "gt_file": str(gt_path.relative_to(PROJECT_ROOT)),
        "eval_file": str(output_path.relative_to(PROJECT_ROOT)),
        "ground_truth_labeled_frames": int(control["ground_truth_labeled_frames"]),
        "evaluated_frames": int(control["evaluated_frames"]),
        "missing_prediction_frames": int(control["missing_prediction_frames"]),
        "prediction_coverage": float(control["prediction_coverage"]),
        "frame_accuracy": float(control["accuracy"]),
        "dominant_state_interval_accuracy": float(continuity["dominant_state_interval_accuracy"]),
        "mean_interval_matching_rate": continuity["mean_interval_matching_rate"],
        "ground_truth_intervals": int(continuity["ground_truth_intervals"]),
        "evaluated_intervals": int(continuity["evaluated_intervals"]),
        "confusion": control["confusion"],
        "intervals": continuity["intervals"],
    }


def _sum_confusions(clips: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    agg: dict[str, Counter[str]] = defaultdict(Counter)
    for clip in clips:
        for truth_state, preds in dict(clip["confusion"]).items():
            for pred_state, count in dict(preds).items():
                agg[str(truth_state)][str(pred_state)] += int(count)
    return agg


def _safe_ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def _build_aggregate(clips: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = _sum_confusions(clips)
    gt_frames = sum(int(c["ground_truth_labeled_frames"]) for c in clips)
    eval_frames = sum(int(c["evaluated_frames"]) for c in clips)
    correct = sum(counter.get(state, 0) for state, counter in confusion.items())
    all_intervals = [row for c in clips for row in c["intervals"] if row.get("observed_frames", 0) > 0]
    dominant_correct = sum(1 for row in all_intervals if row.get("dominant_matches_truth") is True)
    matching_values = [float(row["matching_rate"]) for row in all_intervals if row.get("matching_rate") is not None]

    state_recall: dict[str, Any] = {}
    for truth_state, counter in sorted(confusion.items()):
        total = sum(counter.values())
        state_recall[truth_state] = {
            "frames": total,
            "correct": int(counter.get(truth_state, 0)),
            "recall": _safe_ratio(int(counter.get(truth_state, 0)), total),
        }

    pass_total = sum(confusion.get("pass_flight", Counter()).values())
    shot_total = sum(confusion.get("shot_flight", Counter()).values())
    core_states = ("possessed", "pass_flight", "shot_flight")
    core_total = sum(sum(confusion.get(state, Counter()).values()) for state in core_states)
    core_unknown = sum(confusion.get(state, Counter()).get("unknown", 0) for state in core_states)
    loose_intervals = sum(
        1 for c in clips for row in c["intervals"] if row.get("state") == "loose"
    )

    return {
        "ground_truth_labeled_frames": gt_frames,
        "evaluated_frames": eval_frames,
        "prediction_coverage": round(eval_frames / max(gt_frames, 1), 4),
        "frame_accuracy": round(correct / max(eval_frames, 1), 4),
        "evaluated_intervals": len(all_intervals),
        "dominant_state_interval_accuracy": round(dominant_correct / max(len(all_intervals), 1), 4),
        "mean_interval_matching_rate": (
            None if not matching_values else round(sum(matching_values) / len(matching_values), 4)
        ),
        "state_recall": state_recall,
        "critical_confusions": {
            "pass_flight_as_shot_flight": _safe_ratio(
                int(confusion.get("pass_flight", Counter()).get("shot_flight", 0)), pass_total
            ),
            "shot_flight_as_pass_flight": _safe_ratio(
                int(confusion.get("shot_flight", Counter()).get("pass_flight", 0)), shot_total
            ),
            "core_truth_predicted_unknown": _safe_ratio(core_unknown, core_total),
        },
        "loose_gt_intervals": loose_intervals,
        "confusion": {
            state: dict(sorted(counter.items())) for state, counter in sorted(confusion.items())
        },
    }


def _gate_checks(clips: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, expected: str, passed: bool | None) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "passed": passed})

    add("aggregate_prediction_coverage", aggregate["prediction_coverage"], f">= {COVERAGE_AGG_MIN}", aggregate["prediction_coverage"] >= COVERAGE_AGG_MIN)
    add("aggregate_dominant_interval_accuracy", aggregate["dominant_state_interval_accuracy"], f">= {DOMINANT_AGG_MIN}", aggregate["dominant_state_interval_accuracy"] >= DOMINANT_AGG_MIN)
    mean_match = aggregate["mean_interval_matching_rate"]
    add("aggregate_mean_interval_matching_rate", mean_match, f">= {MATCHING_AGG_MIN}", mean_match is not None and mean_match >= MATCHING_AGG_MIN)

    for clip in clips:
        add(f"{clip['clip']}_prediction_coverage", clip["prediction_coverage"], f">= {COVERAGE_CLIP_MIN}", clip["prediction_coverage"] >= COVERAGE_CLIP_MIN)
        add(f"{clip['clip']}_dominant_interval_accuracy", clip["dominant_state_interval_accuracy"], f">= {DOMINANT_CLIP_MIN}", clip["dominant_state_interval_accuracy"] >= DOMINANT_CLIP_MIN)
        clip_match = clip["mean_interval_matching_rate"]
        add(f"{clip['clip']}_mean_interval_matching_rate", clip_match, f">= {MATCHING_CLIP_MIN}", clip_match is not None and clip_match >= MATCHING_CLIP_MIN)

    for state, threshold in STATE_RECALL_MIN.items():
        row = aggregate["state_recall"].get(state)
        actual = None if row is None else row["recall"]
        add(f"aggregate_{state}_recall", actual, f">= {threshold}", actual is not None and actual >= threshold)

    loose_row = aggregate["state_recall"].get("loose")
    if aggregate["loose_gt_intervals"] >= 2:
        actual = None if loose_row is None else loose_row["recall"]
        add("aggregate_loose_recall", actual, f">= {LOOSE_RECALL_MIN}", actual is not None and actual >= LOOSE_RECALL_MIN)
    else:
        add("aggregate_loose_recall", None if loose_row is None else loose_row["recall"], "insufficient GT (<2 loose intervals)", None)

    critical = aggregate["critical_confusions"]
    for key, limit in (
        ("pass_flight_as_shot_flight", PASS_AS_SHOT_MAX),
        ("shot_flight_as_pass_flight", SHOT_AS_PASS_MAX),
        ("core_truth_predicted_unknown", CORE_UNKNOWN_MAX),
    ):
        actual = critical[key]
        add(key, actual, f"<= {limit}", actual is not None and actual <= limit)

    required = [row for row in checks if row["passed"] is not None]
    return {
        "status": "PASS" if required and all(bool(row["passed"]) for row in required) else "FAIL",
        "checks": checks,
        "required_checks": len(required),
        "passed_checks": sum(1 for row in required if row["passed"] is True),
        "failed_checks": sum(1 for row in required if row["passed"] is False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and aggregate P2 Gate A across V1-V5.")
    for clip, default in DEFAULT_RUNS.items():
        parser.add_argument(f"--{clip.lower()}-run", default=default)
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument(
        "--output",
        default="evaluation/gate_a_scoreboard.json",
        help="Aggregate JSON output path relative to repository root unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_map = {clip: str(getattr(args, f"{clip.lower()}_run")) for clip in DEFAULT_RUNS}
    clips = [_clip_metrics(clip, run_map[clip], args.tolerance_frames) for clip in DEFAULT_RUNS]
    aggregate = _build_aggregate(clips)
    gate = _gate_checks(clips, aggregate)
    result = {
        "version": "courtflux_p2_gate_a_scoreboard_v1",
        "contract": "live control_state; V1-V5 frozen human GT",
        "runs": run_map,
        "clips": clips,
        "aggregate": aggregate,
        "gate": gate,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "gate": gate,
        "aggregate": aggregate,
        "clips": [
            {
                "clip": c["clip"],
                "run_id": c["run_id"],
                "prediction_coverage": c["prediction_coverage"],
                "frame_accuracy": c["frame_accuracy"],
                "dominant_state_interval_accuracy": c["dominant_state_interval_accuracy"],
                "mean_interval_matching_rate": c["mean_interval_matching_rate"],
            }
            for c in clips
        ],
        "output": str(output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
