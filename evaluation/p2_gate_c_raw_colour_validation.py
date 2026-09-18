from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from evaluation.p2_gate_c_scoreboard import DEFAULT_GT, _load_gt, _sample_window, _scoring_samples
from evaluation.p2_ground_truth import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CLIPS = ("V1", "V3", "V4", "V5")


def _run_path(run_id: str) -> Path:
    return PROJECT_ROOT / "evaluation" / "runs" / run_id / "p2_states.jsonl"


def _as_track_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prediction_for_track(row: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    for prediction in list(row.get("team_predictions") or []):
        if not isinstance(prediction, dict):
            continue
        if _as_track_id(prediction.get("track_id")) != track_id:
            continue
        if prediction.get("rescored") is not True:
            continue
        raw_scores = prediction.get("raw_scores")
        if isinstance(raw_scores, dict) and raw_scores:
            return prediction
    return None


def _best_raw_team(prediction: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    scores: list[tuple[str, float]] = []
    for key, raw in dict(prediction.get("raw_scores") or {}).items():
        try:
            scores.append((str(key), float(raw)))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None, None, None
    scores.sort(key=lambda item: (-item[1], item[0]))
    best_team, best_score = scores[0]
    second = scores[1][1] if len(scores) > 1 else 0.0
    return best_team, best_score, best_score - second


def _sample_raw_result(
    sample: dict[str, Any],
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    start, end = _sample_window(sample)
    truth = str(sample["team_id"])
    verified_holder = _as_track_id(sample.get("verified_holder_track_id"))
    observations: list[dict[str, Any]] = []

    for row in states:
        try:
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if frame_id < start or frame_id > end:
            continue

        if verified_holder is not None:
            track_id = verified_holder
            holder_source = "human_verified_holder_track"
        else:
            if str(row.get("control_state") or "") != "possessed":
                continue
            track_id = _as_track_id(row.get("current_track_id"))
            holder_source = "runtime_current_track"
        if track_id is None:
            continue

        prediction = _prediction_for_track(row, track_id)
        if prediction is None:
            continue
        best, score, margin = _best_raw_team(prediction)
        if best is None:
            continue
        observations.append(
            {
                "frame": frame_id,
                "track": track_id,
                "holder_source": holder_source,
                "best": best,
                "truth": truth,
                "correct": best == truth,
                "score": None if score is None else round(score, 4),
                "margin": None if margin is None else round(margin, 4),
                "raw": {
                    str(key): round(float(value), 6)
                    for key, value in dict(prediction.get("raw_scores") or {}).items()
                },
            }
        )

    votes = Counter(str(row["best"]) for row in observations)
    correct = sum(1 for row in observations if row["correct"])
    margins = [float(row["margin"]) for row in observations if row.get("margin") is not None]
    return {
        "window": [start, end],
        "truth": truth,
        "human_note": sample.get("human_note"),
        "source_kind": sample.get("source_kind", "sample"),
        "verified_holder_track_id": verified_holder,
        "raw_observations": len(observations),
        "raw_correct": correct,
        "raw_accuracy": None if not observations else round(correct / len(observations), 4),
        "raw_team_votes": dict(sorted(votes.items())),
        "mean_raw_margin": None if not margins else round(sum(margins) / len(margins), 4),
        "misclassified": [row for row in observations if not row["correct"]][:10],
    }


def evaluate_raw_colour(
    *,
    gt: dict[str, Any],
    run_ids: dict[str, str],
) -> dict[str, Any]:
    clips_gt = dict(gt.get("clips", {}) or {})
    clip_results: list[dict[str, Any]] = []
    global_correct = 0
    global_observations = 0

    for clip in DEVELOPMENT_CLIPS:
        clip_gt = dict(clips_gt[clip])
        possessed_samples = [
            row for row in _scoring_samples(clip_gt) if str(row.get("state") or "") == "possessed"
        ]
        samples_by_run: dict[str, list[dict[str, Any]]] = {}
        for sample in possessed_samples:
            samples_by_run.setdefault(str(sample["run_key"]), []).append(sample)

        run_results: list[dict[str, Any]] = []
        for run_key, samples in samples_by_run.items():
            run_id = str(run_ids.get(run_key) or "").strip()
            if not run_id:
                raise ValueError(f"missing run id for {run_key}")
            path = _run_path(run_id)
            if not path.exists():
                raise FileNotFoundError(path)
            states = load_jsonl(path)
            sample_results = [_sample_raw_result(sample, states) for sample in samples]
            run_correct = sum(int(row["raw_correct"]) for row in sample_results)
            run_observations = sum(int(row["raw_observations"]) for row in sample_results)
            global_correct += run_correct
            global_observations += run_observations
            run_results.append(
                {
                    "run_key": run_key,
                    "run_id": run_id,
                    "correct": run_correct,
                    "observations": run_observations,
                    "accuracy": None if not run_observations else round(run_correct / run_observations, 4),
                    "samples": sample_results,
                }
            )
        clip_results.append({"clip": clip, "runs": run_results})

    return {
        "version": "courtflux_p2_gate_c_raw_colour_validation_v1",
        "ground_truth_version": gt.get("version"),
        "contract": (
            "development raw-colour diagnostic only; uses Gate-C scoring possession intervals; "
            "clips with explicit human possession_intervals do not use historical broad possession windows; "
            "verified_holder_track_id overrides runtime holder only where human review explicitly verified it"
        ),
        "clips": clip_results,
        "global": {
            "correct": global_correct,
            "observations": global_observations,
            "accuracy": None if not global_observations else round(global_correct / global_observations, 4),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw Gate-C team colour on clean possession scoring intervals.")
    parser.add_argument("--v1-run", required=True)
    parser.add_argument("--v3-run", required=True)
    parser.add_argument("--v4a-run", required=True)
    parser.add_argument("--v4b-run", required=True)
    parser.add_argument("--v4c-run", required=True)
    parser.add_argument("--v5-run", required=True)
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT))
    parser.add_argument("--output", default="evaluation/gate_c_raw_colour_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    gt = _load_gt(gt_path)
    result = evaluate_raw_colour(
        gt=gt,
        run_ids={
            "V1": args.v1_run,
            "V3": args.v3_run,
            "V4_A": args.v4a_run,
            "V4_B": args.v4b_run,
            "V4_C": args.v4c_run,
            "V5": args.v5_run,
        },
    )

    print("===== RAW COLOUR CLASSIFIER VALIDATION / CLEAN POSSESSION GT =====")
    for clip in result["clips"]:
        for run in clip["runs"]:
            print(f"\n================ {run['run_key']} / {run['run_id']} ================")
            for sample in run["samples"]:
                printable = {key: value for key, value in sample.items() if key != "misclassified"}
                print(printable)
                if sample["misclassified"]:
                    print("MISCLASSIFIED (first 10):")
                    for row in sample["misclassified"]:
                        print(row)
    print("\n===== GLOBAL RAW COLOUR RESULT / CLEAN POSSESSION GT =====")
    print(result["global"])

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[RAW_COLOUR_RESULT] {output}")


if __name__ == "__main__":
    main()
