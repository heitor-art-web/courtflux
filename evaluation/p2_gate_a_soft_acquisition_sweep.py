from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_ground_truth import load_ground_truth, load_jsonl

STREAK_OPTIONS = (2, 3, 4, 5)
MEAN_SCORE_OPTIONS = (0.20, 0.30, 0.35, 0.40, 0.45)


def _truth_state_by_frame(gt: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in list(gt.get("control_intervals", []) or []):
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or "unknown")
        start = int(row.get("frame_start", row.get("frame", 0)))
        end = int(row.get("frame_end", start))
        if end < start:
            start, end = end, start
        for frame_id in range(start, end + 1):
            result[frame_id] = state
    return result


def _eligible(row: dict[str, Any]) -> bool:
    predicted = str(row.get("live_control_state", row.get("control_state")) or "unknown")
    return bool(
        predicted == "unknown"
        and row.get("engine_eligible", False)
        and row.get("current_track_id") is None
        and row.get("candidate_track_id") is None
        and not bool(row.get("flight_active", False))
        and not bool(row.get("shot_active", False))
        and row.get("raw_best_track_id") is not None
    )


def analyze(run_id: str, ground_truth: Path) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not ground_truth.exists():
        raise FileNotFoundError(ground_truth)

    states = sorted(load_jsonl(states_path), key=lambda row: int(row.get("frame_id", -1)))
    truth = _truth_state_by_frame(load_ground_truth(ground_truth))

    policies: dict[tuple[int, float], dict[str, Any]] = {
        (streak, mean_score): {
            "trigger_truth": Counter(),
            "triggers": [],
        }
        for streak in STREAK_OPTIONS
        for mean_score in MEAN_SCORE_OPTIONS
    }
    triggered_in_episode = {key: False for key in policies}

    previous_frame: int | None = None
    active_track: int | None = None
    streak = 0
    score_sum = 0.0
    score_peak = 0.0

    for row in states:
        frame_id = int(row.get("frame_id", -1))
        if frame_id < 0:
            continue
        contiguous = previous_frame is not None and frame_id == previous_frame + 1
        previous_frame = frame_id

        if not _eligible(row):
            active_track = None
            streak = 0
            score_sum = 0.0
            score_peak = 0.0
            for key in triggered_in_episode:
                triggered_in_episode[key] = False
            continue

        track_id = int(row["raw_best_track_id"])
        score = float(row.get("raw_best_score", 0.0) or 0.0)
        if not contiguous or active_track != track_id:
            active_track = track_id
            streak = 1
            score_sum = score
            score_peak = score
            for key in triggered_in_episode:
                triggered_in_episode[key] = False
        else:
            streak += 1
            score_sum += score
            score_peak = max(score_peak, score)

        mean_score = score_sum / max(streak, 1)
        truth_state = truth.get(frame_id, "unlabeled")

        for key, payload in policies.items():
            min_streak, min_mean = key
            if triggered_in_episode[key]:
                continue
            if streak < min_streak or mean_score < min_mean:
                continue
            triggered_in_episode[key] = True
            payload["trigger_truth"][truth_state] += 1
            payload["triggers"].append(
                {
                    "frame_id": frame_id,
                    "truth_state": truth_state,
                    "track_id": track_id,
                    "streak_frames": streak,
                    "mean_score": round(mean_score, 4),
                    "peak_score": round(score_peak, 4),
                }
            )

    rendered = []
    for (min_streak, min_mean), payload in sorted(policies.items()):
        counts: Counter[str] = payload["trigger_truth"]
        labeled_core = sum(counts.get(state, 0) for state in ("possessed", "pass_flight", "shot_flight", "loose", "unknown"))
        possessed = counts.get("possessed", 0)
        rendered.append(
            {
                "min_streak_frames": min_streak,
                "min_mean_score": min_mean,
                "trigger_count": sum(counts.values()),
                "trigger_truth": dict(counts),
                "possessed_share_of_labeled_triggers": (
                    None if labeled_core == 0 else round(possessed / labeled_core, 4)
                ),
                "triggers": payload["triggers"],
            }
        )

    return {
        "version": "courtflux_p2_gate_a_soft_acquisition_sweep_v1",
        "contract": (
            "offline trigger study only; does not simulate post-acquisition runtime state; "
            "scope is UNKNOWN + engine eligible + ownerless + no thresholded candidate + no flight + no shot"
        ),
        "run_id": run_id,
        "ground_truth": str(ground_truth.relative_to(PROJECT_ROOT)),
        "policies": rendered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep conservative soft-acquisition trigger policies on Gate A V1.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--ground-truth", default="evaluation/ground_truth/v1.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    result = analyze(str(args.run), gt_path)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
