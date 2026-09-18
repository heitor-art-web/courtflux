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

from evaluation.p2_ground_truth import _event_window, _interval_distance, load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_CLIPS = ("V1", "V2", "V3", "V4", "V5")
DEVELOPMENT_CLIPS = ("V1", "V3", "V4", "V5")
CHALLENGE_CLIPS = ("V2",)
DEFAULT_GT = PROJECT_ROOT / "evaluation" / "ground_truth" / "p2_gate_c_team_attribution.json"

TEAM_ATTRIBUTION_COVERAGE_MIN = 0.80
TEAM_ATTRIBUTION_ACCURACY_MIN = 0.90
CROSS_TEAM_CONTAMINATION_MAX = 0.10
PER_CLIP_ACCURACY_MIN = 0.70

UNKNOWN_TEAMS = {None, "", "unknown", "UNKNOWN", "unattributed", "UNATTRIBUTED"}
EVENT_TYPE_BY_STATE = {
    "pass_flight": "pass_candidate",
    "shot_flight": "shot_attempt",
}


def _run_path(run_id: str, filename: str) -> Path:
    return PROJECT_ROOT / "evaluation" / "runs" / run_id / filename


def _clean_team(raw: Any) -> str | None:
    if raw in UNKNOWN_TEAMS:
        return None
    value = str(raw).strip()
    if not value or value.lower() in {"unknown", "none", "null", "unattributed"}:
        return None
    return value


def _load_gt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gate C ground truth must be a JSON object")
    if str(payload.get("version") or "") != "courtflux_p2_gate_c_team_gt_v1":
        raise ValueError("Gate C ground truth version must be courtflux_p2_gate_c_team_gt_v1")
    return payload


def _sample_window(sample: dict[str, Any]) -> tuple[int, int]:
    start = int(sample.get("frame_start", sample.get("frame_end", 0)))
    end = int(sample.get("frame_end", start))
    return (end, start) if end < start else (start, end)


def _modal_team(teams: list[str]) -> tuple[str | None, Counter[str]]:
    counts: Counter[str] = Counter(teams)
    if not counts:
        return None, counts
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered[0][0], counts


def _evaluate_possessed_sample(
    sample: dict[str, Any],
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    start, end = _sample_window(sample)
    truth_team = str(sample["team_id"])
    evidence = []
    for row in states:
        frame_id = int(row.get("frame_id", -1))
        if frame_id < start or frame_id > end:
            continue
        team = _clean_team(row.get("current_team_id"))
        if team is not None:
            evidence.append(team)

    predicted_team, counts = _modal_team(evidence)
    attributed = predicted_team is not None
    correct = bool(attributed and predicted_team == truth_team)
    contamination = (
        None
        if not evidence
        else sum(1 for team in evidence if team != truth_team) / len(evidence)
    )
    return {
        "truth_team_id": truth_team,
        "predicted_team_id": predicted_team,
        "attributed": attributed,
        "correct": correct,
        "evidence_points": len(evidence),
        "team_votes": dict(sorted(counts.items())),
        "contamination_rate": None if contamination is None else round(contamination, 4),
        "matched_event_index": None,
        "event_distance_frames": None,
    }


def _event_candidates(
    sample: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    tolerance_frames: int,
    used_event_indices: set[int],
) -> list[tuple[int, int]]:
    start, end = _sample_window(sample)
    event_type = EVENT_TYPE_BY_STATE[str(sample["state"])]
    candidates: list[tuple[int, int]] = []
    for index, row in enumerate(events):
        if index in used_event_indices:
            continue
        if str(row.get("event_type") or "") != event_type:
            continue
        pred_start, pred_end = _event_window(row)
        distance = _interval_distance(start, end, pred_start, pred_end)
        if distance <= tolerance_frames:
            candidates.append((distance, index))
    return sorted(candidates)


def _evaluate_event_sample(
    sample: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    tolerance_frames: int,
    used_event_indices: set[int],
) -> dict[str, Any]:
    truth_team = str(sample["team_id"])
    candidates = _event_candidates(
        sample,
        events,
        tolerance_frames=tolerance_frames,
        used_event_indices=used_event_indices,
    )
    if not candidates:
        return {
            "truth_team_id": truth_team,
            "predicted_team_id": None,
            "attributed": False,
            "correct": False,
            "evidence_points": 0,
            "team_votes": {},
            "contamination_rate": None,
            "matched_event_index": None,
            "event_distance_frames": None,
        }

    distance, index = candidates[0]
    used_event_indices.add(index)
    team = _clean_team(events[index].get("team_id"))
    attributed = team is not None
    correct = bool(attributed and team == truth_team)
    return {
        "truth_team_id": truth_team,
        "predicted_team_id": team,
        "attributed": attributed,
        "correct": correct,
        "evidence_points": 1 if attributed else 0,
        "team_votes": {} if team is None else {team: 1},
        "contamination_rate": None if team is None else (0.0 if correct else 1.0),
        "matched_event_index": index,
        "event_distance_frames": distance,
    }


def evaluate_clip(
    *,
    clip: str,
    run_id: str,
    clip_gt: dict[str, Any],
    tolerance_frames: int,
) -> dict[str, Any]:
    states_path = _run_path(run_id, "p2_states.jsonl")
    events_path = _run_path(run_id, "p2_events.jsonl")
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    states = load_jsonl(states_path)
    events = load_jsonl(events_path)
    samples = [row for row in list(clip_gt.get("samples", []) or []) if isinstance(row, dict)]
    used_event_indices: set[int] = set()
    rows: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples):
        state = str(sample.get("state") or "")
        if state == "possessed":
            result = _evaluate_possessed_sample(sample, states)
        elif state in EVENT_TYPE_BY_STATE:
            result = _evaluate_event_sample(
                sample,
                events,
                tolerance_frames=tolerance_frames,
                used_event_indices=used_event_indices,
            )
        else:
            raise ValueError(f"unsupported Gate C sample state: {state!r}")

        start, end = _sample_window(sample)
        rows.append(
            {
                "sample_index": sample_index,
                "state": state,
                "frame_start": start,
                "frame_end": end,
                "human_note": sample.get("human_note"),
                **result,
            }
        )

    total = len(rows)
    attributed = sum(1 for row in rows if row["attributed"])
    correct = sum(1 for row in rows if row["correct"])
    contamination_values = [
        float(row["contamination_rate"])
        for row in rows
        if row.get("contamination_rate") is not None
    ]
    coverage = attributed / max(total, 1)
    accuracy = correct / max(attributed, 1)
    contamination = (
        None
        if not contamination_values
        else sum(contamination_values) / len(contamination_values)
    )
    return {
        "clip": clip,
        "run_id": run_id,
        "match_config": clip_gt.get("match_config"),
        "teams": list(clip_gt.get("teams", []) or []),
        "samples": total,
        "attributed_samples": attributed,
        "correct_samples": correct,
        "coverage": round(coverage, 4),
        "accuracy": round(accuracy, 4),
        "cross_team_contamination": None if contamination is None else round(contamination, 4),
        "sample_results": rows,
    }


def _aggregate(clips: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(int(row["samples"]) for row in clips)
    attributed = sum(int(row["attributed_samples"]) for row in clips)
    correct = sum(int(row["correct_samples"]) for row in clips)

    contamination_values: list[float] = []
    for clip in clips:
        for sample in clip.get("sample_results", []):
            value = sample.get("contamination_rate")
            if value is not None:
                contamination_values.append(float(value))

    coverage = attributed / max(samples, 1)
    accuracy = correct / max(attributed, 1)
    contamination = (
        None
        if not contamination_values
        else sum(contamination_values) / len(contamination_values)
    )
    return {
        "samples": samples,
        "attributed_samples": attributed,
        "correct_samples": correct,
        "coverage": round(coverage, 4),
        "accuracy": round(accuracy, 4),
        "cross_team_contamination": None if contamination is None else round(contamination, 4),
    }


def _gate(aggregate: dict[str, Any], clips: list[dict[str, Any]]) -> dict[str, Any]:
    contamination = aggregate["cross_team_contamination"]
    checks = [
        {
            "name": "team_attribution_coverage",
            "actual": aggregate["coverage"],
            "expected": f">= {TEAM_ATTRIBUTION_COVERAGE_MIN}",
            "passed": aggregate["coverage"] >= TEAM_ATTRIBUTION_COVERAGE_MIN,
        },
        {
            "name": "team_attribution_accuracy",
            "actual": aggregate["accuracy"],
            "expected": f">= {TEAM_ATTRIBUTION_ACCURACY_MIN}",
            "passed": aggregate["accuracy"] >= TEAM_ATTRIBUTION_ACCURACY_MIN,
        },
        {
            "name": "cross_team_contamination",
            "actual": contamination,
            "expected": f"<= {CROSS_TEAM_CONTAMINATION_MAX}",
            "passed": contamination is not None and contamination <= CROSS_TEAM_CONTAMINATION_MAX,
        },
    ]
    for clip in clips:
        checks.append(
            {
                "name": f"{clip['clip']}_team_attribution_accuracy",
                "actual": clip["accuracy"],
                "expected": f">= {PER_CLIP_ACCURACY_MIN}",
                "passed": clip["attributed_samples"] > 0 and clip["accuracy"] >= PER_CLIP_ACCURACY_MIN,
            }
        )
    return {
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks": checks,
        "required_checks": len(checks),
        "passed_checks": sum(1 for row in checks if row["passed"]),
        "failed_checks": sum(1 for row in checks if not row["passed"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P2 Gate C team attribution.")
    for clip in ALL_CLIPS:
        parser.add_argument(f"--{clip.lower()}-run", required=True)
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT))
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output", default="evaluation/gate_c_scoreboard.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path
    gt = _load_gt(gt_path)
    clip_gt = dict(gt.get("clips", {}) or {})
    run_map = {clip: str(getattr(args, f"{clip.lower()}_run")) for clip in ALL_CLIPS}

    clip_results = [
        evaluate_clip(
            clip=clip,
            run_id=run_map[clip],
            clip_gt=dict(clip_gt[clip]),
            tolerance_frames=max(0, int(args.tolerance_frames)),
        )
        for clip in ALL_CLIPS
    ]
    development = [row for row in clip_results if row["clip"] in DEVELOPMENT_CLIPS]
    challenge = [row for row in clip_results if row["clip"] in CHALLENGE_CLIPS]

    aggregate = _aggregate(development)
    gate = _gate(aggregate, development)
    challenge_aggregate = _aggregate(challenge)
    challenge_gate = _gate(challenge_aggregate, challenge)

    result = {
        "version": "courtflux_p2_gate_c_team_attribution_v1",
        "contract": (
            "Gate C uses prediction-blind human visual-team colour windows from the separate "
            "Gate C sidecar; V1/V3/V4/V5 block development, V2 is challenge-only; one human "
            "window is one scoring unit; unknown/unattributed is separate from wrong-team "
            "attribution"
        ),
        "thresholds": {
            "coverage_min": TEAM_ATTRIBUTION_COVERAGE_MIN,
            "accuracy_min": TEAM_ATTRIBUTION_ACCURACY_MIN,
            "cross_team_contamination_max": CROSS_TEAM_CONTAMINATION_MAX,
            "per_clip_accuracy_min": PER_CLIP_ACCURACY_MIN,
        },
        "tolerance_frames": max(0, int(args.tolerance_frames)),
        "runs": run_map,
        "development_clips": list(DEVELOPMENT_CLIPS),
        "gate": gate,
        "aggregate": aggregate,
        "challenge": {
            "clips": list(CHALLENGE_CLIPS),
            "blocking": False,
            "diagnostic_gate": challenge_gate,
            "aggregate": challenge_aggregate,
        },
        "clips": clip_results,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
