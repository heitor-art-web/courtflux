from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

UNKNOWN_TEAMS = {None, "", "unknown", "UNKNOWN", "unattributed", "UNATTRIBUTED"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_team(value: Any) -> str | None:
    if value in UNKNOWN_TEAMS:
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"unknown", "none", "null", "unattributed"} else text


def in_block(t: float, block: dict[str, Any], final_end: float) -> bool:
    start, end = float(block["start_sec"]), float(block["end_sec"])
    return t >= start and (t < end or (end == final_end and t <= end))


def block_rows(states: list[dict[str, Any]], block: dict[str, Any], final_end: float) -> list[dict[str, Any]]:
    return [row for row in states if in_block(float(row["timestamp_sec"]), block, final_end)]


def event_fully_in_block(event: dict[str, Any], block: dict[str, Any], final_end: float) -> bool:
    start = float(event.get("timestamp_start_sec", 0.0))
    end = float(event.get("timestamp_end_sec", start))
    b0, b1 = float(block["start_sec"]), float(block["end_sec"])
    return start >= b0 and (end < b1 or (b1 == final_end and end <= b1))


def binary_metrics(rows: list[tuple[bool, bool]]) -> dict[str, Any]:
    counts = Counter(rows)
    tp, fn = counts[(True, True)], counts[(True, False)]
    fp, tn = counts[(False, True)], counts[(False, False)]
    total = tp + fn + fp + tn
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": round((tp + tn) / max(total, 1), 4),
        "gameplay_recall": round(tp / max(tp + fn, 1), 4),
        "non_game_rejection": round(tn / max(tn + fp, 1), 4),
        "gameplay_precision": round(tp / max(tp + fp, 1), 4),
    }


def score_gate_a(gt: dict[str, Any], states: list[dict[str, Any]]) -> dict[str, Any]:
    final_end = float(gt["video"]["end_time_sec"])
    blocks = list(gt["scoring_blocks"])
    labeled = [block for block in blocks if block.get("gameplay") in {"gameplay", "non_game"}]
    frame_truth: list[tuple[bool, dict[str, Any]]] = []
    for block in labeled:
        truth = block["gameplay"] == "gameplay"
        frame_truth.extend((truth, row) for row in block_rows(states, block, final_end))

    gameplay_present = binary_metrics(
        [(truth, bool(row.get("gameplay_present"))) for truth, row in frame_truth]
    )
    metric_eligible = binary_metrics(
        [(truth, bool(row.get("metric_eligible"))) for truth, row in frame_truth]
    )

    gameplay_frames = [row for truth, row in frame_truth if truth]
    non_game_frames = [row for truth, row in frame_truth if not truth]
    gameplay_control = Counter(str(row.get("control_state") or "unknown") for row in gameplay_frames)
    non_game_control = Counter(str(row.get("control_state") or "unknown") for row in non_game_frames)

    semantic_specs = {
        "possessed": [
            block
            for block in blocks
            if block.get("gameplay") == "gameplay"
            and block.get("team_id") in {"team_a", "team_b", "team_b_then_team_a"}
        ],
        "pass_flight": [
            block
            for block in blocks
            if isinstance(block.get("human_constraints"), dict)
            and isinstance(block["human_constraints"].get("passes"), dict)
        ],
        "shot_flight": [
            block
            for block in blocks
            if block.get("scoring_status") != "gate_a_control_excluded"
            and isinstance(block.get("human_constraints"), dict)
            and isinstance(block["human_constraints"].get("shots"), dict)
            and int(block["human_constraints"]["shots"].get("exact", 0) or 0) > 0
        ],
    }
    semantic: dict[str, Any] = {}
    for state, expected_blocks in semantic_specs.items():
        rows = []
        for block in expected_blocks:
            observed = any(
                str(row.get("control_state") or "unknown") == state
                for row in block_rows(states, block, final_end)
            )
            rows.append({"block_id": block["id"], "observed": observed})
        observed_count = sum(1 for row in rows if row["observed"])
        semantic[state] = {
            "expected_blocks": len(rows),
            "observed_blocks": observed_count,
            "block_presence_recall": round(observed_count / max(len(rows), 1), 4),
            "rows": rows,
        }

    return {
        "scope": "V6 macro validation from human possession/non-game blocks; not the frame-exact Gate-A control-interval contract",
        "frame_exact_gate_a_contract_ready": False,
        "gameplay_present": gameplay_present,
        "metric_eligible": metric_eligible,
        "gameplay_control_state_distribution": {
            "frames": len(gameplay_frames),
            "counts": dict(gameplay_control),
            "unknown_rate": round(gameplay_control["unknown"] / max(len(gameplay_frames), 1), 4),
        },
        "non_game_control_state_distribution": {
            "frames": len(non_game_frames),
            "counts": dict(non_game_control),
            "shot_flight_rate": round(non_game_control["shot_flight"] / max(len(non_game_frames), 1), 4),
        },
        "semantic_block_presence": semantic,
    }


def match_shots(gt: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    truth = list(gt.get("shot_events", []))
    predicted = [
        dict(row, _index=index)
        for index, row in enumerate(events)
        if str(row.get("event_type") or "") == "shot_attempt"
    ]
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    grouped_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth:
        grouped_truth[str(row["block_id"])].append(row)

    for block_id, gt_rows in grouped_truth.items():
        gt_rows = sorted(
            gt_rows,
            key=lambda row: (int(row.get("ordered_index", 0)), float(row["window_start_sec"])),
        )
        window_start = min(float(row["window_start_sec"]) for row in gt_rows)
        window_end = max(float(row["window_end_sec"]) for row in gt_rows)
        candidates = [
            row
            for row in predicted
            if row["_index"] not in used
            and float(row["timestamp_start_sec"]) >= window_start
            and float(row["timestamp_start_sec"]) < window_end
        ]
        candidates.sort(key=lambda row: float(row["timestamp_start_sec"]))

        for truth_row in gt_rows:
            truth_start = float(truth_row["window_start_sec"])
            truth_end = float(truth_row["window_end_sec"])
            eligible = [
                row
                for row in candidates
                if row["_index"] not in used
                and truth_start <= float(row["timestamp_start_sec"]) < truth_end
            ]
            prediction = eligible[0] if eligible else None
            if prediction is not None:
                used.add(prediction["_index"])
            matches.append(
                {
                    "truth_id": truth_row["id"],
                    "block_id": block_id,
                    "truth_team_id": truth_row.get("team_id"),
                    "truth_outcome": truth_row.get("outcome"),
                    "matched": prediction is not None,
                    "prediction_index": None if prediction is None else prediction["_index"],
                    "predicted_start_sec": None if prediction is None else prediction.get("timestamp_start_sec"),
                    "predicted_end_sec": None if prediction is None else prediction.get("timestamp_end_sec"),
                    "predicted_team_id": None if prediction is None else clean_team(prediction.get("team_id")),
                    "predicted_outcome": None
                    if prediction is None
                    else dict(prediction.get("attributes", {}) or {}).get("outcome"),
                }
            )

    true_positives = sum(1 for row in matches if row["matched"])
    false_negatives = len(matches) - true_positives
    false_positive_rows = [row for row in predicted if row["_index"] not in used]
    false_positives = len(false_positive_rows)
    labeled_matches = [
        row for row in matches if row["matched"] and row.get("truth_outcome") is not None
    ]
    outcome_correct = sum(
        1 for row in labeled_matches if row["truth_outcome"] == row["predicted_outcome"]
    )

    duplicate_predictions = 0
    duplicate_rows = []
    for block_id, gt_rows in grouped_truth.items():
        window_start = min(float(row["window_start_sec"]) for row in gt_rows)
        window_end = max(float(row["window_end_sec"]) for row in gt_rows)
        predictions_in_window = [
            row
            for row in predicted
            if window_start <= float(row["timestamp_start_sec"]) < window_end
        ]
        extra = max(0, len(predictions_in_window) - len(gt_rows))
        if extra:
            duplicate_predictions += extra
            duplicate_rows.append(
                {
                    "block_id": block_id,
                    "truth_shots": len(gt_rows),
                    "predicted_shots": len(predictions_in_window),
                    "duplicates": extra,
                }
            )

    return {
        "ground_truth_shots": len(truth),
        "predicted_shots": len(predicted),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": round(true_positives / max(true_positives + false_positives, 1), 4),
        "recall": round(true_positives / max(true_positives + false_negatives, 1), 4),
        "f1": round((2 * true_positives) / max(2 * true_positives + false_positives + false_negatives, 1), 4),
        "outcome_labeled_matches": len(labeled_matches),
        "outcome_correct": outcome_correct,
        "outcome_accuracy": None
        if not labeled_matches
        else round(outcome_correct / len(labeled_matches), 4),
        "duplicate_predictions": duplicate_predictions,
        "duplicates": duplicate_rows,
        "matches": matches,
        "false_positive_predictions": [
            {
                "prediction_index": row["_index"],
                "start_sec": row.get("timestamp_start_sec"),
                "end_sec": row.get("timestamp_end_sec"),
                "outcome": dict(row.get("attributes", {}) or {}).get("outcome"),
            }
            for row in false_positive_rows
        ],
    }


def score_transition_constraints(gt: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    final_end = float(gt["video"]["end_time_sec"])
    blocks = {block["id"]: block for block in gt["scoring_blocks"]}

    def count(block: dict[str, Any], event_type: str) -> int:
        return sum(
            1
            for event in events
            if str(event.get("event_type") or "") == event_type
            and event_fully_in_block(event, block, final_end)
        )

    pass_rows = []
    turnover_rows = []
    for block in blocks.values():
        constraints = block.get("human_constraints") or {}
        if isinstance(constraints.get("passes"), dict):
            constraint = constraints["passes"]
            system_count = count(block, "pass_candidate")
            minimum = int(constraint.get("exact", constraint.get("min", 0)))
            pass_rows.append(
                {
                    "block_id": block["id"],
                    "human_exact": constraint.get("exact"),
                    "human_min": constraint.get("min"),
                    "system_candidates": system_count,
                    "constraint_satisfied": system_count >= minimum,
                }
            )
        if isinstance(constraints.get("turnovers"), dict):
            constraint = constraints["turnovers"]
            system_count = count(block, "turnover_candidate")
            minimum = int(constraint.get("exact", constraint.get("min", 0)))
            turnover_rows.append(
                {
                    "block_id": block["id"],
                    "human_exact": constraint.get("exact"),
                    "human_min": constraint.get("min"),
                    "system_candidates": system_count,
                    "constraint_satisfied": system_count >= minimum,
                }
            )

    non_game_blocks = [block for block in blocks.values() if block.get("gameplay") == "non_game"]
    pass_leakage = sum(count(block, "pass_candidate") for block in non_game_blocks)
    turnover_leakage = sum(count(block, "turnover_candidate") for block in non_game_blocks)

    rebound_gt = sum(
        int((block.get("human_constraints") or {}).get("rebounds", {}).get("exact", 0) or 0)
        for block in blocks.values()
    )
    rebound_predicted = sum(
        1 for event in events if str(event.get("event_type") or "") == "rebound"
    )

    return {
        "passes": {
            "explicit_positive_blocks": len(pass_rows),
            "satisfied_blocks": sum(1 for row in pass_rows if row["constraint_satisfied"]),
            "positive_block_recall": round(
                sum(1 for row in pass_rows if row["constraint_satisfied"]) / max(len(pass_rows), 1),
                4,
            ),
            "non_game_candidate_leakage": pass_leakage,
            "rows": pass_rows,
            "precision_recall_ready": False,
            "reason": "Human review gives exact pass count only for P1 and minimum presence elsewhere; exhaustive event-level pass GT is not available.",
        },
        "turnovers": {
            "explicit_positive_blocks": len(turnover_rows),
            "satisfied_blocks": sum(1 for row in turnover_rows if row["constraint_satisfied"]),
            "positive_block_recall": round(
                sum(1 for row in turnover_rows if row["constraint_satisfied"]) / max(len(turnover_rows), 1),
                4,
            ),
            "non_game_candidate_leakage": turnover_leakage,
            "rows": turnover_rows,
            "precision_recall_ready": False,
            "reason": "Human review explicitly labels two loss/turnover blocks but is not exhaustive for all negative gameplay blocks.",
        },
        "rebounds": {
            "ground_truth": rebound_gt,
            "predicted": rebound_predicted,
            "recall": None
            if rebound_gt == 0
            else round(min(rebound_predicted, rebound_gt) / rebound_gt, 4),
        },
    }


def score_gate_b(gt: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    shots = match_shots(gt, events)
    checks = [
        {"name": "shot_recall", "actual": shots["recall"], "expected": ">= 0.80", "passed": shots["recall"] >= 0.80},
        {"name": "shot_precision", "actual": shots["precision"], "expected": ">= 0.80", "passed": shots["precision"] >= 0.80},
        {
            "name": "shot_outcome_accuracy",
            "actual": shots["outcome_accuracy"],
            "expected": ">= 0.75",
            "passed": shots["outcome_accuracy"] is not None and shots["outcome_accuracy"] >= 0.75,
        },
        {
            "name": "same_gt_window_duplicates",
            "actual": shots["duplicate_predictions"],
            "expected": "<= 0",
            "passed": shots["duplicate_predictions"] <= 0,
        },
    ]
    return {
        "shots": shots,
        "shot_gate": {"status": "PASS" if all(check["passed"] for check in checks) else "FAIL", "checks": checks},
        "transition_constraints": score_transition_constraints(gt, events),
    }


def score_gate_c(
    gt: dict[str, Any],
    states: list[dict[str, Any]],
    events: list[dict[str, Any]],
    gate_b: dict[str, Any],
) -> dict[str, Any]:
    final_end = float(gt["video"]["end_time_sec"])
    blocks = {block["id"]: block for block in gt["scoring_blocks"]}
    sample_rows = []
    possessed_frame_total = 0
    possessed_frame_attributed = 0
    possessed_frame_correct = 0

    for block_id in gt["gate_c_possession_blocks"]:
        block = blocks[block_id]
        truth_team = str(block["team_id"])
        state_rows = block_rows(states, block, final_end)
        evidence = [clean_team(row.get("current_team_id")) for row in state_rows]
        evidence = [team for team in evidence if team is not None]
        votes = Counter(evidence)
        predicted_team = votes.most_common(1)[0][0] if votes else None
        contamination = None if not evidence else sum(1 for team in evidence if team != truth_team) / len(evidence)
        sample_rows.append(
            {
                "sample_kind": "possession_block",
                "sample_id": block_id,
                "truth_team_id": truth_team,
                "predicted_team_id": predicted_team,
                "attributed": predicted_team is not None,
                "correct": predicted_team == truth_team if predicted_team is not None else False,
                "contamination_rate": None if contamination is None else round(contamination, 4),
                "team_votes": dict(votes),
            }
        )

        possessed_rows = [row for row in state_rows if str(row.get("control_state") or "") == "possessed"]
        possessed_frame_total += len(possessed_rows)
        possessed_evidence = [clean_team(row.get("current_team_id")) for row in possessed_rows]
        possessed_evidence = [team for team in possessed_evidence if team is not None]
        possessed_frame_attributed += len(possessed_evidence)
        possessed_frame_correct += sum(1 for team in possessed_evidence if team == truth_team)

    matches_by_id = {row["truth_id"]: row for row in gate_b["shots"]["matches"]}
    for shot in gt["shot_events"]:
        truth_team = shot.get("team_id")
        if truth_team not in {"team_a", "team_b"}:
            continue
        match = matches_by_id[shot["id"]]
        predicted_team = match.get("predicted_team_id") if match.get("matched") else None
        sample_rows.append(
            {
                "sample_kind": "shot_attempt",
                "sample_id": shot["id"],
                "truth_team_id": truth_team,
                "predicted_team_id": predicted_team,
                "attributed": predicted_team is not None,
                "correct": predicted_team == truth_team if predicted_team is not None else False,
                "contamination_rate": None if predicted_team is None else (0.0 if predicted_team == truth_team else 1.0),
                "matched_shot": bool(match.get("matched")),
            }
        )

    total = len(sample_rows)
    attributed = sum(1 for row in sample_rows if row["attributed"])
    correct = sum(1 for row in sample_rows if row["correct"])
    contamination_values = [
        float(row["contamination_rate"])
        for row in sample_rows
        if row.get("contamination_rate") is not None
    ]
    coverage = attributed / max(total, 1)
    accuracy = correct / max(attributed, 1)
    contamination = None if not contamination_values else sum(contamination_values) / len(contamination_values)
    checks = [
        {"name": "team_attribution_coverage", "actual": round(coverage, 4), "expected": ">= 0.80", "passed": coverage >= 0.80},
        {"name": "team_attribution_accuracy", "actual": round(accuracy, 4), "expected": ">= 0.90", "passed": accuracy >= 0.90},
        {
            "name": "cross_team_contamination",
            "actual": None if contamination is None else round(contamination, 4),
            "expected": "<= 0.10",
            "passed": contamination is not None and contamination <= 0.10,
        },
        {"name": "V6_team_attribution_accuracy", "actual": round(accuracy, 4), "expected": ">= 0.70", "passed": attributed > 0 and accuracy >= 0.70},
    ]
    return {
        "samples": total,
        "attributed_samples": attributed,
        "correct_samples": correct,
        "coverage": round(coverage, 4),
        "accuracy": round(accuracy, 4),
        "cross_team_contamination": None if contamination is None else round(contamination, 4),
        "gate": {"status": "PASS" if all(check["passed"] for check in checks) else "FAIL", "checks": checks},
        "sample_results": sample_rows,
        "possessed_frame_diagnostics": {
            "frames": possessed_frame_total,
            "attributed_frames": possessed_frame_attributed,
            "coverage": round(possessed_frame_attributed / max(possessed_frame_total, 1), 4),
            "correct_attributed_frames": possessed_frame_correct,
            "accuracy": round(possessed_frame_correct / max(possessed_frame_attributed, 1), 4),
            "cross_team_contamination": round(1 - possessed_frame_correct / max(possessed_frame_attributed, 1), 4),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score V6 validation against human possession/non-game blocks without inventing unresolved boundaries."
    )
    parser.add_argument("--gt", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    states = load_jsonl(Path(args.states))
    events = load_jsonl(Path(args.events))
    gate_a = score_gate_a(gt, states)
    gate_b = score_gate_b(gt, events)
    gate_c = score_gate_c(gt, states, events, gate_b)
    result: dict[str, Any] = {
        "version": "courtflux_p2_v6_validation_score_v1",
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
    }
    if args.metrics:
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        result["runtime_diagnostics"] = {
            "event_team_attribution_rate": dict(metrics.get("events", {}) or {}).get("team_attribution_rate"),
            "actor_attribution_rate": dict(metrics.get("events", {}) or {}).get("actor_attribution_rate"),
            "passes": metrics.get("passes"),
            "turnovers": metrics.get("turnovers"),
            "rebounds": metrics.get("rebounds"),
            "shots": metrics.get("shots"),
        }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
