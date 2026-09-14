from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MIN_REVIEWED_TRACKS = 40
DEFAULT_PRECISION_MIN = 0.90
DEFAULT_UNREADABLE_FALSE_ACCEPT_MAX = 0.05
DEFAULT_CLOSE_CALL_PP = 10.0
DEFAULT_EXTENDED_GT_TARGET = 100


@dataclass(frozen=True)
class Candidate:
    name: str
    path: Path
    metrics: dict[str, Any]


def _load_candidate(spec: str) -> Candidate:
    if "=" not in spec:
        raise ValueError("--candidate must use NAME=SUMMARY_JSON")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path).expanduser().resolve()
    if not name:
        raise ValueError("candidate name cannot be empty")
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    metrics = summary.get("evaluation", {}).get("all")
    if not isinstance(metrics, dict):
        metrics = summary.get("gate_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing evaluation.all / gate_metrics")
    return Candidate(name=name, path=path, metrics=metrics)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_candidate(
    candidate: Candidate,
    *,
    min_reviewed_tracks: int,
    precision_min: float,
    unreadable_false_accept_max: float,
) -> dict[str, Any]:
    m = candidate.metrics
    numeric = int(m.get("numeric_tracks") or 0)
    unreadable = int(m.get("unreadable_tracks") or 0)
    reviewed = numeric + unreadable
    precision = _number(m.get("decision_precision"))
    false_accept = _number(m.get("unreadable_false_accept_rate"))
    recall = _number(m.get("correct_recall"))

    reasons: list[str] = []
    if reviewed < min_reviewed_tracks:
        reasons.append(f"reviewed_tracks<{min_reviewed_tracks}")
    if precision is None:
        reasons.append("decision_precision_missing")
    elif precision < precision_min:
        reasons.append(f"decision_precision<{precision_min:.3f}")
    if false_accept is None:
        reasons.append("unreadable_false_accept_rate_missing")
    elif false_accept > unreadable_false_accept_max:
        reasons.append(f"unreadable_false_accept_rate>{unreadable_false_accept_max:.3f}")
    if recall is None:
        reasons.append("correct_recall_missing")

    return {
        "name": candidate.name,
        "summary": str(candidate.path),
        "reviewed_tracks": reviewed,
        "numeric_tracks": numeric,
        "unreadable_tracks": unreadable,
        "decision_precision": precision,
        "unreadable_false_accept_rate": false_accept,
        "correct_recall": recall,
        "passes": not reasons,
        "reasons": reasons,
    }


def decide(
    results: list[dict[str, Any]],
    *,
    close_call_pp: float,
    extended_gt_target: int,
) -> dict[str, Any]:
    passing = [row for row in results if row["passes"]]
    if not passing:
        return {
            "status": "NO_PROMOTION",
            "winner": None,
            "reason": "No candidate passes the P1 Jersey correctness gate.",
        }

    passing.sort(key=lambda row: float(row["correct_recall"]), reverse=True)
    winner = passing[0]
    if len(passing) >= 2:
        delta_pp = 100.0 * (
            float(passing[0]["correct_recall"]) - float(passing[1]["correct_recall"])
        )
        reviewed = min(int(passing[0]["reviewed_tracks"]), int(passing[1]["reviewed_tracks"]))
        if delta_pp <= close_call_pp and reviewed < extended_gt_target:
            return {
                "status": "EXTEND_GT",
                "winner": None,
                "provisional_leader": winner["name"],
                "correct_recall_delta_pp": round(delta_pp, 3),
                "target_reviewed_tracks": extended_gt_target,
                "reason": (
                    "Multiple candidates pass but correct-recall separation is within the close-call "
                    "band; extend human GT before promotion."
                ),
            }

    return {
        "status": "PROMOTE",
        "winner": winner["name"],
        "reason": "Candidate passes precision/unreadable gates and has the highest correct recall.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CourtFlux P1 Jersey human-GT exit gate")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate summary as NAME=path/to/summary.json; repeat for each recognizer",
    )
    parser.add_argument("--min-reviewed-tracks", type=int, default=DEFAULT_MIN_REVIEWED_TRACKS)
    parser.add_argument("--precision-min", type=float, default=DEFAULT_PRECISION_MIN)
    parser.add_argument(
        "--unreadable-false-accept-max",
        type=float,
        default=DEFAULT_UNREADABLE_FALSE_ACCEPT_MAX,
    )
    parser.add_argument("--close-call-pp", type=float, default=DEFAULT_CLOSE_CALL_PP)
    parser.add_argument("--extended-gt-target", type=int, default=DEFAULT_EXTENDED_GT_TARGET)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    candidates = [_load_candidate(spec) for spec in args.candidate]
    results = [
        evaluate_candidate(
            candidate,
            min_reviewed_tracks=max(1, args.min_reviewed_tracks),
            precision_min=args.precision_min,
            unreadable_false_accept_max=args.unreadable_false_accept_max,
        )
        for candidate in candidates
    ]
    decision = decide(
        results,
        close_call_pp=max(0.0, args.close_call_pp),
        extended_gt_target=max(1, args.extended_gt_target),
    )
    report = {
        "contract": {
            "min_reviewed_tracks": max(1, args.min_reviewed_tracks),
            "decision_precision_min": args.precision_min,
            "unreadable_false_accept_rate_max": args.unreadable_false_accept_max,
            "winner_metric": "correct_recall",
            "close_call_pp": max(0.0, args.close_call_pp),
            "extended_gt_target": max(1, args.extended_gt_target),
        },
        "candidates": results,
        "decision": decision,
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if decision["status"] == "PROMOTE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
