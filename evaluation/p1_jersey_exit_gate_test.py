from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.p1_jersey_exit_gate import Candidate, decide, evaluate_candidate


def _candidate(name: str, **metrics):
    return Candidate(name=name, path=Path(f"/{name}.json"), metrics=metrics)


def main() -> int:
    common = {
        "numeric_tracks": 35,
        "unreadable_tracks": 10,
        "decision_precision": 0.94,
        "unreadable_false_accept_rate": 0.03,
        "correct_recall": 0.52,
    }
    passing = evaluate_candidate(
        _candidate("pass", **common),
        min_reviewed_tracks=40,
        precision_min=0.90,
        unreadable_false_accept_max=0.05,
    )
    assert passing["passes"] is True

    failing = evaluate_candidate(
        _candidate("fail", **{**common, "decision_precision": 0.89}),
        min_reviewed_tracks=40,
        precision_min=0.90,
        unreadable_false_accept_max=0.05,
    )
    assert failing["passes"] is False
    assert any("decision_precision" in reason for reason in failing["reasons"])

    promoted = decide([passing], close_call_pp=10.0, extended_gt_target=100)
    assert promoted["status"] == "PROMOTE"
    assert promoted["winner"] == "pass"

    close_a = dict(passing, name="a", correct_recall=0.55)
    close_b = dict(passing, name="b", correct_recall=0.50)
    extended = decide([close_a, close_b], close_call_pp=10.0, extended_gt_target=100)
    assert extended["status"] == "EXTEND_GT"

    clear_a = dict(passing, name="a", correct_recall=0.70)
    clear_b = dict(passing, name="b", correct_recall=0.50)
    winner = decide([clear_a, clear_b], close_call_pp=10.0, extended_gt_target=100)
    assert winner["status"] == "PROMOTE"
    assert winner["winner"] == "a"

    none = decide([failing], close_call_pp=10.0, extended_gt_target=100)
    assert none["status"] == "NO_PROMOTION"

    print("p1_jersey_exit_gate_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
