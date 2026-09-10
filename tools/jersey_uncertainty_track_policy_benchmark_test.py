from __future__ import annotations

import numpy as np
import pandas as pd

from tools.jersey_uncertainty_track_policy_benchmark import _evaluate, _groups


def _row(group: str, split: str, kind: str, gt: int | None, pred: int, absent: bool, u: float, score: float) -> dict:
    probs = np.full(100, 1e-6, dtype=float)
    probs[pred] = 0.9
    probs[(pred + 1) % 100] = 0.1
    return {
        "benchmark_group": group,
        "split": split,
        "label_kind": kind,
        "jersey_gt_int": float(gt) if gt is not None else np.nan,
        "pred_number": pred,
        "pred_score": score,
        "uncertainty": u,
        "absent_argmax": absent,
        "_number_probs": probs,
    }


def main() -> int:
    frame = pd.DataFrame(
        [
            _row("good", "train", "number", 10, 10, False, 0.2, 0.8),
            _row("good", "train", "unreadable", None, 10, True, 0.9, 0.01),
            _row("wrong", "train", "number", 22, 12, False, 0.3, 0.7),
            _row("unread", "train", "unreadable", None, 7, False, 0.2, 0.8),
            _row("amb", "train", "number", 4, 4, False, 0.2, 0.9),
            _row("amb", "train", "number", 6, 6, False, 0.2, 0.9),
        ]
    )
    groups = _groups(frame)
    assert {g["group"] for g in groups} == {"good", "wrong", "unread"}

    policy = {
        "require_not_absent": True,
        "uncertainty_max": None,
        "pred_score_min": None,
        "aggregation": "best_uncertainty",
        "min_frames": 1,
        "agg_margin_min": 0.0,
        "vote_fraction_min": 0.0,
    }
    metrics = _evaluate(groups, policy, "train")
    assert metrics["numeric_groups"] == 2
    assert metrics["unreadable_groups"] == 1
    assert metrics["accepted_groups"] == 3
    assert metrics["correct"] == 1
    assert abs(metrics["decision_precision"] - (1 / 3)) < 1e-9
    assert metrics["unreadable_false_accept_rate"] == 1.0

    print("jersey uncertainty track policy benchmark smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
