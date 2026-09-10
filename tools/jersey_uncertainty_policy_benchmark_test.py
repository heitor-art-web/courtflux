from __future__ import annotations

import pandas as pd

from jersey_uncertainty_policy_benchmark import (
    _accept_mask,
    _calibrate,
    _metrics,
    _prepare,
)


def main() -> int:
    raw = pd.DataFrame(
        [
            {"mode": "official_player", "split": "train", "label_kind": "number", "jersey_gt_int": 10, "pred_number": 10, "pred_score": 0.9, "uncertainty": 0.1, "absent_argmax": False},
            {"mode": "official_player", "split": "train", "label_kind": "number", "jersey_gt_int": 11, "pred_number": 12, "pred_score": 0.8, "uncertainty": 0.2, "absent_argmax": False},
            {"mode": "official_player", "split": "train", "label_kind": "unreadable", "jersey_gt_int": "", "pred_number": 10, "pred_score": 0.7, "uncertainty": 0.3, "absent_argmax": False},
            {"mode": "official_player", "split": "train", "label_kind": "unreadable", "jersey_gt_int": "", "pred_number": 5, "pred_score": 0.1, "uncertainty": 0.9, "absent_argmax": True},
            # Deliberately perfect test rows. Calibration must not depend on them.
            {"mode": "official_player", "split": "test", "label_kind": "number", "jersey_gt_int": 22, "pred_number": 22, "pred_score": 1.0, "uncertainty": 0.0, "absent_argmax": False},
        ]
    )
    frame = _prepare(raw)
    train = frame[frame["split"] == "train"].copy()

    keep = _accept_mask(train, uncertainty_max=None, pred_score_min=None, require_not_absent=True)
    metrics = _metrics(train, keep)
    assert metrics["accepted_rows"] == 3
    assert metrics["correct_accepted"] == 1
    assert abs(metrics["decision_precision"] - (1.0 / 3.0)) < 1e-9
    assert abs(metrics["unreadable_false_accept_rate"] - 0.5) < 1e-9

    calibrated_a = _calibrate(train, 0.70)
    assert calibrated_a["found"]

    changed_test = frame.copy()
    changed_test.loc[changed_test["split"] == "test", "pred_number"] = 99
    calibrated_b = _calibrate(changed_test[changed_test["split"] == "train"].copy(), 0.70)
    assert calibrated_a["uncertainty_max"] == calibrated_b["uncertainty_max"]
    assert calibrated_a["pred_score_min"] == calibrated_b["pred_score_min"]

    print("jersey uncertainty policy benchmark smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
