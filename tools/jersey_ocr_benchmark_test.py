from __future__ import annotations

import numpy as np
import pandas as pd

from jersey_ocr_benchmark import (
    _candidate_keep,
    _gate_metrics,
    _numeric_metrics,
    _readability_auc,
    _temporal_metrics,
)


class _FakeAnalyzer:
    @staticmethod
    def _raw_prediction(tens_probs, ones_probs):
        tens = int(np.argmax(tens_probs))
        ones = int(np.argmax(ones_probs))
        raw = f"{tens}{ones}"
        return {
            "raw_canonical": str(int(raw)),
            "raw_probability": float(tens_probs[tens] * ones_probs[ones]),
        }


def main() -> int:
    frame = pd.DataFrame(
        [
            {
                "label_kind": "number",
                "jersey_gt_canonical": "14",
                "raw_canonical": "14",
                "top3_hit": True,
                "top5_hit": True,
                "raw_probability": 0.80,
                "torso_sharpness": 200.0,
                "torso_area": 9000.0,
            },
            {
                "label_kind": "number",
                "jersey_gt_canonical": "23",
                "raw_canonical": "24",
                "top3_hit": True,
                "top5_hit": True,
                "raw_probability": 0.60,
                "torso_sharpness": 150.0,
                "torso_area": 8000.0,
            },
            {
                "label_kind": "unreadable",
                "jersey_gt_canonical": "",
                "raw_canonical": "4",
                "top3_hit": False,
                "top5_hit": False,
                "raw_probability": 0.20,
                "torso_sharpness": 60.0,
                "torso_area": 4000.0,
            },
        ]
    )

    metrics = _numeric_metrics(frame)
    assert metrics["rows"] == 2
    assert metrics["correct"] == 1
    assert metrics["exact_accuracy"] == 0.5
    assert metrics["top3_accuracy"] == 1.0

    auc = _readability_auc(frame)
    assert auc == 1.0

    gate = {
        "family": "sharpness_and_area",
        "sharpness_threshold": 100.0,
        "area_threshold": 8500.0,
    }
    keep = _candidate_keep(frame, gate)
    assert keep.tolist() == [True, False, False]
    gate_metrics = _gate_metrics(frame, gate)
    assert gate_metrics["kept_rows"] == 1
    assert gate_metrics["coverage"] == 0.5
    assert gate_metrics["accuracy_kept"] == 1.0

    temporal = pd.DataFrame(
        [
            {
                "label_kind": "number",
                "jersey_gt_canonical": "14",
                "identity_group": "p14",
                "video": "game.mp4",
                "track_id": "1",
                "_tens_probs": np.array([0.0, 0.9, 0.1]),
                "_ones_probs": np.array([0.0, 0.0, 0.0, 0.1, 0.9]),
            },
            {
                "label_kind": "number",
                "jersey_gt_canonical": "14",
                "identity_group": "p14",
                "video": "game.mp4",
                "track_id": "2",
                "_tens_probs": np.array([0.0, 0.8, 0.2]),
                "_ones_probs": np.array([0.0, 0.0, 0.0, 0.2, 0.8]),
            },
        ]
    )
    temporal_metrics = _temporal_metrics(temporal, _FakeAnalyzer())
    assert temporal_metrics["groups"] == 1
    assert temporal_metrics["multi_crop_groups"] == 1
    assert temporal_metrics["exact_accuracy"] == 1.0

    print("jersey OCR benchmark smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
