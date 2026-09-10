from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from jersey_uncertainty_ocr_benchmark import (
    TiedDigitAwareClassifier,
    _binary_auc,
    _numeric_metrics,
    _preprocess_rgb,
    _track_metrics,
)


def main() -> int:
    image = np.zeros((120, 60, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    official = _preprocess_rgb(image, "official_player")
    torso = _preprocess_rgb(image, "torso_direct")
    assert tuple(official.shape) == (3, 224, 224)
    assert tuple(torso.shape) == (3, 224, 224)
    assert float(official.min()) >= -1.0001
    assert float(official.max()) <= 1.0001

    classifier = TiedDigitAwareClassifier(
        embed_dim=8,
        embedding_type="multiplicative",
        per_digit_bias=False,
    )
    logits = classifier(torch.zeros(2, 8))
    assert tuple(logits.shape) == (2, 101)

    auc = _binary_auc(
        pd.Series([False, False, True, True]),
        pd.Series([0.1, 0.2, 0.8, 0.9]),
    )
    assert auc is not None and abs(auc - 1.0) < 1e-9

    metrics_frame = pd.DataFrame(
        [
            {
                "label_kind": "number",
                "pred_number": 14,
                "jersey_gt_int": 14,
                "top3_hit": True,
                "top5_hit": True,
                "pred_score": 0.8,
                "uncertainty": 0.2,
            },
            {
                "label_kind": "number",
                "pred_number": 4,
                "jersey_gt_int": 23,
                "top3_hit": False,
                "top5_hit": True,
                "pred_score": 0.6,
                "uncertainty": 0.4,
            },
            {
                "label_kind": "unreadable",
                "pred_number": 8,
                "jersey_gt_int": np.nan,
                "top3_hit": False,
                "top5_hit": False,
                "pred_score": 0.2,
                "uncertainty": 0.9,
            },
        ]
    )
    metrics = _numeric_metrics(metrics_frame)
    assert metrics["rows"] == 2
    assert metrics["correct"] == 1
    assert abs(metrics["exact_accuracy"] - 0.5) < 1e-9
    assert abs(metrics["top5_accuracy"] - 1.0) < 1e-9

    probs_14 = np.zeros(100, dtype=float)
    probs_14[14] = 1.0
    probs_23 = np.zeros(100, dtype=float)
    probs_23[23] = 1.0

    tracks = pd.DataFrame(
        [
            {
                "video": "game",
                "track_id": "1",
                "identity_group": "",
                "label_kind": "number",
                "jersey_gt_int": 14,
                "uncertainty": 0.1,
                "_number_probs": probs_14,
                "split": "train",
            },
            {
                "video": "game",
                "track_id": "1",
                "identity_group": "",
                "label_kind": "unreadable",
                "jersey_gt_int": np.nan,
                "uncertainty": 0.3,
                "_number_probs": probs_14,
                "split": "train",
            },
            {
                "video": "game",
                "track_id": "2",
                "identity_group": "",
                "label_kind": "number",
                "jersey_gt_int": 14,
                "uncertainty": 0.2,
                "_number_probs": probs_14,
                "split": "test",
            },
            {
                "video": "game",
                "track_id": "2",
                "identity_group": "",
                "label_kind": "number",
                "jersey_gt_int": 23,
                "uncertainty": 0.2,
                "_number_probs": probs_23,
                "split": "test",
            },
        ]
    )
    track_metrics = _track_metrics(tracks)
    assert track_metrics["groups"] == 1
    assert track_metrics["skipped_ambiguous_groups"] == 1
    assert (
        track_metrics["schemes"]["mean_all_frames"]["exact_accuracy"] == 1.0
    )

    print("jersey uncertainty OCR benchmark smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
