from __future__ import annotations

import pandas as pd

from jersey_temporal_ocr_benchmark import _metrics, _sequence_groups, _uniform_indices


def main() -> int:
    assert _uniform_indices(1, 8) == [0] * 8
    assert _uniform_indices(8, 8) == list(range(8))
    assert len(_uniform_indices(3, 8)) == 8
    assert min(_uniform_indices(3, 8)) == 0
    assert max(_uniform_indices(3, 8)) == 2

    rows = [
        {
            "sample_id": "a1",
            "video": "/tmp/game.mp4",
            "track_id": "10",
            "timestamp_sec": "1.0",
            "identity_group": "",
            "jersey_gt": "14",
            "label_kind": "number",
            "split": "train",
        },
        {
            "sample_id": "a2",
            "video": "/tmp/game.mp4",
            "track_id": "10",
            "timestamp_sec": "2.0",
            "identity_group": "",
            "jersey_gt": "unreadable",
            "label_kind": "unreadable",
            "split": "train",
        },
        {
            "sample_id": "b1",
            "video": "/tmp/game.mp4",
            "track_id": "20",
            "timestamp_sec": "1.0",
            "identity_group": "",
            "jersey_gt": "10",
            "label_kind": "number",
            "split": "test",
        },
        {
            "sample_id": "b2",
            "video": "/tmp/game.mp4",
            "track_id": "20",
            "timestamp_sec": "2.0",
            "identity_group": "",
            "jersey_gt": "16",
            "label_kind": "number",
            "split": "test",
        },
    ]
    groups, skipped = _sequence_groups(pd.DataFrame(rows))
    assert len(groups) == 1
    assert groups[0]["gt"] == "14"
    assert groups[0]["source_crops"] == 2
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "multiple_numeric_labels_without_identity_segmentation"

    metrics = _metrics(
        pd.DataFrame(
            [
                {"correct": True, "top3_hit": True, "top5_hit": True, "source_crops": 8},
                {"correct": False, "top3_hit": True, "top5_hit": True, "source_crops": 4},
            ]
        )
    )
    assert metrics["groups"] == 2
    assert abs(metrics["exact_accuracy"] - 0.5) < 1e-9
    assert abs(metrics["top3_accuracy"] - 1.0) < 1e-9

    print("jersey temporal OCR benchmark smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
