from __future__ import annotations

from evaluation.detector_multiclass_calibration import (
    binary_metrics,
    confusion_matrix,
    iou,
    normalize_name,
)


def item(class_name: str, box: list[float], confidence: float | None = None):
    row = {"class_name": class_name, "box": box}
    if confidence is not None:
        row["confidence"] = confidence
    return row


def main() -> int:
    assert normalize_name("Ref") == "referee"
    assert normalize_name("basketball-hoop") == "rim"
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    records = [
        {
            "ground_truth": [
                item("player", [0, 0, 100, 100]),
                item("referee", [200, 0, 300, 100]),
                item("rim", [400, 0, 450, 50]),
            ],
            "predictions": [
                item("player", [0, 0, 100, 100], 0.90),
                # Deliberate class confusion: referee is predicted as player.
                item("player", [200, 0, 300, 100], 0.80),
                item("rim", [400, 0, 450, 50], 0.85),
                # Unmatched basketball false positive.
                item("basketball", [500, 0, 520, 20], 0.70),
            ],
        }
    ]

    player = binary_metrics(records, "player", threshold=0.5, match_iou=0.5)
    assert player["tp"] == 1
    assert player["fp"] == 1
    assert player["fn"] == 0
    assert player["precision"] == 0.5
    assert player["recall"] == 1.0

    referee = binary_metrics(records, "referee", threshold=0.5, match_iou=0.5)
    assert referee["tp"] == 0
    assert referee["fp"] == 0
    assert referee["fn"] == 1

    classes = ["basketball", "player", "referee", "rim"]
    matrix = confusion_matrix(
        records,
        classes,
        {name: 0.5 for name in classes},
        match_iou=0.5,
    )
    assert matrix["matrix"]["player"]["player"] == 1
    assert matrix["matrix"]["referee"]["player"] == 1
    assert matrix["matrix"]["rim"]["rim"] == 1
    assert matrix["matrix"]["__background__"]["basketball"] == 1
    assert matrix["cross_confusions"]["referee_as_player"] == 1

    print("detector_multiclass_calibration_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
