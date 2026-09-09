from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.rfdetr_botsort_adapter import RFDETRBoTSORTAdapter
from services.vision.rfdetr_player_detector import RFDETRPlayerDetection, RFDETRPlayerDetector


class _FakeTracker:
    def update(self, batch, img=None):
        assert len(batch) == 2
        # xyxy, track_id, score, cls, original detection index
        return np.asarray([[0, 0, 10, 20, 42, 0.9, 0, 0]], dtype=np.float32)


class _FakeDetections:
    def __init__(self, class_names):
        self.data = {"class_name": np.asarray(class_names, dtype=object)}


def test_person_class_mapping():
    assert RFDETRPlayerDetector._resolve_person_class_ids({1: "person", 2: "car"}) == {1}
    # A contiguous class-name list can still expose person at index 0, but
    # pretrained COCO predictions themselves use sparse category ID 1.
    assert RFDETRPlayerDetector._resolve_person_class_ids(["person", "bicycle"]) == {0}


def test_detection_class_name_metadata_is_preferred():
    detections = _FakeDetections(["person", "car"])
    assert RFDETRPlayerDetector._class_names_from_detections(detections, 2) == ["person", "car"]


def test_unassigned_detections_are_preserved():
    adapter = RFDETRBoTSORTAdapter.__new__(RFDETRBoTSORTAdapter)
    adapter.tracker = _FakeTracker()
    detections = [
        RFDETRPlayerDetection([0, 0, 10, 20], 0.9),
        RFDETRPlayerDetection([20, 0, 30, 20], 0.8),
    ]
    output = adapter.update(detections, np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(output) == 2
    assert output[0].track_id == 42
    assert output[1].track_id == -1
    assert output[1].confidence == 0.8


if __name__ == "__main__":
    test_person_class_mapping()
    test_detection_class_name_metadata_is_preferred()
    test_unassigned_detections_are_preserved()
    print("rfdetr_challenger_test: PASS")
