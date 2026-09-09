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


def test_roboflow_response_filters_only_player():
    response = {
        "predictions": [
            {"x": 50, "y": 60, "width": 20, "height": 40, "confidence": 0.91, "class": "player"},
            {"x": 80, "y": 60, "width": 20, "height": 40, "confidence": 0.95, "class": "referee"},
            {"x": 20, "y": 20, "width": 10, "height": 10, "confidence": 0.99, "class": "basketball"},
            {"x": 120, "y": 60, "width": 20, "height": 40, "confidence": 0.05, "class": "player"},
        ]
    }
    players = RFDETRPlayerDetector._players_from_roboflow_response(
        response,
        target_class="player",
        player_threshold=0.10,
    )
    assert len(players) == 1
    assert players[0].box == [40.0, 40.0, 60.0, 80.0]
    assert players[0].confidence == 0.91


def test_invalid_boxes_are_rejected():
    response = {
        "predictions": [
            {"x": 10, "y": 10, "width": 0, "height": 20, "confidence": 0.9, "class": "player"}
        ]
    }
    players = RFDETRPlayerDetector._players_from_roboflow_response(
        response,
        target_class="player",
        player_threshold=0.10,
    )
    assert players == []


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
    test_roboflow_response_filters_only_player()
    test_invalid_boxes_are_rejected()
    test_unassigned_detections_are_preserved()
    print("rfdetr_challenger_test: PASS")
