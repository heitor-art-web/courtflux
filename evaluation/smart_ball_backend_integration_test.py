from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_candidate_pipeline import (
    SmartBallCandidatePipeline,
    SmartBallPipelineConfig,
)


class FakeBackend:
    name = "fake_hf"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        assert frame.shape[:2] == (640, 640)
        assert kwargs["conf"] == 0.10
        assert kwargs["imgsz"] == 960
        assert kwargs["device"] == "cpu"
        return [
            {
                "box": [10, 20, 30, 40],
                "center": (20.0, 30.0),
                "confidence": 0.42,
                "detector_confidence": 0.42,
                "detector_backend": "huggingface",
                "detector_model": "example/model",
                "detector_label": "sports ball",
                "source": "full_frame",
                "interpolated": False,
            }
        ]


def test_crop_inference_reuses_active_backend_and_offsets_coordinates() -> None:
    backend = FakeBackend()
    pipeline = SmartBallCandidatePipeline(
        model=None,
        ball_class_ids=[],
        conf=0.10,
        iou=0.5,
        imgsz=960,
        max_det=100,
        device="mps",
        config=SmartBallPipelineConfig(enabled=True),
        candidate_backend=backend,
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    candidates = pipeline._predict_crop(
        frame,
        crop_box=(100, 200, 740, 840),
        source="zoom_temporal",
        reason="no_full_frame_candidate",
    )

    assert backend.calls == 1
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["box"] == [110, 220, 130, 240]
    assert candidate["center"] == (120.0, 230.0)
    assert candidate["confidence"] == 0.42
    assert candidate["detector_backend"] == "huggingface"
    assert candidate["source"] == "zoom_temporal"
    assert candidate["zoom_reason"] == "no_full_frame_candidate"
    assert candidate["crop_box"] == [100, 200, 740, 840]


def main() -> None:
    test_crop_inference_reuses_active_backend_and_offsets_coordinates()
    print("[PASS] smart ball backend integration")


if __name__ == "__main__":
    main()
