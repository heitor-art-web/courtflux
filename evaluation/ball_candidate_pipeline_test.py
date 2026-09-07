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


class FakeModel:
    def __init__(self, results=None):
        self.results = [] if results is None else results
        self.calls = 0

    def predict(self, *args, **kwargs):
        self.calls += 1
        return self.results


class FakeTracker:
    def __init__(self):
        self.last_center = None
        self.last_frame_id = None
        self.velocity = np.zeros(2, dtype=np.float64)

    def inspect_candidate(self, candidate, frame_id, frame_shape):
        return None


def build_pipeline(*, model=None, verify=False, zoom=False) -> SmartBallCandidatePipeline:
    return SmartBallCandidatePipeline(
        model=model or FakeModel(),
        ball_class_ids=[0],
        conf=0.10,
        iou=0.5,
        imgsz=960,
        max_det=100,
        device="cpu",
        config=SmartBallPipelineConfig(
            enabled=True,
            verify_enabled=verify,
            zoom_enabled=zoom,
            crop_size=640,
        ),
    )


def test_crop_box_preserves_size_near_edges() -> None:
    crop = SmartBallCandidatePipeline.crop_box_around_center
    assert crop(1920, 1080, (10.0, 10.0), 640) == (0, 0, 640, 640)
    assert crop(1920, 1080, (1910.0, 1070.0), 640) == (1280, 440, 1920, 1080)


def test_pipeline_preserves_distinct_full_frame_candidates() -> None:
    pipeline = build_pipeline(verify=False, zoom=False)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker = FakeTracker()
    raw = [
        {"box": [100, 100, 120, 120], "center": (110.0, 110.0), "confidence": 0.25},
        {"box": [600, 300, 620, 320], "center": (610.0, 310.0), "confidence": 0.40},
    ]
    enhanced, diagnostics = pipeline.enhance(frame, raw, frame_id=0, tracker=tracker)
    assert len(enhanced) == 2
    assert diagnostics["raw_candidate_count"] == 2
    assert diagnostics["output_candidate_count"] == 2
    assert all(candidate["source"] == "full_frame" for candidate in enhanced)
    assert all(candidate["detector_confidence"] > 0.0 for candidate in enhanced)
    assert all(candidate["rank_score"] > 0.0 for candidate in enhanced)


def test_failed_verification_does_not_delete_raw_candidate() -> None:
    model = FakeModel(results=[])
    pipeline = build_pipeline(model=model, verify=True, zoom=False)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker = FakeTracker()
    raw = [
        {"box": [300, 200, 320, 220], "center": (310.0, 210.0), "confidence": 0.20},
    ]
    enhanced, diagnostics = pipeline.enhance(frame, raw, frame_id=0, tracker=tracker)
    assert len(enhanced) == 1
    assert model.calls == 1
    assert diagnostics["verify_attempts"] == 1
    assert enhanced[0]["source"] == "full_frame"
    assert enhanced[0]["verification_score"] == 0.0
    assert enhanced[0]["verified"] is False


def test_temporal_prior_only_requests_zoom_and_never_creates_ball() -> None:
    model = FakeModel(results=[])
    pipeline = build_pipeline(model=model, verify=False, zoom=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker = FakeTracker()
    tracker.last_center = np.asarray([640.0, 360.0], dtype=np.float64)
    tracker.last_frame_id = 10
    tracker.velocity = np.asarray([8.0, -3.0], dtype=np.float64)

    enhanced, diagnostics = pipeline.enhance(frame, [], frame_id=11, tracker=tracker)
    assert enhanced == []
    assert diagnostics["zoom_triggered"] is True
    assert diagnostics["zoom_reason"] == "no_full_frame_candidate"
    assert diagnostics["zoom_candidate_count"] == 0
    assert diagnostics["zoom_crop_box"] is not None
    assert model.calls == 1


def test_disabled_pipeline_is_exact_candidate_passthrough() -> None:
    pipeline = SmartBallCandidatePipeline(
        model=FakeModel(),
        ball_class_ids=[0],
        conf=0.10,
        iou=0.5,
        imgsz=960,
        max_det=100,
        device="cpu",
        config=SmartBallPipelineConfig(enabled=False),
    )
    raw = [{"box": [1, 2, 3, 4], "center": (2.0, 3.0), "confidence": 0.31}]
    enhanced, diagnostics = pipeline.enhance(
        np.zeros((100, 100, 3), dtype=np.uint8),
        raw,
        frame_id=0,
        tracker=FakeTracker(),
    )
    assert enhanced == raw
    assert diagnostics["enabled"] is False


def main() -> None:
    test_crop_box_preserves_size_near_edges()
    test_pipeline_preserves_distinct_full_frame_candidates()
    test_failed_verification_does_not_delete_raw_candidate()
    test_temporal_prior_only_requests_zoom_and_never_creates_ball()
    test_disabled_pipeline_is_exact_candidate_passthrough()
    print("[PASS] smart ball candidate pipeline")


if __name__ == "__main__":
    main()
