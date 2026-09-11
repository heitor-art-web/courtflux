from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile

import cv2
import numpy as np

from core.frame_state import FrameState
from services.gameplay_filter import GameplayFilter
from services.vision.diagnostic_preview import DiagnosticPreview
from services.vision.hypothesis_p0_detector import PerformanceDetector


def _court_frame(offset: int = 0) -> np.ndarray:
    hsv = np.zeros((240, 320, 3), dtype=np.uint8)
    hsv[:, :, 0] = 15
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 200
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    x = 20 + (offset % 6) * 28
    cv2.rectangle(frame, (x, 80), (x + 38, 160), (245, 245, 245), -1)
    return frame


def test_gameplay_scene_cut_resets_grace() -> None:
    filt = GameplayFilter(processing_scale=1.0, max_grace_frames=15)
    results = [filt.is_gameplay(_court_frame(index)) for index in range(10)]
    assert any(results[-4:]), "court-like sequence should establish gameplay"
    assert filt.grace_counter > 0

    cut = np.zeros((240, 320, 3), dtype=np.uint8)
    assert filt.is_gameplay(cut) is False
    assert filt.last_diagnostics["scene_cut"] is True
    assert filt.last_diagnostics["reason"] == "scene_cut_reset"
    assert filt.grace_counter == 0

    for _ in range(4):
        assert filt.is_gameplay(cut) is False
        assert filt.last_diagnostics["has_court_evidence"] is False


def test_referee_class_is_tracked_but_separate() -> None:
    detector = PerformanceDetector.__new__(PerformanceDetector)
    detector.model = SimpleNamespace(
        names={0: "player", 1: "referee", 2: "basketball", 3: "rim"}
    )
    assert detector._tracked_entity_class_ids() == [0, 1]
    assert detector._is_referee_class("referee")
    assert not detector._is_referee_class("player")

    state = FrameState(
        frame_id=0,
        timestamp=0.0,
        frame=np.zeros((120, 160, 3), dtype=np.uint8),
    )
    state.referees = [
        {
            "track_id": 7,
            "box": [40, 20, 80, 100],
            "center": (60.0, 60.0),
            "confidence": 0.91,
        }
    ]
    assert state.players == []
    assert len(state.referees) == 1


def test_preview_renders_referee_entity() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    state = FrameState(frame_id=0, timestamp=1.5, frame=frame)
    state.referees = [
        {"track_id": 3, "box": [20, 15, 60, 105], "confidence": 0.88}
    ]

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "preview.jpg"
        preview = DiagnosticPreview(output_path=output, interval_frames=1)
        rendered = preview.render(state)
        preview.close()
        assert rendered == output
        assert output.exists() and output.stat().st_size > 0


def main() -> None:
    test_gameplay_scene_cut_resets_grace()
    test_referee_class_is_tracked_but_separate()
    test_preview_renders_referee_entity()
    print("p1_gameplay_referee_qa_test: PASS")


if __name__ == "__main__":
    main()
