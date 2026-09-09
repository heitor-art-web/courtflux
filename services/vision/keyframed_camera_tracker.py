from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from services.vision.camera_tracker import CameraTracker as BaseCameraTracker


class CameraTracker(BaseCameraTracker):
    """CameraTracker with periodic keypoint refresh for remote court inference.

    Serverless inference is intentionally not called on every frame. A validated direct
    keypoint calibration is refreshed at a fixed interval and the existing optical-flow
    hold propagates camera motion between refreshes. Legacy/local inference keeps the
    historical every-frame behaviour unless the interval is explicitly configured.
    """

    def __init__(self, *args, **kwargs) -> None:
        requested_backend = os.getenv(
            "BASKETBALL_COURT_INFERENCE_BACKEND",
            "legacy",
        ).strip().lower() or "legacy"
        raw_interval = os.getenv("BASKETBALL_COURT_KEYPOINT_INTERVAL", "").strip()
        fallback = 30 if requested_backend in {"serverless", "roboflow", "http"} else 1
        try:
            interval = int(raw_interval) if raw_interval else fallback
        except ValueError:
            interval = fallback
        self.keypoint_interval_frames = max(1, min(interval, 900))
        self._keypoint_detection_calls = 0
        super().__init__(*args, **kwargs)

    def _detect_keypoints(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        call_index = int(self._keypoint_detection_calls)
        self._keypoint_detection_calls += 1
        should_refresh = call_index == 0 or call_index % self.keypoint_interval_frames == 0
        if not should_refresh:
            # Preserve the most recent calibration quality diagnostics. Base `update`
            # will run optical flow and mark the source as optical_flow_hold when it
            # successfully propagates motion.
            self.last_calibration_diagnostics["source"] = "keypoint_refresh_skipped"
            self.last_calibration_diagnostics["keypoint_interval_frames"] = int(
                self.keypoint_interval_frames
            )
            self.last_calibration_diagnostics["keypoint_backend"] = self.keypoint_backend
            return None, None

        result = super()._detect_keypoints(frame)
        self.last_calibration_diagnostics["keypoint_interval_frames"] = int(
            self.keypoint_interval_frames
        )
        self.last_calibration_diagnostics["keypoint_refresh_index"] = int(call_index)
        return result

    def process(self, frame_state):
        frame_state = super().process(frame_state)
        camera = frame_state.diagnostics.setdefault("camera", {})
        camera["keypoint_interval_frames"] = int(self.keypoint_interval_frames)
        camera["keypoint_detection_calls"] = int(self._keypoint_detection_calls)
        return frame_state
