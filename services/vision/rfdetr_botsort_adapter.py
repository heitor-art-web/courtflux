from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.vision.rfdetr_player_detector import RFDETRPlayerDetection


class _DetectionBatch:
    """Minimal Results-like object consumed by current Ultralytics BYTETracker/BOTSORT."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index):
        return _DetectionBatch(self.xyxy[index], self.conf[index], self.cls[index])

    @property
    def xywh(self) -> np.ndarray:
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        result = self.xyxy.copy()
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result


@dataclass(slots=True)
class TrackedPlayer:
    track_id: int
    box: list[float]
    confidence: float


class RFDETRBoTSORTAdapter:
    """Feed RF-DETR person detections through Ultralytics' BoT-SORT implementation."""

    def __init__(self, tracker_yaml: str = "botsort.yaml"):
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.utils import IterableSimpleNamespace, YAML
            from ultralytics.utils.checks import check_yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Ultralytics tracking components are unavailable") from exc

        config = YAML.load(check_yaml(tracker_yaml))
        # External detections do not provide native YOLO features. Keep ReID disabled
        # so the challenger compares detector quality under the same association family.
        config["with_reid"] = False
        self.tracker = BOTSORT(args=IterableSimpleNamespace(**config))

    @staticmethod
    def _batch(detections: list[RFDETRPlayerDetection]) -> _DetectionBatch:
        if not detections:
            return _DetectionBatch(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return _DetectionBatch(
            np.asarray([item.box for item in detections], dtype=np.float32),
            np.asarray([item.confidence for item in detections], dtype=np.float32),
            np.zeros((len(detections),), dtype=np.float32),
        )

    def update(self, detections: list[RFDETRPlayerDetection], frame_bgr: np.ndarray) -> list[TrackedPlayer]:
        batch = self._batch(detections)
        tracked = np.asarray(self.tracker.update(batch, img=frame_bgr), dtype=np.float32)

        track_id_by_detection: dict[int, int] = {}
        if tracked.size:
            tracked = tracked.reshape(-1, tracked.shape[-1])
            # Ultralytics STrack.result = xyxy, track_id, score, cls, detection_index.
            for row in tracked:
                if len(row) < 8:
                    continue
                detection_index = int(round(float(row[7])))
                track_id_by_detection[detection_index] = int(round(float(row[4])))

        output: list[TrackedPlayer] = []
        for index, detection in enumerate(detections):
            output.append(
                TrackedPlayer(
                    track_id=track_id_by_detection.get(index, -1),
                    box=list(detection.box),
                    confidence=float(detection.confidence),
                )
            )
        return output
