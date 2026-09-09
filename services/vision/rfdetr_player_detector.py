from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class RFDETRPlayerDetection:
    box: list[float]
    confidence: float


class RFDETRPlayerDetector:
    """Optional P1 challenger adapter around RF-DETR pretrained COCO person detections."""

    def __init__(self, *, inference_threshold: float = 0.03, player_threshold: float = 0.10):
        try:
            from rfdetr import RFDETRSmall
        except ImportError as exc:  # pragma: no cover - exercised by local smoke test
            raise RuntimeError(
                "RF-DETR challenger dependency is missing. Install with: "
                "pip install -r requirements-rfdetr.txt"
            ) from exc

        self.model = RFDETRSmall()
        self.inference_threshold = float(inference_threshold)
        self.player_threshold = float(player_threshold)
        self.person_class_ids = self._resolve_person_class_ids(getattr(self.model, "class_names", {}))
        if not self.person_class_ids:
            raise RuntimeError("RF-DETR pretrained class map does not contain a 'person' class")

    @staticmethod
    def _resolve_person_class_ids(class_names: Any) -> set[int]:
        if isinstance(class_names, dict):
            return {
                int(class_id)
                for class_id, name in class_names.items()
                if str(name).strip().lower() == "person"
            }
        if isinstance(class_names, (list, tuple)):
            return {index for index, name in enumerate(class_names) if str(name).strip().lower() == "person"}
        return set()

    def detect(self, frame_bgr: np.ndarray) -> list[RFDETRPlayerDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(frame_rgb, threshold=self.inference_threshold)

        boxes = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
        confidences = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
        class_ids = np.asarray(getattr(detections, "class_id", []), dtype=np.int64)

        output: list[RFDETRPlayerDetection] = []
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if int(class_id) not in self.person_class_ids:
                continue
            confidence_value = float(confidence)
            if confidence_value < self.player_threshold:
                continue
            output.append(
                RFDETRPlayerDetection(
                    box=[float(value) for value in box[:4]],
                    confidence=confidence_value,
                )
            )
        return output
