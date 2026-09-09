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

        # Pretrained RF-DETR COCO checkpoints emit sparse COCO category IDs
        # (person == 1), while model.class_names may be a contiguous list.
        # Keep class_names-derived IDs only as a fallback; detection-level
        # class_name metadata is preferred in detect().
        self.person_class_ids = {1}
        self.person_class_ids.update(
            self._resolve_person_class_ids(getattr(self.model, "class_names", {}))
        )

    @staticmethod
    def _resolve_person_class_ids(class_names: Any) -> set[int]:
        if isinstance(class_names, dict):
            return {
                int(class_id)
                for class_id, name in class_names.items()
                if str(name).strip().lower() == "person"
            }
        if isinstance(class_names, (list, tuple)):
            return {
                index
                for index, name in enumerate(class_names)
                if str(name).strip().lower() == "person"
            }
        return set()

    @staticmethod
    def _class_names_from_detections(detections: Any, count: int) -> list[str | None]:
        data = getattr(detections, "data", None)
        if not isinstance(data, dict):
            return [None] * count
        values = data.get("class_name")
        if values is None:
            return [None] * count
        try:
            names = list(values)
        except TypeError:
            return [None] * count
        return [str(names[index]) if index < len(names) else None for index in range(count)]

    def detect(self, frame_bgr: np.ndarray) -> list[RFDETRPlayerDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(frame_rgb, threshold=self.inference_threshold)

        boxes = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
        confidences = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
        class_ids = np.asarray(getattr(detections, "class_id", []), dtype=np.int64)
        class_names = self._class_names_from_detections(detections, len(boxes))

        output: list[RFDETRPlayerDetection] = []
        for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            class_name = class_names[index]
            if class_name is not None:
                is_person = class_name.strip().lower() == "person"
            else:
                is_person = int(class_id) in self.person_class_ids
            if not is_person:
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
