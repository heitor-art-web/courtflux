from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


DEFAULT_BASKETBALL_MODEL_ID = "basketball-player-detection-3-ycjdo/13"
DEFAULT_ROBOFLOW_API_URL = "https://serverless.roboflow.com"


@dataclass(slots=True)
class RFDETRPlayerDetection:
    box: list[float]
    confidence: float


class RFDETRPlayerDetector:
    """P1 challenger adapter for basketball-specific RF-DETR player detections.

    The promotion path uses the public basketball model and filters the semantic
    class ``player``. A local COCO-person backend remains available only as a
    diagnostic; it is not quality-gate eligible because ``person`` also includes
    referees, staff and spectators.
    """

    def __init__(
        self,
        *,
        inference_threshold: float = 0.03,
        player_threshold: float = 0.10,
        backend: str = "roboflow",
        model_id: str = DEFAULT_BASKETBALL_MODEL_ID,
        target_class: str = "player",
        api_url: str = DEFAULT_ROBOFLOW_API_URL,
        api_key: str | None = None,
    ):
        self.inference_threshold = float(inference_threshold)
        self.player_threshold = float(player_threshold)
        self.backend = str(backend).strip().lower()
        self.model_id = str(model_id)
        self.target_class = str(target_class).strip().lower()
        self.api_url = str(api_url)
        self.model = None
        self.client = None

        if self.backend == "roboflow":
            try:
                from inference_sdk import InferenceConfiguration, InferenceHTTPClient
            except ImportError as exc:  # pragma: no cover - local environment dependent
                raise RuntimeError(
                    "Roboflow inference dependency is missing. Install with: "
                    "pip install -r requirements-rfdetr.txt"
                ) from exc

            resolved_key = api_key or os.getenv("ROBOFLOW_API_KEY")
            if not resolved_key:
                raise RuntimeError(
                    "ROBOFLOW_API_KEY is required for the basketball-specific RF-DETR challenger. "
                    "Export it in the shell; do not commit it to the repository."
                )
            self.client = InferenceHTTPClient(api_url=self.api_url, api_key=resolved_key)
            # Current serverless API supports header transport and avoids leaking
            # the key in query parameters/logs.
            self.client.configure(InferenceConfiguration(api_key_transport="header"))
            return

        if self.backend == "coco":
            try:
                from rfdetr import RFDETRSmall
            except ImportError as exc:  # pragma: no cover - local environment dependent
                raise RuntimeError(
                    "RF-DETR local diagnostic dependency is missing. Install with: "
                    "pip install -r requirements-rfdetr.txt"
                ) from exc
            self.model = RFDETRSmall()
            return

        raise ValueError(f"Unsupported RF-DETR challenger backend: {backend}")

    @staticmethod
    def _xywh_center_to_xyxy(prediction: dict[str, Any]) -> list[float] | None:
        try:
            x = float(prediction["x"])
            y = float(prediction["y"])
            width = float(prediction["width"])
            height = float(prediction["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if width <= 0.0 or height <= 0.0:
            return None
        return [
            x - width / 2.0,
            y - height / 2.0,
            x + width / 2.0,
            y + height / 2.0,
        ]

    @classmethod
    def _players_from_roboflow_response(
        cls,
        response: dict[str, Any],
        *,
        target_class: str,
        player_threshold: float,
    ) -> list[RFDETRPlayerDetection]:
        output: list[RFDETRPlayerDetection] = []
        predictions = response.get("predictions", []) if isinstance(response, dict) else []
        for prediction in predictions or []:
            if not isinstance(prediction, dict):
                continue
            class_name = str(prediction.get("class", "")).strip().lower()
            if class_name != str(target_class).strip().lower():
                continue
            try:
                confidence = float(prediction.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if confidence < float(player_threshold):
                continue
            box = cls._xywh_center_to_xyxy(prediction)
            if box is None:
                continue
            output.append(RFDETRPlayerDetection(box=box, confidence=confidence))
        return output

    def _detect_roboflow(self, frame_bgr: np.ndarray) -> list[RFDETRPlayerDetection]:
        assert self.client is not None
        response = self.client.infer(frame_bgr, model_id=self.model_id)
        return self._players_from_roboflow_response(
            response,
            target_class=self.target_class,
            player_threshold=self.player_threshold,
        )

    def _detect_coco_diagnostic(self, frame_bgr: np.ndarray) -> list[RFDETRPlayerDetection]:
        assert self.model is not None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(frame_rgb, threshold=self.inference_threshold)
        boxes = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
        confidences = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
        class_ids = np.asarray(getattr(detections, "class_id", []), dtype=np.int64)
        data = getattr(detections, "data", {})
        names = list(data.get("class_name", [])) if isinstance(data, dict) else []

        output: list[RFDETRPlayerDetection] = []
        for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            class_name = str(names[index]).strip().lower() if index < len(names) else None
            is_person = class_name == "person" if class_name is not None else int(class_id) == 1
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

    def detect(self, frame_bgr: np.ndarray) -> list[RFDETRPlayerDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if self.backend == "roboflow":
            return self._detect_roboflow(frame_bgr)
        return self._detect_coco_diagnostic(frame_bgr)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id if self.backend == "roboflow" else "rfdetr-small-coco",
            "target_class": self.target_class if self.backend == "roboflow" else "person",
            "semantic_quality_gate_eligible": self.backend == "roboflow" and self.target_class == "player",
            "api_url": self.api_url if self.backend == "roboflow" else None,
        }
