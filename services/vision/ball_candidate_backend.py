from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def parse_label_set(
    raw: str | None,
    fallback: tuple[str, ...] = ("sports ball", "ball", "basketball"),
) -> set[str]:
    values = raw.split(",") if raw and raw.strip() else list(fallback)
    return {normalize_label(value) for value in values if normalize_label(value)}


def build_hf_candidates(
    *, boxes: Any, scores: Any, labels: Any, id2label: dict[Any, Any],
    allowed_labels: set[str], max_det: int, model_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for box, score, label_id in zip(boxes, scores, labels):
        class_id = int(label_id.item() if hasattr(label_id, "item") else label_id)
        raw_label = id2label.get(class_id, id2label.get(str(class_id), class_id))
        label = normalize_label(raw_label)
        if label not in allowed_labels:
            continue
        confidence = float(score.item() if hasattr(score, "item") else score)
        coords = box.tolist() if hasattr(box, "tolist") else list(box)
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in coords]
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append({
            "box": [x1, y1, x2, y2],
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            "confidence": confidence,
            "detector_confidence": confidence,
            "detector_backend": "huggingface",
            "detector_model": model_id,
            "detector_class_id": class_id,
            "detector_label": label,
            "source": "full_frame",
            "interpolated": False,
        })
    candidates.sort(key=lambda candidate: float(candidate["confidence"]), reverse=True)
    return candidates[: max(0, int(max_det))]


class YOLOBallCandidateBackend:
    name = "yolo"
    timing_name = "yolo_predict_ball"

    def __init__(self, *, model_path: str, device: str, ball_predicate: Callable[[str], bool]) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.raw_model = YOLO(self.model_path)
        self.raw_model.to(self.device)
        self.class_ids = self._class_ids(self.raw_model, ball_predicate)
        if not self.class_ids:
            raise RuntimeError("YOLO ball backend exposes no ball class.")

    @staticmethod
    def _class_ids(model: Any, predicate: Callable[[str], bool]) -> list[int]:
        names = getattr(model, "names", {}) or {}
        items = names.items() if isinstance(names, dict) else enumerate(names)
        return [int(class_id) for class_id, name in items if predicate(str(name).lower())]

    def predict(self, frame: Any, *, conf: float, iou: float, imgsz: int, max_det: int, device: str | None = None) -> list[dict[str, Any]]:
        results = self.raw_model.predict(
            frame, conf=float(conf), iou=float(iou), imgsz=int(imgsz), max_det=int(max_det),
            classes=self.class_ids, device=str(device or self.device), verbose=False,
        )
        candidates: list[dict[str, Any]] = []
        if not results or getattr(results[0], "boxes", None) is None:
            return candidates
        names = getattr(self.raw_model, "names", {}) or {}
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            raw_label = names.get(class_id, class_id) if isinstance(names, dict) else names[class_id]
            candidates.append({
                "box": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "confidence": confidence,
                "detector_confidence": confidence,
                "detector_backend": self.name,
                "detector_model": self.model_path,
                "detector_class_id": class_id,
                "detector_label": normalize_label(raw_label),
                "source": "full_frame",
                "interpolated": False,
            })
        return candidates

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name, "model": self.model_path, "class_ids": list(self.class_ids),
            "device": self.device, "threshold_semantics": "ultralytics_confidence",
            "image_size_semantics": "explicit_imgsz",
        }


class HuggingFaceBallCandidateBackend:
    name = "huggingface"
    timing_name = "hf_predict_ball"

    def __init__(self, *, model_id: str, revision: str | None, device: str, allowed_labels: set[str]) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face ball detection requires the optional HF dependencies. "
                "Install them with: pip install -r requirements-hf.txt"
            ) from exc
        self.torch = torch
        self.model_id = str(model_id)
        self.revision = str(revision).strip() if revision and str(revision).strip() else None
        self.device = str(device)
        self.allowed_labels = {normalize_label(label) for label in allowed_labels}
        load_kwargs: dict[str, Any] = {"trust_remote_code": False}
        if self.revision:
            load_kwargs["revision"] = self.revision
        hf_token = os.getenv("HF_TOKEN", "").strip()
        if hf_token:
            load_kwargs["token"] = hf_token
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, **load_kwargs)
        self.raw_model = AutoModelForObjectDetection.from_pretrained(self.model_id, **load_kwargs)
        self.raw_model.to(self.device)
        self.raw_model.eval()
        self.id2label = dict(getattr(self.raw_model.config, "id2label", {}) or {})
        available = {normalize_label(label) for label in self.id2label.values()}
        self.matched_labels = sorted(self.allowed_labels & available)
        if not self.matched_labels:
            raise RuntimeError(
                "Hugging Face ball backend found none of the requested ball labels. "
                f"requested={sorted(self.allowed_labels)} available={sorted(available)}. "
                "Set BASKETBALL_HF_BALL_LABELS to exact model labels."
            )

    def predict(self, frame: Any, *, conf: float, iou: float, imgsz: int, max_det: int, device: str | None = None) -> list[dict[str, Any]]:
        del iou, imgsz
        from PIL import Image
        inference_device = str(device or self.device)
        if inference_device != self.device:
            raise RuntimeError(
                f"HF backend was initialized on {self.device!r} but inference requested {inference_device!r}. "
                "Set BASKETBALL_HF_DEVICE before startup."
            )
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return []
        height, width = frame.shape[:2]
        rgb = frame[..., ::-1] if frame.ndim == 3 and frame.shape[2] >= 3 else frame
        image = Image.fromarray(np.ascontiguousarray(rgb))
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.raw_model(**inputs)
        target_sizes = self.torch.tensor([[height, width]], device=self.device)
        processed = self.processor.post_process_object_detection(outputs, threshold=float(conf), target_sizes=target_sizes)[0]
        return build_hf_candidates(
            boxes=processed.get("boxes", []), scores=processed.get("scores", []), labels=processed.get("labels", []),
            id2label=self.id2label, allowed_labels=self.allowed_labels, max_det=max_det, model_id=self.model_id,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name, "model": self.model_id, "revision": self.revision,
            "labels": sorted(self.allowed_labels), "matched_labels": list(self.matched_labels),
            "device": self.device, "threshold_semantics": "transformers_post_process_threshold",
            "image_size_semantics": "model_processor",
            "hf_auth": "HF_TOKEN" if bool(os.getenv("HF_TOKEN", "").strip()) else "local_auth_or_public",
        }


class RoboflowBallCandidateBackend:
    """Optional challenger using a public/private Roboflow Inference model."""
    name = "roboflow"
    timing_name = "roboflow_predict_ball"

    def __init__(self, *, model_id: str, allowed_labels: set[str]) -> None:
        try:
            from inference import get_model
        except ImportError as exc:
            raise RuntimeError(
                "Roboflow backend requires optional dependencies. Install: "
                "python3 -m pip install -r requirements-roboflow.txt"
            ) from exc
        self.model_id = str(model_id)
        self.allowed_labels = {normalize_label(label) for label in allowed_labels}
        self.device = "roboflow_inference"
        # inference uses ROBOFLOW_API_KEY from the environment when model access needs it.
        self.raw_model = get_model(model_id=self.model_id)

    @staticmethod
    def _prediction_value(prediction: Any, name: str, default: Any = None) -> Any:
        if isinstance(prediction, dict):
            return prediction.get(name, default)
        return getattr(prediction, name, default)

    def predict(self, frame: Any, *, conf: float, iou: float, imgsz: int, max_det: int, device: str | None = None) -> list[dict[str, Any]]:
        del iou, imgsz, device
        raw = self.raw_model.infer(frame, confidence=float(conf))
        result = raw[0] if isinstance(raw, list) and raw else raw
        if result is None:
            return []
        if isinstance(result, dict):
            predictions = result.get("predictions", []) or []
        else:
            predictions = getattr(result, "predictions", []) or []
        candidates: list[dict[str, Any]] = []
        for prediction in predictions:
            raw_label = self._prediction_value(prediction, "class_name", None)
            if raw_label is None:
                raw_label = self._prediction_value(prediction, "class", "")
            label = normalize_label(raw_label)
            if label not in self.allowed_labels:
                continue
            confidence = float(self._prediction_value(prediction, "confidence", 0.0) or 0.0)
            x = float(self._prediction_value(prediction, "x", 0.0) or 0.0)
            y = float(self._prediction_value(prediction, "y", 0.0) or 0.0)
            width = float(self._prediction_value(prediction, "width", 0.0) or 0.0)
            height = float(self._prediction_value(prediction, "height", 0.0) or 0.0)
            x1, y1, x2, y2 = x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = self._prediction_value(prediction, "class_id", None)
            candidates.append({
                "box": [x1, y1, x2, y2], "center": (x, y), "confidence": confidence,
                "detector_confidence": confidence, "detector_backend": self.name,
                "detector_model": self.model_id, "detector_class_id": class_id,
                "detector_label": label, "source": "full_frame", "interpolated": False,
            })
        candidates.sort(key=lambda candidate: float(candidate["confidence"]), reverse=True)
        return candidates[: max(0, int(max_det))]

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name, "model": self.model_id, "labels": sorted(self.allowed_labels),
            "device": self.device, "threshold_semantics": "roboflow_inference_confidence",
            "image_size_semantics": "model_managed",
            "roboflow_auth": "ROBOFLOW_API_KEY" if bool(os.getenv("ROBOFLOW_API_KEY", "").strip()) else "public_or_local_cache",
        }


def create_ball_candidate_backend(
    *, backend_name: str, yolo_model_path: str | None, device: str,
    ball_predicate: Callable[[str], bool],
) -> YOLOBallCandidateBackend | HuggingFaceBallCandidateBackend | RoboflowBallCandidateBackend:
    backend = str(backend_name or "yolo").strip().lower()
    if backend in {"yolo", "ultralytics"}:
        if not yolo_model_path:
            raise RuntimeError("YOLO ball backend requires a resolvable checkpoint path.")
        return YOLOBallCandidateBackend(model_path=yolo_model_path, device=device, ball_predicate=ball_predicate)
    if backend in {"hf", "huggingface", "rf_detr", "rfdetr"}:
        model_id = os.getenv("BASKETBALL_HF_BALL_MODEL", "Roboflow/rf-detr-medium").strip()
        revision = os.getenv("BASKETBALL_HF_MODEL_REVISION")
        hf_device = os.getenv("BASKETBALL_HF_DEVICE", device).strip() or device
        labels = parse_label_set(os.getenv("BASKETBALL_HF_BALL_LABELS"))
        return HuggingFaceBallCandidateBackend(
            model_id=model_id, revision=revision, device=hf_device, allowed_labels=labels
        )
    if backend in {"roboflow", "rf_inference"}:
        model_id = os.getenv(
            "BASKETBALL_ROBOFLOW_BALL_MODEL", "basketball-player-detection-3-ycjdo/18"
        ).strip()
        labels = parse_label_set(os.getenv("BASKETBALL_ROBOFLOW_BALL_LABELS"), fallback=("ball",))
        return RoboflowBallCandidateBackend(model_id=model_id, allowed_labels=labels)
    raise ValueError(
        f"Unsupported BASKETBALL_BALL_DETECTOR_BACKEND={backend_name!r}; "
        "expected yolo, huggingface, or roboflow."
    )
