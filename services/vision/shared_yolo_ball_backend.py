from __future__ import annotations

from typing import Any, Callable


class SharedYOLOBallCandidateBackend:
    """Ball-candidate backend that reuses the already-loaded primary YOLO model.

    Full-frame candidates are supplied by the single-forward detector path. This backend
    therefore exists for Smart Ball crop verification/zoom only and avoids loading a
    second copy of the same checkpoint into MPS/CUDA memory.
    """

    name = "yolo_shared"
    timing_name = "shared_yolo_predict_ball"

    def __init__(
        self,
        *,
        model: Any,
        device: str,
        ball_predicate: Callable[[str], bool],
    ) -> None:
        self.raw_model = model
        self.device = str(device)
        self.model_path = self._resolve_model_path(model)
        self.class_ids = self._class_ids(model, ball_predicate)
        if not self.class_ids:
            raise RuntimeError("Shared YOLO backend exposes no ball class.")

    @staticmethod
    def _resolve_model_path(model: Any) -> str | None:
        candidates = [
            getattr(model, "ckpt_path", None),
            (getattr(model, "overrides", {}) or {}).get("model"),
        ]
        inner = getattr(model, "model", None)
        candidates.append(getattr(inner, "pt_path", None))
        return next((str(value) for value in candidates if value), None)

    @staticmethod
    def _class_ids(model: Any, predicate: Callable[[str], bool]) -> list[int]:
        names = getattr(model, "names", {}) or {}
        items = names.items() if isinstance(names, dict) else enumerate(names)
        return [
            int(class_id)
            for class_id, name in items
            if predicate(str(name).lower())
        ]

    def predict(
        self,
        frame: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        device: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self.raw_model.predict(
            frame,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            max_det=int(max_det),
            classes=self.class_ids,
            device=str(device or self.device),
            verbose=False,
        )
        candidates: list[dict[str, Any]] = []
        if not results or getattr(results[0], "boxes", None) is None:
            return candidates
        boxes = results[0].boxes.cpu()
        xyxy = boxes.xyxy.numpy()
        confs = boxes.conf.numpy()
        classes = boxes.cls.numpy()
        names = getattr(self.raw_model, "names", {}) or {}
        for coords, confidence, class_value in zip(xyxy, confs, classes):
            class_id = int(class_value)
            x1, y1, x2, y2 = [float(value) for value in coords[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            raw_label = (
                names.get(class_id, class_id)
                if isinstance(names, dict)
                else names[class_id]
            )
            candidates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "confidence": float(confidence),
                    "detector_confidence": float(confidence),
                    "detector_backend": self.name,
                    "detector_model": self.model_path,
                    "detector_class_id": class_id,
                    "detector_label": str(raw_label).strip().lower(),
                    "source": "crop",
                    "interpolated": False,
                }
            )
        return candidates

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model_path,
            "class_ids": list(self.class_ids),
            "device": self.device,
            "shared_primary_model": True,
            "full_frame_inference": False,
            "purpose": "smart_ball_crop_only",
        }
