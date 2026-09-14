from __future__ import annotations

from typing import Any

import numpy as np


class SingleForwardEntityTracker:
    """Feed already-computed YOLO detections into an Ultralytics tracker.

    This deliberately separates detector inference from tracking. The caller owns the
    single YOLO forward; this adapter sees only the player/referee subset and therefore
    cannot suppress raw ball/rim/backboard evidence.

    Ultralytics has changed tracker constructor/YAML helpers across 8.3.x releases, so
    imports and constructor calls are intentionally compatibility-tolerant.
    """

    def __init__(
        self,
        tracker_yaml: str,
        *,
        device: str,
        frame_rate: float = 30.0,
    ) -> None:
        self.tracker_yaml = str(tracker_yaml)
        self.device = str(device)
        self.frame_rate = max(1.0, float(frame_rate))
        self.tracker = self._build_tracker()

    def _load_config(self) -> Any:
        from ultralytics.utils import IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        resolved = check_yaml(self.tracker_yaml)
        data = None
        try:
            from ultralytics.utils import YAML

            loader = getattr(YAML, "load", None)
            if callable(loader):
                data = loader(resolved)
        except (ImportError, AttributeError, TypeError):
            data = None
        if data is None:
            try:
                from ultralytics.utils import yaml_load

                data = yaml_load(resolved)
            except ImportError:
                import yaml

                with open(resolved, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)

        cfg = IterableSimpleNamespace(**dict(data or {}))
        setattr(cfg, "device", self.device)
        return cfg

    @staticmethod
    def _tracker_map() -> dict[str, Any]:
        try:
            from ultralytics.trackers.track import TRACKER_MAP

            return dict(TRACKER_MAP)
        except (ImportError, AttributeError):
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.trackers.byte_tracker import BYTETracker

            return {"botsort": BOTSORT, "bytetrack": BYTETracker}

    def _build_tracker(self) -> Any:
        cfg = self._load_config()
        tracker_type = str(getattr(cfg, "tracker_type", "botsort")).lower()
        tracker_map = self._tracker_map()
        if tracker_type not in tracker_map:
            raise ValueError(
                f"Unsupported tracker_type={tracker_type!r} for single-forward path; "
                f"available={sorted(tracker_map)}"
            )
        tracker_cls = tracker_map[tracker_type]
        try:
            return tracker_cls(args=cfg)
        except TypeError:
            try:
                return tracker_cls(args=cfg, frame_rate=int(round(self.frame_rate)))
            except TypeError:
                return tracker_cls(cfg, frame_rate=int(round(self.frame_rate)))

    def reset(self) -> None:
        reset = getattr(self.tracker, "reset", None)
        if callable(reset):
            reset()

    @staticmethod
    def _rows_from_array(tracks: Any) -> list[dict[str, Any]]:
        if tracks is None:
            return []
        values = np.asarray(tracks)
        if values.size == 0 or values.ndim == 0:
            return []
        if values.ndim == 1:
            values = values.reshape(1, -1)
        rows: list[dict[str, Any]] = []
        for row in values:
            if len(row) < 7:
                continue
            # Current Ultralytics tracker output is
            # [x1, y1, x2, y2, track_id, score, cls, detection_idx].
            rows.append(
                {
                    "box": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                    "track_id": int(round(float(row[4]))),
                    "confidence": float(row[5]),
                    "class_id": int(round(float(row[6]))),
                    "detection_index": (
                        int(round(float(row[-1]))) if len(row) >= 8 else None
                    ),
                }
            )
        return rows

    @staticmethod
    def _rows_from_objects(tracks: Any) -> list[dict[str, Any]]:
        if tracks is None:
            return []
        rows: list[dict[str, Any]] = []
        for track in list(tracks or []):
            raw_box = getattr(track, "tlbr", None)
            if raw_box is None:
                raw_box = getattr(track, "xyxy", None)
            if raw_box is None:
                continue
            box = np.asarray(raw_box, dtype=np.float64).reshape(-1)
            if len(box) < 4:
                continue
            rows.append(
                {
                    "box": [float(value) for value in box[:4]],
                    "track_id": int(getattr(track, "track_id", -1)),
                    "confidence": float(
                        getattr(track, "score", getattr(track, "conf", 0.0))
                    ),
                    "class_id": int(getattr(track, "cls", -1)),
                    "detection_index": getattr(track, "idx", None),
                }
            )
        return rows

    def update(self, detections: Any, frame: np.ndarray) -> list[dict[str, Any]]:
        """Advance the tracker using an Ultralytics ``Boxes``-like detection object."""
        if detections is None:
            return []
        cpu = detections.cpu() if hasattr(detections, "cpu") else detections
        tracker_input = cpu.numpy() if hasattr(cpu, "numpy") else cpu
        tracks = self.tracker.update(tracker_input, frame)
        if tracks is None:
            return []
        try:
            return self._rows_from_array(tracks)
        except (TypeError, ValueError):
            return self._rows_from_objects(tracks)

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "single_forward_entity_tracker_v1",
            "tracker_yaml": self.tracker_yaml,
            "device": self.device,
            "frame_rate": self.frame_rate,
            "detector_inference_owned_by_tracker": False,
            "raw_ball_evidence_passes_through_tracker": False,
        }
