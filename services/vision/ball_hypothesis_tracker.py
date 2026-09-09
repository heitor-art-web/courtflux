from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:  # Optional dependency; production defaults must keep working without it.
    import supervision as sv
    from trackers import OCSORTTracker
    from trackers.utils.iou import CIoU

    TRACKERS_AVAILABLE = True
    TRACKERS_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    sv = None  # type: ignore[assignment]
    OCSORTTracker = None  # type: ignore[assignment]
    CIoU = None  # type: ignore[assignment]
    TRACKERS_AVAILABLE = False
    TRACKERS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


IDENTITY_FEATURES = [
    "detector_confidence",
    "rank_score",
    "confidence_rank_in_confirmed",
    "confirmed_competitors",
    "track_observations",
    "track_streak",
    "track_mean_confidence",
    "track_area_cv",
    "track_speed_px_per_step",
    "box_area",
]


def lifecycle_state(*, tracker_id: int, time_since_update: int) -> str:
    """Map an OC-SORT tracklet to CourtFlux's explicit safe lifecycle."""
    if int(tracker_id) < 0:
        return "TENTATIVE"
    if int(time_since_update) <= 0:
        return "TRACKED"
    return "LOST"


def allows_concrete_output(
    lifecycle: str,
    *,
    observed: bool,
    identity_seed: bool,
) -> bool:
    """Only an observed, identity-approved TRACKED hypothesis may emit a location."""
    return bool(lifecycle == "TRACKED" and observed and identity_seed)


def _sigmoid(value: float) -> float:
    clipped = max(-35.0, min(float(value), 35.0))
    return 1.0 / (1.0 + math.exp(-clipped))


def _candidate_detector_confidence(candidate: dict[str, Any]) -> float:
    for key in ("detector_confidence", "confidence", "rank_score"):
        value = candidate.get(key)
        try:
            if value is not None:
                return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def _candidate_rank_score(candidate: dict[str, Any]) -> float:
    try:
        value = candidate.get("rank_score")
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return _candidate_detector_confidence(candidate)


def _box(candidate: dict[str, Any]) -> list[float] | None:
    value = candidate.get("box")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray(
        ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0),
        dtype=np.float64,
    )


def _box_area(box: np.ndarray) -> float:
    return max(float(box[2] - box[0]), 1.0) * max(float(box[3] - box[1]), 1.0)


@dataclass(frozen=True)
class BallIdentitySeedModel:
    version: str
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "BallIdentitySeedModel":
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        features = list(payload.get("features") or [])
        if features != IDENTITY_FEATURES:
            raise ValueError(
                "Ball identity model feature schema mismatch: "
                f"expected {IDENTITY_FEATURES}, got {features}"
            )
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        weights = np.asarray(payload["weights"], dtype=np.float64)
        if mean.shape != (len(IDENTITY_FEATURES),):
            raise ValueError("Ball identity model mean has invalid shape")
        if scale.shape != mean.shape:
            raise ValueError("Ball identity model scale has invalid shape")
        if weights.shape != (len(IDENTITY_FEATURES) + 1,):
            raise ValueError("Ball identity model weights have invalid shape")
        scale = scale.copy()
        scale[np.abs(scale) < 1e-6] = 1.0
        threshold = float(payload["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Ball identity model threshold must be finite")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"mean", "scale", "weights"}
        }
        return cls(
            version=str(payload.get("version") or "unknown"),
            mean=mean,
            scale=scale,
            weights=weights,
            threshold=threshold,
            metadata=metadata,
        )

    def score(self, features: dict[str, float]) -> float:
        vector = feature_vector(features)
        standardized = (vector - self.mean) / self.scale
        logit = float(self.weights[0] + np.dot(standardized, self.weights[1:]))
        return _sigmoid(logit)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "version": self.version,
            "threshold": round(float(self.threshold), 6),
            "seed_precision_target": self.metadata.get("seed_precision_target"),
            "oof_precision_pct": self.metadata.get("oof_precision_pct"),
            "oof_recall_pct": self.metadata.get("oof_recall_pct"),
        }


def feature_vector(features: dict[str, float]) -> np.ndarray:
    """Apply the exact feature transform used by the frozen LOVO diagnostic."""
    return np.asarray(
        [
            float(features["detector_confidence"]),
            float(features["rank_score"]),
            float(features["confidence_rank_in_confirmed"]),
            float(features["confirmed_competitors"]),
            math.log1p(max(float(features["track_observations"]), 0.0)),
            math.log1p(max(float(features["track_streak"]), 0.0)),
            float(features["track_mean_confidence"]),
            math.log1p(max(float(features["track_area_cv"]), 0.0)),
            math.log1p(max(float(features["track_speed_px_per_step"]), 0.0)),
            math.log1p(max(float(features["box_area"]), 0.0)),
        ],
        dtype=np.float64,
    )


@dataclass
class _TrackFeatureHistory:
    observations: int = 0
    last_frame_id: int | None = None
    streak: int = 0
    confidences: list[float] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    last_center: np.ndarray | None = None
    last_observation_frame_id: int | None = None

    def update(
        self,
        *,
        frame_id: int,
        detector_confidence: float,
        box: np.ndarray,
    ) -> dict[str, float]:
        self.observations += 1
        self.streak = (
            self.streak + 1
            if self.last_frame_id is not None and int(frame_id) == self.last_frame_id + 1
            else 1
        )
        self.last_frame_id = int(frame_id)
        self.confidences.append(float(detector_confidence))
        area = _box_area(box)
        self.areas.append(area)
        center = _box_center(box)

        speed = 0.0
        if self.last_center is not None and self.last_observation_frame_id is not None:
            dt = max(int(frame_id) - int(self.last_observation_frame_id), 1)
            speed = float(np.linalg.norm(center - self.last_center) / float(dt))
        self.last_center = center
        self.last_observation_frame_id = int(frame_id)

        mean_confidence = float(np.mean(self.confidences))
        area_mean = float(np.mean(self.areas))
        area_std = float(np.std(self.areas)) if len(self.areas) > 1 else 0.0
        return {
            "track_observations": float(self.observations),
            "track_streak": float(self.streak),
            "track_mean_confidence": mean_confidence,
            "track_area_cv": float(area_std / max(area_mean, 1e-6)),
            "track_speed_px_per_step": speed,
            "box_area": area,
        }


class OCSortBallHypothesisTracker:
    """OC-SORT+CIoU hypothesis layer with safe CourtFlux lifecycle semantics.

    This layer deliberately does not assume that every confirmed OC-SORT track is the
    basketball. It keeps multiple hypotheses, exposes TENTATIVE/TRACKED/LOST/REMOVED
    state, and optionally scores observed TRACKED hypotheses with a frozen temporal
    identity model. LOST hypotheses are internal recovery state only and can never emit
    a concrete ball location.
    """

    def __init__(
        self,
        *,
        frame_rate: float = 30.0,
        identity_model_path: str | Path | None = None,
    ) -> None:
        self.frame_rate = max(float(frame_rate), 1.0)
        self.available = bool(TRACKERS_AVAILABLE)
        self.unavailable_reason = TRACKERS_IMPORT_ERROR
        self.tracker: Any | None = None
        self.identity_model: BallIdentitySeedModel | None = None
        self.identity_model_error: str | None = None
        self.histories: dict[int, _TrackFeatureHistory] = {}
        self._previous_internal_ids: set[int] = set()

        if self.available:
            self.tracker = OCSORTTracker(
                frame_rate=self.frame_rate,
                minimum_consecutive_frames=3,
                lost_track_buffer=8,
                minimum_iou_threshold=-0.3,
                direction_consistency_weight=0.2,
                high_conf_det_threshold=0.10,
                delta_t=3,
                iou=CIoU(),
            )

        if identity_model_path is not None and str(identity_model_path).strip():
            try:
                self.identity_model = BallIdentitySeedModel.load(identity_model_path)
            except Exception as exc:
                self.identity_model_error = f"{type(exc).__name__}: {exc}"

    @property
    def profile(self) -> dict[str, Any]:
        return {
            "tracker": "ocsort",
            "iou": "CIoU",
            "frame_rate": self.frame_rate,
            "minimum_consecutive_frames": 3,
            "lost_track_buffer": 8,
            "minimum_iou_threshold": -0.3,
            "direction_consistency_weight": 0.2,
            "high_conf_det_threshold": 0.10,
            "delta_t": 3,
        }

    def _detections(self, candidates: list[dict[str, Any]]):
        if sv is None:
            raise RuntimeError("supervision is unavailable")
        boxes: list[list[float]] = []
        confidences: list[float] = []
        source_indices: list[int] = []
        for index, candidate in enumerate(candidates):
            box = _box(candidate)
            if box is None:
                continue
            boxes.append(box)
            confidences.append(_candidate_detector_confidence(candidate))
            source_indices.append(index)

        if not boxes:
            detections = sv.Detections.empty()
            detections.confidence = np.asarray([], dtype=np.float32)
            detections.data["courtflux_candidate_index"] = np.asarray([], dtype=np.int32)
            return detections

        return sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            confidence=np.asarray(confidences, dtype=np.float32),
            data={
                "courtflux_candidate_index": np.asarray(source_indices, dtype=np.int32)
            },
        )

    def _identity_model_diagnostics(self) -> dict[str, Any]:
        if self.identity_model is not None:
            return self.identity_model.diagnostics()
        return {
            "loaded": False,
            "error": self.identity_model_error,
        }

    def update(
        self,
        candidates: list[dict[str, Any]],
        *,
        frame_id: int,
    ) -> dict[str, Any]:
        if not self.available or self.tracker is None:
            return {
                "enabled": True,
                "available": False,
                "error": self.unavailable_reason,
                "profile": self.profile,
                "identity_model": self._identity_model_diagnostics(),
                "hypotheses": [],
                "selected_seed": None,
                "lifecycle_counts": {
                    "TENTATIVE": 0,
                    "TRACKED": 0,
                    "LOST": 0,
                    "REMOVED": 0,
                },
            }

        detections = self._detections(candidates)
        tracked = self.tracker.update(detections)
        tids = (
            np.asarray(tracked.tracker_id, dtype=int)
            if tracked.tracker_id is not None
            else np.full(len(tracked), -1, dtype=int)
        )
        source_indices = tracked.data.get("courtflux_candidate_index")
        if source_indices is None:
            source_indices = np.full(len(tracked), -1, dtype=int)
        source_indices = np.asarray(source_indices, dtype=int)

        confirmed_indices = [index for index, tid in enumerate(tids.tolist()) if int(tid) >= 0]
        confirmed_confidences = []
        for index in confirmed_indices:
            source_index = int(source_indices[index]) if index < len(source_indices) else -1
            source = candidates[source_index] if 0 <= source_index < len(candidates) else {}
            confirmed_confidences.append(_candidate_detector_confidence(source))

        hypotheses: list[dict[str, Any]] = []
        observed_confirmed_ids: set[int] = set()
        seed_candidates: list[tuple[float, dict[str, Any]]] = []

        for index, tracker_id_value in enumerate(tids.tolist()):
            tracker_id = int(tracker_id_value)
            source_index = int(source_indices[index]) if index < len(source_indices) else -1
            source = candidates[source_index] if 0 <= source_index < len(candidates) else {}
            box = np.asarray(tracked.xyxy[index], dtype=np.float64)
            center = _box_center(box)
            detector_confidence = _candidate_detector_confidence(source)
            rank_score = _candidate_rank_score(source)

            if tracker_id < 0:
                hypotheses.append(
                    {
                        "tracker_id": None,
                        "hypothesis_id": f"tentative:{int(frame_id)}:{source_index}",
                        "lifecycle": "TENTATIVE",
                        "observed": True,
                        "candidate_index": source_index,
                        "box": [round(float(v), 4) for v in box.tolist()],
                        "center": [round(float(center[0]), 4), round(float(center[1]), 4)],
                        "detector_confidence": round(detector_confidence, 6),
                        "rank_score": round(rank_score, 6),
                        "identity_probability": None,
                        "identity_seed": False,
                        "concrete_output_allowed": False,
                    }
                )
                continue

            observed_confirmed_ids.add(tracker_id)
            history = self.histories.setdefault(tracker_id, _TrackFeatureHistory())
            history_features = history.update(
                frame_id=int(frame_id),
                detector_confidence=detector_confidence,
                box=box,
            )
            confidence_rank = 1 + sum(
                1 for value in confirmed_confidences if value > detector_confidence
            )
            features = {
                "detector_confidence": float(detector_confidence),
                "rank_score": float(rank_score),
                "confidence_rank_in_confirmed": float(confidence_rank),
                "confirmed_competitors": float(len(confirmed_indices)),
                **history_features,
            }
            probability = (
                None
                if self.identity_model is None
                else float(self.identity_model.score(features))
            )
            identity_seed = bool(
                probability is not None
                and self.identity_model is not None
                and probability >= self.identity_model.threshold
            )
            hypothesis = {
                "tracker_id": tracker_id,
                "hypothesis_id": f"ocsort:{tracker_id}",
                "lifecycle": "TRACKED",
                "observed": True,
                "candidate_index": source_index,
                "box": [round(float(v), 4) for v in box.tolist()],
                "center": [round(float(center[0]), 4), round(float(center[1]), 4)],
                "detector_confidence": round(detector_confidence, 6),
                "rank_score": round(rank_score, 6),
                "features": {key: round(float(value), 6) for key, value in features.items()},
                "identity_probability": (
                    None if probability is None else round(probability, 6)
                ),
                "identity_seed": identity_seed,
                "concrete_output_allowed": allows_concrete_output(
                    "TRACKED",
                    observed=True,
                    identity_seed=identity_seed,
                ),
            }
            hypotheses.append(hypothesis)
            if identity_seed and probability is not None and source_index >= 0:
                seed_candidates.append((probability, hypothesis))

        internal_confirmed_ids: set[int] = set()
        for track in list(getattr(self.tracker, "tracks", []) or []):
            tracker_id = int(getattr(track, "tracker_id", -1))
            if tracker_id < 0:
                continue
            internal_confirmed_ids.add(tracker_id)
            if tracker_id in observed_confirmed_ids:
                continue
            time_since_update = int(getattr(track, "time_since_update", 1))
            state = lifecycle_state(
                tracker_id=tracker_id,
                time_since_update=time_since_update,
            )
            if state != "LOST":
                continue
            box = np.asarray(track.get_state_bbox(), dtype=np.float64)
            center = _box_center(box)
            hypotheses.append(
                {
                    "tracker_id": tracker_id,
                    "hypothesis_id": f"ocsort:{tracker_id}",
                    "lifecycle": "LOST",
                    "observed": False,
                    "candidate_index": None,
                    "box": [round(float(v), 4) for v in box.tolist()],
                    "center": [round(float(center[0]), 4), round(float(center[1]), 4)],
                    "time_since_update": time_since_update,
                    "identity_probability": None,
                    "identity_seed": False,
                    "concrete_output_allowed": False,
                }
            )

        removed_ids = sorted(self._previous_internal_ids - internal_confirmed_ids)
        for tracker_id in removed_ids:
            hypotheses.append(
                {
                    "tracker_id": tracker_id,
                    "hypothesis_id": f"ocsort:{tracker_id}",
                    "lifecycle": "REMOVED",
                    "observed": False,
                    "candidate_index": None,
                    "identity_probability": None,
                    "identity_seed": False,
                    "concrete_output_allowed": False,
                }
            )
            self.histories.pop(tracker_id, None)
        self._previous_internal_ids = internal_confirmed_ids

        selected_seed = None
        if seed_candidates:
            probability, hypothesis = max(seed_candidates, key=lambda item: item[0])
            selected_seed = {
                "tracker_id": int(hypothesis["tracker_id"]),
                "hypothesis_id": hypothesis["hypothesis_id"],
                "candidate_index": int(hypothesis["candidate_index"]),
                "identity_probability": round(float(probability), 6),
            }

        lifecycle_counts = {
            state: sum(1 for item in hypotheses if item.get("lifecycle") == state)
            for state in ("TENTATIVE", "TRACKED", "LOST", "REMOVED")
        }
        return {
            "enabled": True,
            "available": True,
            "profile": self.profile,
            "identity_model": self._identity_model_diagnostics(),
            "hypotheses": hypotheses,
            "selected_seed": selected_seed,
            "lifecycle_counts": lifecycle_counts,
            "safety": {
                "lost_can_emit_concrete_output": False,
                "tentative_can_emit_concrete_output": False,
                "requires_identity_seed_for_output": True,
            },
        }
