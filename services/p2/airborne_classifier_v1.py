from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from core.ball_state import BallStateStatus


class AirborneClassifierV1:
    """Causal, trainable PASS/SHOT classifier for an already-airborne hypothesis.

    The classifier is deliberately separate from P2 event generation. It consumes only the
    current frame plus temporal feature memory from earlier frames and returns one of:

    - ``unresolved``
    - ``pass_flight``
    - ``shot_flight``

    A JSON softmax-linear model is optional. Without a model, features are still emitted for
    development-data collection while the caller may retain its existing fallback policy.
    """

    VERSION = "courtflux_airborne_classifier_v1"
    CLASSES = ("unresolved", "pass_flight", "shot_flight")
    FEATURE_NAMES = (
        "flight_supported",
        "flight_frames",
        "flight_direct_observations",
        "flight_displacement_player_heights",
        "receiver_evidence",
        "receiver_thresholded",
        "raw_best_distinct_from_origin",
        "raw_best_score",
        "candidate_score",
        "receiver_same_track_streak",
        "receiver_approach_delta_heights",
        "receiver_motion_cosine",
        "shot_active",
        "semantic_shot_signal",
        "rim_available",
        "rim_above",
        "rim_distance_norm",
        "rim_approach_delta_norm",
        "rim_rising_delta_heights",
        "direct_ball_observed",
        "recent_owner_age_frames",
    )

    def __init__(self, model_path: str | Path | None = None) -> None:
        env_path = os.getenv("P2_AIRBORNE_MODEL", "").strip()
        self.model_path = Path(model_path or env_path) if (model_path or env_path) else None
        self.model: dict[str, Any] | None = None
        self._previous_direct_ball: tuple[float, float] | None = None
        self._previous_direct_frame: int | None = None
        self._previous_receiver_track_id: int | None = None
        self._receiver_streak = 0
        self._previous_rim_distance_norm: float | None = None
        self._previous_rim_relative_y: float | None = None
        self._previous_rim_center: tuple[float, float] | None = None
        self._previous_rim_frame: int | None = None
        if self.model_path is not None and self.model_path.exists():
            self.load(self.model_path)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self._previous_direct_ball = None
        self._previous_direct_frame = None
        self._previous_receiver_track_id = None
        self._receiver_streak = 0
        self._previous_rim_distance_norm = None
        self._previous_rim_relative_y = None
        self._previous_rim_center = None
        self._previous_rim_frame = None

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("version") or "") != self.VERSION:
            raise ValueError(f"airborne model version must be {self.VERSION}")
        if tuple(payload.get("classes", [])) != self.CLASSES:
            raise ValueError("airborne model classes do not match runtime contract")
        if tuple(payload.get("features", [])) != self.FEATURE_NAMES:
            raise ValueError("airborne model feature order does not match runtime contract")
        n_features = len(self.FEATURE_NAMES)
        means = list(payload.get("feature_mean", []))
        scales = list(payload.get("feature_scale", []))
        weights = list(payload.get("weights", []))
        bias = list(payload.get("bias", []))
        if len(means) != n_features or len(scales) != n_features:
            raise ValueError("airborne model normalization dimensions are invalid")
        if len(weights) != len(self.CLASSES) or any(len(row) != n_features for row in weights):
            raise ValueError("airborne model weight dimensions are invalid")
        if len(bias) != len(self.CLASSES):
            raise ValueError("airborne model bias dimensions are invalid")
        self.model = payload
        self.model_path = Path(path)

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @staticmethod
    def _int(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ball_position(frame_state: Any, *, direct_only: bool = False) -> tuple[float, float] | None:
        ball = getattr(frame_state, "ball_state", None)
        if ball is None or not bool(getattr(ball, "available", False)):
            return None
        if direct_only and getattr(ball, "status", None) is not BallStateStatus.OBSERVED:
            return None
        position_fn = getattr(ball, "position", None)
        if not callable(position_fn):
            return None
        position = position_fn()
        if position is None:
            return None
        try:
            return float(position[0]), float(position[1])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _player_geometry(frame_state: Any, track_id: int | None) -> tuple[tuple[float, float], float] | None:
        if track_id is None:
            return None
        for player in list(getattr(frame_state, "player_states", []) or []):
            if getattr(player, "track_id", None) != track_id:
                continue
            bbox = getattr(player, "bbox", None)
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return None
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                return None
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), max(1.0, y2 - y1)
        return None

    @staticmethod
    def _semantic_shot_signal(frame_state: Any) -> bool:
        diagnostics = dict(getattr(frame_state, "diagnostics", {}) or {})
        detector = dict(diagnostics.get("detector", {}) or {})
        rows: list[dict[str, Any]] = []
        for key in ("p1_detections", "semantic_evidence", "semantic_detections"):
            value = detector.get(key)
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
        for row in rows:
            confidence = AirborneClassifierV1._float(row.get("confidence"), 0.0)
            if confidence < 0.35:
                continue
            label = str(
                row.get("class_name")
                or row.get("label")
                or row.get("kind")
                or row.get("class")
                or ""
            ).strip().lower().replace("_", "-")
            if "player-jump-shot" in label or "player-layup-dunk" in label:
                return True
        return False

    @staticmethod
    def _rim_boxes(frame_state: Any) -> list[tuple[float, float, float, float]]:
        court_state = getattr(frame_state, "court_state", None)
        if court_state is None:
            return []
        rims: list[tuple[float, float, float, float]] = []
        for evidence in list(getattr(court_state, "basket_evidence", None) or []):
            if not isinstance(evidence, dict) or str(evidence.get("kind") or "").lower() != "rim":
                continue
            raw = evidence.get("bbox_px", evidence.get("box", evidence.get("bbox")))
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in raw[:4]]
            except (TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                rims.append((x1, y1, x2, y2))
        return rims

    def _rim_geometry(
        self, frame_state: Any, frame_id: int, ball_position: tuple[float, float] | None
    ) -> dict[str, float]:
        rims = self._rim_boxes(frame_state)
        if ball_position is None or not rims:
            self._previous_rim_distance_norm = None
            self._previous_rim_relative_y = None
            self._previous_rim_center = None
            self._previous_rim_frame = None
            return {
                "rim_available": 0.0,
                "rim_above": 0.0,
                "rim_distance_norm": 0.0,
                "rim_approach_delta_norm": 0.0,
                "rim_rising_delta_heights": 0.0,
            }

        bx, by = ball_position
        if self._previous_rim_center is not None:
            px, py = self._previous_rim_center
            rim = min(
                rims,
                key=lambda candidate: (((candidate[0] + candidate[2]) / 2.0 - px) ** 2 + ((candidate[1] + candidate[3]) / 2.0 - py) ** 2),
            )
        else:
            rim = min(
                rims,
                key=lambda candidate: (((candidate[0] + candidate[2]) / 2.0 - bx) ** 2 + ((candidate[1] + candidate[3]) / 2.0 - by) ** 2),
            )

        x1, y1, x2, y2 = rim
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        rx = (x1 + x2) / 2.0
        ry = (y1 + y2) / 2.0
        relative_x = (bx - rx) / width
        relative_y = (by - ry) / height
        distance = math.hypot(relative_x, relative_y)
        consecutive = bool(
            self._previous_rim_frame is not None
            and frame_id == self._previous_rim_frame + 1
            and self._previous_rim_distance_norm is not None
            and self._previous_rim_relative_y is not None
        )
        approach = 0.0 if not consecutive else self._previous_rim_distance_norm - distance
        rising = 0.0 if not consecutive else self._previous_rim_relative_y - relative_y
        self._previous_rim_distance_norm = distance
        self._previous_rim_relative_y = relative_y
        self._previous_rim_center = (rx, ry)
        self._previous_rim_frame = frame_id
        return {
            "rim_available": 1.0,
            "rim_above": 1.0 if relative_y < -0.20 else 0.0,
            "rim_distance_norm": distance,
            "rim_approach_delta_norm": approach,
            "rim_rising_delta_heights": rising,
        }

    def features(self, *, frame_state: Any, p2: dict[str, Any]) -> dict[str, float]:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        shot = dict(p2.get("shot", {}) or {})
        origin_track = self._int(flight.get("origin_track_id"))
        candidate_track = self._int(possession.get("candidate_track_id"))
        raw_track = self._int(possession.get("raw_best_track_id"))
        receiver_track = candidate_track
        receiver_thresholded = candidate_track is not None
        if receiver_track is None and raw_track is not None and raw_track != origin_track:
            receiver_track = raw_track

        if receiver_track is not None and receiver_track == self._previous_receiver_track_id:
            self._receiver_streak += 1
        elif receiver_track is not None:
            self._receiver_streak = 1
        else:
            self._receiver_streak = 0
        self._previous_receiver_track_id = receiver_track

        direct_ball = self._ball_position(frame_state, direct_only=True)
        receiver_geometry = self._player_geometry(frame_state, receiver_track)
        approach_heights = 0.0
        motion_cosine = 0.0
        if (
            direct_ball is not None
            and self._previous_direct_ball is not None
            and self._previous_direct_frame is not None
            and frame_id == self._previous_direct_frame + 1
            and receiver_geometry is not None
        ):
            (rx, ry), receiver_height = receiver_geometry
            px, py = self._previous_direct_ball
            cx, cy = direct_ball
            previous_distance = math.hypot(rx - px, ry - py)
            current_distance = math.hypot(rx - cx, ry - cy)
            approach_heights = (previous_distance - current_distance) / max(receiver_height, 1.0)
            move_x, move_y = cx - px, cy - py
            target_x, target_y = rx - px, ry - py
            move_norm = math.hypot(move_x, move_y)
            target_norm = math.hypot(target_x, target_y)
            if move_norm > 1e-6 and target_norm > 1e-6:
                motion_cosine = (move_x * target_x + move_y * target_y) / (move_norm * target_norm)

        ball_for_rim = self._ball_position(frame_state, direct_only=False)
        rim = self._rim_geometry(frame_state, frame_id, ball_for_rim)

        recent_frame = self._int(possession.get("recent_owner_frame"))
        recent_owner_age = 0.0 if recent_frame is None else float(max(0, frame_id - recent_frame))

        values = {
            "flight_supported": 1.0 if bool(flight.get("flight_supported", False)) else 0.0,
            "flight_frames": self._float(flight.get("flight_frames"), 0.0),
            "flight_direct_observations": self._float(flight.get("flight_direct_observations"), 0.0),
            "flight_displacement_player_heights": self._float(flight.get("flight_displacement_player_heights"), 0.0),
            "receiver_evidence": 1.0 if bool(possession.get("pass_flight_receiver_evidence", False)) else 0.0,
            "receiver_thresholded": 1.0 if receiver_thresholded else 0.0,
            "raw_best_distinct_from_origin": 1.0 if bool(possession.get("raw_best_distinct_from_flight_origin", False)) else 0.0,
            "raw_best_score": self._float(possession.get("raw_best_score"), 0.0),
            "candidate_score": self._float(possession.get("candidate_score"), 0.0),
            "receiver_same_track_streak": float(self._receiver_streak),
            "receiver_approach_delta_heights": approach_heights,
            "receiver_motion_cosine": motion_cosine,
            "shot_active": 1.0 if bool(shot.get("active", False)) else 0.0,
            "semantic_shot_signal": 1.0 if self._semantic_shot_signal(frame_state) else 0.0,
            "rim_available": rim["rim_available"],
            "rim_above": rim["rim_above"],
            "rim_distance_norm": rim["rim_distance_norm"],
            "rim_approach_delta_norm": rim["rim_approach_delta_norm"],
            "rim_rising_delta_heights": rim["rim_rising_delta_heights"],
            "direct_ball_observed": 1.0 if direct_ball is not None else 0.0,
            "recent_owner_age_frames": recent_owner_age,
        }

        if direct_ball is not None:
            self._previous_direct_ball = direct_ball
            self._previous_direct_frame = frame_id
        else:
            self._previous_direct_ball = None
            self._previous_direct_frame = None

        return {name: float(values[name]) for name in self.FEATURE_NAMES}

    def _probabilities(self, features: dict[str, float]) -> dict[str, float] | None:
        if self.model is None:
            return None
        means = [float(value) for value in self.model["feature_mean"]]
        scales = [max(abs(float(value)), 1e-6) for value in self.model["feature_scale"]]
        vector = [
            (float(features[name]) - means[index]) / scales[index]
            for index, name in enumerate(self.FEATURE_NAMES)
        ]
        logits: list[float] = []
        for row, bias in zip(self.model["weights"], self.model["bias"]):
            logits.append(sum(float(w) * x for w, x in zip(row, vector)) + float(bias))
        maximum = max(logits)
        exps = [math.exp(value - maximum) for value in logits]
        total = sum(exps)
        return {
            label: exps[index] / max(total, 1e-12)
            for index, label in enumerate(self.CLASSES)
        }

    def classify(self, *, frame_state: Any, p2: dict[str, Any]) -> dict[str, Any]:
        features = self.features(frame_state=frame_state, p2=p2)
        probabilities = self._probabilities(features)
        flight = dict(dict(p2.get("possession", {}) or {}).get("flight", {}) or {})
        flight_active = bool(flight.get("active", False))
        if probabilities is None or not flight_active:
            label = "unresolved"
            confidence = 0.0
        else:
            best_label = max(self.CLASSES, key=lambda key: probabilities[key])
            best_probability = float(probabilities[best_label])
            minimum = self._float(self.model.get("minimum_probability"), 0.55)
            label = best_label if best_probability >= minimum else "unresolved"
            confidence = best_probability
        return {
            "version": self.VERSION,
            "model_loaded": self.model_loaded,
            "model_path": None if self.model_path is None else str(self.model_path),
            "flight_active": flight_active,
            "label": label,
            "confidence": round(confidence, 6),
            "probabilities": None
            if probabilities is None
            else {key: round(float(value), 6) for key, value in probabilities.items()},
            "features": {key: round(float(value), 6) for key, value in features.items()},
        }
