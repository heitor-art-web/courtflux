from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


class AirborneClassifierV1:
    """Causal, trainable PASS/SHOT classifier for an active airborne hypothesis.

    The feature contract is FPS-normalized where time is involved so models trained on
    mixed frame-rate clips can generalize by physical duration rather than raw frame count.
    No future frame is used. Without a trained JSON artifact the classifier only emits
    features and returns ``unresolved``; the caller may retain its existing fallback.
    """

    VERSION = "courtflux_airborne_classifier_v1"
    CLASSES = ("unresolved", "pass_flight", "shot_flight")
    FEATURE_NAMES = (
        "flight_supported",
        "flight_duration_sec",
        "flight_direct_observation_fraction",
        "flight_displacement_player_heights",
        "receiver_evidence",
        "raw_best_distinct_from_origin",
        "raw_best_score",
        "candidate_score",
        "shot_active",
        "shot_preflight_available",
        "shot_preflight_direct_observed",
        "shot_preflight_near_x",
        "shot_preflight_above",
        "shot_preflight_ball_to_rim_distance_norm",
        "shot_preflight_horizontal_distance_rim_widths",
        "shot_preflight_relative_y_rim_heights",
        "shot_preflight_distance_velocity_norm_per_sec",
        "shot_preflight_vertical_velocity_heights_per_sec",
        "shot_preflight_approach_duration_sec",
        "shot_preflight_rising_duration_sec",
    )

    def __init__(self, model_path: str | Path | None = None) -> None:
        env_path = os.getenv("P2_AIRBORNE_MODEL", "").strip()
        self.model_path = Path(model_path or env_path) if (model_path or env_path) else None
        self.model: dict[str, Any] | None = None
        self._last_preflight_frame: int | None = None
        self._last_distance_norm: float | None = None
        self._last_relative_y: float | None = None
        self._approach_streak = 0
        self._rising_streak = 0
        if self.model_path is not None and self.model_path.exists():
            self.load(self.model_path)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self._last_preflight_frame = None
        self._last_distance_norm = None
        self._last_relative_y = None
        self._approach_streak = 0
        self._rising_streak = 0

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

    @classmethod
    def _fps(cls, frame_state: Any) -> float:
        return max(1.0, cls._float(getattr(frame_state, "fps", 30.0), 30.0))

    @staticmethod
    def _ball_position(frame_state: Any) -> tuple[float, float] | None:
        ball = getattr(frame_state, "ball_state", None)
        if ball is None or not bool(getattr(ball, "available", False)):
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
    def _direct_observed(frame_state: Any) -> bool:
        ball = getattr(frame_state, "ball_state", None)
        if ball is None:
            return False
        status = getattr(ball, "status", None)
        status_name = str(getattr(status, "name", status) or "").strip().lower()
        return status_name == "observed" and AirborneClassifierV1._ball_position(frame_state) is not None

    @staticmethod
    def _rims(frame_state: Any) -> list[tuple[float, float, float, float]]:
        court_state = getattr(frame_state, "court_state", None)
        if court_state is None:
            return []
        result: list[tuple[float, float, float, float]] = []
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
                result.append((x1, y1, x2, y2))
        return result

    def _preflight(self, frame_state: Any, frame_id: int, fps: float) -> dict[str, float]:
        position = self._ball_position(frame_state)
        rims = self._rims(frame_state)
        direct = self._direct_observed(frame_state)
        if position is None or not rims:
            self.reset()
            return {
                "shot_preflight_available": 0.0,
                "shot_preflight_direct_observed": 1.0 if direct else 0.0,
                "shot_preflight_near_x": 0.0,
                "shot_preflight_above": 0.0,
                "shot_preflight_ball_to_rim_distance_norm": 0.0,
                "shot_preflight_horizontal_distance_rim_widths": 0.0,
                "shot_preflight_relative_y_rim_heights": 0.0,
                "shot_preflight_distance_velocity_norm_per_sec": 0.0,
                "shot_preflight_vertical_velocity_heights_per_sec": 0.0,
                "shot_preflight_approach_duration_sec": 0.0,
                "shot_preflight_rising_duration_sec": 0.0,
            }

        bx, by = position
        rim = min(
            rims,
            key=lambda candidate: (
                ((candidate[0] + candidate[2]) / 2.0 - bx) ** 2
                + ((candidate[1] + candidate[3]) / 2.0 - by) ** 2
            ),
        )
        x1, y1, x2, y2 = rim
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        rim_x = (x1 + x2) / 2.0
        rim_y = (y1 + y2) / 2.0
        dx = bx - rim_x
        dy = by - rim_y
        horizontal = abs(dx) / width
        relative_y = dy / height
        distance = math.hypot(dx / width, dy / height)
        near_x = horizontal <= 2.5
        above = relative_y < -0.20

        consecutive = bool(
            direct
            and self._last_preflight_frame is not None
            and frame_id == self._last_preflight_frame + 1
            and self._last_distance_norm is not None
            and self._last_relative_y is not None
        )
        distance_velocity = 0.0
        vertical_velocity = 0.0
        if consecutive:
            delta_distance = distance - float(self._last_distance_norm)
            delta_relative_y = relative_y - float(self._last_relative_y)
            distance_velocity = delta_distance * fps
            vertical_velocity = delta_relative_y * fps
            self._approach_streak = self._approach_streak + 1 if delta_distance < 0.0 else 0
            self._rising_streak = self._rising_streak + 1 if delta_relative_y < 0.0 else 0
        else:
            self._approach_streak = 0
            self._rising_streak = 0

        if direct:
            self._last_preflight_frame = frame_id
            self._last_distance_norm = distance
            self._last_relative_y = relative_y
        else:
            self.reset()

        return {
            "shot_preflight_available": 1.0,
            "shot_preflight_direct_observed": 1.0 if direct else 0.0,
            "shot_preflight_near_x": 1.0 if near_x else 0.0,
            "shot_preflight_above": 1.0 if above else 0.0,
            "shot_preflight_ball_to_rim_distance_norm": distance,
            "shot_preflight_horizontal_distance_rim_widths": horizontal,
            "shot_preflight_relative_y_rim_heights": relative_y,
            "shot_preflight_distance_velocity_norm_per_sec": distance_velocity,
            "shot_preflight_vertical_velocity_heights_per_sec": vertical_velocity,
            "shot_preflight_approach_duration_sec": float(self._approach_streak) / fps,
            "shot_preflight_rising_duration_sec": float(self._rising_streak) / fps,
        }

    def features(self, *, frame_state: Any, p2: dict[str, Any]) -> dict[str, float]:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        fps = self._fps(frame_state)
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        shot = dict(p2.get("shot", {}) or {})
        flight_frames = max(0.0, self._float(flight.get("flight_frames"), 0.0))
        direct_observations = max(
            0.0, self._float(flight.get("flight_direct_observations"), 0.0)
        )
        values = {
            "flight_supported": 1.0 if bool(flight.get("flight_supported", False)) else 0.0,
            "flight_duration_sec": flight_frames / fps,
            "flight_direct_observation_fraction": (
                direct_observations / max(flight_frames, 1.0)
            ),
            "flight_displacement_player_heights": self._float(
                flight.get("flight_displacement_player_heights"), 0.0
            ),
            "receiver_evidence": 1.0
            if bool(possession.get("pass_flight_receiver_evidence", False))
            else 0.0,
            "raw_best_distinct_from_origin": 1.0
            if bool(possession.get("raw_best_distinct_from_flight_origin", False))
            else 0.0,
            "raw_best_score": self._float(possession.get("raw_best_score"), 0.0),
            "candidate_score": self._float(possession.get("candidate_score"), 0.0),
            "shot_active": 1.0 if bool(shot.get("active", False)) else 0.0,
            **self._preflight(frame_state, frame_id, fps),
        }
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
