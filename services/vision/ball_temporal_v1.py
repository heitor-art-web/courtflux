from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BallTemporalDecision:
    ball: dict[str, Any] | None
    source: str
    selected_index: int | None
    age_since_detection: int
    distance_norm: float | None
    predicted_center: tuple[float, float] | None
    velocity_px_per_frame: tuple[float, float]


class TemporalBallTrackerV1:
    """Conservative detector-to-track temporal bridge for the frozen CourtFlux ball detector.

    Scope is deliberately narrow:
      * associate direct detector observations across adjacent frames;
      * estimate smoothed image-space velocity;
      * bridge only short detector gaps with constant-velocity prediction;
      * return no ball after the short prediction budget expires.

    Direct observations are the only state updates. Predicted points never feed back into
    the track, preventing recursive drift. Step 8 adds a richer trajectory-consistency
    scorer on top of this contract; v1 itself uses only a distance gate plus confidence
    tie-breaking.
    """

    def __init__(
        self,
        *,
        max_prediction_gap: int = 4,
        reset_after_frames: int = 10,
        association_gate_norm: float = 0.12,
        gate_growth_per_frame: float = 0.02,
        max_gate_norm: float = 0.24,
        velocity_alpha: float = 0.45,
    ) -> None:
        self.max_prediction_gap = max(0, int(max_prediction_gap))
        self.reset_after_frames = max(self.max_prediction_gap + 1, int(reset_after_frames))
        self.association_gate_norm = max(0.005, float(association_gate_norm))
        self.gate_growth_per_frame = max(0.0, float(gate_growth_per_frame))
        self.max_gate_norm = max(self.association_gate_norm, float(max_gate_norm))
        self.velocity_alpha = max(0.0, min(float(velocity_alpha), 1.0))

        self.last_center: np.ndarray | None = None
        self.last_box: list[int] | None = None
        self.last_frame_id: int | None = None
        self.last_confidence: float | None = None
        self.velocity = np.zeros(2, dtype=np.float64)

    @staticmethod
    def _diagonal(frame_shape: tuple[int, ...]) -> float:
        height, width = frame_shape[:2]
        return max((float(width) ** 2 + float(height) ** 2) ** 0.5, 1.0)

    @staticmethod
    def _center(candidate: dict[str, Any]) -> np.ndarray:
        center = candidate.get("center")
        if center is None:
            box = candidate.get("box") or [0, 0, 0, 0]
            center = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
        return np.asarray(center, dtype=np.float64)

    def _gate(self, dt: int) -> float:
        return min(
            self.association_gate_norm + self.gate_growth_per_frame * max(int(dt) - 1, 0),
            self.max_gate_norm,
        )

    def predict_center(self, frame_id: int) -> np.ndarray | None:
        if self.last_center is None or self.last_frame_id is None:
            return None
        dt = max(int(frame_id) - int(self.last_frame_id), 0)
        return self.last_center + self.velocity * float(dt)

    def _reset(self) -> None:
        self.last_center = None
        self.last_box = None
        self.last_frame_id = None
        self.last_confidence = None
        self.velocity[:] = 0.0

    def _accept_direct(
        self,
        candidate: dict[str, Any],
        *,
        frame_id: int,
        selected_index: int,
        distance_norm: float | None,
        source: str,
    ) -> BallTemporalDecision:
        center = self._center(candidate)
        if self.last_center is not None and self.last_frame_id is not None:
            dt = int(frame_id) - int(self.last_frame_id)
            if 0 < dt <= self.reset_after_frames:
                measured = (center - self.last_center) / float(dt)
                alpha = self.velocity_alpha
                self.velocity = (1.0 - alpha) * self.velocity + alpha * measured
            else:
                self.velocity[:] = 0.0
        else:
            self.velocity[:] = 0.0

        self.last_center = center.copy()
        box = candidate.get("box") or [0, 0, 0, 0]
        self.last_box = [int(round(float(value))) for value in box]
        self.last_frame_id = int(frame_id)
        self.last_confidence = float(candidate.get("confidence", 0.0))

        ball = dict(candidate)
        ball["interpolated"] = False
        ball["source"] = source
        return BallTemporalDecision(
            ball=ball,
            source=source,
            selected_index=int(selected_index),
            age_since_detection=0,
            distance_norm=distance_norm,
            predicted_center=(float(center[0]), float(center[1])),
            velocity_px_per_frame=(float(self.velocity[0]), float(self.velocity[1])),
        )

    def _short_prediction(
        self,
        *,
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> BallTemporalDecision | None:
        if self.last_center is None or self.last_frame_id is None or self.last_box is None:
            return None
        age = int(frame_id) - int(self.last_frame_id)
        if age <= 0 or age > self.max_prediction_gap:
            return None
        predicted = self.predict_center(frame_id)
        if predicted is None:
            return None

        height, width = frame_shape[:2]
        if not (0.0 <= predicted[0] < width and 0.0 <= predicted[1] < height):
            return None

        old_box = self.last_box
        box_w = max(int(old_box[2] - old_box[0]), 2)
        box_h = max(int(old_box[3] - old_box[1]), 2)
        cx, cy = float(predicted[0]), float(predicted[1])
        x1 = int(round(cx - box_w / 2.0))
        y1 = int(round(cy - box_h / 2.0))
        x1 = max(0, min(x1, max(width - box_w, 0)))
        y1 = max(0, min(y1, max(height - box_h, 0)))
        x2 = min(width - 1, x1 + box_w)
        y2 = min(height - 1, y1 + box_h)
        confidence = float(self.last_confidence or 0.0) * (0.80 ** age)
        ball = {
            "box": [x1, y1, x2, y2],
            "center": (cx, cy),
            "confidence": confidence,
            "interpolated": True,
            "source": "predicted_short_gap",
            "age_since_detection": age,
        }
        return BallTemporalDecision(
            ball=ball,
            source="predicted_short_gap",
            selected_index=None,
            age_since_detection=age,
            distance_norm=None,
            predicted_center=(cx, cy),
            velocity_px_per_frame=(float(self.velocity[0]), float(self.velocity[1])),
        )

    def update(
        self,
        candidates: list[dict[str, Any]],
        *,
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> BallTemporalDecision:
        candidates = list(candidates or [])
        if self.last_frame_id is not None:
            age = int(frame_id) - int(self.last_frame_id)
            if age > self.max_prediction_gap:
                # Crossing the short-gap budget always emits an explicit missing boundary.
                # The old state is discarded before any candidate can be considered; a
                # later frame may seed a new track, but it cannot reacquire the expired one.
                expired_age = max(age, 0)
                self._reset()
                return BallTemporalDecision(
                    ball=None,
                    source="missing",
                    selected_index=None,
                    age_since_detection=expired_age,
                    distance_norm=None,
                    predicted_center=None,
                    velocity_px_per_frame=(0.0, 0.0),
                )

        if self.last_center is None or self.last_frame_id is None:
            if candidates:
                index = max(
                    range(len(candidates)),
                    key=lambda idx: float(candidates[idx].get("confidence", 0.0)),
                )
                return self._accept_direct(
                    candidates[index],
                    frame_id=frame_id,
                    selected_index=index,
                    distance_norm=None,
                    source="detector_seed",
                )
            return BallTemporalDecision(
                ball=None,
                source="missing",
                selected_index=None,
                age_since_detection=0,
                distance_norm=None,
                predicted_center=None,
                velocity_px_per_frame=(0.0, 0.0),
            )

        predicted = self.predict_center(frame_id)
        assert predicted is not None
        diagonal = self._diagonal(frame_shape)
        dt = max(int(frame_id) - int(self.last_frame_id), 1)
        gate = self._gate(dt)

        plausible: list[tuple[float, float, int]] = []
        for index, candidate in enumerate(candidates):
            distance_norm = float(np.linalg.norm(self._center(candidate) - predicted) / diagonal)
            if distance_norm <= gate:
                confidence = float(candidate.get("confidence", 0.0))
                plausible.append((distance_norm, -confidence, index))

        if plausible:
            plausible.sort()
            distance_norm, _, index = plausible[0]
            return self._accept_direct(
                candidates[index],
                frame_id=frame_id,
                selected_index=index,
                distance_norm=distance_norm,
                source="detector_temporal_association",
            )

        prediction = self._short_prediction(frame_id=frame_id, frame_shape=frame_shape)
        if prediction is not None:
            return prediction

        age = max(int(frame_id) - int(self.last_frame_id), 0)
        return BallTemporalDecision(
            ball=None,
            source="missing",
            selected_index=None,
            age_since_detection=age,
            distance_norm=None,
            predicted_center=(float(predicted[0]), float(predicted[1])),
            velocity_px_per_frame=(float(self.velocity[0]), float(self.velocity[1])),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "temporal_ball_tracker_v1",
            "detector_policy": "frozen_external_observations",
            "max_prediction_gap": self.max_prediction_gap,
            "reset_after_frames": self.reset_after_frames,
            "association_gate_norm": self.association_gate_norm,
            "gate_growth_per_frame": self.gate_growth_per_frame,
            "max_gate_norm": self.max_gate_norm,
            "velocity_alpha": self.velocity_alpha,
            "prediction_updates_state": False,
            "reacquisition": False,
            "trajectory_consistency": False,
        }
