from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BallTrajectoryScore:
    score: float
    plausible: bool
    detector_confidence: float
    position_consistency: float
    velocity_consistency: float
    acceleration_consistency: float
    size_consistency: float
    distance_norm: float
    speed_residual_norm: float
    acceleration_norm: float


class BallTrajectoryConsistencyV1:
    """Score detector candidates against the active image-space ball trajectory.

    This layer does not predict or own track state. It consumes the temporal track state
    from TemporalBallTrackerV1 and ranks direct detector observations by physical/temporal
    consistency. All quantities are normalized by the frame diagonal so the thresholds
    are resolution-independent.
    """

    def __init__(
        self,
        *,
        confidence_weight: float = 0.18,
        position_weight: float = 0.34,
        velocity_weight: float = 0.24,
        acceleration_weight: float = 0.16,
        size_weight: float = 0.08,
        max_distance_norm: float = 0.16,
        max_speed_residual_norm: float = 0.12,
        max_acceleration_norm: float = 0.14,
    ) -> None:
        weights = np.asarray(
            [
                confidence_weight,
                position_weight,
                velocity_weight,
                acceleration_weight,
                size_weight,
            ],
            dtype=np.float64,
        )
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("At least one trajectory score weight must be positive")
        weights /= total
        (
            self.confidence_weight,
            self.position_weight,
            self.velocity_weight,
            self.acceleration_weight,
            self.size_weight,
        ) = [float(value) for value in weights]
        self.max_distance_norm = max(0.001, float(max_distance_norm))
        self.max_speed_residual_norm = max(0.001, float(max_speed_residual_norm))
        self.max_acceleration_norm = max(0.001, float(max_acceleration_norm))

    @staticmethod
    def _center(candidate: dict[str, Any]) -> np.ndarray:
        center = candidate.get("center")
        if center is None:
            box = candidate.get("box") or [0, 0, 0, 0]
            center = (
                (float(box[0]) + float(box[2])) / 2.0,
                (float(box[1]) + float(box[3])) / 2.0,
            )
        return np.asarray(center, dtype=np.float64)

    @staticmethod
    def _area(candidate: dict[str, Any]) -> float:
        box = candidate.get("box") or [0, 0, 0, 0]
        return max(float(box[2]) - float(box[0]), 1.0) * max(
            float(box[3]) - float(box[1]), 1.0
        )

    @staticmethod
    def _size_consistency(candidate_area: float, last_area: float | None) -> float:
        if last_area is None or last_area <= 0.0:
            return 0.5
        ratio = max(candidate_area / float(last_area), 1e-6)
        return float(exp(-abs(log(ratio))))

    def score_candidate(
        self,
        candidate: dict[str, Any],
        *,
        last_center: np.ndarray,
        predicted_center: np.ndarray,
        current_velocity: np.ndarray,
        last_area: float | None,
        dt: int,
        frame_diagonal: float,
    ) -> BallTrajectoryScore:
        dt = max(int(dt), 1)
        diagonal = max(float(frame_diagonal), 1.0)
        center = self._center(candidate)
        confidence = max(0.0, min(float(candidate.get("confidence", 0.0)), 1.0))

        distance_norm = float(np.linalg.norm(center - predicted_center) / diagonal)
        position_consistency = float(exp(-distance_norm / self.max_distance_norm))

        measured_velocity = (center - last_center) / float(dt)
        velocity_delta = measured_velocity - current_velocity
        speed_residual_norm = float(np.linalg.norm(velocity_delta) / diagonal)
        velocity_consistency = float(exp(-speed_residual_norm / self.max_speed_residual_norm))

        # Acceleration is the change from the active velocity estimate to the new measured
        # velocity. Comparing against an older pre-update velocity would penalize a truly
        # constant trajectory for historical acceleration that has already been absorbed.
        measured_acceleration = velocity_delta / float(dt)
        acceleration_norm = float(np.linalg.norm(measured_acceleration) / diagonal)
        acceleration_consistency = float(exp(-acceleration_norm / self.max_acceleration_norm))

        size_consistency = self._size_consistency(self._area(candidate), last_area)
        score = (
            self.confidence_weight * confidence
            + self.position_weight * position_consistency
            + self.velocity_weight * velocity_consistency
            + self.acceleration_weight * acceleration_consistency
            + self.size_weight * size_consistency
        )
        plausible = (
            distance_norm <= self.max_distance_norm
            and speed_residual_norm <= self.max_speed_residual_norm
            and acceleration_norm <= self.max_acceleration_norm
        )
        return BallTrajectoryScore(
            score=float(score),
            plausible=bool(plausible),
            detector_confidence=confidence,
            position_consistency=position_consistency,
            velocity_consistency=velocity_consistency,
            acceleration_consistency=acceleration_consistency,
            size_consistency=size_consistency,
            distance_norm=distance_norm,
            speed_residual_norm=speed_residual_norm,
            acceleration_norm=acceleration_norm,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "ball_trajectory_consistency_v1",
            "weights": {
                "detector_confidence": self.confidence_weight,
                "position": self.position_weight,
                "velocity": self.velocity_weight,
                "acceleration": self.acceleration_weight,
                "size": self.size_weight,
            },
            "gates": {
                "max_distance_norm": self.max_distance_norm,
                "max_speed_residual_norm": self.max_speed_residual_norm,
                "max_acceleration_norm": self.max_acceleration_norm,
            },
            "normalization": "frame_diagonal",
            "owns_track_state": False,
        }
