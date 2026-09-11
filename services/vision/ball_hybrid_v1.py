from __future__ import annotations

from typing import Any

from services.vision.ball_temporal_v1 import BallTemporalDecision, TemporalBallTrackerV1
from services.vision.ball_trajectory_consistency import (
    BallTrajectoryConsistencyV1,
    BallTrajectoryScore,
)


class TrajectoryAwareBallTrackerV1(TemporalBallTrackerV1):
    """TemporalBallTrackerV1 with explicit trajectory-consistency candidate ranking.

    The optional override-policy controls let evaluation distinguish trajectory evidence
    from promotion policy. Defaults preserve the original v1 behavior.
    """

    def __init__(
        self,
        *args,
        trajectory: BallTrajectoryConsistencyV1 | None = None,
        min_trajectory_score: float = 0.0,
        protect_direct_confidence: float | None = None,
        override_score_margin: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trajectory = trajectory or BallTrajectoryConsistencyV1()
        self.min_trajectory_score = max(0.0, min(float(min_trajectory_score), 1.0))
        self.protect_direct_confidence = (
            None
            if protect_direct_confidence is None
            else max(0.0, min(float(protect_direct_confidence), 1.0))
        )
        self.override_score_margin = max(0.0, float(override_score_margin))
        self.last_area: float | None = None

    @staticmethod
    def _area(candidate: dict[str, Any]) -> float:
        box = candidate.get("box") or [0, 0, 0, 0]
        return max(float(box[2]) - float(box[0]), 1.0) * max(float(box[3]) - float(box[1]), 1.0)

    def _reset(self) -> None:
        super()._reset()
        self.last_area = None

    def _accept_trajectory_direct(
        self,
        candidate: dict[str, Any],
        *,
        frame_id: int,
        selected_index: int,
        details: BallTrajectoryScore | None,
        source: str,
    ) -> BallTemporalDecision:
        decision = super()._accept_direct(
            candidate,
            frame_id=frame_id,
            selected_index=selected_index,
            distance_norm=None if details is None else details.distance_norm,
            source=source,
        )
        self.last_area = self._area(candidate)
        if decision.ball is not None and details is not None:
            decision.ball["trajectory_score"] = round(details.score, 6)
            decision.ball["trajectory_consistency"] = {
                "position": round(details.position_consistency, 6),
                "velocity": round(details.velocity_consistency, 6),
                "acceleration": round(details.acceleration_consistency, 6),
                "size": round(details.size_consistency, 6),
                "distance_norm": round(details.distance_norm, 6),
                "speed_residual_norm": round(details.speed_residual_norm, 6),
                "acceleration_norm": round(details.acceleration_norm, 6),
            }
        return decision

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
            if not candidates:
                return super().update([], frame_id=frame_id, frame_shape=frame_shape)
            index = max(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("confidence", 0.0)),
            )
            return self._accept_trajectory_direct(
                candidates[index],
                frame_id=frame_id,
                selected_index=index,
                details=None,
                source="detector_seed",
            )

        predicted = self.predict_center(frame_id)
        assert predicted is not None
        diagonal = self._diagonal(frame_shape)
        dt = max(int(frame_id) - int(self.last_frame_id), 1)
        assert self.last_center is not None

        scored: list[tuple[float, int, BallTrajectoryScore]] = []
        details_by_index: dict[int, BallTrajectoryScore] = {}
        for index, candidate in enumerate(candidates):
            details = self.trajectory.score_candidate(
                candidate,
                last_center=self.last_center,
                predicted_center=predicted,
                current_velocity=self.velocity,
                last_area=self.last_area,
                dt=dt,
                frame_diagonal=diagonal,
            )
            details_by_index[index] = details
            if details.plausible and details.score >= self.min_trajectory_score:
                scored.append((details.score, index, details))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            _, best_index, best_details = scored[0]
            direct_index = max(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("confidence", 0.0)),
            )
            direct_details = details_by_index[direct_index]
            direct_is_eligible = (
                direct_details.plausible
                and direct_details.score >= self.min_trajectory_score
            )
            direct_confidence = float(candidates[direct_index].get("confidence", 0.0))

            selected_index = best_index
            selected_details = best_details
            source = "detector_trajectory_consistent"

            if best_index != direct_index and direct_is_eligible:
                protected = (
                    self.protect_direct_confidence is not None
                    and direct_confidence >= self.protect_direct_confidence
                )
                margin_met = best_details.score >= (
                    direct_details.score + self.override_score_margin
                )
                if protected or not margin_met:
                    selected_index = direct_index
                    selected_details = direct_details
                    source = "detector_direct_protected"

            return self._accept_trajectory_direct(
                candidates[selected_index],
                frame_id=frame_id,
                selected_index=selected_index,
                details=selected_details,
                source=source,
            )

        return super().update([], frame_id=frame_id, frame_shape=frame_shape)

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "version": "trajectory_aware_ball_tracker_v1",
                "trajectory_consistency": True,
                "trajectory": self.trajectory.metadata(),
                "override_policy": {
                    "min_trajectory_score": self.min_trajectory_score,
                    "protect_direct_confidence": self.protect_direct_confidence,
                    "override_score_margin": self.override_score_margin,
                },
                "reacquisition": False,
            }
        )
        return metadata
