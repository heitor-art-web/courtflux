from __future__ import annotations

from typing import Any

from core.ball_state import BallState, BallStateStatus
from core.config import Settings, settings
from services.vision.ball_hybrid_v1 import TrajectoryAwareBallTrackerV1
from services.vision.ball_temporal_v1 import BallTemporalDecision
from services.vision.ball_trajectory_consistency import BallTrajectoryConsistencyV1


class BallStateEstimatorV1:
    """Convert P1 ball detector/tracker evidence into canonical ``BallState``.

    The estimator deliberately owns no basketball-event semantics. It exposes only
    position/state evidence, uncertainty, timestamps and provenance.

    ``update`` remains the standalone canonical tracker used by the validated BallState
    benchmark. ``from_runtime_evidence`` is the production adapter for an already-selected
    runtime ball hypothesis, keeping the Lab/UI contract canonical while the detector
    association implementation is consolidated.
    """

    def __init__(self, *, config: Settings | None = None) -> None:
        self.config = config or settings
        self.tracker = TrajectoryAwareBallTrackerV1(
            trajectory=BallTrajectoryConsistencyV1(
                max_distance_norm=self.config.ball_trajectory_max_distance_norm,
                max_speed_residual_norm=self.config.ball_trajectory_max_speed_residual_norm,
                max_acceleration_norm=self.config.ball_trajectory_max_acceleration_norm,
            ),
            min_trajectory_score=self.config.ball_trajectory_min_score,
            protect_direct_confidence=self.config.ball_trajectory_protect_direct_confidence,
            override_score_margin=self.config.ball_trajectory_override_score_margin,
            max_prediction_gap=self.config.ball_max_prediction_gap,
        )
        self._runtime_last_observed_center: tuple[float, float] | None = None
        self._runtime_last_observed_frame: int | None = None
        self._runtime_velocity: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def _status(decision: BallTemporalDecision) -> BallStateStatus:
        if decision.ball is None or decision.source == "missing":
            return BallStateStatus.MISSING
        if decision.source == "predicted_short_gap" or bool(decision.ball.get("interpolated")):
            return BallStateStatus.PREDICTED
        return BallStateStatus.OBSERVED

    @staticmethod
    def _runtime_status(ball: dict[str, Any] | None, source: str) -> BallStateStatus:
        if ball is None or source == "missing":
            return BallStateStatus.MISSING
        if bool(ball.get("interpolated", False)) or source in {
            "predicted_hold",
            "predicted_short_gap",
        }:
            return BallStateStatus.PREDICTED
        return BallStateStatus.OBSERVED

    @staticmethod
    def _center(ball: dict[str, Any]) -> tuple[float, float]:
        center = ball.get("center")
        if center is not None:
            return float(center[0]), float(center[1])
        box = ball.get("box") or [0, 0, 0, 0]
        return (
            (float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0,
        )

    @staticmethod
    def _box(ball: dict[str, Any]) -> tuple[int, int, int, int] | None:
        box = ball.get("box")
        if box is None or len(box) < 4:
            return None
        return tuple(int(round(float(value))) for value in box[:4])  # type: ignore[return-value]

    def _provenance(
        self,
        decision: BallTemporalDecision,
        *,
        status: BallStateStatus,
    ) -> dict[str, Any]:
        provenance: dict[str, Any] = {
            "detector_model": self.config.model,
            "detector_version": self.config.detector_version,
            "detector_baseline_status": self.config.detector_baseline_status,
            "ball_candidate_confidence": self.config.ball_candidate_confidence,
            "tracker": "trajectory_aware_ball_tracker_v1",
            "tracker_source": decision.source,
            "selected_candidate_index": decision.selected_index,
            "distance_norm": decision.distance_norm,
            "direct_observation": status is BallStateStatus.OBSERVED,
            "short_gap_prediction": status is BallStateStatus.PREDICTED,
            "prediction_updates_state": False,
            "reacquisition": False,
            "uncertainty_policy": "1_minus_upstream_confidence",
        }
        if decision.ball is not None and decision.ball.get("trajectory_consistency") is not None:
            provenance["trajectory_consistency"] = dict(
                decision.ball["trajectory_consistency"]
            )
        return provenance

    def from_decision(
        self,
        decision: BallTemporalDecision,
        *,
        frame_id: int,
        timestamp_sec: float | None,
    ) -> BallState:
        status = self._status(decision)
        if status is BallStateStatus.MISSING or decision.ball is None:
            return BallState(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                status=BallStateStatus.MISSING,
                velocity_px_per_frame=decision.velocity_px_per_frame,
                age_since_detection=decision.age_since_detection,
                source=decision.source,
                provenance=self._provenance(decision, status=BallStateStatus.MISSING),
            )

        center = self._center(decision.ball)
        confidence = max(0.0, min(float(decision.ball.get("confidence", 0.0)), 1.0))
        trajectory_score = decision.ball.get("trajectory_score")
        return BallState(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            status=status,
            x_px=center[0],
            y_px=center[1],
            box=self._box(decision.ball),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            velocity_px_per_frame=decision.velocity_px_per_frame,
            age_since_detection=decision.age_since_detection,
            source=decision.source,
            trajectory_score=(
                None if trajectory_score is None else float(trajectory_score)
            ),
            provenance=self._provenance(decision, status=status),
        )

    def from_runtime_evidence(
        self,
        ball: dict[str, Any] | None,
        association: dict[str, Any] | None,
        *,
        frame_id: int,
        timestamp_sec: float | None,
    ) -> BallState:
        """Canonicalize the current production ball selection without re-tracking it.

        Direct observations update only a small velocity memory. Predicted runtime points
        never feed back into that memory, preserving the BallState contract that prediction
        must not recursively create future state.
        """
        association = dict(association or {})
        source = str(association.get("source") or (ball or {}).get("source") or "missing")
        status = self._runtime_status(ball, source)
        age = int(
            association.get(
                "hold_age_frames",
                (ball or {}).get("age_since_detection", 0),
            )
            or 0
        )

        if status is BallStateStatus.MISSING or ball is None:
            return BallState(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                status=BallStateStatus.MISSING,
                velocity_px_per_frame=self._runtime_velocity,
                age_since_detection=age,
                source="missing",
                provenance={
                    "version": "ball_state_v1_runtime_adapter",
                    "detector_model": self.config.model,
                    "detector_version": self.config.detector_version,
                    "upstream_source": source,
                    "selected_candidate_index": association.get("selected_index"),
                    "prediction_updates_state": False,
                    "uncertainty_policy": "1_minus_upstream_confidence",
                },
            )

        center = self._center(ball)
        if status is BallStateStatus.OBSERVED:
            previous_center = self._runtime_last_observed_center
            previous_frame = self._runtime_last_observed_frame
            if previous_center is not None and previous_frame is not None:
                dt = int(frame_id) - int(previous_frame)
                if dt > 0:
                    self._runtime_velocity = (
                        (center[0] - previous_center[0]) / dt,
                        (center[1] - previous_center[1]) / dt,
                    )
            else:
                self._runtime_velocity = (0.0, 0.0)
            self._runtime_last_observed_center = center
            self._runtime_last_observed_frame = int(frame_id)
            age = 0

        confidence = max(0.0, min(float(ball.get("confidence", 0.0)), 1.0))
        trajectory_score = ball.get("trajectory_score")
        return BallState(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            status=status,
            x_px=center[0],
            y_px=center[1],
            box=self._box(ball),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            velocity_px_per_frame=self._runtime_velocity,
            age_since_detection=age,
            source=source,
            trajectory_score=(
                None if trajectory_score is None else float(trajectory_score)
            ),
            provenance={
                "version": "ball_state_v1_runtime_adapter",
                "detector_model": self.config.model,
                "detector_version": self.config.detector_version,
                "upstream_source": source,
                "selected_candidate_index": association.get("selected_index"),
                "distance_norm": association.get("distance_norm"),
                "gate_norm": association.get("gate_norm"),
                "direct_observation": status is BallStateStatus.OBSERVED,
                "short_gap_prediction": status is BallStateStatus.PREDICTED,
                "prediction_updates_state": False,
                "uncertainty_policy": "1_minus_upstream_confidence",
            },
        )

    def update(
        self,
        candidates: list[dict[str, Any]],
        *,
        frame_id: int,
        timestamp_sec: float | None,
        frame_shape: tuple[int, ...],
    ) -> BallState:
        eligible = [
            candidate
            for candidate in list(candidates or [])
            if float(candidate.get("confidence", 0.0))
            >= self.config.ball_candidate_confidence
        ]
        decision = self.tracker.update(
            eligible,
            frame_id=frame_id,
            frame_shape=frame_shape,
        )
        return self.from_decision(
            decision,
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "ball_state_v1",
            "detector_model": self.config.model,
            "detector_version": self.config.detector_version,
            "ball_candidate_confidence": self.config.ball_candidate_confidence,
            "tracker": self.tracker.metadata(),
            "runtime_adapter": "canonicalizes_existing_association_without_retracking",
            "states": [status.value for status in BallStateStatus],
            "p2_event_semantics": False,
        }
