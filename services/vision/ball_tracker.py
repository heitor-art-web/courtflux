from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any

import numpy as np


@dataclass
class BallAssociationDecision:
    ball: dict[str, Any] | None
    source: str
    selected_index: int | None
    score: float | None
    distance_norm: float | None
    gate_norm: float | None
    rejected_candidates: int
    hold_age_frames: int
    track_state: str = "LOST"
    reacquired: bool = False
    challenger_streak: int = 0
    challenger_confidence: float | None = None


@dataclass(frozen=True)
class BallCandidateScore:
    score: float
    confidence: float
    proximity: float
    size_similarity: float
    confidence_contribution: float
    proximity_contribution: float
    size_contribution: float
    distance_norm: float
    gate_norm: float
    plausible: bool
    high_confidence_gate_used: bool


class TemporalBallTracker:
    """Associate per-frame ball detections into one conservative image-space trajectory.

    The detector remains the source of direct observations. This class only chooses the
    most temporally plausible candidate and bridges very short gaps with constant-velocity
    prediction. Predicted points never update the direct-observation state, which prevents
    a prediction error from recursively drifting the trajectory.

    Association weights and the high-confidence gate extension are configurable so they
    can be benchmarked offline without changing the production defaults.

    Controlled reacquisition provides an escape hatch when the active trajectory is no
    longer supported. It never widens the normal gate. Instead, an out-of-gate challenger
    must be sufficiently confident and persist consistently across multiple frames before
    it is allowed to replace the active trajectory.
    """

    def __init__(
        self,
        max_hold_frames: int = 5,
        reset_after_frames: int = 12,
        base_gate_norm: float = 0.12,
        gate_growth_per_frame: float = 0.025,
        max_gate_norm: float = 0.28,
        velocity_alpha: float = 0.40,
        score_confidence_weight: float = 0.58,
        score_proximity_weight: float = 0.32,
        score_size_weight: float = 0.10,
        high_confidence_gate_threshold: float = 0.85,
        high_confidence_gate_multiplier: float = 1.35,
        reacquire_enabled: bool = True,
        reacquire_min_confidence: float = 0.45,
        reacquire_confidence_margin: float = 0.15,
        reacquire_confirm_frames: int = 2,
        reacquire_radius_norm: float = 0.04,
        reacquire_min_size_similarity: float = 0.25,
    ) -> None:
        self.max_hold_frames = max(0, int(max_hold_frames))
        self.reset_after_frames = max(self.max_hold_frames + 1, int(reset_after_frames))
        self.base_gate_norm = max(0.02, float(base_gate_norm))
        self.gate_growth_per_frame = max(0.0, float(gate_growth_per_frame))
        self.max_gate_norm = max(self.base_gate_norm, float(max_gate_norm))
        self.velocity_alpha = max(0.0, min(float(velocity_alpha), 1.0))

        weights = [
            max(0.0, float(score_confidence_weight)),
            max(0.0, float(score_proximity_weight)),
            max(0.0, float(score_size_weight)),
        ]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("At least one ball association score weight must be positive.")
        if abs(weight_sum - 1.0) > 1e-12:
            weights = [value / weight_sum for value in weights]
        self.score_confidence_weight = weights[0]
        self.score_proximity_weight = weights[1]
        self.score_size_weight = weights[2]
        self.high_confidence_gate_threshold = max(
            0.0, min(float(high_confidence_gate_threshold), 1.0)
        )
        self.high_confidence_gate_multiplier = max(
            1.0, float(high_confidence_gate_multiplier)
        )

        self.reacquire_enabled = bool(reacquire_enabled)
        self.reacquire_min_confidence = max(
            0.0, min(float(reacquire_min_confidence), 1.0)
        )
        self.reacquire_confidence_margin = max(
            0.0, min(float(reacquire_confidence_margin), 1.0)
        )
        self.reacquire_confirm_frames = max(1, int(reacquire_confirm_frames))
        self.reacquire_radius_norm = max(0.005, float(reacquire_radius_norm))
        self.reacquire_min_size_similarity = max(
            0.0, min(float(reacquire_min_size_similarity), 1.0)
        )

        self.last_center: np.ndarray | None = None
        self.last_box: list[int] | None = None
        self.last_frame_id: int | None = None
        self.last_area: float | None = None
        self.last_confidence: float | None = None
        self.velocity = np.zeros(2, dtype=np.float64)

        self.track_state = "LOST"
        self._challenger_center: np.ndarray | None = None
        self._challenger_area: float | None = None
        self._challenger_last_frame_id: int | None = None
        self._challenger_streak = 0
        self._challenger_confidence: float | None = None

    @staticmethod
    def _frame_diagonal(frame_shape: tuple[int, ...]) -> float:
        height, width = frame_shape[:2]
        return max((float(width) ** 2 + float(height) ** 2) ** 0.5, 1.0)

    @staticmethod
    def _candidate_area(candidate: dict[str, Any]) -> float:
        box = candidate.get("box") or [0, 0, 0, 0]
        return max(float(box[2] - box[0]), 1.0) * max(float(box[3] - box[1]), 1.0)

    @staticmethod
    def _area_similarity(area_a: float | None, area_b: float | None) -> float:
        if area_a is None or area_b is None or area_a <= 0 or area_b <= 0:
            return 0.5
        ratio = max(float(area_a) / float(area_b), 1e-6)
        return float(exp(-abs(log(ratio))))

    def _predict_center(self, frame_id: int) -> np.ndarray | None:
        if self.last_center is None or self.last_frame_id is None:
            return None
        dt = max(int(frame_id) - int(self.last_frame_id), 0)
        return self.last_center + self.velocity * float(dt)

    def _gate(self, dt: int) -> float:
        return min(
            self.base_gate_norm + self.gate_growth_per_frame * max(int(dt) - 1, 0),
            self.max_gate_norm,
        )

    def _size_similarity(self, candidate_area: float) -> float:
        return self._area_similarity(candidate_area, self.last_area)

    def _candidate_score_details(
        self,
        candidate: dict[str, Any],
        predicted_center: np.ndarray,
        diagonal: float,
        gate_norm: float,
    ) -> BallCandidateScore:
        center = np.asarray(candidate["center"], dtype=np.float64)
        distance_norm = float(np.linalg.norm(center - predicted_center) / diagonal)
        proximity = max(0.0, 1.0 - distance_norm / max(gate_norm, 1e-6))
        confidence = max(0.0, min(float(candidate.get("confidence", 0.0)), 1.0))
        size_similarity = self._size_similarity(self._candidate_area(candidate))

        confidence_contribution = self.score_confidence_weight * confidence
        proximity_contribution = self.score_proximity_weight * proximity
        size_contribution = self.score_size_weight * size_similarity
        score = confidence_contribution + proximity_contribution + size_contribution

        plausible = distance_norm <= gate_norm
        high_confidence_gate_used = False
        if not plausible and confidence >= self.high_confidence_gate_threshold:
            extended_gate = min(
                gate_norm * self.high_confidence_gate_multiplier,
                self.max_gate_norm,
            )
            plausible = distance_norm <= extended_gate
            high_confidence_gate_used = plausible and extended_gate > gate_norm

        return BallCandidateScore(
            score=float(score),
            confidence=float(confidence),
            proximity=float(proximity),
            size_similarity=float(size_similarity),
            confidence_contribution=float(confidence_contribution),
            proximity_contribution=float(proximity_contribution),
            size_contribution=float(size_contribution),
            distance_norm=float(distance_norm),
            gate_norm=float(gate_norm),
            plausible=bool(plausible),
            high_confidence_gate_used=bool(high_confidence_gate_used),
        )

    def inspect_candidate(
        self,
        candidate: dict[str, Any],
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> BallCandidateScore | None:
        """Return the current association score breakdown without mutating tracker state."""
        if self.last_center is None or self.last_frame_id is None:
            return None
        dt = max(int(frame_id) - int(self.last_frame_id), 1)
        if dt > self.reset_after_frames:
            return None
        predicted_center = self._predict_center(frame_id)
        if predicted_center is None:
            return None
        diagonal = self._frame_diagonal(frame_shape)
        gate_norm = self._gate(dt)
        return self._candidate_score_details(
            candidate,
            predicted_center,
            diagonal,
            gate_norm,
        )

    def _score_candidate(
        self,
        candidate: dict[str, Any],
        predicted_center: np.ndarray,
        diagonal: float,
        gate_norm: float,
    ) -> tuple[float, float, bool]:
        details = self._candidate_score_details(
            candidate,
            predicted_center,
            diagonal,
            gate_norm,
        )
        return details.score, details.distance_norm, details.plausible

    def _reset_challenger(self) -> None:
        self._challenger_center = None
        self._challenger_area = None
        self._challenger_last_frame_id = None
        self._challenger_streak = 0
        self._challenger_confidence = None

    def _update_challenger(
        self,
        candidate: dict[str, Any],
        frame_id: int,
        diagonal: float,
    ) -> bool:
        center = np.asarray(candidate["center"], dtype=np.float64)
        area = self._candidate_area(candidate)
        confidence = max(0.0, min(float(candidate.get("confidence", 0.0)), 1.0))

        consistent = False
        if (
            self._challenger_center is not None
            and self._challenger_last_frame_id is not None
        ):
            dt = int(frame_id) - int(self._challenger_last_frame_id)
            if 0 < dt <= 2:
                distance_norm = float(
                    np.linalg.norm(center - self._challenger_center) / diagonal
                )
                allowed_radius = self.reacquire_radius_norm * float(max(dt, 1))
                size_similarity = self._area_similarity(area, self._challenger_area)
                consistent = (
                    distance_norm <= allowed_radius
                    and size_similarity >= self.reacquire_min_size_similarity
                )

        if consistent:
            self._challenger_streak += 1
        else:
            self._challenger_streak = 1

        self._challenger_center = center
        self._challenger_area = area
        self._challenger_last_frame_id = int(frame_id)
        self._challenger_confidence = confidence
        self.track_state = "REACQUIRING"
        return self._challenger_streak >= self.reacquire_confirm_frames

    def _direct_ball(
        self,
        candidate: dict[str, Any],
        frame_id: int,
        selected_index: int,
        score: float | None,
        distance_norm: float | None,
        gate_norm: float | None,
        rejected_candidates: int,
        *,
        source: str = "detector_associated",
        reset_velocity: bool = False,
        reacquired: bool = False,
        clear_challenger: bool = True,
    ) -> BallAssociationDecision:
        center = np.asarray(candidate["center"], dtype=np.float64)
        area = self._candidate_area(candidate)

        if reset_velocity:
            self.velocity[:] = 0.0
        elif self.last_center is not None and self.last_frame_id is not None:
            dt = int(frame_id) - int(self.last_frame_id)
            if 0 < dt <= self.reset_after_frames:
                measured_velocity = (center - self.last_center) / float(dt)
                alpha = self.velocity_alpha
                self.velocity = (1.0 - alpha) * self.velocity + alpha * measured_velocity
            else:
                self.velocity[:] = 0.0
        else:
            self.velocity[:] = 0.0

        self.last_center = center
        self.last_box = [int(v) for v in candidate["box"]]
        self.last_frame_id = int(frame_id)
        self.last_area = area
        self.last_confidence = float(candidate.get("confidence", 0.0))
        self.track_state = "TRACKED"
        challenger_streak = self._challenger_streak
        challenger_confidence = self._challenger_confidence
        if clear_challenger:
            self._reset_challenger()

        ball = dict(candidate)
        ball["interpolated"] = False
        ball["source"] = source
        ball["association_score"] = None if score is None else round(float(score), 6)
        ball["association_distance_norm"] = (
            None if distance_norm is None else round(float(distance_norm), 6)
        )
        return BallAssociationDecision(
            ball=ball,
            source=source,
            selected_index=selected_index,
            score=score,
            distance_norm=distance_norm,
            gate_norm=gate_norm,
            rejected_candidates=rejected_candidates,
            hold_age_frames=0,
            track_state=self.track_state,
            reacquired=bool(reacquired),
            challenger_streak=int(challenger_streak),
            challenger_confidence=challenger_confidence,
        )

    def _predicted_hold(
        self,
        frame_id: int,
        frame_shape: tuple[int, ...],
        rejected_candidates: int,
    ) -> BallAssociationDecision | None:
        if (
            self.last_center is None
            or self.last_box is None
            or self.last_frame_id is None
            or self.max_hold_frames <= 0
        ):
            return None

        age = int(frame_id) - int(self.last_frame_id)
        if age <= 0 or age > self.max_hold_frames:
            return None

        predicted_center = self._predict_center(frame_id)
        if predicted_center is None:
            return None

        height, width = frame_shape[:2]
        old_box = self.last_box
        box_w = max(int(old_box[2] - old_box[0]), 2)
        box_h = max(int(old_box[3] - old_box[1]), 2)
        cx = float(np.clip(predicted_center[0], 0, max(width - 1, 0)))
        cy = float(np.clip(predicted_center[1], 0, max(height - 1, 0)))
        x1 = int(round(cx - box_w / 2.0))
        y1 = int(round(cy - box_h / 2.0))
        x1 = max(0, min(x1, max(width - box_w, 0)))
        y1 = max(0, min(y1, max(height - box_h, 0)))
        x2 = min(width - 1, x1 + box_w)
        y2 = min(height - 1, y1 + box_h)

        base_conf = float(self.last_confidence or 0.0)
        predicted_conf = base_conf * (0.82 ** age)
        ball = {
            "box": [x1, y1, x2, y2],
            "center": (cx, cy),
            "confidence": predicted_conf,
            "interpolated": True,
            "source": "predicted_hold",
            "hold_age_frames": age,
        }
        if self._challenger_streak > 0:
            self.track_state = "REACQUIRING"
        else:
            self.track_state = "DEGRADED"
        return BallAssociationDecision(
            ball=ball,
            source="predicted_hold",
            selected_index=None,
            score=None,
            distance_norm=None,
            gate_norm=None,
            rejected_candidates=rejected_candidates,
            hold_age_frames=age,
            track_state=self.track_state,
            reacquired=False,
            challenger_streak=int(self._challenger_streak),
            challenger_confidence=self._challenger_confidence,
        )

    def _reacquire_candidate(
        self,
        candidates: list[dict[str, Any]],
        candidate_details: list[BallCandidateScore],
        selected_confidence: float | None,
        frame_id: int,
        diagonal: float,
    ) -> tuple[int, dict[str, Any], BallCandidateScore] | None:
        if not self.reacquire_enabled:
            self._reset_challenger()
            return None

        outside = [
            (index, candidate, details)
            for index, (candidate, details) in enumerate(zip(candidates, candidate_details))
            if not details.plausible
            and details.confidence >= self.reacquire_min_confidence
        ]
        if not outside:
            self._reset_challenger()
            return None

        outside.sort(key=lambda item: item[2].confidence, reverse=True)
        index, candidate, details = outside[0]

        if (
            selected_confidence is not None
            and details.confidence
            < float(selected_confidence) + self.reacquire_confidence_margin
        ):
            self._reset_challenger()
            return None

        confirmed = self._update_challenger(
            candidate=candidate,
            frame_id=frame_id,
            diagonal=diagonal,
        )
        if not confirmed:
            return None
        return index, candidate, details

    def select(
        self,
        candidates: list[dict[str, Any]],
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> BallAssociationDecision:
        candidates = list(candidates or [])
        if not candidates:
            hold = self._predicted_hold(frame_id, frame_shape, rejected_candidates=0)
            if hold is not None:
                return hold
            self.track_state = "LOST"
            self._reset_challenger()
            return BallAssociationDecision(
                None,
                "missing",
                None,
                None,
                None,
                None,
                0,
                0,
                track_state=self.track_state,
            )

        ranked = sorted(
            enumerate(candidates),
            key=lambda item: float(item[1].get("confidence", 0.0)),
            reverse=True,
        )

        if self.last_center is None or self.last_frame_id is None:
            index, candidate = ranked[0]
            return self._direct_ball(candidate, frame_id, index, None, None, None, 0)

        dt = max(int(frame_id) - int(self.last_frame_id), 1)
        if dt > self.reset_after_frames:
            self.velocity[:] = 0.0
            self._reset_challenger()
            index, candidate = ranked[0]
            return self._direct_ball(
                candidate,
                frame_id,
                index,
                score=float(candidate.get("confidence", 0.0)),
                distance_norm=None,
                gate_norm=None,
                rejected_candidates=0,
                reset_velocity=True,
                reacquired=True,
                source="detector_reacquired",
            )

        predicted_center = self._predict_center(frame_id)
        assert predicted_center is not None
        diagonal = self._frame_diagonal(frame_shape)
        gate_norm = self._gate(dt)

        candidate_details = [
            self._candidate_score_details(
                candidate,
                predicted_center,
                diagonal,
                gate_norm,
            )
            for candidate in candidates
        ]
        scored: list[tuple[float, float, int, dict[str, Any]]] = [
            (details.score, details.distance_norm, index, candidate)
            for index, (candidate, details) in enumerate(zip(candidates, candidate_details))
            if details.plausible
        ]

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            score, distance_norm, index, candidate = scored[0]
            selected_confidence = float(candidate.get("confidence", 0.0))
            reacquire = self._reacquire_candidate(
                candidates=candidates,
                candidate_details=candidate_details,
                selected_confidence=selected_confidence,
                frame_id=frame_id,
                diagonal=diagonal,
            )
            if reacquire is not None:
                reacquire_index, reacquire_candidate, details = reacquire
                return self._direct_ball(
                    reacquire_candidate,
                    frame_id,
                    reacquire_index,
                    score=details.confidence,
                    distance_norm=details.distance_norm,
                    gate_norm=gate_norm,
                    rejected_candidates=max(len(candidates) - len(scored), 0),
                    source="detector_reacquired",
                    reset_velocity=True,
                    reacquired=True,
                )

            if self._challenger_streak > 0:
                self.track_state = "REACQUIRING"
                challenger_streak = self._challenger_streak
                challenger_confidence = self._challenger_confidence
            else:
                self.track_state = "TRACKED"
                challenger_streak = 0
                challenger_confidence = None

            decision = self._direct_ball(
                candidate,
                frame_id,
                index,
                score,
                distance_norm,
                gate_norm,
                rejected_candidates=max(len(candidates) - len(scored), 0),
                clear_challenger=challenger_streak <= 0,
            )
            if challenger_streak > 0:
                decision.track_state = "REACQUIRING"
                decision.challenger_streak = challenger_streak
                decision.challenger_confidence = challenger_confidence
                self.track_state = "REACQUIRING"
            return decision

        reacquire = self._reacquire_candidate(
            candidates=candidates,
            candidate_details=candidate_details,
            selected_confidence=None,
            frame_id=frame_id,
            diagonal=diagonal,
        )
        if reacquire is not None:
            reacquire_index, reacquire_candidate, details = reacquire
            return self._direct_ball(
                reacquire_candidate,
                frame_id,
                reacquire_index,
                score=details.confidence,
                distance_norm=details.distance_norm,
                gate_norm=gate_norm,
                rejected_candidates=len(candidates),
                source="detector_reacquired",
                reset_velocity=True,
                reacquired=True,
            )

        hold = self._predicted_hold(
            frame_id,
            frame_shape,
            rejected_candidates=len(candidates),
        )
        if hold is not None:
            return hold

        self.track_state = "REACQUIRING" if self._challenger_streak > 0 else "LOST"
        return BallAssociationDecision(
            ball=None,
            source="missing",
            selected_index=None,
            score=None,
            distance_norm=None,
            gate_norm=gate_norm,
            rejected_candidates=len(candidates),
            hold_age_frames=0,
            track_state=self.track_state,
            reacquired=False,
            challenger_streak=int(self._challenger_streak),
            challenger_confidence=self._challenger_confidence,
        )
