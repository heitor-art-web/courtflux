from __future__ import annotations

from dataclasses import dataclass
import os
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
    recovery_provisional: bool = False
    challenger_velocity: tuple[float, float] | None = None


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

    Optional controlled reacquisition provides an escape hatch only when the active
    trajectory has no plausible normal association. It never widens the normal gate.
    Any plausible in-gate association has absolute priority and cancels recovery.

    state_aware_v3c keeps an out-of-gate recovery hypothesis separate from the active
    trajectory. A challenger first becomes provisional after temporal confirmation, then
    requires additional consistent support before promotion. Until promotion it never
    mutates last_center/last_area/velocity. On promotion, the challenger trajectory's own
    velocity seeds the new active track instead of forcing velocity to zero.
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
        reacquire_enabled: bool | None = None,
        reacquire_min_confidence: float | None = None,
        reacquire_confidence_margin: float | None = None,
        reacquire_confirm_frames: int | None = None,
        reacquire_radius_norm: float | None = None,
        reacquire_min_size_similarity: float | None = None,
        reacquire_min_unsupported_frames: int | None = None,
        reacquire_promotion_frames: int | None = None,
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

        self.reacquire_enabled = (
            self._env_bool("BASKETBALL_BALL_REACQUIRE", False)
            if reacquire_enabled is None
            else bool(reacquire_enabled)
        )
        reacquire_min_confidence = (
            self._env_float("BASKETBALL_BALL_REACQUIRE_MIN_CONF", 0.45)
            if reacquire_min_confidence is None
            else float(reacquire_min_confidence)
        )
        reacquire_confidence_margin = (
            self._env_float("BASKETBALL_BALL_REACQUIRE_CONF_MARGIN", 0.15)
            if reacquire_confidence_margin is None
            else float(reacquire_confidence_margin)
        )
        reacquire_confirm_frames = (
            self._env_int("BASKETBALL_BALL_REACQUIRE_CONFIRM_FRAMES", 2)
            if reacquire_confirm_frames is None
            else int(reacquire_confirm_frames)
        )
        reacquire_radius_norm = (
            self._env_float("BASKETBALL_BALL_REACQUIRE_RADIUS_NORM", 0.04)
            if reacquire_radius_norm is None
            else float(reacquire_radius_norm)
        )
        reacquire_min_size_similarity = (
            self._env_float("BASKETBALL_BALL_REACQUIRE_MIN_SIZE_SIM", 0.25)
            if reacquire_min_size_similarity is None
            else float(reacquire_min_size_similarity)
        )
        reacquire_min_unsupported_frames = (
            self._env_int("BASKETBALL_BALL_REACQUIRE_MIN_UNSUPPORTED_FRAMES", 2)
            if reacquire_min_unsupported_frames is None
            else int(reacquire_min_unsupported_frames)
        )
        reacquire_promotion_frames = (
            self._env_int("BASKETBALL_BALL_REACQUIRE_PROMOTION_FRAMES", 3)
            if reacquire_promotion_frames is None
            else int(reacquire_promotion_frames)
        )

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
        self.reacquire_min_unsupported_frames = max(
            1, int(reacquire_min_unsupported_frames)
        )
        self.reacquire_promotion_frames = max(
            self.reacquire_confirm_frames + 1,
            int(reacquire_promotion_frames),
        )
        self.reacquire_policy = "state_aware_v3c"

        if os.getenv("BASKETBALL_BALL_REACQUIRE") is not None:
            print(
                "🏀 Ball controlled reacquisition: "
                f"{'on' if self.reacquire_enabled else 'off'} "
                f"policy={self.reacquire_policy} "
                f"min_conf={self.reacquire_min_confidence:.2f} "
                f"margin={self.reacquire_confidence_margin:.2f} "
                f"confirm={self.reacquire_confirm_frames}f "
                f"promote={self.reacquire_promotion_frames}f "
                f"min_unsupported={self.reacquire_min_unsupported_frames}f "
                f"radius={self.reacquire_radius_norm:.3f}diag "
                f"min_size_sim={self.reacquire_min_size_similarity:.2f}",
                flush=True,
            )

        self.last_center: np.ndarray | None = None
        self.last_box: list[int] | None = None
        self.last_frame_id: int | None = None
        self.last_area: float | None = None
        self.last_confidence: float | None = None
        self.velocity = np.zeros(2, dtype=np.float64)

        self.track_state = "LOST"
        self._challenger_center: np.ndarray | None = None
        self._challenger_box: list[int] | None = None
        self._challenger_area: float | None = None
        self._challenger_last_frame_id: int | None = None
        self._challenger_streak = 0
        self._challenger_confidence: float | None = None
        self._challenger_velocity = np.zeros(2, dtype=np.float64)
        self._challenger_provisional = False

    @staticmethod
    def _env_bool(name: str, fallback: bool) -> bool:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return bool(fallback)
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _env_float(name: str, fallback: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return float(fallback)
        try:
            return float(raw)
        except ValueError:
            print(f"⚠️ Invalid {name}={raw!r}; using {fallback:.3f}.", flush=True)
            return float(fallback)

    @staticmethod
    def _env_int(name: str, fallback: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return int(fallback)
        try:
            return int(raw)
        except ValueError:
            print(f"⚠️ Invalid {name}={raw!r}; using {fallback}.", flush=True)
            return int(fallback)

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

    def _challenger_velocity_payload(self) -> tuple[float, float] | None:
        if self._challenger_streak <= 0:
            return None
        return (
            float(self._challenger_velocity[0]),
            float(self._challenger_velocity[1]),
        )

    def _reset_challenger(self) -> None:
        self._challenger_center = None
        self._challenger_box = None
        self._challenger_area = None
        self._challenger_last_frame_id = None
        self._challenger_streak = 0
        self._challenger_confidence = None
        self._challenger_velocity[:] = 0.0
        self._challenger_provisional = False

    def _expire_challenger(self, frame_id: int) -> None:
        """Keep a recovery hypothesis through one brief detector gap, then expire it."""
        if self._challenger_last_frame_id is None:
            return
        if int(frame_id) - int(self._challenger_last_frame_id) > 2:
            self._reset_challenger()

    def _update_challenger(
        self,
        candidate: dict[str, Any],
        frame_id: int,
        diagonal: float,
    ) -> int:
        """Advance the recovery-local trajectory without mutating active-track state."""
        center = np.asarray(candidate["center"], dtype=np.float64)
        area = self._candidate_area(candidate)
        confidence = max(0.0, min(float(candidate.get("confidence", 0.0)), 1.0))

        consistent = False
        measured_velocity: np.ndarray | None = None
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
                    measured_velocity = (
                        center - self._challenger_center
                    ) / float(dt)

        if consistent:
            self._challenger_streak += 1
            if measured_velocity is not None:
                alpha = self.velocity_alpha
                self._challenger_velocity = (
                    (1.0 - alpha) * self._challenger_velocity
                    + alpha * measured_velocity
                )
        else:
            self._challenger_streak = 1
            self._challenger_velocity[:] = 0.0
            self._challenger_provisional = False

        self._challenger_center = center
        self._challenger_box = [int(v) for v in candidate["box"]]
        self._challenger_area = area
        self._challenger_last_frame_id = int(frame_id)
        self._challenger_confidence = confidence
        if self._challenger_streak >= self.reacquire_confirm_frames:
            self._challenger_provisional = True
        self.track_state = "REACQUIRING"
        return self._challenger_streak

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
        velocity_override: np.ndarray | None = None,
    ) -> BallAssociationDecision:
        center = np.asarray(candidate["center"], dtype=np.float64)
        area = self._candidate_area(candidate)

        challenger_streak = self._challenger_streak
        challenger_confidence = self._challenger_confidence
        challenger_velocity = self._challenger_velocity_payload()
        recovery_provisional = bool(self._challenger_provisional)

        if velocity_override is not None:
            velocity = np.asarray(velocity_override, dtype=np.float64)
            if velocity.shape == (2,):
                self.velocity = velocity.copy()
            else:
                self.velocity[:] = 0.0
        elif reset_velocity:
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
            recovery_provisional=recovery_provisional,
            challenger_velocity=challenger_velocity,
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
            recovery_provisional=bool(self._challenger_provisional),
            challenger_velocity=self._challenger_velocity_payload(),
        )

    def _reacquire_candidate(
        self,
        candidates: list[dict[str, Any]],
        candidate_details: list[BallCandidateScore],
        frame_id: int,
        diagonal: float,
    ) -> tuple[int, dict[str, Any], BallCandidateScore, np.ndarray] | None:
        if not self.reacquire_enabled:
            self._reset_challenger()
            return None

        self._expire_challenger(frame_id)

        minimum_confidence = max(
            0.0,
            self.reacquire_min_confidence - self.reacquire_confidence_margin,
        )
        outside = [
            (index, candidate, details)
            for index, (candidate, details) in enumerate(
                zip(candidates, candidate_details)
            )
            if not details.plausible
            and details.confidence >= minimum_confidence
            and details.size_similarity >= self.reacquire_min_size_similarity
        ]
        if not outside:
            # A single miss does not immediately destroy the recovery hypothesis.
            return None

        outside.sort(
            key=lambda item: (
                item[2].confidence * item[2].size_similarity,
                item[2].confidence,
            ),
            reverse=True,
        )
        index, candidate, details = outside[0]

        streak = self._update_challenger(
            candidate=candidate,
            frame_id=frame_id,
            diagonal=diagonal,
        )

        unsupported_age = (
            self.reacquire_min_unsupported_frames
            if self.last_frame_id is None
            else max(int(frame_id) - int(self.last_frame_id), 0)
        )
        if unsupported_age < self.reacquire_min_unsupported_frames:
            return None

        # v3c two-stage recovery: temporal confirmation creates a provisional
        # hypothesis; one or more further consistent observations are required
        # before that hypothesis may replace the active trajectory.
        if not self._challenger_provisional:
            return None
        if streak < self.reacquire_promotion_frames:
            return None

        return index, candidate, details, self._challenger_velocity.copy()

    def select(
        self,
        candidates: list[dict[str, Any]],
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> BallAssociationDecision:
        candidates = list(candidates or [])
        if not candidates:
            self._expire_challenger(frame_id)
            hold = self._predicted_hold(frame_id, frame_shape, rejected_candidates=0)
            if hold is not None:
                return hold
            self.track_state = "REACQUIRING" if self._challenger_streak > 0 else "LOST"
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
                challenger_streak=int(self._challenger_streak),
                challenger_confidence=self._challenger_confidence,
                recovery_provisional=bool(self._challenger_provisional),
                challenger_velocity=self._challenger_velocity_payload(),
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
                reacquired=self.reacquire_enabled,
                source=(
                    "detector_reacquired"
                    if self.reacquire_enabled
                    else "detector_associated"
                ),
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
            for index, (candidate, details) in enumerate(
                zip(candidates, candidate_details)
            )
            if details.plausible
        ]

        if scored:
            # v3a+ invariant: supported direct evidence always wins and cancels
            # the independent recovery hypothesis before it can poison the track.
            scored.sort(key=lambda item: item[0], reverse=True)
            score, distance_norm, index, candidate = scored[0]
            self._reset_challenger()
            return self._direct_ball(
                candidate,
                frame_id,
                index,
                score,
                distance_norm,
                gate_norm,
                rejected_candidates=max(len(candidates) - len(scored), 0),
            )

        reacquire = self._reacquire_candidate(
            candidates=candidates,
            candidate_details=candidate_details,
            frame_id=frame_id,
            diagonal=diagonal,
        )
        if reacquire is not None:
            reacquire_index, reacquire_candidate, details, recovery_velocity = reacquire
            return self._direct_ball(
                reacquire_candidate,
                frame_id,
                reacquire_index,
                score=details.confidence,
                distance_norm=details.distance_norm,
                gate_norm=gate_norm,
                rejected_candidates=len(candidates),
                source="detector_reacquired",
                reacquired=True,
                velocity_override=recovery_velocity,
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
            recovery_provisional=bool(self._challenger_provisional),
            challenger_velocity=self._challenger_velocity_payload(),
        )
