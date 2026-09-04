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


class TemporalBallTracker:
    """Associate per-frame ball detections into one conservative image-space trajectory.

    The detector remains the source of direct observations. This class only chooses the
    most temporally plausible candidate and bridges very short gaps with constant-velocity
    prediction. Predicted points never update the direct-observation state, which prevents
    a prediction error from recursively drifting the trajectory.
    """

    def __init__(
        self,
        max_hold_frames: int = 5,
        reset_after_frames: int = 12,
        base_gate_norm: float = 0.12,
        gate_growth_per_frame: float = 0.025,
        max_gate_norm: float = 0.28,
        velocity_alpha: float = 0.40,
    ) -> None:
        self.max_hold_frames = max(0, int(max_hold_frames))
        self.reset_after_frames = max(self.max_hold_frames + 1, int(reset_after_frames))
        self.base_gate_norm = max(0.02, float(base_gate_norm))
        self.gate_growth_per_frame = max(0.0, float(gate_growth_per_frame))
        self.max_gate_norm = max(self.base_gate_norm, float(max_gate_norm))
        self.velocity_alpha = max(0.0, min(float(velocity_alpha), 1.0))

        self.last_center: np.ndarray | None = None
        self.last_box: list[int] | None = None
        self.last_frame_id: int | None = None
        self.last_area: float | None = None
        self.last_confidence: float | None = None
        self.velocity = np.zeros(2, dtype=np.float64)

    @staticmethod
    def _frame_diagonal(frame_shape: tuple[int, ...]) -> float:
        height, width = frame_shape[:2]
        return max((float(width) ** 2 + float(height) ** 2) ** 0.5, 1.0)

    @staticmethod
    def _candidate_area(candidate: dict[str, Any]) -> float:
        box = candidate.get("box") or [0, 0, 0, 0]
        return max(float(box[2] - box[0]), 1.0) * max(float(box[3] - box[1]), 1.0)

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
        if self.last_area is None or self.last_area <= 0 or candidate_area <= 0:
            return 0.5
        ratio = max(candidate_area / self.last_area, 1e-6)
        return float(exp(-abs(log(ratio))))

    def _score_candidate(
        self,
        candidate: dict[str, Any],
        predicted_center: np.ndarray,
        diagonal: float,
        gate_norm: float,
    ) -> tuple[float, float, bool]:
        center = np.asarray(candidate["center"], dtype=np.float64)
        distance_norm = float(np.linalg.norm(center - predicted_center) / diagonal)
        proximity = max(0.0, 1.0 - distance_norm / max(gate_norm, 1e-6))
        confidence = max(0.0, min(float(candidate.get("confidence", 0.0)), 1.0))
        size_similarity = self._size_similarity(self._candidate_area(candidate))
        score = 0.58 * confidence + 0.32 * proximity + 0.10 * size_similarity

        plausible = distance_norm <= gate_norm
        if not plausible and confidence >= 0.85:
            plausible = distance_norm <= min(gate_norm * 1.35, self.max_gate_norm)
        return float(score), distance_norm, plausible

    def _direct_ball(
        self,
        candidate: dict[str, Any],
        frame_id: int,
        selected_index: int,
        score: float | None,
        distance_norm: float | None,
        gate_norm: float | None,
        rejected_candidates: int,
    ) -> BallAssociationDecision:
        center = np.asarray(candidate["center"], dtype=np.float64)
        area = self._candidate_area(candidate)

        if self.last_center is not None and self.last_frame_id is not None:
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

        ball = dict(candidate)
        ball["interpolated"] = False
        ball["source"] = "detector_associated"
        ball["association_score"] = None if score is None else round(float(score), 6)
        ball["association_distance_norm"] = (
            None if distance_norm is None else round(float(distance_norm), 6)
        )
        return BallAssociationDecision(
            ball=ball,
            source="detector_associated",
            selected_index=selected_index,
            score=score,
            distance_norm=distance_norm,
            gate_norm=gate_norm,
            rejected_candidates=rejected_candidates,
            hold_age_frames=0,
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
        return BallAssociationDecision(
            ball=ball,
            source="predicted_hold",
            selected_index=None,
            score=None,
            distance_norm=None,
            gate_norm=None,
            rejected_candidates=rejected_candidates,
            hold_age_frames=age,
        )

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
            return BallAssociationDecision(None, "missing", None, None, None, None, 0, 0)

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
            index, candidate = ranked[0]
            return self._direct_ball(
                candidate,
                frame_id,
                index,
                score=float(candidate.get("confidence", 0.0)),
                distance_norm=None,
                gate_norm=None,
                rejected_candidates=0,
            )

        predicted_center = self._predict_center(frame_id)
        assert predicted_center is not None
        diagonal = self._frame_diagonal(frame_shape)
        gate_norm = self._gate(dt)

        scored: list[tuple[float, float, int, dict[str, Any]]] = []
        for index, candidate in enumerate(candidates):
            score, distance_norm, plausible = self._score_candidate(
                candidate,
                predicted_center,
                diagonal,
                gate_norm,
            )
            if plausible:
                scored.append((score, distance_norm, index, candidate))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            score, distance_norm, index, candidate = scored[0]
            return self._direct_ball(
                candidate,
                frame_id,
                index,
                score,
                distance_norm,
                gate_norm,
                rejected_candidates=max(len(candidates) - len(scored), 0),
            )

        hold = self._predicted_hold(
            frame_id,
            frame_shape,
            rejected_candidates=len(candidates),
        )
        if hold is not None:
            return hold

        return BallAssociationDecision(
            ball=None,
            source="missing",
            selected_index=None,
            score=None,
            distance_norm=None,
            gate_norm=gate_norm,
            rejected_candidates=len(candidates),
            hold_age_frames=0,
        )
