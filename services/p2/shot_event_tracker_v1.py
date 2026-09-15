from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import (
    BasketballEvent,
    BasketballEventType,
    EventDecision,
    ShotOutcome,
)
from core.court_state import CourtState


@dataclass(slots=True)
class _ShotCandidate:
    frame_start: int
    timestamp_start_sec: float | None
    last_frame: int
    rim_bbox: tuple[float, float, float, float] | None = None
    had_above_rim: bool = False
    had_rim_band: bool = False
    direct_ball_observations: int = 0
    semantic_start: bool = False
    evidence: set[str] = field(default_factory=set)
    last_ball_position: tuple[float, float] | None = None


class ShotEventTrackerV1:
    """Temporal shot state machine inspired by the Roboflow basketball tracker pattern.

    Semantic detections are supporting evidence only. A single ``player-jump-shot`` or
    ``ball-in-basket`` observation never creates an event without an active shot candidate.
    Missing evidence resolves to UNKNOWN rather than being silently counted as a miss.
    """

    def __init__(
        self,
        *,
        fps: float = 30.0,
        reset_time_sec: float = 1.7,
        minimum_between_starts_sec: float = 0.5,
        cooldown_after_made_sec: float = 0.5,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.reset_frames = max(1, int(round(self.fps * float(reset_time_sec))))
        self.minimum_between_starts_frames = max(
            1,
            int(round(self.fps * float(minimum_between_starts_sec))),
        )
        self.cooldown_after_made_frames = max(
            1,
            int(round(self.fps * float(cooldown_after_made_sec))),
        )
        self._active: _ShotCandidate | None = None
        self._last_start_frame = -10**9
        self._cooldown_until_frame = -1

    @staticmethod
    def _label(row: dict[str, Any]) -> str:
        return str(
            row.get("class_name")
            or row.get("label")
            or row.get("kind")
            or row.get("class")
            or ""
        ).strip().lower().replace("_", "-")

    @classmethod
    def _semantic_flags(cls, rows: list[dict[str, Any]]) -> dict[str, bool]:
        flags = {
            "jump_shot": False,
            "layup_dunk": False,
            "ball_in_basket": False,
            "shot_block": False,
        }
        for row in rows:
            if not isinstance(row, dict):
                continue
            confidence = max(0.0, min(float(row.get("confidence", 0.0)), 1.0))
            if confidence < 0.35:
                continue
            label = cls._label(row)
            if "player-jump-shot" in label:
                flags["jump_shot"] = True
            elif "player-layup-dunk" in label:
                flags["layup_dunk"] = True
            elif "ball-in-basket" in label:
                flags["ball_in_basket"] = True
            elif "player-shot-block" in label:
                flags["shot_block"] = True
        return flags

    @staticmethod
    def _rim_candidates(court_state: CourtState | None) -> list[tuple[float, float, float, float]]:
        if court_state is None:
            return []
        rims: list[tuple[float, float, float, float]] = []
        for row in list(court_state.basket_evidence or []):
            if not isinstance(row, dict) or str(row.get("kind") or "").lower() != "rim":
                continue
            raw = row.get("bbox_px", row.get("box", row.get("bbox")))
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in raw[:4]]
            except (TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                rims.append((x1, y1, x2, y2))
        return rims

    @staticmethod
    def _nearest_rim(
        ball_position: tuple[float, float] | None,
        rims: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float] | None:
        if not rims:
            return None
        if ball_position is None:
            return rims[0]
        bx, by = ball_position
        return min(
            rims,
            key=lambda rim: (((rim[0] + rim[2]) / 2.0 - bx) ** 2 + ((rim[1] + rim[3]) / 2.0 - by) ** 2),
        )

    @staticmethod
    def _relative_to_rim(
        ball_position: tuple[float, float] | None,
        rim: tuple[float, float, float, float] | None,
    ) -> dict[str, bool]:
        result = {"above": False, "near_x": False, "rim_band": False, "below": False, "inside_mouth": False}
        if ball_position is None or rim is None:
            return result
        bx, by = ball_position
        x1, y1, x2, y2 = rim
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        rim_y = (y1 + y2) / 2.0
        rim_x = (x1 + x2) / 2.0
        result["above"] = by < rim_y - 0.20 * height
        result["near_x"] = abs(bx - rim_x) <= 2.5 * width
        result["rim_band"] = (y1 - 0.8 * height) <= by <= (y2 + 0.8 * height) and result["near_x"]
        result["below"] = by > rim_y + 0.65 * height
        result["inside_mouth"] = (x1 - 0.20 * width) <= bx <= (x2 + 0.20 * width)
        return result

    def _can_start(self, frame_id: int) -> bool:
        return bool(
            frame_id >= self._cooldown_until_frame
            and frame_id - self._last_start_frame >= self.minimum_between_starts_frames
        )

    @staticmethod
    def _quality(ball_state: BallState | None) -> float:
        if ball_state is None or not ball_state.available:
            return 0.0
        trajectory = (
            float(ball_state.trajectory_score)
            if ball_state.trajectory_score is not None
            else float(ball_state.confidence)
        )
        quality = 0.65 * float(ball_state.confidence) + 0.35 * trajectory
        if ball_state.status is BallStateStatus.PREDICTED:
            quality *= 0.72
        return max(0.0, min(quality, 1.0))

    def _start(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        rim: tuple[float, float, float, float] | None,
        ball_state: BallState | None,
        semantic_start: bool,
    ) -> None:
        candidate = _ShotCandidate(
            frame_start=int(frame_id),
            timestamp_start_sec=timestamp_sec,
            last_frame=int(frame_id),
            rim_bbox=rim,
            semantic_start=bool(semantic_start),
        )
        if semantic_start:
            candidate.evidence.add("semantic_shot_action")
        if ball_state is not None and ball_state.available:
            candidate.last_ball_position = ball_state.position()
            if ball_state.status is BallStateStatus.OBSERVED:
                candidate.direct_ball_observations += 1
        self._active = candidate
        self._last_start_frame = int(frame_id)

    def _resolve(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        outcome: ShotOutcome,
        confidence: float,
        reason: str,
        ball_state: BallState | None,
    ) -> BasketballEvent:
        candidate = self._active
        if candidate is None:
            raise RuntimeError("cannot resolve a shot without an active candidate")
        evidence = set(candidate.evidence)
        evidence.add(reason)
        quality = self._quality(ball_state)
        final_confidence = max(0.0, min(0.75 * float(confidence) + 0.25 * quality, 1.0))
        if outcome is ShotOutcome.UNKNOWN:
            final_confidence = min(final_confidence, 0.55)

        event = BasketballEvent(
            event_id=f"shot-attempt:{candidate.frame_start}:{frame_id}",
            event_type=BasketballEventType.SHOT_ATTEMPT,
            frame_start=candidate.frame_start,
            frame_end=int(frame_id),
            timestamp_start_sec=candidate.timestamp_start_sec,
            timestamp_end_sec=timestamp_sec,
            confidence=final_confidence,
            uncertainty=1.0 - final_confidence,
            decision=(EventDecision.CONFIRMED if outcome is not ShotOutcome.UNKNOWN else EventDecision.CANDIDATE),
            evidence=tuple(sorted(evidence)),
            provenance={
                "version": "shot_event_tracker_v1",
                "semantic_classes_are_supporting_evidence_only": True,
                "missing_evidence_policy": "unknown_not_missed",
            },
            attributes={
                "outcome": outcome.value,
                "resolution_reason": reason,
                "rim_bbox_px": (list(candidate.rim_bbox) if candidate.rim_bbox is not None else None),
                "had_above_rim": candidate.had_above_rim,
                "had_rim_band": candidate.had_rim_band,
                "direct_ball_observations": candidate.direct_ball_observations,
                "semantic_start": candidate.semantic_start,
            },
        )
        if outcome is ShotOutcome.MADE:
            self._cooldown_until_frame = int(frame_id) + self.cooldown_after_made_frames
        self._active = None
        return event

    def update(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        court_state: CourtState | None,
        semantic_evidence: list[dict[str, Any]] | None = None,
    ) -> list[BasketballEvent]:
        rows = list(semantic_evidence or [])
        flags = self._semantic_flags(rows)
        ball_position = None if ball_state is None else ball_state.position()
        rims = self._rim_candidates(court_state)
        rim = self._nearest_rim(ball_position, rims)
        relative = self._relative_to_rim(ball_position, rim)
        semantic_start = bool(flags["jump_shot"] or flags["layup_dunk"])
        trajectory_start = bool(ball_position is not None and rim is not None and relative["above"] and relative["near_x"])

        if self._active is None:
            if self._can_start(frame_id) and (semantic_start or trajectory_start):
                self._start(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    rim=rim,
                    ball_state=ball_state,
                    semantic_start=semantic_start,
                )
            # Crucial abstention: ball-in-basket by itself cannot manufacture a shot.
            return []

        candidate = self._active
        candidate.last_frame = int(frame_id)
        if candidate.rim_bbox is None and rim is not None:
            candidate.rim_bbox = rim
        elif rim is None:
            rim = candidate.rim_bbox
        relative = self._relative_to_rim(ball_position, rim)

        previous_ball = candidate.last_ball_position
        if ball_position is not None:
            candidate.last_ball_position = ball_position
        if ball_state is not None and ball_state.status is BallStateStatus.OBSERVED:
            candidate.direct_ball_observations += 1
        if relative["above"]:
            candidate.had_above_rim = True
            candidate.evidence.add("ball_above_rim")
        if relative["rim_band"]:
            candidate.had_rim_band = True
            candidate.evidence.add("ball_near_rim")
        if flags["jump_shot"]:
            candidate.evidence.add("player_jump_shot_semantic")
        if flags["layup_dunk"]:
            candidate.evidence.add("player_layup_dunk_semantic")
        if flags["shot_block"]:
            candidate.evidence.add("player_shot_block_semantic")

        if flags["ball_in_basket"] and (candidate.had_above_rim or candidate.had_rim_band or candidate.semantic_start):
            candidate.evidence.add("ball_in_basket_semantic")
            return [
                self._resolve(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    outcome=ShotOutcome.MADE,
                    confidence=0.92,
                    reason="ball_in_basket_supported_resolution",
                    ball_state=ball_state,
                )
            ]

        if (
            previous_ball is not None
            and ball_position is not None
            and rim is not None
            and ball_state is not None
            and ball_state.status is BallStateStatus.OBSERVED
        ):
            x1, y1, x2, y2 = rim
            width = max(1.0, x2 - x1)
            rim_y = (y1 + y2) / 2.0
            prev_x, prev_y = previous_ball
            cur_x, cur_y = ball_position
            crossed_downward = prev_y <= rim_y < cur_y
            crossed_inside = (
                (x1 - 0.20 * width) <= prev_x <= (x2 + 0.20 * width)
                and (x1 - 0.20 * width) <= cur_x <= (x2 + 0.20 * width)
            )
            if candidate.had_above_rim and crossed_downward and crossed_inside:
                candidate.evidence.add("observed_downward_rim_crossing")
                return [
                    self._resolve(
                        frame_id=frame_id,
                        timestamp_sec=timestamp_sec,
                        outcome=ShotOutcome.MADE,
                        confidence=0.80,
                        reason="observed_downward_rim_crossing",
                        ball_state=ball_state,
                    )
                ]

            if (
                candidate.had_above_rim
                and candidate.had_rim_band
                and relative["below"]
                and not relative["inside_mouth"]
            ):
                candidate.evidence.add("observed_rim_exit_outside_mouth")
                return [
                    self._resolve(
                        frame_id=frame_id,
                        timestamp_sec=timestamp_sec,
                        outcome=ShotOutcome.MISSED,
                        confidence=0.76,
                        reason="observed_rim_exit_outside_mouth",
                        ball_state=ball_state,
                    )
                ]

        if frame_id - candidate.frame_start >= self.reset_frames:
            return [
                self._resolve(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    outcome=ShotOutcome.UNKNOWN,
                    confidence=0.48 if candidate.had_above_rim else 0.38,
                    reason="shot_window_expired_without_outcome_evidence",
                    ball_state=ball_state,
                )
            ]
        return []

    def diagnostics(self) -> dict[str, Any]:
        candidate = self._active
        return {
            "version": "shot_event_tracker_v1",
            "active": candidate is not None,
            "active_start_frame": None if candidate is None else candidate.frame_start,
            "had_above_rim": False if candidate is None else candidate.had_above_rim,
            "had_rim_band": False if candidate is None else candidate.had_rim_band,
            "direct_ball_observations": 0 if candidate is None else candidate.direct_ball_observations,
            "last_start_frame": self._last_start_frame,
            "cooldown_until_frame": self._cooldown_until_frame,
            "reset_frames": self.reset_frames,
            "minimum_between_starts_frames": self.minimum_between_starts_frames,
            "cooldown_after_made_frames": self.cooldown_after_made_frames,
        }
