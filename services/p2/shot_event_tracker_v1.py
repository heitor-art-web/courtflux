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


RimBBox = tuple[float, float, float, float]


@dataclass(slots=True)
class _PendingBridge:
    pre_frame: int
    pre_ball: tuple[float, float]
    pre_rim: RimBBox
    first_post_frame: int
    first_post_ball: tuple[float, float]
    first_post_rim: RimBBox
    crossing_x_px: float
    crossing_fraction: float
    ball_gap_frames: int
    rim_gap_frames: int
    pre_gap_relative_fall_px_per_frame: float | None = None


@dataclass(slots=True)
class _ShotCandidate:
    frame_start: int
    timestamp_start_sec: float | None
    last_frame: int
    rim_bbox: RimBBox | None = None
    had_above_rim: bool = False
    had_rim_band: bool = False
    direct_ball_observations: int = 0
    semantic_start: bool = False
    evidence: set[str] = field(default_factory=set)

    last_direct_ball_frame: int | None = None
    previous_geometry_ball_position: tuple[float, float] | None = None
    previous_geometry_rim_bbox: RimBBox | None = None
    previous_geometry_frame: int | None = None
    last_geometry_ball_position: tuple[float, float] | None = None
    last_geometry_rim_bbox: RimBBox | None = None
    last_geometry_frame: int | None = None

    current_rim_gap_frames: int = 0
    max_ball_gap_frames: int = 0
    max_rim_gap_frames: int = 0
    outcome_geometry_tainted: bool = False

    pending_bridge: _PendingBridge | None = None
    bridge_count: int = 0
    post_gap_fall_ratio: float | None = None
    pre_gap_relative_fall_px_per_frame: float | None = None
    post_gap_relative_fall_px_per_frame: float | None = None
    last_crossing_x_px: float | None = None
    last_crossing_fraction: float | None = None


class ShotEventTrackerV1:
    """Conservative temporal shot candidate/resolution state machine.

    P1 predictions preserve continuity but do not independently prove an outcome.
    Direct observed ball/rim samples drive geometry.

    Direct consecutive observations may resolve MADE/MISSED immediately. If a downward
    rim-plane crossing is bracketed across an observation or rim-evidence gap, the tracker
    creates a pending bridge and waits for one more direct geometric sample. The post-gap
    fall ratio is exported as diagnostic evidence only; it is not yet calibrated to
    classify MADE. Therefore a settled pending bridge resolves UNKNOWN unless independent
    semantic ball-in-basket evidence confirms MADE.
    """

    def __init__(
        self,
        *,
        fps: float = 30.0,
        reset_time_sec: float = 1.7,
        minimum_between_starts_sec: float = 0.5,
        cooldown_after_made_sec: float = 0.5,
        rearm_clear_observed_frames: int = 4,
        rearm_min_separation_sec: float = 0.24,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.reset_frames = max(1, int(round(self.fps * float(reset_time_sec))))
        self.minimum_between_starts_frames = max(
            1, int(round(self.fps * float(minimum_between_starts_sec)))
        )
        self.cooldown_after_made_frames = max(
            1, int(round(self.fps * float(cooldown_after_made_sec)))
        )
        self.rearm_clear_observed_frames = max(1, int(rearm_clear_observed_frames))
        self.rearm_min_separation_frames = max(
            1, int(round(self.fps * float(rearm_min_separation_sec)))
        )

        self._active: _ShotCandidate | None = None
        self._last_start_frame = -10**9
        self._last_resolved_frame = -10**9
        self._cooldown_until_frame = -1
        self._awaiting_rearm = False
        self._rearm_clear_frames = 0
        self._rearm_count = 0
        self._reset_count = 0
        self._last_reset_reason: str | None = None

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
            try:
                confidence = max(0.0, min(float(row.get("confidence", 0.0)), 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
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
    def _rim_candidates(court_state: CourtState | None) -> list[RimBBox]:
        if court_state is None:
            return []
        rims: list[RimBBox] = []
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
        rims: list[RimBBox],
    ) -> RimBBox | None:
        if not rims:
            return None
        if ball_position is None:
            return rims[0]
        bx, by = ball_position
        return min(
            rims,
            key=lambda rim: (
                ((rim[0] + rim[2]) / 2.0 - bx) ** 2
                + ((rim[1] + rim[3]) / 2.0 - by) ** 2
            ),
        )

    @staticmethod
    def _relative_to_rim(
        ball_position: tuple[float, float] | None,
        rim: RimBBox | None,
    ) -> dict[str, bool]:
        result = {
            "above": False,
            "near_x": False,
            "rim_band": False,
            "below": False,
            "inside_mouth": False,
        }
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
        result["rim_band"] = (
            (y1 - 0.8 * height) <= by <= (y2 + 0.8 * height)
            and result["near_x"]
        )
        result["below"] = by > rim_y + 0.65 * height
        result["inside_mouth"] = (x1 - 0.20 * width) <= bx <= (x2 + 0.20 * width)
        return result

    @staticmethod
    def _direct_observed(ball_state: BallState | None) -> bool:
        return bool(
            ball_state is not None
            and ball_state.status is BallStateStatus.OBSERVED
            and ball_state.position() is not None
        )

    @staticmethod
    def _rim_center_y(rim: RimBBox) -> float:
        return (rim[1] + rim[3]) / 2.0

    @staticmethod
    def _rim_boxes_overlap(a: RimBBox, b: RimBBox) -> bool:
        return bool(
            min(a[2], b[2]) > max(a[0], b[0])
            and min(a[3], b[3]) > max(a[1], b[1])
        )

    @classmethod
    def _crossing_at_rim_plane(
        cls,
        *,
        previous_ball: tuple[float, float],
        previous_rim: RimBBox,
        current_ball: tuple[float, float],
        current_rim: RimBBox,
    ) -> dict[str, float | bool | None]:
        prev_x, prev_y = previous_ball
        cur_x, cur_y = current_ball
        prev_rim_y = cls._rim_center_y(previous_rim)
        cur_rim_y = cls._rim_center_y(current_rim)

        prev_rel_y = prev_y - prev_rim_y
        cur_rel_y = cur_y - cur_rim_y
        denominator = cur_rel_y - prev_rel_y
        if not (prev_rel_y <= 0.0 < cur_rel_y) or denominator <= 0.0:
            return {
                "crossed_downward": False,
                "inside_mouth": False,
                "fraction": None,
                "x_px": None,
            }

        fraction = -prev_rel_y / denominator
        if not 0.0 <= fraction <= 1.0:
            return {
                "crossed_downward": False,
                "inside_mouth": False,
                "fraction": None,
                "x_px": None,
            }

        cross_x = prev_x + fraction * (cur_x - prev_x)
        cross_x1 = previous_rim[0] + fraction * (current_rim[0] - previous_rim[0])
        cross_x2 = previous_rim[2] + fraction * (current_rim[2] - previous_rim[2])
        width = max(1.0, cross_x2 - cross_x1)
        inside = (cross_x1 - 0.20 * width) <= cross_x <= (cross_x2 + 0.20 * width)
        return {
            "crossed_downward": True,
            "inside_mouth": bool(inside),
            "fraction": float(fraction),
            "x_px": float(cross_x),
        }

    @classmethod
    def _relative_fall_per_frame(
        cls,
        *,
        first_frame: int,
        first_ball: tuple[float, float],
        first_rim: RimBBox,
        second_frame: int,
        second_ball: tuple[float, float],
        second_rim: RimBBox,
    ) -> float | None:
        delta_frames = int(second_frame) - int(first_frame)
        if delta_frames <= 0:
            return None
        first_rel = first_ball[1] - cls._rim_center_y(first_rim)
        second_rel = second_ball[1] - cls._rim_center_y(second_rim)
        return (second_rel - first_rel) / float(delta_frames)

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

    def _record_direct_observation(
        self,
        candidate: _ShotCandidate,
        *,
        frame_id: int,
        ball_position: tuple[float, float],
        rim: RimBBox | None,
    ) -> None:
        candidate.last_direct_ball_frame = int(frame_id)
        if rim is None:
            return
        candidate.previous_geometry_ball_position = candidate.last_geometry_ball_position
        candidate.previous_geometry_rim_bbox = candidate.last_geometry_rim_bbox
        candidate.previous_geometry_frame = candidate.last_geometry_frame
        candidate.last_geometry_ball_position = ball_position
        candidate.last_geometry_rim_bbox = rim
        candidate.last_geometry_frame = int(frame_id)

    def _start(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        rim: RimBBox | None,
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
        if self._direct_observed(ball_state):
            position = ball_state.position()
            if position is not None:
                candidate.direct_ball_observations = 1
                self._record_direct_observation(
                    candidate,
                    frame_id=frame_id,
                    ball_position=position,
                    rim=rim,
                )
                relative = self._relative_to_rim(position, rim)
                if relative["above"]:
                    candidate.had_above_rim = True
                    candidate.evidence.add("ball_above_rim")
                if relative["rim_band"]:
                    candidate.had_rim_band = True
                    candidate.evidence.add("ball_near_rim")
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

        bridge = candidate.pending_bridge
        event = BasketballEvent(
            event_id=f"shot-attempt:{candidate.frame_start}:{frame_id}",
            event_type=BasketballEventType.SHOT_ATTEMPT,
            frame_start=candidate.frame_start,
            frame_end=int(frame_id),
            timestamp_start_sec=candidate.timestamp_start_sec,
            timestamp_end_sec=timestamp_sec,
            confidence=final_confidence,
            uncertainty=1.0 - final_confidence,
            decision=(
                EventDecision.CONFIRMED
                if outcome is not ShotOutcome.UNKNOWN
                else EventDecision.CANDIDATE
            ),
            evidence=tuple(sorted(evidence)),
            provenance={
                "version": "shot_event_tracker_v1",
                "semantic_classes_are_supporting_evidence_only": True,
                "missing_evidence_policy": "unknown_not_missed",
                "outcome_geometry_policy": "rim_plane_intersection_plus_pending_bridge",
                "post_gap_fall_ratio_policy": "diagnostic_only_unvalidated",
                "rearm_policy": "direct_observed_clearance_plus_min_separation",
            },
            attributes={
                "outcome": outcome.value,
                "resolution_reason": reason,
                "rim_bbox_px": list(candidate.rim_bbox) if candidate.rim_bbox is not None else None,
                "had_above_rim": candidate.had_above_rim,
                "had_rim_band": candidate.had_rim_band,
                "direct_ball_observations": candidate.direct_ball_observations,
                "max_ball_observation_gap_frames": candidate.max_ball_gap_frames,
                "max_rim_evidence_gap_frames": candidate.max_rim_gap_frames,
                "outcome_geometry_tainted": candidate.outcome_geometry_tainted,
                "semantic_start": candidate.semantic_start,
                "pending_bridge_used": candidate.bridge_count > 0,
                "pending_bridge_count": candidate.bridge_count,
                "crossing_x_px": candidate.last_crossing_x_px,
                "crossing_fraction": candidate.last_crossing_fraction,
                "post_gap_fall_ratio": candidate.post_gap_fall_ratio,
                "pre_gap_relative_fall_px_per_frame": candidate.pre_gap_relative_fall_px_per_frame,
                "post_gap_relative_fall_px_per_frame": candidate.post_gap_relative_fall_px_per_frame,
                "bridge_ball_gap_frames": None if bridge is None else bridge.ball_gap_frames,
                "bridge_rim_gap_frames": None if bridge is None else bridge.rim_gap_frames,
                "post_gap_fall_ratio_validated_for_outcome": False,
            },
        )

        if outcome is ShotOutcome.MADE:
            self._cooldown_until_frame = int(frame_id) + self.cooldown_after_made_frames
        self._active = None
        self._last_resolved_frame = int(frame_id)
        self._awaiting_rearm = True
        self._rearm_clear_frames = 0
        return event

    def _update_rearm(
        self,
        *,
        frame_id: int,
        ball_state: BallState | None,
        relative: dict[str, bool],
        semantic_start: bool,
        trajectory_start: bool,
    ) -> bool:
        if not self._awaiting_rearm:
            return False

        direct_observed = self._direct_observed(ball_state)
        clear_of_previous_shot_zone = bool(
            direct_observed
            and not semantic_start
            and not trajectory_start
            and not relative.get("rim_band", False)
        )
        if clear_of_previous_shot_zone:
            self._rearm_clear_frames += 1
        elif semantic_start or trajectory_start or relative.get("rim_band", False):
            self._rearm_clear_frames = 0

        separated = (
            int(frame_id) - self._last_resolved_frame
            >= self.rearm_min_separation_frames
        )
        if separated and self._rearm_clear_frames >= self.rearm_clear_observed_frames:
            self._awaiting_rearm = False
            self._rearm_clear_frames = 0
            self._rearm_count += 1
            return True
        return False

    def _start_pending_bridge(
        self,
        candidate: _ShotCandidate,
        *,
        previous_ball: tuple[float, float],
        previous_rim: RimBBox,
        previous_frame: int,
        current_ball: tuple[float, float],
        current_rim: RimBBox,
        current_frame: int,
        crossing: dict[str, float | bool | None],
        ball_gap_frames: int,
        rim_gap_frames: int,
    ) -> None:
        pre_gap_rate = None
        if (
            candidate.previous_geometry_ball_position is not None
            and candidate.previous_geometry_rim_bbox is not None
            and candidate.previous_geometry_frame is not None
        ):
            pre_gap_rate = self._relative_fall_per_frame(
                first_frame=candidate.previous_geometry_frame,
                first_ball=candidate.previous_geometry_ball_position,
                first_rim=candidate.previous_geometry_rim_bbox,
                second_frame=previous_frame,
                second_ball=previous_ball,
                second_rim=previous_rim,
            )

        candidate.pending_bridge = _PendingBridge(
            pre_frame=previous_frame,
            pre_ball=previous_ball,
            pre_rim=previous_rim,
            first_post_frame=current_frame,
            first_post_ball=current_ball,
            first_post_rim=current_rim,
            crossing_x_px=float(crossing["x_px"]),
            crossing_fraction=float(crossing["fraction"]),
            ball_gap_frames=int(ball_gap_frames),
            rim_gap_frames=int(rim_gap_frames),
            pre_gap_relative_fall_px_per_frame=pre_gap_rate,
        )
        candidate.bridge_count += 1
        candidate.last_crossing_x_px = float(crossing["x_px"])
        candidate.last_crossing_fraction = float(crossing["fraction"])
        candidate.pre_gap_relative_fall_px_per_frame = pre_gap_rate
        candidate.evidence.add("pending_rim_crossing_bridge")

    def _settle_pending_bridge(
        self,
        candidate: _ShotCandidate,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        ball_position: tuple[float, float] | None,
        rim: RimBBox | None,
    ) -> list[BasketballEvent] | None:
        bridge = candidate.pending_bridge
        if bridge is None or not self._direct_observed(ball_state):
            return None
        if ball_position is None or rim is None:
            return None
        if int(frame_id) <= bridge.first_post_frame:
            return None

        post_rate = self._relative_fall_per_frame(
            first_frame=bridge.first_post_frame,
            first_ball=bridge.first_post_ball,
            first_rim=bridge.first_post_rim,
            second_frame=frame_id,
            second_ball=ball_position,
            second_rim=rim,
        )
        pre_rate = bridge.pre_gap_relative_fall_px_per_frame
        ratio = None
        if pre_rate is not None and pre_rate > 1e-6 and post_rate is not None:
            ratio = post_rate / pre_rate

        candidate.post_gap_relative_fall_px_per_frame = post_rate
        candidate.post_gap_fall_ratio = ratio
        candidate.evidence.add("post_gap_fall_ratio_measured")
        return [
            self._resolve(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                outcome=ShotOutcome.UNKNOWN,
                confidence=0.50,
                reason="pending_bridge_post_gap_ratio_unvalidated",
                ball_state=ball_state,
            )
        ]

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
        direct_observed = self._direct_observed(ball_state)
        semantic_start = bool(flags["jump_shot"] or flags["layup_dunk"])
        trajectory_start = bool(
            ball_position is not None
            and rim is not None
            and relative["above"]
            and relative["near_x"]
        )
        start_signal = bool(semantic_start or trajectory_start)

        if self._active is None:
            if self._awaiting_rearm:
                self._update_rearm(
                    frame_id=frame_id,
                    ball_state=ball_state,
                    relative=relative,
                    semantic_start=semantic_start,
                    trajectory_start=trajectory_start,
                )
                return []
            if self._can_start(frame_id) and start_signal:
                self._start(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    rim=rim,
                    ball_state=ball_state,
                    semantic_start=semantic_start,
                )
            return []

        candidate = self._active
        candidate.last_frame = int(frame_id)
        if candidate.rim_bbox is None and rim is not None:
            candidate.rim_bbox = rim

        rim_gap_before_current = candidate.current_rim_gap_frames
        if rim is None:
            if candidate.had_above_rim or candidate.had_rim_band:
                candidate.outcome_geometry_tainted = True
            candidate.current_rim_gap_frames += 1
            candidate.max_rim_gap_frames = max(
                candidate.max_rim_gap_frames,
                candidate.current_rim_gap_frames,
            )
        else:
            candidate.current_rim_gap_frames = 0

        previous_observed_ball = candidate.last_geometry_ball_position
        previous_observed_rim = candidate.last_geometry_rim_bbox
        previous_observed_frame = candidate.last_geometry_frame
        geometry_gap_frames = 0
        if previous_observed_frame is not None:
            geometry_gap_frames = max(0, int(frame_id) - int(previous_observed_frame) - 1)

        ball_gap_frames = 0
        if direct_observed and candidate.last_direct_ball_frame is not None:
            ball_gap_frames = max(0, int(frame_id) - int(candidate.last_direct_ball_frame) - 1)
            if ball_gap_frames > 0 and (candidate.had_above_rim or candidate.had_rim_band):
                candidate.outcome_geometry_tainted = True
            candidate.max_ball_gap_frames = max(
                candidate.max_ball_gap_frames,
                ball_gap_frames,
            )

        if direct_observed and rim is not None:
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

        if flags["ball_in_basket"] and (
            candidate.had_above_rim
            or candidate.had_rim_band
            or candidate.semantic_start
        ):
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

        settled = self._settle_pending_bridge(
            candidate,
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            ball_state=ball_state,
            ball_position=ball_position,
            rim=rim,
        )
        if settled is not None:
            return settled

        if (
            direct_observed
            and ball_position is not None
            and rim is not None
            and previous_observed_ball is not None
            and previous_observed_rim is not None
            and previous_observed_frame is not None
        ):
            crossing = self._crossing_at_rim_plane(
                previous_ball=previous_observed_ball,
                previous_rim=previous_observed_rim,
                current_ball=ball_position,
                current_rim=rim,
            )
            crossed_inside = bool(
                crossing["crossed_downward"] and crossing["inside_mouth"]
            )

            if candidate.had_above_rim and crossed_inside:
                candidate.last_crossing_x_px = float(crossing["x_px"])
                candidate.last_crossing_fraction = float(crossing["fraction"])
                geometry_coherent = self._rim_boxes_overlap(previous_observed_rim, rim)

                if ball_gap_frames == 0 and geometry_gap_frames == 0 and rim_gap_before_current == 0:
                    candidate.evidence.add("observed_downward_rim_plane_crossing")
                    return [
                        self._resolve(
                            frame_id=frame_id,
                            timestamp_sec=timestamp_sec,
                            outcome=ShotOutcome.MADE,
                            confidence=0.80,
                            reason="observed_downward_rim_plane_crossing",
                            ball_state=ball_state,
                        )
                    ]

                if geometry_coherent:
                    self._start_pending_bridge(
                        candidate,
                        previous_ball=previous_observed_ball,
                        previous_rim=previous_observed_rim,
                        previous_frame=previous_observed_frame,
                        current_ball=ball_position,
                        current_rim=rim,
                        current_frame=frame_id,
                        crossing=crossing,
                        ball_gap_frames=ball_gap_frames,
                        rim_gap_frames=max(rim_gap_before_current, geometry_gap_frames),
                    )
                    self._record_direct_observation(
                        candidate,
                        frame_id=frame_id,
                        ball_position=ball_position,
                        rim=rim,
                    )
                    return []

            if (
                candidate.had_above_rim
                and candidate.had_rim_band
                and relative["below"]
                and not relative["inside_mouth"]
            ):
                if (
                    ball_gap_frames > 0
                    or rim_gap_before_current > 0
                    or candidate.outcome_geometry_tainted
                ):
                    candidate.evidence.add("rim_exit_after_observation_gap_insufficient_for_miss")
                    return [
                        self._resolve(
                            frame_id=frame_id,
                            timestamp_sec=timestamp_sec,
                            outcome=ShotOutcome.UNKNOWN,
                            confidence=0.48,
                            reason="rim_exit_after_observation_gap_insufficient_for_miss",
                            ball_state=ball_state,
                        )
                    ]
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

        if direct_observed and ball_position is not None:
            self._record_direct_observation(
                candidate,
                frame_id=frame_id,
                ball_position=ball_position,
                rim=rim,
            )

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

    def reset(self, *, reason: str = "external_reset") -> None:
        self._active = None
        self._last_start_frame = -10**9
        self._last_resolved_frame = -10**9
        self._cooldown_until_frame = -1
        self._awaiting_rearm = False
        self._rearm_clear_frames = 0
        self._reset_count += 1
        self._last_reset_reason = str(reason)

    def diagnostics(self) -> dict[str, Any]:
        candidate = self._active
        bridge = None if candidate is None else candidate.pending_bridge
        return {
            "version": "shot_event_tracker_v1",
            "active": candidate is not None,
            "active_start_frame": None if candidate is None else candidate.frame_start,
            "had_above_rim": False if candidate is None else candidate.had_above_rim,
            "had_rim_band": False if candidate is None else candidate.had_rim_band,
            "direct_ball_observations": 0 if candidate is None else candidate.direct_ball_observations,
            "last_direct_observed_frame": None if candidate is None else candidate.last_direct_ball_frame,
            "last_geometry_observed_frame": None if candidate is None else candidate.last_geometry_frame,
            "current_rim_gap_frames": 0 if candidate is None else candidate.current_rim_gap_frames,
            "max_ball_observation_gap_frames": 0 if candidate is None else candidate.max_ball_gap_frames,
            "max_rim_evidence_gap_frames": 0 if candidate is None else candidate.max_rim_gap_frames,
            "outcome_geometry_tainted": False if candidate is None else candidate.outcome_geometry_tainted,
            "pending_bridge": bridge is not None,
            "pending_bridge_created_frame": None if bridge is None else bridge.first_post_frame,
            "pending_bridge_crossing_x_px": None if bridge is None else bridge.crossing_x_px,
            "post_gap_fall_ratio": None if candidate is None else candidate.post_gap_fall_ratio,
            "post_gap_fall_ratio_validated_for_outcome": False,
            "last_start_frame": self._last_start_frame,
            "last_resolved_frame": self._last_resolved_frame,
            "cooldown_until_frame": self._cooldown_until_frame,
            "awaiting_rearm": self._awaiting_rearm,
            "rearm_clear_observed_frames": self._rearm_clear_frames,
            "rearm_required_clear_observed_frames": self.rearm_clear_observed_frames,
            "rearm_min_separation_frames": self.rearm_min_separation_frames,
            "rearm_count": self._rearm_count,
            "reset_count": self._reset_count,
            "last_reset_reason": self._last_reset_reason,
        }
