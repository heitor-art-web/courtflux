from __future__ import annotations

from typing import Any

from core.ball_state import BallState
from core.basketball_event import BasketballEvent, ShotOutcome
from core.court_state import CourtState
from services.p2.shot_event_tracker_base_v1 import ShotEventTrackerV1 as _BaseShotEventTrackerV1


class ShotEventTrackerV1(_BaseShotEventTrackerV1):
    """Gate-B outcome-policy revision layered over the frozen V1 tracker.

    Two deliberately conservative changes are applied:

    1. A gap-bridged downward rim-plane crossing may resolve MADE only after a later
       direct observation confirms continued downward motion with coherent rim geometry
       and horizontal proximity. ``post_gap_fall_ratio`` remains diagnostic only; no
       fitted ratio threshold is used for the decision.
    2. A rim exit observed after a ball/rim observation gap is deferred briefly instead
       of being emitted as UNKNOWN immediately. The candidate remains alive so later
       semantic/crossing evidence can resolve it; expiry without confirmation remains
       UNKNOWN.

    Deferred-window diagnostics are observational only. They are exported to support
    Gate-B analysis and do not participate in outcome decisions.
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
        rim_exit_defer_sec: float = 0.30,
    ) -> None:
        super().__init__(
            fps=fps,
            reset_time_sec=reset_time_sec,
            minimum_between_starts_sec=minimum_between_starts_sec,
            cooldown_after_made_sec=cooldown_after_made_sec,
            rearm_clear_observed_frames=rearm_clear_observed_frames,
            rearm_min_separation_sec=rearm_min_separation_sec,
        )
        requested = max(1, int(round(self.fps * float(rim_exit_defer_sec))))
        self.rim_exit_defer_frames = max(1, min(requested, self.reset_frames))
        self._deferred_rim_exit_frame: int | None = None
        self._deferred_rim_exit_deadline_frame: int | None = None
        self._deferred_outcome_diagnostics: dict[str, Any] | None = None

    def _clear_deferred_rim_exit(self) -> None:
        self._deferred_rim_exit_frame = None
        self._deferred_rim_exit_deadline_frame = None
        self._deferred_outcome_diagnostics = None

    @staticmethod
    def _normalized_x_offset(
        ball_position: tuple[float, float],
        rim: tuple[float, float, float, float],
    ) -> float:
        width = max(1.0, float(rim[2]) - float(rim[0]))
        rim_x = (float(rim[0]) + float(rim[2])) / 2.0
        return abs(float(ball_position[0]) - rim_x) / width

    def _begin_deferred_outcome_diagnostics(self, candidate: Any) -> None:
        previous_ball = candidate.last_geometry_ball_position
        previous_rim = candidate.last_geometry_rim_bbox
        previous_relative_y = None
        if previous_ball is not None and previous_rim is not None:
            previous_relative_y = float(previous_ball[1]) - self._rim_center_y(previous_rim)

        self._deferred_outcome_diagnostics = {
            "policy": "diagnostic_only_no_outcome_decision_effect",
            "frames_seen": 0,
            "direct_ball_observations": 0,
            "rim_observations": 0,
            "near_x_observations": 0,
            "inside_mouth_observations": 0,
            "below_rim_observations": 0,
            "descending_steps": 0,
            "ascending_steps": 0,
            "flat_steps": 0,
            "coherent_rim_steps": 0,
            "incoherent_rim_steps": 0,
            "ball_in_basket_semantic_frames": 0,
            "min_abs_x_offset_rim_widths": None,
            "last_abs_x_offset_rim_widths": None,
            "previous_relative_y_px": previous_relative_y,
            "previous_rim_bbox_px": None if previous_rim is None else list(previous_rim),
        }

    def _observe_deferred_outcome_frame(
        self,
        *,
        ball_state: BallState | None,
        court_state: CourtState | None,
        semantic_evidence: list[dict[str, Any]] | None,
    ) -> None:
        diagnostics = self._deferred_outcome_diagnostics
        if diagnostics is None:
            return

        diagnostics["frames_seen"] = int(diagnostics["frames_seen"]) + 1
        flags = self._semantic_flags(list(semantic_evidence or []))
        if flags["ball_in_basket"]:
            diagnostics["ball_in_basket_semantic_frames"] = (
                int(diagnostics["ball_in_basket_semantic_frames"]) + 1
            )

        if not self._direct_observed(ball_state):
            return
        ball_position = None if ball_state is None else ball_state.position()
        if ball_position is None:
            return

        diagnostics["direct_ball_observations"] = (
            int(diagnostics["direct_ball_observations"]) + 1
        )
        rim = self._nearest_rim(ball_position, self._rim_candidates(court_state))
        if rim is None:
            return

        diagnostics["rim_observations"] = int(diagnostics["rim_observations"]) + 1
        relative = self._relative_to_rim(ball_position, rim)
        if relative["near_x"]:
            diagnostics["near_x_observations"] = int(diagnostics["near_x_observations"]) + 1
        if relative["inside_mouth"]:
            diagnostics["inside_mouth_observations"] = (
                int(diagnostics["inside_mouth_observations"]) + 1
            )
        if relative["below"]:
            diagnostics["below_rim_observations"] = (
                int(diagnostics["below_rim_observations"]) + 1
            )

        x_offset = self._normalized_x_offset(ball_position, rim)
        diagnostics["last_abs_x_offset_rim_widths"] = x_offset
        previous_min = diagnostics["min_abs_x_offset_rim_widths"]
        if previous_min is None or x_offset < float(previous_min):
            diagnostics["min_abs_x_offset_rim_widths"] = x_offset

        previous_rim_raw = diagnostics["previous_rim_bbox_px"]
        if isinstance(previous_rim_raw, list) and len(previous_rim_raw) >= 4:
            previous_rim = tuple(float(value) for value in previous_rim_raw[:4])
            if self._rim_boxes_overlap(previous_rim, rim):
                diagnostics["coherent_rim_steps"] = int(diagnostics["coherent_rim_steps"]) + 1
            else:
                diagnostics["incoherent_rim_steps"] = int(diagnostics["incoherent_rim_steps"]) + 1

        relative_y = float(ball_position[1]) - self._rim_center_y(rim)
        previous_relative_y = diagnostics["previous_relative_y_px"]
        if previous_relative_y is not None:
            delta = relative_y - float(previous_relative_y)
            if delta > 1e-6:
                diagnostics["descending_steps"] = int(diagnostics["descending_steps"]) + 1
            elif delta < -1e-6:
                diagnostics["ascending_steps"] = int(diagnostics["ascending_steps"]) + 1
            else:
                diagnostics["flat_steps"] = int(diagnostics["flat_steps"]) + 1

        diagnostics["previous_relative_y_px"] = relative_y
        diagnostics["previous_rim_bbox_px"] = list(rim)

    def _deferred_outcome_diagnostics_snapshot(self) -> dict[str, Any] | None:
        diagnostics = self._deferred_outcome_diagnostics
        if diagnostics is None:
            return None
        return {
            key: value
            for key, value in diagnostics.items()
            if key not in {"previous_relative_y_px", "previous_rim_bbox_px"}
        }

    def _record_current_direct_observation(
        self,
        candidate: Any,
        *,
        frame_id: int,
        ball_state: BallState | None,
        court_state: CourtState | None,
    ) -> None:
        if not self._direct_observed(ball_state):
            return
        ball_position = None if ball_state is None else ball_state.position()
        if ball_position is None:
            return
        rim = self._nearest_rim(ball_position, self._rim_candidates(court_state))
        self._record_direct_observation(
            candidate,
            frame_id=frame_id,
            ball_position=ball_position,
            rim=rim,
        )

    def _restore_deferred_candidate(
        self,
        candidate: Any,
        *,
        last_resolved_frame: int,
        awaiting_rearm: bool,
        rearm_clear_frames: int,
        frame_id: int,
        ball_state: BallState | None,
        court_state: CourtState | None,
    ) -> None:
        # Base V1 has already emitted/cleared the candidate. Restore exactly the
        # pre-resolution tracker state because this policy intentionally defers it.
        self._active = candidate
        self._last_resolved_frame = last_resolved_frame
        self._awaiting_rearm = awaiting_rearm
        self._rearm_clear_frames = rearm_clear_frames
        self._record_current_direct_observation(
            candidate,
            frame_id=frame_id,
            ball_state=ball_state,
            court_state=court_state,
        )

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
        event = super()._resolve(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            outcome=outcome,
            confidence=confidence,
            reason=reason,
            ball_state=ball_state,
        )
        event.provenance["outcome_policy_revision"] = "shot_outcome_v1_1"
        event.provenance["post_gap_motion_policy"] = (
            "continued_relative_descent_plus_coherent_rim_geometry_and_near_x"
        )
        event.provenance["rim_exit_after_gap_policy"] = "defer_then_confirm_or_unknown"
        event.provenance["deferred_outcome_diagnostics_policy"] = (
            "diagnostic_only_no_outcome_decision_effect"
        )
        event.attributes["post_gap_motion_validated_for_outcome"] = (
            reason == "pending_bridge_post_gap_downward_motion_confirmed"
        )
        event.attributes["rim_exit_deferred"] = self._deferred_rim_exit_frame is not None
        event.attributes["rim_exit_deferred_frame"] = self._deferred_rim_exit_frame
        event.attributes["rim_exit_deferred_deadline_frame"] = (
            self._deferred_rim_exit_deadline_frame
        )
        event.attributes["deferred_outcome_diagnostics"] = (
            self._deferred_outcome_diagnostics_snapshot()
        )
        return event

    def _settle_pending_bridge(
        self,
        candidate: Any,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        ball_position: tuple[float, float] | None,
        rim: tuple[float, float, float, float] | None,
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

        relative = self._relative_to_rim(ball_position, rim)
        rim_geometry_coherent = self._rim_boxes_overlap(bridge.first_post_rim, rim)
        downward_confirmed = post_rate is not None and post_rate > 0.0
        spatially_coherent = rim_geometry_coherent and bool(relative["near_x"])

        if downward_confirmed and spatially_coherent:
            candidate.evidence.add("post_gap_downward_motion_confirmed")
            candidate.evidence.add("post_gap_rim_geometry_coherent")
            return [
                self._resolve(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    outcome=ShotOutcome.MADE,
                    confidence=0.78,
                    reason="pending_bridge_post_gap_downward_motion_confirmed",
                    ball_state=ball_state,
                )
            ]

        if not downward_confirmed:
            candidate.evidence.add("pending_bridge_post_gap_not_descending")
        if not rim_geometry_coherent:
            candidate.evidence.add("pending_bridge_post_gap_rim_geometry_incoherent")
        if not relative["near_x"]:
            candidate.evidence.add("pending_bridge_post_gap_ball_not_near_rim")
        return [
            self._resolve(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                outcome=ShotOutcome.UNKNOWN,
                confidence=0.50,
                reason="pending_bridge_post_gap_confirmation_incoherent",
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
        if self._deferred_rim_exit_frame is not None:
            self._observe_deferred_outcome_frame(
                ball_state=ball_state,
                court_state=court_state,
                semantic_evidence=semantic_evidence,
            )

        candidate_before = self._active
        last_resolved_before = self._last_resolved_frame
        awaiting_rearm_before = self._awaiting_rearm
        rearm_clear_before = self._rearm_clear_frames

        events = super().update(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            ball_state=ball_state,
            court_state=court_state,
            semantic_evidence=semantic_evidence,
        )

        if candidate_before is not None and len(events) == 1:
            event = events[0]
            attributes = dict(event.attributes or {})
            outcome = str(attributes.get("outcome") or "unknown")
            reason = str(attributes.get("resolution_reason") or "")

            rim_exit_unknown = bool(
                outcome == ShotOutcome.UNKNOWN.value
                and reason == "rim_exit_after_observation_gap_insufficient_for_miss"
            )
            deferred_window_expiry = bool(
                self._deferred_rim_exit_frame is not None
                and outcome == ShotOutcome.UNKNOWN.value
                and reason == "shot_window_expired_without_outcome_evidence"
            )

            if rim_exit_unknown or deferred_window_expiry:
                if self._deferred_rim_exit_frame is None:
                    self._deferred_rim_exit_frame = int(frame_id)
                    self._deferred_rim_exit_deadline_frame = (
                        int(frame_id) + self.rim_exit_defer_frames
                    )
                    candidate_before.evidence.add("rim_exit_after_observation_gap_deferred")
                    self._begin_deferred_outcome_diagnostics(candidate_before)
                    self._observe_deferred_outcome_frame(
                        ball_state=ball_state,
                        court_state=court_state,
                        semantic_evidence=semantic_evidence,
                    )

                self._restore_deferred_candidate(
                    candidate_before,
                    last_resolved_frame=last_resolved_before,
                    awaiting_rearm=awaiting_rearm_before,
                    rearm_clear_frames=rearm_clear_before,
                    frame_id=frame_id,
                    ball_state=ball_state,
                    court_state=court_state,
                )

                deadline = self._deferred_rim_exit_deadline_frame
                if deadline is not None and int(frame_id) >= deadline:
                    resolved = self._resolve(
                        frame_id=frame_id,
                        timestamp_sec=timestamp_sec,
                        outcome=ShotOutcome.UNKNOWN,
                        confidence=0.50,
                        reason="deferred_rim_exit_expired_without_confirmation",
                        ball_state=ball_state,
                    )
                    self._clear_deferred_rim_exit()
                    return [resolved]
                return []

        if events:
            self._clear_deferred_rim_exit()
            return events

        deadline = self._deferred_rim_exit_deadline_frame
        if (
            self._deferred_rim_exit_frame is not None
            and deadline is not None
            and int(frame_id) >= deadline
            and self._active is not None
        ):
            resolved = self._resolve(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                outcome=ShotOutcome.UNKNOWN,
                confidence=0.50,
                reason="deferred_rim_exit_expired_without_confirmation",
                ball_state=ball_state,
            )
            self._clear_deferred_rim_exit()
            return [resolved]

        return []

    def reset(self, *, reason: str = "external_reset") -> None:
        self._clear_deferred_rim_exit()
        super().reset(reason=reason)

    def diagnostics(self) -> dict[str, Any]:
        result = dict(super().diagnostics())
        result.update(
            {
                "outcome_policy_revision": "shot_outcome_v1_1",
                "rim_exit_defer_frames": self.rim_exit_defer_frames,
                "rim_exit_deferred": self._deferred_rim_exit_frame is not None,
                "rim_exit_deferred_frame": self._deferred_rim_exit_frame,
                "rim_exit_deferred_deadline_frame": self._deferred_rim_exit_deadline_frame,
                "deferred_outcome_diagnostics": self._deferred_outcome_diagnostics_snapshot(),
            }
        )
        return result
