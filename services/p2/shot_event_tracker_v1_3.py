from __future__ import annotations

from typing import Any

from core.ball_state import BallState
from core.basketball_event import BasketballEvent, EventDecision, ShotOutcome
from core.court_state import CourtState
from services.p2.shot_event_tracker_v1_2 import ShotEventTrackerV12


class ShotEventTrackerV13(ShotEventTrackerV12):
    """V6-focused shot lifecycle policy layered over V1.2.

    The V6 validation trace exposed four concrete failure modes which are handled here
    without changing the frozen V1/V1.1/V1.2 implementations:

    * a supported airborne flight can reach the rim band without satisfying the strict
      ``above + near_x`` start gate (the V6 P4 false negative);
    * a trajectory-only start that never obtains any direct above/rim-band evidence is
      not published as a shot attempt (the V6 870.3 false positive);
    * a deferred rim exit with tight mouth alignment, coherent geometry and sustained
      downward/below-rim evidence may confirm MADE (the V6 free throw);
    * a long, well-observed above+rim-band attempt that expires without made evidence is
      resolved MISSED, enabling the existing possession-driven rebound path.

    Rearm is causal in V1.3: direct visual clearance is still required, but a new shot is
    not armed until control has also been reacquired after the previous resolved attempt.
    """

    OUTCOME_POLICY_REVISION = "shot_outcome_v1_3_v6_batch"
    MIN_STRONG_EXPIRY_OBSERVATIONS = 12
    DEFERRED_TIGHT_MOUTH_MAX_X_RIM_WIDTHS = 0.25
    DEFERRED_MIN_BELOW_OBSERVATIONS = 3
    DEFERRED_MIN_DESCENDING_STEPS = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._supported_flight_hint = False
        self._control_reacquired_frame: int | None = None
        self._suppressed_public_shots = 0
        self._last_suppressed_reason: str | None = None
        self._supported_flight_bridge_starts = 0

    def set_supported_flight_hint(self, supported: bool) -> None:
        """Provide the previous-frame generic airborne-control diagnostic.

        The hint is consumed once by ``update`` and cannot independently publish a shot;
        direct ball + rim-band geometry is still required in the current frame.
        """
        self._supported_flight_hint = bool(supported)

    def confirm_control_reacquisition(self, frame_id: int) -> None:
        """Record a confirmed post-shot possession/control reacquisition."""
        self._control_reacquired_frame = int(frame_id)

    @staticmethod
    def _deferred_confirmation_reason(
        diagnostics: dict[str, Any] | None,
    ) -> str | None:
        inherited = ShotEventTrackerV12._deferred_confirmation_reason(diagnostics)
        if inherited is not None or not diagnostics:
            return inherited

        rim_observations = int(diagnostics.get("rim_observations", 0) or 0)
        inside_mouth_observations = int(
            diagnostics.get("inside_mouth_observations", 0) or 0
        )
        below_rim_observations = int(
            diagnostics.get("below_rim_observations", 0) or 0
        )
        descending_steps = int(diagnostics.get("descending_steps", 0) or 0)
        incoherent_rim_steps = int(diagnostics.get("incoherent_rim_steps", 0) or 0)
        raw_min_x = diagnostics.get("min_abs_x_offset_rim_widths")
        try:
            min_x = None if raw_min_x is None else float(raw_min_x)
        except (TypeError, ValueError):
            min_x = None

        if incoherent_rim_steps > 0 or min_x is None:
            return None
        if (
            rim_observations >= 3
            and inside_mouth_observations >= 1
            and below_rim_observations
            >= ShotEventTrackerV13.DEFERRED_MIN_BELOW_OBSERVATIONS
            and descending_steps >= ShotEventTrackerV13.DEFERRED_MIN_DESCENDING_STEPS
            and min_x <= ShotEventTrackerV13.DEFERRED_TIGHT_MOUTH_MAX_X_RIM_WIDTHS
        ):
            return "deferred_rim_exit_mouth_below_descent_confirmed"
        return None

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
            int(frame_id) - int(self._last_resolved_frame)
            >= int(self.rearm_min_separation_frames)
        )
        control_reacquired = bool(
            self._control_reacquired_frame is not None
            and int(self._control_reacquired_frame) >= int(self._last_resolved_frame)
        )
        if (
            separated
            and control_reacquired
            and self._rearm_clear_frames >= self.rearm_clear_observed_frames
        ):
            self._awaiting_rearm = False
            self._rearm_clear_frames = 0
            self._control_reacquired_frame = None
            self._rearm_count += 1
            return True
        return False

    @staticmethod
    def _is_unconfirmed_geometry_start(event: BasketballEvent) -> bool:
        attributes = dict(event.attributes or {})
        return bool(
            str(attributes.get("outcome") or "unknown") == ShotOutcome.UNKNOWN.value
            and not bool(attributes.get("semantic_start", False))
            and not bool(attributes.get("had_above_rim", False))
            and not bool(attributes.get("had_rim_band", False))
        )

    @classmethod
    def _strong_expiry_is_miss(cls, event: BasketballEvent) -> bool:
        attributes = dict(event.attributes or {})
        reason = str(attributes.get("resolution_reason") or "")
        try:
            direct_observations = int(attributes.get("direct_ball_observations", 0) or 0)
        except (TypeError, ValueError):
            direct_observations = 0
        return bool(
            str(attributes.get("outcome") or "unknown") == ShotOutcome.UNKNOWN.value
            and reason == "shot_window_expired_without_outcome_evidence"
            and bool(attributes.get("had_above_rim", False))
            and bool(attributes.get("had_rim_band", False))
            and direct_observations >= cls.MIN_STRONG_EXPIRY_OBSERVATIONS
            and not bool(attributes.get("pending_bridge_used", False))
        )

    def _postprocess_public_event(
        self,
        event: BasketballEvent,
    ) -> BasketballEvent | None:
        event.provenance["outcome_policy_revision"] = self.OUTCOME_POLICY_REVISION
        event.provenance["start_confirmation_policy"] = (
            "supported_flight_rim_band_bridge_plus_direct_post_start_confirmation"
        )
        event.provenance["rearm_policy"] = (
            "direct_observed_clearance_plus_confirmed_control_reacquisition"
        )

        if self._is_unconfirmed_geometry_start(event):
            self._suppressed_public_shots += 1
            self._last_suppressed_reason = (
                "unconfirmed_start_without_direct_above_or_rim_band_evidence"
            )
            return None

        if self._strong_expiry_is_miss(event):
            reason = "strong_rim_evidence_expiry_missed"
            event.attributes["outcome"] = ShotOutcome.MISSED.value
            event.attributes["resolution_reason"] = reason
            event.attributes["expiry_miss_confirmation"] = True
            event.evidence = tuple(sorted(set(event.evidence) | {reason}))
            event.decision = EventDecision.CONFIRMED
            event.confidence = max(float(event.confidence), 0.72)
            event.uncertainty = 1.0 - event.confidence
        return event

    def _maybe_start_supported_flight_bridge(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        court_state: CourtState | None,
    ) -> bool:
        if (
            not self._supported_flight_hint
            or self._active is not None
            or self._awaiting_rearm
            or not self._can_start(frame_id)
            or not self._direct_observed(ball_state)
        ):
            return False

        ball_position = None if ball_state is None else ball_state.position()
        if ball_position is None:
            return False
        rim = self._nearest_rim(ball_position, self._rim_candidates(court_state))
        if rim is None:
            return False
        relative = self._relative_to_rim(ball_position, rim)
        if not (relative["near_x"] and relative["rim_band"]):
            return False

        self._start(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            rim=rim,
            ball_state=ball_state,
            semantic_start=False,
        )
        if self._active is not None:
            self._active.evidence.add("supported_airborne_rim_band_bridge")
        self._supported_flight_bridge_starts += 1
        diagnostics = getattr(self, "_last_start_gate_diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics["supported_flight_hint"] = True
            diagnostics["supported_flight_rim_band_bridge"] = True
            diagnostics["active_after"] = True
            diagnostics["started_this_frame"] = True
        return True

    def update(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        court_state: CourtState | None,
        semantic_evidence: list[dict[str, Any]] | None = None,
    ) -> list[BasketballEvent]:
        try:
            events = super().update(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                ball_state=ball_state,
                court_state=court_state,
                semantic_evidence=semantic_evidence,
            )
            if not events:
                self._maybe_start_supported_flight_bridge(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    ball_state=ball_state,
                    court_state=court_state,
                )
                return []

            output: list[BasketballEvent] = []
            for event in events:
                normalized = self._postprocess_public_event(event)
                if normalized is not None:
                    output.append(normalized)
            return output
        finally:
            # The control-layer hint is causal and valid only for this frame.
            self._supported_flight_hint = False

    def reset(self, *, reason: str = "external_reset") -> None:
        self._supported_flight_hint = False
        self._control_reacquired_frame = None
        super().reset(reason=reason)

    def diagnostics(self) -> dict[str, Any]:
        result = dict(super().diagnostics())
        result.update(
            {
                "outcome_policy_revision": self.OUTCOME_POLICY_REVISION,
                "supported_flight_rim_band_bridge": True,
                "supported_flight_bridge_starts": self._supported_flight_bridge_starts,
                "causal_rearm_requires_control_reacquisition": True,
                "control_reacquired_frame": self._control_reacquired_frame,
                "suppressed_public_shots": self._suppressed_public_shots,
                "last_suppressed_reason": self._last_suppressed_reason,
                "strong_expiry_miss_min_direct_observations": self.MIN_STRONG_EXPIRY_OBSERVATIONS,
            }
        )
        return result


__all__ = ["ShotEventTrackerV13"]
