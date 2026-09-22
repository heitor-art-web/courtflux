from __future__ import annotations

from typing import Any

from core.ball_state import BallState
from core.basketball_event import BasketballEvent, EventDecision, ShotOutcome
from core.court_state import CourtState
from services.p2.shot_event_tracker_v1_1 import ShotEventTrackerV1


class ShotEventTrackerV12(ShotEventTrackerV1):
    """Gate-B outcome-policy revision V1.2.

    V1.1 remains responsible for direct crossings, pending bridges, and deferred
    rim-exit collection. V1.2 adds two conservative confirmations evaluated only when
    a deferred rim-exit window would otherwise expire UNKNOWN:

    * repeated mouth support: at least two direct observations inside the rim mouth;
    * sustained lateral alignment: at least five rim observations and every one is
      horizontally near the rim.

    Both require coherent rim geometry throughout the observed deferred window. These
    rules use no clip identity and no fitted fall-ratio threshold.

    ``start_gate_diagnostics`` is observational only. It records the inputs to the
    existing shot-start gate so development probes can diagnose false negatives without
    changing shot decisions.
    """

    OUTCOME_POLICY_REVISION = "shot_outcome_v1_2"
    MIN_PERSISTENT_NEAR_X_OBSERVATIONS = 5

    @staticmethod
    def _deferred_confirmation_reason(
        diagnostics: dict[str, Any] | None,
    ) -> str | None:
        if not diagnostics:
            return None

        rim_observations = int(diagnostics.get("rim_observations", 0) or 0)
        near_x_observations = int(diagnostics.get("near_x_observations", 0) or 0)
        inside_mouth_observations = int(
            diagnostics.get("inside_mouth_observations", 0) or 0
        )
        incoherent_rim_steps = int(diagnostics.get("incoherent_rim_steps", 0) or 0)

        if incoherent_rim_steps > 0:
            return None

        if inside_mouth_observations >= 2:
            return "deferred_rim_exit_repeated_inside_mouth_confirmed"

        if (
            rim_observations >= ShotEventTrackerV12.MIN_PERSISTENT_NEAR_X_OBSERVATIONS
            and near_x_observations == rim_observations
        ):
            return "deferred_rim_exit_persistent_near_x_confirmed"

        return None

    def _promote_deferred_event_if_supported(
        self,
        event: BasketballEvent,
        *,
        frame_id: int,
    ) -> BasketballEvent:
        attributes = dict(event.attributes or {})
        if str(attributes.get("outcome") or "unknown") != ShotOutcome.UNKNOWN.value:
            return event
        if (
            str(attributes.get("resolution_reason") or "")
            != "deferred_rim_exit_expired_without_confirmation"
        ):
            return event

        diagnostics = attributes.get("deferred_outcome_diagnostics")
        reason = self._deferred_confirmation_reason(
            diagnostics if isinstance(diagnostics, dict) else None
        )
        event.provenance["outcome_policy_revision"] = self.OUTCOME_POLICY_REVISION
        event.provenance["deferred_confirmation_policy"] = (
            "repeated_inside_mouth_or_sustained_near_x_with_coherent_rim"
        )
        if reason is None:
            return event

        event.attributes["outcome"] = ShotOutcome.MADE.value
        event.attributes["resolution_reason"] = reason
        event.attributes["deferred_spatial_confirmation"] = reason
        event.evidence = tuple(sorted(set(event.evidence) | {reason}))
        event.decision = EventDecision.CONFIRMED
        event.confidence = max(float(event.confidence), 0.78)
        event.uncertainty = 1.0 - event.confidence

        # V1.1 already closed the candidate and entered rearm. A promoted MADE also
        # receives the same post-made cooldown used by normal MADE resolution.
        self._cooldown_until_frame = max(
            self._cooldown_until_frame,
            int(frame_id) + self.cooldown_after_made_frames,
        )
        return event

    def _capture_start_gate_diagnostics(
        self,
        *,
        frame_id: int,
        ball_state: BallState | None,
        court_state: CourtState | None,
        semantic_evidence: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        rows = list(semantic_evidence or [])
        flags = self._semantic_flags(rows)
        ball_position = None if ball_state is None else ball_state.position()
        rims = self._rim_candidates(court_state)
        rim = self._nearest_rim(ball_position, rims)
        relative = self._relative_to_rim(ball_position, rim)
        semantic_start = bool(flags["jump_shot"] or flags["layup_dunk"])
        trajectory_start = bool(
            ball_position is not None
            and rim is not None
            and relative["above"]
            and relative["near_x"]
        )
        active_before = self._active is not None
        awaiting_rearm_before = bool(self._awaiting_rearm)
        can_start = bool(self._can_start(frame_id))

        x_offset_rim_widths = None
        y_offset_rim_heights = None
        delta_y_offset_rim_heights = None
        coherent_with_previous = False
        geometry_consecutive = False

        current_geometry = None
        if self._direct_observed(ball_state) and ball_position is not None and rim is not None:
            bx, by = ball_position
            x1, y1, x2, y2 = rim
            width = max(1.0, float(x2) - float(x1))
            height = max(1.0, float(y2) - float(y1))
            rim_x = (float(x1) + float(x2)) / 2.0
            rim_y = (float(y1) + float(y2)) / 2.0
            x_offset_rim_widths = abs(float(bx) - rim_x) / width
            y_offset_rim_heights = (float(by) - rim_y) / height
            current_geometry = {
                "frame_id": int(frame_id),
                "rim": rim,
                "y_offset_rim_heights": y_offset_rim_heights,
            }

            previous = getattr(self, "_last_start_gate_geometry", None)
            if isinstance(previous, dict):
                previous_frame = int(previous.get("frame_id", -10**9))
                previous_rim = previous.get("rim")
                previous_y = previous.get("y_offset_rim_heights")
                geometry_consecutive = previous_frame + 1 == int(frame_id)
                coherent_with_previous = bool(
                    geometry_consecutive
                    and isinstance(previous_rim, tuple)
                    and len(previous_rim) == 4
                    and self._rim_boxes_overlap(previous_rim, rim)
                )
                if coherent_with_previous and previous_y is not None:
                    delta_y_offset_rim_heights = (
                        float(y_offset_rim_heights) - float(previous_y)
                    )

        # Diagnostic history is deliberately strict: any missing direct/rim geometry
        # breaks the consecutive-motion chain rather than bridging evidence gaps.
        self._last_start_gate_geometry = current_geometry

        return {
            "policy": "diagnostic_only_existing_start_gate_inputs",
            "frame_id": int(frame_id),
            "active_before": active_before,
            "awaiting_rearm_before": awaiting_rearm_before,
            "can_start": can_start,
            "direct_ball_observed": bool(self._direct_observed(ball_state)),
            "ball_available": bool(ball_state is not None and ball_state.available),
            "rim_available": rim is not None,
            "rim_candidate_count": len(rims),
            "relative_above": bool(relative["above"]),
            "relative_near_x": bool(relative["near_x"]),
            "relative_rim_band": bool(relative["rim_band"]),
            "relative_below": bool(relative["below"]),
            "relative_inside_mouth": bool(relative["inside_mouth"]),
            "x_offset_rim_widths": x_offset_rim_widths,
            "y_offset_rim_heights": y_offset_rim_heights,
            "delta_y_offset_rim_heights": delta_y_offset_rim_heights,
            "rim_geometry_coherent_with_previous": coherent_with_previous,
            "geometry_consecutive": geometry_consecutive,
            "semantic_jump_shot": bool(flags["jump_shot"]),
            "semantic_layup_dunk": bool(flags["layup_dunk"]),
            "semantic_start": semantic_start,
            "trajectory_start": trajectory_start,
            "start_signal": bool(semantic_start or trajectory_start),
            "cooldown_remaining_frames": max(
                0,
                int(self._cooldown_until_frame) - int(frame_id),
            ),
            "frames_since_last_start": int(frame_id) - int(self._last_start_frame),
        }

    def update(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        court_state: CourtState | None,
        semantic_evidence: list[dict[str, Any]] | None = None,
    ) -> list[BasketballEvent]:
        start_gate = self._capture_start_gate_diagnostics(
            frame_id=frame_id,
            ball_state=ball_state,
            court_state=court_state,
            semantic_evidence=semantic_evidence,
        )
        events = super().update(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            ball_state=ball_state,
            court_state=court_state,
            semantic_evidence=semantic_evidence,
        )
        start_gate["active_after"] = self._active is not None
        start_gate["started_this_frame"] = bool(
            not start_gate["active_before"] and self._active is not None
        )
        self._last_start_gate_diagnostics = start_gate
        return [
            self._promote_deferred_event_if_supported(event, frame_id=frame_id)
            for event in events
        ]

    def reset(self, *, reason: str = "external_reset") -> None:
        self._last_start_gate_geometry = None
        super().reset(reason=reason)

    def diagnostics(self) -> dict[str, Any]:
        result = dict(super().diagnostics())
        result["outcome_policy_revision"] = self.OUTCOME_POLICY_REVISION
        result["deferred_confirmation_policy"] = (
            "repeated_inside_mouth_or_sustained_near_x_with_coherent_rim"
        )
        result["min_persistent_near_x_observations"] = (
            self.MIN_PERSISTENT_NEAR_X_OBSERVATIONS
        )
        result["start_gate_diagnostics"] = dict(
            getattr(self, "_last_start_gate_diagnostics", {}) or {}
        )
        return result
