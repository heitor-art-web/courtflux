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

    def update(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        court_state: CourtState | None,
        semantic_evidence: list[dict[str, Any]] | None = None,
    ) -> list[BasketballEvent]:
        events = super().update(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            ball_state=ball_state,
            court_state=court_state,
            semantic_evidence=semantic_evidence,
        )
        return [
            self._promote_deferred_event_if_supported(event, frame_id=frame_id)
            for event in events
        ]

    def diagnostics(self) -> dict[str, Any]:
        result = dict(super().diagnostics())
        result["outcome_policy_revision"] = self.OUTCOME_POLICY_REVISION
        result["deferred_confirmation_policy"] = (
            "repeated_inside_mouth_or_sustained_near_x_with_coherent_rim"
        )
        result["min_persistent_near_x_observations"] = (
            self.MIN_PERSISTENT_NEAR_X_OBSERVATIONS
        )
        return result
