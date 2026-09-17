from __future__ import annotations

from typing import Any

from core.basketball_event import BallControlState, BasketballEvent
from core.ball_state import BallState
from core.player_state import PlayerState
from services.p2.possession_tracker_base_v1 import PossessionTrackerV1 as _BasePossessionTrackerV1


class PossessionTrackerV1(_BasePossessionTrackerV1):
    """Canonical possession tracker with conservative flight classification.

    ``flight_supported`` is generic airborne/release evidence. It must not, by itself,
    classify the control state as PASS_FLIGHT. PASS_FLIGHT requires a distinct receiver
    candidate from the flight origin; otherwise the tracker abstains to UNKNOWN while
    preserving the flight evidence for later shot/pass classification and recovery.
    """

    FLIGHT_CLASSIFICATION_POLICY = (
        "flight_supported_is_generic_airborne;pass_requires_distinct_receiver_candidate"
    )

    def _pass_receiver_evidence(self) -> bool:
        flight = self._flight
        if flight is None or self._candidate_track_id is None:
            return False
        return int(self._candidate_track_id) != int(flight.origin_track_id)

    def update(
        self,
        *,
        frame_id: int,
        players: list[PlayerState],
        ball_state: BallState | None,
        team_evidence: dict[int, tuple[str | None, float]] | None = None,
        semantic_evidence: list[dict[str, Any]] | None = None,
    ) -> list[BasketballEvent]:
        events = super().update(
            frame_id=frame_id,
            players=players,
            ball_state=ball_state,
            team_evidence=team_evidence,
            semantic_evidence=semantic_evidence,
        )

        # The base V1 historically equated supported release flight with PASS_FLIGHT.
        # Real-video GT shows that this also fires on dribble separation and shot flight.
        # Keep the accumulated flight evidence, but abstain unless a distinct receiver
        # candidate is actually present.
        if (
            self.control_state is BallControlState.PASS_FLIGHT
            and not self._pass_receiver_evidence()
        ):
            self.control_state = BallControlState.UNKNOWN
        return events

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        threshold = (
            self.acquire_threshold
            if self.current_track_id is None
            else self.switch_threshold
        )
        required_frames = (
            self.lock_frames
            if self.current_track_id is None
            else self.switch_frames
        )
        candidate_score = round(float(self._candidate_score), 6)
        diagnostics.update(
            {
                "candidate_score": candidate_score,
                "candidate_peak_score": candidate_score,
                "candidate_threshold": round(float(threshold), 6),
                "candidate_required_frames": int(required_frames),
                "pass_flight_receiver_evidence": self._pass_receiver_evidence(),
                "flight_classification_policy": self.FLIGHT_CLASSIFICATION_POLICY,
            }
        )
        return diagnostics
