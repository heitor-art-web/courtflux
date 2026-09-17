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

    def _rank(
        self,
        players: list[PlayerState],
        ball_state: BallState | None,
        *,
        team_evidence: dict[int, tuple[str | None, float]] | None,
        semantic_evidence: list[dict[str, Any]] | None,
    ):
        """Capture pre-threshold rank evidence without changing ranking or decisions."""
        best = super()._rank(
            players,
            ball_state,
            team_evidence=team_evidence,
            semantic_evidence=semantic_evidence,
        )
        self._raw_best_track_id = None if best is None else int(best.track_id)
        self._raw_best_score = 0.0 if best is None else float(best.score)
        return best

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
        raw_best_track_id = getattr(self, "_raw_best_track_id", None)
        raw_best_score = round(float(getattr(self, "_raw_best_score", 0.0)), 6)
        flight = self._flight
        raw_best_distinct_from_flight_origin = bool(
            flight is not None
            and raw_best_track_id is not None
            and int(raw_best_track_id) != int(flight.origin_track_id)
        )
        diagnostics.update(
            {
                "candidate_score": candidate_score,
                "candidate_peak_score": candidate_score,
                "candidate_threshold": round(float(threshold), 6),
                "candidate_required_frames": int(required_frames),
                "raw_best_track_id": raw_best_track_id,
                "raw_best_score": raw_best_score,
                "raw_best_above_threshold": bool(
                    raw_best_track_id is not None and raw_best_score >= float(threshold)
                ),
                "raw_best_distinct_from_flight_origin": raw_best_distinct_from_flight_origin,
                "pass_flight_receiver_evidence": self._pass_receiver_evidence(),
                "flight_classification_policy": self.FLIGHT_CLASSIFICATION_POLICY,
            }
        )
        return diagnostics
