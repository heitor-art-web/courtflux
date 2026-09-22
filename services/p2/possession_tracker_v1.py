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

    When a confirmed owner expires before flight evidence was able to start, the last
    confirmed control anchor seeds an ownerless flight hypothesis on that exact expiry
    frame. The seed does not change control state and does not count as supported flight;
    normal direct-observation, release-duration and displacement requirements still apply.
    """

    FLIGHT_CLASSIFICATION_POLICY = (
        "flight_supported_is_generic_airborne;pass_requires_distinct_receiver_candidate"
    )
    RECENT_OWNER_FLIGHT_SEED_POLICY = (
        "seed_on_confirmed_owner_expiry_only;requires_last_control_ball_anchor;"
        "seed_is_hypothesis_only;existing_flight_support_thresholds_unchanged"
    )
    TEAM_ATTRIBUTION_POLICY = (
        "publish_team_for_supported_owner_or_stale_owner_still_raw_best;"
        "distinct_pending_challenger_or_unsupported_stale_owner_abstains;"
        "recent_strong_team_evidence_may_fill_public_unknown_across_owner_lock;"
        "internal_possession_memory_unchanged"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._recent_owner_flight_seed_count = 0
        self._recent_owner_flight_last_seed_frame: int | None = None
        self._recent_owner_flight_last_origin_track_id: int | None = None
        self._public_team_hint_cache: dict[int, tuple[str, float, int]] = {}
        self._last_update_frame: int | None = None

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

    def _public_team_hint_max_age_frames(self) -> int:
        """Keep a trusted team label only long enough to survive owner confirmation."""
        return max(int(self.lock_frames), int(self.switch_frames)) + 1

    def _observe_public_team_hints(
        self,
        *,
        frame_id: int,
        team_evidence: dict[int, tuple[str | None, float]] | None,
    ) -> None:
        frame_id = int(frame_id)
        self._last_update_frame = frame_id
        max_age = self._public_team_hint_max_age_frames()
        self._public_team_hint_cache = {
            track_id: hint
            for track_id, hint in self._public_team_hint_cache.items()
            if 0 <= frame_id - int(hint[2]) <= max_age
        }
        for raw_track_id, raw_evidence in dict(team_evidence or {}).items():
            try:
                track_id = int(raw_track_id)
                team_id, confidence = raw_evidence
                confidence = max(0.0, min(float(confidence), 1.0))
            except (TypeError, ValueError):
                continue
            if team_id is None or confidence < float(self.team_evidence_threshold):
                continue
            self._public_team_hint_cache[track_id] = (str(team_id), confidence, frame_id)

    def _recent_public_team_hint(
        self,
        track_id: int | None,
    ) -> tuple[str | None, float, int | None]:
        if track_id is None or self._last_update_frame is None:
            return None, 0.0, None
        hint = self._public_team_hint_cache.get(int(track_id))
        if hint is None:
            return None, 0.0, None
        team_id, confidence, hint_frame = hint
        age = int(self._last_update_frame) - int(hint_frame)
        if age < 0 or age > self._public_team_hint_max_age_frames():
            return None, 0.0, None
        return str(team_id), float(confidence), int(age)

    def _seed_flight_on_owner_expiry(
        self,
        *,
        frame_id: int,
        owner_before_update: int | None,
        ball_state: BallState | None,
    ) -> bool:
        """Preserve release origin when ownership expires before flight can start.

        This is deliberately narrower than a generic "recent owner" lookup. A flight may
        be seeded only on the exact update where a live owner transitions to no owner, and
        only when the last confirmed-control anchor belongs to that same owner. Missing
        current-ball evidence is allowed: the seed remains unsupported until later frames
        satisfy the existing flight evidence requirements.
        """
        if owner_before_update is None or self.current_track_id is not None or self._flight is not None:
            return False
        if self._last_control_track_id is None or self._last_control_frame is None:
            return False
        if int(self._last_control_track_id) != int(owner_before_update):
            return False
        if self._last_control_ball_position is None:
            return False

        self._begin_or_update_flight(
            frame_id=frame_id,
            timestamp_sec=self._timestamp(None, ball_state),
            ball_state=ball_state,
            origin_track_id=int(self._last_control_track_id),
            origin_team_id=self._last_control_team_id,
        )
        self._recent_owner_flight_seed_count += 1
        self._recent_owner_flight_last_seed_frame = int(frame_id)
        self._recent_owner_flight_last_origin_track_id = int(self._last_control_track_id)
        return True

    def update(
        self,
        *,
        frame_id: int,
        players: list[PlayerState],
        ball_state: BallState | None,
        team_evidence: dict[int, tuple[str | None, float]] | None = None,
        semantic_evidence: list[dict[str, Any]] | None = None,
    ) -> list[BasketballEvent]:
        self._observe_public_team_hints(frame_id=frame_id, team_evidence=team_evidence)
        owner_before_update = self.current_track_id
        events = super().update(
            frame_id=frame_id,
            players=players,
            ball_state=ball_state,
            team_evidence=team_evidence,
            semantic_evidence=semantic_evidence,
        )

        # If the confirmed owner expired on this update before a release flight could be
        # created, preserve the last confirmed ball/control anchor as an unsupported
        # ownerless flight hypothesis. Later frames must still satisfy the unchanged
        # flight-support thresholds before this evidence can affect classification/events.
        self._seed_flight_on_owner_expiry(
            frame_id=frame_id,
            owner_before_update=owner_before_update,
            ball_state=ball_state,
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

    def _team_attribution_abstention_reason(self) -> str | None:
        """Return why public team attribution must abstain without mutating owner memory."""
        if self.current_track_id is None:
            return None
        if (
            self._candidate_track_id is not None
            and int(self._candidate_track_id) != int(self.current_track_id)
            and int(self._candidate_frames) > 0
        ):
            return "distinct_challenger_pending"
        raw_best_track_id = getattr(self, "_raw_best_track_id", None)
        raw_best_matches_current_owner = bool(
            raw_best_track_id is not None
            and int(raw_best_track_id) == int(self.current_track_id)
        )
        if int(self._lost_frames) > 0 and not raw_best_matches_current_owner:
            return "owner_not_currently_supported"
        return None

    def reset(self, *, reason: str = "external_reset") -> None:
        super().reset(reason=reason)
        self._public_team_hint_cache.clear()
        self._last_update_frame = None

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
        raw_best_matches_current_owner = bool(
            self.current_track_id is not None
            and raw_best_track_id is not None
            and int(raw_best_track_id) == int(self.current_track_id)
        )
        flight = self._flight
        raw_best_distinct_from_flight_origin = bool(
            flight is not None
            and raw_best_track_id is not None
            and int(raw_best_track_id) != int(flight.origin_track_id)
        )
        current_flight_seeded_from_recent_owner = bool(
            flight is not None
            and self._recent_owner_flight_last_seed_frame is not None
            and flight.frame_start == self._recent_owner_flight_last_seed_frame
            and self._recent_owner_flight_last_origin_track_id is not None
            and int(flight.origin_track_id) == self._recent_owner_flight_last_origin_track_id
        )
        attribution_reason = self._team_attribution_abstention_reason()
        internal_current_team_id = self.current_team_id
        public_team_hint_id = None
        public_team_hint_confidence = 0.0
        public_team_hint_age_frames = None
        if attribution_reason is not None:
            diagnostics["current_team_id"] = None
        elif internal_current_team_id is None and self.current_track_id is not None:
            (
                public_team_hint_id,
                public_team_hint_confidence,
                public_team_hint_age_frames,
            ) = self._recent_public_team_hint(self.current_track_id)
            if public_team_hint_id is not None:
                diagnostics["current_team_id"] = public_team_hint_id
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
                "raw_best_matches_current_owner": raw_best_matches_current_owner,
                "raw_best_distinct_from_flight_origin": raw_best_distinct_from_flight_origin,
                "pass_flight_receiver_evidence": self._pass_receiver_evidence(),
                "flight_classification_policy": self.FLIGHT_CLASSIFICATION_POLICY,
                "recent_owner_flight_seed_policy": self.RECENT_OWNER_FLIGHT_SEED_POLICY,
                "recent_owner_flight_seed_count": int(self._recent_owner_flight_seed_count),
                "recent_owner_flight_last_seed_frame": self._recent_owner_flight_last_seed_frame,
                "recent_owner_flight_last_origin_track_id": self._recent_owner_flight_last_origin_track_id,
                "flight_seeded_from_recent_owner": current_flight_seeded_from_recent_owner,
                "team_attribution_policy": self.TEAM_ATTRIBUTION_POLICY,
                "team_attribution_abstained": attribution_reason is not None,
                "team_attribution_abstention_reason": attribution_reason,
                "internal_current_team_id": internal_current_team_id,
                "public_team_hint_applied": public_team_hint_id is not None,
                "public_team_hint_team_id": public_team_hint_id,
                "public_team_hint_confidence": round(float(public_team_hint_confidence), 6),
                "public_team_hint_age_frames": public_team_hint_age_frames,
                "public_team_hint_max_age_frames": self._public_team_hint_max_age_frames(),
            }
        )
        return diagnostics
