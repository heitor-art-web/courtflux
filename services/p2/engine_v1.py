from __future__ import annotations

from typing import Any

from core.basketball_event import (
    BasketballEvent,
    BasketballEventType,
    EventDecision,
    ShotOutcome,
)
from services.p2.possession_tracker_v1 import PossessionTrackerV1
from services.p2.shot_event_tracker_v1 import ShotEventTrackerV1


class P2EngineV1:
    """CourtFlux P2 basketball-understanding coordinator.

    P1 owns state/evidence. This engine consumes canonical PlayerState/BallState/
    CourtState plus optional detector/team evidence and emits event primitives without
    mutating P1 truth. P3 statistics should aggregate these events later.
    """

    def __init__(self, *, fps: float = 30.0, rebound_window_sec: float = 2.5) -> None:
        self.fps = max(1.0, float(fps))
        self.possession_tracker = PossessionTrackerV1()
        self.shot_tracker = ShotEventTrackerV1(fps=self.fps)
        self.rebound_window_frames = max(1, int(round(self.fps * float(rebound_window_sec))))
        self._shot_actor_track_id: int | None = None
        self._shot_team_id: str | None = None
        self._pending_missed_shot: BasketballEvent | None = None

    @staticmethod
    def _team_evidence(frame_state: Any) -> dict[int, tuple[str | None, float]]:
        evidence: dict[int, tuple[str | None, float]] = {}
        for row in list(getattr(frame_state, "players", []) or []):
            if not isinstance(row, dict):
                continue
            raw_track_id = row.get("track_id", row.get("id"))
            team_id = row.get("team", row.get("team_id"))
            if raw_track_id is None or team_id is None:
                continue
            try:
                track_id = int(raw_track_id)
                confidence = float(row.get("team_confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            confidence = max(0.0, min(confidence, 1.0))
            previous = evidence.get(track_id)
            if previous is None or confidence > previous[1]:
                evidence[track_id] = (str(team_id), confidence)
        return evidence

    @staticmethod
    def _semantic_evidence(frame_state: Any) -> list[dict[str, Any]]:
        diagnostics = dict(getattr(frame_state, "diagnostics", {}) or {})
        detector = dict(diagnostics.get("detector", {}) or {})
        rows: list[dict[str, Any]] = []
        for key in ("p1_detections", "semantic_evidence", "semantic_detections"):
            value = detector.get(key)
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
        return rows

    @staticmethod
    def _timestamp(frame_state: Any) -> float | None:
        raw = getattr(frame_state, "timestamp", None)
        if raw is None:
            ball_state = getattr(frame_state, "ball_state", None)
            raw = getattr(ball_state, "timestamp_sec", None)
        return None if raw is None else float(raw)

    @staticmethod
    def _player_location(frame_state: Any, track_id: int | None) -> tuple[float, float] | None:
        if track_id is None:
            return None
        for player in list(getattr(frame_state, "player_states", []) or []):
            if getattr(player, "track_id", None) == track_id:
                return getattr(player, "court_position_m", None)
        return None

    def _attribute_shot(self, event: BasketballEvent, frame_state: Any) -> BasketballEvent:
        if event.event_type is not BasketballEventType.SHOT_ATTEMPT:
            return event
        event.actor_track_id = self._shot_actor_track_id
        event.team_id = self._shot_team_id
        if event.location_court_m is None:
            event.location_court_m = self._player_location(frame_state, self._shot_actor_track_id)
        if self._shot_actor_track_id is not None:
            event.evidence = tuple(sorted(set(event.evidence) | {"pre_release_possession_actor"}))
        event.attributes["actor_attribution_source"] = (
            "pre_release_possession" if self._shot_actor_track_id is not None else "unknown"
        )
        return event

    def _rebound_from_possession(
        self,
        *,
        frame_id: int,
        possession_events: list[BasketballEvent],
        frame_state: Any,
    ) -> BasketballEvent | None:
        shot = self._pending_missed_shot
        if shot is None:
            return None
        if int(frame_id) - shot.frame_end > self.rebound_window_frames:
            self._pending_missed_shot = None
            return None

        possession = next(
            (
                event
                for event in possession_events
                if event.event_type
                in {BasketballEventType.POSSESSION_ACQUIRED, BasketballEventType.POSSESSION_CHANGE}
                and event.actor_track_id is not None
            ),
            None,
        )
        if possession is None:
            return None

        confidence = max(0.0, min(0.82 * possession.confidence, 1.0))
        rebound = BasketballEvent(
            event_id=f"rebound-candidate:{shot.frame_start}:{frame_id}:{possession.actor_track_id}",
            event_type=BasketballEventType.REBOUND_CANDIDATE,
            frame_start=shot.frame_end,
            frame_end=int(frame_id),
            timestamp_start_sec=shot.timestamp_end_sec,
            timestamp_end_sec=self._timestamp(frame_state),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            decision=EventDecision.CANDIDATE,
            actor_track_id=possession.actor_track_id,
            team_id=possession.team_id,
            location_court_m=self._player_location(frame_state, possession.actor_track_id),
            evidence=("resolved_missed_shot", "subsequent_possession_lock"),
            provenance={"version": "p2_engine_v1", "source_shot_event_id": shot.event_id},
            attributes={"source_shot_event_id": shot.event_id},
        )
        self._pending_missed_shot = None
        return rebound

    def process(self, frame_state: Any) -> Any:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        timestamp = self._timestamp(frame_state)
        players = list(getattr(frame_state, "player_states", []) or [])
        ball_state = getattr(frame_state, "ball_state", None)
        court_state = getattr(frame_state, "court_state", None)
        team_evidence = self._team_evidence(frame_state)
        semantic_evidence = self._semantic_evidence(frame_state)

        possession_events = self.possession_tracker.update(
            frame_id=frame_id,
            players=players,
            ball_state=ball_state,
            team_evidence=team_evidence,
            semantic_evidence=semantic_evidence,
        )

        shot_was_active = bool(self.shot_tracker.diagnostics().get("active", False))
        shot_events = self.shot_tracker.update(
            frame_id=frame_id,
            timestamp_sec=timestamp,
            ball_state=ball_state,
            court_state=court_state,
            semantic_evidence=semantic_evidence,
        )
        shot_is_active = bool(self.shot_tracker.diagnostics().get("active", False))

        if not shot_was_active and shot_is_active:
            self._shot_actor_track_id = self.possession_tracker.current_track_id
            self._shot_team_id = self.possession_tracker.current_team_id
            self.possession_tracker.release_for_airborne_ball()

        attributed_shots: list[BasketballEvent] = []
        for event in shot_events:
            event = self._attribute_shot(event, frame_state)
            attributed_shots.append(event)
            outcome = str(event.attributes.get("outcome") or ShotOutcome.UNKNOWN.value)
            if outcome == ShotOutcome.MISSED.value:
                self._pending_missed_shot = event
            elif outcome == ShotOutcome.MADE.value:
                self._pending_missed_shot = None
            self._shot_actor_track_id = None
            self._shot_team_id = None

        rebound = self._rebound_from_possession(
            frame_id=frame_id,
            possession_events=possession_events,
            frame_state=frame_state,
        )

        emitted = possession_events + attributed_shots
        if rebound is not None:
            emitted.append(rebound)

        if not hasattr(frame_state, "p2_events") or frame_state.p2_events is None:
            frame_state.p2_events = []
        frame_state.p2_events.extend(emitted)

        diagnostics = getattr(frame_state, "diagnostics", None)
        if diagnostics is None:
            frame_state.diagnostics = {}
            diagnostics = frame_state.diagnostics
        diagnostics["p2"] = {
            "version": "p2_engine_v1",
            "event_count": len(emitted),
            "events": [event.as_dict() for event in emitted],
            "possession": self.possession_tracker.diagnostics(),
            "shot": self.shot_tracker.diagnostics(),
            "pending_missed_shot_event_id": (
                None if self._pending_missed_shot is None else self._pending_missed_shot.event_id
            ),
            "semantic_evidence_rows": len(semantic_evidence),
            "team_evidence_tracks": len(team_evidence),
            "p1_state_is_not_mutated": True,
        }
        return frame_state
