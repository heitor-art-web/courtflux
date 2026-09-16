from __future__ import annotations

from typing import Any

from core.basketball_event import (
    BallControlState,
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
    mutating P1 truth. Temporal P2 state is reset at meaningful segment boundaries so
    possession/shot candidates never bridge replay/camera/non-gameplay gaps.
    """

    def __init__(
        self,
        *,
        fps: float = 30.0,
        rebound_window_sec: float = 2.5,
        reset_after_ineligible_frames: int = 3,
        max_frame_discontinuity: int = 2,
        recent_shot_actor_sec: float = 0.9,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.possession_tracker = PossessionTrackerV1()
        self.shot_tracker = ShotEventTrackerV1(fps=self.fps)
        self.rebound_window_frames = max(1, int(round(self.fps * float(rebound_window_sec))))
        self.reset_after_ineligible_frames = max(1, int(reset_after_ineligible_frames))
        self.max_frame_discontinuity = max(1, int(max_frame_discontinuity))
        self.recent_shot_actor_frames = max(1, int(round(self.fps * float(recent_shot_actor_sec))))

        self._shot_actor_track_id: int | None = None
        self._shot_team_id: str | None = None
        self._shot_actor_source: str = "unknown"
        self._shot_actor_location: tuple[float, float] | None = None
        self._pending_missed_shot: BasketballEvent | None = None

        self._ineligible_frames = 0
        self._reset_done_for_ineligible_gap = False
        self._last_frame_id: int | None = None
        self._last_timestamp: float | None = None
        self._reset_count = 0
        self._last_reset_reason: str | None = None
        self._last_reset_frame: int | None = None

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

    @staticmethod
    def _analysis_eligible(frame_state: Any) -> bool:
        """Return P2 engine eligibility without conflating it with metric eligibility.

        New CourtState instances expose high-recall ``gameplay_present`` separately from
        conservative ``analysis_eligible``. P2 continuity follows gameplay presence; old
        callers without that field retain the historical analysis-eligible behaviour.
        """
        gameplay = bool(getattr(frame_state, "valid_gameplay", True))
        court_state = getattr(frame_state, "court_state", None)
        if court_state is None:
            return gameplay
        gameplay_present = getattr(court_state, "gameplay_present", None)
        if gameplay_present is not None:
            return bool(gameplay_present)
        return bool(gameplay and getattr(court_state, "analysis_eligible", False))

    def _loose_snapshot(self) -> dict[str, Any] | None:
        possession = self.possession_tracker.diagnostics()
        raw_start = possession.get("loose_start_frame")
        if raw_start is None:
            return None
        try:
            start_frame = int(raw_start)
        except (TypeError, ValueError):
            return None
        return {
            "frame_start": start_frame,
            "origin_track_id": possession.get("current_track_id"),
            "origin_team_id": possession.get("current_team_id"),
        }

    def _loose_start_timestamp(
        self,
        *,
        loose_snapshot: dict[str, Any],
        frame_id: int,
        timestamp: float | None,
    ) -> float | None:
        if timestamp is None:
            return None
        frames = max(0, int(frame_id) - int(loose_snapshot["frame_start"]))
        return max(0.0, float(timestamp) - frames / self.fps)

    def _loose_ended_event(
        self,
        *,
        loose_snapshot: dict[str, Any] | None,
        frame_id: int,
        timestamp: float | None,
        terminal_reason: str,
    ) -> BasketballEvent | None:
        if loose_snapshot is None:
            return None
        start_frame = int(loose_snapshot["frame_start"])
        confidence = 1.0
        return BasketballEvent(
            event_id=f"loose-ball-ended:{start_frame}:{frame_id}:{terminal_reason}",
            event_type=BasketballEventType.LOOSE_BALL_ENDED,
            frame_start=start_frame,
            frame_end=int(frame_id),
            timestamp_start_sec=self._loose_start_timestamp(
                loose_snapshot=loose_snapshot,
                frame_id=frame_id,
                timestamp=timestamp,
            ),
            timestamp_end_sec=timestamp,
            confidence=confidence,
            uncertainty=0.0,
            decision=EventDecision.CONFIRMED,
            actor_track_id=loose_snapshot.get("origin_track_id"),
            team_id=loose_snapshot.get("origin_team_id"),
            evidence=("loose_ball_interval", "terminal_state_transition"),
            provenance={"version": "p2_engine_v1"},
            attributes={
                "terminal_reason": str(terminal_reason),
                "recovered": False,
                "loose_frames": max(1, int(frame_id) - start_frame + 1),
            },
        )

    def _normalize_loose_recovery(
        self,
        *,
        loose_snapshot: dict[str, Any] | None,
        possession_events: list[BasketballEvent],
        frame_id: int,
        timestamp: float | None,
    ) -> None:
        if loose_snapshot is None:
            return

        existing_recovery = next(
            (
                event
                for event in possession_events
                if event.event_type is BasketballEventType.LOOSE_BALL_RECOVERED
            ),
            None,
        )
        if existing_recovery is not None:
            same_handler = bool(
                existing_recovery.actor_track_id is not None
                and existing_recovery.actor_track_id == existing_recovery.secondary_track_id
            )
            existing_recovery.attributes.setdefault(
                "terminal_reason",
                "same_handler_recovered" if same_handler else "new_handler_recovered",
            )
            existing_recovery.attributes.setdefault("same_handler_recovery", same_handler)
            existing_recovery.attributes.setdefault("recovered", True)
            return

        possession_terminal = next(
            (
                event
                for event in possession_events
                if event.event_type
                in {
                    BasketballEventType.POSSESSION_CHANGE,
                    BasketballEventType.POSSESSION_ACQUIRED,
                }
                and event.actor_track_id is not None
            ),
            None,
        )
        if possession_terminal is None:
            return

        start_frame = int(loose_snapshot["frame_start"])
        origin_track_id = loose_snapshot.get("origin_track_id")
        if possession_terminal.event_type is BasketballEventType.POSSESSION_CHANGE:
            origin_track_id = possession_terminal.secondary_track_id
            terminal_reason = "new_handler_switch"
        else:
            terminal_reason = "possession_reacquired"

        recovery = BasketballEvent(
            event_id=(
                f"loose-ball-recovered:{start_frame}:{frame_id}:"
                f"{possession_terminal.actor_track_id}"
            ),
            event_type=BasketballEventType.LOOSE_BALL_RECOVERED,
            frame_start=start_frame,
            frame_end=int(frame_id),
            timestamp_start_sec=self._loose_start_timestamp(
                loose_snapshot=loose_snapshot,
                frame_id=frame_id,
                timestamp=timestamp,
            ),
            timestamp_end_sec=timestamp,
            confidence=possession_terminal.confidence,
            uncertainty=possession_terminal.uncertainty,
            decision=EventDecision.CONFIRMED,
            actor_track_id=possession_terminal.actor_track_id,
            secondary_track_id=origin_track_id,
            team_id=possession_terminal.team_id,
            location_court_m=possession_terminal.location_court_m,
            evidence=("loose_ball_interval", "subsequent_possession_lock"),
            provenance={"version": "p2_engine_v1"},
            attributes={
                "terminal_reason": terminal_reason,
                "same_handler_recovery": False,
                "recovered": True,
                "loose_frames": max(1, int(frame_id) - start_frame + 1),
            },
        )
        insert_at = possession_events.index(possession_terminal) + 1
        possession_events.insert(insert_at, recovery)

    def _reset_temporal_state(
        self,
        *,
        reason: str,
        frame_id: int,
        timestamp: float | None,
    ) -> BasketballEvent | None:
        loose_end = self._loose_ended_event(
            loose_snapshot=self._loose_snapshot(),
            frame_id=frame_id,
            timestamp=timestamp,
            terminal_reason=f"reset:{reason}",
        )
        self.possession_tracker.reset(reason=reason)
        self.shot_tracker.reset(reason=reason)
        self._shot_actor_track_id = None
        self._shot_team_id = None
        self._shot_actor_source = "unknown"
        self._shot_actor_location = None
        self._pending_missed_shot = None
        self._reset_count += 1
        self._last_reset_reason = str(reason)
        self._last_reset_frame = int(frame_id)
        return loose_end

    def _detect_discontinuity(
        self,
        *,
        frame_id: int,
        timestamp: float | None,
    ) -> str | None:
        if self._last_frame_id is not None:
            frame_gap = int(frame_id) - int(self._last_frame_id)
            if frame_gap <= 0 or frame_gap > self.max_frame_discontinuity:
                return "frame_discontinuity"
        if self._last_timestamp is not None and timestamp is not None:
            delta = float(timestamp) - float(self._last_timestamp)
            if delta < -1e-6 or delta > max(1.0, 4.0 / self.fps):
                return "timestamp_discontinuity"
        return None

    def _attribute_shot(self, event: BasketballEvent, frame_state: Any) -> BasketballEvent:
        if event.event_type is not BasketballEventType.SHOT_ATTEMPT:
            return event
        event.actor_track_id = self._shot_actor_track_id
        event.team_id = self._shot_team_id
        if event.location_court_m is None:
            event.location_court_m = self._shot_actor_location or self._player_location(
                frame_state,
                self._shot_actor_track_id,
            )
        if self._shot_actor_track_id is not None:
            event.evidence = tuple(
                sorted(set(event.evidence) | {"pre_release_possession_actor"})
            )
        event.attributes["actor_attribution_source"] = self._shot_actor_source
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
                in {
                    BasketballEventType.POSSESSION_ACQUIRED,
                    BasketballEventType.POSSESSION_CHANGE,
                    BasketballEventType.LOOSE_BALL_RECOVERED,
                }
                and event.actor_track_id is not None
            ),
            None,
        )
        if possession is None:
            return None

        confidence = max(0.0, min(0.82 * possession.confidence, 1.0))
        rebound = BasketballEvent(
            event_id=(
                f"rebound-candidate:{shot.frame_start}:{frame_id}:"
                f"{possession.actor_track_id}"
            ),
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
            provenance={
                "version": "p2_engine_v1",
                "source_shot_event_id": shot.event_id,
            },
            attributes={"source_shot_event_id": shot.event_id},
        )
        self._pending_missed_shot = None
        return rebound

    def _write_diagnostics(
        self,
        *,
        frame_state: Any,
        emitted: list[BasketballEvent],
        semantic_evidence: list[dict[str, Any]],
        team_evidence: dict[int, tuple[str | None, float]],
        engine_eligible: bool,
        skipped_reason: str | None = None,
    ) -> None:
        diagnostics = getattr(frame_state, "diagnostics", None)
        if diagnostics is None:
            frame_state.diagnostics = {}
            diagnostics = frame_state.diagnostics
        diagnostics["p2"] = {
            "version": "p2_engine_v1",
            "behavior_revision": "control_state_v4_gameplay_presence",
            "eligibility_policy": "court_state_gameplay_present_fallback_analysis_eligible",
            "engine_eligible": bool(engine_eligible),
            "skipped_reason": skipped_reason,
            "event_count": len(emitted),
            "events": [event.as_dict() for event in emitted],
            "possession": self.possession_tracker.diagnostics(),
            "shot": self.shot_tracker.diagnostics(),
            "pending_missed_shot_event_id": (
                None
                if self._pending_missed_shot is None
                else self._pending_missed_shot.event_id
            ),
            "semantic_evidence_rows": len(semantic_evidence),
            "team_evidence_tracks": len(team_evidence),
            "temporal_reset": {
                "count": self._reset_count,
                "last_reason": self._last_reset_reason,
                "last_frame": self._last_reset_frame,
                "ineligible_gap_frames": self._ineligible_frames,
            },
            "p1_state_is_not_mutated": True,
        }

    @staticmethod
    def _append_frame_events(frame_state: Any, emitted: list[BasketballEvent]) -> None:
        if not emitted:
            return
        if not hasattr(frame_state, "p2_events") or frame_state.p2_events is None:
            frame_state.p2_events = []
        frame_state.p2_events.extend(emitted)

    def process(self, frame_state: Any) -> Any:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        timestamp = self._timestamp(frame_state)
        players = list(getattr(frame_state, "player_states", []) or [])
        ball_state = getattr(frame_state, "ball_state", None)
        court_state = getattr(frame_state, "court_state", None)
        team_evidence = self._team_evidence(frame_state)
        semantic_evidence = self._semantic_evidence(frame_state)
        boundary_events: list[BasketballEvent] = []

        discontinuity = self._detect_discontinuity(frame_id=frame_id, timestamp=timestamp)
        if discontinuity is not None:
            loose_end = self._reset_temporal_state(
                reason=discontinuity,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            if loose_end is not None:
                boundary_events.append(loose_end)
            self._ineligible_frames = 0
            self._reset_done_for_ineligible_gap = False

        self._last_frame_id = frame_id
        self._last_timestamp = timestamp

        if not self._analysis_eligible(frame_state):
            self._ineligible_frames += 1
            if (
                self._ineligible_frames >= self.reset_after_ineligible_frames
                and not self._reset_done_for_ineligible_gap
            ):
                loose_end = self._reset_temporal_state(
                    reason="gameplay_absent_gap",
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
                if loose_end is not None:
                    boundary_events.append(loose_end)
                self._reset_done_for_ineligible_gap = True
            self._append_frame_events(frame_state, boundary_events)
            self._write_diagnostics(
                frame_state=frame_state,
                emitted=boundary_events,
                semantic_evidence=semantic_evidence,
                team_evidence=team_evidence,
                engine_eligible=False,
                skipped_reason="gameplay_not_present",
            )
            return frame_state

        self._ineligible_frames = 0
        self._reset_done_for_ineligible_gap = False
        loose_before_control = self._loose_snapshot()

        # Shot state is evaluated before possession for the current frame. If a shot is
        # active, airborne ball must not be reinterpreted as possession/pass/turnover.
        pre_track_id = self.possession_tracker.current_track_id
        pre_team_id = self.possession_tracker.current_team_id
        pre_source = "pre_release_possession" if pre_track_id is not None else "unknown"
        if pre_track_id is None:
            recent_track, recent_team, _ = self.possession_tracker.recent_owner(
                frame_id=frame_id,
                max_age_frames=self.recent_shot_actor_frames,
            )
            if recent_track is not None:
                pre_track_id = recent_track
                pre_team_id = recent_team
                pre_source = "recent_possession_before_release"

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
            self._shot_actor_track_id = pre_track_id
            self._shot_team_id = pre_team_id
            self._shot_actor_source = pre_source
            self._shot_actor_location = self._player_location(frame_state, pre_track_id)
            loose_end = self._loose_ended_event(
                loose_snapshot=loose_before_control,
                frame_id=frame_id,
                timestamp=timestamp,
                terminal_reason="shot_flight_started",
            )
            if loose_end is not None:
                boundary_events.append(loose_end)
            self.possession_tracker.release_for_airborne_ball(
                reason="shot",
                frame_id=frame_id,
            )

        possession_events: list[BasketballEvent] = []
        if not shot_was_active and not shot_is_active and not shot_events:
            possession_events = self.possession_tracker.update(
                frame_id=frame_id,
                players=players,
                ball_state=ball_state,
                team_evidence=team_evidence,
                semantic_evidence=semantic_evidence,
            )
            self._normalize_loose_recovery(
                loose_snapshot=loose_before_control,
                possession_events=possession_events,
                frame_id=frame_id,
                timestamp=timestamp,
            )
        elif shot_was_active or shot_is_active:
            self.possession_tracker.control_state = BallControlState.SHOT_FLIGHT

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
            self._shot_actor_source = "unknown"
            self._shot_actor_location = None

        rebound = self._rebound_from_possession(
            frame_id=frame_id,
            possession_events=possession_events,
            frame_state=frame_state,
        )

        emitted = boundary_events + possession_events + attributed_shots
        if rebound is not None:
            emitted.append(rebound)

        self._append_frame_events(frame_state, emitted)
        self._write_diagnostics(
            frame_state=frame_state,
            emitted=emitted,
            semantic_evidence=semantic_evidence,
            team_evidence=team_evidence,
            engine_eligible=True,
        )
        return frame_state
