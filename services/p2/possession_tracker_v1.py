from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import (
    BallControlState,
    BasketballEvent,
    BasketballEventType,
    EventDecision,
)
from core.player_state import PlayerState


@dataclass(slots=True)
class _Candidate:
    track_id: int
    score: float
    team_id: str | None
    team_confidence: float


@dataclass(slots=True)
class _FlightEvidence:
    origin_track_id: int
    origin_team_id: str | None
    frame_start: int
    timestamp_start_sec: float | None
    origin_ball_position: tuple[float, float] | None
    origin_player_height: float
    last_frame: int
    last_ball_position: tuple[float, float] | None = None
    direct_observations: int = 0
    predicted_observations: int = 0
    max_displacement_px: float = 0.0

    def observe(self, frame_id: int, ball_state: BallState | None) -> None:
        self.last_frame = int(frame_id)
        if ball_state is None or not ball_state.available:
            return
        position = ball_state.position()
        if position is None:
            return
        self.last_ball_position = position
        if ball_state.status is BallStateStatus.OBSERVED:
            self.direct_observations += 1
        elif ball_state.status is BallStateStatus.PREDICTED:
            self.predicted_observations += 1
        if self.origin_ball_position is not None:
            dx = float(position[0]) - float(self.origin_ball_position[0])
            dy = float(position[1]) - float(self.origin_ball_position[1])
            self.max_displacement_px = max(self.max_displacement_px, math.hypot(dx, dy))

    @property
    def frame_count(self) -> int:
        return max(1, self.last_frame - self.frame_start + 1)

    @property
    def displacement_heights(self) -> float:
        return self.max_displacement_px / max(1.0, self.origin_player_height)


class PossessionTrackerV1:
    """Conservative P2 ball-control inference over canonical P1 state.

    P2 distinguishes POSSESSED, LOOSE, PASS_FLIGHT and SHOT_FLIGHT. Team evidence is
    optional. Same-team transitions become pass candidates only when direct ball-flight
    evidence exists; team changes can still be turnover candidates for direct takeaways.
    """

    def __init__(
        self,
        *,
        lock_frames: int = 3,
        switch_frames: int = 3,
        lost_grace_frames: int = 6,
        loose_confirm_frames: int = 3,
        release_confirm_frames: int = 2,
        acquire_threshold: float = 0.58,
        switch_threshold: float = 0.64,
        team_evidence_threshold: float = 0.65,
        pass_min_direct_observations: int = 2,
        pass_min_displacement_heights: float = 0.65,
    ) -> None:
        self.lock_frames = max(1, int(lock_frames))
        self.switch_frames = max(1, int(switch_frames))
        self.lost_grace_frames = max(0, int(lost_grace_frames))
        self.loose_confirm_frames = max(1, int(loose_confirm_frames))
        self.release_confirm_frames = max(1, int(release_confirm_frames))
        self.acquire_threshold = max(0.0, min(float(acquire_threshold), 1.0))
        self.switch_threshold = max(0.0, min(float(switch_threshold), 1.0))
        self.team_evidence_threshold = max(0.0, min(float(team_evidence_threshold), 1.0))
        self.pass_min_direct_observations = max(1, int(pass_min_direct_observations))
        self.pass_min_displacement_heights = max(0.05, float(pass_min_displacement_heights))

        self.current_track_id: int | None = None
        self.current_team_id: str | None = None
        self.current_confidence = 0.0
        self.control_state = BallControlState.UNKNOWN

        self._candidate_track_id: int | None = None
        self._candidate_frames = 0
        self._candidate_score = 0.0
        self._lost_frames = 0

        self._last_control_track_id: int | None = None
        self._last_control_team_id: str | None = None
        self._last_control_confidence = 0.0
        self._last_control_frame: int | None = None
        self._last_control_ball_position: tuple[float, float] | None = None
        self._last_control_player_height = 1.0

        self._flight: _FlightEvidence | None = None
        self._loose_start_frame: int | None = None
        self._loose_start_timestamp_sec: float | None = None
        self._loose_origin_track_id: int | None = None
        self._loose_origin_team_id: str | None = None

        self._reset_count = 0
        self._last_reset_reason: str | None = None

    @staticmethod
    def _point_to_box_distance(
        point: tuple[float, float],
        box: tuple[int, int, int, int],
    ) -> float:
        x, y = point
        x1, y1, x2, y2 = box
        dx = max(float(x1) - x, 0.0, x - float(x2))
        dy = max(float(y1) - y, 0.0, y - float(y2))
        return math.hypot(dx, dy)

    @staticmethod
    def _semantic_possession_boost(
        player: PlayerState,
        semantic_evidence: list[dict[str, Any]],
    ) -> float:
        px1, py1, px2, py2 = player.bbox
        best = 0.0
        for row in semantic_evidence:
            if not isinstance(row, dict):
                continue
            label = str(
                row.get("class_name") or row.get("label") or row.get("kind") or ""
            ).lower()
            if "player-in-possession" not in label and "player_in_possession" not in label:
                continue
            raw_box = row.get("box", row.get("bbox", row.get("bbox_px")))
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                confidence = max(0.0, min(float(row.get("confidence", 0.0)), 1.0))
            except (TypeError, ValueError):
                continue
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                best = max(best, 0.18 * confidence)
        return best

    def _rank(
        self,
        players: list[PlayerState],
        ball_state: BallState | None,
        *,
        team_evidence: dict[int, tuple[str | None, float]] | None,
        semantic_evidence: list[dict[str, Any]] | None,
    ) -> _Candidate | None:
        if ball_state is None or not ball_state.available:
            return None
        ball_position = ball_state.position()
        if ball_position is None:
            return None

        semantics = list(semantic_evidence or [])
        teams = dict(team_evidence or {})
        best: _Candidate | None = None
        for player in players:
            height = max(1.0, float(player.bbox[3] - player.bbox[1]))
            distance = self._point_to_box_distance(ball_position, player.bbox)
            proximity = math.exp(-distance / max(0.55 * height, 1.0))
            evidence_quality = 0.55 * float(ball_state.confidence) + 0.45 * float(player.confidence)
            trajectory_quality = (
                float(ball_state.trajectory_score)
                if ball_state.trajectory_score is not None
                else float(ball_state.confidence)
            )
            score = proximity * (0.75 * evidence_quality + 0.25 * trajectory_quality)
            if ball_state.status is BallStateStatus.PREDICTED:
                score *= 0.72
            score += self._semantic_possession_boost(player, semantics)
            score = max(0.0, min(score, 1.0))

            team_id, team_confidence = teams.get(player.track_id, (None, 0.0))
            candidate = _Candidate(
                track_id=player.track_id,
                score=score,
                team_id=None if team_id is None else str(team_id),
                team_confidence=max(0.0, min(float(team_confidence), 1.0)),
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _advance_candidate(self, candidate: _Candidate | None) -> None:
        track_id = None if candidate is None else candidate.track_id
        score = 0.0 if candidate is None else candidate.score
        if track_id is not None and track_id == self._candidate_track_id:
            self._candidate_frames += 1
            self._candidate_score = max(self._candidate_score, score)
            return
        self._candidate_track_id = track_id
        self._candidate_frames = 1 if track_id is not None else 0
        self._candidate_score = score

    @staticmethod
    def _timestamp(player: PlayerState | None, ball_state: BallState | None) -> float | None:
        if ball_state is not None and ball_state.timestamp_sec is not None:
            return ball_state.timestamp_sec
        return None if player is None else player.timestamp_sec

    @staticmethod
    def _player_by_id(players: list[PlayerState], track_id: int | None) -> PlayerState | None:
        if track_id is None:
            return None
        return next((player for player in players if player.track_id == track_id), None)

    @staticmethod
    def _player_height(player: PlayerState | None) -> float:
        if player is None:
            return 1.0
        return max(1.0, float(player.bbox[3] - player.bbox[1]))

    def _remember_control(
        self,
        *,
        frame_id: int,
        player: PlayerState | None,
        ball_state: BallState | None,
    ) -> None:
        if self.current_track_id is None:
            return
        self._last_control_track_id = self.current_track_id
        self._last_control_team_id = self.current_team_id
        self._last_control_confidence = self.current_confidence
        self._last_control_frame = int(frame_id)
        self._last_control_player_height = self._player_height(player)
        if ball_state is not None and ball_state.available:
            position = ball_state.position()
            if position is not None:
                self._last_control_ball_position = position

    def recent_owner(
        self,
        *,
        frame_id: int,
        max_age_frames: int,
    ) -> tuple[int | None, str | None, float]:
        if self._last_control_track_id is None or self._last_control_frame is None:
            return None, None, 0.0
        if int(frame_id) - self._last_control_frame > max(0, int(max_age_frames)):
            return None, None, 0.0
        return (
            self._last_control_track_id,
            self._last_control_team_id,
            self._last_control_confidence,
        )

    def _begin_or_update_flight(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        ball_state: BallState | None,
        origin_track_id: int,
        origin_team_id: str | None,
    ) -> None:
        if self._flight is None or self._flight.origin_track_id != origin_track_id:
            self._flight = _FlightEvidence(
                origin_track_id=origin_track_id,
                origin_team_id=origin_team_id,
                frame_start=int(frame_id),
                timestamp_start_sec=timestamp_sec,
                origin_ball_position=self._last_control_ball_position,
                origin_player_height=max(1.0, self._last_control_player_height),
                last_frame=int(frame_id),
            )
        self._flight.observe(frame_id, ball_state)

    def _flight_supported(self) -> bool:
        flight = self._flight
        return bool(
            flight is not None
            and flight.direct_observations >= self.pass_min_direct_observations
            and flight.frame_count >= self.release_confirm_frames
            and flight.displacement_heights >= self.pass_min_displacement_heights
        )

    def _flight_confidence(self) -> float:
        flight = self._flight
        if flight is None:
            return 0.0
        obs_score = min(
            1.0,
            flight.direct_observations / max(1, self.pass_min_direct_observations + 1),
        )
        displacement_score = min(
            1.0,
            flight.displacement_heights / max(self.pass_min_displacement_heights, 1e-6),
        )
        return max(0.0, min(0.45 * obs_score + 0.55 * displacement_score, 1.0))

    def _flight_attributes(self) -> dict[str, Any]:
        flight = self._flight
        if flight is None:
            return {
                "flight_supported": False,
                "flight_frames": 0,
                "flight_direct_observations": 0,
                "flight_predicted_observations": 0,
                "flight_displacement_px": 0.0,
                "flight_displacement_player_heights": 0.0,
            }
        return {
            "flight_supported": self._flight_supported(),
            "flight_frames": flight.frame_count,
            "flight_direct_observations": flight.direct_observations,
            "flight_predicted_observations": flight.predicted_observations,
            "flight_displacement_px": round(float(flight.max_displacement_px), 3),
            "flight_displacement_player_heights": round(float(flight.displacement_heights), 4),
        }

    def _emit_loose_started(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
    ) -> BasketballEvent | None:
        if self._loose_start_frame is not None:
            return None
        self._loose_start_frame = int(frame_id)
        self._loose_start_timestamp_sec = timestamp_sec
        self._loose_origin_track_id = self.current_track_id
        self._loose_origin_team_id = self.current_team_id
        confidence = max(0.35, min(0.78, 0.48 + 0.04 * self._lost_frames))
        return BasketballEvent(
            event_id=f"loose-ball-started:{frame_id}:{self.current_track_id}",
            event_type=BasketballEventType.LOOSE_BALL_STARTED,
            frame_start=max(0, int(frame_id) - self._lost_frames + 1),
            frame_end=int(frame_id),
            timestamp_start_sec=timestamp_sec,
            timestamp_end_sec=timestamp_sec,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            decision=EventDecision.CANDIDATE,
            actor_track_id=self.current_track_id,
            team_id=self.current_team_id,
            evidence=("possession_proximity_lost", "ball_still_available"),
            provenance={"version": "possession_tracker_v1"},
            attributes={"lost_frames": self._lost_frames, **self._flight_attributes()},
        )

    def _emit_loose_recovered(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        new_track_id: int,
        new_team_id: str | None,
        confidence: float,
    ) -> BasketballEvent | None:
        if self._loose_start_frame is None:
            return None
        confidence = max(0.0, min(confidence, 1.0))
        return BasketballEvent(
            event_id=f"loose-ball-recovered:{self._loose_start_frame}:{frame_id}:{new_track_id}",
            event_type=BasketballEventType.LOOSE_BALL_RECOVERED,
            frame_start=self._loose_start_frame,
            frame_end=int(frame_id),
            timestamp_start_sec=self._loose_start_timestamp_sec,
            timestamp_end_sec=timestamp_sec,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            decision=EventDecision.CONFIRMED,
            actor_track_id=new_track_id,
            secondary_track_id=self._loose_origin_track_id,
            team_id=new_team_id,
            evidence=("loose_ball_interval", "subsequent_possession_lock"),
            provenance={"version": "possession_tracker_v1"},
            attributes={
                "origin_team_id": self._loose_origin_team_id,
                "recovery_team_id": new_team_id,
                "loose_frames": int(frame_id) - self._loose_start_frame + 1,
                **self._flight_attributes(),
            },
        )

    def _transition_candidate(
        self,
        *,
        frame_id: int,
        timestamp_sec: float | None,
        previous_track: int,
        previous_team: str | None,
        new_track: int,
        new_team: str | None,
        possession_confidence: float,
        location_court_m: tuple[float, float] | None,
        transition_kind: str,
    ) -> BasketballEvent | None:
        if previous_team is None or new_team is None or previous_track == new_track:
            return None

        flight_supported = self._flight_supported()
        flight_confidence = self._flight_confidence()
        attributes = {
            "from_team_id": previous_team,
            "to_team_id": new_team,
            "classification_requires_team_evidence": True,
            "transition_kind": transition_kind,
            **self._flight_attributes(),
        }

        if previous_team == new_team:
            if not flight_supported:
                return None
            event_type = BasketballEventType.PASS_CANDIDATE
            prefix = "pass-candidate"
            evidence = (
                "team_consistent_transition",
                "ball_flight_between_handlers",
                "temporal_possession_transition",
            )
            confidence = min(
                1.0,
                0.72 * possession_confidence + 0.28 * flight_confidence,
            )
        else:
            event_type = BasketballEventType.TURNOVER_CANDIDATE
            prefix = "turnover-candidate"
            evidence = (
                "team_changed_between_handlers",
                "ball_flight_between_handlers" if flight_supported else "direct_control_change",
                "temporal_possession_transition",
            )
            confidence = min(
                1.0,
                0.82 * possession_confidence
                + (0.18 * flight_confidence if flight_supported else 0.0),
            )

        return BasketballEvent(
            event_id=f"{prefix}:{frame_id}:{previous_track}:{new_track}",
            event_type=event_type,
            frame_start=(self._flight.frame_start if self._flight is not None else int(frame_id)),
            frame_end=int(frame_id),
            timestamp_start_sec=(
                self._flight.timestamp_start_sec if self._flight is not None else timestamp_sec
            ),
            timestamp_end_sec=timestamp_sec,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            decision=EventDecision.CANDIDATE,
            actor_track_id=previous_track,
            secondary_track_id=new_track,
            team_id=previous_team,
            location_court_m=location_court_m,
            evidence=evidence,
            provenance={"version": "possession_tracker_v1"},
            attributes=attributes,
        )

    def _clear_transition_memory(self) -> None:
        self._flight = None
        self._loose_start_frame = None
        self._loose_start_timestamp_sec = None
        self._loose_origin_track_id = None
        self._loose_origin_team_id = None

    def update(
        self,
        *,
        frame_id: int,
        players: list[PlayerState],
        ball_state: BallState | None,
        team_evidence: dict[int, tuple[str | None, float]] | None = None,
        semantic_evidence: list[dict[str, Any]] | None = None,
    ) -> list[BasketballEvent]:
        events: list[BasketballEvent] = []
        best = self._rank(
            players,
            ball_state,
            team_evidence=team_evidence,
            semantic_evidence=semantic_evidence,
        )
        threshold = self.acquire_threshold if self.current_track_id is None else self.switch_threshold
        usable = best if best is not None and best.score >= threshold else None

        # Acquisition/switch thresholds are intentionally stricter than continuity. If
        # the already-locked handler is still the best-ranked player but temporarily
        # dips below the switch threshold, preserve POSSESSED for the existing grace
        # window instead of immediately manufacturing release/flight evidence. This uses
        # the existing lost_grace_frames contract and introduces no new numeric threshold.
        same_handler_below_threshold = bool(
            self.current_track_id is not None
            and best is not None
            and best.track_id == self.current_track_id
            and usable is None
        )
        if same_handler_below_threshold and self._lost_frames < self.lost_grace_frames:
            self._advance_candidate(best)
            self._lost_frames += 1
            self.current_confidence = max(self.current_confidence * 0.92, float(best.score))
            if best.team_confidence >= self.team_evidence_threshold:
                self.current_team_id = best.team_id
            self.control_state = BallControlState.POSSESSED
            return events

        if usable is None:
            self._advance_candidate(None)
            self._lost_frames += 1
            current_player = self._player_by_id(players, self.current_track_id)
            timestamp = self._timestamp(current_player, ball_state)

            if self.current_track_id is not None and ball_state is not None and ball_state.available:
                self._begin_or_update_flight(
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    ball_state=ball_state,
                    origin_track_id=self.current_track_id,
                    origin_team_id=self.current_team_id,
                )
                if self._flight_supported():
                    self.control_state = BallControlState.PASS_FLIGHT
                elif self._lost_frames >= self.loose_confirm_frames:
                    self.control_state = BallControlState.LOOSE
                    loose_event = self._emit_loose_started(
                        frame_id=frame_id,
                        timestamp_sec=timestamp,
                    )
                    if loose_event is not None:
                        events.append(loose_event)
            elif self._lost_frames >= self.loose_confirm_frames:
                self.control_state = BallControlState.UNKNOWN

            if self._lost_frames > self.lost_grace_frames:
                self.current_track_id = None
                self.current_team_id = None
                self.current_confidence = 0.0
                if ball_state is None or not ball_state.available:
                    self.control_state = BallControlState.UNKNOWN
            return events

        self._advance_candidate(usable)
        player = self._player_by_id(players, usable.track_id)
        timestamp = self._timestamp(player, ball_state)

        if self.current_track_id is None:
            self._lost_frames = 0
            if self._candidate_frames < self.lock_frames:
                return events

            previous_track = None if self._flight is None else self._flight.origin_track_id
            previous_team = None if self._flight is None else self._flight.origin_team_id
            new_team = usable.team_id if usable.team_confidence >= self.team_evidence_threshold else None
            confidence = usable.score
            events.append(
                BasketballEvent(
                    event_id=f"possession-acquired:{frame_id}:{usable.track_id}",
                    event_type=BasketballEventType.POSSESSION_ACQUIRED,
                    frame_start=max(0, int(frame_id) - self._candidate_frames + 1),
                    frame_end=int(frame_id),
                    timestamp_start_sec=timestamp,
                    timestamp_end_sec=timestamp,
                    confidence=confidence,
                    uncertainty=1.0 - confidence,
                    decision=EventDecision.CONFIRMED,
                    actor_track_id=usable.track_id,
                    team_id=new_team,
                    location_court_m=None if player is None else player.court_position_m,
                    evidence=("ball_player_proximity", "temporal_possession_lock"),
                    provenance={"version": "possession_tracker_v1"},
                    attributes={
                        "from_control_state": self.control_state.value,
                        **self._flight_attributes(),
                    },
                )
            )

            recovered = self._emit_loose_recovered(
                frame_id=frame_id,
                timestamp_sec=timestamp,
                new_track_id=usable.track_id,
                new_team_id=new_team,
                confidence=confidence,
            )
            if recovered is not None:
                events.append(recovered)

            if previous_track is not None and previous_track != usable.track_id:
                transition = self._transition_candidate(
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    previous_track=previous_track,
                    previous_team=previous_team,
                    new_track=usable.track_id,
                    new_team=new_team,
                    possession_confidence=confidence,
                    location_court_m=None if player is None else player.court_position_m,
                    transition_kind="reacquisition_after_control_gap",
                )
                if transition is not None:
                    events.append(transition)

            self.current_track_id = usable.track_id
            self.current_team_id = new_team
            self.current_confidence = confidence
            self.control_state = BallControlState.POSSESSED
            self._remember_control(frame_id=frame_id, player=player, ball_state=ball_state)
            self._clear_transition_memory()
            return events

        if usable.track_id == self.current_track_id:
            self._lost_frames = 0
            self.current_confidence = max(self.current_confidence * 0.8, usable.score)
            if usable.team_confidence >= self.team_evidence_threshold:
                self.current_team_id = usable.team_id
            recovered = self._emit_loose_recovered(
                frame_id=frame_id,
                timestamp_sec=timestamp,
                new_track_id=usable.track_id,
                new_team_id=self.current_team_id,
                confidence=self.current_confidence,
            )
            if recovered is not None:
                events.append(recovered)
            self.control_state = BallControlState.POSSESSED
            self._clear_transition_memory()
            # Only a frame still supported as the same handler may advance the flight
            # origin. This freezes pass-flight origin at the last genuinely controlled
            # ball position instead of overwriting it with the receiver-side position.
            self._remember_control(frame_id=frame_id, player=player, ball_state=ball_state)
            return events

        previous_track = self.current_track_id
        previous_team = self.current_team_id
        self._begin_or_update_flight(
            frame_id=frame_id,
            timestamp_sec=timestamp,
            ball_state=ball_state,
            origin_track_id=previous_track,
            origin_team_id=previous_team,
        )
        if self._flight_supported():
            self.control_state = BallControlState.PASS_FLIGHT

        if self._candidate_frames < self.switch_frames:
            return events

        new_team = usable.team_id if usable.team_confidence >= self.team_evidence_threshold else None
        confidence = usable.score
        self.current_track_id = usable.track_id
        self.current_team_id = new_team
        self.current_confidence = confidence
        self._lost_frames = 0

        events.append(
            BasketballEvent(
                event_id=f"possession-change:{frame_id}:{previous_track}:{usable.track_id}",
                event_type=BasketballEventType.POSSESSION_CHANGE,
                frame_start=max(0, int(frame_id) - self._candidate_frames + 1),
                frame_end=int(frame_id),
                timestamp_start_sec=timestamp,
                timestamp_end_sec=timestamp,
                confidence=confidence,
                uncertainty=1.0 - confidence,
                decision=EventDecision.CONFIRMED,
                actor_track_id=usable.track_id,
                secondary_track_id=previous_track,
                team_id=new_team,
                location_court_m=None if player is None else player.court_position_m,
                evidence=("ball_player_proximity", "temporal_possession_switch"),
                provenance={"version": "possession_tracker_v1"},
                attributes={**self._flight_attributes()},
            )
        )

        transition = self._transition_candidate(
            frame_id=frame_id,
            timestamp_sec=timestamp,
            previous_track=previous_track,
            previous_team=previous_team,
            new_track=usable.track_id,
            new_team=new_team,
            possession_confidence=confidence,
            location_court_m=None if player is None else player.court_position_m,
            transition_kind="direct_temporal_switch",
        )
        if transition is not None:
            events.append(transition)

        self.control_state = BallControlState.POSSESSED
        self._remember_control(frame_id=frame_id, player=player, ball_state=ball_state)
        self._clear_transition_memory()
        return events

    def release_for_airborne_ball(
        self,
        *,
        reason: str = "shot",
        frame_id: int | None = None,
    ) -> None:
        if self.current_track_id is not None:
            self._last_control_track_id = self.current_track_id
            self._last_control_team_id = self.current_team_id
            self._last_control_confidence = self.current_confidence
            if frame_id is not None:
                self._last_control_frame = int(frame_id)
        self.current_track_id = None
        self.current_team_id = None
        self.current_confidence = 0.0
        self._candidate_track_id = None
        self._candidate_frames = 0
        self._candidate_score = 0.0
        self._lost_frames = 0
        self._clear_transition_memory()
        self.control_state = (
            BallControlState.SHOT_FLIGHT
            if str(reason).lower() == "shot"
            else BallControlState.UNKNOWN
        )

    def reset(self, *, reason: str = "external_reset") -> None:
        self.current_track_id = None
        self.current_team_id = None
        self.current_confidence = 0.0
        self.control_state = BallControlState.UNKNOWN
        self._candidate_track_id = None
        self._candidate_frames = 0
        self._candidate_score = 0.0
        self._lost_frames = 0
        self._last_control_track_id = None
        self._last_control_team_id = None
        self._last_control_confidence = 0.0
        self._last_control_frame = None
        self._last_control_ball_position = None
        self._last_control_player_height = 1.0
        self._clear_transition_memory()
        self._reset_count += 1
        self._last_reset_reason = str(reason)

    def diagnostics(self) -> dict[str, Any]:
        flight = self._flight
        return {
            "version": "possession_tracker_v1",
            "control_state": self.control_state.value,
            "current_track_id": self.current_track_id,
            "current_team_id": self.current_team_id,
            "current_confidence": round(float(self.current_confidence), 6),
            "candidate_track_id": self._candidate_track_id,
            "candidate_frames": self._candidate_frames,
            "candidate_score": round(float(self._candidate_score), 6),
            "lost_frames": self._lost_frames,
            "loose_start_frame": self._loose_start_frame,
            "flight": {
                "active": flight is not None,
                "origin_track_id": None if flight is None else flight.origin_track_id,
                "origin_team_id": None if flight is None else flight.origin_team_id,
                "frame_start": None if flight is None else flight.frame_start,
                **self._flight_attributes(),
            },
            "recent_owner_track_id": self._last_control_track_id,
            "recent_owner_team_id": self._last_control_team_id,
            "recent_owner_frame": self._last_control_frame,
            "reset_count": self._reset_count,
            "last_reset_reason": self._last_reset_reason,
        }
