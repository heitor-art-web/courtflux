from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import BasketballEvent, BasketballEventType, EventDecision
from core.player_state import PlayerState


@dataclass(slots=True)
class _Candidate:
    track_id: int
    score: float
    team_id: str | None
    team_confidence: float


class PossessionTrackerV1:
    """Conservative temporal possession inference over canonical P1 state.

    A nearby ball is evidence, not possession truth. Locks and switches require a
    multi-frame streak. Team evidence is optional and is used only to classify an
    already-supported possession transition as a pass/turnover candidate.
    """

    def __init__(
        self,
        *,
        lock_frames: int = 3,
        switch_frames: int = 3,
        lost_grace_frames: int = 6,
        acquire_threshold: float = 0.58,
        switch_threshold: float = 0.64,
        team_evidence_threshold: float = 0.65,
    ) -> None:
        self.lock_frames = max(1, int(lock_frames))
        self.switch_frames = max(1, int(switch_frames))
        self.lost_grace_frames = max(0, int(lost_grace_frames))
        self.acquire_threshold = max(0.0, min(float(acquire_threshold), 1.0))
        self.switch_threshold = max(0.0, min(float(switch_threshold), 1.0))
        self.team_evidence_threshold = max(0.0, min(float(team_evidence_threshold), 1.0))

        self.current_track_id: int | None = None
        self.current_team_id: str | None = None
        self.current_confidence: float = 0.0
        self._candidate_track_id: int | None = None
        self._candidate_frames = 0
        self._candidate_score = 0.0
        self._lost_frames = 0

    @staticmethod
    def _point_to_box_distance(point: tuple[float, float], box: tuple[int, int, int, int]) -> float:
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
            label = str(row.get("class_name") or row.get("label") or row.get("kind") or "").lower()
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
            box = player.bbox
            height = max(1.0, float(box[3] - box[1]))
            distance = self._point_to_box_distance(ball_position, box)
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
        if player is not None:
            return player.timestamp_sec
        return None

    @staticmethod
    def _player_by_id(players: list[PlayerState], track_id: int | None) -> PlayerState | None:
        if track_id is None:
            return None
        return next((player for player in players if player.track_id == track_id), None)

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

        if usable is None:
            self._advance_candidate(None)
            self._lost_frames += 1
            if self._lost_frames > self.lost_grace_frames:
                self.current_track_id = None
                self.current_team_id = None
                self.current_confidence = 0.0
            return events

        self._lost_frames = 0
        self._advance_candidate(usable)
        player = self._player_by_id(players, usable.track_id)
        timestamp = self._timestamp(player, ball_state)

        if self.current_track_id is None:
            if self._candidate_frames < self.lock_frames:
                return events
            self.current_track_id = usable.track_id
            self.current_team_id = (
                usable.team_id if usable.team_confidence >= self.team_evidence_threshold else None
            )
            self.current_confidence = usable.score
            events.append(
                BasketballEvent(
                    event_id=f"possession-acquired:{frame_id}:{usable.track_id}",
                    event_type=BasketballEventType.POSSESSION_ACQUIRED,
                    frame_start=max(0, int(frame_id) - self._candidate_frames + 1),
                    frame_end=int(frame_id),
                    timestamp_start_sec=timestamp,
                    timestamp_end_sec=timestamp,
                    confidence=usable.score,
                    uncertainty=1.0 - usable.score,
                    decision=EventDecision.CONFIRMED,
                    actor_track_id=usable.track_id,
                    team_id=self.current_team_id,
                    location_court_m=(None if player is None else player.court_position_m),
                    evidence=("ball_player_proximity", "temporal_possession_lock"),
                    provenance={"version": "possession_tracker_v1"},
                )
            )
            return events

        if usable.track_id == self.current_track_id:
            self.current_confidence = max(self.current_confidence * 0.8, usable.score)
            if usable.team_confidence >= self.team_evidence_threshold:
                self.current_team_id = usable.team_id
            return events

        if self._candidate_frames < self.switch_frames:
            return events

        previous_track = self.current_track_id
        previous_team = self.current_team_id
        new_team = usable.team_id if usable.team_confidence >= self.team_evidence_threshold else None
        confidence = usable.score
        self.current_track_id = usable.track_id
        self.current_team_id = new_team
        self.current_confidence = confidence

        common = dict(
            frame_start=max(0, int(frame_id) - self._candidate_frames + 1),
            frame_end=int(frame_id),
            timestamp_start_sec=timestamp,
            timestamp_end_sec=timestamp,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            actor_track_id=usable.track_id,
            secondary_track_id=previous_track,
            team_id=new_team,
            location_court_m=(None if player is None else player.court_position_m),
            evidence=("ball_player_proximity", "temporal_possession_switch"),
            provenance={"version": "possession_tracker_v1"},
        )
        events.append(
            BasketballEvent(
                event_id=f"possession-change:{frame_id}:{previous_track}:{usable.track_id}",
                event_type=BasketballEventType.POSSESSION_CHANGE,
                decision=EventDecision.CONFIRMED,
                **common,
            )
        )

        if previous_team is not None and new_team is not None:
            if previous_team == new_team:
                event_type = BasketballEventType.PASS_CANDIDATE
                prefix = "pass-candidate"
                team_id = new_team
            else:
                event_type = BasketballEventType.TURNOVER_CANDIDATE
                prefix = "turnover-candidate"
                team_id = previous_team
            events.append(
                BasketballEvent(
                    event_id=f"{prefix}:{frame_id}:{previous_track}:{usable.track_id}",
                    event_type=event_type,
                    decision=EventDecision.CANDIDATE,
                    attributes={
                        "from_team_id": previous_team,
                        "to_team_id": new_team,
                        "classification_requires_team_evidence": True,
                    },
                    **{**common, "team_id": team_id},
                )
            )
        return events

    def release_for_airborne_ball(self) -> None:
        """Release the handler without inventing a possession-loss event.

        A supported shot start means the ball is intentionally airborne. The next lock is
        therefore a fresh possession (or rebound evidence), not automatically a pass or
        turnover from the shooter.
        """
        self.current_track_id = None
        self.current_team_id = None
        self.current_confidence = 0.0
        self._candidate_track_id = None
        self._candidate_frames = 0
        self._candidate_score = 0.0
        self._lost_frames = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "version": "possession_tracker_v1",
            "current_track_id": self.current_track_id,
            "current_team_id": self.current_team_id,
            "current_confidence": round(float(self.current_confidence), 6),
            "candidate_track_id": self._candidate_track_id,
            "candidate_frames": self._candidate_frames,
            "lost_frames": self._lost_frames,
        }
