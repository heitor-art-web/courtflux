from __future__ import annotations

from typing import Any

from core.basketball_event import BasketballEvent, BasketballEventType, EventDecision
from services.p2.two_stage_airborne_live_control_v1 import (
    LiveControlP2EngineV1 as _BaseLiveControlP2EngineV1,
)


class LiveControlP2EngineV1(_BaseLiveControlP2EngineV1):
    """Add conservative public-possession continuity across short release episodes.

    The inherited trackers remain the only control-state decision makers. This layer keeps
    only public, already-supported owner/team snapshots and uses them in two narrow ways:

    * a recent public owner may attribute a shot that starts just after possession clears;
    * a release episode may be finalized retrospectively as a pass after a different
      receiver is confirmed and the receiver is publicly supported as the same team.

    No possession thresholds, team thresholds, track decisions, or P1 state are changed.
    Temporal state is cleared on the same reset boundaries as the canonical engine.
    """

    TEMPORAL_CONTINUITY_POLICY = (
        "public_supported_owner_snapshot_only;"
        "shot_fallback_requires_no_current_owner_and_recent_same_actor_snapshot;"
        "pass_requires_release_interval_confirmed_distinct_receiver_and_same_public_team;"
        "conflicting_or_missing_team_abstains;resets_clear_all_continuity_memory"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._public_owner_snapshot: dict[str, Any] | None = None
        self._pending_release_episode: dict[str, Any] | None = None
        self._last_continuity_pass_event_id: str | None = None
        self._receiver_resolution_max_frames = max(12, int(round(0.6 * self.fps)))
        self._release_terminal_grace_frames = max(2, int(self.possession_tracker.switch_frames))

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _snapshot_is_recent(self, snapshot: dict[str, Any] | None, frame_id: int) -> bool:
        if not snapshot:
            return False
        snapshot_frame = self._safe_int(snapshot.get("frame_id"))
        if snapshot_frame is None:
            return False
        age = int(frame_id) - snapshot_frame
        return bool(0 <= age <= int(self.recent_shot_actor_frames))

    def _pre_release_public_attribution(
        self,
        *,
        frame_id: int,
    ) -> tuple[int | None, str | None, str]:
        track_id, team_id, source = super()._pre_release_public_attribution(frame_id=frame_id)
        if team_id is not None:
            return track_id, team_id, source

        # If a current owner exists but public team publication abstains, do not revive an
        # older team's snapshot. This protects challenger/contradiction cases such as V5#2.
        if self.possession_tracker.current_track_id is not None:
            return track_id, team_id, source

        snapshot = self._public_owner_snapshot
        if not self._snapshot_is_recent(snapshot, int(frame_id)):
            return track_id, team_id, source
        assert snapshot is not None
        snapshot_track = self._safe_int(snapshot.get("track_id"))
        snapshot_team = snapshot.get("team_id")
        if snapshot_track is None or snapshot_team is None:
            return track_id, team_id, source
        if track_id is not None and int(track_id) != snapshot_track:
            return track_id, team_id, source
        return snapshot_track, str(snapshot_team), "recent_public_possession_snapshot"

    def _reset_temporal_state(
        self,
        *,
        reason: str,
        frame_id: int,
        timestamp: float | None,
    ) -> BasketballEvent | None:
        event = super()._reset_temporal_state(
            reason=reason,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        self._public_owner_snapshot = None
        self._pending_release_episode = None
        self._last_continuity_pass_event_id = None
        return event

    def _public_owner_from_p2(
        self,
        *,
        p2: dict[str, Any],
        frame_state: Any,
    ) -> dict[str, Any] | None:
        possession = dict(p2.get("possession", {}) or {})
        track_id = self._safe_int(possession.get("current_track_id"))
        team_id = possession.get("current_team_id")
        if track_id is None or team_id is None:
            return None
        if bool(possession.get("team_attribution_abstained", False)):
            return None
        frame_id = int(getattr(frame_state, "frame_id", 0))
        return {
            "track_id": track_id,
            "team_id": str(team_id),
            "frame_id": frame_id,
            "timestamp": self._timestamp(frame_state),
            "confidence": max(
                0.0,
                min(self._safe_float(possession.get("current_confidence"), 0.0), 1.0),
            ),
            "location_court_m": self._player_location(frame_state, track_id),
            "source": (
                possession.get("team_attribution_fallback_source")
                if bool(possession.get("team_attribution_fallback_applied", False))
                else "public_possession"
            ),
        }

    def _update_public_owner_snapshot(
        self,
        *,
        p2: dict[str, Any],
        frame_state: Any,
    ) -> None:
        possession = dict(p2.get("possession", {}) or {})
        current_track = self._safe_int(possession.get("current_track_id"))
        supported = self._public_owner_from_p2(p2=p2, frame_state=frame_state)
        if supported is not None:
            self._public_owner_snapshot = supported
            return

        # A confirmed different owner with no publishable team invalidates the older owner
        # snapshot. Ownerless flight does not: that is exactly the gap continuity bridges.
        if current_track is None or self._public_owner_snapshot is None:
            return
        previous_track = self._safe_int(self._public_owner_snapshot.get("track_id"))
        if previous_track is not None and current_track != previous_track:
            self._public_owner_snapshot = None

    def _origin_snapshot_for_flight(
        self,
        *,
        p2: dict[str, Any],
        frame_state: Any,
        origin_track_id: int,
    ) -> dict[str, Any] | None:
        current = self._public_owner_from_p2(p2=p2, frame_state=frame_state)
        if current is not None and self._safe_int(current.get("track_id")) == int(origin_track_id):
            return current
        snapshot = self._public_owner_snapshot
        if snapshot is None:
            return None
        if self._safe_int(snapshot.get("track_id")) != int(origin_track_id):
            return None
        return dict(snapshot)

    def _start_or_update_release_episode(
        self,
        *,
        p2: dict[str, Any],
        frame_state: Any,
    ) -> None:
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        if not bool(flight.get("active", False)):
            return
        origin_track = self._safe_int(flight.get("origin_track_id"))
        if origin_track is None:
            return
        frame_id = int(getattr(frame_state, "frame_id", 0))
        origin = self._origin_snapshot_for_flight(
            p2=p2,
            frame_state=frame_state,
            origin_track_id=origin_track,
        )
        if origin is None or origin.get("team_id") is None:
            return

        episode = self._pending_release_episode
        if episode is None or self._safe_int(episode.get("origin_track_id")) != origin_track:
            flight_start = self._safe_int(flight.get("frame_start"))
            if flight_start is None:
                flight_start = frame_id
            timestamp = self._timestamp(frame_state)
            timestamp_start = None
            if timestamp is not None:
                timestamp_start = max(0.0, float(timestamp) - max(0, frame_id - flight_start) / self.fps)
            episode = {
                "origin_track_id": origin_track,
                "origin_team_id": str(origin["team_id"]),
                "origin_confidence": self._safe_float(origin.get("confidence"), 0.0),
                "origin_snapshot_frame": self._safe_int(origin.get("frame_id")),
                "frame_start": flight_start,
                "timestamp_start": timestamp_start,
                "last_flight_frame": frame_id,
                "flight_frames_max": 0,
                "flight_direct_observations_max": 0,
                "flight_predicted_observations_max": 0,
                "flight_supported_seen": False,
                "receiver_track_id": None,
                "receiver_confirmed_frame": None,
                "supersedes_transition_event_id": None,
            }
            self._pending_release_episode = episode

        episode["last_flight_frame"] = frame_id
        episode["flight_frames_max"] = max(
            int(episode.get("flight_frames_max", 0) or 0),
            int(flight.get("flight_frames", 0) or 0),
        )
        episode["flight_direct_observations_max"] = max(
            int(episode.get("flight_direct_observations_max", 0) or 0),
            int(flight.get("flight_direct_observations", 0) or 0),
        )
        episode["flight_predicted_observations_max"] = max(
            int(episode.get("flight_predicted_observations_max", 0) or 0),
            int(flight.get("flight_predicted_observations", 0) or 0),
        )
        episode["flight_supported_seen"] = bool(
            episode.get("flight_supported_seen", False)
            or flight.get("flight_supported", False)
        )

    @staticmethod
    def _event_rows(p2: dict[str, Any]) -> list[dict[str, Any]]:
        return [row for row in list(p2.get("events", []) or []) if isinstance(row, dict)]

    def _confirmed_receiver_from_events(
        self,
        *,
        p2: dict[str, Any],
        origin_track_id: int,
    ) -> tuple[int | None, str | None]:
        terminal_types = {
            BasketballEventType.POSSESSION_ACQUIRED.value,
            BasketballEventType.POSSESSION_CHANGE.value,
            BasketballEventType.LOOSE_BALL_RECOVERED.value,
        }
        receiver = None
        supersedes = None
        for row in self._event_rows(p2):
            event_type = str(row.get("event_type") or "")
            actor = self._safe_int(row.get("actor_track_id"))
            if event_type in terminal_types and actor is not None and actor != int(origin_track_id):
                receiver = actor
            if event_type == BasketballEventType.TURNOVER_CANDIDATE.value:
                actor = self._safe_int(row.get("actor_track_id"))
                secondary = self._safe_int(row.get("secondary_track_id"))
                if actor == int(origin_track_id) and secondary is not None:
                    supersedes = str(row.get("event_id") or "") or None
        return receiver, supersedes

    def _release_episode_has_minimum_evidence(self, episode: dict[str, Any]) -> bool:
        frames = int(episode.get("flight_frames_max", 0) or 0)
        observations = int(episode.get("flight_direct_observations_max", 0) or 0) + int(
            episode.get("flight_predicted_observations_max", 0) or 0
        )
        return bool(
            frames >= int(self.possession_tracker.release_confirm_frames)
            and observations > 0
        )

    def _same_team_receiver_supported(
        self,
        *,
        p2: dict[str, Any],
        receiver_track_id: int,
        origin_team_id: str,
    ) -> tuple[bool, float, str]:
        possession = dict(p2.get("possession", {}) or {})
        current_track = self._safe_int(possession.get("current_track_id"))
        public_team = possession.get("current_team_id")
        abstained = bool(possession.get("team_attribution_abstained", False))
        if current_track == int(receiver_track_id) and public_team is not None and not abstained:
            if str(public_team) == str(origin_team_id):
                confidence = max(
                    0.0,
                    min(self._safe_float(possession.get("current_confidence"), 0.0), 1.0),
                )
                return True, confidence, "receiver_public_possession_team"
            return False, 0.0, "receiver_public_team_conflict"

        hint_team, hint_confidence, _ = self._recent_event_team_hint(
            int(receiver_track_id),
            frame_id=int(self._last_frame_id or 0),
        )
        if hint_team is not None and str(hint_team) == str(origin_team_id):
            return True, float(hint_confidence), "receiver_fresh_on_court_team_hint"
        return False, 0.0, "receiver_team_unresolved"

    def _existing_pass_for_pair(
        self,
        *,
        p2: dict[str, Any],
        origin_track_id: int,
        receiver_track_id: int,
    ) -> bool:
        for row in self._event_rows(p2):
            if str(row.get("event_type") or "") != BasketballEventType.PASS_CANDIDATE.value:
                continue
            if self._safe_int(row.get("actor_track_id")) != int(origin_track_id):
                continue
            if self._safe_int(row.get("secondary_track_id")) != int(receiver_track_id):
                continue
            return True
        return False

    def _make_delayed_pass(
        self,
        *,
        episode: dict[str, Any],
        receiver_track_id: int,
        receiver_confidence: float,
        receiver_source: str,
        frame_state: Any,
    ) -> BasketballEvent:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        origin_confidence = max(0.0, min(self._safe_float(episode.get("origin_confidence"), 0.0), 1.0))
        receiver_confidence = max(0.0, min(float(receiver_confidence), 1.0))
        confidence = max(0.35, min(0.88 * min(max(origin_confidence, 0.55), max(receiver_confidence, 0.55)), 1.0))
        origin_track = int(episode["origin_track_id"])
        origin_team = str(episode["origin_team_id"])
        event = BasketballEvent(
            event_id=f"pass-candidate:delayed-continuity:{episode['frame_start']}:{frame_id}:{origin_track}:{receiver_track_id}",
            event_type=BasketballEventType.PASS_CANDIDATE,
            frame_start=int(episode["frame_start"]),
            frame_end=frame_id,
            timestamp_start_sec=episode.get("timestamp_start"),
            timestamp_end_sec=self._timestamp(frame_state),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            decision=EventDecision.CANDIDATE,
            actor_track_id=origin_track,
            secondary_track_id=int(receiver_track_id),
            team_id=origin_team,
            location_court_m=self._player_location(frame_state, int(receiver_track_id)),
            evidence=(
                "public_origin_possession_snapshot",
                "release_interval",
                "confirmed_distinct_receiver",
                "same_public_team_after_release",
            ),
            provenance={"version": "temporal_continuity_live_control_v1"},
            attributes={
                "transition_kind": "delayed_same_team_release_resolution",
                "temporal_continuity_policy": self.TEMPORAL_CONTINUITY_POLICY,
                "team_attribution_publication_source": receiver_source,
                "team_attribution_publication_reason": "same_public_team_after_confirmed_release",
                "origin_snapshot_frame": episode.get("origin_snapshot_frame"),
                "receiver_confirmed_frame": episode.get("receiver_confirmed_frame"),
                "resolution_delay_frames": (
                    None
                    if episode.get("receiver_confirmed_frame") is None
                    else frame_id - int(episode["receiver_confirmed_frame"])
                ),
                "flight_frames": int(episode.get("flight_frames_max", 0) or 0),
                "flight_direct_observations": int(episode.get("flight_direct_observations_max", 0) or 0),
                "flight_predicted_observations": int(episode.get("flight_predicted_observations_max", 0) or 0),
                "flight_supported_seen": bool(episode.get("flight_supported_seen", False)),
                "supersedes_transition_event_id": episode.get("supersedes_transition_event_id"),
            },
        )
        return event

    def _resolve_release_episode(
        self,
        *,
        p2: dict[str, Any],
        frame_state: Any,
    ) -> BasketballEvent | None:
        episode = self._pending_release_episode
        if episode is None:
            return None
        frame_id = int(getattr(frame_state, "frame_id", 0))
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        if bool(flight.get("active", False)):
            return None

        shot = dict(p2.get("shot", {}) or {})
        if bool(shot.get("active", False)) or any(
            str(row.get("event_type") or "") == BasketballEventType.SHOT_ATTEMPT.value
            for row in self._event_rows(p2)
        ):
            self._pending_release_episode = None
            return None

        origin_track = int(episode["origin_track_id"])
        receiver_track = self._safe_int(episode.get("receiver_track_id"))
        if receiver_track is None:
            receiver_track, supersedes = self._confirmed_receiver_from_events(
                p2=p2,
                origin_track_id=origin_track,
            )
            if receiver_track is not None:
                episode["receiver_track_id"] = receiver_track
                episode["receiver_confirmed_frame"] = frame_id
                if supersedes is not None:
                    episode["supersedes_transition_event_id"] = supersedes

        if receiver_track is None:
            current_track = self._safe_int(possession.get("current_track_id"))
            last_flight = int(episode.get("last_flight_frame", frame_id) or frame_id)
            if current_track == origin_track:
                self._pending_release_episode = None
            elif frame_id - last_flight > int(self._release_terminal_grace_frames):
                self._pending_release_episode = None
            return None

        if not self._release_episode_has_minimum_evidence(episode):
            self._pending_release_episode = None
            return None
        if self._existing_pass_for_pair(
            p2=p2,
            origin_track_id=origin_track,
            receiver_track_id=receiver_track,
        ):
            self._pending_release_episode = None
            return None

        supported, receiver_confidence, receiver_source = self._same_team_receiver_supported(
            p2=p2,
            receiver_track_id=receiver_track,
            origin_team_id=str(episode["origin_team_id"]),
        )
        if supported:
            event = self._make_delayed_pass(
                episode=episode,
                receiver_track_id=receiver_track,
                receiver_confidence=receiver_confidence,
                receiver_source=receiver_source,
                frame_state=frame_state,
            )
            self._last_continuity_pass_event_id = event.event_id
            self._pending_release_episode = None
            return event

        confirmed_frame = self._safe_int(episode.get("receiver_confirmed_frame"))
        if confirmed_frame is not None and frame_id - confirmed_frame > int(self._receiver_resolution_max_frames):
            self._pending_release_episode = None
        return None

    @staticmethod
    def _append_recovered_event(frame_state: Any, p2: dict[str, Any], event: BasketballEvent) -> None:
        if not hasattr(frame_state, "p2_events") or frame_state.p2_events is None:
            frame_state.p2_events = []
        frame_state.p2_events.append(event)
        rows = list(p2.get("events", []) or [])
        rows.append(event.as_dict())
        p2["events"] = rows
        p2["event_count"] = len(rows)

    def _write_continuity_diagnostics(self, p2: dict[str, Any]) -> None:
        snapshot = None if self._public_owner_snapshot is None else dict(self._public_owner_snapshot)
        if snapshot is not None and snapshot.get("location_court_m") is not None:
            snapshot["location_court_m"] = list(snapshot["location_court_m"])
        episode = None if self._pending_release_episode is None else dict(self._pending_release_episode)
        p2["temporal_event_continuity"] = {
            "policy": self.TEMPORAL_CONTINUITY_POLICY,
            "public_owner_snapshot": snapshot,
            "pending_release_episode": episode,
            "receiver_resolution_max_frames": int(self._receiver_resolution_max_frames),
            "last_recovered_pass_event_id": self._last_continuity_pass_event_id,
        }
        p2["behavior_revision"] = "control_state_v9_temporal_event_continuity_v1"

    def process(self, frame_state: Any) -> Any:
        result = super().process(frame_state)
        diagnostics = getattr(result, "diagnostics", None)
        p2 = diagnostics.get("p2") if isinstance(diagnostics, dict) else None
        if not isinstance(p2, dict):
            return result

        engine_eligible = bool(p2.get("engine_eligible", False))
        skipped_reason = p2.get("skipped_reason")
        if engine_eligible and skipped_reason not in {"gameplay_not_present"}:
            self._start_or_update_release_episode(p2=p2, frame_state=result)
            recovered = self._resolve_release_episode(p2=p2, frame_state=result)
            if recovered is not None:
                self._append_recovered_event(result, p2, recovered)
            self._update_public_owner_snapshot(p2=p2, frame_state=result)
        self._write_continuity_diagnostics(p2)
        return result
