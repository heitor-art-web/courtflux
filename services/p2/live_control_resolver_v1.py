from __future__ import annotations

import math
from typing import Any

from core.ball_state import BallStateStatus
from core.basketball_event import BallControlState
from services.p2.engine_v1 import P2EngineV1 as _BaseP2EngineV1


class LiveControlP2EngineV1(_BaseP2EngineV1):
    """Single-engine P2 runtime with a causal published control-state resolver.

    The underlying possession and shot trackers remain the only event-producing state
    machines. This class calls the base engine exactly once on eligible frames and only
    resolves the published control state afterwards. The raw tracker decision is retained
    as ``tracker_control_state`` for auditability.

    Release flight has an internal AIRBORNE_UNCLASSIFIED phase which is never published as
    Gate-A truth. SHOT classification always has precedence. PASS requires a distinct
    receiver plus direct ball motion converging toward that receiver. While airborne is
    still unclassified, recent-owner continuity remains published as POSSESSED rather than
    falling back immediately to UNKNOWN/LOOSE.
    """

    AIRBORNE_UNCLASSIFIED = "airborne_unclassified"
    AIRBORNE_SHOT = "shot_flight"
    AIRBORNE_PASS = "pass_flight"

    SHOT_PREFLIGHT_MAX_DISTANCE_NORM = 8.0
    LIVE_CONTROL_POLICY = (
        "causal_live_resolver_v2;internal_airborne_unclassified;shot_first;"
        "pass_requires_distinct_receiver_and_converging_direct_trajectory;"
        "recent_owner_continuity;shot_preflight_above_distance<=8;"
        "gameplay_gap_grace_uses_recent_shot_actor_horizon;events_unchanged"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        explicit_reset_contract = "reset_after_ineligible_frames" in kwargs
        super().__init__(*args, **kwargs)
        self._live_control_state = BallControlState.UNKNOWN
        self._live_control_source = "initial_unknown"
        self._live_control_latch: BallControlState | None = None
        self._live_ineligible_frames = 0
        self._live_seen_reset_count = int(self._reset_count)
        self._airborne_state: str | None = None
        self._previous_direct_ball_position: tuple[float, float] | None = None
        self._previous_direct_ball_frame: int | None = None
        self._receiver_trajectory_track_id: int | None = None
        self._receiver_trajectory_converging = False
        # Normal runtime reuses the existing recent-shot-actor continuity horizon. When
        # reset_after_ineligible_frames is explicitly configured, preserve the legacy
        # boundary: reset on frame N means only N-1 frames may be held as grace.
        self.gameplay_continuity_grace_frames = (
            max(0, self.reset_after_ineligible_frames - 1)
            if explicit_reset_contract
            else max(self.reset_after_ineligible_frames, self.recent_shot_actor_frames)
        )

    def _reset_live_control(self, reason: str) -> None:
        self._live_control_state = BallControlState.UNKNOWN
        self._live_control_source = f"reset:{reason}"
        self._live_control_latch = None
        self._airborne_state = None
        self._previous_direct_ball_position = None
        self._previous_direct_ball_frame = None
        self._receiver_trajectory_track_id = None
        self._receiver_trajectory_converging = False

    @staticmethod
    def _semantic_shot_signal(rows: list[dict[str, Any]]) -> bool:
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                confidence = max(0.0, min(float(row.get("confidence", 0.0)), 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.35:
                continue
            label = str(
                row.get("class_name")
                or row.get("label")
                or row.get("kind")
                or row.get("class")
                or ""
            ).strip().lower().replace("_", "-")
            if "player-jump-shot" in label or "player-layup-dunk" in label:
                return True
        return False

    @staticmethod
    def _ball_position(frame_state: Any) -> tuple[float, float] | None:
        ball_state = getattr(frame_state, "ball_state", None)
        if ball_state is None or not bool(getattr(ball_state, "available", False)):
            return None
        position_fn = getattr(ball_state, "position", None)
        if not callable(position_fn):
            return None
        position = position_fn()
        if position is None:
            return None
        try:
            return float(position[0]), float(position[1])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _direct_ball_position(frame_state: Any) -> tuple[float, float] | None:
        ball_state = getattr(frame_state, "ball_state", None)
        if ball_state is None or getattr(ball_state, "status", None) is not BallStateStatus.OBSERVED:
            return None
        return LiveControlP2EngineV1._ball_position(frame_state)

    @staticmethod
    def _player_center(frame_state: Any, track_id: int | None) -> tuple[float, float] | None:
        if track_id is None:
            return None
        for player in list(getattr(frame_state, "player_states", []) or []):
            if getattr(player, "track_id", None) != track_id:
                continue
            bbox = getattr(player, "bbox", None)
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return None
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                return None
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return None

    @staticmethod
    def _rim_boxes(frame_state: Any) -> list[tuple[float, float, float, float]]:
        court_state = getattr(frame_state, "court_state", None)
        if court_state is None:
            return []
        result: list[tuple[float, float, float, float]] = []
        for row in list(getattr(court_state, "basket_evidence", None) or []):
            if not isinstance(row, dict) or str(row.get("kind") or "").lower() != "rim":
                continue
            raw = row.get("bbox_px", row.get("box", row.get("bbox")))
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in raw[:4]]
            except (TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2, y2))
        return result

    def _shot_preflight(self, frame_state: Any) -> dict[str, Any]:
        position = self._ball_position(frame_state)
        rims = self._rim_boxes(frame_state)
        if position is None or not rims:
            return {
                "available": False,
                "above": False,
                "distance_norm": None,
                "signal": False,
            }

        bx, by = position
        rim = min(
            rims,
            key=lambda candidate: (
                ((candidate[0] + candidate[2]) / 2.0 - bx) ** 2
                + ((candidate[1] + candidate[3]) / 2.0 - by) ** 2
            ),
        )
        x1, y1, x2, y2 = rim
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        rim_x = (x1 + x2) / 2.0
        rim_y = (y1 + y2) / 2.0
        relative_x = (bx - rim_x) / width
        relative_y = (by - rim_y) / height
        distance_norm = math.hypot(relative_x, relative_y)
        above = relative_y < -0.20
        signal = bool(above and distance_norm <= self.SHOT_PREFLIGHT_MAX_DISTANCE_NORM)
        return {
            "available": True,
            "above": bool(above),
            "distance_norm": round(float(distance_norm), 6),
            "signal": signal,
        }

    def _receiver_trajectory(
        self,
        *,
        frame_state: Any,
        possession: dict[str, Any],
        flight_active: bool,
        receiver_evidence: bool,
    ) -> dict[str, Any]:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        raw_receiver = possession.get("candidate_track_id")
        receiver_track_id: int | None = None
        try:
            if raw_receiver is not None:
                receiver_track_id = int(raw_receiver)
        except (TypeError, ValueError):
            receiver_track_id = None

        current_ball = self._direct_ball_position(frame_state)
        receiver_center = self._player_center(frame_state, receiver_track_id)
        previous_ball = self._previous_direct_ball_position
        previous_frame = self._previous_direct_ball_frame
        consecutive_direct = bool(
            current_ball is not None
            and previous_ball is not None
            and previous_frame is not None
            and frame_id == int(previous_frame) + 1
        )

        previous_distance = None
        current_distance = None
        projection = None
        converging = False
        if (
            flight_active
            and receiver_evidence
            and receiver_track_id is not None
            and receiver_center is not None
            and consecutive_direct
            and current_ball is not None
            and previous_ball is not None
        ):
            rx, ry = receiver_center
            px, py = previous_ball
            cx, cy = current_ball
            previous_distance = math.hypot(rx - px, ry - py)
            current_distance = math.hypot(rx - cx, ry - cy)
            move_x = cx - px
            move_y = cy - py
            target_x = rx - px
            target_y = ry - py
            projection = move_x * target_x + move_y * target_y
            converging = bool(current_distance < previous_distance and projection > 0.0)

        self._receiver_trajectory_track_id = receiver_track_id
        self._receiver_trajectory_converging = converging
        return {
            "receiver_track_id": receiver_track_id,
            "receiver_center_available": receiver_center is not None,
            "consecutive_direct_ball_observations": consecutive_direct,
            "previous_distance_px": (
                None if previous_distance is None else round(float(previous_distance), 3)
            ),
            "current_distance_px": (
                None if current_distance is None else round(float(current_distance), 3)
            ),
            "directional_projection": None if projection is None else round(float(projection), 3),
            "converging": converging,
        }

    def _remember_direct_ball(self, frame_state: Any) -> None:
        position = self._direct_ball_position(frame_state)
        if position is None:
            self._previous_direct_ball_position = None
            self._previous_direct_ball_frame = None
            return
        self._previous_direct_ball_position = position
        self._previous_direct_ball_frame = int(getattr(frame_state, "frame_id", 0))

    def _recent_owner_context(
        self,
        *,
        possession: dict[str, Any],
        frame_id: int,
    ) -> tuple[bool, int | None]:
        raw_track = possession.get("recent_owner_track_id")
        raw_frame = possession.get("recent_owner_frame")
        if raw_track is None or raw_frame is None:
            return False, None
        try:
            age = int(frame_id) - int(raw_frame)
        except (TypeError, ValueError):
            return False, None
        return bool(0 <= age <= self.recent_shot_actor_frames), age

    def _sync_live_reset(self, p2: dict[str, Any]) -> None:
        temporal = dict(p2.get("temporal_reset", {}) or {})
        try:
            reset_count = int(temporal.get("count", 0) or 0)
        except (TypeError, ValueError):
            reset_count = self._live_seen_reset_count
        if reset_count > self._live_seen_reset_count:
            self._reset_live_control(str(temporal.get("last_reason") or "temporal_reset"))
            self._live_seen_reset_count = reset_count

    def _resolve_eligible_control(
        self,
        *,
        frame_state: Any,
        p2: dict[str, Any],
        semantic_evidence: list[dict[str, Any]],
    ) -> tuple[BallControlState, str, dict[str, Any]]:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        possession = dict(p2.get("possession", {}) or {})
        shot = dict(p2.get("shot", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        tracker_state_raw = str(possession.get("control_state") or "unknown")
        try:
            tracker_state = BallControlState(tracker_state_raw)
        except ValueError:
            tracker_state = BallControlState.UNKNOWN

        flight_active = bool(flight.get("active", False))
        receiver_evidence = bool(possession.get("pass_flight_receiver_evidence", False))
        shot_active = bool(shot.get("active", False))
        shot_event_frame = any(
            isinstance(event, dict) and str(event.get("event_type") or "") == "shot_attempt"
            for event in list(p2.get("events", []) or [])
        )
        semantic_shot = self._semantic_shot_signal(semantic_evidence)
        preflight = self._shot_preflight(frame_state)
        # SHOT is evaluated before PASS. Receiver evidence never suppresses the frozen
        # geometry signal; this deliberately removes the old pass-first ambiguity.
        preflight_shot = bool(
            flight_active
            and tracker_state is not BallControlState.POSSESSED
            and preflight["signal"]
        )
        recent_owner, recent_owner_age = self._recent_owner_context(
            possession=possession,
            frame_id=frame_id,
        )
        receiver_trajectory = self._receiver_trajectory(
            frame_state=frame_state,
            possession=possession,
            flight_active=flight_active,
            receiver_evidence=receiver_evidence,
        )

        resolver_diag = {
            "tracker_control_state": tracker_state.value,
            "flight_active": flight_active,
            "receiver_evidence": receiver_evidence,
            "recent_owner": recent_owner,
            "recent_owner_age_frames": recent_owner_age,
            "shot_active": shot_active,
            "shot_event_frame": shot_event_frame,
            "semantic_shot_signal": semantic_shot,
            "shot_preflight": preflight,
            "receiver_trajectory": receiver_trajectory,
            "internal_airborne_state": self._airborne_state,
        }

        shot_signal = bool(
            shot_active
            or shot_event_frame
            or semantic_shot
            or tracker_state is BallControlState.SHOT_FLIGHT
            or preflight_shot
        )
        if shot_signal:
            self._airborne_state = self.AIRBORNE_SHOT
            self._live_control_latch = BallControlState.SHOT_FLIGHT
            if shot_active:
                source = "shot_tracker_active"
            elif shot_event_frame:
                source = "shot_event_frame"
            elif semantic_shot:
                source = "semantic_shot_action"
            elif preflight_shot:
                source = "frozen_shot_preflight"
            else:
                source = "tracker_shot_flight"
            resolver_diag["internal_airborne_state"] = self._airborne_state
            return BallControlState.SHOT_FLIGHT, source, resolver_diag

        if self._live_control_latch is BallControlState.SHOT_FLIGHT:
            if shot_active or flight_active:
                self._airborne_state = self.AIRBORNE_SHOT
                resolver_diag["internal_airborne_state"] = self._airborne_state
                return BallControlState.SHOT_FLIGHT, "latched_shot_flight", resolver_diag
            self._live_control_latch = None
            self._airborne_state = None

        pass_signal = bool(
            flight_active
            and receiver_evidence
            and bool(receiver_trajectory.get("converging", False))
        )
        if pass_signal:
            self._airborne_state = self.AIRBORNE_PASS
            self._live_control_latch = BallControlState.PASS_FLIGHT
            resolver_diag["internal_airborne_state"] = self._airborne_state
            return BallControlState.PASS_FLIGHT, "receiver_trajectory_pass", resolver_diag

        if self._live_control_latch is BallControlState.PASS_FLIGHT:
            if flight_active and tracker_state is not BallControlState.POSSESSED:
                self._airborne_state = self.AIRBORNE_PASS
                resolver_diag["internal_airborne_state"] = self._airborne_state
                return BallControlState.PASS_FLIGHT, "latched_pass_flight", resolver_diag
            self._live_control_latch = None
            self._airborne_state = None

        if (
            tracker_state is BallControlState.POSSESSED
            and possession.get("current_track_id") is not None
        ):
            self._live_control_latch = None
            self._airborne_state = None
            resolver_diag["internal_airborne_state"] = None
            return BallControlState.POSSESSED, "tracker_possessed", resolver_diag

        if flight_active:
            self._airborne_state = self.AIRBORNE_UNCLASSIFIED
            resolver_diag["internal_airborne_state"] = self._airborne_state
            if recent_owner:
                return BallControlState.POSSESSED, "airborne_unclassified_owner_continuity", resolver_diag

        if recent_owner and tracker_state in {
            BallControlState.UNKNOWN,
            BallControlState.LOOSE,
            BallControlState.PASS_FLIGHT,
        }:
            return BallControlState.POSSESSED, "recent_owner_continuity", resolver_diag

        if not flight_active:
            self._airborne_state = None
            resolver_diag["internal_airborne_state"] = None
        return tracker_state, "tracker_fallback", resolver_diag

    def _apply_live_override(
        self,
        *,
        frame_state: Any,
        state: BallControlState,
        source: str,
        resolver_diag: dict[str, Any] | None = None,
    ) -> None:
        diagnostics = getattr(frame_state, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            return
        p2 = diagnostics.get("p2")
        if not isinstance(p2, dict):
            return
        possession = dict(p2.get("possession", {}) or {})
        tracker_state = str(possession.get("control_state") or "unknown")
        possession["tracker_control_state"] = tracker_state
        possession["control_state"] = state.value
        possession["live_control_source"] = str(source)
        possession["live_control_policy"] = self.LIVE_CONTROL_POLICY
        possession["live_control_latch"] = (
            None if self._live_control_latch is None else self._live_control_latch.value
        )
        possession["internal_airborne_state"] = self._airborne_state
        p2["possession"] = possession
        p2["behavior_revision"] = "control_state_v6_airborne_classifier"
        p2["eligibility_policy"] = (
            "court_state_gameplay_present_with_recent_shot_actor_horizon_grace"
        )
        p2["live_control"] = {
            "state": state.value,
            "source": str(source),
            "policy": self.LIVE_CONTROL_POLICY,
            "latch": None if self._live_control_latch is None else self._live_control_latch.value,
            "internal_airborne_state": self._airborne_state,
            "gameplay_continuity_grace_frames": int(self.gameplay_continuity_grace_frames),
            "ineligible_gap_frames": int(self._live_ineligible_frames),
            **dict(resolver_diag or {}),
        }
        self._live_control_state = state
        self._live_control_source = str(source)

    def _process_ineligible(self, frame_state: Any) -> Any:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        timestamp = self._timestamp(frame_state)
        semantic_evidence = self._semantic_evidence(frame_state)
        team_evidence = self._team_evidence(frame_state)
        boundary_events = []

        discontinuity = self._detect_discontinuity(frame_id=frame_id, timestamp=timestamp)
        if discontinuity is not None:
            loose_end = self._reset_temporal_state(
                reason=discontinuity,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            if loose_end is not None:
                boundary_events.append(loose_end)
            self._reset_live_control(discontinuity)
            self._live_seen_reset_count = int(self._reset_count)
            self._reset_done_for_ineligible_gap = True

        self._last_frame_id = frame_id
        self._last_timestamp = timestamp
        self._live_ineligible_frames += 1
        self._ineligible_frames = self._live_ineligible_frames

        within_grace = bool(
            discontinuity is None
            and self._live_ineligible_frames <= self.gameplay_continuity_grace_frames
            and self._live_control_state is not BallControlState.UNKNOWN
        )
        if within_grace:
            self._append_frame_events(frame_state, boundary_events)
            self._write_diagnostics(
                frame_state=frame_state,
                emitted=boundary_events,
                semantic_evidence=semantic_evidence,
                team_evidence=team_evidence,
                engine_eligible=True,
                skipped_reason="gameplay_continuity_grace",
            )
            self._apply_live_override(
                frame_state=frame_state,
                state=self._live_control_state,
                source="gameplay_continuity_grace",
                resolver_diag={
                    "grace_preserved_source": self._live_control_source,
                    "raw_gameplay_present": False,
                },
            )
            return frame_state

        if not self._reset_done_for_ineligible_gap:
            loose_end = self._reset_temporal_state(
                reason="gameplay_absent_gap",
                frame_id=frame_id,
                timestamp=timestamp,
            )
            if loose_end is not None:
                boundary_events.append(loose_end)
            self._reset_done_for_ineligible_gap = True
            self._reset_live_control("gameplay_absent_gap")
            self._live_seen_reset_count = int(self._reset_count)

        self._append_frame_events(frame_state, boundary_events)
        self._write_diagnostics(
            frame_state=frame_state,
            emitted=boundary_events,
            semantic_evidence=semantic_evidence,
            team_evidence=team_evidence,
            engine_eligible=False,
            skipped_reason="gameplay_not_present",
        )
        self._apply_live_override(
            frame_state=frame_state,
            state=BallControlState.UNKNOWN,
            source="gameplay_not_present",
            resolver_diag={"raw_gameplay_present": False},
        )
        return frame_state

    def process(self, frame_state: Any) -> Any:
        if not self._analysis_eligible(frame_state):
            return self._process_ineligible(frame_state)

        self._live_ineligible_frames = 0
        self._ineligible_frames = 0
        result = super().process(frame_state)
        self._reset_done_for_ineligible_gap = False

        diagnostics = getattr(result, "diagnostics", None)
        p2 = diagnostics.get("p2") if isinstance(diagnostics, dict) else None
        if not isinstance(p2, dict):
            self._remember_direct_ball(result)
            return result
        self._sync_live_reset(p2)
        semantic_evidence = self._semantic_evidence(result)
        state, source, resolver_diag = self._resolve_eligible_control(
            frame_state=result,
            p2=p2,
            semantic_evidence=semantic_evidence,
        )
        self._apply_live_override(
            frame_state=result,
            state=state,
            source=source,
            resolver_diag=resolver_diag,
        )
        self._remember_direct_ball(result)
        return result
