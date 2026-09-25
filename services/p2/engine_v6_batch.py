from __future__ import annotations

from typing import Any

from core.basketball_event import BasketballEventType
from services.p2.engine_v1 import P2EngineV1


class P2EngineV6Batch(P2EngineV1):
    """V6 validation coordinator layered over the frozen P2 engine.

    The base engine remains the owner of possession, transition and rebound semantics.
    This wrapper supplies only the V6 evidence bridges that require coordination across
    existing components:

    * previous-frame generic flight support may seed the V1.3 shot rim-band bridge;
    * shot team publication may use fresh evidence for the *exact actor track* instead of
      abstaining solely because the public possession team is temporarily unavailable;
    * the actor/team evidence used at release is frozen for the lifetime of that shot;
    * post-shot possession reacquisition explicitly rearms the V1.3 shot tracker;
    * a suppressed provisional shot clears its pre-release actor snapshot.
    """

    EVENT_TEAM_ATTRIBUTION_POLICY = (
        "shot_team_accepts_current_or_fresh_exact_actor_team_snapshot;"
        "exact_actor_conflict_abstains;generic_recent_team_never_promoted;"
        "shot_snapshot_frozen_at_release;"
        "pass_team_requires_recent_fresh_on_court_endpoint_support;"
        "turnover_team_requires_recent_fresh_on_court_actor_support;"
        "fresh_same_team_endpoints_may_reclassify_turnover_as_pass;"
        "confirmed_same_team_handler_switch_with_release_interval_may_emit_pass;"
        "event_publication_never_mutates_possession_memory"
    )

    CONTROL_REACQUISITION_EVENT_TYPES = {
        BasketballEventType.POSSESSION_ACQUIRED,
        BasketballEventType.POSSESSION_CHANGE,
        BasketballEventType.LOOSE_BALL_RECOVERED,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._v6_team_evidence: dict[int, tuple[str | None, float]] = {}
        self._v6_team_hints: dict[int, tuple[str, float]] = {}
        self._pre_release_team_snapshot_confidence = 0.0
        self._pre_release_team_snapshot_age_frames: int | None = None
        self._pre_release_team_snapshot_source = "unknown"
        self._active_shot_team_snapshot_confidence = 0.0
        self._active_shot_team_snapshot_age_frames: int | None = None
        self._active_shot_team_snapshot_source = "unknown"

    def _clear_pre_release_snapshot(self) -> None:
        self._pre_release_team_snapshot_confidence = 0.0
        self._pre_release_team_snapshot_age_frames = None
        self._pre_release_team_snapshot_source = "unknown"

    def _clear_active_shot_snapshot(self) -> None:
        self._active_shot_team_snapshot_confidence = 0.0
        self._active_shot_team_snapshot_age_frames = None
        self._active_shot_team_snapshot_source = "unknown"

    def _freeze_pre_release_snapshot_for_active_shot(self) -> None:
        self._active_shot_team_snapshot_confidence = float(
            self._pre_release_team_snapshot_confidence
        )
        self._active_shot_team_snapshot_age_frames = (
            self._pre_release_team_snapshot_age_frames
        )
        self._active_shot_team_snapshot_source = self._pre_release_team_snapshot_source

    def _exact_actor_team_snapshot(
        self,
        actor_track_id: int | None,
        *,
        frame_id: int,
    ) -> tuple[str | None, float, int | None, str]:
        if actor_track_id is None:
            return None, 0.0, None, "unknown"

        team_id, confidence, source = self._attributed_team_for_track(
            int(actor_track_id),
            team_evidence=self._v6_team_evidence,
            team_hints=self._v6_team_hints,
        )
        if team_id is not None:
            return str(team_id), float(confidence), 0, f"exact_actor_{source}"

        recent_team, recent_confidence, age = self._recent_event_team_hint(
            int(actor_track_id),
            frame_id=int(frame_id),
        )
        if recent_team is not None and age is not None:
            return (
                str(recent_team),
                float(recent_confidence),
                int(age),
                "exact_actor_recent_fresh_on_court_hint",
            )
        return None, 0.0, None, "unknown"

    def _pre_release_public_attribution(
        self,
        *,
        frame_id: int,
    ) -> tuple[int | None, str | None, str]:
        actor_track_id, public_team_id, base_source = super()._pre_release_public_attribution(
            frame_id=frame_id,
        )
        self._clear_pre_release_snapshot()
        if actor_track_id is None:
            return actor_track_id, public_team_id, base_source

        exact_team, confidence, age, exact_source = self._exact_actor_team_snapshot(
            actor_track_id,
            frame_id=frame_id,
        )
        if exact_team is not None:
            if public_team_id is not None and str(public_team_id) != str(exact_team):
                self._pre_release_team_snapshot_confidence = float(confidence)
                self._pre_release_team_snapshot_age_frames = age
                self._pre_release_team_snapshot_source = "exact_actor_conflicts_public_team"
                return (
                    int(actor_track_id),
                    None,
                    "pre_release_exact_actor_team_conflict_abstained",
                )
            self._pre_release_team_snapshot_confidence = float(confidence)
            self._pre_release_team_snapshot_age_frames = age
            self._pre_release_team_snapshot_source = exact_source
            return int(actor_track_id), str(exact_team), exact_source

        if public_team_id is not None:
            self._pre_release_team_snapshot_source = base_source
            return int(actor_track_id), str(public_team_id), base_source
        return int(actor_track_id), None, base_source

    def _shot_snapshot_for_publication(self) -> tuple[str, float, int | None]:
        if self._active_shot_team_snapshot_source != "unknown":
            return (
                self._active_shot_team_snapshot_source,
                float(self._active_shot_team_snapshot_confidence),
                self._active_shot_team_snapshot_age_frames,
            )
        # Covers an extremely short shot that begins and resolves inside one process call.
        return (
            self._pre_release_team_snapshot_source,
            float(self._pre_release_team_snapshot_confidence),
            self._pre_release_team_snapshot_age_frames,
        )

    def _attribute_shot(self, event: Any, frame_state: Any) -> Any:
        event = super()._attribute_shot(event, frame_state)
        if event.event_type is not BasketballEventType.SHOT_ATTEMPT:
            return event

        snapshot_source, snapshot_confidence, snapshot_age = (
            self._shot_snapshot_for_publication()
        )
        event.attributes["actor_team_snapshot_source"] = snapshot_source
        event.attributes["actor_team_snapshot_confidence"] = round(
            float(snapshot_confidence),
            6,
        )
        event.attributes["actor_team_snapshot_age_frames"] = snapshot_age
        event.attributes["actor_team_snapshot_frozen_at_release"] = True
        event.attributes["team_attribution_publication_policy"] = self.EVENT_TEAM_ATTRIBUTION_POLICY
        if event.team_id is not None and snapshot_source.startswith("exact_actor_"):
            event.attributes["team_attribution_publication_source"] = snapshot_source
            event.attributes["team_attribution_publication_reason"] = (
                "fresh_exact_actor_team_snapshot_supported"
            )
        elif snapshot_source == "exact_actor_conflicts_public_team":
            event.team_id = None
            event.attributes["team_attribution_publication_source"] = "abstained"
            event.attributes["team_attribution_publication_reason"] = (
                "fresh_exact_actor_team_conflict"
            )
        return event

    @staticmethod
    def _current_frame_events(frame_state: Any, frame_id: int) -> list[Any]:
        output: list[Any] = []
        for event in list(getattr(frame_state, "p2_events", []) or []):
            try:
                event_frame = int(getattr(event, "frame_end", -1))
            except (TypeError, ValueError):
                continue
            if event_frame == int(frame_id):
                output.append(event)
        return output

    def process(self, frame_state: Any) -> Any:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        self._v6_team_evidence = self._team_evidence(frame_state)
        self._v6_team_hints = self._team_attribution_hints(frame_state)

        possession_before = self.possession_tracker.diagnostics()
        supported_flight_before = bool(possession_before.get("flight_supported", False))
        if hasattr(self.shot_tracker, "set_supported_flight_hint"):
            self.shot_tracker.set_supported_flight_hint(supported_flight_before)

        shot_active_before = bool(self.shot_tracker.diagnostics().get("active", False))
        result = super().process(frame_state)
        shot_active_after = bool(self.shot_tracker.diagnostics().get("active", False))

        if not shot_active_before and shot_active_after:
            self._freeze_pre_release_snapshot_for_active_shot()

        current_events = self._current_frame_events(frame_state, frame_id)
        shot_published = any(
            getattr(event, "event_type", None) is BasketballEventType.SHOT_ATTEMPT
            for event in current_events
        )
        reacquired = any(
            getattr(event, "event_type", None) in self.CONTROL_REACQUISITION_EVENT_TYPES
            and getattr(event, "actor_track_id", None) is not None
            for event in current_events
        )
        if reacquired and hasattr(self.shot_tracker, "confirm_control_reacquisition"):
            self.shot_tracker.confirm_control_reacquisition(frame_id)

        # V1.3 may intentionally suppress a provisional event after closing its internal
        # candidate. The frozen base engine would otherwise retain that stale shot actor.
        suppressed_terminal = bool(
            shot_active_before and not shot_active_after and not shot_published
        )
        if suppressed_terminal:
            self._shot_actor_track_id = None
            self._shot_team_id = None
            self._shot_actor_source = "unknown"
            self._shot_actor_location = None

        diagnostics = getattr(frame_state, "diagnostics", {}) or {}
        p2 = diagnostics.get("p2", {}) if isinstance(diagnostics, dict) else {}
        if isinstance(p2, dict):
            snapshot_source, snapshot_confidence, snapshot_age = (
                self._shot_snapshot_for_publication()
            )
            p2["behavior_revision"] = "v6_abcd_batch_20260925"
            p2["supported_flight_hint_before_shot_update"] = supported_flight_before
            p2["post_shot_control_reacquired"] = reacquired
            p2["shot_actor_team_snapshot"] = {
                "source": snapshot_source,
                "confidence": round(float(snapshot_confidence), 6),
                "age_frames": snapshot_age,
                "frozen_at_release": bool(shot_active_after or shot_published),
            }

        if shot_published or suppressed_terminal:
            self._clear_active_shot_snapshot()
        return result


__all__ = ["P2EngineV6Batch"]
