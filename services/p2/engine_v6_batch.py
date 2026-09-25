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
    * post-shot possession reacquisition explicitly rearms the V1.3 shot tracker;
    * a suppressed provisional shot clears its pre-release actor snapshot.
    """

    EVENT_TEAM_ATTRIBUTION_POLICY = (
        "shot_team_accepts_current_or_fresh_exact_actor_team_snapshot;"
        "exact_actor_conflict_abstains;generic_recent_team_never_promoted;"
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
        self._shot_team_snapshot_confidence = 0.0
        self._shot_team_snapshot_age_frames: int | None = None
        self._shot_team_snapshot_source = "unknown"

    def _clear_v6_shot_snapshot(self) -> None:
        self._shot_team_snapshot_confidence = 0.0
        self._shot_team_snapshot_age_frames = None
        self._shot_team_snapshot_source = "unknown"

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
        self._clear_v6_shot_snapshot()
        if actor_track_id is None:
            return actor_track_id, public_team_id, base_source

        exact_team, confidence, age, exact_source = self._exact_actor_team_snapshot(
            actor_track_id,
            frame_id=frame_id,
        )
        if exact_team is not None:
            if public_team_id is not None and str(public_team_id) != str(exact_team):
                self._shot_team_snapshot_confidence = float(confidence)
                self._shot_team_snapshot_age_frames = age
                self._shot_team_snapshot_source = "exact_actor_conflicts_public_team"
                return (
                    int(actor_track_id),
                    None,
                    "pre_release_exact_actor_team_conflict_abstained",
                )
            self._shot_team_snapshot_confidence = float(confidence)
            self._shot_team_snapshot_age_frames = age
            self._shot_team_snapshot_source = exact_source
            return int(actor_track_id), str(exact_team), exact_source

        if public_team_id is not None:
            self._shot_team_snapshot_source = base_source
            return int(actor_track_id), str(public_team_id), base_source
        return int(actor_track_id), None, base_source

    def _attribute_shot(self, event: Any, frame_state: Any) -> Any:
        event = super()._attribute_shot(event, frame_state)
        if event.event_type is not BasketballEventType.SHOT_ATTEMPT:
            return event

        event.attributes["actor_team_snapshot_source"] = self._shot_team_snapshot_source
        event.attributes["actor_team_snapshot_confidence"] = round(
            float(self._shot_team_snapshot_confidence),
            6,
        )
        event.attributes["actor_team_snapshot_age_frames"] = self._shot_team_snapshot_age_frames
        event.attributes["team_attribution_publication_policy"] = self.EVENT_TEAM_ATTRIBUTION_POLICY
        if event.team_id is not None and self._shot_team_snapshot_source.startswith("exact_actor_"):
            event.attributes["team_attribution_publication_source"] = (
                self._shot_team_snapshot_source
            )
            event.attributes["team_attribution_publication_reason"] = (
                "fresh_exact_actor_team_snapshot_supported"
            )
        elif self._shot_team_snapshot_source == "exact_actor_conflicts_public_team":
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
        if shot_active_before and not shot_active_after and not shot_published:
            self._shot_actor_track_id = None
            self._shot_team_id = None
            self._shot_actor_source = "unknown"
            self._shot_actor_location = None
            self._clear_v6_shot_snapshot()

        diagnostics = getattr(frame_state, "diagnostics", {}) or {}
        p2 = diagnostics.get("p2", {}) if isinstance(diagnostics, dict) else {}
        if isinstance(p2, dict):
            p2["behavior_revision"] = "v6_abcd_batch_20260925"
            p2["supported_flight_hint_before_shot_update"] = supported_flight_before
            p2["post_shot_control_reacquired"] = reacquired
            p2["shot_actor_team_snapshot"] = {
                "source": self._shot_team_snapshot_source,
                "confidence": round(float(self._shot_team_snapshot_confidence), 6),
                "age_frames": self._shot_team_snapshot_age_frames,
            }
        return result


__all__ = ["P2EngineV6Batch"]
