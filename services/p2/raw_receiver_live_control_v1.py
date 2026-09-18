from __future__ import annotations

from typing import Any

from services.p2.live_control_resolver_v1 import (
    LiveControlP2EngineV1 as _BaseLiveControlP2EngineV1,
)


class LiveControlP2EngineV1(_BaseLiveControlP2EngineV1):
    """Gate-A live resolver with a broader but still causal PASS receiver path.

    The previous resolver proved that SHOT-first classification removes enough false
    shot->pass confusion to satisfy that guardrail, but requiring the possession tracker's
    thresholded candidate left PASS recall effectively unchanged. This layer preserves all
    existing shot/continuity behavior and only broadens receiver evidence while a flight is
    active.

    A raw-best player may stand in for the thresholded candidate only when:
    - it is distinct from the flight origin;
    - the same raw-best track is present on two consecutive frames; and
    - the inherited direct-ball trajectory test shows the ball converging toward it.

    No raw score threshold is introduced. SHOT remains evaluated first by the inherited
    resolver, so raw receiver evidence can never suppress shot evidence.
    """

    LIVE_CONTROL_POLICY = (
        "causal_live_resolver_v3;internal_airborne_unclassified;shot_first;"
        "pass_receiver=thresholded_or_two_frame_raw_best_distinct_origin;"
        "pass_requires_converging_direct_trajectory;recent_owner_continuity;"
        "shot_preflight_above_distance<=8;"
        "gameplay_gap_grace_uses_recent_shot_actor_horizon;events_unchanged"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._raw_receiver_previous_track_id: int | None = None
        self._raw_receiver_previous_frame: int | None = None

    def _reset_live_control(self, reason: str) -> None:
        super()._reset_live_control(reason)
        self._raw_receiver_previous_track_id = None
        self._raw_receiver_previous_frame = None

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    def _raw_receiver_context(
        self,
        *,
        frame_state: Any,
        possession: dict[str, Any],
    ) -> dict[str, Any]:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        flight = dict(possession.get("flight", {}) or {})
        flight_active = bool(flight.get("active", False))
        origin_track_id = self._int_or_none(flight.get("origin_track_id"))
        raw_best_track_id = self._int_or_none(possession.get("raw_best_track_id"))
        thresholded_track_id = self._int_or_none(possession.get("candidate_track_id"))

        distinct_from_origin = bool(
            raw_best_track_id is not None
            and origin_track_id is not None
            and raw_best_track_id != origin_track_id
        )
        consecutive_same_raw = bool(
            raw_best_track_id is not None
            and self._raw_receiver_previous_track_id == raw_best_track_id
            and self._raw_receiver_previous_frame is not None
            and frame_id == self._raw_receiver_previous_frame + 1
        )
        player_visible = self._player_center(frame_state, raw_best_track_id) is not None
        raw_receiver_eligible = bool(
            flight_active
            and thresholded_track_id is None
            and distinct_from_origin
            and consecutive_same_raw
            and player_visible
        )

        context = {
            "raw_best_track_id": raw_best_track_id,
            "thresholded_candidate_track_id": thresholded_track_id,
            "flight_origin_track_id": origin_track_id,
            "raw_best_distinct_from_origin": distinct_from_origin,
            "raw_best_same_track_consecutive": consecutive_same_raw,
            "raw_best_player_visible": player_visible,
            "raw_receiver_eligible": raw_receiver_eligible,
        }

        if flight_active and raw_best_track_id is not None:
            self._raw_receiver_previous_track_id = raw_best_track_id
            self._raw_receiver_previous_frame = frame_id
        elif not flight_active:
            self._raw_receiver_previous_track_id = None
            self._raw_receiver_previous_frame = None

        return context

    def _resolve_eligible_control(
        self,
        *,
        frame_state: Any,
        p2: dict[str, Any],
        semantic_evidence: list[dict[str, Any]],
    ):
        # Work on shallow copies so the event engine's raw tracker diagnostics remain
        # untouched. The inherited resolver will publish the resolved state separately.
        p2_for_resolver = dict(p2)
        possession = dict(p2.get("possession", {}) or {})
        p2_for_resolver["possession"] = possession

        raw_context = self._raw_receiver_context(
            frame_state=frame_state,
            possession=possession,
        )
        raw_receiver_eligible = bool(raw_context["raw_receiver_eligible"])

        if raw_receiver_eligible:
            possession["candidate_track_id"] = raw_context["raw_best_track_id"]
            possession["pass_flight_receiver_evidence"] = True

        state, source, diagnostics = super()._resolve_eligible_control(
            frame_state=frame_state,
            p2=p2_for_resolver,
            semantic_evidence=semantic_evidence,
        )
        diagnostics = dict(diagnostics)
        trajectory = dict(diagnostics.get("receiver_trajectory", {}) or {})
        trajectory["receiver_evidence_source"] = (
            "raw_best_temporal"
            if raw_receiver_eligible
            else "thresholded_candidate"
            if bool(possession.get("pass_flight_receiver_evidence", False))
            else "none"
        )
        diagnostics["receiver_trajectory"] = trajectory
        diagnostics["raw_receiver"] = raw_context

        if source == "receiver_trajectory_pass" and raw_receiver_eligible:
            source = "raw_receiver_trajectory_pass"

        return state, source, diagnostics
