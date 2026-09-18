from __future__ import annotations

from typing import Any

from core.basketball_event import BallControlState
from services.p2.airborne_classifier_v1 import AirborneClassifierV1
from services.p2.live_control_resolver_v1 import (
    LiveControlP2EngineV1 as _BaseLiveControlP2EngineV1,
)


class LiveControlP2EngineV1(_BaseLiveControlP2EngineV1):
    """Canonical live resolver with optional trained airborne semantics.

    Event generation remains entirely in the inherited P2 engine. When no airborne model is
    loaded, behavior is identical to the AIRBORNE-v2 resolver. When a model is loaded, only
    the internal airborne classification is replaced by AirborneClassifierV1.
    """

    LIVE_CONTROL_POLICY = (
        "causal_live_resolver_v4;AirborneClassifierV1_when_model_loaded;"
        "strong_shot_signals_precede_classifier;classifier_pass_shot_unresolved;"
        "recent_owner_continuity;events_unchanged"
    )

    HARD_SHOT_SOURCES = {
        "shot_tracker_active",
        "shot_event_frame",
        "semantic_shot_action",
        "frozen_shot_preflight",
        "tracker_shot_flight",
        "latched_shot_flight",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        airborne_model_path = kwargs.pop("airborne_model_path", None)
        super().__init__(*args, **kwargs)
        self.airborne_classifier = AirborneClassifierV1(airborne_model_path)
        self._classifier_latch: BallControlState | None = None

    def _reset_live_control(self, reason: str) -> None:
        super()._reset_live_control(reason)
        self._classifier_latch = None
        classifier = getattr(self, "airborne_classifier", None)
        if classifier is not None:
            classifier.reset()

    @staticmethod
    def _state(value: Any) -> BallControlState:
        try:
            return BallControlState(str(value or "unknown"))
        except ValueError:
            return BallControlState.UNKNOWN

    def _resolve_eligible_control(
        self,
        *,
        frame_state: Any,
        p2: dict[str, Any],
        semantic_evidence: list[dict[str, Any]],
    ):
        classification = self.airborne_classifier.classify(
            frame_state=frame_state,
            p2=p2,
        )

        # Feature-collection/development mode: no trained artifact means zero behavior
        # change from the already validated AIRBORNE-v2 fallback.
        state, source, diagnostics = super()._resolve_eligible_control(
            frame_state=frame_state,
            p2=p2,
            semantic_evidence=semantic_evidence,
        )
        diagnostics = dict(diagnostics)
        diagnostics["airborne_classifier"] = classification

        if not classification["model_loaded"]:
            return state, source, diagnostics

        flight_active = bool(classification.get("flight_active", False))

        # Existing hard/causal shot evidence remains a safety precedence rule. A trained
        # PASS prediction cannot override an active/confirmed semantic shot signal.
        if source in self.HARD_SHOT_SOURCES and state is BallControlState.SHOT_FLIGHT:
            self._classifier_latch = BallControlState.SHOT_FLIGHT
            diagnostics["classifier_decision_source"] = "hard_shot_precedence"
            return state, source, diagnostics

        if not flight_active:
            self._classifier_latch = None
            diagnostics["classifier_decision_source"] = "not_airborne"
            return state, source, diagnostics

        label = str(classification.get("label") or "unresolved")
        if label == "shot_flight":
            self._classifier_latch = BallControlState.SHOT_FLIGHT
            self._live_control_latch = BallControlState.SHOT_FLIGHT
            self._airborne_state = self.AIRBORNE_SHOT
            diagnostics["internal_airborne_state"] = self._airborne_state
            diagnostics["classifier_decision_source"] = "trained_classifier"
            return BallControlState.SHOT_FLIGHT, "airborne_classifier_shot", diagnostics

        if label == "pass_flight":
            self._classifier_latch = BallControlState.PASS_FLIGHT
            self._live_control_latch = BallControlState.PASS_FLIGHT
            self._airborne_state = self.AIRBORNE_PASS
            diagnostics["internal_airborne_state"] = self._airborne_state
            diagnostics["classifier_decision_source"] = "trained_classifier"
            return BallControlState.PASS_FLIGHT, "airborne_classifier_pass", diagnostics

        # A prior trained classification remains stable for the physical flight. This is a
        # temporal latch of a model decision, not a new per-frame guess.
        if self._classifier_latch is BallControlState.SHOT_FLIGHT:
            self._airborne_state = self.AIRBORNE_SHOT
            diagnostics["internal_airborne_state"] = self._airborne_state
            diagnostics["classifier_decision_source"] = "trained_classifier_latch"
            return BallControlState.SHOT_FLIGHT, "airborne_classifier_shot_latched", diagnostics
        if self._classifier_latch is BallControlState.PASS_FLIGHT:
            self._airborne_state = self.AIRBORNE_PASS
            diagnostics["internal_airborne_state"] = self._airborne_state
            diagnostics["classifier_decision_source"] = "trained_classifier_latch"
            return BallControlState.PASS_FLIGHT, "airborne_classifier_pass_latched", diagnostics

        # Model explicitly abstained. Do not let the inherited receiver heuristic silently
        # turn that abstention into PASS. Preserve recent-owner continuity where available;
        # otherwise expose the raw non-PASS tracker state.
        self._live_control_latch = None
        self._airborne_state = self.AIRBORNE_UNCLASSIFIED
        diagnostics["internal_airborne_state"] = self._airborne_state
        diagnostics["classifier_decision_source"] = "trained_classifier_unresolved"
        if bool(diagnostics.get("recent_owner", False)):
            return (
                BallControlState.POSSESSED,
                "airborne_classifier_unresolved_owner_continuity",
                diagnostics,
            )

        tracker_state = self._state(diagnostics.get("tracker_control_state"))
        if tracker_state is BallControlState.PASS_FLIGHT:
            tracker_state = BallControlState.UNKNOWN
        return tracker_state, "airborne_classifier_unresolved", diagnostics

    def _apply_live_override(
        self,
        *,
        frame_state: Any,
        state: BallControlState,
        source: str,
        resolver_diag: dict[str, Any] | None = None,
    ) -> None:
        super()._apply_live_override(
            frame_state=frame_state,
            state=state,
            source=source,
            resolver_diag=resolver_diag,
        )
        diagnostics = getattr(frame_state, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            return
        p2 = diagnostics.get("p2")
        if not isinstance(p2, dict):
            return
        airborne = dict((resolver_diag or {}).get("airborne_classifier", {}) or {})
        p2["airborne_classifier"] = airborne
        p2["behavior_revision"] = "control_state_v7_trainable_airborne_classifier"
