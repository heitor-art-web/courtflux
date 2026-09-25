from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v2 import CourtStateEstimatorV2


class CourtStateEstimatorV3(CourtStateEstimatorV2):
    """V6 temporal-boundary policy layered over the long-form V2 estimator.

    V6 shows two opposite errors in the same long clip: short broadcast/view-evidence
    collapses become hard NON_GAMEPLAY during real possessions, while weak live context
    can keep true stoppages alive for several seconds. V3 changes only the hysteresis
    horizons used by the validation pipeline:

    * transient negative evidence must persist for 3.0 s before creating a hard boundary;
    * weak live evidence is allowed to hold continuity for only 1.0 s before it starts the
      hard-boundary debounce.

    Positive strong-wide and basket/ball LIMITED resume semantics are inherited unchanged,
    so free-throw context remains GAMEPLAY_LIMITED rather than being forced to NON_GAMEPLAY.
    """

    POLICY_VERSION = "court_state_gameplay_v3_v6_batch"

    def __init__(
        self,
        *args: Any,
        non_gameplay_enter_sec: float = 3.0,
        weak_live_hold_sec: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            non_gameplay_enter_sec=non_gameplay_enter_sec,
            weak_live_hold_sec=weak_live_hold_sec,
            **kwargs,
        )

    def update(
        self,
        frame_state: Any,
        *,
        gameplay_diagnostics: dict[str, Any] | None = None,
    ):
        state = super().update(
            frame_state,
            gameplay_diagnostics=gameplay_diagnostics,
        )
        policy = state.provenance.get("gameplay_presence")
        if isinstance(policy, dict):
            policy["version"] = self.POLICY_VERSION
            policy["v6_boundary_objective"] = (
                "suppress_sub_3s_false_hard_boundaries_and_expire_weak_live_hold_after_1s"
            )
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_boundary_objective"] = (
            "3s_negative_debounce_plus_1s_weak_live_hold"
        )
        return payload


__all__ = ["CourtStateEstimatorV3"]
