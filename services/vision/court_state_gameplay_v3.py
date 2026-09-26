from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v2 import CourtStateEstimatorV2


class CourtStateEstimatorV3(CourtStateEstimatorV2):
    """V6 temporal-boundary policy layered over the long-form V2 estimator.

    The first V6 batch showed two distinct temporal failure modes:

    * short hard negatives (broadcast cuts/close-ups) still became false NON_GAMEPLAY
      boundaries inside live possessions;
    * prolonged weak-live evidence never became NON_GAMEPLAY because V2's weak timeout
      emitted only a one-frame negative pulse, resetting before the hard-boundary debounce
      could accumulate.

    V3 therefore separates those cases instead of solving both with one threshold:

    * hard negative evidence must persist for 3.5 s before creating a boundary;
    * non-basket weak-live evidence receives a 1.0 s grace period, then stays negative
      until strong live or safe basket+ball context returns, allowing the normal debounce
      to accumulate;
    * safe basket+ball LIMITED context does not consume the weak-live budget.

    Positive strong-wide and basket/ball resume semantics remain inherited from V2.
    """

    POLICY_VERSION = "court_state_gameplay_v3_v6_batch_2"
    BOUNDED_WEAK_REASONS = {
        "on_court_motion_hold",
        "prefilter_and_on_court_motion",
        "motion_supported_prefilter",
    }

    def __init__(
        self,
        *args: Any,
        non_gameplay_enter_sec: float = 3.5,
        weak_live_hold_sec: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            non_gameplay_enter_sec=non_gameplay_enter_sec,
            weak_live_hold_sec=weak_live_hold_sec,
            **kwargs,
        )
        self._v3_weak_context_frames = 0
        self._v3_weak_timeout_latched = False

    def _reset_temporal_presence(self) -> None:
        super()._reset_temporal_presence()
        self._v3_weak_context_frames = 0
        self._v3_weak_timeout_latched = False

    def _resolve_temporal_presence(
        self,
        *,
        candidate_present: bool,
        candidate_status: str,
        candidate_reason: str,
        fps: float,
        strong_live_evidence: bool = True,
        safe_limited_resume: bool = False,
        weak_live_evidence: bool = False,
    ) -> tuple[bool, str, str, dict[str, Any]]:
        """Make weak-live expiry persistent instead of a one-frame pulse.

        V2 increments ``_weak_live_frames`` while weak evidence is positive. On the exact
        timeout frame it flips the candidate negative, but the following frame starts the
        weak counter again from zero. With a multi-second NON_GAMEPLAY debounce, a true
        stoppage can therefore remain GAMEPLAY_LIMITED forever.

        V3 owns that weak budget explicitly. Once bounded weak context expires, the
        candidate remains negative until strong wide live evidence or safe basket+ball
        context returns. Hard negatives keep using the normal long debounce.
        """

        bounded_weak = bool(
            candidate_present
            and weak_live_evidence
            and not strong_live_evidence
            and not safe_limited_resume
            and str(candidate_reason) in self.BOUNDED_WEAK_REASONS
        )

        if strong_live_evidence or safe_limited_resume:
            self._v3_weak_context_frames = 0
            self._v3_weak_timeout_latched = False
        elif bounded_weak:
            self._v3_weak_context_frames += 1
            weak_hold_frames = self._threshold_frames(self.weak_live_hold_sec, fps)
            if self._v3_weak_context_frames >= weak_hold_frames:
                self._v3_weak_timeout_latched = True
        elif candidate_present:
            self._v3_weak_context_frames = 0
            self._v3_weak_timeout_latched = False

        if (
            self._v3_weak_timeout_latched
            and not strong_live_evidence
            and not safe_limited_resume
        ):
            candidate_present = False
            candidate_status = "non_gameplay"
            candidate_reason = "v3_persistent_weak_live_timeout"

        # Disable V2's one-frame weak timeout. V3 has already converted expired weak
        # evidence into a persistent negative candidate above.
        present, status, reason, diagnostics = super()._resolve_temporal_presence(
            candidate_present=candidate_present,
            candidate_status=candidate_status,
            candidate_reason=candidate_reason,
            fps=fps,
            strong_live_evidence=strong_live_evidence,
            safe_limited_resume=safe_limited_resume,
            weak_live_evidence=False,
        )
        diagnostics["v3_weak_context_frames"] = int(self._v3_weak_context_frames)
        diagnostics["v3_weak_timeout_latched"] = bool(self._v3_weak_timeout_latched)
        diagnostics["v3_bounded_weak_reason"] = bool(bounded_weak)
        return present, status, reason, diagnostics

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
                "3.5s_hard_negative_debounce_plus_persistent_bounded_weak_timeout"
            )
            policy["v3_weak_context_frames"] = int(self._v3_weak_context_frames)
            policy["v3_weak_timeout_latched"] = bool(self._v3_weak_timeout_latched)
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_boundary_objective"] = (
            "3.5s_hard_negative_debounce_plus_persistent_bounded_weak_timeout"
        )
        payload["bounded_weak_reasons"] = sorted(self.BOUNDED_WEAK_REASONS)
        return payload


__all__ = ["CourtStateEstimatorV3"]
