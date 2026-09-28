from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v4 import CourtStateEstimatorV4


class CourtStateEstimatorV5(CourtStateEstimatorV4):
    """V6 stoppage-density policy with bounded inferred holds.

    V4 proved that causal static-formation evidence can recover dead-ball footage, but the
    long-form V6 run exposed two opposing failure modes:

    * a long static stoppage may contain intermittent ball/player evidence and reopen too
      easily before enough of the stoppage is rejected;
    * once a stoppage is confirmed, the generic temporal latch can bleed into the next
      possession when ball reacquisition is delayed.

    V5 therefore separates *confirmation* from *hold duration*. Static formations confirm
    after 2.2 s and receive a bounded four-second NON_GAMEPLAY hold. A second stalled-wide
    signal confirms after one second when the preceding two-second context has low analysis
    eligibility, weak court-surface support and multiple on-court players; it receives a
    bounded 2.5-second hold. Once a bounded hold expires, the same evidence source is
    disarmed until it clears, so one long visual pattern cannot repeatedly re-close gameplay.

    These paths are causal and timestamp-agnostic. Existing V3/V4 evidence remains intact.
    """

    POLICY_VERSION = "court_state_gameplay_v5_v6_stoppage_density"

    # V6 replay separation: true gameplay static streaks stayed <= 2.08 s, while the
    # problematic NG2/NG4 formations persisted materially longer.
    STATIC_FORMATION_CONFIRM_SEC = 2.20
    STATIC_HOLD_SEC = 4.00

    # V4's stalled-context gate was too strict because ball detections persisted during
    # NG3. Ball presence is supporting evidence only; it cannot by itself prove live play.
    STALLED_CONTEXT_BALL_RATE_MAX = 0.65
    STALLED_WIDE_CONFIRM_SEC = 1.00
    STALLED_WIDE_HOLD_SEC = 2.50

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._v5_stalled_ready_frames = 0
        self._v5_hold_source: str | None = None
        self._v5_hold_frames = 0
        self._v5_hold_limit_frames = 0
        self._v5_static_source_blocked = False
        self._v5_stalled_source_blocked = False
        self._v5_last_diagnostics: dict[str, Any] = {}

    def _reset_temporal_presence(self) -> None:
        super()._reset_temporal_presence()
        self._v5_stalled_ready_frames = 0
        self._v5_hold_source = None
        self._v5_hold_frames = 0
        self._v5_hold_limit_frames = 0
        self._v5_static_source_blocked = False
        self._v5_stalled_source_blocked = False
        self._v5_last_diagnostics = {}

    def _candidate(
        self,
        frame_state: Any,
        state: Any,
        gameplay: dict[str, Any],
    ) -> tuple[bool, str, str, dict[str, Any]] | None:
        candidate = super()._candidate(frame_state, state, gameplay)
        if candidate is None:
            return None

        candidate_present, candidate_status, candidate_reason, evidence = candidate
        evidence = dict(evidence)
        fps = self._fps(frame_state)

        static_confirmed = bool(evidence.get("v4_static_formation_confirmed", False))
        stalled_ready = bool(evidence.get("v4_stalled_context_ready", False))
        hard_break = bool(
            evidence.get("strong_closeup", False)
            or evidence.get("prefilter_scene_cut", False)
        )

        if not static_confirmed:
            self._v5_static_source_blocked = False

        if stalled_ready and not hard_break:
            self._v5_stalled_ready_frames += 1
        elif not stalled_ready:
            self._v5_stalled_ready_frames = 0
            self._v5_stalled_source_blocked = False

        stalled_confirm_frames = self._threshold_frames(
            self.STALLED_WIDE_CONFIRM_SEC,
            fps,
        )
        stalled_confirmed = bool(
            self._v5_stalled_ready_frames >= stalled_confirm_frames
        )

        # After a bounded hold expires, do not allow the same still-active source to
        # immediately re-close gameplay. For static formations, also suppress V3's
        # low-action direct path while the same static evidence remains confirmed.
        if self._v5_static_source_blocked and static_confirmed:
            if candidate_reason in {
                "v4_static_formation_candidate",
                "v3_sustained_low_action_candidate",
            }:
                candidate_present = True
                candidate_status = (
                    "gameplay_analysis"
                    if bool(getattr(state, "analysis_eligible", False))
                    else "gameplay_limited"
                )
                candidate_reason = "v5_static_hold_expired_fail_open_guard"
                evidence["strong_live_evidence"] = False
                evidence["safe_limited_resume"] = False
                evidence["weak_live_evidence"] = False

        if (
            self._v5_stalled_source_blocked
            and stalled_ready
            and candidate_reason == "v4_stalled_context_broadcast_break_candidate"
        ):
            candidate_present = True
            candidate_status = "gameplay_limited"
            candidate_reason = "v5_stalled_hold_expired_fail_open_guard"
            evidence["strong_live_evidence"] = False
            evidence["safe_limited_resume"] = False
            evidence["weak_live_evidence"] = False

        direct_source: str | None = None
        if static_confirmed and not self._v5_static_source_blocked:
            direct_source = "static_formation"
        elif stalled_confirmed and not self._v5_stalled_source_blocked:
            direct_source = "stalled_wide"
        elif (
            candidate_reason == "v4_stalled_context_broadcast_break_candidate"
            and not self._v5_stalled_source_blocked
        ):
            direct_source = "stalled_wide"

        if direct_source is not None:
            candidate_present = False
            candidate_status = "non_gameplay"
            candidate_reason = f"v5_{direct_source}_candidate"
            evidence["strong_live_evidence"] = False
            evidence["safe_limited_resume"] = False
            evidence["weak_live_evidence"] = False

        diagnostics = {
            "v5_static_confirm_sec": self.STATIC_FORMATION_CONFIRM_SEC,
            "v5_static_hold_sec": self.STATIC_HOLD_SEC,
            "v5_static_source_blocked": bool(self._v5_static_source_blocked),
            "v5_stalled_ball_rate_max": self.STALLED_CONTEXT_BALL_RATE_MAX,
            "v5_stalled_ready_frames": int(self._v5_stalled_ready_frames),
            "v5_stalled_confirm_frames": int(stalled_confirm_frames),
            "v5_stalled_confirmed": bool(stalled_confirmed),
            "v5_stalled_hold_sec": self.STALLED_WIDE_HOLD_SEC,
            "v5_stalled_source_blocked": bool(self._v5_stalled_source_blocked),
            "v5_direct_source_candidate": direct_source,
            "v5_hold_source": self._v5_hold_source,
            "v5_hold_frames": int(self._v5_hold_frames),
            "v5_hold_limit_frames": int(self._v5_hold_limit_frames),
        }
        evidence.update(diagnostics)
        self._v5_last_diagnostics = diagnostics
        return candidate_present, candidate_status, candidate_reason, evidence

    def _start_hold(self, source: str, fps: float) -> None:
        self._v5_hold_source = source
        self._v5_hold_frames = 0
        hold_sec = (
            self.STATIC_HOLD_SEC
            if source == "static_formation"
            else self.STALLED_WIDE_HOLD_SEC
        )
        self._v5_hold_limit_frames = self._threshold_frames(hold_sec, fps)

    def _expire_hold_fail_open(
        self,
        *,
        source: str,
        fps: float,
    ) -> tuple[bool, str, str, dict[str, Any]]:
        if source == "static_formation":
            self._v5_static_source_blocked = True
        elif source == "stalled_wide":
            self._v5_stalled_source_blocked = True

        self._v5_hold_source = None
        self._v5_hold_frames = 0
        self._v5_hold_limit_frames = 0

        # A bounded inferred stoppage is intentionally fail-open after its evidence budget.
        # This prevents dead-ball evidence from bleeding indefinitely into a restarted
        # possession when direct ball reacquisition is delayed.
        self._gameplay_presence_latched = True
        self._non_gameplay_candidate_frames = 0
        self._gameplay_candidate_frames = 0
        return (
            True,
            "gameplay_limited",
            f"v5_{source}_bounded_hold_expired_fail_open",
            {
                "v5_bounded_hold_expired": True,
                "v5_bounded_hold_source": source,
                "v5_static_source_blocked": bool(self._v5_static_source_blocked),
                "v5_stalled_source_blocked": bool(self._v5_stalled_source_blocked),
                "non_gameplay_enter_frames": self._threshold_frames(
                    self.non_gameplay_enter_sec, fps
                ),
                "gameplay_resume_frames": self._threshold_frames(
                    self.gameplay_resume_sec, fps
                ),
                "strong_gameplay_resume_frames": self._threshold_frames(
                    self.strong_gameplay_resume_sec, fps
                ),
                "weak_live_hold_frames": self._threshold_frames(
                    self.weak_live_hold_sec, fps
                ),
            },
        )

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
        requested_source: str | None = None
        if candidate_reason == "v5_static_formation_candidate":
            requested_source = "static_formation"
        elif candidate_reason == "v5_stalled_wide_candidate":
            requested_source = "stalled_wide"

        if self._v5_hold_source is None and requested_source is not None:
            self._start_hold(requested_source, fps)

        if self._v5_hold_source is not None:
            source = self._v5_hold_source
            self._v5_hold_frames += 1
            if self._v5_hold_frames >= self._v5_hold_limit_frames:
                return self._expire_hold_fail_open(source=source, fps=fps)

            self._gameplay_presence_latched = False
            self._non_gameplay_candidate_frames = 0
            self._gameplay_candidate_frames = 0
            diagnostics = {
                "non_gameplay_enter_frames": self._threshold_frames(
                    self.non_gameplay_enter_sec, fps
                ),
                "gameplay_resume_frames": self._threshold_frames(
                    self.gameplay_resume_sec, fps
                ),
                "strong_gameplay_resume_frames": self._threshold_frames(
                    self.strong_gameplay_resume_sec, fps
                ),
                "weak_live_hold_frames": self._threshold_frames(
                    self.weak_live_hold_sec, fps
                ),
                "v5_bounded_hold_active": True,
                "v5_bounded_hold_source": source,
                "v5_hold_frames": int(self._v5_hold_frames),
                "v5_hold_limit_frames": int(self._v5_hold_limit_frames),
            }
            diagnostics.update(self._v5_last_diagnostics)
            return (
                False,
                "non_gameplay",
                f"v5_{source}_bounded_hold_active",
                diagnostics,
            )

        result = super()._resolve_temporal_presence(
            candidate_present=candidate_present,
            candidate_status=candidate_status,
            candidate_reason=candidate_reason,
            fps=fps,
            strong_live_evidence=strong_live_evidence,
            safe_limited_resume=safe_limited_resume,
            weak_live_evidence=weak_live_evidence,
        )
        result[3].update(self._v5_last_diagnostics)
        return result

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
                "bounded_static_and_stalled_stoppage_holds_without_cross_possession_bleed"
            )
            policy.update(self._v5_last_diagnostics)
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_boundary_objective"] = (
            "bounded_static_and_stalled_stoppage_holds_without_cross_possession_bleed"
        )
        payload["v5_bounded_stoppage_policy"] = {
            "static_confirm_sec": self.STATIC_FORMATION_CONFIRM_SEC,
            "static_hold_sec": self.STATIC_HOLD_SEC,
            "stalled_ball_rate_max": self.STALLED_CONTEXT_BALL_RATE_MAX,
            "stalled_confirm_sec": self.STALLED_WIDE_CONFIRM_SEC,
            "stalled_hold_sec": self.STALLED_WIDE_HOLD_SEC,
            "same_source_retrigger_blocked_until_clear": True,
        }
        return payload


__all__ = ["CourtStateEstimatorV5"]
