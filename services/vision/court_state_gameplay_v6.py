from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v5 import CourtStateEstimatorV5


class CourtStateEstimatorV6(CourtStateEstimatorV5):
    """V6 refinement: longer stoppage budget with sustained-live early release.

    V5 materially improved V6 non-game rejection, but one bounded static hold bled for
    almost its full four-second budget into real P4 gameplay. V6 keeps the aggressive
    confirmation policy, extends the *maximum* inferred hold budget for true stoppages,
    and adds an early-release path when strong wide live-play evidence is sustained.

    The release signal reuses V2's existing ``strong_live_evidence`` contract:
    wide view, at least two explicit on-court players, sufficient motion, and either
    direct ball presence or a live prefilter. A short persistence requirement prevents
    one-frame motion spikes inside dead-ball footage from reopening gameplay.

    No shot, team-attribution, possession, or identity logic is changed here.
    """

    POLICY_VERSION = "court_state_gameplay_v6_v6_hold_release"

    # The V5 run reached 46.84% non-game rejection. Extending the maximum budgets by
    # one half-second / one second on already-confirmed stoppages is enough to target
    # >50% on V6 without lowering any confirmation threshold.
    STATIC_HOLD_SEC = 5.00
    STALLED_WIDE_HOLD_SEC = 3.00

    HOLD_EARLY_RELEASE_STRONG_LIVE_SEC = 0.30

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._v6_hold_strong_live_frames = 0
        self._v6_early_release_count = 0
        super().__init__(*args, **kwargs)

    def _reset_temporal_presence(self) -> None:
        super()._reset_temporal_presence()
        self._v6_hold_strong_live_frames = 0

    def _release_hold_early(
        self,
        *,
        source: str,
        candidate_present: bool,
        candidate_status: str,
        fps: float,
    ) -> tuple[bool, str, str, dict[str, Any]]:
        # Disarm the source until its underlying evidence clears, matching V5's bounded
        # expiry semantics so a still-active static pattern cannot immediately re-close.
        if source == "static_formation":
            self._v5_static_source_blocked = True
        elif source == "stalled_wide":
            self._v5_stalled_source_blocked = True

        self._v5_hold_source = None
        self._v5_hold_frames = 0
        self._v5_hold_limit_frames = 0
        self._v6_hold_strong_live_frames = 0
        self._v6_early_release_count += 1

        self._gameplay_presence_latched = True
        self._non_gameplay_candidate_frames = 0
        self._gameplay_candidate_frames = 0

        status = (
            candidate_status
            if candidate_present and candidate_status in {"gameplay_analysis", "gameplay_limited"}
            else "gameplay_limited"
        )
        return (
            True,
            status,
            f"v6_{source}_hold_early_release_strong_live",
            {
                "v6_hold_early_release": True,
                "v6_hold_early_release_source": source,
                "v6_hold_early_release_count": int(self._v6_early_release_count),
                "v6_hold_strong_live_frames": 0,
                "v6_hold_strong_live_required_frames": self._threshold_frames(
                    self.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC,
                    fps,
                ),
                "v5_static_source_blocked": bool(self._v5_static_source_blocked),
                "v5_stalled_source_blocked": bool(self._v5_stalled_source_blocked),
                "non_gameplay_enter_frames": self._threshold_frames(
                    self.non_gameplay_enter_sec,
                    fps,
                ),
                "gameplay_resume_frames": self._threshold_frames(
                    self.gameplay_resume_sec,
                    fps,
                ),
                "strong_gameplay_resume_frames": self._threshold_frames(
                    self.strong_gameplay_resume_sec,
                    fps,
                ),
                "weak_live_hold_frames": self._threshold_frames(
                    self.weak_live_hold_sec,
                    fps,
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
        active_source = self._v5_hold_source
        required_live_frames = self._threshold_frames(
            self.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC,
            fps,
        )

        if active_source is not None:
            if candidate_present and strong_live_evidence:
                self._v6_hold_strong_live_frames += 1
            else:
                self._v6_hold_strong_live_frames = 0

            if self._v6_hold_strong_live_frames >= required_live_frames:
                return self._release_hold_early(
                    source=active_source,
                    candidate_present=candidate_present,
                    candidate_status=candidate_status,
                    fps=fps,
                )
        else:
            self._v6_hold_strong_live_frames = 0

        result = super()._resolve_temporal_presence(
            candidate_present=candidate_present,
            candidate_status=candidate_status,
            candidate_reason=candidate_reason,
            fps=fps,
            strong_live_evidence=strong_live_evidence,
            safe_limited_resume=safe_limited_resume,
            weak_live_evidence=weak_live_evidence,
        )
        result[3].update(
            {
                "v6_hold_strong_live_frames": int(self._v6_hold_strong_live_frames),
                "v6_hold_strong_live_required_frames": int(required_live_frames),
                "v6_hold_early_release_count": int(self._v6_early_release_count),
                "v6_static_hold_sec": self.STATIC_HOLD_SEC,
                "v6_stalled_hold_sec": self.STALLED_WIDE_HOLD_SEC,
            }
        )
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
                "longer_confirmed_stoppage_budget_with_sustained_strong_live_early_release"
            )
            policy["v6_static_hold_sec"] = self.STATIC_HOLD_SEC
            policy["v6_stalled_hold_sec"] = self.STALLED_WIDE_HOLD_SEC
            policy["v6_hold_early_release_strong_live_sec"] = (
                self.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC
            )
            policy["v6_hold_early_release_count"] = int(self._v6_early_release_count)
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_boundary_objective"] = (
            "longer_confirmed_stoppage_budget_with_sustained_strong_live_early_release"
        )
        payload["v6_hold_release_policy"] = {
            "static_hold_sec": self.STATIC_HOLD_SEC,
            "stalled_hold_sec": self.STALLED_WIDE_HOLD_SEC,
            "strong_live_release_sec": self.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC,
            "same_source_retrigger_blocked_until_clear": True,
        }
        return payload


__all__ = ["CourtStateEstimatorV6"]
