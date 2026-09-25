from __future__ import annotations

from typing import Any

from core.court_state import CourtState
from services.vision.court_state_v1 import CourtStateEstimatorV1


class CourtStateEstimatorV2(CourtStateEstimatorV1):
    """Long-form gameplay-presence policy layered on the conservative V1 view estimator.

    V1 deliberately treated multiple detected players as sufficient to preserve gameplay
    continuity. That was useful on short highlights, but V6 showed that real dead-ball
    periods can still contain many players, court pixels and basket geometry for several
    seconds. V2 keeps the cheap image filter fail-open for detector/tracker inference, then
    combines its diagnostic evidence with post-detector basketball context and temporal
    hysteresis before deciding ``gameplay_present``.

    The policy is intentionally asymmetric:
    * entering NON_GAMEPLAY requires sustained negative evidence;
    * returning to gameplay requires a shorter sustained positive run;
    * a basket + visible ball + several players may preserve GAMEPLAY_LIMITED (for
      free-throw/dead-ball basketball context) without making the frame metric eligible;
    * explicit on-court player motion may preserve GAMEPLAY_LIMITED when the cheap court
      colour heuristic abstains;
    * scene cuts never count as positive evidence by themselves.

    This class does not change detector/tracker execution. It changes only the canonical
    CourtState gameplay/analysis scope consumed by P2 and Gate D.
    """

    def __init__(
        self,
        *args: Any,
        non_gameplay_enter_sec: float = 0.60,
        gameplay_resume_sec: float = 0.20,
        limited_motion_ratio: float = 0.006,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.non_gameplay_enter_sec = max(0.10, float(non_gameplay_enter_sec))
        self.gameplay_resume_sec = max(0.05, float(gameplay_resume_sec))
        self.limited_motion_ratio = max(0.0, float(limited_motion_ratio))
        self._gameplay_presence_latched: bool | None = None
        self._non_gameplay_candidate_frames = 0
        self._gameplay_candidate_frames = 0

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fps(frame_state: Any) -> float:
        try:
            return max(1.0, float(getattr(frame_state, "fps", 30.0) or 30.0))
        except (TypeError, ValueError):
            return 30.0

    def _threshold_frames(self, seconds: float, fps: float) -> int:
        return max(1, int(round(max(0.0, float(seconds)) * max(1.0, float(fps)))))

    def _reset_temporal_presence(self) -> None:
        self._gameplay_presence_latched = None
        self._non_gameplay_candidate_frames = 0
        self._gameplay_candidate_frames = 0

    def _candidate(
        self,
        frame_state: Any,
        state: CourtState,
        gameplay: dict[str, Any],
    ) -> tuple[bool, str, str, dict[str, Any]] | None:
        raw_likely = gameplay.get("likely_gameplay")
        if not isinstance(raw_likely, bool):
            # Legacy/non-canonical callers do not have the cheap-filter diagnostic.
            # Preserve V1 behaviour exactly rather than inventing a new gate.
            return None

        motion_ratio = self._safe_float(gameplay.get("motion_ratio"))
        scene_cut = bool(gameplay.get("scene_cut", False))
        ball_present = getattr(frame_state, "ball", None) is not None
        basket_ball_context = bool(
            state.basket_evidence
            and ball_present
            and state.player_count >= 4
        )
        explicit_on_court_motion = bool(
            state.on_court_player_count >= 2
            and motion_ratio is not None
            and motion_ratio >= self.limited_motion_ratio
        )
        motion_supported_prefilter = bool(
            raw_likely
            and not scene_cut
            and (
                motion_ratio is None
                or motion_ratio >= self.limited_motion_ratio
            )
        )

        diagnostics = {
            "prefilter_likely_gameplay": raw_likely,
            "prefilter_scene_cut": scene_cut,
            "prefilter_motion_ratio": motion_ratio,
            "limited_motion_ratio": self.limited_motion_ratio,
            "ball_present": bool(ball_present),
            "basket_ball_context": basket_ball_context,
            "explicit_on_court_motion": explicit_on_court_motion,
            "motion_supported_prefilter": motion_supported_prefilter,
        }

        # Inter-frame motion across a broadcast cut is not player-motion evidence: the
        # whole image changed. Treat the cut itself as a negative candidate and let the
        # temporal debounce decide whether it becomes a real NON_GAMEPLAY boundary.
        if scene_cut:
            return False, "non_gameplay", "scene_cut_candidate", diagnostics

        if basket_ball_context and not motion_supported_prefilter:
            return True, "gameplay_limited", "basket_ball_context_hold", diagnostics
        if explicit_on_court_motion:
            status = "gameplay_analysis" if state.analysis_eligible and raw_likely else "gameplay_limited"
            reason = (
                "prefilter_and_on_court_motion"
                if raw_likely
                else "on_court_motion_hold"
            )
            return True, status, reason, diagnostics
        if motion_supported_prefilter:
            status = "gameplay_analysis" if state.analysis_eligible else "gameplay_limited"
            return True, status, "motion_supported_prefilter", diagnostics

        return False, "non_gameplay", "sustained_low_live_evidence_candidate", diagnostics

    def _resolve_temporal_presence(
        self,
        *,
        candidate_present: bool,
        candidate_status: str,
        candidate_reason: str,
        fps: float,
    ) -> tuple[bool, str, str, dict[str, Any]]:
        enter_frames = self._threshold_frames(self.non_gameplay_enter_sec, fps)
        resume_frames = self._threshold_frames(self.gameplay_resume_sec, fps)

        if candidate_present:
            self._non_gameplay_candidate_frames = 0
            self._gameplay_candidate_frames += 1
            if self._gameplay_presence_latched is False:
                if self._gameplay_candidate_frames < resume_frames:
                    return (
                        False,
                        "non_gameplay",
                        "gameplay_resume_pending",
                        {
                            "non_gameplay_enter_frames": enter_frames,
                            "gameplay_resume_frames": resume_frames,
                        },
                    )
                self._gameplay_presence_latched = True
            elif self._gameplay_presence_latched is None:
                # High recall on acquisition: positive evidence may start immediately.
                self._gameplay_presence_latched = True
            return (
                True,
                candidate_status,
                candidate_reason,
                {
                    "non_gameplay_enter_frames": enter_frames,
                    "gameplay_resume_frames": resume_frames,
                },
            )

        self._gameplay_candidate_frames = 0
        self._non_gameplay_candidate_frames += 1
        if self._gameplay_presence_latched is False:
            return (
                False,
                "non_gameplay",
                candidate_reason,
                {
                    "non_gameplay_enter_frames": enter_frames,
                    "gameplay_resume_frames": resume_frames,
                },
            )

        if self._non_gameplay_candidate_frames >= enter_frames:
            self._gameplay_presence_latched = False
            return (
                False,
                "non_gameplay",
                "sustained_non_gameplay_confirmed",
                {
                    "non_gameplay_enter_frames": enter_frames,
                    "gameplay_resume_frames": resume_frames,
                },
            )

        # Do not create a hard boundary from a single weak frame or camera cut. During
        # the debounce window keep continuity alive, but downgrade analysis to LIMITED.
        if self._gameplay_presence_latched is None:
            self._gameplay_presence_latched = True
        return (
            True,
            "gameplay_limited",
            "non_gameplay_confirmation_pending",
            {
                "non_gameplay_enter_frames": enter_frames,
                "gameplay_resume_frames": resume_frames,
            },
        )

    def update(
        self,
        frame_state: Any,
        *,
        gameplay_diagnostics: dict[str, Any] | None = None,
    ) -> CourtState:
        gameplay = dict(gameplay_diagnostics or {})
        state = super().update(
            frame_state,
            gameplay_diagnostics=gameplay,
        )
        candidate = self._candidate(frame_state, state, gameplay)
        if candidate is None:
            self._reset_temporal_presence()
            return state

        candidate_present, candidate_status, candidate_reason, evidence = candidate
        present, status, reason, temporal = self._resolve_temporal_presence(
            candidate_present=candidate_present,
            candidate_status=candidate_status,
            candidate_reason=candidate_reason,
            fps=self._fps(frame_state),
        )

        state.gameplay_present = bool(present)
        state.gameplay_status = str(status)
        state.gameplay_reason = str(reason)

        # Metric eligibility requires confirmed GAMEPLAY_ANALYSIS. Limited basketball
        # context and pending/confirmed non-gameplay stay available only to the continuity
        # layers explicitly designed to consume them.
        if status != "gameplay_analysis":
            state.analysis_eligible = False
            if status == "non_gameplay":
                state.eligibility_reason = "non_gameplay_temporal_gate"
            elif state.eligibility_reason == "court_and_entities_consistent":
                state.eligibility_reason = "gameplay_limited_temporal_gate"

        policy = {
            "version": "court_state_gameplay_v2",
            "policy": "prefilter_motion_plus_post_detector_context_with_hysteresis",
            "non_gameplay_enter_sec": self.non_gameplay_enter_sec,
            "gameplay_resume_sec": self.gameplay_resume_sec,
            "non_gameplay_candidate_frames": self._non_gameplay_candidate_frames,
            "gameplay_candidate_frames": self._gameplay_candidate_frames,
            "latched_gameplay_present": self._gameplay_presence_latched,
            "candidate_present": bool(candidate_present),
            "candidate_status": candidate_status,
            "candidate_reason": candidate_reason,
            **evidence,
            **temporal,
        }
        state.provenance["gameplay_presence"] = policy
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload.update(
            {
                "gameplay_presence_version": "court_state_gameplay_v2",
                "non_gameplay_enter_sec": self.non_gameplay_enter_sec,
                "gameplay_resume_sec": self.gameplay_resume_sec,
                "limited_motion_ratio": self.limited_motion_ratio,
                "non_gameplay_policy": "sustained_negative_evidence",
                "limited_policy": "basket_ball_context_or_explicit_on_court_motion",
            }
        )
        return payload
