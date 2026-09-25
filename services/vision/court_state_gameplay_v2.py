from __future__ import annotations

import statistics
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
    * confirmed NON_GAMEPLAY resumes quickly only on strong wide live-play evidence;
    * a recent broadcast close-up/cut may use a short, geometry-heavy wide reacquisition
      window so live play can resume before ball/rim evidence has recovered;
    * basket + ball context may resume/hold GAMEPLAY_LIMITED for bounded dead-ball or
      free-throw context without making the frame metric eligible;
    * strong close-ups/cutaways are negative evidence even when frame motion is high;
    * weak "players are moving" evidence cannot keep gameplay alive forever;
    * scene cuts never count as positive evidence by themselves.

    This class does not change detector/tracker execution. It changes only the canonical
    CourtState gameplay/analysis scope consumed by P2 and Gate D.
    """

    def __init__(
        self,
        *args: Any,
        non_gameplay_enter_sec: float = 0.60,
        gameplay_resume_sec: float = 0.20,
        strong_gameplay_resume_sec: float = 0.05,
        limited_motion_ratio: float = 0.006,
        wide_resume_motion_ratio: float = 0.012,
        wide_resume_min_players: int = 6,
        wide_resume_max_height_ratio: float = 0.35,
        strong_closeup_height_ratio: float = 0.55,
        weak_live_hold_sec: float = 6.0,
        wide_reacquire_window_sec: float = 1.50,
        wide_reacquire_min_players: int = 7,
        wide_reacquire_max_height_ratio: float = 0.30,
        wide_reacquire_min_on_court: int = 2,
        wide_reacquire_motion_ratio: float = 0.006,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.non_gameplay_enter_sec = max(0.10, float(non_gameplay_enter_sec))
        self.gameplay_resume_sec = max(0.05, float(gameplay_resume_sec))
        self.strong_gameplay_resume_sec = max(0.01, float(strong_gameplay_resume_sec))
        self.limited_motion_ratio = max(0.0, float(limited_motion_ratio))
        self.wide_resume_motion_ratio = max(
            self.limited_motion_ratio,
            float(wide_resume_motion_ratio),
        )
        self.wide_resume_min_players = max(2, int(wide_resume_min_players))
        self.wide_resume_max_height_ratio = max(
            0.05,
            min(float(wide_resume_max_height_ratio), 1.0),
        )
        self.strong_closeup_height_ratio = max(
            self.wide_resume_max_height_ratio,
            min(float(strong_closeup_height_ratio), 1.0),
        )
        self.weak_live_hold_sec = max(1.0, float(weak_live_hold_sec))
        self.wide_reacquire_window_sec = max(0.10, float(wide_reacquire_window_sec))
        self.wide_reacquire_min_players = max(
            self.wide_resume_min_players,
            int(wide_reacquire_min_players),
        )
        self.wide_reacquire_max_height_ratio = max(
            0.05,
            min(
                float(wide_reacquire_max_height_ratio),
                self.wide_resume_max_height_ratio,
            ),
        )
        self.wide_reacquire_min_on_court = max(1, int(wide_reacquire_min_on_court))
        self.wide_reacquire_motion_ratio = max(
            self.limited_motion_ratio,
            min(float(wide_reacquire_motion_ratio), self.wide_resume_motion_ratio),
        )
        self._gameplay_presence_latched: bool | None = None
        self._non_gameplay_candidate_frames = 0
        self._gameplay_candidate_frames = 0
        self._weak_live_frames = 0
        self._wide_reacquire_frames_remaining = 0

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
        self._weak_live_frames = 0
        self._wide_reacquire_frames_remaining = 0

    @staticmethod
    def _height_ratios_from_frame(frame_state: Any) -> list[float]:
        frame = getattr(frame_state, "frame", None)
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return []
        try:
            frame_h = max(1.0, float(shape[0]))
        except (TypeError, ValueError):
            return []

        ratios: list[float] = []
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            raw = player.get("box", player.get("bbox"))
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                ratio = max(0.0, float(raw[3]) - float(raw[1])) / frame_h
            except (TypeError, ValueError):
                continue
            ratios.append(ratio)
        return ratios

    def _view_evidence(
        self,
        frame_state: Any,
        state: CourtState,
        *,
        motion_ratio: float | None,
        raw_likely: bool,
        ball_present: bool,
        scene_cut: bool,
    ) -> dict[str, Any]:
        ratios = self._height_ratios_from_frame(frame_state)
        median_ratio = statistics.median(ratios) if ratios else None
        max_ratio = max(ratios) if ratios else None
        player_count = int(state.player_count or len(ratios) or 0)

        wide_resume_view = bool(
            player_count >= self.wide_resume_min_players
            and median_ratio is not None
            and median_ratio <= self.wide_resume_max_height_ratio
        )
        wide_reacquire_view = bool(
            player_count >= self.wide_reacquire_min_players
            and median_ratio is not None
            and median_ratio <= self.wide_reacquire_max_height_ratio
        )
        strong_closeup = bool(
            median_ratio is not None
            and (
                median_ratio >= self.strong_closeup_height_ratio
                or (
                    player_count <= 2
                    and max_ratio is not None
                    and max_ratio >= max(0.50, self.strong_closeup_height_ratio - 0.05)
                )
            )
        )
        strong_wide_live = bool(
            not scene_cut
            and wide_resume_view
            and state.on_court_player_count >= 2
            and motion_ratio is not None
            and motion_ratio >= self.wide_resume_motion_ratio
            and (ball_present or raw_likely)
        )
        return {
            "player_height_ratio_median": (
                None if median_ratio is None else round(float(median_ratio), 6)
            ),
            "player_height_ratio_max": (
                None if max_ratio is None else round(float(max_ratio), 6)
            ),
            "wide_resume_view": wide_resume_view,
            "wide_reacquire_view": wide_reacquire_view,
            "strong_closeup": strong_closeup,
            "strong_wide_live": strong_wide_live,
        }

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
        view = self._view_evidence(
            frame_state,
            state,
            motion_ratio=motion_ratio,
            raw_likely=raw_likely,
            ball_present=ball_present,
            scene_cut=scene_cut,
        )
        strong_wide_live = bool(view["strong_wide_live"])
        strong_closeup = bool(view["strong_closeup"])

        transition_window_frames = self._threshold_frames(
            self.wide_reacquire_window_sec,
            self._fps(frame_state),
        )
        if scene_cut or strong_closeup:
            # Arm a short recovery window while the broadcast is visibly away from the
            # normal game camera. The window is consumed only after that view has ended.
            self._wide_reacquire_frames_remaining = transition_window_frames
        elif self._wide_reacquire_frames_remaining > 0:
            self._wide_reacquire_frames_remaining -= 1

        transition_wide_reacquisition = bool(
            not scene_cut
            and not strong_closeup
            and self._wide_reacquire_frames_remaining > 0
            and bool(view["wide_reacquire_view"])
            and state.on_court_player_count >= self.wide_reacquire_min_on_court
            and motion_ratio is not None
            and motion_ratio >= self.wide_reacquire_motion_ratio
        )
        strong_live_evidence = bool(strong_wide_live or transition_wide_reacquisition)

        diagnostics = {
            "prefilter_likely_gameplay": raw_likely,
            "prefilter_scene_cut": scene_cut,
            "prefilter_motion_ratio": motion_ratio,
            "limited_motion_ratio": self.limited_motion_ratio,
            "wide_resume_motion_ratio": self.wide_resume_motion_ratio,
            "wide_reacquire_motion_ratio": self.wide_reacquire_motion_ratio,
            "wide_reacquire_window_frames": transition_window_frames,
            "wide_reacquire_frames_remaining": self._wide_reacquire_frames_remaining,
            "transition_wide_reacquisition": transition_wide_reacquisition,
            "ball_present": bool(ball_present),
            "basket_ball_context": basket_ball_context,
            "explicit_on_court_motion": explicit_on_court_motion,
            "motion_supported_prefilter": motion_supported_prefilter,
            "strong_live_evidence": strong_live_evidence,
            "safe_limited_resume": bool(basket_ball_context and not strong_closeup),
            **view,
        }

        # Inter-frame motion across a broadcast cut is not player-motion evidence: the
        # whole image changed. Treat the cut itself as a negative candidate and let the
        # temporal debounce decide whether it becomes a real NON_GAMEPLAY boundary.
        if scene_cut:
            diagnostics["weak_live_evidence"] = False
            return False, "non_gameplay", "scene_cut_candidate", diagnostics

        # V6 long-form review showed close-ups with very high whole-frame motion. They
        # must not be rescued by motion/on-court heuristics: a tight broadcast shot is a
        # negative continuity candidate until a coherent wide live view returns.
        if strong_closeup:
            diagnostics["weak_live_evidence"] = False
            return False, "non_gameplay", "strong_closeup_candidate", diagnostics

        if transition_wide_reacquisition:
            diagnostics["weak_live_evidence"] = False
            status = "gameplay_analysis" if state.analysis_eligible and raw_likely else "gameplay_limited"
            return True, status, "broadcast_wide_reacquisition", diagnostics

        if strong_wide_live:
            diagnostics["weak_live_evidence"] = False
            status = "gameplay_analysis" if state.analysis_eligible and raw_likely else "gameplay_limited"
            return True, status, "strong_wide_live_evidence", diagnostics

        if basket_ball_context and not motion_supported_prefilter:
            diagnostics["weak_live_evidence"] = True
            return True, "gameplay_limited", "basket_ball_context_hold", diagnostics
        if explicit_on_court_motion:
            diagnostics["weak_live_evidence"] = True
            status = "gameplay_analysis" if state.analysis_eligible and raw_likely else "gameplay_limited"
            reason = (
                "prefilter_and_on_court_motion"
                if raw_likely
                else "on_court_motion_hold"
            )
            return True, status, reason, diagnostics
        if motion_supported_prefilter:
            diagnostics["weak_live_evidence"] = True
            status = "gameplay_analysis" if state.analysis_eligible else "gameplay_limited"
            return True, status, "motion_supported_prefilter", diagnostics

        diagnostics["weak_live_evidence"] = False
        return False, "non_gameplay", "sustained_low_live_evidence_candidate", diagnostics

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
        enter_frames = self._threshold_frames(self.non_gameplay_enter_sec, fps)
        resume_frames = self._threshold_frames(self.gameplay_resume_sec, fps)
        strong_resume_frames = self._threshold_frames(self.strong_gameplay_resume_sec, fps)
        weak_hold_frames = self._threshold_frames(self.weak_live_hold_sec, fps)

        weak_timeout = False
        if candidate_present and weak_live_evidence and not strong_live_evidence:
            self._weak_live_frames += 1
            weak_timeout = self._weak_live_frames >= weak_hold_frames
            if weak_timeout:
                candidate_present = False
                candidate_status = "non_gameplay"
                candidate_reason = "sustained_weak_live_context_candidate"
        elif strong_live_evidence:
            self._weak_live_frames = 0
        elif not candidate_present:
            self._weak_live_frames = 0

        if candidate_present:
            self._non_gameplay_candidate_frames = 0

            if self._gameplay_presence_latched is False:
                # A confirmed stoppage must not reopen just because several players are
                # still walking on court. Strong wide live evidence gets a fast resume;
                # basket+ball context may resume LIMITED on the normal debounce horizon.
                if not strong_live_evidence and not safe_limited_resume:
                    self._gameplay_candidate_frames = 0
                    return (
                        False,
                        "non_gameplay",
                        "gameplay_resume_waiting_strong_live_evidence",
                        {
                            "non_gameplay_enter_frames": enter_frames,
                            "gameplay_resume_frames": resume_frames,
                            "strong_gameplay_resume_frames": strong_resume_frames,
                            "weak_live_hold_frames": weak_hold_frames,
                            "weak_live_frames": self._weak_live_frames,
                        },
                    )

                self._gameplay_candidate_frames += 1
                required = strong_resume_frames if strong_live_evidence else resume_frames
                if self._gameplay_candidate_frames < required:
                    return (
                        False,
                        "non_gameplay",
                        "gameplay_resume_pending",
                        {
                            "non_gameplay_enter_frames": enter_frames,
                            "gameplay_resume_frames": resume_frames,
                            "strong_gameplay_resume_frames": strong_resume_frames,
                            "weak_live_hold_frames": weak_hold_frames,
                            "weak_live_frames": self._weak_live_frames,
                        },
                    )
                self._gameplay_presence_latched = True
            else:
                self._gameplay_candidate_frames += 1
                if self._gameplay_presence_latched is None:
                    # High recall on acquisition: positive evidence may start immediately.
                    self._gameplay_presence_latched = True

            return (
                True,
                candidate_status,
                candidate_reason,
                {
                    "non_gameplay_enter_frames": enter_frames,
                    "gameplay_resume_frames": resume_frames,
                    "strong_gameplay_resume_frames": strong_resume_frames,
                    "weak_live_hold_frames": weak_hold_frames,
                    "weak_live_frames": self._weak_live_frames,
                },
            )

        self._gameplay_candidate_frames = 0
        if not weak_timeout:
            self._weak_live_frames = 0
        self._non_gameplay_candidate_frames += 1
        if self._gameplay_presence_latched is False:
            return (
                False,
                "non_gameplay",
                candidate_reason,
                {
                    "non_gameplay_enter_frames": enter_frames,
                    "gameplay_resume_frames": resume_frames,
                    "strong_gameplay_resume_frames": strong_resume_frames,
                    "weak_live_hold_frames": weak_hold_frames,
                    "weak_live_frames": self._weak_live_frames,
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
                    "strong_gameplay_resume_frames": strong_resume_frames,
                    "weak_live_hold_frames": weak_hold_frames,
                    "weak_live_frames": self._weak_live_frames,
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
                "strong_gameplay_resume_frames": strong_resume_frames,
                "weak_live_hold_frames": weak_hold_frames,
                "weak_live_frames": self._weak_live_frames,
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
            strong_live_evidence=bool(evidence.get("strong_live_evidence", False)),
            safe_limited_resume=bool(evidence.get("safe_limited_resume", False)),
            weak_live_evidence=bool(evidence.get("weak_live_evidence", False)),
        )

        # The transition-specific escape hatch is single-use. Once a coherent wide view
        # has actually reopened gameplay, subsequent frames must satisfy the normal policy
        # rather than inheriting stale broadcast-transition state.
        if present and bool(evidence.get("transition_wide_reacquisition", False)):
            self._wide_reacquire_frames_remaining = 0

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
            "policy": "broadcast_view_plus_bounded_live_context_with_hysteresis",
            "non_gameplay_enter_sec": self.non_gameplay_enter_sec,
            "gameplay_resume_sec": self.gameplay_resume_sec,
            "strong_gameplay_resume_sec": self.strong_gameplay_resume_sec,
            "weak_live_hold_sec": self.weak_live_hold_sec,
            "wide_reacquire_window_sec": self.wide_reacquire_window_sec,
            "wide_reacquire_frames_remaining": self._wide_reacquire_frames_remaining,
            "non_gameplay_candidate_frames": self._non_gameplay_candidate_frames,
            "gameplay_candidate_frames": self._gameplay_candidate_frames,
            "weak_live_frames": self._weak_live_frames,
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
                "strong_gameplay_resume_sec": self.strong_gameplay_resume_sec,
                "limited_motion_ratio": self.limited_motion_ratio,
                "wide_resume_motion_ratio": self.wide_resume_motion_ratio,
                "wide_resume_min_players": self.wide_resume_min_players,
                "wide_resume_max_height_ratio": self.wide_resume_max_height_ratio,
                "strong_closeup_height_ratio": self.strong_closeup_height_ratio,
                "weak_live_hold_sec": self.weak_live_hold_sec,
                "wide_reacquire_window_sec": self.wide_reacquire_window_sec,
                "wide_reacquire_min_players": self.wide_reacquire_min_players,
                "wide_reacquire_max_height_ratio": self.wide_reacquire_max_height_ratio,
                "wide_reacquire_min_on_court": self.wide_reacquire_min_on_court,
                "wide_reacquire_motion_ratio": self.wide_reacquire_motion_ratio,
                "non_gameplay_policy": "sustained_negative_or_bounded_weak_live_evidence",
                "limited_policy": "bounded_basket_ball_context_or_explicit_on_court_motion",
                "resume_policy": "fast_strong_wide_live_or_recent_broadcast_wide_reacquisition_or_debounced_basket_ball_context",
            }
        )
        return payload