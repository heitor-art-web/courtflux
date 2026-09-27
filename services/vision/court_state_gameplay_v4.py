from __future__ import annotations

import statistics
from collections import deque
from typing import Any

from services.vision.court_state_gameplay_v3 import CourtStateEstimatorV3


class CourtStateEstimatorV4(CourtStateEstimatorV3):
    """V6 NG2/NG3 stoppage policy layered over the validated V3 batch.

    V3 materially improved dead-ball rejection, but two V6 patterns remained open:

    * NG2 can stay visually wide with many plausible on-court players and an intermittently
      detected ball even though the cohort is nearly static for several seconds;
    * NG3 can end in a broadcast close-up after several seconds of already-stalled court
      context. Treating that close-up like an isolated in-play camera cut makes the generic
      hard-negative debounce too slow.

    V4 keeps V3's existing low-action and weak-live policies unchanged and adds two
    conservative causal paths:

    * ``static_formation`` ignores ball availability but requires a wide multi-player view
      with camera-compensated relative motion <= 0.007 player-heights/frame for 4 seconds;
    * ``stalled_context_broadcast_break`` accelerates a close-up/cut only when the preceding
      2 seconds already have low analysis eligibility, low direct-ball presence, multiple
      on-court players, and weak local court-surface support.

    Generic hard negatives use a 4.5 second debounce. This deliberately protects the short
    live-play close-ups seen in P3/P5/P6 while the two V4 paths target the longer stoppages.
    """

    POLICY_VERSION = "court_state_gameplay_v4_v6_ng23"

    STATIC_FORMATION_MOTION_MAX = 0.0070
    STATIC_FORMATION_CONFIRM_SEC = 4.0
    STATIC_FORMATION_MIN_PLAYERS = 6
    STATIC_FORMATION_MIN_ON_COURT = 3
    STATIC_FORMATION_MAX_HEIGHT_RATIO = 0.35

    STALLED_CONTEXT_WINDOW_SEC = 2.0
    STALLED_CONTEXT_ANALYSIS_RATE_MAX = 0.15
    STALLED_CONTEXT_BALL_RATE_MAX = 0.35
    STALLED_CONTEXT_MIN_ON_COURT_MEAN = 3.0
    STALLED_CONTEXT_SURFACE_MAX = 0.20
    STALLED_CONTEXT_MIN_SURFACE_VALID_RATIO = 0.50

    DIRECT_CONFIRM_REASONS = {
        "v4_static_formation_candidate",
        "v4_stalled_context_broadcast_break_candidate",
    }

    def __init__(
        self,
        *args: Any,
        non_gameplay_enter_sec: float = 4.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            non_gameplay_enter_sec=non_gameplay_enter_sec,
            **kwargs,
        )
        self._v4_static_formation_frames = 0
        self._v4_stalled_context_history: deque[dict[str, Any]] = deque()
        self._v4_last_diagnostics: dict[str, Any] = {}

    def _reset_temporal_presence(self) -> None:
        super()._reset_temporal_presence()
        self._v4_static_formation_frames = 0
        self._v4_stalled_context_history.clear()
        self._v4_last_diagnostics = {}

    @staticmethod
    def _surface_ratio_median(frame_state: Any) -> float | None:
        values: list[float] = []
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            raw = player.get("local_court_surface_ratio")
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        return None if not values else float(statistics.median(values))

    def _update_static_formation(
        self,
        *,
        state: Any,
        evidence: dict[str, Any],
        fps: float,
    ) -> dict[str, Any]:
        motion = evidence.get("v3_action_window_motion_mean")
        valid_ratio = evidence.get("v3_action_window_valid_motion_ratio")
        height_ratio = evidence.get("player_height_ratio_median")
        try:
            player_count = int(getattr(state, "player_count", 0) or 0)
        except (TypeError, ValueError):
            player_count = 0
        try:
            on_court_count = int(getattr(state, "on_court_player_count", 0) or 0)
        except (TypeError, ValueError):
            on_court_count = 0

        signal = False
        try:
            signal = bool(
                motion is not None
                and valid_ratio is not None
                and float(valid_ratio) >= self.LOW_ACTION_MIN_VALID_MOTION_RATIO
                and float(motion) <= self.STATIC_FORMATION_MOTION_MAX
                and player_count >= self.STATIC_FORMATION_MIN_PLAYERS
                and on_court_count >= self.STATIC_FORMATION_MIN_ON_COURT
                and height_ratio is not None
                and float(height_ratio) <= self.STATIC_FORMATION_MAX_HEIGHT_RATIO
                and not bool(evidence.get("strong_closeup", False))
                and not bool(evidence.get("prefilter_scene_cut", False))
            )
        except (TypeError, ValueError):
            signal = False

        if signal:
            self._v4_static_formation_frames += 1
        else:
            self._v4_static_formation_frames = 0

        confirm_frames = self._threshold_frames(
            self.STATIC_FORMATION_CONFIRM_SEC,
            fps,
        )
        return {
            "v4_static_formation_signal": bool(signal),
            "v4_static_formation_frames": int(self._v4_static_formation_frames),
            "v4_static_formation_confirm_frames": int(confirm_frames),
            "v4_static_formation_confirmed": bool(
                self._v4_static_formation_frames >= confirm_frames
            ),
        }

    def _stalled_context_summary(self, *, fps: float) -> dict[str, Any]:
        window_frames = self._threshold_frames(self.STALLED_CONTEXT_WINDOW_SEC, fps)
        history = list(self._v4_stalled_context_history)[-window_frames:]
        ready = len(history) >= window_frames
        analysis_rate = (
            sum(1 for row in history if row["analysis_eligible"]) / len(history)
            if history
            else 0.0
        )
        ball_rate = (
            sum(1 for row in history if row["ball_present"]) / len(history)
            if history
            else 0.0
        )
        on_court_mean = (
            sum(float(row["on_court_count"]) for row in history) / len(history)
            if history
            else 0.0
        )
        surfaces = [
            float(row["surface_ratio_median"])
            for row in history
            if row["surface_ratio_median"] is not None
        ]
        surface_valid_ratio = len(surfaces) / len(history) if history else 0.0
        surface_mean = sum(surfaces) / len(surfaces) if surfaces else None
        stalled = bool(
            ready
            and analysis_rate <= self.STALLED_CONTEXT_ANALYSIS_RATE_MAX
            and ball_rate <= self.STALLED_CONTEXT_BALL_RATE_MAX
            and on_court_mean >= self.STALLED_CONTEXT_MIN_ON_COURT_MEAN
            and surface_valid_ratio >= self.STALLED_CONTEXT_MIN_SURFACE_VALID_RATIO
            and surface_mean is not None
            and surface_mean <= self.STALLED_CONTEXT_SURFACE_MAX
        )
        return {
            "v4_stalled_context_window_frames": int(window_frames),
            "v4_stalled_context_history_frames": int(len(history)),
            "v4_stalled_context_analysis_rate": round(float(analysis_rate), 6),
            "v4_stalled_context_ball_rate": round(float(ball_rate), 6),
            "v4_stalled_context_on_court_mean": round(float(on_court_mean), 6),
            "v4_stalled_context_surface_valid_ratio": round(
                float(surface_valid_ratio), 6
            ),
            "v4_stalled_context_surface_mean": (
                None if surface_mean is None else round(float(surface_mean), 6)
            ),
            "v4_stalled_context_ready": bool(stalled),
        }

    def _append_stalled_context_sample(
        self,
        frame_state: Any,
        *,
        state: Any,
        evidence: dict[str, Any],
        fps: float,
    ) -> None:
        hard_break = bool(
            evidence.get("strong_closeup", False)
            or evidence.get("prefilter_scene_cut", False)
        )
        if hard_break:
            return

        try:
            on_court_count = int(getattr(state, "on_court_player_count", 0) or 0)
        except (TypeError, ValueError):
            on_court_count = 0
        self._v4_stalled_context_history.append(
            {
                "analysis_eligible": bool(getattr(state, "analysis_eligible", False)),
                "ball_present": bool(evidence.get("ball_present", False)),
                "on_court_count": on_court_count,
                "surface_ratio_median": self._surface_ratio_median(frame_state),
            }
        )
        window_frames = self._threshold_frames(self.STALLED_CONTEXT_WINDOW_SEC, fps)
        while len(self._v4_stalled_context_history) > window_frames:
            self._v4_stalled_context_history.popleft()

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

        static_diag = self._update_static_formation(
            state=state,
            evidence=evidence,
            fps=fps,
        )
        stalled_diag = self._stalled_context_summary(fps=fps)
        hard_break = bool(
            evidence.get("strong_closeup", False)
            or evidence.get("prefilter_scene_cut", False)
        )
        stalled_break = bool(
            hard_break and stalled_diag.get("v4_stalled_context_ready", False)
        )

        self._append_stalled_context_sample(
            frame_state,
            state=state,
            evidence=evidence,
            fps=fps,
        )

        evidence.update(static_diag)
        evidence.update(stalled_diag)
        evidence["v4_stalled_context_broadcast_break"] = stalled_break
        self._v4_last_diagnostics = {
            **static_diag,
            **stalled_diag,
            "v4_stalled_context_broadcast_break": stalled_break,
        }

        if bool(static_diag.get("v4_static_formation_confirmed", False)):
            return (
                False,
                "non_gameplay",
                "v4_static_formation_candidate",
                evidence,
            )
        if stalled_break:
            return (
                False,
                "non_gameplay",
                "v4_stalled_context_broadcast_break_candidate",
                evidence,
            )
        return candidate_present, candidate_status, candidate_reason, evidence

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
        if not candidate_present and candidate_reason in self.DIRECT_CONFIRM_REASONS:
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
                "v4_direct_confirmation": True,
                "v4_direct_confirmation_reason": candidate_reason,
            }
            diagnostics.update(self._v4_last_diagnostics)
            return (
                False,
                "non_gameplay",
                candidate_reason.replace("_candidate", "_confirmed"),
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
        result[3].update(self._v4_last_diagnostics)
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
                "v3_low_action_plus_4s_static_formation_plus_stalled_context_broadcast_break"
            )
            policy.update(self._v4_last_diagnostics)
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_boundary_objective"] = (
            "v3_low_action_plus_4s_static_formation_plus_stalled_context_broadcast_break"
        )
        payload["v4_static_formation"] = {
            "motion_max": self.STATIC_FORMATION_MOTION_MAX,
            "confirm_sec": self.STATIC_FORMATION_CONFIRM_SEC,
            "min_players": self.STATIC_FORMATION_MIN_PLAYERS,
            "min_on_court": self.STATIC_FORMATION_MIN_ON_COURT,
            "max_height_ratio": self.STATIC_FORMATION_MAX_HEIGHT_RATIO,
        }
        payload["v4_stalled_context"] = {
            "window_sec": self.STALLED_CONTEXT_WINDOW_SEC,
            "analysis_rate_max": self.STALLED_CONTEXT_ANALYSIS_RATE_MAX,
            "ball_rate_max": self.STALLED_CONTEXT_BALL_RATE_MAX,
            "min_on_court_mean": self.STALLED_CONTEXT_MIN_ON_COURT_MEAN,
            "surface_max": self.STALLED_CONTEXT_SURFACE_MAX,
        }
        return payload


__all__ = ["CourtStateEstimatorV4"]
