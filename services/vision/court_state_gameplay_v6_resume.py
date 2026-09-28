from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v6 import CourtStateEstimatorV6


class CourtStateEstimatorV6Resume(CourtStateEstimatorV6):
    """V6 hold refinement with an FPS-robust wide/dense live-reacquisition path.

    The first resume experiment required 0.30 s of *consecutive predicted-ball-only*
    evidence. That looked safe in a 10 fps unit test, but the real V6 probe runs at 60 fps:
    intermittent direct ball detections repeatedly reset the 18-frame release streak, so the
    policy changed no CourtState decisions at all.

    This revision keeps every V6 non-game confirmation threshold and maximum hold duration
    intact. During an already-active ``static_formation`` hold it looks for a broader visual
    basketball context that is causal and upstream of P2:

    * at least eight players are visible and at least five are explicitly on-court;
    * the view is wide (median player-height ratio <= 0.24);
    * local court-surface support is strong on average (>= 0.30);
    * a ball state is available, whether directly detected or temporally held;
    * the frame is neither a scene cut nor a strong close-up.

    The base context must persist for 0.70 s. Only then does it upgrade to V6's existing
    strong-live release path, which still requires another 0.30 s. The effective release
    horizon is therefore about one second and scales with the real FPS. In the frozen V6
    replay, real P4 sustains this context for ~2.1 s while the closest NG2 streak is ~0.70 s,
    so NG2 does not complete the second phase.
    """

    POLICY_VERSION = "court_state_gameplay_v6_v6_wide_dense_ball_resume"

    WIDE_DENSE_RESUME_MIN_PLAYERS = 8
    WIDE_DENSE_RESUME_MIN_ON_COURT = 5
    WIDE_DENSE_RESUME_MAX_HEIGHT_RATIO = 0.24
    WIDE_DENSE_RESUME_MIN_SURFACE_MEAN = 0.30
    WIDE_DENSE_RESUME_BASE_CONFIRM_SEC = 0.70

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._v6_resume_signal_frames = 0
        self._v6_resume_signal_hits = 0
        self._v6_resume_last_diagnostics: dict[str, Any] = {}

    def _reset_temporal_presence(self) -> None:
        super()._reset_temporal_presence()
        self._v6_resume_signal_frames = 0
        self._v6_resume_last_diagnostics = {}

    @staticmethod
    def _surface_ratio_mean(frame_state: Any) -> float | None:
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
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _direct_selected_ball(frame_state: Any) -> bool:
        diagnostics = getattr(frame_state, "diagnostics", {}) or {}
        detector = diagnostics.get("detector", {}) if isinstance(diagnostics, dict) else {}
        if not isinstance(detector, dict):
            return False
        return bool(detector.get("selected_ball", False))

    def _wide_dense_ball_resume_base_signal(
        self,
        frame_state: Any,
        *,
        state: Any,
        evidence: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        try:
            player_count = int(getattr(state, "player_count", 0) or 0)
        except (TypeError, ValueError):
            player_count = 0
        try:
            on_court_count = int(getattr(state, "on_court_player_count", 0) or 0)
        except (TypeError, ValueError):
            on_court_count = 0

        height_ratio = evidence.get("player_height_ratio_median")
        try:
            height_ratio_value = None if height_ratio is None else float(height_ratio)
        except (TypeError, ValueError):
            height_ratio_value = None

        surface_mean = self._surface_ratio_mean(frame_state)
        ball_present = bool(
            evidence.get("ball_present", getattr(frame_state, "ball", None) is not None)
        )
        direct_ball = self._direct_selected_ball(frame_state)
        hard_break = bool(
            evidence.get("strong_closeup", False)
            or evidence.get("prefilter_scene_cut", False)
        )

        signal = bool(
            self._v5_hold_source == "static_formation"
            and player_count >= self.WIDE_DENSE_RESUME_MIN_PLAYERS
            and on_court_count >= self.WIDE_DENSE_RESUME_MIN_ON_COURT
            and height_ratio_value is not None
            and height_ratio_value <= self.WIDE_DENSE_RESUME_MAX_HEIGHT_RATIO
            and surface_mean is not None
            and surface_mean >= self.WIDE_DENSE_RESUME_MIN_SURFACE_MEAN
            and ball_present
            and not hard_break
        )
        diagnostics = {
            "v6_wide_dense_resume_base_signal": signal,
            "v6_wide_dense_resume_player_count": player_count,
            "v6_wide_dense_resume_on_court_count": on_court_count,
            "v6_wide_dense_resume_height_ratio": (
                None if height_ratio_value is None else round(height_ratio_value, 6)
            ),
            "v6_wide_dense_resume_surface_mean": (
                None if surface_mean is None else round(float(surface_mean), 6)
            ),
            "v6_wide_dense_resume_ball_present": ball_present,
            "v6_wide_dense_resume_direct_ball": direct_ball,
            "v6_wide_dense_resume_hard_break": hard_break,
            "v6_wide_dense_resume_hold_source": self._v5_hold_source,
        }
        return signal, diagnostics

    def _update_wide_dense_ball_resume_signal(
        self,
        frame_state: Any,
        *,
        state: Any,
        evidence: dict[str, Any],
        fps: float,
    ) -> tuple[bool, dict[str, Any]]:
        base_signal, diagnostics = self._wide_dense_ball_resume_base_signal(
            frame_state,
            state=state,
            evidence=evidence,
        )
        if base_signal:
            self._v6_resume_signal_frames += 1
            self._v6_resume_signal_hits += 1
        else:
            self._v6_resume_signal_frames = 0

        confirm_frames = self._threshold_frames(
            self.WIDE_DENSE_RESUME_BASE_CONFIRM_SEC,
            fps,
        )
        confirmed = bool(self._v6_resume_signal_frames >= confirm_frames)
        diagnostics.update(
            {
                "v6_wide_dense_resume_signal_frames": int(
                    self._v6_resume_signal_frames
                ),
                "v6_wide_dense_resume_confirm_frames": int(confirm_frames),
                "v6_wide_dense_resume_confirmed": confirmed,
                "v6_wide_dense_resume_signal_hits": int(self._v6_resume_signal_hits),
            }
        )
        return confirmed, diagnostics

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
        confirmed, diagnostics = self._update_wide_dense_ball_resume_signal(
            frame_state,
            state=state,
            evidence=evidence,
            fps=self._fps(frame_state),
        )

        if confirmed:
            candidate_present = True
            candidate_status = (
                "gameplay_analysis"
                if bool(getattr(state, "analysis_eligible", False))
                else "gameplay_limited"
            )
            candidate_reason = "v6_wide_dense_ball_resume_evidence"
            evidence["strong_live_evidence"] = True
            evidence["safe_limited_resume"] = False
            evidence["weak_live_evidence"] = False

        evidence.update(diagnostics)
        self._v6_resume_last_diagnostics = diagnostics
        return candidate_present, candidate_status, candidate_reason, evidence

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
                "preserve_50pct_non_game_while_releasing_sustained_wide_dense_ball_live_context"
            )
            policy.update(self._v6_resume_last_diagnostics)
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_wide_dense_ball_resume"] = {
            "min_players": self.WIDE_DENSE_RESUME_MIN_PLAYERS,
            "min_on_court": self.WIDE_DENSE_RESUME_MIN_ON_COURT,
            "max_height_ratio": self.WIDE_DENSE_RESUME_MAX_HEIGHT_RATIO,
            "min_surface_mean": self.WIDE_DENSE_RESUME_MIN_SURFACE_MEAN,
            "requires_ball_present": True,
            "requires_static_hold": True,
            "base_confirm_sec": self.WIDE_DENSE_RESUME_BASE_CONFIRM_SEC,
            "strong_live_release_sec": self.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC,
            "effective_release_horizon_sec": round(
                self.WIDE_DENSE_RESUME_BASE_CONFIRM_SEC
                + self.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC,
                3,
            ),
        }
        return payload


__all__ = ["CourtStateEstimatorV6Resume"]
