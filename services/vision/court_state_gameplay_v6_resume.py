from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v6 import CourtStateEstimatorV6


class CourtStateEstimatorV6Resume(CourtStateEstimatorV6):
    """V6 hold refinement with a causal wide predicted-ball resume path.

    The previous V6 policy crossed the 50% non-game rejection milestone, but the five-second
    static hold still consumed its full budget inside real P4 gameplay. The existing
    ``strong_wide_live`` release path did not fire there because camera-compensated motion
    stayed below its threshold even though the visual basketball context had clearly
    reacquired.

    This refinement keeps every V6 confirmation threshold and maximum hold duration intact.
    It adds one hold-only resume signal for a distinctive live-play reacquisition pattern:

    * at least eight players are visible and at least five are explicitly on-court;
    * the view is wide (median player-height ratio <= 0.24);
    * local court-surface support is strong on average (>= 0.30);
    * a ball state is available, but it is an interpolated/predicted hold rather than a
      direct selected-ball detection;
    * the frame is neither a scene cut nor a strong close-up.

    The signal only upgrades ``strong_live_evidence`` while an inferred stoppage hold is
    already active, so it cannot create a new gameplay acquisition outside that bounded
    context. The existing 0.30-second sustained-live persistence requirement still applies.
    """

    POLICY_VERSION = "court_state_gameplay_v6_v6_wide_predicted_ball_resume"

    WIDE_PREDICTED_RESUME_MIN_PLAYERS = 8
    WIDE_PREDICTED_RESUME_MIN_ON_COURT = 5
    WIDE_PREDICTED_RESUME_MAX_HEIGHT_RATIO = 0.24
    WIDE_PREDICTED_RESUME_MIN_SURFACE_MEAN = 0.30

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

    def _wide_predicted_ball_resume_signal(
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
        ball_present = bool(evidence.get("ball_present", getattr(frame_state, "ball", None) is not None))
        direct_ball = self._direct_selected_ball(frame_state)
        hard_break = bool(
            evidence.get("strong_closeup", False)
            or evidence.get("prefilter_scene_cut", False)
        )

        signal = bool(
            self._v5_hold_source is not None
            and player_count >= self.WIDE_PREDICTED_RESUME_MIN_PLAYERS
            and on_court_count >= self.WIDE_PREDICTED_RESUME_MIN_ON_COURT
            and height_ratio_value is not None
            and height_ratio_value <= self.WIDE_PREDICTED_RESUME_MAX_HEIGHT_RATIO
            and surface_mean is not None
            and surface_mean >= self.WIDE_PREDICTED_RESUME_MIN_SURFACE_MEAN
            and ball_present
            and not direct_ball
            and not hard_break
        )
        diagnostics = {
            "v6_wide_predicted_resume_signal": signal,
            "v6_wide_predicted_resume_player_count": player_count,
            "v6_wide_predicted_resume_on_court_count": on_court_count,
            "v6_wide_predicted_resume_height_ratio": (
                None if height_ratio_value is None else round(height_ratio_value, 6)
            ),
            "v6_wide_predicted_resume_surface_mean": (
                None if surface_mean is None else round(float(surface_mean), 6)
            ),
            "v6_wide_predicted_resume_ball_present": ball_present,
            "v6_wide_predicted_resume_direct_ball": direct_ball,
            "v6_wide_predicted_resume_hard_break": hard_break,
        }
        return signal, diagnostics

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
        signal, diagnostics = self._wide_predicted_ball_resume_signal(
            frame_state,
            state=state,
            evidence=evidence,
        )

        if signal:
            self._v6_resume_signal_frames += 1
            self._v6_resume_signal_hits += 1
            candidate_present = True
            candidate_status = (
                "gameplay_analysis"
                if bool(getattr(state, "analysis_eligible", False))
                else "gameplay_limited"
            )
            candidate_reason = "v6_wide_predicted_ball_resume_evidence"
            evidence["strong_live_evidence"] = True
            evidence["safe_limited_resume"] = False
            evidence["weak_live_evidence"] = False
        else:
            self._v6_resume_signal_frames = 0

        diagnostics["v6_wide_predicted_resume_signal_frames"] = int(
            self._v6_resume_signal_frames
        )
        diagnostics["v6_wide_predicted_resume_signal_hits"] = int(
            self._v6_resume_signal_hits
        )
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
                "preserve_50pct_non_game_while_releasing_wide_predicted_ball_live_reacquisition"
            )
            policy.update(self._v6_resume_last_diagnostics)
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_wide_predicted_ball_resume"] = {
            "min_players": self.WIDE_PREDICTED_RESUME_MIN_PLAYERS,
            "min_on_court": self.WIDE_PREDICTED_RESUME_MIN_ON_COURT,
            "max_height_ratio": self.WIDE_PREDICTED_RESUME_MAX_HEIGHT_RATIO,
            "min_surface_mean": self.WIDE_PREDICTED_RESUME_MIN_SURFACE_MEAN,
            "requires_ball_present": True,
            "requires_non_direct_selected_ball": True,
            "hold_only": True,
            "release_persistence_sec": self.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC,
        }
        return payload


__all__ = ["CourtStateEstimatorV6Resume"]
