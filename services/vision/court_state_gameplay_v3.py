from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any

from services.vision.court_state_gameplay_v2 import CourtStateEstimatorV2


class CourtStateEstimatorV3(CourtStateEstimatorV2):
    """V6 temporal-boundary policy layered over the long-form V2 estimator.

    V6 exposed three distinct temporal failure modes:

    * short hard negatives (broadcast cuts/close-ups) can become false NON_GAMEPLAY
      boundaries inside live possessions;
    * prolonged weak-live evidence can stay GAMEPLAY_LIMITED indefinitely unless its
      timeout is persistent;
    * wide dead-ball periods can still look superficially live because camera motion,
      many players and intermittent ball detections satisfy the cheap visual heuristics.

    V3 keeps those cases separate:

    * hard negative evidence must persist for 3.5 s before creating a boundary;
    * non-basket weak-live evidence receives a 1.0 s grace period, then stays negative
      until strong live or safe basket+ball context returns;
    * a conservative low-action watchdog removes global camera translation from tracked
      player motion and combines the residual with direct ball availability. Four seconds
      of sustained low action are themselves sufficient evidence for NON_GAMEPLAY.

    The low-action path is intentionally slow. It is not a replacement for normal
    gameplay detection; it exists only to reject long dead-ball stretches that otherwise
    remain latched as gameplay forever.
    """

    POLICY_VERSION = "court_state_gameplay_v3_v6_batch_3"
    BOUNDED_WEAK_REASONS = {
        "on_court_motion_hold",
        "prefilter_and_on_court_motion",
        "motion_supported_prefilter",
    }

    ACTION_WINDOW_SEC = 1.50
    LOW_ACTION_CONFIRM_SEC = 4.00
    LOW_ACTION_MIN_MATCHED_PLAYERS = 3
    LOW_ACTION_MIN_VALID_MOTION_RATIO = 0.50
    LOW_ACTION_MOTION_MAX = 0.0065
    LOW_ACTION_BALL_RATE_MAX = 0.55
    LOW_BALL_MOTION_MAX = 0.0090
    LOW_BALL_RATE_MAX = 0.25

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
        self._v3_previous_players: dict[int, tuple[float, float, float]] = {}
        self._v3_action_history: deque[tuple[float | None, bool]] = deque()
        self._v3_low_action_frames = 0
        self._v3_last_action_watchdog: dict[str, Any] = {}

    def _reset_temporal_presence(self) -> None:
        super()._reset_temporal_presence()
        self._v3_weak_context_frames = 0
        self._v3_weak_timeout_latched = False
        self._v3_previous_players.clear()
        self._v3_action_history.clear()
        self._v3_low_action_frames = 0
        self._v3_last_action_watchdog = {}

    @staticmethod
    def _player_motion_snapshot(
        frame_state: Any,
    ) -> dict[int, tuple[float, float, float]]:
        snapshot: dict[int, tuple[float, float, float]] = {}
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            try:
                track_id = int(player.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            raw_box = player.get("box", player.get("bbox"))
            if track_id < 0 or not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
                continue
            try:
                x1, y1, x2, y2 = (float(raw_box[index]) for index in range(4))
            except (TypeError, ValueError):
                continue
            height = max(1.0, y2 - y1)
            snapshot[track_id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, height)
        return snapshot

    def _relative_player_motion(self, frame_state: Any) -> tuple[float | None, int]:
        """Return camera-translation-compensated player motion in player heights/frame."""
        current = self._player_motion_snapshot(frame_state)
        deltas: list[tuple[float, float, float]] = []
        for track_id, (cx, cy, height) in current.items():
            previous = self._v3_previous_players.get(track_id)
            if previous is None:
                continue
            px, py, previous_height = previous
            deltas.append((
                cx - px,
                cy - py,
                max(1.0, (height + previous_height) / 2.0),
            ))
        self._v3_previous_players = current

        if len(deltas) < self.LOW_ACTION_MIN_MATCHED_PLAYERS:
            return None, len(deltas)

        # A broadcast pan moves most tracked players in roughly the same image-space
        # direction. Subtract the cohort median translation before measuring residual
        # player motion so camera movement cannot by itself keep dead-ball footage live.
        global_dx = statistics.median(delta[0] for delta in deltas)
        global_dy = statistics.median(delta[1] for delta in deltas)
        residuals = [
            math.hypot(dx - global_dx, dy - global_dy) / height
            for dx, dy, height in deltas
        ]
        return float(statistics.median(residuals)), len(deltas)

    def _update_low_action_watchdog(
        self,
        frame_state: Any,
        *,
        fps: float,
    ) -> dict[str, Any]:
        motion, matched_players = self._relative_player_motion(frame_state)
        direct_ball_observed = getattr(frame_state, "ball", None) is not None
        self._v3_action_history.append((motion, direct_ball_observed))

        window_frames = self._threshold_frames(self.ACTION_WINDOW_SEC, fps)
        while len(self._v3_action_history) > window_frames:
            self._v3_action_history.popleft()

        valid_motion = [
            value
            for value, _ in self._v3_action_history
            if value is not None
        ]
        history_frames = len(self._v3_action_history)
        valid_ratio = (
            len(valid_motion) / history_frames if history_frames > 0 else 0.0
        )
        rolling_motion = (
            sum(valid_motion) / len(valid_motion) if valid_motion else None
        )
        ball_rate = (
            sum(1 for _, observed in self._v3_action_history if observed)
            / history_frames
            if history_frames > 0
            else 0.0
        )

        history_ready = bool(
            history_frames >= window_frames
            and valid_ratio >= self.LOW_ACTION_MIN_VALID_MOTION_RATIO
            and rolling_motion is not None
        )
        low_action_signal = bool(
            history_ready
            and (
                (
                    float(rolling_motion) <= self.LOW_ACTION_MOTION_MAX
                    and ball_rate <= self.LOW_ACTION_BALL_RATE_MAX
                )
                or (
                    float(rolling_motion) <= self.LOW_BALL_MOTION_MAX
                    and ball_rate <= self.LOW_BALL_RATE_MAX
                )
            )
        )

        if low_action_signal:
            self._v3_low_action_frames += 1
        else:
            self._v3_low_action_frames = 0

        confirm_frames = self._threshold_frames(self.LOW_ACTION_CONFIRM_SEC, fps)
        confirmed = self._v3_low_action_frames >= confirm_frames
        diagnostics = {
            "v3_low_action_watchdog_enabled": True,
            "v3_player_relative_motion": (
                None if motion is None else round(float(motion), 6)
            ),
            "v3_player_motion_matched_tracks": int(matched_players),
            "v3_action_window_frames": int(window_frames),
            "v3_action_window_valid_motion_ratio": round(float(valid_ratio), 6),
            "v3_action_window_motion_mean": (
                None if rolling_motion is None else round(float(rolling_motion), 6)
            ),
            "v3_action_window_direct_ball_rate": round(float(ball_rate), 6),
            "v3_low_action_signal": low_action_signal,
            "v3_low_action_frames": int(self._v3_low_action_frames),
            "v3_low_action_confirm_frames": int(confirm_frames),
            "v3_low_action_confirmed": bool(confirmed),
        }
        self._v3_last_action_watchdog = diagnostics
        return diagnostics

    def _candidate(
        self,
        frame_state: Any,
        state: Any,
        gameplay: dict[str, Any],
    ) -> tuple[bool, str, str, dict[str, Any]] | None:
        candidate = super()._candidate(frame_state, state, gameplay)
        watchdog = self._update_low_action_watchdog(
            frame_state,
            fps=self._fps(frame_state),
        )
        if candidate is None:
            return None

        candidate_present, candidate_status, candidate_reason, evidence = candidate
        evidence = {**evidence, **watchdog}
        if bool(watchdog.get("v3_low_action_confirmed", False)):
            return (
                False,
                "non_gameplay",
                "v3_sustained_low_action_candidate",
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
        """Make weak-live expiry persistent and accept confirmed low-action stoppages."""

        # The watchdog has already required a 1.5 s causal evidence window followed by
        # 4.0 s of uninterrupted low-action evidence. Do not add the generic 3.5 s hard
        # negative debounce again; that would make this path too slow to reject a useful
        # fraction of a real stoppage.
        if (
            not candidate_present
            and candidate_reason == "v3_sustained_low_action_candidate"
        ):
            self._gameplay_presence_latched = False
            self._non_gameplay_candidate_frames = 0
            self._gameplay_candidate_frames = 0
            return (
                False,
                "non_gameplay",
                "v3_sustained_low_action_confirmed",
                {
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
                    "weak_live_frames": self._weak_live_frames,
                    "v3_low_action_direct_confirmation": True,
                },
            )

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
        diagnostics.update(self._v3_last_action_watchdog)
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
                "3.5s_hard_negative_debounce_plus_persistent_weak_timeout_plus_4s_low_action_watchdog"
            )
            policy["v3_weak_context_frames"] = int(self._v3_weak_context_frames)
            policy["v3_weak_timeout_latched"] = bool(self._v3_weak_timeout_latched)
            policy.update(self._v3_last_action_watchdog)
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        payload = dict(super().metadata())
        payload["gameplay_presence_version"] = self.POLICY_VERSION
        payload["v6_boundary_objective"] = (
            "3.5s_hard_negative_debounce_plus_persistent_weak_timeout_plus_4s_low_action_watchdog"
        )
        payload["bounded_weak_reasons"] = sorted(self.BOUNDED_WEAK_REASONS)
        payload["low_action_watchdog"] = {
            "window_sec": self.ACTION_WINDOW_SEC,
            "confirm_sec": self.LOW_ACTION_CONFIRM_SEC,
            "motion_max": self.LOW_ACTION_MOTION_MAX,
            "ball_rate_max": self.LOW_ACTION_BALL_RATE_MAX,
            "low_ball_motion_max": self.LOW_BALL_MOTION_MAX,
            "low_ball_rate_max": self.LOW_BALL_RATE_MAX,
            "camera_translation_compensated": True,
        }
        return payload


__all__ = ["CourtStateEstimatorV3"]
