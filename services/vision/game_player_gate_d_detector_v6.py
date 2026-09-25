from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v3 import CourtStateEstimatorV3
from services.vision.game_player_gate_d_detector_v5 import PerformanceDetector as GateDDetectorV5


class PerformanceDetector(GateDDetectorV5):
    """V6 Gate-D scope recovery on top of the stoppage-aware V5 resolver.

    V5 intentionally requires explicit ``active_on_court=True`` evidence during
    GAMEPLAY_LIMITED. In V6 that leaves a small but real set of LIMITED frames with zero
    identity observations. V6 keeps the explicit set whenever it exists and uses a bounded
    fallback only when that set is empty.

    The fallback never runs in NON_GAMEPLAY and never invents an identity. It only exposes
    strong player detections to the existing unknown-preferred resolver when image-space
    court evidence says they are plausible on-court observations.
    """

    LIMITED_BOOTSTRAP_MAX_PLAYERS = 8
    LIMITED_BOOTSTRAP_MIN_PLAYER_CONFIDENCE = 0.82
    LIMITED_BOOTSTRAP_MIN_SURFACE_RATIO = 0.18
    LIMITED_BOOTSTRAP_STRONG_SURFACE_RATIO = 0.30
    LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE = 0.80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.identity_court_estimator = CourtStateEstimatorV3()
        self._v6_last_limited_bootstrap_count = 0
        self._v6_limited_scope_empty_frames = 0
        self._v6_waiting_for_identity_scope_after_resume = False
        self._v6_resume_scope_wait_frames = 0
        self._v6_last_frames_to_identity_scope_after_resume: int | None = None
        print(
            "🧍 Gate D V6 scope: explicit LIMITED players + bounded empty-scope bootstrap",
            flush=True,
        )

    @classmethod
    def _gate_d_limited_bootstrap_candidates(
        cls,
        frame_state: Any,
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, float, float, dict[str, Any]]] = []
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            if player.get("active_on_court") is False:
                continue
            if player.get("identity_eligible") is False:
                continue

            try:
                confidence = float(player.get("confidence", 0.0) or 0.0)
                surface_ratio = float(player.get("local_court_surface_ratio", 0.0) or 0.0)
                team_confidence = float(player.get("team_confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if confidence < cls.LIMITED_BOOTSTRAP_MIN_PLAYER_CONFIDENCE:
                continue
            if surface_ratio < cls.LIMITED_BOOTSTRAP_MIN_SURFACE_RATIO:
                continue

            source = str(player.get("court_eligibility_source") or "")
            strong_court = bool(
                source == "image_space_court_surface"
                or surface_ratio >= cls.LIMITED_BOOTSTRAP_STRONG_SURFACE_RATIO
            )
            if not strong_court:
                continue

            has_strong_team = bool(
                player.get("team") is not None
                and team_confidence >= cls.LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE
            )
            if not has_strong_team and surface_ratio < cls.LIMITED_BOOTSTRAP_STRONG_SURFACE_RATIO:
                continue

            copy = dict(player)
            # This flag exists only in the temporary Gate-D observation view. The original
            # detector row is restored by V5 after observe() and is not mutated.
            copy["active_on_court"] = True
            copy["gate_d_limited_bootstrap"] = True
            copy["gate_d_limited_bootstrap_reason"] = (
                "strong_team_and_court_evidence"
                if has_strong_team
                else "strong_court_surface_evidence"
            )
            ranked.append((surface_ratio, confidence, team_confidence, copy))

        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2]))
        return [
            row[3]
            for row in ranked[: cls.LIMITED_BOOTSTRAP_MAX_PLAYERS]
        ]

    def _gate_d_limited_players(self, frame_state: Any) -> list[dict[str, Any]]:
        explicit = GateDDetectorV5._gate_d_limited_players(frame_state)
        if explicit:
            self._v6_last_limited_bootstrap_count = 0
            return explicit
        bootstrap = self._gate_d_limited_bootstrap_candidates(frame_state)
        self._v6_last_limited_bootstrap_count = len(bootstrap)
        return bootstrap

    def _observe_game_players(self, frame_state: Any) -> None:
        mode = self._gate_d_continuity_mode(frame_state)
        explicit_count = sum(
            1
            for player in list(getattr(frame_state, "players", []) or [])
            if isinstance(player, dict) and player.get("active_on_court") is True
        )

        if mode == "frozen_non_gameplay":
            self._v6_waiting_for_identity_scope_after_resume = True
            self._v6_resume_scope_wait_frames = 0
            self._v6_last_limited_bootstrap_count = 0
        elif mode == "gameplay_limited" and explicit_count == 0:
            self._v6_limited_scope_empty_frames += 1

        super()._observe_game_players(frame_state)

        diagnostics = getattr(frame_state, "diagnostics", {}) or {}
        detector = diagnostics.get("detector", {}) if isinstance(diagnostics, dict) else {}
        runtime = detector.get("game_player_runtime", {}) if isinstance(detector, dict) else {}
        if not isinstance(runtime, dict):
            return

        if mode != "frozen_non_gameplay" and self._v6_waiting_for_identity_scope_after_resume:
            self._v6_resume_scope_wait_frames += 1
            try:
                eligible_players = int(runtime.get("eligible_players", 0) or 0)
            except (TypeError, ValueError):
                eligible_players = 0
            if eligible_players > 0:
                self._v6_last_frames_to_identity_scope_after_resume = (
                    self._v6_resume_scope_wait_frames
                )
                self._v6_waiting_for_identity_scope_after_resume = False

        runtime["version"] = "game_player_resolver_v6_v6_batch"
        runtime["limited_empty_scope_bootstrap_enabled"] = True
        runtime["limited_scope_empty_before_bootstrap"] = bool(
            mode == "gameplay_limited" and explicit_count == 0
        )
        runtime["limited_scope_bootstrap_players"] = int(
            self._v6_last_limited_bootstrap_count
            if mode == "gameplay_limited" and explicit_count == 0
            else 0
        )
        runtime["limited_scope_empty_frames_cumulative"] = int(
            self._v6_limited_scope_empty_frames
        )
        runtime["frames_to_identity_scope_after_resume"] = (
            self._v6_last_frames_to_identity_scope_after_resume
        )
        runtime["limited_bootstrap_never_runs_in_non_gameplay"] = True


__all__ = ["PerformanceDetector"]
