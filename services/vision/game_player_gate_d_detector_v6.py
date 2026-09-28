from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v5 import CourtStateEstimatorV5
from services.vision.game_player_gate_d_detector_v5 import PerformanceDetector as GateDDetectorV5
from services.vision.gate_d_limited_scope_v6 import select_limited_bootstrap_candidates


class PerformanceDetector(GateDDetectorV5):
    """V6 Gate-D scope recovery on top of the stoppage-aware V5 resolver.

    LIMITED fallback is split into two risk tiers:

    * already-resolved raw tracks may continue with plausible image-space court evidence;
    * unseen tracks require stronger evidence plus three consecutive candidate frames.

    NON_GAMEPLAY still freezes the resolver completely.
    """

    NEW_TRACK_PERSISTENCE_FRAMES = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.identity_court_estimator = CourtStateEstimatorV5()
        self._v6_last_limited_bootstrap_count = 0
        self._v6_last_limited_bootstrap_known_count = 0
        self._v6_last_limited_bootstrap_new_count = 0
        self._v6_last_limited_bootstrap_candidate_count = 0
        self._v6_new_track_candidate_streaks: dict[int, int] = {}
        self._v6_limited_scope_empty_frames = 0
        self._v6_waiting_for_identity_scope_after_resume = False
        self._v6_resume_scope_wait_frames = 0
        self._v6_last_frames_to_identity_scope_after_resume: int | None = None
        print(
            "🧍 Gate D V6 scope: resolved-track continuation + persistent new-track bootstrap",
            flush=True,
        )

    def _resolved_current_track_ids(self) -> set[int]:
        resolver = self.game_player_resolver
        current = dict(getattr(resolver, "current_episode_by_track", {}) or {})
        episodes = dict(getattr(resolver, "episodes", {}) or {})
        resolved: set[int] = set()
        for raw_track_id, key in current.items():
            episode = episodes.get(key)
            if episode is None or getattr(episode, "game_player_id", None) is None:
                continue
            try:
                resolved.add(int(raw_track_id))
            except (TypeError, ValueError):
                continue
        return resolved

    def _gate_d_limited_bootstrap_candidates(
        self,
        frame_state: Any,
    ) -> list[dict[str, Any]]:
        candidates = select_limited_bootstrap_candidates(
            list(getattr(frame_state, "players", []) or []),
            known_track_ids=self._resolved_current_track_ids(),
        )
        self._v6_last_limited_bootstrap_candidate_count = len(candidates)

        new_ids: set[int] = set()
        for row in candidates:
            if str(row.get("gate_d_limited_bootstrap_tier") or "") != (
                "persistent_new_track_candidate"
            ):
                continue
            try:
                new_ids.add(int(row.get("track_id", -1)))
            except (TypeError, ValueError):
                continue

        self._v6_new_track_candidate_streaks = {
            track_id: streak
            for track_id, streak in self._v6_new_track_candidate_streaks.items()
            if track_id in new_ids
        }
        for track_id in new_ids:
            self._v6_new_track_candidate_streaks[track_id] = (
                self._v6_new_track_candidate_streaks.get(track_id, 0) + 1
            )

        selected: list[dict[str, Any]] = []
        known_count = 0
        new_count = 0
        for row in candidates:
            tier = str(row.get("gate_d_limited_bootstrap_tier") or "")
            try:
                track_id = int(row.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if tier == "known_resolved_track":
                copy = dict(row)
                copy["gate_d_limited_bootstrap_persistence_frames"] = None
                selected.append(copy)
                known_count += 1
                continue
            streak = int(self._v6_new_track_candidate_streaks.get(track_id, 0))
            if streak < self.NEW_TRACK_PERSISTENCE_FRAMES:
                continue
            copy = dict(row)
            copy["gate_d_limited_bootstrap_persistence_frames"] = streak
            selected.append(copy)
            new_count += 1

        self._v6_last_limited_bootstrap_known_count = known_count
        self._v6_last_limited_bootstrap_new_count = new_count
        return selected

    def _gate_d_limited_players(self, frame_state: Any) -> list[dict[str, Any]]:
        explicit = GateDDetectorV5._gate_d_limited_players(frame_state)
        if explicit:
            self._v6_new_track_candidate_streaks.clear()
            self._v6_last_limited_bootstrap_count = 0
            self._v6_last_limited_bootstrap_known_count = 0
            self._v6_last_limited_bootstrap_new_count = 0
            self._v6_last_limited_bootstrap_candidate_count = 0
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

        if mode != "gameplay_limited":
            self._v6_new_track_candidate_streaks.clear()

        if mode == "frozen_non_gameplay":
            self._v6_waiting_for_identity_scope_after_resume = True
            self._v6_resume_scope_wait_frames = 0
            self._v6_last_limited_bootstrap_count = 0
            self._v6_last_limited_bootstrap_known_count = 0
            self._v6_last_limited_bootstrap_new_count = 0
            self._v6_last_limited_bootstrap_candidate_count = 0
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

        runtime["version"] = "game_player_resolver_v6_v6_batch_2"
        runtime["limited_empty_scope_bootstrap_enabled"] = True
        runtime["limited_scope_empty_before_bootstrap"] = bool(
            mode == "gameplay_limited" and explicit_count == 0
        )
        runtime["limited_scope_bootstrap_candidates"] = int(
            self._v6_last_limited_bootstrap_candidate_count
            if mode == "gameplay_limited" and explicit_count == 0
            else 0
        )
        runtime["limited_scope_bootstrap_players"] = int(
            self._v6_last_limited_bootstrap_count
            if mode == "gameplay_limited" and explicit_count == 0
            else 0
        )
        runtime["limited_scope_bootstrap_known_players"] = int(
            self._v6_last_limited_bootstrap_known_count
            if mode == "gameplay_limited" and explicit_count == 0
            else 0
        )
        runtime["limited_scope_bootstrap_new_players"] = int(
            self._v6_last_limited_bootstrap_new_count
            if mode == "gameplay_limited" and explicit_count == 0
            else 0
        )
        runtime["limited_scope_new_track_persistence_frames"] = (
            self.NEW_TRACK_PERSISTENCE_FRAMES
        )
        runtime["limited_scope_empty_frames_cumulative"] = int(
            self._v6_limited_scope_empty_frames
        )
        runtime["frames_to_identity_scope_after_resume"] = (
            self._v6_last_frames_to_identity_scope_after_resume
        )
        runtime["limited_bootstrap_never_runs_in_non_gameplay"] = True


__all__ = ["PerformanceDetector"]
