from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v6_resume import CourtStateEstimatorV6Resume
from services.vision.game_player_gate_d_detector_v5 import PerformanceDetector as GateDDetectorV5
from services.vision.game_player_resolver_gate_d_v6 import GateDGamePlayerResolverV6
from services.vision.gate_d_limited_scope_v6 import select_limited_bootstrap_candidates
from services.vision.gate_d_rosterless_jersey_v6 import (
    GateDRosterlessJerseyEvidenceV6,
)


class PerformanceDetector(GateDDetectorV5):
    """V6 Gate-D scope + identity recovery on top of the frozen V5 detector.

    V6 keeps the CourtState/limited-scope behavior already validated for A-C, while the
    identity path adds only Gate-D-specific evidence:

    * resolved raw tracks may continue through LIMITED gameplay;
    * unseen tracks require persistent LIMITED bootstrap evidence;
    * rosterless jersey OCR rejected solely because the validation config has no roster is
      accumulated as soft visual evidence after repeated agreement;
    * the V6 resolver consumes that soft evidence without changing canonical P1 identity.

    NON_GAMEPLAY still freezes the resolver completely.
    """

    NEW_TRACK_PERSISTENCE_FRAMES = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.identity_court_estimator = CourtStateEstimatorV6Resume()

        self.game_player_resolver = GateDGamePlayerResolverV6()
        self._v6_rosterless_jersey = GateDRosterlessJerseyEvidenceV6()

        self._v6_last_limited_bootstrap_count = 0
        self._v6_last_limited_bootstrap_known_count = 0
        self._v6_last_limited_bootstrap_new_count = 0
        self._v6_last_limited_bootstrap_candidate_count = 0
        self._v6_new_track_candidate_streaks: dict[int, int] = {}
        self._v6_limited_scope_empty_frames = 0
        self._v6_waiting_for_identity_scope_after_resume = False
        self._v6_resume_scope_wait_frames = 0
        self._v6_last_frames_to_identity_scope_after_resume: int | None = None
        self._v6_last_soft_jersey_tracks = 0
        print(
            "🧍 Gate D V6: scope bootstrap + rosterless soft-jersey identity recovery",
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

    def _v6_prepare_soft_jersey_evidence(self, frame_state: Any) -> None:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        diagnostics = getattr(frame_state, "diagnostics", {}) or {}
        detector = diagnostics.get("detector", {}) if isinstance(diagnostics, dict) else {}
        predictions = (
            list(detector.get("jersey_predictions", []) or [])
            if isinstance(detector, dict)
            else []
        )
        self._v6_rosterless_jersey.observe_predictions(
            frame_id=frame_id,
            predictions=[row for row in predictions if isinstance(row, dict)],
        )

        stable_tracks = 0
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            try:
                track_id = int(player.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if track_id < 0:
                continue
            number, confidence, evidence = self._v6_rosterless_jersey.evidence(
                track_id=track_id,
                frame_id=frame_id,
            )
            if number is None:
                player.pop("gate_d_v6_soft_jersey_number", None)
                player.pop("gate_d_v6_soft_jersey_confidence", None)
                player.pop("gate_d_v6_soft_jersey_evidence", None)
                continue
            stable_tracks += 1
            player["gate_d_v6_soft_jersey_number"] = number
            player["gate_d_v6_soft_jersey_confidence"] = confidence
            player["gate_d_v6_soft_jersey_evidence"] = evidence
        self._v6_last_soft_jersey_tracks = stable_tracks

    def _jersey_evidence(self, player: dict[str, Any]) -> tuple[str | None, float]:
        canonical_number, canonical_confidence = super()._jersey_evidence(player)
        if canonical_number is not None:
            return canonical_number, canonical_confidence

        number = player.get("gate_d_v6_soft_jersey_number")
        if number is None:
            return None, 0.0
        try:
            confidence = float(
                player.get("gate_d_v6_soft_jersey_confidence", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            return None, 0.0
        return str(number), max(0.0, min(0.95, confidence))

    def _observe_game_players(self, frame_state: Any) -> None:
        mode = self._gate_d_continuity_mode(frame_state)
        explicit_count = sum(
            1
            for player in list(getattr(frame_state, "players", []) or [])
            if isinstance(player, dict) and player.get("active_on_court") is True
        )

        if mode != "frozen_non_gameplay":
            self._v6_prepare_soft_jersey_evidence(frame_state)
        else:
            self._v6_last_soft_jersey_tracks = 0

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

        config = self._v6_rosterless_jersey.config
        runtime["version"] = "game_player_resolver_v6_soft_jersey_reappearance"
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

        runtime["rosterless_soft_jersey_enabled"] = True
        runtime["rosterless_soft_jersey_tracks"] = int(
            self._v6_last_soft_jersey_tracks
        )
        runtime["rosterless_soft_jersey_raw_observations"] = int(
            self._v6_rosterless_jersey.accepted_raw_observations
        )
        runtime["rosterless_soft_jersey_min_observations"] = int(
            config.min_observations
        )
        runtime["rosterless_soft_jersey_min_posterior"] = float(
            config.min_candidate_posterior
        )
        runtime["rosterless_soft_jersey_window_frames"] = int(config.window_frames)
        runtime["team_only_split_suppressed_v6"] = True
        runtime["long_reappearance_requires_soft_jersey"] = True


__all__ = ["PerformanceDetector"]
