from __future__ import annotations

from typing import Any

from services.vision.court_state_gameplay_v2 import CourtStateEstimatorV2
from services.vision.game_player_gate_d_detector_v4 import PerformanceDetector as GateDDetectorV4
from services.vision.game_player_resolver_gate_d_v5 import GateDDeferredAssociationConfig
from services.vision.game_player_resolver_gate_d_v5_stoppage import (
    GateDStoppageAwareGamePlayerResolverV5,
)


class PerformanceDetector(GateDDetectorV4):
    """Gate-D detector using v5 persistent ambiguity recovery.

    The bounded long-gap reappearance experiment remains available in the resolver for
    synthetic coverage, but is disabled in the development runtime after V3 showed that
    it delayed many legitimate new identities without producing a safe reappearance.

    Gate-D continuity follows the three-state CourtState gameplay contract:
    * GAMEPLAY_ANALYSIS observes the normal identity-eligible player set;
    * GAMEPLAY_LIMITED observes only players with explicit positive on-court evidence;
    * NON_GAMEPLAY freezes the resolver completely: no new episode, split, appearance
      update or association decision is produced, while all existing resolver memory is
      preserved for the next gameplay frame. Frozen source frames are also removed from
      the resolver's temporal-gap calculations, so a real stoppage does not become an
      artificial ``track_gap`` or stale cross-track association boundary on resume.

    Within-track team changes additionally reuse Identity-v2's existing confirmed-reversal
    diagnostic. An opposite-team observation that Identity-v2 has not confirmed cannot by
    itself open a Gate-D episode; cross-track hard team incompatibility remains unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Keep the reappearance ceiling equal to the canonical association window so the
        # long-gap path has no runtime-only window. Persistent within-window ambiguity
        # recovery remains enabled.
        self.game_player_resolver = GateDStoppageAwareGamePlayerResolverV5(
            deferred_association_config=GateDDeferredAssociationConfig(
                max_reappearance_gap_frames=180,
            )
        )
        # Gate-D/P2 validation needs the long-form gameplay-presence contract before
        # Identity-v2 team/jersey evidence and before this post-process continuity hook.
        self.identity_court_estimator = CourtStateEstimatorV2()
        print(
            "🧍 Gate D identity resolver: v5 deferred ambiguity + confirmed team reversals "
            "+ gameplay-state continuity scope",
            flush=True,
        )

    @staticmethod
    def _gate_d_continuity_mode(frame_state: Any) -> str:
        court_state = getattr(frame_state, "court_state", None)
        if court_state is None or not hasattr(court_state, "gameplay_present"):
            return "legacy"
        gameplay_present = getattr(court_state, "gameplay_present", None)
        if gameplay_present is False:
            return "frozen_non_gameplay"
        if gameplay_present is True and bool(
            getattr(court_state, "analysis_eligible", False)
        ):
            return "gameplay_analysis"
        if gameplay_present is True:
            return "gameplay_limited"
        return "legacy"

    @classmethod
    def _gate_d_widen_continuity_scope(cls, frame_state: Any) -> bool:
        return cls._gate_d_continuity_mode(frame_state) == "gameplay_limited"

    @staticmethod
    def _gate_d_limited_players(frame_state: Any) -> list[dict[str, Any]]:
        return [
            player
            for player in list(getattr(frame_state, "players", []) or [])
            if isinstance(player, dict) and player.get("active_on_court") is True
        ]

    @staticmethod
    def _gate_d_team_change_confirmations(frame_state: Any) -> dict[int, bool]:
        diagnostics = getattr(frame_state, "diagnostics", {}) or {}
        detector = diagnostics.get("detector", {}) if isinstance(diagnostics, dict) else {}
        predictions = (
            list(detector.get("team_predictions", []) or [])
            if isinstance(detector, dict)
            else []
        )
        confirmations: dict[int, bool] = {}
        for row in predictions:
            if not isinstance(row, dict) or "recent_reversal_confirmed" not in row:
                continue
            try:
                track_id = int(row.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            value = row.get("recent_reversal_confirmed")
            if track_id >= 0 and isinstance(value, bool):
                confirmations[track_id] = value
        return confirmations

    def _stoppage_runtime(self) -> dict[str, object]:
        if hasattr(self.game_player_resolver, "stoppage_diagnostics"):
            return dict(self.game_player_resolver.stoppage_diagnostics())
        return {"stoppage_gap_suspension": False}

    def _freeze_non_gameplay_continuity(
        self,
        frame_state: Any,
        *,
        confirmations: dict[int, bool],
    ) -> None:
        detector = frame_state.diagnostics.setdefault("detector", {})
        detector["game_player_predictions"] = []
        detector["game_player_runtime"] = {
            "version": "game_player_resolver_v5_gate_d_team_stable_stoppage",
            "identity_key": "track_id+temporal_episode",
            "unknown_preferred": True,
            "eligible_players": 0,
            "tracked_players": 0,
            "appearance_available": 0,
            "resolved_players": 0,
            "unknown_players": 0,
            "deferred_ambiguous_reassociation": True,
            "deferred_long_gap_reappearance": False,
            "bounded_reappearance_probe": False,
            "continuity_scope": "court_state_gameplay_status",
            "continuity_mode": "frozen_non_gameplay",
            "continuity_frozen": True,
            "non_gameplay_freezes_resolver": True,
            "limited_gameplay_requires_explicit_on_court": True,
            "limited_gameplay_scope_widened": False,
            "team_conflict_requires_fresh_reversal_confirmation": True,
            "team_change_confirmation_tracks": len(confirmations),
            "team_change_confirmed_tracks": sum(confirmations.values()),
            **self._stoppage_runtime(),
        }

    def _observe_game_players(self, frame_state: Any) -> None:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        court_state = getattr(frame_state, "court_state", None)
        mode = self._gate_d_continuity_mode(frame_state)
        confirmations = self._gate_d_team_change_confirmations(frame_state)

        # A true stoppage is a temporal boundary for observations, not for memory. Do not
        # call prepare_frame()/observe() at all: the resolver's entities/episodes remain
        # untouched. Only the stoppage clock is advanced so temporal-gap checks can remove
        # the frozen interval when gameplay resumes.
        if mode == "frozen_non_gameplay":
            if hasattr(self.game_player_resolver, "freeze_non_gameplay"):
                self.game_player_resolver.freeze_non_gameplay(frame_id)
            self._freeze_non_gameplay_continuity(
                frame_state,
                confirmations=confirmations,
            )
            return

        if hasattr(self.game_player_resolver, "resume_gameplay"):
            self.game_player_resolver.resume_gameplay(frame_id)

        widen_scope = mode == "gameplay_limited"
        original_analysis_eligible = (
            bool(getattr(court_state, "analysis_eligible", False))
            if court_state is not None
            else False
        )
        original_players = getattr(frame_state, "players", None)
        limited_players = self._gate_d_limited_players(frame_state) if widen_scope else None

        if limited_players is not None:
            allowed_tracks = {
                int(player.get("track_id", -1))
                for player in limited_players
                if int(player.get("track_id", -1)) >= 0
            }
            confirmations = {
                track_id: confirmed
                for track_id, confirmed in confirmations.items()
                if track_id in allowed_tracks
            }

        self.game_player_resolver.prepare_team_stability(confirmations)
        if widen_scope and court_state is not None:
            court_state.analysis_eligible = True
            frame_state.players = limited_players
        try:
            super()._observe_game_players(frame_state)
        finally:
            if widen_scope and court_state is not None:
                court_state.analysis_eligible = original_analysis_eligible
                frame_state.players = original_players
            self.game_player_resolver.clear_team_stability()

        detector = frame_state.diagnostics.setdefault("detector", {})
        runtime = detector.setdefault("game_player_runtime", {})
        runtime["version"] = "game_player_resolver_v5_gate_d_team_stable_stoppage"
        runtime["deferred_ambiguous_reassociation"] = True
        runtime["deferred_long_gap_reappearance"] = False
        runtime["bounded_reappearance_probe"] = False
        runtime["continuity_scope"] = "court_state_gameplay_status"
        runtime["continuity_mode"] = mode
        runtime["continuity_frozen"] = False
        runtime["non_gameplay_freezes_resolver"] = True
        runtime["limited_gameplay_requires_explicit_on_court"] = True
        runtime["limited_gameplay_scope_widened"] = bool(widen_scope)
        runtime["limited_gameplay_explicit_players"] = (
            len(limited_players) if limited_players is not None else None
        )
        runtime["team_conflict_requires_fresh_reversal_confirmation"] = True
        runtime["team_change_confirmation_tracks"] = len(confirmations)
        runtime["team_change_confirmed_tracks"] = sum(confirmations.values())
        runtime.update(self._stoppage_runtime())
